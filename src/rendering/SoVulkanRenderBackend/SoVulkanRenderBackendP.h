// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h
//
// Private internal header: helper code formerly in the file-local anonymous namespace,
// lifted into CoinVulkanDetail so the split SoVulkanRenderBackend*.cpp TUs can share it.
// Provides debug counters; push-constant/lighting-UBO structs (VulkanPushConstants,
// VulkanBackgroundPush, VulkanLightingUbo, VulkanDrawUbo); Vulkan enum converters;
// FNV content-hash helpers; draw/overlay/composite counters; createImageView().

#ifndef COIN_SOVULKANRENDERBACKENDP_H
#define COIN_SOVULKANRENDERBACKENDP_H

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>

#include "rendering/SoVulkanPlatform.h"
#include <vulkan/vulkan.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <rendering/SoFnv1a.h>
#include <rendering/SoVulkanConfig.h>

#ifndef _WIN32
#include <unistd.h>
#endif

// Declared in SoRenderBackend.h; only the frame-stats helper references it, so a
// forward declaration keeps the backend interface out of this header.
struct SoRenderParams;

namespace CoinVulkanDetail {

  // ---- [TRC] per-step recording traces (COIN_VULKAN_TRACE) ----
  // One line per pipeline step, tagged with sequence, frame and thread id so an
  // interleaved multi-thread log reads in order.  Gated by COIN_VULKAN_TRACE, cached.
  inline bool vkBackendTraceEnabled()
  {
    static const bool enabled = SoVulkanShared::envString("COIN_VULKAN_TRACE") != nullptr;
    return enabled;
  }

  inline void vkBackendTrace(const uint32_t frame, const char * stage,
                             const char * fmt, ...)
  {
    if (!vkBackendTraceEnabled()) return;
    static std::atomic<long> seq(0);
    const long n = seq.fetch_add(1, std::memory_order_relaxed);
    char buf[224];
    int len = std::snprintf(buf, sizeof(buf), "[TRC] %ld f=%u", n, frame);
    const size_t th =
      std::hash<std::thread::id>()(std::this_thread::get_id());
    len += std::snprintf(buf + len, sizeof(buf) - static_cast<size_t>(len),
                         " t=%zx %s ", th, stage);
    va_list ap;
    va_start(ap, fmt);
    len += std::vsnprintf(buf + len,
                          sizeof(buf) - static_cast<size_t>(len), fmt, ap);
    va_end(ap);
    if (len > 223) len = 223;
    buf[len++] = '\n';
    // write(2) to fd 2: unbuffered/lock-free so tracing perturbs thread interleaving
    // minimally and the last pre-crash line survives.  (fprintf+fflush on Windows.)
#ifdef _WIN32
    std::fwrite(buf, 1, static_cast<size_t>(len), stderr);
#else
    const ssize_t written = ::write(2, buf, static_cast<size_t>(len));
    (void)written;
#endif
  }


  inline uint32_t s_debugPushCount = 0;
  inline int s_lightLog = 0;

// Number of per-draw lighting UBO slots a frame will consume: each non-overlay
// command is drawn once, again when the overlay redraw is active (opaque,
// non-transparent only; tessellation redraws triangles, LINES redraws lines,
// POINTS everything), plus the on-top pass for depth-disabled commands; overlay
// commands draw a second time in the overlay block.  recordDrawCommand() bails
// before claiming a slot for skipped commands, so this worst case is a safe bound.
  inline uint32_t
countDrawCommands(const SoDrawList & drawlist, const int wireframeFillMode,
                  const bool tessellationOverlay)
{
  uint32_t draws = 0;
  const int num = drawlist.getNumCommands();
  for (int i = 0; i < num; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    ++draws;
    if ((wireframeFillMode >= 0 || tessellationOverlay) &&
        command.pass != SO_RENDERPASS_TRANSPARENT) {
      ++draws;
      const SoPrimitiveTopology topo = command.geometry.topology;
      const bool triTopo = topo == SO_TOPOLOGY_TRIANGLES ||
        topo == SO_TOPOLOGY_TRIANGLE_STRIP;
      // At most one overlay redraw per command (tessellation wins over fill-mode).
      if (tessellationOverlay && triTopo) {
        ++draws;
      }
      else if (wireframeFillMode >= 0) {
        ++draws;
      }
    }
    // The on-top pass re-records every depth-disabled command, consuming a second slot.
    if (!command.state.depth.enabled) {
      ++draws;
    }
  }
  for (int i = 0; i < num; ++i) {
    if (drawlist.getCommand(i).pass == SO_RENDERPASS_OVERLAY) ++draws;
  }
  return draws;
}

  inline uint32_t
countCompositeCommands(const SoDrawList & drawlist)
{
  // Overlay composite: every OVERLAY command plus every non-triangle
  // OPAQUE/TRANSPARENT command (BRep edge/point residue); one draw + one slot each.
  uint32_t draws = 0;
  const int num = drawlist.getNumCommands();
  for (int i = 0; i < num; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) {
      ++draws;
      continue;
    }
    const SoPrimitiveTopology topo = command.geometry.topology;
    if (topo == SO_TOPOLOGY_TRIANGLES || topo == SO_TOPOLOGY_TRIANGLE_STRIP) {
      continue;
    }
    ++draws;
  }
  return draws;
}

// --- Viewport / scissor coordinate helpers --------------------------------
// Coin/OpenGL rectangles are bottom-left anchored, Vulkan's top-left; this Y-flip (clamped) replaces six hand-copied sites.

struct FlippedRect {
  int32_t x0 = 0;
  int32_t y0 = 0;
  int32_t x1 = 0;
  int32_t y1 = 0;
};

// Clamp a bottom-left rect (origin x/y, size w/h) into a top-left target, flipping Y.
  inline FlippedRect
clampFlippedRect(const int32_t originX, const int32_t originY,
                 const int32_t width, const int32_t height,
                 const VkExtent2D & target)
{
  FlippedRect r;
  r.x0 = std::max(0, originX);
  r.y0 = std::max(0, static_cast<int32_t>(target.height) - originY - height);
  r.x1 = std::min(static_cast<int32_t>(target.width), originX + width);
  r.y1 = std::min(static_cast<int32_t>(target.height),
                  static_cast<int32_t>(target.height) - originY);
  return r;
}

  inline VkRect2D
toVkRect(const FlippedRect & r)
{
  VkRect2D rect {};
  rect.offset = {r.x0, r.y0};
  rect.extent = {static_cast<uint32_t>(std::max(0, r.x1 - r.x0)),
                 static_cast<uint32_t>(std::max(0, r.y1 - r.y0))};
  return rect;
}

// --- Wide-line predicate --------------------------------------------------
// CPU wide-line expansion is used when width > 1px or a stipple is present; the overlay redraw stays plain-path.

  inline bool
isPatternedLine(const SoRenderCommand & command)
{
  const uint16_t pattern = command.state.raster.linePattern;
  return pattern != 0xFFFF && pattern != 0;
}

  inline bool
isWideLine(const SoRenderCommand & command, const int fillModeOverride,
           const bool interactionLod = false)
{
  const SoPrimitiveTopology topology = command.geometry.topology;
  const bool lineTopology = topology == SO_TOPOLOGY_LINES ||
    topology == SO_TOPOLOGY_LINE_STRIP;
  if (!lineTopology || fillModeOverride >= 0) return false;
  // A stipple needs the expanded path regardless of width, so interaction LOD must not downgrade it.
  if (isPatternedLine(command)) return true;
  // During camera interaction draw wide lines as plain 1px GPU lines rather than expand
  // segments to quads; full width returns when the camera stops.
  return command.state.raster.lineWidth > 1.0f && !interactionLod;
}

  // True when a wide line is drawn by the GPU-instanced vertex shader rather than the
  // CPU quad expansion: a non-stippled LINE_LIST whose segments each reference their own
  // two vertices.  Uses the same per-draw view/projection the record path resolves.
  inline bool
instancedWideLineForceCpu()
{
  // COIN_VULKAN_WLINE_CPU forces CPU quad expansion for A/B and as a driver escape hatch.
  return SoVulkanConfig::get().raster.wideLineCpu;
}

  inline bool
isInstancedWideLine(const SoRenderCommand & command)
{
  return !instancedWideLineForceCpu() &&
    command.geometry.topology == SO_TOPOLOGY_LINES &&
    !isPatternedLine(command) &&
    command.state.raster.lineWidth > 1.0f;
}

// True when an overlay spans the whole frame viewport (selection/preselection highlight):
// frame-camera geometry using the frame matrices, not the command's own camera.
  inline bool
isFrameCameraOverlay(const SoRenderCommand & command,
                     const SoRenderParams & params)
{
  if (command.pass != SO_RENDERPASS_OVERLAY) return false;
  const SbVec2s frameSize = params.viewport.getViewportSizePixels();
  return command.state.raster.viewportWidth == frameSize[0] &&
    command.state.raster.viewportHeight == frameSize[1];
}

// FNV-1a over `sampleCount` elements spread uniformly across a buffer (first and last
// always included), folding bit patterns via `toBits`.  The producer arena reuses pointers
// for unchanged layouts, so sampling catches in-place edits cheaply.
  template <typename T, typename ToBits>
  inline uint64_t
hashSampled(const T * values, size_t count, size_t sampleCount, ToBits toBits)
{
  uint64_t hash = 1469598103934665603ULL;
  if (!values || count == 0) return hash;
  CoinRenderDetail::fnvMix(hash, static_cast<uint64_t>(count));
  if (count <= sampleCount) {
    for (size_t i = 0; i < count; ++i) {
      CoinRenderDetail::fnvMix(hash, toBits(values[i]));
    }
    return hash;
  }
  for (size_t s = 0; s < sampleCount; ++s) {
    const size_t i = s * (count - 1) / (sampleCount - 1);
    CoinRenderDetail::fnvMix(hash, toBits(values[i]));
  }
  return hash;
}

  inline uint64_t
hashFloats(const float * values, size_t count, size_t sampleCount)
{
  return hashSampled(values, count, sampleCount, [](const float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint64_t>(bits);
  });
}

  inline uint64_t
hashUint32(const uint32_t * values, size_t count, size_t sampleCount)
{
  return hashSampled(values, count, sampleCount, [](const uint32_t value) {
    return static_cast<uint64_t>(value);
  });
}

  inline uint64_t
hashGeometryContent(const SoGeometryDesc & geometry)
{
  uint64_t hash = 1469598103934665603ULL;
  const uint32_t vertexStride =
    geometry.vertexStride ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = vertexStride / sizeof(float);
  const size_t posCount =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  hash ^= hashFloats(geometry.positions, posCount, 1024);
  if (geometry.normals && geometry.normalCount > 0) {
    const size_t normalCount =
      static_cast<size_t>(geometry.normalCount) * posStrideFloats;
    hash ^= hashFloats(geometry.normals, normalCount, 512);
  }
  if (geometry.colors) {
    const size_t colorCount = static_cast<size_t>(geometry.vertexCount) * 4;
    hash ^= hashFloats(geometry.colors, colorCount, 512);
  }
  if (geometry.texcoords) {
    const uint32_t texcoordStrideFloats =
      (geometry.texcoordStride ? geometry.texcoordStride : sizeof(float) * 4)
      / sizeof(float);
    const size_t texcoordCount =
      static_cast<size_t>(geometry.vertexCount) * texcoordStrideFloats;
    hash ^= hashFloats(geometry.texcoords, texcoordCount, 512);
  }
  hash ^= hashUint32(geometry.indices, geometry.indexCount, 512);
  hash = hash * 1099511628211ULL ^
    static_cast<uint64_t>(geometry.vertexCount);
  hash = hash * 1099511628211ULL ^
    static_cast<uint64_t>(geometry.indexCount);
  hash = hash * 1099511628211ULL ^
    static_cast<uint64_t>(geometry.normalCount);
  hash = hash * 1099511628211ULL ^
    static_cast<uint64_t>(vertexStride);
  hash = hash * 1099511628211ULL ^
    static_cast<uint64_t>(geometry.texcoordStride);
  return hash;
}

  inline uint64_t
hashTextureContent(const SoTextureData & texture)
{
  uint64_t hash = 1469598103934665603ULL;
  if (texture.pixels) {
    const size_t count =
      static_cast<size_t>(texture.width) * texture.height *
      static_cast<size_t>(texture.numComponents);
    const size_t sampleCount = std::min<size_t>(4096, count);
    if (sampleCount > 0) {
      if (count <= sampleCount) {
        for (size_t i = 0; i < count; ++i) {
          CoinRenderDetail::fnvMix(hash, static_cast<uint64_t>(texture.pixels[i]));
        }
      }
      else {
        for (size_t s = 0; s < sampleCount; ++s) {
          const size_t i = s * (count - 1) / (sampleCount - 1);
          CoinRenderDetail::fnvMix(hash, static_cast<uint64_t>(texture.pixels[i]));
        }
      }
    }
  }
  CoinRenderDetail::fnvMix(hash, static_cast<uint64_t>(texture.width));
  CoinRenderDetail::fnvMix(hash, static_cast<uint64_t>(texture.height));
  CoinRenderDetail::fnvMix(hash, static_cast<uint64_t>(texture.numComponents));
  return hash;
}

// COIN_VULKAN_ENV_FLAG/envFlagEnabled() come from SoVulkanShared.h (via SoVulkanRenderBackend.h); don't redefine.

// Fixed 32-byte interleaved vertex layout shared by every retained command: pos f32 @0,
// normal f32 @12, color R8G8B8A8_UNORM @24, texcoord R16G16_SFLOAT @28.  Positions/normals
// stay f32 (CAD precision); color is 8-bit UNORM and texcoord half-float, cutting the
// per-vertex fetch from 48 to 32 bytes.  One static vertex-input description serves all.
constexpr uint32_t VULKAN_VERTEX_STRIDE = 32;
// Largest vertex count a single command may upload.  Flat-shaded CAD expands to 3 unique
// vertices per triangle, so a Voron-class assembly reaches tens of millions in one command;
// the old 10M ceiling silently dropped such objects.  Override COIN_VULKAN_MAX_VERTEX_COUNT.
constexpr int MAX_VERTEX_COUNT = 64000000;

// The projection matrix lives in the per-draw DrawBlock UBO, not here, so the push block
// stays at 112 B within the 128 B maxPushConstantsSize minimum (minimum-spec devices).
struct alignas(16) VulkanPushConstants {
  float color[4];       // offset 0: uniform diffuse color
  float flags[4];       // offset 16: x = useVertexColor
                        // y = vertexColorAlphaIncludesOpacity; z = textureEnabled; w = textureAlphaIncludesOpacity
  float texParams[4];   // offset 32: x = textureModel, y = alphaTestFunction,
                        // z = alphaTestReference
  float texBlend[4];    // offset 48: texture blend color
  float pointSize;      // offset 64: gl_PointSize (points/polygon mode)
  float pointSizePad[3];// std140: pointSize occupies a full vec4 slot
  float lineParams[4];  // offset 80: x = stipple factor (px/bit, glLineStipple
                        // factor), y = stipple bits (wide-line) / round points (visual),
                        // z = line primitive, w = point primitive
  float lineGeom[4];    // offset 96: x = line width (device px), y = viewport
                        // width, z = viewport height, w = device pixel ratio (instanced wide-line shader only).
};
static_assert(offsetof(VulkanPushConstants, lineParams) == 80,
              "lineParams must land at shader offset 80");
static_assert(offsetof(VulkanPushConstants, lineGeom) == 96,
              "lineGeom must land at shader offset 96");
static_assert(sizeof(VulkanPushConstants) == 112,
              "push-constant block must be 112 bytes (<= the 128-byte Vulkan "
              "minimum)");

// Push-constant block for the background gradient pass (BackgroundFragment.glsl).
struct alignas(16) VulkanBackgroundPush {
  float topColor[4];       // offset 0
  float bottomColor[4];    // offset 16
  float viewport[4];       // offset 32: x = width, y = height
};
static_assert(sizeof(VulkanBackgroundPush) == 48,
              "VulkanBackgroundPush must match BackgroundPush layout");

// The lighting constant block is the standardized SoLightingBlock (std140 mirror of
// the shaders' LightingBlock, set 0 binding 0), written once per frame into a small
// ring (SoRenderIR::fillLightingBlock() transforms setups to eye space with the frame
// view); each draw binds its slot by dynamic offset.  One slot normally holds the
// host-pushed set, else one per distinct lightingHandle.
using VulkanLightingUbo = SoLightingBlock;
static_assert(sizeof(VulkanLightingUbo) == 784,
              "VulkanLightingUbo must match LightingBlock std140 layout");

// std140 mirror of the DrawBlock uniform (set 1, binding 0): the per-draw
// view/model/material, bound by dynamic offset into the per-draw UBO ring.  The
// projection matrix lives here (not push constants) to keep the push block within
// the 128-byte Vulkan minimum.
struct alignas(16) VulkanDrawUbo {
  float view[16];                 // offset 0
  float model[16];                // offset 64
  float emissive[4];              // offset 128
  float materialAmbient[4];       // offset 144
  float materialSpecular[4];      // offset 160
  float materialParams[4];        // offset 176
  float proj[16];                 // offset 192
};
static_assert(sizeof(VulkanDrawUbo) == 256,
              "VulkanDrawUbo must match DrawBlock std140 layout");

  inline VkCompareOp
depthFunctionToVk(const SoDepthFunction function)
{
  switch (function) {
  case SO_DEPTH_NEVER: return VK_COMPARE_OP_NEVER;
  case SO_DEPTH_ALWAYS: return VK_COMPARE_OP_ALWAYS;
  case SO_DEPTH_LESS: return VK_COMPARE_OP_LESS;
  case SO_DEPTH_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
  case SO_DEPTH_EQUAL: return VK_COMPARE_OP_EQUAL;
  case SO_DEPTH_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case SO_DEPTH_GREATER: return VK_COMPARE_OP_GREATER;
  case SO_DEPTH_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
  default: return VK_COMPARE_OP_LESS_OR_EQUAL;
  }
}

  inline VkCompareOp
stencilFunctionToVk(const SoStencilFunction function)
{
  switch (function) {
  case SO_STENCIL_FUNC_NEVER: return VK_COMPARE_OP_NEVER;
  case SO_STENCIL_FUNC_ALWAYS: return VK_COMPARE_OP_ALWAYS;
  case SO_STENCIL_FUNC_LESS: return VK_COMPARE_OP_LESS;
  case SO_STENCIL_FUNC_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
  case SO_STENCIL_FUNC_EQUAL: return VK_COMPARE_OP_EQUAL;
  case SO_STENCIL_FUNC_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case SO_STENCIL_FUNC_GREATER: return VK_COMPARE_OP_GREATER;
  case SO_STENCIL_FUNC_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
  default: return VK_COMPARE_OP_ALWAYS;
  }
}

  inline VkStencilOp
stencilOpToVk(const SoStencilOp op)
{
  switch (op) {
  case SO_STENCIL_OP_ZERO: return VK_STENCIL_OP_ZERO;
  case SO_STENCIL_OP_REPLACE: return VK_STENCIL_OP_REPLACE;
  case SO_STENCIL_OP_INCREMENT: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
  case SO_STENCIL_OP_DECREMENT: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
  case SO_STENCIL_OP_INVERT: return VK_STENCIL_OP_INVERT;
  case SO_STENCIL_OP_INCREMENT_WRAP: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
  case SO_STENCIL_OP_DECREMENT_WRAP: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
  case SO_STENCIL_OP_KEEP:
  default: return VK_STENCIL_OP_KEEP;
  }
}

  inline VkBlendFactor
blendFactorToVk(const SoBlendFactor factor)
{
  switch (factor) {
  case SO_BLEND_FACTOR_ZERO: return VK_BLEND_FACTOR_ZERO;
  case SO_BLEND_FACTOR_ONE: return VK_BLEND_FACTOR_ONE;
  case SO_BLEND_FACTOR_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
  case SO_BLEND_FACTOR_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_DST_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case SO_BLEND_FACTOR_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case SO_BLEND_FACTOR_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  case SO_BLEND_FACTOR_CONSTANT_COLOR:
    return VK_BLEND_FACTOR_CONSTANT_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
  case SO_BLEND_FACTOR_CONSTANT_ALPHA:
    return VK_BLEND_FACTOR_CONSTANT_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
  case SO_BLEND_FACTOR_SRC_ALPHA_SATURATE:
    return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
  case SO_BLEND_FACTOR_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
  case SO_BLEND_FACTOR_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  default: return VK_BLEND_FACTOR_ONE;
  }
}

  inline VkBlendOp
blendEquationToVk(const SoBlendEquation equation)
{
  switch (equation) {
  case SO_BLEND_EQUATION_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
  case SO_BLEND_EQUATION_REVERSE_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
  case SO_BLEND_EQUATION_MIN: return VK_BLEND_OP_MIN;
  case SO_BLEND_EQUATION_MAX: return VK_BLEND_OP_MAX;
  case SO_BLEND_EQUATION_ADD:
  default: return VK_BLEND_OP_ADD;
  }
}

  inline VkPrimitiveTopology
topologyToVk(const SoPrimitiveTopology topology)
{
  switch (topology) {
  case SO_TOPOLOGY_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  case SO_TOPOLOGY_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  case SO_TOPOLOGY_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case SO_TOPOLOGY_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  case SO_TOPOLOGY_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
}

  inline VkFormat
textureFormatToVk(const int numComponents)
{
  switch (numComponents) {
  case 1: return VK_FORMAT_R8_UNORM;
  case 2: return VK_FORMAT_R8G8_UNORM;
  case 3: return VK_FORMAT_R8G8B8_UNORM;
  case 4:
  default: return VK_FORMAT_R8G8B8A8_UNORM;
  }
}

  inline VkFilter
textureFilterToVk(const SoTextureFilter filter)
{
  switch (filter) {
  case SO_TEXTURE_FILTER_NEAREST:
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    return VK_FILTER_NEAREST;
  case SO_TEXTURE_FILTER_LINEAR:
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
  default:
    return VK_FILTER_LINEAR;
  }
}

  inline VkSamplerAddressMode
textureWrapToVk(const SoTextureWrap wrap)
{
  switch (wrap) {
  case SO_TEXTURE_WRAP_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case SO_TEXTURE_WRAP_CLAMP_TO_EDGE:
  case SO_TEXTURE_WRAP_CLAMP_TO_BORDER:
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

  inline VkImageView
createImageView(VkDevice device,
                VkImage image,
                VkFormat format,
                VkImageAspectFlags aspect,
                const VkAllocationCallbacks * allocator)
{
  VkImageViewCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ci.image = image;
  ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ci.format = format;
  ci.subresourceRange.aspectMask = aspect;
  ci.subresourceRange.baseMipLevel = 0;
  ci.subresourceRange.levelCount = 1;
  ci.subresourceRange.baseArrayLayer = 0;
  ci.subresourceRange.layerCount = 1;
  VkImageView view = VK_NULL_HANDLE;
  const VkResult result = vkCreateImageView(device, &ci, allocator, &view);
  if (result != VK_SUCCESS) {
    SoDebugError::postWarning("CoinVulkanDetail::createImageView",
                              "vkCreateImageView failed (VkResult=%d)",
                              static_cast<int>(result));
    return VK_NULL_HANDLE;
  }
  return view;
}

} // namespace CoinVulkanDetail

#endif // COIN_SOVULKANRENDERBACKENDP_H
