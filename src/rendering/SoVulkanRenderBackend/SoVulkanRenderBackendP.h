// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h
//
// Private internal header for the Vulkan render backend.  Holds the helper
// code that was formerly the file-local anonymous namespace, lifted here so
// it can be shared across the split SoVulkanRenderBackend*.cpp translation
// units (in the CoinVulkanDetail namespace).  Provides:
//
//   - Debug counters
//   - Push-constant / lighting-UBO structs (VulkanPushConstants,
//     VulkanBackgroundPush, VulkanLightingUbo, VulkanDrawUbo)
//   - Vulkan enum-conversion helpers
//   - FNV content-hash helpers (hashFloats, hashUint32, hashGeometryContent,
//     hashTextureContent)
//   - Draw/overlay/composite command counters
//   - createImageView()

#ifndef COIN_SOVULKANRENDERBACKENDP_H
#define COIN_SOVULKANRENDERBACKENDP_H

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

#include <vulkan/vulkan.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <rendering/SoFnv1a.h>

#ifndef _WIN32
#include <unistd.h>
#endif

// Declared in SoRenderBackend.h; only referenced by the frame-stats helper
// below, so a forward declaration keeps this header from pulling the backend
// interface in.
struct SoRenderParams;

namespace CoinVulkanDetail {

  // ---- [TRC] per-step recording traces (FC_VULKAN_TRACE) ----
  // One line per pipeline step, tagged with a monotonic sequence, frame
  // index and thread id so an interleaved multi-thread log can be read in
  // order.  Gated by FC_VULKAN_TRACE, off by default, and cached (the
  // environment does not change mid-process).
  inline bool vkBackendTraceEnabled()
  {
    static const bool enabled = SoVulkanShared::envString("FC_VULKAN_TRACE") != nullptr;
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
    // write(2) to fd 2: unbuffered, lock-free syscalls so tracing perturbs
    // thread interleaving as little as possible and the final line before a
    // crash is never lost in a stdio buffer.  (fprintf + fflush fallback on
    // Windows, where fd 2 is not a POSIX descriptor.)
#ifdef _WIN32
    std::fwrite(buf, 1, static_cast<size_t>(len), stderr);
    std::fflush(stderr);
#else
    const ssize_t written = ::write(2, buf, static_cast<size_t>(len));
    (void)written;
#endif
  }


  inline int s_debugFrame = 0;
  inline uint32_t s_debugPushCount = 0;
  inline int s_dumpCmdCount = 0;
  inline int s_lightLog = 0;

// Number of per-draw lighting UBO slots a frame will consume.  A command is
// recorded once in its own pass, again when the wireframe/point overlay
// redraw is active (opaque commands only), and overlay commands are recorded
// a second time in the overlay block.  recordDrawCommand() bails out before
// claiming a slot for skipped commands, so this worst case is a safe upper
// bound.
  inline uint32_t
countDrawCommands(const SoDrawList & drawlist, const int wireframeFillMode)
{
  uint32_t draws = 0;
  const int num = drawlist.getNumCommands();
  for (int i = 0; i < num; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    ++draws;
    if (wireframeFillMode >= 0 &&
        command.pass != SO_RENDERPASS_TRANSPARENT) {
      ++draws;
    }
    // The on-top annotations pass re-records every depth-disabled command
    // after both passes (recordFrame), consuming a second lighting slot.
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
  // Ray-tracing composite: every OVERLAY command plus every non-triangle
  // OPAQUE/TRANSPARENT command (the BRep edge/point residue the RT backend
  // did not trace).  Each is recorded as one draw and consumes one lighting
  // slot, so the reservation must account for both.
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
// Coin/OpenGL viewport, scissor and clear rectangles are anchored at the
// bottom-left; Vulkan's are top-left.  Every site that converts one used to
// re-derive the same clamped, Y-flipped x0/y0/x1/y1 by hand (six copies), so
// the flip math lives here once.

struct FlippedRect {
  int32_t x0 = 0;
  int32_t y0 = 0;
  int32_t x1 = 0;
  int32_t y1 = 0;
};

// Clamp a bottom-left rectangle (origin x/y and size w/h) into a top-left
// target, flipping Y around the target height.  x0<=x1 and y0<=y1 always
// hold; an empty result has x0==x1 or y0==y1.
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
// A line is drawn through the CPU wide-line expansion path when its width
// exceeds 1px or it carries a stipple pattern.  The overlay wireframe/point
// redraw (fillModeOverride >= 0) stays on the plain line path.  Four copies
// of this rule existed; they now share one definition.

  inline bool
isPatternedLine(const SoRenderCommand & command)
{
  const uint16_t pattern = command.state.raster.linePattern;
  return pattern != 0xFFFF && pattern != 0;
}

  inline bool
isWideLine(const SoRenderCommand & command, const int fillModeOverride)
{
  const SoPrimitiveTopology topology = command.geometry.topology;
  const bool lineTopology = topology == SO_TOPOLOGY_LINES ||
    topology == SO_TOPOLOGY_LINE_STRIP;
  return lineTopology && fillModeOverride < 0 &&
    (command.state.raster.lineWidth > 1.0f || isPatternedLine(command));
}

// True when an overlay command spans the whole frame viewport (the selection/
// preselection highlight): such geometry is frame-camera geometry and must be
// projected/viewed with the frame matrices, not the command's own recorded
// camera.  Overlays that carry their own sub-viewport (the navigation cube)
// return false and keep their own camera.  The identical test lived in
// updateLightingUniforms() and recordDrawCommand().
  inline bool
isFrameCameraOverlay(const SoRenderCommand & command,
                     const SoRenderParams & params)
{
  if (command.pass != SO_RENDERPASS_OVERLAY) return false;
  const SbVec2s frameSize = params.viewport.getViewportSizePixels();
  return command.state.raster.viewportWidth == frameSize[0] &&
    command.state.raster.viewportHeight == frameSize[1];
}

// --- [BLACK] frame diagnostic ---------------------------------------------
// Gated by FC_VULKAN_BLACK_DEBUG, this counts the draw list by pass/topology
// and prints one line.  The identical counting loop + fprintf appeared in
// renderInternal() and recordFrame(); both call this now.

struct VulkanFrameStats {
  int tri = 0;
  int triLit = 0;
  int triUnlit = 0;
  int line = 0;
  int overlay = 0;
  int trans = 0;
};

  inline VulkanFrameStats
collectFrameStats(const SoDrawList & drawlist)
{
  VulkanFrameStats s;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & c = drawlist.getCommand(i);
    if (c.pass == SO_RENDERPASS_OVERLAY) s.overlay++;
    else if (c.pass == SO_RENDERPASS_TRANSPARENT) s.trans++;
    if (c.geometry.topology == SO_TOPOLOGY_TRIANGLES) {
      s.tri++;
      if (c.material.shadingModel == SO_SHADING_LEGACY_GOURAUD) s.triLit++;
      else s.triUnlit++;
    }
    if (c.geometry.topology == SO_TOPOLOGY_LINES ||
        c.geometry.topology == SO_TOPOLOGY_LINE_STRIP) {
      s.line++;
    }
  }
  return s;
}

// `overlaysOnly` is printed when >= 0 (renderInternal); pass -1 to omit it
// (recordFrame's line format).
  inline void
logBlackFrameStats(const SoDrawList & drawlist, const SoRenderParams & params,
                   const int frame, const int overlaysOnly)
{
  const VulkanFrameStats s = collectFrameStats(drawlist);
  if (overlaysOnly >= 0) {
    std::fprintf(stderr,
                 "[BLACK] frame=%d overlaysOnly=%d flags=0x%x "
                 "clear=(%.2f,%.2f,%.2f,%.2f) "
                 "cmds=%d tri=%d(lit=%d unlit=%d) line=%d overlay=%d trans=%d\n",
                 frame, overlaysOnly, static_cast<unsigned>(params.flags),
                 params.clearColor[0], params.clearColor[1],
                 params.clearColor[2], params.clearColor[3],
                 drawlist.getNumCommands(), s.tri, s.triLit, s.triUnlit,
                 s.line, s.overlay, s.trans);
  }
  else {
    std::fprintf(stderr,
                 "[BLACK] recordFrame frame=%d flags=0x%x "
                 "clear=(%.2f,%.2f,%.2f,%.2f) "
                 "cmds=%d tri=%d(lit=%d unlit=%d) line=%d overlay=%d trans=%d\n",
                 frame, static_cast<unsigned>(params.flags),
                 params.clearColor[0], params.clearColor[1],
                 params.clearColor[2], params.clearColor[3],
                 drawlist.getNumCommands(), s.tri, s.triLit, s.triUnlit,
                 s.line, s.overlay, s.trans);
  }
}

// FNV-1a over a float stream, sampling up to sampleCount elements spread
// uniformly across the buffer (the first and last elements are always
// included).  The producer's per-frame arena hands out the same pointers
// for unchanged layouts, so pointer identity alone cannot detect in-place
// content edits; the hash closes that hole at a fraction of the cost of a
// full scan.
// Sample `sampleCount` elements spread uniformly across a buffer (first and
// last always included) and fold their bit patterns into an FNV-1a hash.
// `toBits` converts one element to the uint64 the mixer consumes.  Shared by
// the float and uint32 entry points, which differ only in that conversion.
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

// COIN_VULKAN_ENV_FLAG and envFlagEnabled() are provided by SoVulkanShared.h,
// which every consumer of this header includes transitively (via
// SoVulkanRenderBackend.h).  Defining them here again would both redefine the
// macro (with a different body) and double the env-flag cache.

// Fixed interleaved vertex layout shared by every retained command.
//
//   offset 0 : vec3 position  (R32G32B32_SFLOAT)
//   offset 12: vec3 normal    (R32G32B32_SFLOAT)
//   offset 24: vec4 color     (R8G8B8A8_UNORM)
//   offset 28: vec2 texcoord  (R16G16_SFLOAT)
//
// Positions and normals stay full f32 (CAD geometry needs the precision for
// correct normals/lighting); the diffuse color is quantized to 8-bit UNORM and
// the texture coordinate to half-float, cutting the per-vertex fetch from 48
// to 32 bytes.  This keeps a single static vertex-input description usable
// across all pipelines, mirroring the GL backend's VAO-per-command bookkeeping
// without any per-command vertex-state objects.
constexpr uint32_t VULKAN_VERTEX_STRIDE = 32;
constexpr int MAX_VERTEX_COUNT = 10000000;

struct alignas(16) VulkanPushConstants {
  float proj[16];       // projection matrix (view/model live in the UBO)
  float color[4];       // uniform diffuse color
  float flags[4];       // x = useVertexColor
                        // y = vertexColorAlphaIncludesOpacity
                        // z = textureEnabled
                        // w = textureAlphaIncludesOpacity
  float texParams[4];   // x = textureModel, y = alphaTestFunction,
                        // z = alphaTestReference
  float texBlend[4];    // texture blend color
  float pointSize;      // gl_PointSize (point primitives and polygon mode)
  float pointSizePad[3];// std140: pointSize occupies a full vec4 slot
  float lineParams[4];  // x = stipple factor (px/bit, glLineStipple factor),
                        // y = stipple pattern bits (wide-line) / round
                        //     points (visual), z = line primitive,
                        // w = point primitive
};
static_assert(offsetof(VulkanPushConstants, lineParams) == 144,
              "lineParams must land at shader offset 144");
static_assert(sizeof(VulkanPushConstants) == 160,
              "push-constant block must be 160 bytes");

// Push-constant block for the background gradient pass (BackgroundFragment.glsl).
struct alignas(16) VulkanBackgroundPush {
  float topColor[4];       // offset 0
  float bottomColor[4];    // offset 16
  float viewport[4];       // offset 32: x = width, y = height
};
static_assert(sizeof(VulkanBackgroundPush) == 48,
              "VulkanBackgroundPush must match BackgroundPush layout");

// The lighting constant block is the standardized SoLightingBlock (the
// single authoritative std140 mirror of the visual shaders' LightingBlock
// uniform, set 0 binding 0).  It is written once per frame into a small ring
// -- world-space setups transformed to eye space by
// SoRenderIR::fillLightingBlock() with the frame view -- and every draw binds
// its slot through a dynamic offset, so a setup is never recomputed or
// re-written per draw.  Normally a single slot holds the host-pushed
// authoritative set (every handle maps to it); without one, a slot is packed
// per distinct lightingHandle.
using VulkanLightingUbo = SoLightingBlock;
static_assert(sizeof(VulkanLightingUbo) == 784,
              "VulkanLightingUbo must match LightingBlock std140 layout");

// std140 mirror of the DrawBlock uniform (set 1, binding 0) in the visual
// shaders.  This is the per-draw member that actually varies per command
// (view/model/material); it is pointed at through a dynamic offset into the
// per-draw UBO ring.
struct alignas(16) VulkanDrawUbo {
  float view[16];                 // offset 0
  float model[16];                // offset 64
  float emissive[4];              // offset 128
  float materialAmbient[4];       // offset 144
  float materialSpecular[4];      // offset 160
  float materialParams[4];        // offset 176
};
static_assert(sizeof(VulkanDrawUbo) == 192,
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
  vkCreateImageView(device, &ci, allocator, &view);
  return view;
}

} // namespace CoinVulkanDetail

#endif // COIN_SOVULKANRENDERBACKENDP_H
