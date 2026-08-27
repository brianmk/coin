// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h
//
// Private internal header for the Vulkan render backend.  Holds the helper
// code that was formerly the file-local anonymous namespace, lifted here so
// it can be shared across the split SoVulkanRenderBackend*.cpp translation
// units (in the CoinVulkanDetail namespace).  Provides:
//
//   - Debug counters + env-flag cache (envFlagEnabled)
//   - Push-constant / lighting-UBO structs (VulkanPushConstants,
//     VulkanBackgroundPush, VulkanVisualUbo)
//   - Vulkan enum-conversion helpers
//   - FNV content-hash helpers (hashFloats, hashUint32, hashGeometryContent,
//     hashTextureContent)
//   - Draw/overlay/composite command counters
//   - createImageView()

#ifndef COIN_SOVULKANRENDERBACKENDP_H
#define COIN_SOVULKANRENDERBACKENDP_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include <vulkan/vulkan.h>
#include <Inventor/rendering/SoRenderIR.h>

namespace CoinVulkanDetail {


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
countDrawCommands(const SoDrawList & drawlist, const int overlayFillMode)
{
  uint32_t draws = 0;
  const int num = drawlist.getNumCommands();
  for (int i = 0; i < num; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    ++draws;
    if (overlayFillMode >= 0 &&
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

// FNV-1a mixing step.
inline void
mix64(uint64_t & hash, uint64_t value)
{
  hash ^= value;
  hash *= 1099511628211ULL;
}

// FNV-1a over a float stream, sampling up to sampleCount elements spread
// uniformly across the buffer (the first and last elements are always
// included).  The producer's per-frame arena hands out the same pointers
// for unchanged layouts, so pointer identity alone cannot detect in-place
// content edits; the hash closes that hole at a fraction of the cost of a
// full scan.
  inline uint64_t
hashFloats(const float * values, size_t count, size_t sampleCount)
{
  uint64_t hash = 1469598103934665603ULL;
  if (!values || count == 0) return hash;
  mix64(hash, static_cast<uint64_t>(count));
  if (count <= sampleCount) {
    for (size_t i = 0; i < count; ++i) {
      uint32_t bits = 0;
      std::memcpy(&bits, &values[i], sizeof(bits));
      mix64(hash, static_cast<uint64_t>(bits));
    }
    return hash;
  }
  for (size_t s = 0; s < sampleCount; ++s) {
    const size_t i = s * (count - 1) / (sampleCount - 1);
    uint32_t bits = 0;
    std::memcpy(&bits, &values[i], sizeof(bits));
    mix64(hash, static_cast<uint64_t>(bits));
  }
  return hash;
}

  inline uint64_t
hashUint32(const uint32_t * values, size_t count, size_t sampleCount)
{
  uint64_t hash = 1469598103934665603ULL;
  if (!values || count == 0) return hash;
  mix64(hash, static_cast<uint64_t>(count));
  if (count <= sampleCount) {
    for (size_t i = 0; i < count; ++i) {
      mix64(hash, static_cast<uint64_t>(values[i]));
    }
    return hash;
  }
  for (size_t s = 0; s < sampleCount; ++s) {
    const size_t i = s * (count - 1) / (sampleCount - 1);
    mix64(hash, static_cast<uint64_t>(values[i]));
  }
  return hash;
}

  inline uint64_t
hashGeometryContent(const SoGeometryDesc & geometry)
{
  uint64_t hash = 1469598103934665603ULL;
  const uint32_t vertexStride =
    geometry.vertexStride ? geometry.vertexStride : sizeof(float) * 3;
  const size_t posCount =
    static_cast<size_t>(geometry.vertexCount) * vertexStride / sizeof(float);
  hash ^= hashFloats(geometry.positions, posCount, 1024);
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
          mix64(hash, static_cast<uint64_t>(texture.pixels[i]));
        }
      }
      else {
        for (size_t s = 0; s < sampleCount; ++s) {
          const size_t i = s * (count - 1) / (sampleCount - 1);
          mix64(hash, static_cast<uint64_t>(texture.pixels[i]));
        }
      }
    }
  }
  mix64(hash, static_cast<uint64_t>(texture.width));
  mix64(hash, static_cast<uint64_t>(texture.height));
  mix64(hash, static_cast<uint64_t>(texture.numComponents));
  return hash;
}

// Environment flags are enabled by presence, but honor the conventional
// "VAR=0"/"false"/"off" opt-out values.  Results are cached on first use:
// these are diagnostic switches, and the call sites sit in per-frame hot
// paths where a getenv() per call is pure overhead.
  inline bool
envFlagEnabled(const char * name)
{
  static std::unordered_map<std::string, bool> cache;
  const auto found = cache.find(name);
  if (found != cache.end()) return found->second;
  const char * value = getenv(name);
  const bool enabled = value != nullptr && std::strcmp(value, "0") != 0 &&
    std::strcmp(value, "false") != 0 && std::strcmp(value, "off") != 0;
  cache[name] = enabled;
  return enabled;
}

// Fixed interleaved vertex layout shared by every retained command.
//
//   offset 0 : vec3 position
//   offset 12: vec3 normal
//   offset 24: vec4 color
//   offset 40: vec2 texcoord
//
// This keeps a single static vertex-input description usable across all
// pipelines, mirroring the GL backend's VAO-per-command bookkeeping without
// any per-command vertex-state objects.
constexpr uint32_t VULKAN_VERTEX_STRIDE = 48;
constexpr int MAX_VERTEX_COUNT = 10000000;
constexpr int MAX_SHADER_LIGHTS = 8;

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

// std140 mirror of the VisualBlock uniform in Vertex.glsl.  The layout must
// match the shader byte-for-byte; the C++ side uses plain float arrays with
// alignas(16) so vec3 members consume 16 bytes like std140 vec4s.
struct alignas(16) VulkanVisualUbo {
  float view[16];                 // offset 0
  float model[16];                // offset 64
  float emissive[4];              // offset 128
  float ambientLight[4];          // offset 144
  float materialAmbient[4];       // offset 160
  float materialSpecular[4];      // offset 176
  float materialParams[4];        // offset 192
  float lightType[MAX_SHADER_LIGHTS * 4];        // offset 208
  float lightColor[MAX_SHADER_LIGHTS * 4];       // offset 336
  float lightDirection[MAX_SHADER_LIGHTS * 4];   // offset 464
  float lightPosition[MAX_SHADER_LIGHTS * 4];    // offset 592
  float lightAttenuation[MAX_SHADER_LIGHTS * 4]; // offset 720
  float lightSpotParams[MAX_SHADER_LIGHTS * 4];  // offset 848
};
static_assert(sizeof(VulkanVisualUbo) == 976,
              "VulkanVisualUbo must match VisualBlock std140 layout");

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
