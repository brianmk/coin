// src/rendering/SoVulkanRenderBackend.cpp

#include "rendering/SoVulkanRenderBackend.h"

#include <Inventor/C/glue/gl.h>
#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#include "vulkan/visual/Fragment.spv.h"
#include "vulkan/visual/Vertex.spv.h"
#include "vulkan/visual/WideLineFragment.spv.h"
#include "vulkan/visual/WideLineVertex.spv.h"
#include "vulkan/visual/BackgroundVertex.spv.h"
#include "vulkan/visual/BackgroundFragment.spv.h"

namespace {

int s_debugFrame = 0;
static uint32_t s_debugPushCount = 0;
int s_dumpCmdCount = 0;
static int s_lightLog = 0;

// Number of per-draw lighting UBO slots a frame will consume.  A command is
// recorded once in its own pass, again when the wireframe/point overlay
// redraw is active (opaque commands only), and overlay commands are recorded
// a second time in the overlay block.  recordDrawCommand() bails out before
// claiming a slot for skipped commands, so this worst case is a safe upper
// bound.
uint32_t
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

uint32_t
countOverlayCommands(const SoDrawList & drawlist)
{
  uint32_t draws = 0;
  const int num = drawlist.getNumCommands();
  for (int i = 0; i < num; ++i) {
    if (drawlist.getCommand(i).pass == SO_RENDERPASS_OVERLAY) ++draws;
  }
  return draws;
}

uint32_t
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
uint64_t
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

uint64_t
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

uint64_t
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

uint64_t
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
bool
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

VkCompareOp
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

VkCompareOp
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

VkStencilOp
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

VkBlendFactor
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

VkBlendOp
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

VkPrimitiveTopology
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

VkFormat
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

VkFilter
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

VkSamplerAddressMode
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

VkImageView
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

} // namespace

SoVulkanRenderBackend::SoVulkanRenderBackend()
{
  this->pendingDestroys.resize(this->maxFramesInFlight);
}

SoVulkanRenderBackend::~SoVulkanRenderBackend()
{
  if (this->isInitialized()) this->shutdown();
}

void
SoVulkanRenderBackend::setMaxFramesInFlight(const uint32_t count)
{
  if (count == 0) return;
  if (count == this->maxFramesInFlight) return;

  // Resizing while submissions are still in flight would free command
  // buffers/fences that a pending submission references and orphan the ring
  // batches below the new size; wait for every pending submission first.
  if (this->isInitialized()) {
    this->waitForInFlightFrames();
  }

  const size_t oldSize = this->pendingDestroys.size();
  for (size_t i = count; i < oldSize; ++i) {
    for (auto & fn : this->pendingDestroys[i]) {
      if (fn) fn();
    }
  }

  this->maxFramesInFlight = count;
  this->pendingDestroys.resize(count);

  // Re-allocate the per-frame-slot command buffers/fences at the new count.
  // Only valid while the queue is idle (guaranteed by the wait above).
  if (this->isInitialized()) {
    this->releaseFrameResources();
    if (!this->allocateFrameResources()) {
      this->emitError(
        "setMaxFramesInFlight: failed to reallocate frame resources");
    }
    // The lighting UBO ring is sized maxFramesInFlight * slotsPerFrame; grow
    // it to match the new in-flight count or the ring-offset math would run
    // past the allocation.  swapLightingBuffer() waits the (already idle)
    // in-flight frames and repoints every descriptor set at the new buffer.
    if (this->lightingBuffer != VK_NULL_HANDLE) {
      const VkDeviceSize totalBytes =
        static_cast<VkDeviceSize>(this->maxFramesInFlight) *
        this->uboSlotsPerFrame * this->uboSlotStride;
      VkBuffer newBuffer = VK_NULL_HANDLE;
      VkDeviceMemory newMemory = VK_NULL_HANDLE;
      void * newMapped = nullptr;
      if (!this->createBuffer(totalBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              newBuffer, newMemory, nullptr) ||
          vkMapMemory(this->device, newMemory, 0, totalBytes, 0, &newMapped) !=
            VK_SUCCESS) {
        this->emitError(
          "setMaxFramesInFlight: failed to resize lighting UBO");
        if (newBuffer != VK_NULL_HANDLE) {
          vkDestroyBuffer(this->device, newBuffer, this->allocator);
        }
        if (newMemory != VK_NULL_HANDLE) {
          vkFreeMemory(this->device, newMemory, this->allocator);
        }
      }
      else {
        this->swapLightingBuffer(newBuffer, newMemory, newMapped,
                                 this->uboSlotsPerFrame);
      }
    }
  }
}

const char *
SoVulkanRenderBackend::getName() const
{
  return "VulkanRenderBackend";
}

void
SoVulkanRenderBackend::setWireframeOverlay(SbBool enabled)
{
  this->wireframeOverlay = enabled;
}

void
SoVulkanRenderBackend::setPointsOverlay(SbBool enabled)
{
  this->pointsOverlay = enabled;
}

void
SoVulkanRenderBackend::setEdgeColor(const SbColor4f & color)
{
  this->edgeColor = color;
}

SbBool
SoVulkanRenderBackend::initialize(const SoRenderBackendInitParams & params)
{
  if (this->isInitialized()) return TRUE;

  this->setInitParams(params);
  const auto * deviceContext =
    static_cast<const SoVulkanDeviceContext *>(params.userData);
  if (!deviceContext || deviceContext->instance == VK_NULL_HANDLE ||
      deviceContext->physicalDevice == VK_NULL_HANDLE ||
      deviceContext->device == VK_NULL_HANDLE ||
      deviceContext->graphicsQueue == VK_NULL_HANDLE) {
    this->emitError(
      "SoVulkanRenderBackend requires a SoVulkanDeviceContext in "
      "SoRenderBackendInitParams::userData");
    return FALSE;
  }

  this->instance = deviceContext->instance;
  this->physicalDevice = deviceContext->physicalDevice;
  this->device = deviceContext->device;
  this->queue = deviceContext->graphicsQueue;
  this->queueFamilyIndex = deviceContext->graphicsQueueFamilyIndex;
  this->allocator = deviceContext->allocator;

  // Mark initialized before creating resources so that a failure in any
  // create*() below runs the full (null-tolerant) shutdown() cleanup
  // instead of leaking every handle created so far.
  this->setInitialized(TRUE);

  if (!this->createCommandPool()) {
    this->emitError("failed to create Vulkan command pool");
    this->shutdown();
    return FALSE;
  }

  if (!this->createDescriptorSetLayout()) {
    this->emitError("failed to create Vulkan descriptor set layout");
    this->shutdown();
    return FALSE;
  }

  if (!this->createDescriptorPool()) {
    this->emitError("failed to create Vulkan descriptor pool");
    this->shutdown();
    return FALSE;
  }

  if (!this->createLightingUniformBuffer()) {
    this->emitError("failed to create Vulkan lighting uniform buffer");
    this->shutdown();
    return FALSE;
  }

  if (!this->createWhiteTexture()) {
    this->emitError("failed to create Vulkan white fallback texture");
    this->shutdown();
    return FALSE;
  }

  if (!this->createPipelineLayout()) {
    this->emitError("failed to create Vulkan pipeline layout");
    this->shutdown();
    return FALSE;
  }

  if (!this->createShaders(this->vertexModule, this->fragmentModule)) {
    this->emitError("failed to create Vulkan shader modules");
    this->shutdown();
    return FALSE;
  }

  if (!this->createWideLineShaders()) {
    this->emitError("failed to create Vulkan wide-line shader modules");
    this->shutdown();
    return FALSE;
  }

  if (!this->createBackgroundResources()) {
    this->emitError("failed to create Vulkan background resources");
    this->shutdown();
    return FALSE;
  }

  this->emitLog("initialized");
  return TRUE;
}

bool
SoVulkanRenderBackend::createCommandPool()
{
  VkCommandPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
             VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  ci.queueFamilyIndex = this->queueFamilyIndex;
  if (vkCreateCommandPool(this->device, &ci, this->allocator,
                          &this->commandPool) != VK_SUCCESS) {
    return false;
  }
  return this->allocateFrameResources();
}

bool
SoVulkanRenderBackend::allocateFrameResources()
{
  if (this->commandPool == VK_NULL_HANDLE) return false;
  if (this->maxFramesInFlight == 0) return false;

  this->frameCommandBuffers.assign(this->maxFramesInFlight, VK_NULL_HANDLE);
  this->frameFences.assign(this->maxFramesInFlight, VK_NULL_HANDLE);
  this->frameFencePending.assign(this->maxFramesInFlight, 0);

  VkCommandBufferAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = this->commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = this->maxFramesInFlight;
  if (vkAllocateCommandBuffers(this->device, &ai,
                               this->frameCommandBuffers.data()) !=
      VK_SUCCESS) {
    return false;
  }

  VkFenceCreateInfo fi {};
  fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (VkFence & fence : this->frameFences) {
    if (vkCreateFence(this->device, &fi, this->allocator, &fence) !=
        VK_SUCCESS) {
      return false;
    }
  }
  return true;
}

void
SoVulkanRenderBackend::releaseFrameResources()
{
  // The caller must have made the queue idle (shutdown waits) or have waited
  // the pending fences (setMaxFramesInFlight) before this runs.
  for (VkCommandBuffer buffer : this->frameCommandBuffers) {
    if (buffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(this->device, this->commandPool, 1, &buffer);
    }
  }
  this->frameCommandBuffers.clear();
  for (VkFence fence : this->frameFences) {
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(this->device, fence, this->allocator);
    }
  }
  this->frameFences.clear();
  this->frameFencePending.clear();
}

VkCommandBuffer
SoVulkanRenderBackend::currentCommandBuffer()
{
  if (this->frameCommandBuffers.empty()) return VK_NULL_HANDLE;
  return this->frameCommandBuffers[this->uboFrameIndex %
                                   this->frameCommandBuffers.size()];
}

void
SoVulkanRenderBackend::waitForInFlightFrames()
{
  // Wait every fence that is actually pending.  Fences are only signaled by
  // endAndSubmit() on the own-queue path; the external path never submits
  // through this backend, so its fences are never signaled and must never be
  // waited on (waiting an unsignaled fence would block forever).  The
  // current frame's slot is never pending here (beginFrame() cleared it),
  // so growLightingUbo() cannot deadlock on the frame it is recording.
  // Called from growLightingUbo() and setMaxFramesInFlight(), both of which
  // must rewrite/teardown resources bound in submitted command buffers, so
  // this is a deliberately rare, synchronized event.
  std::vector<VkFence> pending;
  for (size_t i = 0; i < this->frameFences.size(); ++i) {
    if (i < this->frameFencePending.size() && this->frameFencePending[i] &&
        this->frameFences[i] != VK_NULL_HANDLE) {
      pending.push_back(this->frameFences[i]);
    }
  }
  if (pending.empty()) return;
  vkWaitForFences(this->device, static_cast<uint32_t>(pending.size()),
                  pending.data(), VK_TRUE, UINT64_MAX);
}

bool
SoVulkanRenderBackend::createDescriptorSetLayout()
{
  VkDescriptorSetLayoutBinding bindings[2] {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  bindings[0].descriptorCount = 1;
  // Lighting is evaluated per fragment (Phong), so the view/model/lighting
  // UBO must be visible to both stages.
  bindings[0].stageFlags =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[0].pImmutableSamplers = nullptr;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[1].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 2;
  ci.pBindings = bindings;
  return vkCreateDescriptorSetLayout(this->device, &ci, this->allocator,
                                     &this->descriptorSetLayout) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createDescriptorPool()
{
  VkDescriptorPoolSize poolSizes[2] {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  poolSizes[0].descriptorCount = 1024;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = 1024;

  VkDescriptorPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  ci.maxSets = 1024;
  ci.poolSizeCount = 2;
  ci.pPoolSizes = poolSizes;

  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(this->device, &ci, this->allocator, &pool) !=
      VK_SUCCESS) {
    return false;
  }
  this->descriptorPool = pool;
  this->descriptorPools.push_back(pool);
  return true;
}

bool
SoVulkanRenderBackend::allocateTextureDescriptorSet(VkImageView view,
                                                    VkSampler sampler,
                                                    VkDescriptorSet & set)
{
  VkDescriptorSetAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = this->descriptorPool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &this->descriptorSetLayout;
  if (vkAllocateDescriptorSets(this->device, &ai, &set) != VK_SUCCESS) {
    return false;
  }
  ++this->descriptorSetCount;

  VkDescriptorBufferInfo bufferInfo {};
  bufferInfo.buffer = this->lightingBuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = this->uboSlotStride;

  VkDescriptorImageInfo imageInfo {};
  imageInfo.sampler = sampler;
  imageInfo.imageView = view;
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkWriteDescriptorSet writes[2] {};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = set;
  writes[0].dstBinding = 0;
  writes[0].dstArrayElement = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  writes[0].pBufferInfo = &bufferInfo;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = set;
  writes[1].dstBinding = 1;
  writes[1].dstArrayElement = 0;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(this->device, 2, writes, 0, nullptr);
  return true;
}

bool
SoVulkanRenderBackend::createLightingUniformBuffer()
{
  // Per-command slots in a ring buffer sized for maxFramesInFlight frames.
  // Each draw binds its slot with a dynamic offset, so the GPU reads the
  // uniform block that was recorded for that specific draw instead of a
  // shared buffer that later commands overwrite.
  VkPhysicalDeviceProperties deviceProps;
  vkGetPhysicalDeviceProperties(this->physicalDevice, &deviceProps);
  const VkDeviceSize alignment = std::max<VkDeviceSize>(
    1, deviceProps.limits.minUniformBufferOffsetAlignment);
  this->uboSlotStride =
    (sizeof(VulkanVisualUbo) + alignment - 1) / alignment * alignment;
  this->uboSlotsPerFrame = 4096;
  const VkDeviceSize totalBytes =
    static_cast<VkDeviceSize>(this->maxFramesInFlight) *
    static_cast<VkDeviceSize>(this->uboSlotsPerFrame) * this->uboSlotStride;
  if (!this->createBuffer(totalBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          this->lightingBuffer, this->lightingMemory,
                          nullptr)) {
    return false;
  }
  if (vkMapMemory(this->device, this->lightingMemory, 0, totalBytes, 0,
                  &this->lightingMapped) != VK_SUCCESS) {
    this->emitError("createLightingUniformBuffer: vkMapMemory failed");
    vkDestroyBuffer(this->device, this->lightingBuffer, this->allocator);
    vkFreeMemory(this->device, this->lightingMemory, this->allocator);
    this->lightingBuffer = VK_NULL_HANDLE;
    this->lightingMemory = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::growLightingUbo(const uint32_t minSlots)
{
  uint32_t slots = 4096;
  while (slots < minSlots) slots <<= 1;
  if (slots <= this->uboSlotsPerFrame) return true;

  VkBuffer newBuffer = VK_NULL_HANDLE;
  VkDeviceMemory newMemory = VK_NULL_HANDLE;
  void * newMapped = nullptr;
  const VkDeviceSize totalBytes =
    static_cast<VkDeviceSize>(this->maxFramesInFlight) *
    static_cast<VkDeviceSize>(slots) * this->uboSlotStride;
  if (!this->createBuffer(totalBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          newBuffer, newMemory, nullptr)) {
    this->emitError("growLightingUbo: failed to allocate larger UBO");
    return false;
  }
  if (vkMapMemory(this->device, newMemory, 0, totalBytes, 0, &newMapped) !=
      VK_SUCCESS) {
    this->emitError("growLightingUbo: vkMapMemory failed");
    vkDestroyBuffer(this->device, newBuffer, this->allocator);
    vkFreeMemory(this->device, newMemory, this->allocator);
    return false;
  }

  return this->swapLightingBuffer(newBuffer, newMemory, newMapped, slots);
}

bool
SoVulkanRenderBackend::swapLightingBuffer(VkBuffer newBuffer,
                                          VkDeviceMemory newMemory,
                                          void * newMapped,
                                          const uint32_t newSlotsPerFrame)
{
  const VkBuffer oldBuffer = this->lightingBuffer;
  const VkDeviceMemory oldMemory = this->lightingMemory;
  void * oldMapped = this->lightingMapped;
  this->lightingBuffer = newBuffer;
  this->lightingMemory = newMemory;
  this->lightingMapped = newMapped;
  this->uboSlotsPerFrame = newSlotsPerFrame;

  // The old buffer may still be referenced by a pending frame; destroy it
  // only after the batch ring wraps back around (flushPendingDestroys()).
  const VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  this->deferDestroy([device, allocator, oldBuffer, oldMemory, oldMapped]() {
    if (oldMapped != nullptr) vkUnmapMemory(device, oldMemory);
    if (oldBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, oldBuffer, allocator);
    }
    if (oldMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, oldMemory, allocator);
    }
  });

  // Every descriptor set captured the old buffer handle at allocation time
  // (binding 0 is the lighting UBO).  Rewriting the binding of a set that is
  // bound in an already-submitted command buffer is a spec violation, so
  // first wait for every in-flight submission to complete.  This is a rare
  // event (ring growth or a maxFramesInFlight change), so the stall is
  // acceptable; the current frame's buffer has not been submitted yet, so
  // updating sets bound only in the recording buffer is safe.
  this->waitForInFlightFrames();
  std::vector<VkWriteDescriptorSet> writes;
  std::vector<VkDescriptorBufferInfo> bufferInfos;
  const auto collect = [&](const VkDescriptorSet set) {
    if (set == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo info {};
    info.buffer = this->lightingBuffer;
    info.offset = 0;
    info.range = this->uboSlotStride;
    bufferInfos.push_back(info);
    VkWriteDescriptorSet write {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    write.pBufferInfo = &bufferInfos.back();
    writes.push_back(write);
  };
  collect(this->whiteDescriptorSet);
  for (const VulkanCachedTexture & tex : this->textureCache) {
    collect(tex.descriptorSet);
  }
  if (!writes.empty()) {
    vkUpdateDescriptorSets(this->device,
                           static_cast<uint32_t>(writes.size()), writes.data(),
                           0, nullptr);
  }
  return true;
}

bool
SoVulkanRenderBackend::prepareLightingSlots(const uint32_t neededDraws)
{
  if (neededDraws > this->uboSlotsPerFrame) {
    if (!this->growLightingUbo(neededDraws)) return false;
  }
  // The frame index was advanced by beginFrame(); every render starts from
  // slot zero of its own ring half.
  this->uboCmdIndex = 0;
  return true;
}

void
SoVulkanRenderBackend::beginFrame()
{
  // One frame boundary: advance the ring cursor, then, on the own-queue
  // path, wait the slot's fence.  The slot we are about to record into was
  // last used maxFramesInFlight frames ago; its fence covers that
  // submission, so the slot's UBO ring half, command buffer, and deferred
  // resources are all safe to reuse.  The external path never signals these
  // fences (the caller owns submission), so frameFencePending stays false
  // there and no wait occurs -- external correctness rests on the caller
  // honoring setMaxFramesInFlight().
  this->uboFrameIndex++;
  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  if (slot < this->frameFencePending.size() &&
      this->frameFencePending[slot] &&
      this->frameFences[slot] != VK_NULL_HANDLE) {
    vkWaitForFences(this->device, 1, &this->frameFences[slot], VK_TRUE,
                    UINT64_MAX);
    vkResetFences(this->device, 1, &this->frameFences[slot]);
    this->frameFencePending[slot] = 0;
  }
  this->flushPendingDestroys();
}

void
SoVulkanRenderBackend::flushPendingDestroys()
{
  if (this->pendingDestroys.empty()) return;
  auto & batch =
    this->pendingDestroys[this->uboFrameIndex % this->pendingDestroys.size()];
  for (const auto & fn : batch) {
    if (fn) fn();
  }
  batch.clear();
}

void
SoVulkanRenderBackend::flushAllPendingDestroys()
{
  for (auto & batch : this->pendingDestroys) {
    for (const auto & fn : batch) {
      if (fn) fn();
    }
    batch.clear();
  }
}

void
SoVulkanRenderBackend::deferDestroy(std::function<void()> && fn)
{
  this->pendingDestroys[this->uboFrameIndex % this->pendingDestroys.size()]
    .push_back(std::move(fn));
}

void
SoVulkanRenderBackend::deferDestroyCacheEntry(VulkanCachedCommand & entry)
{
  if (entry.vertexBuffer == VK_NULL_HANDLE &&
      entry.indexBuffer == VK_NULL_HANDLE &&
      entry.wideLineBuffers.empty()) {
    entry = VulkanCachedCommand();
    return;
  }
  VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  const VkBuffer vertexBuffer = entry.vertexBuffer;
  const VkDeviceMemory vertexMemory = entry.vertexMemory;
  const VkBuffer indexBuffer = entry.indexBuffer;
  const VkDeviceMemory indexMemory = entry.indexMemory;
  std::vector<VulkanCachedCommand::VulkanWideLineBuffer> wideLine =
    std::move(entry.wideLineBuffers);
  this->deferDestroy(
    [device, allocator, vertexBuffer, vertexMemory, indexBuffer,
     indexMemory, wideLine]() {
      for (const VulkanCachedCommand::VulkanWideLineBuffer & slot : wideLine) {
        if (slot.buffer != VK_NULL_HANDLE) {
          vkDestroyBuffer(device, slot.buffer, allocator);
        }
        if (slot.memory != VK_NULL_HANDLE) {
          vkFreeMemory(device, slot.memory, allocator);
        }
      }
      if (indexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, indexBuffer, allocator);
      }
      if (indexMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, indexMemory, allocator);
      }
      if (vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, vertexBuffer, allocator);
      }
      if (vertexMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, vertexMemory, allocator);
      }
    });
  entry = VulkanCachedCommand();
}

void
SoVulkanRenderBackend::deferDestroyTextureEntry(VulkanCachedTexture & entry)
{
  // The set is only returned to its pool after the batch ring wraps back
  // around: a pending frame may still reference it, and vkFreeDescriptorSets
  // on an in-use set is a spec violation.  Pools are append-only (never
  // reset), so the pool handle captured here stays valid until shutdown.
  if (entry.descriptorSet != VK_NULL_HANDLE) {
    VkDevice device = this->device;
    const VkDescriptorPool pool = entry.descriptorPool;
    const VkDescriptorSet set = entry.descriptorSet;
    this->deferDestroy([device, pool, set]() {
      if (pool != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device, pool, 1, &set);
      }
    });
    if (this->descriptorSetCount > 0) --this->descriptorSetCount;
  }
  if (entry.image == VK_NULL_HANDLE) {
    entry = VulkanCachedTexture();
    return;
  }
  VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  const VkImage image = entry.image;
  const VkDeviceMemory memory = entry.memory;
  const VkImageView view = entry.view;
  const VkSampler sampler = entry.sampler;
  this->deferDestroy([device, allocator, image, memory, view, sampler]() {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, view, allocator);
    }
    if (sampler != VK_NULL_HANDLE) {
      vkDestroySampler(device, sampler, allocator);
    }
    if (image != VK_NULL_HANDLE) {
      vkDestroyImage(device, image, allocator);
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, allocator);
    }
  });
  entry = VulkanCachedTexture();
}

bool
SoVulkanRenderBackend::createWhiteTexture()
{
  const uint8_t white = 255;
  const uint32_t extent = 1;

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.extent = {extent, extent, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(this->device, &ci, this->allocator, &this->whiteImage) !=
      VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(this->device, this->whiteImage, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = 0;
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &props);
  bool found = false;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      ai.memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    this->emitError("createWhiteTexture: no device-local memory type");
    return false;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator,
                       &this->whiteImageMemory) != VK_SUCCESS) {
    return false;
  }
  vkBindImageMemory(this->device, this->whiteImage, this->whiteImageMemory, 0);

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging,
                          stagingMemory, &white)) {
    return false;
  }

  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer uploadBuffer = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(this->device, &allocInfo, &uploadBuffer) !=
      VK_SUCCESS) {
    this->emitError("createWhiteTexture: failed to allocate upload buffer");
    vkDestroyBuffer(this->device, staging, this->allocator);
    vkFreeMemory(this->device, stagingMemory, this->allocator);
    return false;
  }
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(uploadBuffer, &bi);

  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = this->whiteImage;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(uploadBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region {};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {extent, extent, 1};
  vkCmdCopyBufferToImage(uploadBuffer, staging, this->whiteImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(uploadBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);

  vkEndCommandBuffer(uploadBuffer);
  VkSubmitInfo submit {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &uploadBuffer;
  const VkResult submitResult =
    vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE);
  if (submitResult == VK_SUCCESS) {
    vkQueueWaitIdle(this->queue);
  }
  else {
    this->emitError("createWhiteTexture: vkQueueSubmit failed");
  }
  vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
  vkDestroyBuffer(this->device, staging, this->allocator);
  vkFreeMemory(this->device, stagingMemory, this->allocator);

  this->whiteImageView =
    createImageView(this->device, this->whiteImage, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (this->whiteImageView == VK_NULL_HANDLE) {
    return false;
  }

  SoTextureData fallback;
  fallback.minFilter = SO_TEXTURE_FILTER_NEAREST;
  fallback.magFilter = SO_TEXTURE_FILTER_NEAREST;
  fallback.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  fallback.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  if (!this->createSampler(fallback, this->whiteSampler)) {
    return false;
  }
  const bool allocated = this->allocateTextureDescriptorSet(
    this->whiteImageView, this->whiteSampler, this->whiteDescriptorSet);
  this->whiteDescriptorPool = this->descriptorPool;
  return allocated;
}

bool
SoVulkanRenderBackend::createPipelineLayout()
{
  // The visual push-constant block carries the projection matrix, colors,
  // texture state, point size, and line params.  Verify the device can hold
  // it (desktop GPUs advertise 256 bytes; some embedded parts only 128).
  VkPhysicalDeviceProperties deviceProps {};
  vkGetPhysicalDeviceProperties(this->physicalDevice, &deviceProps);
  if (deviceProps.limits.maxPushConstantsSize < sizeof(VulkanPushConstants)) {
    this->emitError(
      "device push-constant limit too small for the visual pipeline");
    return false;
  }

  constexpr VkPushConstantRange range {
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    0,
    sizeof(VulkanPushConstants)
  };

  VkPipelineLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = this->descriptorSetLayout != VK_NULL_HANDLE ? 1u : 0u;
  ci.pSetLayouts = &this->descriptorSetLayout;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &range;
  return vkCreatePipelineLayout(this->device, &ci, this->allocator,
                                &this->pipelineLayout) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createShaders(VkShaderModule & vertex,
                                     VkShaderModule & fragment)
{
  auto load = [this](const uint32_t * code, size_t count,
                     VkShaderModule & module) {
    VkShaderModuleCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = count * sizeof(uint32_t);
    ci.pCode = code;
    return vkCreateShaderModule(this->device, &ci, this->allocator,
                                &module) == VK_SUCCESS;
  };

  vertex = VK_NULL_HANDLE;
  fragment = VK_NULL_HANDLE;
  if (!load(coin_vulkan_visual_vertex_spirv,
            coin_vulkan_visual_vertex_spirv_count, vertex)) {
    return false;
  }
  if (!load(coin_vulkan_visual_fragment_spirv,
            coin_vulkan_visual_fragment_spirv_count, fragment)) {
    vkDestroyShaderModule(this->device, vertex, this->allocator);
    vertex = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createWideLineShaders()
{
  auto load = [this](const uint32_t * code, size_t count,
                     VkShaderModule & module) {
    VkShaderModuleCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = count * sizeof(uint32_t);
    ci.pCode = code;
    return vkCreateShaderModule(this->device, &ci, this->allocator,
                                &module) == VK_SUCCESS;
  };

  if (!load(coin_vulkan_wide_line_vertex_spirv,
            coin_vulkan_wide_line_vertex_spirv_count,
            this->wideLineVertexModule)) {
    return false;
  }
  if (!load(coin_vulkan_wide_line_fragment_spirv,
            coin_vulkan_wide_line_fragment_spirv_count,
            this->wideLineFragmentModule)) {
    vkDestroyShaderModule(this->device, this->wideLineVertexModule,
                          this->allocator);
    this->wideLineVertexModule = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createBackgroundResources()
{
  if (getenv("FC_VULKAN_BREADCRUMBS")) {
    fprintf(stderr, "[VK-TRACE] SoVulkanRenderBackend::createBackgroundResources enter\n");
  }
  auto load = [this](const uint32_t * code, size_t count,
                     VkShaderModule & module) {
    VkShaderModuleCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = count * sizeof(uint32_t);
    ci.pCode = code;
    return vkCreateShaderModule(this->device, &ci, this->allocator,
                                &module) == VK_SUCCESS;
  };

  if (!load(coin_vulkan_background_vertex_spirv,
            coin_vulkan_background_vertex_spirv_count,
            this->backgroundVertexModule)) {
    return false;
  }
  if (!load(coin_vulkan_background_fragment_spirv,
            coin_vulkan_background_fragment_spirv_count,
            this->backgroundFragmentModule)) {
    vkDestroyShaderModule(this->device, this->backgroundVertexModule,
                          this->allocator);
    this->backgroundVertexModule = VK_NULL_HANDLE;
    return false;
  }

  // Push-constant-only layout: the gradient shader has no descriptor sets.
  constexpr VkPushConstantRange range {
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    0,
    sizeof(VulkanBackgroundPush)
  };
  VkPipelineLayoutCreateInfo li {};
  li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  li.setLayoutCount = 0;
  li.pSetLayouts = nullptr;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &range;
  return vkCreatePipelineLayout(this->device, &li, this->allocator,
                                &this->backgroundPipelineLayout) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createBackgroundPipeline(
  const SoVulkanRenderTarget & target,
  VkRenderPass renderPass,
  VkPipeline & pipeline)
{
  BackgroundPipelineKey key;
  key.renderPass = renderPass;
  key.sampleCount = target.sampleCount;
  const auto found = this->backgroundPipelineCache.find(key);
  if (found != this->backgroundPipelineCache.end()) {
    pipeline = found->second;
    return pipeline != VK_NULL_HANDLE;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = this->backgroundVertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = this->backgroundFragmentModule;
  stages[1].pName = "main";

  // Fullscreen triangle: no vertex inputs.
  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 0;
  vertexInput.vertexAttributeDescriptionCount = 0;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  rasterization.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization.cullMode = VK_CULL_MODE_NONE;
  rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType =
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = target.sampleCount;

  // The gradient fills the whole viewport and writes no depth so geometry
  // drawn afterwards is unaffected.
  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_FALSE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType =
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.logicOpEnable = VK_FALSE;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkGraphicsPipelineCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci.stageCount = 2;
  ci.pStages = stages;
  ci.pVertexInputState = &vertexInput;
  ci.pInputAssemblyState = &inputAssembly;
  ci.pViewportState = &viewportState;
  ci.pRasterizationState = &rasterization;
  ci.pMultisampleState = &multisample;
  ci.pDepthStencilState = &depthStencil;
  ci.pColorBlendState = &colorBlend;
  ci.pDynamicState = &dynamicState;
  ci.layout = this->backgroundPipelineLayout;
  ci.renderPass = renderPass;
  ci.subpass = 0;

  VkPipeline created = VK_NULL_HANDLE;
  const VkResult result =
    vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &ci,
                              this->allocator, &created);
  if (result != VK_SUCCESS) {
    this->emitError("failed to create Vulkan background pipeline");
    this->backgroundPipelineCache[key] = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->backgroundPipelineCache[key] = created;
  pipeline = created;
  return true;
}

void
SoVulkanRenderBackend::recordBackground(const SoRenderParams & params,
                                        const SoVulkanRenderTarget & target,
                                        VkRenderPass renderPass)
{
  if (!params.backgroundGradient) {
    return;
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->createBackgroundPipeline(target, renderPass, pipeline) ||
      pipeline == VK_NULL_HANDLE) {
    return;
  }

  // The gradient covers exactly the viewport region (same Y-flip math as
  // applyViewport()); geometry drawn afterwards restores its own viewport.
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(origin[1]) -
         static_cast<int32_t>(size[1]));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(origin[0]) +
                                static_cast<int32_t>(size[0]));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(origin[1]));
  const int32_t w = std::max(0, x1 - x0);
  const int32_t h = std::max(0, y1 - y0);
  if (w == 0 || h == 0) return;

  VkViewport viewport {};
  viewport.x = static_cast<float>(x0);
  viewport.y = static_cast<float>(y0);
  viewport.width = static_cast<float>(w);
  viewport.height = static_cast<float>(h);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(this->activeCommandBuffer, 0, 1, &viewport);

  VkRect2D scissor {};
  scissor.offset = {x0, y0};
  scissor.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);

  vkCmdBindPipeline(this->activeCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline);

  VulkanBackgroundPush push {};
  push.topColor[0] = params.backgroundTopColor[0];
  push.topColor[1] = params.backgroundTopColor[1];
  push.topColor[2] = params.backgroundTopColor[2];
  push.topColor[3] = params.backgroundTopColor[3];
  push.bottomColor[0] = params.backgroundBottomColor[0];
  push.bottomColor[1] = params.backgroundBottomColor[1];
  push.bottomColor[2] = params.backgroundBottomColor[2];
  push.bottomColor[3] = params.backgroundBottomColor[3];
  push.viewport[0] = static_cast<float>(w);
  push.viewport[1] = static_cast<float>(h);
  push.viewport[2] = static_cast<float>(x0);
  push.viewport[3] = static_cast<float>(y0);
  vkCmdPushConstants(this->activeCommandBuffer, this->backgroundPipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);

  vkCmdDraw(this->activeCommandBuffer, 3, 1, 0, 0);
}

bool
SoVulkanRenderBackend::createRenderPass(const SoVulkanRenderTarget & target,
                                        VkRenderPass & pass)
{
  VkAttachmentDescription attachments[2];
  uint32_t attachmentCount = 1;

  attachments[0].flags = 0;
  attachments[0].format = target.colorFormat;
  attachments[0].samples = target.sampleCount;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = target.colorLayout;
  attachments[0].finalLayout = target.colorLayout;

  VkAttachmentReference colorRef {};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef {};
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  if (hasDepth) {
    attachments[1].flags = 0;
    attachments[1].format = target.depthFormat;
    attachments[1].samples = target.sampleCount;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = target.depthLayout;
    attachments[1].finalLayout = target.depthLayout;
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachmentCount = 2;
  }

  VkSubpassDescription subpass {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

  VkRenderPassCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = attachmentCount;
  ci.pAttachments = attachments;
  ci.subpassCount = 1;
  ci.pSubpasses = &subpass;
  ci.dependencyCount = 0;
  ci.pDependencies = nullptr;

  return vkCreateRenderPass(this->device, &ci, this->allocator, &pass) ==
         VK_SUCCESS;
}

SoVulkanRenderBackend::RenderPassIdentity
SoVulkanRenderBackend::renderPassIdentity(const SoVulkanRenderTarget & target) const
{
  RenderPassIdentity identity;
  identity.colorFormat = target.colorFormat;
  identity.sampleCount = target.sampleCount;
  identity.colorLayout = target.colorLayout;
  // createRenderPass() only adds a depth attachment when a depth view is
  // present, so a configured-but-viewless depth format must not be part of
  // the identity.
  identity.depthFormat =
    (target.depthImageView != VK_NULL_HANDLE) ? target.depthFormat
                                              : VK_FORMAT_UNDEFINED;
  identity.depthLayout = target.depthLayout;
  return identity;
}

VkRenderPass
SoVulkanRenderBackend::getOrCreateRenderPass(const SoVulkanRenderTarget & target)
{
  const RenderPassIdentity identity = this->renderPassIdentity(target);
  const auto found = this->renderPassCache.find(identity);
  if (found != this->renderPassCache.end()) return found->second;

  VkRenderPass pass = VK_NULL_HANDLE;
  if (!this->createRenderPass(target, pass)) return VK_NULL_HANDLE;
  this->renderPassCache.emplace(identity, pass);
  return pass;
}

bool
SoVulkanRenderBackend::getOrCreatePipeline(const SoRenderCommand & command,
                                           const SoVulkanRenderTarget & target,
                                           VkRenderPass pass,
                                           VkPipeline & pipeline,
                                           const bool transparent,
                                           const int fillModeOverride,
                                           const bool overlayPass)
{
  // Pipelines are immutable in Vulkan.  Key the cache on every retained
  // state value that changes the created pipeline so commands of different
  // topology, fill mode, depth/blend state, or sample count never reuse an
  // incompatible pipeline.  Shading model, vertex-color, texture, and
  // lighting remain uniform/push-constant concerns in this milestone and do
  // not need to participate in the key yet.
  const bool blending = transparent || command.state.blend.enabled ||
                        command.material.diffuse[3] < 0.999f;
  const bool overlay = fillModeOverride >= 0;
  // SoPolygonOffsetElement contributes an explicit depth bias captured into
  // the raster state.  Selection/overlay faces use it to pull themselves in
  // front of the coplanar base geometry (GL glPolygonOffset semantics).
  // Respect it in the key so selection overlays stop z-fighting with the
  // geometry underneath them.
  const bool polygonOffset =
    command.state.raster.polygonOffsetFactor != 0.0f ||
    command.state.raster.polygonOffsetUnits != 0.0f;
  const bool depthBias = overlay || polygonOffset;
  // GL polygon-offset units map ~1:1 onto Vulkan's depthBiasConstantFactor,
  // but the two differ in how `r` (the minimum resolvable depth step) is
  // derived: GL uses the (fixed-point, 24-bit) depth range while this backend
  // commonly owns a float (D32_SFLOAT) depth attachment, whose resolvable
  // step is far finer.  A GL-sized offset therefore leaves the coplanar
  // selection/hover overlay Z-fighting with the base (a dark seam along the
  // face boundary).  Scale the GL decal up so the overlay wins the depth test
  // decisively; the slope factor keeps it from detaching at grazing,
  // silhouette edges.
  constexpr float kDecalScale = 512.0f;
  const float kUseDecal = envFlagEnabled("FC_VULKAN_RASTER_DECAL")
    ? kDecalScale : 1.0f;
  const float depthBiasConstant = polygonOffset
    ? command.state.raster.polygonOffsetUnits * kUseDecal
    : (overlay ? -0.5f : 0.0f);
  const float depthBiasSlope = polygonOffset
    ? command.state.raster.polygonOffsetFactor * kUseDecal
    : (overlay ? -0.5f : 0.0f);
  PipelineKey key;
  // Wide-line rendering (line width > 1 and/or a stipple pattern) expands
  // segments into quads on the CPU and draws them with the wide-line
  // pipeline as triangle lists, mirroring the GL wide-line geometry shader.
  // The overlay wireframe redraw stays on the plain line path.
  const bool lineTopology = command.geometry.topology == SO_TOPOLOGY_LINES ||
    command.geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  const bool patternedLine =
    command.state.raster.linePattern != 0xFFFF &&
    command.state.raster.linePattern != 0;
  const bool useWideLine =
    lineTopology && fillModeOverride < 0 &&
    (command.state.raster.lineWidth > 1.0f || patternedLine);
  key.wideLine = useWideLine;
  key.renderPass = pass;
  key.topology = command.geometry.topology;
  key.fillMode = overlay ? static_cast<uint8_t>(fillModeOverride)
                          : command.state.raster.fillMode;
  key.cullMode = overlay ? 0 : command.state.raster.cullMode;
  key.ccwFrontFace = command.state.raster.ccwFrontFace;
  key.depthTestEnable = command.state.depth.enabled || overlay;
  // Overlay-pass geometry (e.g. the navigation cube) draws last into its own
  // viewport and keeps depth writes so it can self-occlude correctly; the
  // wireframe/point redraw overlays deliberately disable depth writes.
  key.depthWriteEnable = overlayPass
    ? command.state.depth.writeEnabled
    : (!transparent && !overlay && command.state.depth.writeEnabled);
  key.depthFunction = overlayPass ? static_cast<uint8_t>(command.state.depth.func)
                                  : (overlay ? static_cast<uint8_t>(SO_DEPTH_LEQUAL)
                                             : command.state.depth.func);
  key.depthBiasEnable = depthBias;
  key.depthBiasConstantFactor = depthBiasConstant;
  key.depthBiasSlopeFactor = depthBiasSlope;
  key.blendEnable = blending;
  key.sampleCount = target.sampleCount;
  if (blending) {
    key.blendSrcRGB = command.state.blend.srcRGBFactor;
    key.blendDstRGB = command.state.blend.dstRGBFactor;
    key.blendSrcAlpha = command.state.blend.srcAlphaFactor;
    key.blendDstAlpha = command.state.blend.dstAlphaFactor;
    key.blendEquationRGB = command.state.blend.rgbEquation;
    key.blendEquationAlpha = command.state.blend.alphaEquation;
  }
  const SoStencilState & stencil = command.state.stencil;
  key.stencilEnable = stencil.enabled;
  if (stencil.enabled) {
    key.stencilFunction = stencil.function;
    key.stencilReference = stencil.reference;
    key.stencilCompareMask = stencil.compareMask;
    key.stencilWriteMask = stencil.writeMask;
    key.stencilFailOp = stencil.failOp;
    key.stencilZFailOp = stencil.zfailOp;
    key.stencilZPassOp = stencil.zpassOp;
  }

  const auto found = this->pipelineCache.find(key);
  if (found != this->pipelineCache.end()) {
    pipeline = found->second;
    return pipeline != VK_NULL_HANDLE;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = key.wideLine ? this->wideLineVertexModule
                                  : this->vertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = key.wideLine ? this->wideLineFragmentModule
                                  : this->fragmentModule;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding {};
  binding.binding = 0;
  binding.stride = key.wideLine ? 36u : VULKAN_VERTEX_STRIDE;
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributes[4] {};
  attributes[0].location = 0;
  attributes[0].binding = 0;
  attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset = 0;
  attributes[1].location = 1;
  attributes[1].binding = 0;
  attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset = 12;
  attributes[2].location = 2;
  attributes[2].binding = 0;
  attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[2].offset = 24;
  attributes[3].location = 3;
  attributes[3].binding = 0;
  attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
  attributes[3].offset = 40;

  // Wide-line layout: clip-space position (0), color (16), polyline
  // distance (32).
  VkVertexInputAttributeDescription wideLineAttributes[3] {};
  wideLineAttributes[0].location = 0;
  wideLineAttributes[0].binding = 0;
  wideLineAttributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  wideLineAttributes[0].offset = 0;
  wideLineAttributes[1].location = 2;
  wideLineAttributes[1].binding = 0;
  wideLineAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  wideLineAttributes[1].offset = 16;
  wideLineAttributes[2].location = 4;
  wideLineAttributes[2].binding = 0;
  wideLineAttributes[2].format = VK_FORMAT_R32_SFLOAT;
  wideLineAttributes[2].offset = 32;

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount = key.wideLine ? 3u : 4u;
  vertexInput.pVertexAttributeDescriptions =
    key.wideLine ? wideLineAttributes : attributes;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = key.wideLine
    ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    : topologyToVk(command.geometry.topology);
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  const uint8_t fillMode = fillModeOverride >= 0
                             ? static_cast<uint8_t>(fillModeOverride)
                             : command.state.raster.fillMode;
  // The overlay fill mode passed in by recordFrame() uses SoDrawStyleElement
  // style values, and the retained IR stores the same encoding (see
  // SoRenderIR::fillRenderStateFromState): FILLED=0, LINES=1, POINTS=2.
  //
  // Wide lines expand each segment into FILLED quads (drawn as a triangle
  // list), so the polygon mode must be FILL regardless of the underlying
  // draw style.  Using the inherited LINES mode here rasterizes the quad's
  // edges as hairline wireframe instead of the solid line, which makes the
  // expanded quads (a few pixels wide) effectively invisible -- the
  // "wide lines don't render" symptom.
  rasterization.polygonMode =
    key.wideLine ? VK_POLYGON_MODE_FILL
    : fillMode == SoDrawStyleElement::LINES ? VK_POLYGON_MODE_LINE
    : fillMode == SoDrawStyleElement::POINTS ? VK_POLYGON_MODE_POINT
    : VK_POLYGON_MODE_FILL;
  // The vertex shader flips Y to match Coin's bottom-left origin; that
  // reflection reverses screen winding, so the Vulkan front face is the
  // inverse of the GL vertex ordering captured in the IR.  Back-face
  // culling matches GL: only shapes declaring an explicit winding plus
  // SOLID shape type cull (ccwFrontFace/cullMode above).  FreeCAD BRep
  // tessellations declare COUNTERCLOCKWISE/SOLID, so closed parts cull
  // back faces here exactly like the GL pipeline does.
  rasterization.cullMode =
    key.wideLine || !key.cullMode ? VK_CULL_MODE_NONE
                                  : VK_CULL_MODE_BACK_BIT;
  rasterization.frontFace = key.ccwFrontFace
    ? VK_FRONT_FACE_CLOCKWISE
    : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;
  // Depth bias: wireframe/point overlays pull toward the camera so they pass
  // the depth test against coplanar filled geometry; selection/overlay faces
  // carry an explicit SoPolygonOffsetElement captured into the raster state.
  rasterization.depthBiasEnable = depthBias ? VK_TRUE : VK_FALSE;
  rasterization.depthBiasConstantFactor = depthBiasConstant;
  rasterization.depthBiasSlopeFactor = depthBiasSlope;

  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType =
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = target.sampleCount;

  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable =
    (command.state.depth.enabled || overlay) ? VK_TRUE : VK_FALSE;
  depthStencil.depthWriteEnable =
    (!transparent && !overlay && command.state.depth.writeEnabled)
      ? VK_TRUE : VK_FALSE;
  depthStencil.depthCompareOp = overlay
    ? VK_COMPARE_OP_LESS_OR_EQUAL
    : depthFunctionToVk(command.state.depth.func);
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = stencil.enabled ? VK_TRUE : VK_FALSE;
  VkStencilOpState stencilState {};
  if (stencil.enabled) {
    stencilState.failOp = stencilOpToVk(stencil.failOp);
    stencilState.passOp = stencilOpToVk(stencil.zpassOp);
    stencilState.depthFailOp = stencilOpToVk(stencil.zfailOp);
    stencilState.compareOp = stencilFunctionToVk(stencil.function);
    stencilState.compareMask = stencil.compareMask;
    stencilState.writeMask = stencil.writeMask;
    stencilState.reference = stencil.reference;
  }
  depthStencil.front = stencilState;
  depthStencil.back = stencilState;

  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendAttachment.blendEnable = blending ? VK_TRUE : VK_FALSE;
  if (command.state.blend.enabled) {
    blendAttachment.srcColorBlendFactor =
      blendFactorToVk(command.state.blend.srcRGBFactor);
    blendAttachment.dstColorBlendFactor =
      blendFactorToVk(command.state.blend.dstRGBFactor);
    blendAttachment.colorBlendOp =
      blendEquationToVk(command.state.blend.rgbEquation);
    blendAttachment.srcAlphaBlendFactor =
      blendFactorToVk(command.state.blend.srcAlphaFactor);
    blendAttachment.dstAlphaBlendFactor =
      blendFactorToVk(command.state.blend.dstAlphaFactor);
    blendAttachment.alphaBlendOp =
      blendEquationToVk(command.state.blend.alphaEquation);
  }
  else {
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  }

  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType =
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.logicOpEnable = VK_FALSE;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkGraphicsPipelineCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci.stageCount = 2;
  ci.pStages = stages;
  ci.pVertexInputState = &vertexInput;
  ci.pInputAssemblyState = &inputAssembly;
  ci.pViewportState = &viewportState;
  ci.pRasterizationState = &rasterization;
  ci.pMultisampleState = &multisample;
  ci.pDepthStencilState = &depthStencil;
  ci.pColorBlendState = &colorBlend;
  ci.pDynamicState = &dynamicState;
  ci.layout = this->pipelineLayout;
  ci.renderPass = pass;
  ci.subpass = 0;

  VkPipeline created = VK_NULL_HANDLE;
  const VkResult result =
    vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &ci,
                              this->allocator, &created);
  if (result != VK_SUCCESS) {
    this->emitError("failed to create Vulkan graphics pipeline");
    this->pipelineCache[key] = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->pipelineCache[key] = created;
  pipeline = created;
  return true;
}

// --- Geometry cache -------------------------------------------------------

VulkanCachedCommand &
SoVulkanRenderBackend::getOrCreateCache(const SoRenderCommand * command)
{
  const auto found = this->commandToCache.find(command);
  if (found != this->commandToCache.end()) {
    return this->gpuCache[found->second];
  }
  const size_t index = this->gpuCache.size();
  this->gpuCache.emplace_back();
  this->gpuCache.back().commandKey = command;
  this->commandToCache[command] = index;
  return this->gpuCache.back();
}

bool
SoVulkanRenderBackend::createBuffer(VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkBuffer & buffer,
                                    VkDeviceMemory & memory,
                                    const void * data)
{
  VkBufferCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(this->device, &ci, this->allocator, &buffer) !=
      VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(this->device, buffer, &requirements);

  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  // Milestone: no per-memory-type discovery; assume a host-visible coherent
  // heap.  A follow-up replaces this with a memory-type index lookup.
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &memoryProperties);
  uint32_t memoryTypeIndex = 0;
  const VkMemoryPropertyFlags desired =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  bool found = false;
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (memoryProperties.memoryTypes[i].propertyFlags & desired) == desired) {
      memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  ai.memoryTypeIndex = memoryTypeIndex;

  if (vkAllocateMemory(this->device, &ai, this->allocator, &memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(this->device, buffer, memory, 0);

  if (data) {
    void * mapped = nullptr;
    if (vkMapMemory(this->device, memory, 0, size, 0, &mapped) != VK_SUCCESS) {
      this->emitError("createBuffer: vkMapMemory failed");
      vkDestroyBuffer(this->device, buffer, this->allocator);
      vkFreeMemory(this->device, memory, this->allocator);
      buffer = VK_NULL_HANDLE;
      memory = VK_NULL_HANDLE;
      return false;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(this->device, memory);
  }
  return true;
}

void
SoVulkanRenderBackend::ensureDeviceMemoryProperties()
{
  if (!this->deviceMemoryPropertiesValid) {
    vkGetPhysicalDeviceMemoryProperties(this->physicalDevice,
                                        &this->deviceMemoryProperties);
    this->deviceMemoryPropertiesValid = true;
  }
}

bool
SoVulkanRenderBackend::createBufferDeviceLocal(VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VkBuffer & buffer,
                                               VkDeviceMemory & memory,
                                               const void * data)
{
  // Retained static geometry is read by the GPU every frame, so it belongs in
  // device-local VRAM rather than host-visible memory.  `data` is copied from
  // a transient host-visible staging buffer with a one-shot transfer that is
  // fenced before this function returns.  The GPU then reads the mesh from
  // device memory instead of walking the PCIe/system bus every frame.
  //
  // This is only invoked from the geometry-change path (not steady-state), so
  // the synchronous transfer is acceptable.  On any failure the buffer/memory
  // are left null and the caller falls back to the host-visible createBuffer().
  VkBufferCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = size;
  ci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(this->device, &ci, this->allocator, &buffer) != VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(this->device, buffer, &requirements);

  this->ensureDeviceMemoryProperties();
  uint32_t memoryTypeIndex = UINT32_MAX;
  for (uint32_t i = 0; i < this->deviceMemoryProperties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (this->deviceMemoryProperties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      memoryTypeIndex = i;
      break;
    }
  }
  if (memoryTypeIndex == UINT32_MAX) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }

  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = memoryTypeIndex;
  if (vkAllocateMemory(this->device, &ai, this->allocator, &memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(this->device, buffer, memory, 0);

  if (!data) return true;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMemory, data)) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    vkFreeMemory(this->device, memory, this->allocator);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    return false;
  }

  // One-shot transfer command buffer.  The per-frame buffers are not yet begun
  // at this point (updateGeometryCache runs before beginCommandBuffer), so we
  // allocate a transient buffer from the shared pool and fence-wait it.
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer transfer = VK_NULL_HANDLE;
  bool ok = vkAllocateCommandBuffers(this->device, &allocInfo, &transfer) ==
    VK_SUCCESS;

  if (ok) {
    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ok = vkBeginCommandBuffer(transfer, &bi) == VK_SUCCESS;
  }
  if (ok) {
    VkBufferCopy copy {};
    copy.size = size;
    vkCmdCopyBuffer(transfer, staging, buffer, 1, &copy);
    // Make the device-local writes visible to a later vertex-input read.  The
    // transfer is fenced so the copy has executed, but fence completion alone
    // does not establish a memory dependency for the buffer being read as
    // vertex/index attributes in a subsequent submit.  This barrier transitions
    // it from TRANSFER_WRITE to VERTEX_ATTRIBUTE/INDEX read.
    VkBufferMemoryBarrier bufBarrier {};
    bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                               VK_ACCESS_INDEX_READ_BIT;
    bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufBarrier.buffer = buffer;
    bufBarrier.offset = 0;
    bufBarrier.size = size;
    vkCmdPipelineBarrier(transfer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         0, 0, nullptr, 1, &bufBarrier, 0, nullptr);
    ok = vkEndCommandBuffer(transfer) == VK_SUCCESS;
  }

  VkFence fence = VK_NULL_HANDLE;
  if (ok) {
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ok = vkCreateFence(this->device, &fci, this->allocator, &fence) == VK_SUCCESS;
  }
  if (ok) {
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transfer;
    ok = vkQueueSubmit(this->queue, 1, &submit, fence) == VK_SUCCESS;
  }
  if (ok) {
    ok = vkWaitForFences(this->device, 1, &fence, VK_TRUE, UINT64_MAX) ==
      VK_SUCCESS;
  }
  if (fence != VK_NULL_HANDLE) {
    vkDestroyFence(this->device, fence, this->allocator);
  }
  if (transfer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &transfer);
  }
  vkDestroyBuffer(this->device, staging, this->allocator);
  vkFreeMemory(this->device, stagingMemory, this->allocator);

  if (!ok) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    vkFreeMemory(this->device, memory, this->allocator);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

void
SoVulkanRenderBackend::uploadGeometry(VulkanCachedCommand & entry,
                                      const SoRenderCommand & command)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;

  // Pack interleaved vertices with deterministic defaults for absent streams.
  // The buffer is a reusable member scratch vector: resize() preserves
  // capacity, so the heap is only touched on the first (largest) upload that
  // reaches this size rather than on every geometry change.  The lock spans the
  // packing plus the synchronous uploads below, which read `vertices`.
  std::lock_guard<std::mutex> scratchLock(this->uploadScratchMutex);
  this->uploadScratch.resize(static_cast<size_t>(vertexCount) * 12);
  float * const vertices = this->uploadScratch.data();
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t normalStrideFloats =
    (geometry.normals ? posStrideFloats : 0);
  const uint32_t texcoordStride = geometry.texcoordStride
    ? geometry.texcoordStride : sizeof(float) * 4;
  const uint32_t texcoordStrideFloats = texcoordStride / sizeof(float);

  for (uint32_t i = 0; i < vertexCount; ++i) {
    float * out = vertices + static_cast<size_t>(i) * 12;

    const float * pos = geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    out[0] = pos[0];
    out[1] = pos[1];
    out[2] = pos[2];

    if (geometry.normals && i < geometry.normalCount) {
      const float * normal = geometry.normals + static_cast<size_t>(i) * normalStrideFloats;
      out[3] = normal[0];
      out[4] = normal[1];
      out[5] = normal[2];
    }
    else {
      out[3] = 0.0f;
      out[4] = 0.0f;
      out[5] = 1.0f;
    }

    if (geometry.colors) {
      const float * color = geometry.colors + static_cast<size_t>(i) * 4;
      out[6] = color[0];
      out[7] = color[1];
      out[8] = color[2];
      out[9] = color[3];
    }
    else {
      out[6] = 1.0f;
      out[7] = 1.0f;
      out[8] = 1.0f;
      out[9] = 1.0f;
    }

    if (geometry.texcoords) {
      const float * uv = geometry.texcoords + static_cast<size_t>(i) * texcoordStrideFloats;
      out[10] = uv[0];
      out[11] = uv[1];
    }
    else {
      out[10] = 0.0f;
      out[11] = 0.0f;
    }
  }

  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(vertexCount) * VULKAN_VERTEX_STRIDE;
  // Cached static geometry (retained) lives in device-local VRAM so the GPU
  // does not read large meshes across the PCIe/system bus every frame.  The
  // upload is staged through a transient one-shot transfer; if device-local is
  // unavailable or the copy fails, fall back to the host-visible path so
  // rendering still works.  Non-retained geometry (per-frame overlays,
  // highlights, text, images -- which rewrite every frame) stays host-visible
  // so its frequent re-uploads never take the synchronous transfer stall.
  bool vertexCreated = false;
  if (geometry.retained) {
    vertexCreated = this->createBufferDeviceLocal(vertexBytes,
                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                  entry.vertexBuffer,
                                                  entry.vertexMemory, vertices);
  }
  if (!vertexCreated) {
    vertexCreated = this->createBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       entry.vertexBuffer, entry.vertexMemory,
                                       vertices);
  }
  if (!vertexCreated) {
    this->emitError("uploadGeometry: failed to create vertex buffer");
    return;
  }

  if (geometry.indexCount && geometry.indices) {
    const VkDeviceSize indexBytes =
      static_cast<VkDeviceSize>(geometry.indexCount) * sizeof(uint32_t);
    bool indexCreated = false;
    if (geometry.retained) {
      indexCreated = this->createBufferDeviceLocal(indexBytes,
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                    entry.indexBuffer,
                                                    entry.indexMemory,
                                                    geometry.indices);
    }
    if (!indexCreated) {
      indexCreated = this->createBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                        entry.indexBuffer, entry.indexMemory,
                                        geometry.indices);
    }
    if (!indexCreated) {
      this->emitError("uploadGeometry: failed to create index buffer");
      return;
    }
  }

  entry.posKey = geometry.positions;
  entry.normalKey = geometry.normals;
  entry.colorKey = geometry.colors;
  entry.texcoordKey = geometry.texcoords;
  entry.idxKey = geometry.indices;
  entry.vertexCount = vertexCount;
  entry.indexCount = geometry.indexCount;
  entry.vertexStride = posStride;
  entry.texcoordStride = geometry.texcoordStride;
  entry.normalCount = geometry.normalCount;
  entry.contentHash = hashGeometryContent(geometry);
}

void
SoVulkanRenderBackend::destroyCacheEntry(VulkanCachedCommand & entry)
{
  if (entry.indexBuffer) {
    vkDestroyBuffer(this->device, entry.indexBuffer, this->allocator);
    entry.indexBuffer = VK_NULL_HANDLE;
  }
  if (entry.indexMemory) {
    vkFreeMemory(this->device, entry.indexMemory, this->allocator);
    entry.indexMemory = VK_NULL_HANDLE;
  }
  if (entry.vertexBuffer) {
    vkDestroyBuffer(this->device, entry.vertexBuffer, this->allocator);
    entry.vertexBuffer = VK_NULL_HANDLE;
  }
  if (entry.vertexMemory) {
    vkFreeMemory(this->device, entry.vertexMemory, this->allocator);
    entry.vertexMemory = VK_NULL_HANDLE;
  }
  for (const VulkanCachedCommand::VulkanWideLineBuffer & slot : entry.wideLineBuffers) {
    if (slot.buffer) {
      vkDestroyBuffer(this->device, slot.buffer, this->allocator);
    }
    if (slot.memory) {
      vkFreeMemory(this->device, slot.memory, this->allocator);
    }
  }
  entry.wideLineBuffers.clear();
  entry = VulkanCachedCommand();
}

void
SoVulkanRenderBackend::invalidateCache()
{
  for (VulkanCachedCommand & entry : this->gpuCache) {
    this->destroyCacheEntry(entry);
  }
  this->gpuCache.clear();
  this->commandToCache.clear();
  this->invalidateTextureCache();
}

void
SoVulkanRenderBackend::updateGeometryCache(const SoDrawList & drawlist,
                                           const bool overlaysOnly)
{
  // The frame boundary was handled by beginFrame() at the entry point.

  // Release uploads left over from a frame that aborted between the cache
  // update and the flush/finalize step (e.g. a failed framebuffer create).
  // Their copies were never recorded, but deferring the staging destruction
  // is uniformly safe and keeps a single cleanup path.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging == VK_NULL_HANDLE) continue;
    const VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkBuffer staging = upload.staging;
    const VkDeviceMemory stagingMemory = upload.stagingMemory;
    this->deferDestroy([device, allocator, staging, stagingMemory]() {
      if (staging != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, staging, allocator);
      }
      if (stagingMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, stagingMemory, allocator);
      }
    });
  }
  this->pendingUploads.clear();

  const uint32_t generation = drawlist.getGeneration();

  // Make sure the descriptor pool can hold one set per distinct texture in
  // this frame before any allocation happens.  Pool growth never
  // invalidates existing sets, so this is safe regardless of recording
  // state.
  if (!this->ensureDescriptorPoolSpace()) {
    this->emitError("updateGeometryCache: failed to grow descriptor pool");
    // Continue with the current pool: allocateTextureDescriptorSet()
    // failures fall back to the white texture per command.
  }

  this->pendingUploads.clear();

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    // Overlay-only renders (ray-tracing compositing) draw SO_RENDERPASS_OVERLAY
    // commands (nav cube, axis cross, selection/hover highlights) plus the
    // non-triangle residue the RT backend did not trace (BRep edge lines,
    // point markers, polylines): those must be uploaded so the composite can
    // rasterize them onto the traced surface.  Pure triangle geometry is
    // already traced and skipping it here keeps the composite cheap.
    const bool isResidual =
      command.geometry.topology != SO_TOPOLOGY_TRIANGLES &&
      command.pass != SO_RENDERPASS_OVERLAY;
    if (overlaysOnly && command.pass != SO_RENDERPASS_OVERLAY &&
        !isResidual) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) {
      continue;
    }

    VulkanCachedCommand & entry = this->getOrCreateCache(&command);
    const uint32_t vertexStride = geometry.vertexStride
      ? geometry.vertexStride : sizeof(float) * 3;
    // The draw-list generation changes every frame (clear() bumps it), so
    // it is only a visit stamp for cache eviction below -- never a signal
    // to re-upload.  Re-uploads are driven purely by the producer-owned
    // content keys.
    //
    // Identity is the primary change signal.  For retained geometry (the
    // shape-owned retained buffers produced by SoShape::IRRender) the pointers
    // are stable across frames for an unchanged shape and change whenever the
    // shape re-tessellates, so a full identity match means the content is
    // unchanged and the content hash (recomputed on upload) is authoritative --
    // no need to re-hash every command every frame.  For non-retained geometry
    // (per-frame arena pools such as SoText2/SoImage, which rewrite the same
    // pointer in place) the content hash is still verified each frame to catch
    // in-place edits.
    const bool identityMatches = entry.vertexBuffer != VK_NULL_HANDLE &&
      entry.posKey == geometry.positions &&
      entry.normalKey == geometry.normals &&
      entry.colorKey == geometry.colors &&
      entry.texcoordKey == geometry.texcoords &&
      entry.idxKey == geometry.indices &&
      entry.vertexCount == geometry.vertexCount &&
      entry.indexCount == geometry.indexCount &&
      entry.normalCount == geometry.normalCount &&
      entry.vertexStride == vertexStride &&
      entry.texcoordStride == geometry.texcoordStride;
    const bool geometryMatches = identityMatches &&
      (geometry.retained ||
       entry.contentHash == hashGeometryContent(geometry));
    if (!geometryMatches) {
      this->deferDestroyCacheEntry(entry);
      this->uploadGeometry(entry, command);
    }
    entry.commandKey = &command;
    entry.cacheGeneration = generation;

    const SoTextureData & texture = command.material.texture;
    if (texture.pixels && texture.width > 0 && texture.height > 0) {
      VulkanCachedTexture & texEntry = this->getOrCreateTexture(&command);
      // Texture pixels come from per-frame action storage (arena), which may
      // rewrite the same pointer in place, so the content hash is always
      // re-verified -- pointer identity alone is not sound for textures.
      const bool textureMatches = texEntry.image != VK_NULL_HANDLE &&
        texEntry.pixelsKey == texture.pixels &&
        texEntry.width == texture.width &&
        texEntry.height == texture.height &&
        texEntry.numComponents == texture.numComponents &&
        texEntry.minFilter == texture.minFilter &&
        texEntry.magFilter == texture.magFilter &&
        texEntry.wrapS == texture.wrapS &&
        texEntry.wrapT == texture.wrapT &&
        texEntry.model == texture.model &&
        texEntry.contentHash == hashTextureContent(texture);
      if (!textureMatches) {
        this->deferDestroyTextureEntry(texEntry);
        // A command that appears twice in one draw list would otherwise
        // prepare two uploads for the same entry (leaking the first image);
        // the first pending upload for this index wins.
        bool alreadyPending = false;
        for (const PendingTextureUpload & prior : this->pendingUploads) {
          if (prior.index == this->commandToTexture[&command]) {
            alreadyPending = true;
            break;
          }
        }
        if (!alreadyPending) {
          PendingTextureUpload upload;
          upload.command = &command;
          upload.index = this->commandToTexture[&command];
          upload.texture = &texture;
          if (this->prepareTextureUpload(texEntry, texture, upload.staging,
                                         upload.stagingMemory)) {
            this->pendingUploads.push_back(upload);
          }
          // On failure the entry was reset by prepareTextureUpload();
          // leaving the content keys unstamped makes the next frame retry.
        }
      }
      texEntry.commandKey = &command;
      texEntry.cacheGeneration = generation;
    }
  }

  // Evict entries that were not visited this frame: their command has
  // disappeared from the draw list (or its pointer is no longer part of
  // this frame's arena).  Entries surviving eviction keep their index
  // identity, so rebuild the pointer maps from the stored commandKey.
  // Destruction is deferred: a pending frame may still reference the
  // evicted buffers/images.
  // Overlay-only renders skip the sweep: their traversal deliberately
  // visits only overlay commands, so a sweep here would evict the entire
  // scene cache and force a full re-upload on the next full render.
  if (!overlaysOnly) {
    const auto evictStale = [&](auto & cache, auto destroyEntry,
                                auto & indexMap) {
      bool anyStale = false;
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        if (cache[idx].cacheGeneration != generation) {
          destroyEntry(cache[idx]);
          anyStale = true;
        }
      }
      if (!anyStale) return;
      size_t write = 0;
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        if (cache[idx].cacheGeneration == generation) {
          if (write != idx) cache[write] = std::move(cache[idx]);
          ++write;
        }
      }
      cache.resize(write);
      indexMap.clear();
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        indexMap[cache[idx].commandKey] = idx;
      }
    };
    evictStale(this->gpuCache,
               [this](VulkanCachedCommand & entry) {
                 this->deferDestroyCacheEntry(entry);
               },
               this->commandToCache);
    evictStale(this->textureCache,
               [this](VulkanCachedTexture & entry) {
                 this->deferDestroyTextureEntry(entry);
               },
               this->commandToTexture);

    // Eviction compacts the texture cache and reindexes it, so the upload
    // indices captured above are stale.  Re-resolve each pending upload
    // through its command pointer; entries that were just prepared carry the
    // current generation and survive the sweep.
    for (PendingTextureUpload & upload : this->pendingUploads) {
      const auto it = this->commandToTexture.find(upload.command);
      if (it != this->commandToTexture.end()) {
        upload.index = it->second;
      }
      else {
        upload.index = std::numeric_limits<size_t>::max();
      }
    }
  }
}

// --- Texture cache --------------------------------------------------------

void
SoVulkanRenderBackend::destroyTextureEntry(VulkanCachedTexture & entry)
{
  if (entry.descriptorSet != VK_NULL_HANDLE) {
    if (entry.descriptorPool != VK_NULL_HANDLE) {
      vkFreeDescriptorSets(this->device, entry.descriptorPool, 1,
                           &entry.descriptorSet);
    }
    if (this->descriptorSetCount > 0) --this->descriptorSetCount;
    entry.descriptorSet = VK_NULL_HANDLE;
  }
  if (entry.sampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, entry.sampler, this->allocator);
    entry.sampler = VK_NULL_HANDLE;
  }
  if (entry.view != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, entry.view, this->allocator);
    entry.view = VK_NULL_HANDLE;
  }
  if (entry.image != VK_NULL_HANDLE) {
    vkDestroyImage(this->device, entry.image, this->allocator);
    entry.image = VK_NULL_HANDLE;
  }
  if (entry.memory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.memory, this->allocator);
    entry.memory = VK_NULL_HANDLE;
  }
  entry = VulkanCachedTexture();
}

void
SoVulkanRenderBackend::invalidateTextureCache()
{
  for (VulkanCachedTexture & entry : this->textureCache) {
    this->destroyTextureEntry(entry);
  }
  this->textureCache.clear();
  this->commandToTexture.clear();
}

VulkanCachedTexture &
SoVulkanRenderBackend::getOrCreateTexture(const SoRenderCommand * command)
{
  const auto found = this->commandToTexture.find(command);
  if (found != this->commandToTexture.end()) {
    return this->textureCache[found->second];
  }
  const size_t index = this->textureCache.size();
  this->textureCache.emplace_back();
  this->textureCache.back().commandKey = command;
  this->commandToTexture[command] = index;
  return this->textureCache.back();
}

bool
SoVulkanRenderBackend::createSampler(const SoTextureData & texture,
                                     VkSampler & sampler)
{
  VkSamplerCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  ci.magFilter = textureFilterToVk(texture.magFilter);
  ci.minFilter = textureFilterToVk(texture.minFilter);
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  ci.addressModeU = textureWrapToVk(texture.wrapS);
  ci.addressModeV = textureWrapToVk(texture.wrapT);
  ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.mipLodBias = 0.0f;
  ci.anisotropyEnable = VK_FALSE;
  ci.maxAnisotropy = 1.0f;
  ci.compareEnable = VK_FALSE;
  ci.minLod = 0.0f;
  ci.maxLod = 0.0f;
  ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  ci.unnormalizedCoordinates = VK_FALSE;
  return vkCreateSampler(this->device, &ci, this->allocator, &sampler) ==
         VK_SUCCESS;
}

bool
SoVulkanRenderBackend::prepareTextureUpload(VulkanCachedTexture & entry,
                                            const SoTextureData & texture,
                                            VkBuffer & staging,
                                            VkDeviceMemory & stagingMemory)
{
  // VK_FORMAT_R8G8B8_UNORM is not guaranteed to be sampleable, so expand
  // 3-component (RGB) textures to 4-component RGBA on the host.  Other
  // component counts map directly.
  const int components =
    (texture.numComponents == 3) ? 4 : texture.numComponents;
  if (components < 1 || components > 4) {
    this->emitError("prepareTextureUpload: unsupported component count");
    return false;
  }
  const VkFormat format = (texture.numComponents == 3)
    ? VK_FORMAT_R8G8B8A8_UNORM : textureFormatToVk(texture.numComponents);
  const VkDeviceSize byteSize =
    static_cast<VkDeviceSize>(texture.width) * texture.height * components;

  std::vector<unsigned char> converted;
  const unsigned char * uploadPixels = texture.pixels;
  if (texture.numComponents == 3) {
    const size_t pixelCount =
      static_cast<size_t>(texture.width) * texture.height;
    converted.resize(pixelCount * 4);
    for (size_t i = 0; i < pixelCount; ++i) {
      converted[i * 4 + 0] = texture.pixels[i * 3 + 0];
      converted[i * 4 + 1] = texture.pixels[i * 3 + 1];
      converted[i * 4 + 2] = texture.pixels[i * 3 + 2];
      converted[i * 4 + 3] = 255;
    }
    uploadPixels = converted.data();
  }

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {static_cast<uint32_t>(texture.width),
               static_cast<uint32_t>(texture.height), 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(this->device, &ci, this->allocator, &entry.image) !=
      VK_SUCCESS) {
    this->emitError("prepareTextureUpload: vkCreateImage failed");
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(this->device, entry.image, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = 0;
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &props);
  bool found = false;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      ai.memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    this->emitError("prepareTextureUpload: no device-local memory type");
    this->destroyTextureEntry(entry);
    return false;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator, &entry.memory) !=
      VK_SUCCESS) {
    this->emitError("prepareTextureUpload: vkAllocateMemory failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  vkBindImageMemory(this->device, entry.image, entry.memory, 0);

  if (!this->createBuffer(byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMemory, uploadPixels)) {
    this->emitError("prepareTextureUpload: staging buffer creation failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  return true;
}

void
SoVulkanRenderBackend::recordTextureUpload(
  VkCommandBuffer commandBuffer,
  const VulkanCachedTexture & entry,
  const SoTextureData & texture,
  VkBuffer staging)
{
  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = entry.image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {static_cast<uint32_t>(texture.width),
                        static_cast<uint32_t>(texture.height), 1};
  vkCmdCopyBufferToImage(commandBuffer, staging, entry.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);
}

bool
SoVulkanRenderBackend::finalizeTexture(VulkanCachedTexture & entry,
                                       const SoTextureData & texture)
{
  // The image format matches what prepareTextureUpload() created (RGB
  // textures are expanded to RGBA there).
  const VkFormat format = (texture.numComponents == 3)
    ? VK_FORMAT_R8G8B8A8_UNORM : textureFormatToVk(texture.numComponents);
  entry.view = createImageView(this->device, entry.image, format,
                               VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (entry.view == VK_NULL_HANDLE ||
      !this->createSampler(texture, entry.sampler) ||
      !this->allocateTextureDescriptorSet(entry.view, entry.sampler,
                                          entry.descriptorSet)) {
    this->emitError("finalizeTexture: view/sampler/descriptor creation failed");
    // Leave the entry half-initialized for the caller to dispose of.  On the
    // own-queue path the image is already referenced by recorded copies in
    // an unsubmitted command buffer, so the caller must defer the
    // destruction rather than destroy synchronously.
    return false;
  }
  entry.descriptorPool = this->descriptorPool;
  return true;
}

bool
SoVulkanRenderBackend::recordPendingTextureUploads()
{
  // Own-queue path: record the copies into the frame command buffer, ahead
  // of the render pass that samples them.  No separate submit is needed, so
  // no extra queue drain per frame.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    this->recordTextureUpload(this->currentCommandBuffer(),
                              this->textureCache[upload.index],
                              *upload.texture, upload.staging);
  }
  return true;
}

void
SoVulkanRenderBackend::finalizePendingTextureUploads()
{
  // Own-queue path: create the views/samplers/descriptor sets (bound by the
  // draws recorded below), stamp the content identity, and defer the staging
  // buffers to the frame's deferred-destruction batch.  Staging buffers are
  // referenced by the just-recorded submission, so they are released only
  // after the slot fence signals.
  VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    VulkanCachedTexture & texEntry = this->textureCache[upload.index];
    if (this->finalizeTexture(texEntry, *upload.texture)) {
      const SoTextureData & texture = *upload.texture;
      texEntry.pixelsKey = texture.pixels;
      texEntry.width = texture.width;
      texEntry.height = texture.height;
      texEntry.numComponents = texture.numComponents;
      texEntry.minFilter = texture.minFilter;
      texEntry.magFilter = texture.magFilter;
      texEntry.wrapS = texture.wrapS;
      texEntry.wrapT = texture.wrapT;
      texEntry.model = texture.model;
      texEntry.contentHash = hashTextureContent(texture);
    }
    else {
      // The entry's image is referenced by the recorded copies, so the
      // half-initialized resources must be destroyed through the deferred
      // ring, not synchronously.  Keys stay unstamped so the next frame
      // retries the upload.
      this->deferDestroyTextureEntry(texEntry);
    }
    if (upload.staging != VK_NULL_HANDLE) {
      const VkBuffer staging = upload.staging;
      const VkDeviceMemory stagingMemory = upload.stagingMemory;
      this->deferDestroy([device, allocator, staging, stagingMemory]() {
        if (staging != VK_NULL_HANDLE) {
          vkDestroyBuffer(device, staging, allocator);
        }
        if (stagingMemory != VK_NULL_HANDLE) {
          vkFreeMemory(device, stagingMemory, allocator);
        }
      });
    }
  }
  this->pendingUploads.clear();
}

bool
SoVulkanRenderBackend::flushPendingTextureUploadsExternal()
{
  if (this->pendingUploads.empty()) return true;

  // External path: the caller owns the frame command buffer and is already
  // inside a render pass, so the copies cannot be merged into it.  Record
  // them into one transient command buffer, submit, and wait.  The wait also
  // retires any in-flight frames submitted by the external caller, which
  // keeps ring-buffer reuse in renderExternal() safe even when the caller
  // pipelines more frames than maxFramesInFlight.
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer uploadBuffer = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(this->device, &allocInfo, &uploadBuffer) !=
      VK_SUCCESS) {
    this->emitError(
      "flushPendingTextureUploadsExternal: failed to allocate upload "
      "command buffer");
    goto fail;
  }

  {
    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(uploadBuffer, &bi) != VK_SUCCESS) {
      this->emitError(
        "flushPendingTextureUploadsExternal: failed to begin upload buffer");
      vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
      goto fail;
    }
  }

  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    this->recordTextureUpload(uploadBuffer, this->textureCache[upload.index],
                              *upload.texture, upload.staging);
  }

  if (vkEndCommandBuffer(uploadBuffer) != VK_SUCCESS) {
    this->emitError(
      "flushPendingTextureUploadsExternal: failed to end upload buffer");
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
    goto fail;
  }

  {
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &uploadBuffer;
    const VkResult submitResult =
      vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(this->queue);
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
    if (submitResult != VK_SUCCESS) {
      this->emitError("flushPendingTextureUploadsExternal: vkQueueSubmit "
                      "failed");
      goto fail;
    }
  }

  // Staging buffers are no longer referenced by the completed submission.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, upload.staging, this->allocator);
      vkFreeMemory(this->device, upload.stagingMemory, this->allocator);
    }
  }

  // Host-side completion (views/samplers/descriptor sets) and content
  // identity stamping.  The queue is idle here, so synchronous destruction
  // on failure is safe.  A failure leaves the content keys unstamped, so the
  // next frame retries the upload.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    VulkanCachedTexture & texEntry = this->textureCache[upload.index];
    if (this->finalizeTexture(texEntry, *upload.texture)) {
      const SoTextureData & texture = *upload.texture;
      texEntry.pixelsKey = texture.pixels;
      texEntry.width = texture.width;
      texEntry.height = texture.height;
      texEntry.numComponents = texture.numComponents;
      texEntry.minFilter = texture.minFilter;
      texEntry.magFilter = texture.magFilter;
      texEntry.wrapS = texture.wrapS;
      texEntry.wrapT = texture.wrapT;
      texEntry.model = texture.model;
      texEntry.contentHash = hashTextureContent(texture);
    }
    else {
      this->destroyTextureEntry(texEntry);
    }
  }
  this->pendingUploads.clear();
  return true;

fail:
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, upload.staging, this->allocator);
      vkFreeMemory(this->device, upload.stagingMemory, this->allocator);
    }
    if (upload.index < this->textureCache.size()) {
      // Reset the half-initialized entry so the next frame retries cleanly.
      this->destroyTextureEntry(this->textureCache[upload.index]);
    }
  }
  this->pendingUploads.clear();
  return false;
}

bool
SoVulkanRenderBackend::ensureDescriptorPoolSpace()
{
  // Each pool is sized for 1024 sets.  Textures accumulate per unique
  // command until the cache is invalidated (scene change, backend re-init),
  // so long-lived scenes with many distinct textures can exhaust the active
  // pool.  Resetting a pool wholesale would invalidate every set allocated
  // from it -- including sets referenced by frames the caller still has in
  // flight -- so instead a fresh pool is appended and becomes current.
  // Sets live in whatever pool allocated them and are freed back to that
  // pool (or destroyed with it at shutdown); never reset.
  if (this->descriptorSetCount < 1000) {
    return true;
  }
  this->descriptorSetCount = 0;
  return this->createDescriptorPool();
}

VkDescriptorSet
SoVulkanRenderBackend::resolveTextureSet(const SoRenderCommand & command)
{
  const auto found = this->commandToTexture.find(&command);
  if (found != this->commandToTexture.end() &&
      this->textureCache[found->second].descriptorSet != VK_NULL_HANDLE) {
    return this->textureCache[found->second].descriptorSet;
  }
  return this->whiteDescriptorSet;
}

// --- Render recording -----------------------------------------------------

void
SoVulkanRenderBackend::applyViewport(const SoRenderParams & params,
                                     const SoVulkanRenderTarget & target)
{
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();

  if (envFlagEnabled("FC_VULKAN_MATRIX_DUMP") && s_debugFrame > 0
      && (s_debugFrame % 100 == 0)) {
    fprintf(stderr,
            "[VPRT] frame=%d origin=(%d,%d) size=(%d,%d) target=(%u,%u)\n",
            s_debugFrame, origin[0], origin[1], size[0], size[1],
            target.extent.width, target.extent.height);
  }

  // Coin/OpenGL viewport origins are bottom-left; Vulkan's are top-left.
  // The vertex shader flips Y in clip space, so the viewport rectangle must
  // be re-anchored to the top edge for the two to cancel out (and for
  // non-fullscreen viewports to land in the correct sub-region).
  VkViewport viewport {};
  viewport.x = static_cast<float>(origin[0]);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(origin[1]) -
                                 static_cast<int32_t>(size[1]));
  viewport.width = static_cast<float>(size[0]);
  viewport.height = static_cast<float>(size[1]);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(this->activeCommandBuffer, 0, 1, &viewport);

  // Clamp the clear region to the target so an off-screen viewport (origin
  // outside the target, or a size exceeding the extent) never generates a
  // clear outside the render area.
  const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(origin[1]) -
         static_cast<int32_t>(size[1]));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(origin[0]) +
                                static_cast<int32_t>(size[0]));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(origin[1]));
  VkRect2D scissor {};
  scissor.offset = {x0, y0};
  scissor.extent = {static_cast<uint32_t>(std::max(0, x1 - x0)),
                    static_cast<uint32_t>(std::max(0, y1 - y0))};
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

// Apply a per-command viewport (recorded by the IR producer from
// SoViewportRegionElement).  Draws that carry their own viewport render
// into that sub-region; commands without one keep the frame viewport set
// by applyViewport().  Same Y-flip math as applyViewport().
void
SoVulkanRenderBackend::applyCommandViewport(const SoRenderCommand & command,
                                            const SoVulkanRenderTarget & target)
{
  const SoRasterState & raster = command.state.raster;
  if (!raster.viewportEnabled || raster.viewportWidth <= 0 ||
      raster.viewportHeight <= 0) {
    return;
  }
  VkViewport viewport {};
  viewport.x = static_cast<float>(raster.viewportX);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(raster.viewportY) -
                                 static_cast<int32_t>(raster.viewportHeight));
  viewport.width = static_cast<float>(raster.viewportWidth);
  viewport.height = static_cast<float>(raster.viewportHeight);
  // Depth range from the retained SoDepthBufferElement state; GL applies
  // glDepthRange() per command and restores (0,1) after each draw.  The
  // viewport is dynamic state here, so each command gets its own range and
  // nothing needs restoring.  Clamp to the legal [0,1] window.
  viewport.minDepth =
    std::clamp(command.state.depth.range[0], 0.0f, 1.0f);
  viewport.maxDepth =
    std::clamp(command.state.depth.range[1], 0.0f, 1.0f);
  vkCmdSetViewport(this->activeCommandBuffer, 0, 1, &viewport);

  // The per-command viewport also bounds the draw region; mirror the
  // scissor clamp used by applyViewport().
  const int32_t x0 = std::max(0, static_cast<int32_t>(raster.viewportX));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(raster.viewportY) -
         static_cast<int32_t>(raster.viewportHeight));
  const int32_t x1 =
    std::min(static_cast<int32_t>(target.extent.width),
             static_cast<int32_t>(raster.viewportX) +
               static_cast<int32_t>(raster.viewportWidth));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(raster.viewportY));
  VkRect2D scissor {};
  scissor.offset = {x0, y0};
  scissor.extent = {static_cast<uint32_t>(std::max(0, x1 - x0)),
                    static_cast<uint32_t>(std::max(0, y1 - y0))};
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

void
SoVulkanRenderBackend::applyScissor(const SoRenderCommand & command,
                                    const SoVulkanRenderTarget & target)
{
  VkRect2D scissor {};
  const SoRasterState & raster = command.state.raster;
  if (raster.scissorEnabled && raster.scissorWidth > 0 &&
      raster.scissorHeight > 0) {
    // Coin/OpenGL scissors are anchored at the bottom-left; Vulkan's are
    // top-left.  Mirror the viewport math: flip the Y offset around the
    // target height so the region lands where the producer intends.
    const int32_t flippedY = static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(raster.scissorY) -
      static_cast<int32_t>(raster.scissorHeight);
    scissor.offset = {static_cast<int32_t>(raster.scissorX), flippedY};
    scissor.extent = {static_cast<uint32_t>(raster.scissorWidth),
                      static_cast<uint32_t>(raster.scissorHeight)};
  }
  else {
    scissor.offset = {0, 0};
    scissor.extent = target.extent;
  }
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

void
SoVulkanRenderBackend::recordClear(const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;

  VkClearAttachment attachments[3];
  uint32_t attachmentCount = 0;

  if (params.flags & SO_PARAM_CLEAR_WINDOW) {
    const SbColor4f & color = params.clearColor;
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.color.float32[0] = color[0];
    clear.clearValue.color.float32[1] = color[1];
    clear.clearValue.color.float32[2] = color[2];
    clear.clearValue.color.float32[3] = color[3];
    attachments[attachmentCount++] = clear;
  }

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_DEPTH)) {
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.depthStencil.depth = params.clearDepth;
    clear.clearValue.depthStencil.stencil = 0;
    attachments[attachmentCount++] = clear;
  }

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_STENCIL)) {
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.depthStencil.depth = 0;
    clear.clearValue.depthStencil.stencil = params.clearStencil;
    attachments[attachmentCount++] = clear;
  }

  if (attachmentCount == 0) return;

  // Clear only the requested viewport region (Y-flipped into Vulkan
  // coordinates like applyViewport()).  Clearing the whole target would
  // overwrite other viewports or the backing image outside the viewport.
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(origin[1]) -
         static_cast<int32_t>(size[1]));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(origin[0]) +
                                static_cast<int32_t>(size[0]));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(origin[1]));
  if (x1 <= x0 || y1 <= y0) return;

  VkClearRect rect {};
  rect.rect.offset = {x0, y0};
  rect.rect.extent = {static_cast<uint32_t>(x1 - x0),
                      static_cast<uint32_t>(y1 - y0)};
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(this->activeCommandBuffer, attachmentCount, attachments, 1,
                        &rect);
}

void
SoVulkanRenderBackend::recordOverlayDepthClear(const SoRenderCommand & command,
                                               const SoVulkanRenderTarget & target)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  if (!hasDepth) {
    return;
  }

  // The overlay rect is stored in Coin/OpenGL (bottom-left) coordinates by
  // the producer; mirror the Y-flip applied by applyScissor().
  const SoRasterState & raster = command.state.raster;
  const int32_t x0 = std::max(0, static_cast<int32_t>(raster.scissorX));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(raster.scissorY) -
         static_cast<int32_t>(raster.scissorHeight));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(raster.scissorX) +
                                static_cast<int32_t>(raster.scissorWidth));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(raster.scissorY));
  if (x1 <= x0 || y1 <= y0) {
    return;
  }

  VkClearAttachment attachment {};
  attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  attachment.colorAttachment = 0;
  attachment.clearValue.depthStencil.depth = 1.0f;
  attachment.clearValue.depthStencil.stencil = 0;

  VkClearRect rect {};
  rect.rect.offset = {x0, y0};
  rect.rect.extent = {static_cast<uint32_t>(x1 - x0),
                      static_cast<uint32_t>(y1 - y0)};
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(this->activeCommandBuffer, 1, &attachment, 1, &rect);
}

void
SoVulkanRenderBackend::updateLightingUniforms(const SoDrawList & drawlist,
                                              const SoRenderCommand & command,
                                              const SoRenderParams & params,
                                              const VkDeviceSize uboOffset,
                                              const bool unlit)
{
  VulkanVisualUbo ubo {};

  SbMat m;
  if (command.state.raster.scissorEnabled
      && command.pass == SO_RENDERPASS_OVERLAY) {
    command.viewMatrix.getValue(m);
  }
  else {
    params.viewMatrix.getValue(m);
  }
  std::memcpy(ubo.view, &m[0][0], sizeof(float) * 16);
  command.modelMatrix.getValue(m);
  std::memcpy(ubo.model, &m[0][0], sizeof(float) * 16);
  if (envFlagEnabled("FC_VULKAN_CLIP_DEBUG")) {
    static int uboLog = 0;
    if (uboLog++ < 6) {
      fprintf(stderr, "[UBO] cmd pass=%d verts=%u model00=%.3f m11=%.3f m22=%.3f "
                      "trans=(%.3f,%.3f,%.3f) view33=%.3f\n",
              static_cast<int>(command.pass),
              static_cast<unsigned>(command.geometry.vertexCount),
              ubo.model[0], ubo.model[5], ubo.model[10],
              ubo.model[12], ubo.model[13], ubo.model[14],
              ubo.view[15]);
    }
  }

  const SoMaterialData & material = command.material;
  if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS)) {
    fprintf(stderr,
            "[UBO] cmd=%p pass=%d topo=%d diffuse=(%.2f,%.2f,%.2f,%.2f) "
            "emissive=(%.2f,%.2f,%.2f) ambient=(%.2f,%.2f,%.2f) "
            "specular=(%.2f,%.2f,%.2f) shading=%d unlit=%d\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            material.diffuse[0], material.diffuse[1], material.diffuse[2],
            material.diffuse[3],
            material.emissive[0], material.emissive[1], material.emissive[2],
            material.ambient[0], material.ambient[1], material.ambient[2],
            material.specular[0], material.specular[1], material.specular[2],
            static_cast<int>(material.shadingModel), unlit ? 1 : 0);
  }
  ubo.emissive[0] = material.emissive[0];
  ubo.emissive[1] = material.emissive[1];
  ubo.emissive[2] = material.emissive[2];
  ubo.emissive[3] = 1.0f;
  ubo.materialAmbient[0] = material.ambient[0];
  ubo.materialAmbient[1] = material.ambient[1];
  ubo.materialAmbient[2] = material.ambient[2];
  ubo.materialAmbient[3] = 1.0f;
  ubo.materialSpecular[0] = material.specular[0];
  ubo.materialSpecular[1] = material.specular[1];
  ubo.materialSpecular[2] = material.specular[2];
  ubo.materialSpecular[3] = 1.0f;
  ubo.materialParams[0] = material.shininess;
  ubo.materialParams[1] = material.twoSidedLighting ? 1.0f : 0.0f;
  ubo.materialParams[3] = unlit
    ? 0.0f
    : (material.shadingModel == SO_SHADING_LEGACY_GOURAUD ? 1.0f : 0.0f);

  const SoLightingData * lighting = drawlist.getLighting(command.lightingHandle);
  static const SoLightingData emptyLighting;
  if (!lighting) {
    lighting = &emptyLighting;
    if (command.lightingHandle != 0) {
      static std::once_flag invalidHandleWarning;
      std::call_once(invalidHandleWarning, []() {
        SoDebugError::postWarning(
          "SoVulkanRenderBackend::updateLightingUniforms",
          "Draw command references missing lighting data; no headlight is "
          "synthesized.");
      });
    }
  }

  ubo.ambientLight[0] = lighting->ambient[0];
  ubo.ambientLight[1] = lighting->ambient[1];
  ubo.ambientLight[2] = lighting->ambient[2];
  ubo.ambientLight[3] = 1.0f;

  const int count = std::min<int>(
    static_cast<int>(lighting->lights.size()), MAX_SHADER_LIGHTS);
  if (static_cast<int>(lighting->lights.size()) > MAX_SHADER_LIGHTS) {
    static std::once_flag lightLimitWarning;
    std::call_once(lightLimitWarning, []() {
      SoDebugError::postWarning(
        "SoVulkanRenderBackend::updateLightingUniforms",
        "The Visual program supports eight lights; additional retained "
        "lights are ignored by this executor.");
    });
  }
  ubo.materialParams[2] = static_cast<float>(count);

  for (int i = 0; i < count; ++i) {
    const SoLightData & light = lighting->lights[static_cast<size_t>(i)];
    float * type = ubo.lightType + i * 4;
    type[0] = static_cast<float>(light.type);
    type[1] = type[2] = 0.0f;
    type[3] = 1.0f;

    float * color = ubo.lightColor + i * 4;
    color[0] = light.color[0];
    color[1] = light.color[1];
    color[2] = light.color[2];
    color[3] = 1.0f;

    float * direction = ubo.lightDirection + i * 4;
    direction[0] = light.direction[0];
    direction[1] = light.direction[1];
    direction[2] = light.direction[2];
    direction[3] = 1.0f;

    float * position = ubo.lightPosition + i * 4;
    position[0] = light.position[0];
    position[1] = light.position[1];
    position[2] = light.position[2];
    position[3] = 1.0f;

    float * attenuation = ubo.lightAttenuation + i * 4;
    attenuation[0] = light.attenuation[0];
    attenuation[1] = light.attenuation[1];
    attenuation[2] = light.attenuation[2];
    attenuation[3] = 1.0f;

    float * spot = ubo.lightSpotParams + i * 4;
    spot[0] = light.spotCutoffCos;
    spot[1] = light.spotExponent;
    spot[2] = 0.0f;
    spot[3] = 1.0f;
  }

  if (envFlagEnabled("FC_VULKAN_LIGHT_DEBUG") && count > 0 &&
      s_lightLog++ < 6) {
    fprintf(stderr,
            "[LIT] cmd=%p pass=%d topo=%d verts=%u lightCount=%d "
            "ambient=(%.3f,%.3f,%.3f) matAmb=(%.3f,%.3f,%.3f) "
            "shininess=%.2f\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            static_cast<unsigned>(command.geometry.vertexCount), count,
            lighting->ambient[0], lighting->ambient[1], lighting->ambient[2],
            material.ambient[0], material.ambient[1], material.ambient[2],
            material.shininess);
    for (int li = 0; li < count && li < 3; ++li) {
      const SoLightData & l = lighting->lights[static_cast<size_t>(li)];
      fprintf(stderr,
              "[LIT]   light%d type=%d color=(%.2f,%.2f,%.2f) "
              "dir=(%.3f,%.3f,%.3f) pos=(%.2f,%.2f,%.2f)\n",
              li, static_cast<int>(l.type), l.color[0], l.color[1],
              l.color[2], l.direction[0], l.direction[1], l.direction[2],
              l.position[0], l.position[1], l.position[2]);
    }
  }

  if (this->lightingMapped && this->uboSlotStride > 0) {
    std::memcpy(static_cast<char *>(this->lightingMapped) + uboOffset,
                &ubo, sizeof(ubo));
  }

  if (envFlagEnabled("FC_VULKAN_MATRIX_DUMP") && s_debugFrame > 0
      && (s_debugFrame % 100 == 0) && s_dumpCmdCount <= 4) {
    fprintf(stderr,
            "[LGT] frame=%d cmd#%d ambient=(%.2f,%.2f,%.2f) lights=%d\n",
            s_debugFrame, s_dumpCmdCount - 1, lighting->ambient[0],
            lighting->ambient[1], lighting->ambient[2], count);
    for (int i = 0; i < count && i < 3; ++i) {
      const SoLightData & light = lighting->lights[static_cast<size_t>(i)];
      fprintf(stderr,
              "[LGT]   light%d type=%.0f dir=(%.3f,%.3f,%.3f) "
              "pos=(%.3f,%.3f,%.3f) color=(%.2f,%.2f,%.2f)\n",
              i, static_cast<float>(light.type),
              light.direction[0], light.direction[1], light.direction[2],
              light.position[0], light.position[1], light.position[2],
              light.color[0], light.color[1], light.color[2]);
    }
  }
}

bool
SoVulkanRenderBackend::expandWideLines(VulkanCachedCommand & entry,
                                       const SoRenderCommand & command,
                                       const SoRenderParams & params,
                                       const SbMat & proj,
                                       const float lineWidth)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;
  if (!vertexCount) return false;

  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t count = geometry.indexCount && geometry.indices
    ? geometry.indexCount : vertexCount;
  const bool strip = geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  const uint32_t segmentCount =
    strip ? (count > 1 ? count - 1 : 0) : count / 2;
  if (!segmentCount) return false;

  static int wlineDiag = 0;
  const bool isSketchCmd = vertexCount >= 900;
  const bool wdiag = envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")
    && (isSketchCmd || wlineDiag < 40) && wlineDiag < 200;
  if (wdiag) {
    ++wlineDiag;
    fprintf(stderr, "[WLINE2] enter frame=%u cmd=%p verts=%u idx=%u strip=%d segs=%u lw=%.2f\n",
            this->uboFrameIndex, (const void*)&command, vertexCount, count, strip ? 1 : 0,
            static_cast<unsigned>(segmentCount),
            static_cast<double>(lineWidth));
  }

  // MVP: clip = P * V * M * pos.  Regular geometry shares the frame view
  // matrix in params (wide lines never render in the overlay pass).
  SbMat model;
  command.modelMatrix.getValue(model);
  auto multiplyMat = [](const SbMat & a, const SbMat & b, SbMat & out) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] +
          a[r][2] * b[2][c] + a[r][3] * b[3][c];
      }
    }
  };
  SbMat view;
  params.viewMatrix.getValue(view);
  // The GPU visual path computes clip = proj_GL * view_GL * model_GL * pos,
  // where a *_GL matrix is the transpose of the stored (row-major) SbMat,
  // because Vulkan/GLSL interprets the raw memory as column-major.  The CPU
  // quad expansion below must produce the exact same clip positions (the
  // wide-line vertex shader passes them through).  With the stored accessor
  // matrices this is equivalent to mvp = transpose(model * view * proj):
  // (W^T * v)_i = sum_{l,k,j} model[l][k] view[k][j] proj[j][i] v_l
  //   = sum_j proj_GL[i][j] * (view_GL * model_GL * v)_j.
  // Multiplying in the naive order (proj*view*model) would place the view
  // translation into the clip w component (-P.x*... *v + 1), producing huge
  // negative w for off-origin geometry, a failing near-plane test, and
  // geometry collapsed onto NDC (0,0).
  SbMat vp;
  multiplyMat(view, proj, vp);
  SbMat wm;
  multiplyMat(model, vp, wm);
  SbMat mvp;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      mvp[r][c] = wm[c][r];
    }
  }
  const SbVec2s viewportSize = params.viewport.getViewportSizePixels();
  const float vpWidth = static_cast<float>(viewportSize[0] > 0
    ? viewportSize[0] : 1);
  const float vpHeight = static_cast<float>(viewportSize[1] > 0
    ? viewportSize[1] : 1);

  // Transform a position to clip space with the same Y-flip and depth remap
  // as the visual vertex shader, so the wide-line vertex shader can pass the
  // quad corners through unchanged.
  auto transformPoint = [&mvp](const float * p, float out[4]) {
    const float x = p[0];
    const float y = p[1];
    const float z = p[2];
    const float cx = mvp[0][0] * x + mvp[0][1] * y + mvp[0][2] * z +
      mvp[0][3];
    const float cy = mvp[1][0] * x + mvp[1][1] * y + mvp[1][2] * z +
      mvp[1][3];
    const float cz = mvp[2][0] * x + mvp[2][1] * y + mvp[2][2] * z +
      mvp[2][3];
    const float cw = mvp[3][0] * x + mvp[3][1] * y + mvp[3][2] * z +
      mvp[3][3];
    out[0] = cx;
    out[1] = -cy;
    out[2] = 0.5f * cz + 0.5f * cw;
    out[3] = cw;
  };

  // Per-vertex clip-space cache plus accumulated polyline distance in
  // WINDOW PIXELS.  Classic GL applies the stipple pattern in screen space
  // (glLineStipple: each bit covers linePatternScaleFactor pixels), so the
  // fragment discard below must operate on pixel distances, not object
  // units -- an object-unit period changes size when zooming.
  std::vector<float> clipCache(static_cast<size_t>(vertexCount) * 4, 0.0f);
  std::vector<float> distances(vertexCount, 0.0f);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t actual = geometry.indices ? geometry.indices[i] : i;
    float * clip = clipCache.data() + static_cast<size_t>(actual) * 4;
    transformPoint(geometry.positions
                   + static_cast<size_t>(actual) * posStrideFloats, clip);
  }
  if (strip) {
    for (uint32_t i = 1; i < count; ++i) {
      const uint32_t previous = geometry.indices
        ? geometry.indices[i - 1] : i - 1;
      const uint32_t current = geometry.indices ? geometry.indices[i] : i;
      const float * c0 = clipCache.data() + static_cast<size_t>(previous) * 4;
      const float * c1 = clipCache.data() + static_cast<size_t>(current) * 4;
      if (c0[3] <= 0.0f || c1[3] <= 0.0f) {
        distances[current] = distances[previous];
        continue;
      }
      const float dx = (c1[0] / c1[3] - c0[0] / c0[3]) * 0.5f * vpWidth;
      const float dy = (c1[1] / c1[3] - c0[1] / c0[3]) * 0.5f * vpHeight;
      distances[current] = distances[previous] +
        std::sqrt(dx * dx + dy * dy);
    }
  }
  else {
    for (uint32_t i = 0; i + 1 < count; i += 2) {
      const uint32_t first = geometry.indices ? geometry.indices[i] : i;
      const uint32_t second = geometry.indices
        ? geometry.indices[i + 1] : i + 1;
      const float * c0 = clipCache.data() + static_cast<size_t>(first) * 4;
      const float * c1 = clipCache.data() + static_cast<size_t>(second) * 4;
      if (c0[3] <= 0.0f || c1[3] <= 0.0f) {
        distances[first] = 0.0f;
        distances[second] = 0.0f;
        continue;
      }
      const float dx = (c1[0] / c1[3] - c0[0] / c0[3]) * 0.5f * vpWidth;
      const float dy = (c1[1] / c1[3] - c0[1] / c0[3]) * 0.5f * vpHeight;
      distances[first] = 0.0f;
      distances[second] = std::sqrt(dx * dx + dy * dy);
    }
  }

  // 6 vertices per segment (two triangles), 9 floats each:
  // clip position (4) + color (4) + distance in pixels (1).
  std::vector<float> quads(static_cast<size_t>(segmentCount) * 6 * 9);
  size_t outIndex = 0;
  size_t diagSkippedW = 0;
  size_t diagSkippedDeg = 0;
  const float nearEps = 1.0e-5f;
  for (uint32_t s = 0; s < segmentCount; ++s) {
    uint32_t i0;
    uint32_t i1;
    if (strip) {
      i0 = geometry.indices ? geometry.indices[s] : s;
      i1 = geometry.indices ? geometry.indices[s + 1] : s + 1;
    }
    else {
      i0 = geometry.indices ? geometry.indices[s * 2] : s * 2;
      i1 = geometry.indices ? geometry.indices[s * 2 + 1] : s * 2 + 1;
    }
    const float * col0 = geometry.colors
      ? geometry.colors + static_cast<size_t>(i0) * 4 : nullptr;
    const float * col1 = geometry.colors
      ? geometry.colors + static_cast<size_t>(i1) * 4 : nullptr;

    const float * c0 = clipCache.data() + static_cast<size_t>(i0) * 4;
    const float * c1 = clipCache.data() + static_cast<size_t>(i1) * 4;

    // Near-plane clip in the remapped clip space.  A point is visible when
    // it is in front of the eye (w > nearEps) AND in front of the near plane
    // (the remapped z, cache[2] >= 0, i.e. z_ndc >= 0).  The old code
    // dropped the WHOLE segment when either endpoint had w <= 0, so a
    // segment merely crossing the near plane vanished at close-in views.
    // Instead, clip against the boundary: when only one endpoint is visible
    // the segment straddles the near plane, so keep the visible half and
    // interpolate the hidden endpoint onto the plane.  Behind-the-eye points
    // also land behind the near plane (cache[2] < 0), so this single clip
    // surface handles both the near-plane and eye-plane crossings.
    const float fa = c0[2];
    const float fb = c1[2];
    const bool visible0 = (c0[3] > nearEps) && (fa >= 0.0f);
    const bool visible1 = (c1[3] > nearEps) && (fb >= 0.0f);
    if (!visible0 && !visible1) {
      if (wdiag) ++diagSkippedW;
      continue;
    }

    // tA/tB give the interpolation fraction along c0 -> c1 for each emitted
    // end (0 = original c0, 1 = original c1, intermediate = clipped onto the
    // near plane).  Exactly one end is ever clipped because a plane meets a
    // line segment at most once.
    float tA;
    float tB;
    if (visible0 && visible1) {
      tA = 0.0f;
      tB = 1.0f;
    }
    else {
      const float denom = fa - fb;
      const float tclip = (denom != 0.0f) ? fa / denom : 0.0f;
      tA = visible0 ? 0.0f : tclip;
      tB = visible1 ? 1.0f : tclip;
    }

    const float cA[4] = {
      c0[0] + tA * (c1[0] - c0[0]),
      c0[1] + tA * (c1[1] - c0[1]),
      c0[2] + tA * (c1[2] - c0[2]),
      c0[3] + tA * (c1[3] - c0[3]),
    };
    const float cB[4] = {
      c0[0] + tB * (c1[0] - c0[0]),
      c0[1] + tB * (c1[1] - c0[1]),
      c0[2] + tB * (c1[2] - c0[2]),
      c0[3] + tB * (c1[3] - c0[3]),
    };
    // Guard against a degenerate clip that lands on/behind the eye plane.
    if (cA[3] <= nearEps || cB[3] <= nearEps) {
      if (wdiag) ++diagSkippedW;
      continue;
    }

    const float dA = distances[i0] + tA * (distances[i1] - distances[i0]);
    const float dB = distances[i0] + tB * (distances[i1] - distances[i0]);

    const float ndc0x = cA[0] / cA[3];
    const float ndc0y = cA[1] / cA[3];
    const float ndc1x = cB[0] / cB[3];
    const float ndc1y = cB[1] / cB[3];
    const float dx = ndc1x - ndc0x;
    const float dy = ndc1y - ndc0y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.0e-8f) {
      if (wdiag) ++diagSkippedDeg;
      continue;
    }
    const float dirx = dx / length;
    const float diry = dy / length;
    // Anisotropic NDC offset mirroring the GL geometry shader
    // (perp * lineWidth / u_vpSize with per-axis division).
    const float offx = -diry * lineWidth / vpWidth;
    const float offy = dirx * lineWidth / vpHeight;

    // Quad corners: [0]=p0+off, [1]=p0-off, [2]=p1+off, [3]=p1-off.
    // Triangles: (0,1,2), (2,1,3) -- same emission order as the GL
    // geometry shader's triangle strip.
    float corners[4][4];
    for (int corner = 0; corner < 4; ++corner) {
      const int endpoint = corner < 2 ? 0 : 1;
      const float sign = (corner % 2 == 0) ? 1.0f : -1.0f;
      const float w = endpoint == 0 ? cA[3] : cB[3];
      corners[corner][0] = (endpoint == 0 ? cA[0] : cB[0]) + sign * offx * w;
      corners[corner][1] = (endpoint == 0 ? cA[1] : cB[1]) + sign * offy * w;
      corners[corner][2] = endpoint == 0 ? cA[2] : cB[2];
      corners[corner][3] = w;
    }
    // Colors, interpolated for a clipped (near-plane) endpoint.  The
    // default is opaque white when the command has no per-vertex colors.
    const float defaultColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    const float * p0 = col0 ? col0 : defaultColor;
    const float * p1 = col1 ? col1 : defaultColor;
    float colA[4];
    float colB[4];
    for (int i = 0; i < 4; ++i) {
      colA[i] = p0[i] + tA * (p1[i] - p0[i]);
      colB[i] = p0[i] + tB * (p1[i] - p0[i]);
    }

    static const int triOrder[6] = { 0, 1, 2, 2, 1, 3 };
    for (int t = 0; t < 6; ++t) {
      const int corner = triOrder[t];
      const int endpoint = corner < 2 ? 0 : 1;
      const float * col = endpoint == 0 ? colA : colB;
      float * out = quads.data() + outIndex;
      outIndex += 9;
      out[0] = corners[corner][0];
      out[1] = corners[corner][1];
      out[2] = corners[corner][2];
      out[3] = corners[corner][3];
      out[4] = col[0];
      out[5] = col[1];
      out[6] = col[2];
      out[7] = col[3];
      out[8] = endpoint == 0 ? dA : dB;
    }
  }
  quads.resize(outIndex);
  if (outIndex < 9) {
    if (wdiag) {
      fprintf(stderr, "[WLINE2] FAIL cmd=%p verts=%u segs=%u outIndex=%zu "
                      "skippedW=%zu skippedDeg=%zu\n",
              (const void*)&command, vertexCount,
              static_cast<unsigned>(segmentCount), outIndex,
              diagSkippedW, diagSkippedDeg);
    }
    return false;
  }
  if (wdiag) {
    const SbMatrix vwd = params.viewMatrix;
    const SbMatrix pwd(proj);
    fprintf(stderr, "[WLINE2] OK cmd=%p verts=%u segs=%u quads=%zu "
                    "skippedW=%zu skippedDeg=%zu firstSegNDC=(%.3f,%.3f)->(%.3f,%.3f)\n",
            (const void*)&command, vertexCount, static_cast<unsigned>(segmentCount),
            outIndex / 9, diagSkippedW, diagSkippedDeg,
            static_cast<double>(clipCache[0] / clipCache[3]),
            static_cast<double>(clipCache[1] / clipCache[3]),
            static_cast<double>(clipCache[4] / clipCache[7]),
            static_cast<double>(clipCache[5] / clipCache[7]));
    fprintf(stderr, "[WLINE2]   viewT=(%.3f,%.3f,%.3f) v11=%.4f v00=%.4f "
                    "isId=%d proj00=%.4f proj11=%.4f proj33=%.4f "
                    "cmdViewT=(%.3f,%.3f,%.3f)\n",
            static_cast<double>(vwd[3][0]), static_cast<double>(vwd[3][1]),
            static_cast<double>(vwd[3][2]),
            static_cast<double>(vwd[1][1]), static_cast<double>(vwd[0][0]),
            (vwd[0][0] == 1.0f && vwd[3][0] == 0.0f && vwd[3][1] == 0.0f) ? 1 : 0,
            static_cast<double>(pwd[0][0]), static_cast<double>(pwd[1][1]),
            static_cast<double>(pwd[3][3]),
            static_cast<double>(command.viewMatrix[3][0]),
            static_cast<double>(command.viewMatrix[3][1]),
            static_cast<double>(command.viewMatrix[3][2]));
  }

  if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
    static int distLog = 0;
    if (distLog++ < 3) {
      fprintf(stderr, "[WLINE] verts=%u segs=%u quads=%zu dists:",
              vertexCount, segmentCount, quads.size() / 9);
      for (size_t q = 0; q < quads.size() && q < 60; q += 9) {
        fprintf(stderr, " %.1f", static_cast<double>(quads[q + 8]));
      }
      fprintf(stderr, "\n");
    }
  }

  const VkDeviceSize needed =
    static_cast<VkDeviceSize>(outIndex) * sizeof(float);
  // Ring of host-visible scratch buffers, one per in-flight frame slot: the
  // quad expansion is recomputed every frame (it depends on the projection
  // matrix), so the slot selected for the current frame index is reused only
  // after the same slot's previous submission has completed (beginFrame
  // waits the slot fence).  Growth defers the old buffer's destruction
  // instead of destroying it synchronously, since a still-executing frame
  // may reference it.
  if (entry.wideLineBuffers.size() < this->maxFramesInFlight) {
    entry.wideLineBuffers.resize(this->maxFramesInFlight);
  }
  VulkanCachedCommand::VulkanWideLineBuffer & slot =
    entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
  if (slot.size < needed) {
    if (slot.buffer != VK_NULL_HANDLE || slot.memory != VK_NULL_HANDLE) {
      const VkDevice device = this->device;
      const VkAllocationCallbacks * allocator = this->allocator;
      const VkBuffer oldBuffer = slot.buffer;
      const VkDeviceMemory oldMemory = slot.memory;
      slot.buffer = VK_NULL_HANDLE;
      slot.memory = VK_NULL_HANDLE;
      slot.size = 0;
      this->deferDestroy([device, allocator, oldBuffer, oldMemory]() {
        if (oldBuffer != VK_NULL_HANDLE) {
          vkDestroyBuffer(device, oldBuffer, allocator);
        }
        if (oldMemory != VK_NULL_HANDLE) {
          vkFreeMemory(device, oldMemory, allocator);
        }
      });
    }
    if (!this->createBuffer(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            slot.buffer, slot.memory, quads.data())) {
      this->emitError("expandWideLines: failed to create quad buffer");
      slot.size = 0;
      return false;
    }
    slot.size = needed;
  }
  else {
    void * mapped = nullptr;
    if (vkMapMemory(this->device, slot.memory, 0, needed, 0,
                    &mapped) != VK_SUCCESS) {
      this->emitError("expandWideLines: vkMapMemory failed");
      return false;
    }
    std::memcpy(mapped, quads.data(), static_cast<size_t>(needed));
    vkUnmapMemory(this->device, slot.memory);
  }

  entry.wideLineVertexCount = static_cast<uint32_t>(outIndex / 9);
  return true;
}

void
SoVulkanRenderBackend::recordDrawCommand(const SoDrawList & drawlist,
                                         const SoRenderCommand & command,
                                         const SoVulkanRenderTarget & target,
                                         const SoRenderParams & params,
                                         VkRenderPass pass,
                                         const bool transparent,
                                         const int fillModeOverride,
                                         const float * uniformColorOverride,
                                         const bool overlayPass)
{
  if (!command.geometry.positions || command.geometry.vertexCount == 0) {
    if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: no positions/verts\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) {
    if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: no gpu cache entry\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }
  if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS)) {
    const VulkanCachedCommand & entryTmp = this->gpuCache[found->second];
    fprintf(stderr,
            "[VKBE] line/point cmd=%p pass=%d topo=%d verts=%u "
            "diffuse=(%.2f,%.2f,%.2f,%.2f) colorKey=%d shading=%d "
            "lineWidth=%.2f pattern=0x%04x fillMode=%d\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            command.geometry.vertexCount,
            command.material.diffuse[0], command.material.diffuse[1],
            command.material.diffuse[2], command.material.diffuse[3],
            entryTmp.colorKey != nullptr ? 1 : 0,
            static_cast<int>(command.material.shadingModel),
            command.state.raster.lineWidth,
            static_cast<unsigned>(command.state.raster.linePattern),
            static_cast<int>(command.state.raster.fillMode));
  }
  VulkanCachedCommand & entry = this->gpuCache[found->second];
  if (entry.vertexBuffer == VK_NULL_HANDLE) {
    if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: vertexBuffer null\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }

  // Wide-line rendering mirrors the GL wide-line path: line width > 1 or a
  // stipple pattern expands each segment into a quad.  The overlay
  // wireframe/point redraws keep the plain line path.
  const bool lineTopology = command.geometry.topology == SO_TOPOLOGY_LINES ||
    command.geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  const bool patternedLine =
    command.state.raster.linePattern != 0xFFFF &&
    command.state.raster.linePattern != 0;
  const bool useWideLine =
    lineTopology && fillModeOverride < 0 &&
    (command.state.raster.lineWidth > 1.0f || patternedLine);
  // Line stipple mirrors classic GL (glLineStipple): each pattern bit
  // covers linePatternScaleFactor PIXELS in screen space.  The fragment
  // shader tests the bit selected by floor(distance / factor) % 16.
  float stippleFactor = 0.0f;
  float stipplePatternBits = 0.0f;
  if (useWideLine && patternedLine) {
    stippleFactor = static_cast<float>(std::max(
      1, static_cast<int>(command.state.raster.linePatternScale)));
    const uint32_t patternBits =
      static_cast<uint32_t>(command.state.raster.linePattern & 0xFFFFu);
    std::memcpy(&stipplePatternBits, &patternBits, sizeof(patternBits));
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->getOrCreatePipeline(command, target, pass, pipeline, transparent,
                                 fillModeOverride, overlayPass) ||
      pipeline == VK_NULL_HANDLE) {
    if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: pipeline creation failed "
                      "(transparent=%d fillOverride=%d overlay=%d)\n",
              (const void*)&command, static_cast<int>(command.pass),
              transparent ? 1 : 0, fillModeOverride, overlayPass ? 1 : 0);
    }
    return;
  }
  if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
    static int drawn = 0;
    static int logged = 0;
    drawn++;
    if (logged++ < 200) {
      fprintf(stderr,
              "[VKBE] draw %d cmd=%p pass=%d verts=%u idx=%u topo=%d "
              "overlay=%d transparent=%d\n",
              drawn, (const void*)&command, static_cast<int>(command.pass),
              command.geometry.vertexCount, command.geometry.indexCount,
              static_cast<int>(command.geometry.topology),
              overlayPass ? 1 : 0, transparent ? 1 : 0);
    }
  }
  vkCmdBindPipeline(this->activeCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline);
  // Commands carrying their own viewport (SoViewportRegionElement) render
  // into that sub-region; otherwise the frame viewport from applyViewport()
  // stays active.
  this->applyCommandViewport(command, target);
  this->applyScissor(command, target);

  const uint32_t slotIndex = this->uboCmdIndex++;
  // prepareLightingSlots() reserves a worst-case slot count before any
  // recording, so this can only trip if a future recording path forgets to
  // pre-count.  Guard at runtime regardless: the mapped UBO write below
  // would otherwise run past the allocation.
  if (slotIndex >= this->uboSlotsPerFrame) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      this->emitError(
        "lighting UBO slot overflow: buffer too small for draw list");
    }
    return;
  }
  const VkDeviceSize uboOffset =
    ((this->uboFrameIndex % this->maxFramesInFlight) *
       this->uboSlotsPerFrame + slotIndex) *
    this->uboSlotStride;
  const uint32_t uboDynamicOffset = static_cast<uint32_t>(uboOffset);

  VkDescriptorSet textureSet = this->resolveTextureSet(command);
  if (textureSet != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(this->activeCommandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            this->pipelineLayout, 0, 1, &textureSet, 1,
                            &uboDynamicOffset);
  }

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(this->activeCommandBuffer, 0, 1, &entry.vertexBuffer,
                         &offset);
  const bool indexed =
    entry.indexBuffer != VK_NULL_HANDLE && command.geometry.indexCount &&
    command.geometry.indices;
  if (indexed && !useWideLine) {
    vkCmdBindIndexBuffer(this->activeCommandBuffer, entry.indexBuffer, 0,
                         VK_INDEX_TYPE_UINT32);
  }

  this->updateLightingUniforms(drawlist, command, params, uboOffset,
                               uniformColorOverride != nullptr);

  VulkanPushConstants push {};
  SbMat projValue;
  // Overlay-pass geometry carries its own camera (view/projection) matrices;
  // regular geometry shares the frame camera in params.
  if (overlayPass) {
    command.projMatrix.getValue(projValue);
  }
  else {
    params.projMatrix.getValue(projValue);
  }
  std::memcpy(push.proj, &projValue[0][0], sizeof(float) * 16);
  const SbVec4f & color = command.material.diffuse;
  const bool useOverrideColor = uniformColorOverride != nullptr;
  push.color[0] = useOverrideColor ? uniformColorOverride[0] : color[0];
  push.color[1] = useOverrideColor ? uniformColorOverride[1] : color[1];
  push.color[2] = useOverrideColor ? uniformColorOverride[2] : color[2];
  push.color[3] = useOverrideColor ? uniformColorOverride[3] : color[3];
  push.flags[0] = (entry.colorKey && !useOverrideColor) ? 1.0f : 0.0f;
  push.flags[1] =
    command.material.vertexColorAlphaIncludesOpacity ? 1.0f : 0.0f;
  const bool textured = command.material.texture.pixels &&
                        command.material.texture.width > 0 &&
                        command.material.texture.height > 0;
  push.flags[2] = (textured && !useOverrideColor) ? 1.0f : 0.0f;
  push.flags[3] = command.material.textureAlphaIncludesOpacity
                    ? 1.0f : 0.0f;
  push.texParams[0] =
    static_cast<float>(command.material.texture.model);
  push.texParams[1] =
    static_cast<float>(command.state.alphaTest.function);
  push.texParams[2] = command.state.alphaTest.reference;
  push.texParams[3] =
    (command.material.flags & SO_MAT_IS_PIXEL_TEXT) ? 1.0f : 0.0f;
  const SbVec4f & blendColor = command.material.texture.blendColor;
  push.texBlend[0] = blendColor[0];
  push.texBlend[1] = blendColor[1];
  push.texBlend[2] = blendColor[2];
  push.texBlend[3] = blendColor[3];
  // Point size from the retained state (SoDrawStyle::pointSize via
  // SoPointSizeElement); GL multiplies by the device pixel ratio because
  // its viewport is in device pixels -- Vulkan viewports are too, so the
  // same value applies directly.  Applies to point primitives and to
  // VK_POLYGON_MODE_POINT (the wireframe/points overlay).
  push.pointSize = std::max(1.0f, command.state.raster.pointSize);
  push.lineParams[0] = push.lineParams[1] = push.lineParams[2] = 0.0f;
  push.lineParams[3] = 0.0f;
  push.lineParams[0] = stippleFactor;
  // Slot y serves two masters: the wide-line shader reads the stipple
  // pattern bits here, the visual shader reads the round-point flag.  A
  // draw never reaches both shaders, so the slot is safe to share.
  push.lineParams[1] = (useWideLine && patternedLine)
    ? stipplePatternBits
    : (command.state.raster.pointShape == SO_POINT_SHAPE_ROUND ? 1.0f : 0.0f);
  // Debug override: no scene-side element sets SO_POINT_SHAPE_ROUND yet;
  // this exercises the round-glyph discard path for testing.
  if (envFlagEnabled("FC_VULKAN_ROUND_POINTS") &&
      command.geometry.topology == SO_TOPOLOGY_POINTS) {
    push.lineParams[1] = 1.0f;
  }
  push.lineParams[2] = useWideLine ? 1.0f : 0.0f;
  push.lineParams[3] = command.geometry.topology == SO_TOPOLOGY_POINTS
    ? 1.0f : 0.0f;

  vkCmdPushConstants(this->activeCommandBuffer, this->pipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);

  if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS ||
       s_debugPushCount++ < 40)) {
    uint32_t patternRaw = 0;
    std::memcpy(&patternRaw, &push.lineParams[1], sizeof(patternRaw));
    fprintf(stderr,
            "[PUSH] cmd=%p pass=%d topo=%d srcDiffuse=(%.2f,%.2f,%.2f,%.2f) "
            "override=%d pushColor=(%.2f,%.2f,%.2f,%.2f) flags=(%.0f,%.0f,%.0f,%.0f) "
            "lineParams=(%.2f,%.2f,%.2f,%.2f) wideLine=%d stippleFactor=%.1f pattern=0x%04x patternRaw=0x%08x "
            "fillMode=%d fillModeOverride=%d overlayPass=%d transparent=%d vbuf=%p vertexCount=%u\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            color[0], color[1], color[2], color[3],
            useOverrideColor ? 1 : 0,
            push.color[0], push.color[1], push.color[2], push.color[3],
            push.flags[0], push.flags[1], push.flags[2], push.flags[3],
            push.lineParams[0], push.lineParams[1], push.lineParams[2],
            push.lineParams[3],
            useWideLine ? 1 : 0, stippleFactor,
            static_cast<unsigned>(command.state.raster.linePattern), patternRaw,
            static_cast<int>(command.state.raster.fillMode),
            fillModeOverride, overlayPass ? 1 : 0, transparent ? 1 : 0,
            (const void*)entry.vertexBuffer,
            static_cast<unsigned>(command.geometry.vertexCount));
  }

  if (envFlagEnabled("FC_VULKAN_MATRIX_DUMP") && s_debugFrame > 0
      && (s_debugFrame % 100 == 0) && s_dumpCmdCount < 12) {
    s_dumpCmdCount++;
    SbMat mm;
    command.modelMatrix.getValue(mm);
    SbMat vm;
    if (command.state.raster.scissorEnabled
        && command.pass == SO_RENDERPASS_OVERLAY) {
      command.viewMatrix.getValue(vm);
    }
    else {
      params.viewMatrix.getValue(vm);
    }
    fprintf(stderr,
            "[MATX] frame=%d cmd#%d pass=%d verts=%u overlay=%d "
            "scissor=%d model=\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n",
            s_debugFrame, s_dumpCmdCount - 1, static_cast<int>(command.pass),
            command.geometry.vertexCount, overlayPass ? 1 : 0,
            command.state.raster.scissorEnabled ? 1 : 0,
            mm[0][0], mm[0][1], mm[0][2], mm[0][3],
            mm[1][0], mm[1][1], mm[1][2], mm[1][3],
            mm[2][0], mm[2][1], mm[2][2], mm[2][3],
            mm[3][0], mm[3][1], mm[3][2], mm[3][3]);
    fprintf(stderr,
            "[MATX]   view=\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "[MATX]   proj=\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n",
            vm[0][0], vm[0][1], vm[0][2], vm[0][3],
            vm[1][0], vm[1][1], vm[1][2], vm[1][3],
            vm[2][0], vm[2][1], vm[2][2], vm[2][3],
            vm[3][0], vm[3][1], vm[3][2], vm[3][3],
            projValue[0][0], projValue[0][1], projValue[0][2], projValue[0][3],
            projValue[1][0], projValue[1][1], projValue[1][2], projValue[1][3],
            projValue[2][0], projValue[2][1], projValue[2][2], projValue[2][3],
            projValue[3][0], projValue[3][1], projValue[3][2], projValue[3][3]);
  }

  if (useWideLine) {
    // CPU-side quad expansion in clip space (line width and/or stipple);
    // the wide-line pipeline draws it as a triangle list.  Binds here so
    // the projection matrix (projValue) is already resolved.
    if (!this->expandWideLines(entry, command, params, projValue,
                               std::max(1.0f, command.state.raster.lineWidth))) {
      return;
    }
    VkDeviceSize wideOffset = 0;
    const VulkanCachedCommand::VulkanWideLineBuffer & wslot =
      entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
    vkCmdBindVertexBuffers(this->activeCommandBuffer, 0, 1, &wslot.buffer,
                           &wideOffset);
  }

  if (useWideLine) {
    static int wldrawDiag = 0;
    if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG") && wldrawDiag++ < 40) {
      fprintf(stderr, "[WLINE2] DRAW cmd=%p wideLineVertexCount=%u pass=%d\n",
              (const void*)&command, entry.wideLineVertexCount,
              static_cast<int>(command.pass));
    }
    vkCmdDraw(this->activeCommandBuffer, entry.wideLineVertexCount, 1, 0, 0);
  }
  else if (indexed) {
    vkCmdDrawIndexed(this->activeCommandBuffer, command.geometry.indexCount, 1, 0,
                     0, 0);
  }
  else {
    vkCmdDraw(this->activeCommandBuffer, command.geometry.vertexCount, 1, 0, 0);
  }
}

bool
SoVulkanRenderBackend::beginCommandBuffer()
{
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return vkBeginCommandBuffer(this->currentCommandBuffer(), &bi) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::endAndSubmit()
{
  VkCommandBuffer cmd = this->currentCommandBuffer();
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  const VkFence fence = this->frameFences[slot];
  if (vkQueueSubmit(this->queue, 1, &si, fence) != VK_SUCCESS) {
    // The fence stays unsignaled (no signal was requested), so beginFrame()
    // would wait forever on the next reuse of this slot.  Signal it by
    // submitting nothing and relying on the next frame's failure path, but
    // mark the slot as not pending so the wait is skipped; a submission
    // failure (typically device loss) leaves the backend unusable anyway.
    this->frameFencePending[slot] = 0;
    return false;
  }
  this->frameFencePending[slot] = 1;
  return true;
}

// --- Lifecycle ------------------------------------------------------------

void
SoVulkanRenderBackend::shutdown()
{
  if (!this->isInitialized()) return;

  vkQueueWaitIdle(this->queue);

  // The queue is idle, so every deferred resource is safe to release now.
  this->flushAllPendingDestroys();

  // Release uploads abandoned by a frame that aborted between the cache
  // update and the flush/finalize step.  Their copies were never recorded,
  // so synchronous destruction is safe here.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, upload.staging, this->allocator);
      vkFreeMemory(this->device, upload.stagingMemory, this->allocator);
    }
    if (upload.index < this->textureCache.size()) {
      this->destroyTextureEntry(this->textureCache[upload.index]);
    }
  }
  this->pendingUploads.clear();

  this->invalidateCache();

  for (auto & entry : this->pipelineCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->pipelineCache.clear();

  for (auto & entry : this->backgroundPipelineCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->backgroundPipelineCache.clear();

  if (this->renderPassFramebuffer != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(this->device, this->renderPassFramebuffer,
                         this->allocator);
    this->renderPassFramebuffer = VK_NULL_HANDLE;
  }
  for (auto & entry : this->renderPassCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyRenderPass(this->device, entry.second, this->allocator);
    }
  }
  this->renderPassCache.clear();
  this->renderPass = VK_NULL_HANDLE;
  if (this->fragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->fragmentModule, this->allocator);
    this->fragmentModule = VK_NULL_HANDLE;
  }
  if (this->vertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->vertexModule, this->allocator);
    this->vertexModule = VK_NULL_HANDLE;
  }
  if (this->wideLineFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->wideLineFragmentModule,
                          this->allocator);
    this->wideLineFragmentModule = VK_NULL_HANDLE;
  }
  if (this->wideLineVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->wideLineVertexModule,
                          this->allocator);
    this->wideLineVertexModule = VK_NULL_HANDLE;
  }
  if (this->backgroundFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->backgroundFragmentModule, this->allocator);
    this->backgroundFragmentModule = VK_NULL_HANDLE;
  }
  if (this->backgroundVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->backgroundVertexModule, this->allocator);
    this->backgroundVertexModule = VK_NULL_HANDLE;
  }
  if (this->backgroundPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->backgroundPipelineLayout, this->allocator);
    this->backgroundPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->pipelineLayout, this->allocator);
    this->pipelineLayout = VK_NULL_HANDLE;
  }
  if (this->lightingMapped != nullptr) {
    vkUnmapMemory(this->device, this->lightingMemory);
    this->lightingMapped = nullptr;
  }
  if (this->lightingBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->lightingBuffer, this->allocator);
    this->lightingBuffer = VK_NULL_HANDLE;
  }
  if (this->lightingMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->lightingMemory, this->allocator);
    this->lightingMemory = VK_NULL_HANDLE;
  }
  if (this->whiteSampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->whiteSampler, this->allocator);
    this->whiteSampler = VK_NULL_HANDLE;
  }
  if (this->whiteImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, this->whiteImageView, this->allocator);
    this->whiteImageView = VK_NULL_HANDLE;
  }
  if (this->whiteImage != VK_NULL_HANDLE) {
    vkDestroyImage(this->device, this->whiteImage, this->allocator);
    this->whiteImage = VK_NULL_HANDLE;
  }
  if (this->whiteImageMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->whiteImageMemory, this->allocator);
    this->whiteImageMemory = VK_NULL_HANDLE;
  }
  this->whiteDescriptorSet = VK_NULL_HANDLE;
  for (VkDescriptorPool pool : this->descriptorPools) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(this->device, pool, this->allocator);
    }
  }
  this->descriptorPools.clear();
  this->descriptorPool = VK_NULL_HANDLE;
  this->descriptorSetCount = 0;
  if (this->descriptorSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->descriptorSetLayout,
                                 this->allocator);
    this->descriptorSetLayout = VK_NULL_HANDLE;
  }
  this->releaseFrameResources();
  if (this->commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->commandPool, this->allocator);
    this->commandPool = VK_NULL_HANDLE;
  }

  this->instance = VK_NULL_HANDLE;
  this->physicalDevice = VK_NULL_HANDLE;
  this->device = VK_NULL_HANDLE;
  this->queue = VK_NULL_HANDLE;
  this->allocator = nullptr;

  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

SbBool
SoVulkanRenderBackend::renderInternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      const bool overlaysOnly)
{
  if (!this->isInitialized()) {
    this->emitError("render called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "render called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }

  this->debugValidateDrawList(drawlist);

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  if (overlaysOnly) {
    bool hasOverlay = false;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      if (drawlist.getCommand(i).pass == SO_RENDERPASS_OVERLAY) {
        hasOverlay = true;
        break;
      }
    }
    if (!hasOverlay) return TRUE;
  }

  // One frame boundary: advances the ring cursor and releases resources
  // deferred maxFramesInFlight frames ago.
  this->beginFrame();

  // Render passes are cached by their attachment identity (formats, sample
  // count, image layouts), not by the target's images: swapchain targets
  // cycle their images every frame, and pipelines are keyed on the render
  // pass handle, so reusing the pass across image changes keeps the pipeline
  // cache warm.
  this->renderPass = this->getOrCreateRenderPass(*target);
  if (this->renderPass == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan render pass");
    return FALSE;
  }

  this->updateGeometryCache(drawlist, overlaysOnly);

  // Composite renders skip recordFrame(), so reserve the ring slots here;
  // beginFrame() above already advanced the frame cursor.  This covers both
  // the OVERLAY commands and the non-triangle residual geometry.
  if (overlaysOnly &&
      !this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return FALSE;
  }

  if (!this->beginCommandBuffer()) {
    this->emitError("failed to begin Vulkan command buffer");
    return FALSE;
  }

  // The framebuffer is cached for the current target identity (image views +
  // extent + render pass) and recreated whenever any of those change.  The
  // old framebuffer is released through the deferred ring: an older in-flight
  // submission may still reference it (the per-frame vkQueueWaitIdle is gone,
  // so only the current slot's fence has been waited by beginFrame()).
  if (this->renderPassFramebuffer == VK_NULL_HANDLE ||
      this->renderPassFramebufferPass != this->renderPass ||
      this->renderPassFramebufferColorImage != target->colorImage ||
      this->renderPassFramebufferColorView != target->colorImageView ||
      this->renderPassFramebufferDepthImage != target->depthImage ||
      this->renderPassFramebufferDepthView != target->depthImageView ||
      this->renderPassFramebufferExtent.width != target->extent.width ||
      this->renderPassFramebufferExtent.height != target->extent.height) {
    if (this->renderPassFramebuffer != VK_NULL_HANDLE) {
      const VkDevice device = this->device;
      const VkAllocationCallbacks * allocator = this->allocator;
      const VkFramebuffer oldFramebuffer = this->renderPassFramebuffer;
      this->deferDestroy([device, allocator, oldFramebuffer]() {
        if (oldFramebuffer != VK_NULL_HANDLE) {
          vkDestroyFramebuffer(device, oldFramebuffer, allocator);
        }
      });
      this->renderPassFramebuffer = VK_NULL_HANDLE;
    }
    VkFramebufferCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass = this->renderPass;
    fci.attachmentCount =
      (target->depthImageView != VK_NULL_HANDLE &&
       target->depthFormat != VK_FORMAT_UNDEFINED)
        ? 2u : 1u;
    const VkImageView attachments[] = {
      target->colorImageView,
      target->depthImageView,
    };
    fci.pAttachments = attachments;
    fci.width = target->extent.width;
    fci.height = target->extent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(this->device, &fci, this->allocator,
                            &this->renderPassFramebuffer) != VK_SUCCESS) {
      this->emitError("failed to create Vulkan framebuffer");
      // The one-shot command buffer was begun above and never submitted; an
      // implicit reset only happens on submission, so reset it explicitly or
      // every later beginCommandBuffer() will fail.
      vkEndCommandBuffer(this->currentCommandBuffer());
      vkResetCommandBuffer(this->currentCommandBuffer(), 0);
      return FALSE;
    }
    this->renderPassFramebufferPass = this->renderPass;
    this->renderPassFramebufferColorImage = target->colorImage;
    this->renderPassFramebufferColorView = target->colorImageView;
    this->renderPassFramebufferDepthImage = target->depthImage;
    this->renderPassFramebufferDepthView = target->depthImageView;
    this->renderPassFramebufferExtent = target->extent;
  }
  const VkFramebuffer framebuffer = this->renderPassFramebuffer;

  // Record the pending texture copies into the frame command buffer (one
  // submit for the whole frame instead of a separate transfer submit) and
  // finalize the host-side resources.  The draws that sample these textures
  // are recorded below, after the copies, and the descriptor sets they bind
  // must already exist.  Staging buffers are released through the deferred
  // ring once the slot fence signals.
  if (!this->recordPendingTextureUploads()) {
    this->emitError("failed to record texture uploads");
  }
  this->finalizePendingTextureUploads();

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = this->renderPass;
  rpbi.framebuffer = framebuffer;
  rpbi.renderArea.offset = {0, 0};
  rpbi.renderArea.extent = target->extent;
  rpbi.clearValueCount = 0;
  rpbi.pClearValues = nullptr;

  vkCmdBeginRenderPass(this->currentCommandBuffer(), &rpbi,
                       VK_SUBPASS_CONTENTS_INLINE);

  this->activeCommandBuffer = this->currentCommandBuffer();
  bool recorded = true;
  if (overlaysOnly) {
    this->recordTracedComposite(drawlist, params, *target, this->renderPass);
    this->recordOverlayBlock(drawlist, params, *target, this->renderPass);
  }
  else {
    recorded = this->recordFrame(drawlist, params, *target, this->renderPass);
  }
  this->activeCommandBuffer = VK_NULL_HANDLE;

  vkCmdEndRenderPass(this->currentCommandBuffer());

  // Submit even when recordFrame() failed: an unsubmitted one-shot command
  // buffer cannot be reused, and a partial frame is preferable to a dead
  // backend.
  const bool submitted = this->endAndSubmit();
  if (!submitted) {
    this->emitError("failed to submit Vulkan command buffer");
    return FALSE;
  }
  if (!recorded) {
    this->emitError("recordFrame failed; submitted a partial frame");
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderBackend::render(const SoDrawList & drawlist,
                              const SoRenderParams & params)
{
  return this->renderInternal(drawlist, params, false);
}

SbBool
SoVulkanRenderBackend::renderOverlaysOnly(const SoDrawList & drawlist,
                                          const SoRenderParams & params)
{
  return this->renderInternal(drawlist, params, true);
}

SbBool
SoVulkanRenderBackend::renderExternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass)
{
  if (!this->isInitialized()) {
    this->emitError("renderExternal called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "renderExternal called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    this->emitError(
      "renderExternal called without a command buffer and render pass");
    return FALSE;
  }

  this->debugValidateDrawList(drawlist);

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  this->beginFrame();
  this->updateGeometryCache(drawlist);
  if (!this->flushPendingTextureUploadsExternal()) {
    this->emitError("renderExternal: texture upload failed");
    return FALSE;
  }

  this->activeCommandBuffer = commandBuffer;
  const bool recorded = this->recordFrame(drawlist, params, *target, renderPass);
  this->activeCommandBuffer = VK_NULL_HANDLE;
  return recorded ? TRUE : FALSE;
}

SbBool
SoVulkanRenderBackend::renderExternalOverlay(const SoDrawList & drawlist,
                                             const SoRenderParams & params,
                                             VkCommandBuffer commandBuffer,
                                             VkRenderPass renderPass)
{
  if (!this->isInitialized()) {
    this->emitError(
      "renderExternalOverlay called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "renderExternalOverlay called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    this->emitError(
      "renderExternalOverlay called without a command buffer and render "
      "pass");
    return FALSE;
  }

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  this->beginFrame();
  this->updateGeometryCache(drawlist, true);

  // This path never goes through recordFrame(), so reserve the slots it will
  // consume (OVERLAY commands plus the non-triangle residual geometry);
  // otherwise the cursor keeps climbing across frames and eventually
  // overflows the lighting UBO.
  if (!this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return FALSE;
  }
  if (!this->flushPendingTextureUploadsExternal()) {
    this->emitError("renderExternalOverlay: texture upload failed");
    return FALSE;
  }

  this->activeCommandBuffer = commandBuffer;
  this->recordTracedComposite(drawlist, params, *target, renderPass);
  this->recordOverlayBlock(drawlist, params, *target, renderPass);
  this->activeCommandBuffer = VK_NULL_HANDLE;
  return TRUE;
}

bool
SoVulkanRenderBackend::recordFrame(const SoDrawList & drawlist,
                                   const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target,
                                   VkRenderPass renderPass)
{
  if (envFlagEnabled("FC_VULKAN_MATRIX_DUMP")) {
    s_debugFrame++;
    s_dumpCmdCount = 0;
  }
  this->applyViewport(params, target);
  this->recordClear(params, target);
  this->recordBackground(params, target, renderPass);
  // The background pass overrides the viewport/scissor for its own draw;
  // restore the viewport from params before recording geometry so draws
  // land in the requested region.
  this->applyViewport(params, target);

  // Vulkan-only display options, configured through setWireframeOverlay()/
  // setPointsOverlay()/setEdgeColor().  Environment variables act as a
  // diagnostic fallback for the command line.
  const bool wireframeOverlay =
    this->wireframeOverlay || envFlagEnabled("FC_VULKAN_WIREFRAME");
  const bool pointsOverlay =
    this->pointsOverlay || envFlagEnabled("FC_VULKAN_POINTS");
  float overlayColor[4] = {
    this->edgeColor[0], this->edgeColor[1], this->edgeColor[2],
    this->edgeColor[3]
  };
  // Parse the FC_VULKAN_EDGE_COLOR override (a diagnostic switch) once; it is
  // process-lifetime and this runs on the overlay path, so two getenv() calls
  // per frame is pure overhead.  Only RGB is taken from the hex value; alpha
  // keeps the configured edgeColor's.
  struct EdgeColorOverride { bool present; float rgb[3]; };
  static const EdgeColorOverride edgeOverride = []() {
    EdgeColorOverride o{false, {0.0f, 0.0f, 0.0f}};
    const char * hex = getenv("FC_VULKAN_EDGE_COLOR");
    if (hex) {
      unsigned int value = 0;
      if (sscanf(hex, "%x", &value) == 1) {
        o.present = true;
        o.rgb[0] = ((value >> 16) & 0xff) / 255.0f;
        o.rgb[1] = ((value >> 8) & 0xff) / 255.0f;
        o.rgb[2] = (value & 0xff) / 255.0f;
      }
    }
    return o;
  }();
  if (edgeOverride.present) {
    overlayColor[0] = edgeOverride.rgb[0];
    overlayColor[1] = edgeOverride.rgb[1];
    overlayColor[2] = edgeOverride.rgb[2];
  }
  // No overlay when neither is requested; otherwise re-draw opaque geometry
  // in the requested draw style (SoDrawStyleElement encoding: LINES=1,
  // POINTS=2) using a uniform edge color.
  const int overlayFillMode = wireframeOverlay
    ? SoDrawStyleElement::LINES
    : (pointsOverlay ? SoDrawStyleElement::POINTS : -1);
  if (envFlagEnabled("FC_VULKAN_BACKEND_DEBUG")) {
    static int overlayLog = 0;
    if (overlayLog++ < 3) {
      fprintf(stderr,
              "[OVL] wireframe=%d points=%d fillMode=%d edgeColor=(%.2f,%.2f,%.2f,%.2f)\n",
              wireframeOverlay ? 1 : 0, pointsOverlay ? 1 : 0, overlayFillMode,
              overlayColor[0], overlayColor[1], overlayColor[2], overlayColor[3]);
    }
  }

  // Reserve per-draw lighting slots for the worst case (main pass plus
  // overlay redraws) before recording, so slotIndex can never overflow the
  // ring allocation (VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972).
  if (!this->prepareLightingSlots(countDrawCommands(drawlist,
                                                    overlayFillMode))) {
    return FALSE;
  }

  // Opaque then transparent, honoring the draw-list sort order.  Overlay
  // commands (SO_RENDERPASS_OVERLAY) are handled exclusively by the
  // dedicated overlay block below: they carry their own view/projection
  // matrices and viewport, so recording them here with the main camera
  // matrices would draw garbage into the scene depth buffer.
  const std::vector<int> & order = drawlist.getSortedOrder();
  for (int passIndex = 0; passIndex < 2; ++passIndex) {
    const bool transparent = passIndex == 1;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      const int index =
        i < static_cast<int>(order.size()) ? order[i] : i;
      const SoRenderCommand & command = drawlist.getCommand(index);
      if (command.pass == SO_RENDERPASS_OVERLAY) continue;
      const bool isTransparent = command.pass == SO_RENDERPASS_TRANSPARENT;
      if (isTransparent != transparent) continue;
      this->recordDrawCommand(drawlist, command, target, params, renderPass,
                              transparent);
    }

    // Wireframe/point overlay: re-draw opaque geometry in the requested fill
    // mode using a uniform edge color (no vertex colors, no texture, unlit).
    if (!transparent && overlayFillMode >= 0) {
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index =
          i < static_cast<int>(order.size()) ? order[i] : i;
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        this->recordDrawCommand(drawlist, command, target, params, renderPass,
                                false, overlayFillMode, overlayColor);
      }
    }
  }

  // On-top annotations: commands recorded with the depth test disabled are
  // drawn after both passes in insertion order.  This mirrors GL's delayed
  // annotations pass (glClear(GL_DEPTH_BUFFER_BIT) + re-render on top),
  // which clarify selection uses to show a highlighted shape through
  // occluding geometry: the base shape and its highlight are recorded with
  // the depth test off during traversal and deferred here so later-drawn
  // occluders cannot paint over them.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    if (command.state.depth.enabled) continue;
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false);
  }

  // Screen-space overlay geometry (navigation cube): drawn after both passes
  // into its own viewport, with the overlay rect's depth cleared first so the
  // overlay self-occludes independently of the main scene.
  this->recordOverlayBlock(drawlist, params, target, renderPass);

  return true;
}

void
SoVulkanRenderBackend::recordOverlayBlock(const SoDrawList & drawlist,
                                          const SoRenderParams & params,
                                          const SoVulkanRenderTarget & target,
                                          VkRenderPass renderPass)
{
  // Overlays are drawn in recorded (insertion) order, matching GL: the
  // draw-list sorted order is a painter's algorithm built from the main
  // scene's camera-space depth, which is meaningless for screen-space
  // overlay geometry and would shuffle the navigation cube's panels
  // relative to each other and to other overlays.
  int lastClearX = -1, lastClearY = -1, lastClearW = -1, lastClearH = -1;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass != SO_RENDERPASS_OVERLAY) continue;
    const SoRasterState & raster = command.state.raster;
    if (!raster.scissorEnabled || raster.scissorWidth <= 0 ||
        raster.scissorHeight <= 0) {
      continue;
    }
    if (raster.scissorX != lastClearX || raster.scissorY != lastClearY ||
        raster.scissorWidth != lastClearW ||
        raster.scissorHeight != lastClearH) {
      this->recordOverlayDepthClear(command, target);
      lastClearX = raster.scissorX;
      lastClearY = raster.scissorY;
      lastClearW = raster.scissorWidth;
      lastClearH = raster.scissorHeight;
    }
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, true);
  }
}

void
SoVulkanRenderBackend::recordTracedComposite(const SoDrawList & drawlist,
                                             const SoRenderParams & params,
                                             const SoVulkanRenderTarget & target,
                                             VkRenderPass renderPass)
{
  // Ray-tracing compositing residue: the RT backend traces only triangles, so
  // the OPAQUE/TRANSPARENT LINES / POINTS / LINE_STRIP commands (BRep edge
  // lines, point markers, polylines) are drawn here as a raster layer on top
  // of the path-traced image.  The present pass wrote the scene depth, so
  // each fragment is depth tested with the overlay's LESS_OR_EQUAL compare
  // (recordDrawCommand's overlay path): a front face's edge lies at the
  // traced surface's depth and passes, while a hidden back-facing edge is
  // farther and is culled -- matching the raster pipeline's silhouette edge
  // look.  Depth is not cleared here and depth write stays off.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    const SoPrimitiveTopology topo = command.geometry.topology;
    if (topo == SO_TOPOLOGY_TRIANGLES) continue;
    if (topo == SO_TOPOLOGY_TRIANGLE_STRIP) continue;

    const SoRasterState & raster = command.state.raster;
    // Apply the command's own viewport/scissor if it carries one, else the
    // whole-surface viewport (the default for scene geometry).
    this->applyCommandViewport(command, target);
    this->applyScissor(command, target);
    // overlayPass=false: these are scene-geometry edge/point commands, so
    // they must draw with the FRAME camera (params.projMatrix), not the
    // command's own projection matrix.  Passing overlayPass=true made them
    // use a stale per-command proj, displacing them off the traced surface
    // (the offset "phantom box").  They inherit the raster depth compare so
    // the present-pass depth occludes hidden edges.
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, false);
  }
}
