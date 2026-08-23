// src/rendering/SoRTXRenderBackend.cpp

#include "rendering/SoRTXRenderBackend.h"

#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vulkan/rt/PathTrace.spv.h"
#include "vulkan/rt/Raygen.spv.h"
#include "vulkan/rt/Miss.spv.h"
#include "vulkan/rt/ShadowMiss.spv.h"
#include "vulkan/rt/ClosestHit.spv.h"
#include "vulkan/rt/ShadowClosestHit.spv.h"
#include "vulkan/rt/PresentVertex.spv.h"
#include "vulkan/rt/PresentFragment.spv.h"

namespace {

// Environment flags are enabled by presence, but honor the conventional
// "VAR=0"/"false"/"off" opt-out values.
bool
envFlagEnabled(const char * name)
{
  const char * value = getenv(name);
  if (value == nullptr) return false;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "off") != 0;
}

// std430 mirror of the RTMaterial struct in PathTrace.glsl.  One record per
// draw command, indexed by the instance custom index (the command index).
// C++ packs the float arrays without padding, which matches std430: 5 vec4
// + 6 arrays of 8 vec4 = 80 + 768 = 848 bytes.
// (struct RTMaterial is defined in SoRTXRenderBackend.h; the static assert
// below pins its size to the shader layout.)
static_assert(sizeof(RTMaterial) == 880,
              "RTMaterial must match PathTrace.glsl std430 layout");

// std140 mirror of the FrameBlock uniform in Raygen/ClosestHit/Miss.glsl.
struct alignas(16) RTXFrameBlock {
  float view[16];
  float viewInverse[16];
  float projInverse[16];
  float cameraPos[4];
  float viewport[4]; // x = width, y = height, z = orthographic, w = unused
  float bgTop[4];
  float bgBottom[4];
  float state[4]; // x = frameIndex, y = pathTracing, z = accumulating,
                  // w = maxBounces
  float adaptive[4]; // x = minSamples, y = relErrorThreshold (0 = off)
};
static_assert(sizeof(RTXFrameBlock) == 3 * 64 + 6 * 16,
              "RTXFrameBlock must match FrameBlock std140 layout");

// Push constant block of the raygen shader (RaygenPush in Raygen.glsl).
// Per-frame state rides here instead of the shared UBO: the chit and miss
// stages never read it, so it stays out of the descriptor sets entirely.
struct alignas(16) RTXRaygenPush {
  uint32_t frameIndex = 0;
  uint32_t flags = 0; // bit 0 = path tracing, bit 1 = accumulating,
                      // bit 2 = debug fill
  uint32_t maxBounces = 4;
  uint32_t pad = 0;
};
static_assert(sizeof(RTXRaygenPush) == 16,
              "RTXRaygenPush must match RaygenPush layout");

constexpr int SBT_GROUP_COUNT = 5; // raygen, miss, shadow miss, chit, shadow chit

constexpr int MAX_SHADER_LIGHTS = 8;
constexpr int MAX_VERTEX_COUNT = 10000000;

// Pick the first memory type matching the desired properties, or any type
// the device offers for this resource as a fallback.
uint32_t
findMemoryType(VkPhysicalDevice physicalDevice,
               const VkMemoryRequirements & requirements,
               VkMemoryPropertyFlags desired)
{
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags & desired) == desired) {
      return i;
    }
  }
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if (requirements.memoryTypeBits & (1u << i)) {
      return i;
    }
  }
  return 0;
}

// FNV-1a content hash of a command's geometry, sampled so full-scene
// hashing stays sub-millisecond.  The producer's geometry storage is a
// per-frame arena, so this hash -- not pointer identity -- is the scene
// change signal for the geometry cache.
uint64_t
hashGeometry(const SoGeometryDesc & geometry, uint32_t vertexStride,
             bool indexed)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  mix(geometry.vertexCount);
  mix(geometry.indexCount);
  mix(vertexStride);

  const size_t posStrideFloats = vertexStride / sizeof(float);
  const size_t totalFloats =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  if (totalFloats > 0) {
    const size_t samples = 512;
    const size_t step = totalFloats > samples ? totalFloats / samples : 1;
    for (size_t i = 0; i < totalFloats; i += step) {
      mix(std::bit_cast<uint32_t>(geometry.positions[i]));
    }
    mix(std::bit_cast<uint32_t>(geometry.positions[totalFloats - 1]));
  }

  if (indexed) {
    const size_t count = geometry.indexCount;
    // Hash every index for scenes up to a practical threshold (covers
    // typical CAD parts in full); beyond that fall back to uniform
    // sampling so the per-frame cost stays bounded.  Either way the first
    // and last indices are always included.
    const size_t samples = 256;
    if (count <= 65536) {
      for (size_t i = 0; i < count; ++i) {
        mix(geometry.indices[i]);
      }
    }
    else {
      const size_t step = count / samples;
      for (size_t i = 0; i < count; i += step) {
        mix(geometry.indices[i]);
      }
    }
    mix(geometry.indices[count - 1]);
  }
  return h;
}

// A cheap per-frame "change signal" for one command's geometry.  It mixes
// only the metadata plus a sampled subset of positions and indices, so it is
// far cheaper than the full hashGeometry() walk (which hashes every index for
// scenes up to 65536 indices).  When this signal is unchanged from the cache
// entry, the geometry is assumed unchanged for cache purposes and the full
// hash is reused, avoiding the per-frame full-index walk over large CAD parts.
uint64_t
hashGeometrySignal(const SoGeometryDesc & geometry, uint32_t vertexStride,
                   bool indexed)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  mix(geometry.vertexCount);
  mix(geometry.indexCount);
  mix(vertexStride);

  const size_t posStrideFloats = vertexStride / sizeof(float);
  const size_t totalFloats =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  if (totalFloats > 0) {
    // Sample the position data (same sampling rate as hashGeometry).
    const size_t samples = 512;
    const size_t step = totalFloats > samples ? totalFloats / samples : 1;
    for (size_t i = 0; i < totalFloats; i += step) {
      mix(std::bit_cast<uint32_t>(geometry.positions[i]));
    }
    mix(std::bit_cast<uint32_t>(geometry.positions[totalFloats - 1]));
  }

  if (indexed && geometry.indexCount > 0) {
    // Sample indices (same rate as the large-scene fallback).
    const size_t count = geometry.indexCount;
    const size_t samples = 256;
    const size_t step = count > samples ? count / samples : 1;
    for (size_t i = 0; i < count; i += step) {
      mix(geometry.indices[i]);
    }
    mix(geometry.indices[count - 1]);
  }
  return h;
}

// FNV-1a hash of a command's vertex positions only.  Separates position
// edits (refit-able: topology unchanged) from index edits (topology change,
// full rebuild required).
uint64_t
hashPositions(const SoGeometryDesc & geometry, uint32_t vertexStride)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  const size_t posStrideFloats = vertexStride / sizeof(float);
  const size_t totalFloats =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  for (size_t i = 0; i < totalFloats; ++i) {
    mix(std::bit_cast<uint32_t>(geometry.positions[i]));
  }
  return h;
}

// FNV-1a hash of a command's index data only (full walk up to the same
// 65536 threshold as hashGeometry, uniform sampling beyond).
uint64_t
hashIndices(const SoGeometryDesc & geometry)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  const size_t count = geometry.indexCount;
  if (count <= 65536) {
    for (size_t i = 0; i < count; ++i) {
      mix(geometry.indices[i]);
    }
  }
  else {
    const size_t samples = 256;
    const size_t step = count / samples;
    for (size_t i = 0; i < count; i += step) {
      mix(geometry.indices[i]);
    }
    mix(geometry.indices[count - 1]);
  }
  return h;
}

} // namespace

SoRTXRenderBackend::SoRTXRenderBackend()
{
}

SoRTXRenderBackend::~SoRTXRenderBackend()
{
  if (this->isInitialized()) this->shutdown();
}

const char *
SoRTXRenderBackend::getName() const
{
  return "RTXRenderBackend";
}

void
SoRTXRenderBackend::setPathTracingEnabled(SbBool enabled)
{
  if (this->ptEnabled == enabled) return;
  this->ptEnabled = enabled;
  // Switching modes invalidates the accumulated image and any in-flight
  // progressive run.
  this->ptAccumulating = FALSE;
  this->ptStartLatch = FALSE;
  this->ptFrameIndex = 0;
  this->ptIdleFrames = 0;
  this->haveLastView = FALSE;
}

SbBool
SoRTXRenderBackend::getPathTracingEnabled(void) const
{
  return this->ptEnabled;
}

void
SoRTXRenderBackend::setPathTracingStart(SbBool start)
{
  if (!this->ptEnabled) {
    this->emitLog("setPathTracingStart ignored: path tracing is disabled");
    return;
  }
  if (start) {
    // Latch: the next frame resets the accumulation and starts a fresh
    // progressive run (even if the camera changed since the last frame).
    this->ptStartLatch = TRUE;
  }
  else {
    this->ptStartLatch = FALSE;
    this->ptAccumulating = FALSE;
    this->ptIdleFrames = 0;
  }
}

SbBool
SoRTXRenderBackend::getPathTracingActive(void) const
{
  return this->ptEnabled && this->ptAccumulating;
}

SbBool
SoRTXRenderBackend::getPathTracingRefining(void) const
{
  // Request continuous frames while working toward a converged image: while
  // accumulating, and during the short post-move settle window (ptIdleFrames
  // below the settle threshold) so the auto-restart has frames to count.
  // After convergence ptIdleFrames is saturated at ptSettleFrames, so this
  // reads FALSE and the viewport can go idle.
  return this->ptEnabled &&
    (this->ptAccumulating || this->ptIdleFrames < this->ptSettleFrames);
}

uint32_t
SoRTXRenderBackend::getPathTracingSampleCount(void) const
{
  return this->ptAccumulating ? this->ptFrameIndex + 1 : 0;
}

void
SoRTXRenderBackend::setPathTracingBounces(const uint32_t bounces)
{
  this->ptMaxBounces = std::max(1u, std::min(16u, bounces));
}

void
SoRTXRenderBackend::setPathTracingSettleFrames(const uint32_t frames)
{
  this->ptSettleFrames = std::max(1u, std::min(120u, frames));
}

void
SoRTXRenderBackend::setPathTracingDenoiseEnabled(SbBool enabled)
{
  this->ptDenoise = enabled;
}

SbBool
SoRTXRenderBackend::initialize(const SoRenderBackendInitParams & params)
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
      "SoRTXRenderBackend requires a SoVulkanDeviceContext in "
      "SoRenderBackendInitParams::userData");
    return FALSE;
  }
  if (deviceContext->apiVersion < VK_API_VERSION_1_2) {
    this->emitError("SoRTXRenderBackend requires a Vulkan 1.2+ device");
    return FALSE;
  }

  this->instance = deviceContext->instance;
  this->physicalDevice = deviceContext->physicalDevice;
  this->device = deviceContext->device;
  this->queue = deviceContext->graphicsQueue;
  this->queueFamilyIndex = deviceContext->graphicsQueueFamilyIndex;
  this->allocator = deviceContext->allocator;

  // The system loader only exports core entry points; resolve the ray
  // tracing KHR functions per-device.  Failing here means the device is
  // missing the acceleration-structure/ray-query extensions (or the loader
  // version cannot reach them), and the RT backend cannot function.
  this->vkDestroyAccelerationStructureKHR =
    reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
      vkGetDeviceProcAddr(this->device, "vkDestroyAccelerationStructureKHR"));
  this->vkGetAccelerationStructureBuildSizesKHR =
    reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
      vkGetDeviceProcAddr(this->device, "vkGetAccelerationStructureBuildSizesKHR"));
  this->vkCreateAccelerationStructureKHR =
    reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
      vkGetDeviceProcAddr(this->device, "vkCreateAccelerationStructureKHR"));
  this->vkCmdBuildAccelerationStructuresKHR =
    reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
      vkGetDeviceProcAddr(this->device, "vkCmdBuildAccelerationStructuresKHR"));
  this->vkGetAccelerationStructureDeviceAddressKHR =
    reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
      vkGetDeviceProcAddr(this->device, "vkGetAccelerationStructureDeviceAddressKHR"));
  if (!this->vkDestroyAccelerationStructureKHR ||
      !this->vkGetAccelerationStructureBuildSizesKHR ||
      !this->vkCreateAccelerationStructureKHR ||
      !this->vkCmdBuildAccelerationStructuresKHR ||
      !this->vkGetAccelerationStructureDeviceAddressKHR) {
    this->emitError(
      "failed to resolve ray tracing KHR entry points; the device or "
      "loader does not provide VK_KHR_acceleration_structure");
    this->shutdown();
    return FALSE;
  }

  // The ray tracing pipeline (VK_KHR_ray_tracing_pipeline) entry points
  // power the shader binding table dispatch.
  this->vkCreateRayTracingPipelinesKHR =
    reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
      vkGetDeviceProcAddr(this->device, "vkCreateRayTracingPipelinesKHR"));
  this->vkGetRayTracingShaderGroupHandlesKHR =
    reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
      vkGetDeviceProcAddr(this->device, "vkGetRayTracingShaderGroupHandlesKHR"));
  this->vkCmdTraceRaysKHR =
    reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
      vkGetDeviceProcAddr(this->device, "vkCmdTraceRaysKHR"));
  if (!this->vkCreateRayTracingPipelinesKHR ||
      !this->vkGetRayTracingShaderGroupHandlesKHR ||
      !this->vkCmdTraceRaysKHR) {
    this->emitError(
      "failed to resolve VK_KHR_ray_tracing_pipeline entry points; the "
      "device or loader does not provide the ray tracing pipeline");
    this->shutdown();
    return FALSE;
  }

  // Dispatch mode: the SBT pipeline is opt-in (FC_VULKAN_RT_SBT=1); the
  // default ray-query compute path avoids a hang in NVIDIA driver 610.x
  // where triangle hit-group execution stalls the GPU.
  this->useSbtPipeline = envFlagEnabled("FC_VULKAN_RT_SBT");

  // All entry points are resolved from here on.  Mark the backend
  // initialized before creating resources so that a failure in any
  // create*() below runs the full (null-tolerant) shutdown() cleanup
  // instead of leaking every handle created so far.
  this->setInitialized(TRUE);

  // Query the pipeline properties needed for the SBT record layout, plus the
  // acceleration-structure properties for the scratch buffer alignment
  // (VUID-vkCmdBuildAccelerationStructuresKHR-scratchData-*): the scratch
  // device address must be aligned to
  // minAccelerationStructureScratchOffsetAlignment, which the buffer's own
  // memory requirements do not guarantee.
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps {};
  rtProps.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps {};
  asProps.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  rtProps.pNext = &asProps;
  VkPhysicalDeviceProperties2 props2 {};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &rtProps;
  vkGetPhysicalDeviceProperties2(this->physicalDevice, &props2);
  this->asScratchAlignment =
    std::max<VkDeviceSize>(asProps.minAccelerationStructureScratchOffsetAlignment, 1u);
  this->sbtGroupHandleSize = rtProps.shaderGroupHandleSize;
  this->sbtGroupBaseAlignment = std::max(rtProps.shaderGroupBaseAlignment, 1u);
  // The record stride must satisfy both the handle alignment and the
  // base alignment, because every strided region address has to be a
  // multiple of shaderGroupBaseAlignment (VUID-vkCmdTraceRaysKHR-*).
  const uint32_t alignment = std::max({
    rtProps.shaderGroupHandleAlignment,
    rtProps.shaderGroupBaseAlignment,
    1u});
  this->sbtRecordSize = this->sbtGroupHandleSize;
  this->sbtRecordSize += alignment - 1;
  this->sbtRecordSize -= this->sbtRecordSize % alignment;

  if (!this->createDescriptorSetLayout()) {
    this->emitError("failed to create RT descriptor set layout");
    this->shutdown();
    return FALSE;
  }
  if (!this->createDescriptorPool()) {
    this->emitError("failed to create RT descriptor pool");
    this->shutdown();
    return FALSE;
  }
  if (!this->createShaderModules()) {
    this->emitError("failed to create RT shader modules");
    this->shutdown();
    return FALSE;
  }
  if (!this->createPipelines()) {
    this->emitError("failed to create ray tracing pipeline");
    this->shutdown();
    return FALSE;
  }
  if (!this->createFrameBuffer()) {
    this->emitError("failed to create RT frame uniform buffer");
    this->shutdown();
    return FALSE;
  }

  // Optional path tracing tuning (kept out of the public API for now).
  if (const char * bounces = getenv("FC_VULKAN_PT_BOUNCES")) {
    const int value = std::atoi(bounces);
    if (value >= 1 && value <= 16) {
      this->ptMaxBounces = static_cast<uint32_t>(value);
    }
  }
  if (const char * settle = getenv("FC_VULKAN_PT_SETTLE")) {
    const int value = std::atoi(settle);
    if (value >= 1 && value <= 120) {
      this->ptSettleFrames = static_cast<uint32_t>(value);
    }
  }
  if (const char * maxsamples = getenv("FC_VULKAN_PT_MAXSAMPLES")) {
    const int value = std::atoi(maxsamples);
    if (value >= 1 && value <= 100000) {
      this->ptMaxSamples = static_cast<uint32_t>(value);
    }
  }
  // Adaptive sampling tuning (see PathTrace.glsl u_adaptive).
  if (const char * adaptive = getenv("FC_VULKAN_PT_ADAPTIVE")) {
    this->ptAdaptiveEnabled = std::atoi(adaptive) != 0 ? TRUE : FALSE;
  }
  if (const char * minsamples = getenv("FC_VULKAN_PT_MIN_SAMPLES")) {
    const int value = std::atoi(minsamples);
    if (value >= 1 && value <= 256) {
      this->ptAdaptiveMinSamples = static_cast<uint32_t>(value);
    }
  }
  if (const char * threshold = getenv("FC_VULKAN_PT_THRESHOLD")) {
    const float value = static_cast<float>(std::atof(threshold));
    if (value > 0.0f && value <= 1.0f) {
      this->ptAdaptiveThreshold = value;
    }
  }
  if (const char * stopfraction = getenv("FC_VULKAN_PT_STOP_FRACTION")) {
    const float value = static_cast<float>(std::atof(stopfraction));
    if (value > 0.0f && value <= 1.0f) {
      this->ptAdaptiveStopFraction = value;
    }
  }

  this->setInitialized(TRUE);
  this->emitLog("initialized (Vulkan ray tracing)");
  return TRUE;
}

bool
SoRTXRenderBackend::createDescriptorSetLayout()
{
  // Ray tracing descriptor set: bindings 0-7 (see Raygen.glsl and
  // ClosestHit.glsl).  Stage flags mirror the consumers: the raygen traces
  // rays and writes the image/accum/G-buffers, the miss shader samples the
  // frame UBO, and the closest-hit shader reads materials, the frame UBO
  // and the triangle-normal pool.
  VkDescriptorSetLayoutBinding bindings[10] {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[3].binding = 3;
  bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

  // Path tracing: accumulation buffer, first-bounce G-buffers (written by
  // the raygen) and the triangle-normal pool (read by the closest hit).
  for (uint32_t b = 4; b <= 6; ++b) {
    bindings[b].binding = b;
    bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[b].descriptorCount = 1;
    bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
      VK_SHADER_STAGE_COMPUTE_BIT;
  }
  bindings[7].binding = 7;
  bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[7].descriptorCount = 1;
  bindings[7].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  // Adaptive sampling: per-pixel sums-of-squares and the active-pixel
  // counter (compute-tracer only).
  bindings[8].binding = 8;
  bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[8].descriptorCount = 1;
  bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[9].binding = 9;
  bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[9].descriptorCount = 1;
  bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 10;
  ci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(this->device, &ci, this->allocator,
                                  &this->rtSetLayout) != VK_SUCCESS) {
    return false;
  }

  // Present descriptor set: combined image sampler at binding 1 (the raw
  // traced image for the preview mode) plus the accumulation and G-buffer
  // storage buffers at bindings 2-4 (the denoising path tracing path).
  VkDescriptorSetLayoutBinding presentBindings[4] {};
  presentBindings[0].binding = 1;
  presentBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  presentBindings[0].descriptorCount = 1;
  presentBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  for (uint32_t b = 2; b <= 4; ++b) {
    presentBindings[b - 1].binding = b;
    presentBindings[b - 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentBindings[b - 1].descriptorCount = 1;
    presentBindings[b - 1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }

  VkDescriptorSetLayoutCreateInfo pci {};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  pci.bindingCount = 4;
  pci.pBindings = presentBindings;
  return vkCreateDescriptorSetLayout(this->device, &pci, this->allocator,
                                     &this->presentSetLayout) == VK_SUCCESS;
}

bool
SoRTXRenderBackend::createDescriptorPool()
{
  VkDescriptorPoolSize sizes[5] {};
  sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  sizes[0].descriptorCount = 2;
  sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  sizes[1].descriptorCount = 2;
  sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sizes[2].descriptorCount = 2;
  sizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sizes[3].descriptorCount = 2;
  sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sizes[4].descriptorCount = 20;

  VkDescriptorPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  ci.maxSets = 4;
  ci.poolSizeCount = 5;
  ci.pPoolSizes = sizes;
  return vkCreateDescriptorPool(this->device, &ci, this->allocator,
                                &this->descriptorPool) == VK_SUCCESS;
}

bool
SoRTXRenderBackend::createShaderModules()
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
  if (!load(coin_vulkan_rt_pathtrace_spirv,
            coin_vulkan_rt_pathtrace_spirv_count, this->pathTraceModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_raygen_spirv,
            coin_vulkan_rt_raygen_spirv_count, this->raygenModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_miss_spirv,
            coin_vulkan_rt_miss_spirv_count, this->missModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_shadowmiss_spirv,
            coin_vulkan_rt_shadowmiss_spirv_count, this->shadowMissModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_closesthit_spirv,
            coin_vulkan_rt_closesthit_spirv_count, this->closestHitModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_shadowclosesthit_spirv,
            coin_vulkan_rt_shadowclosesthit_spirv_count,
            this->shadowClosestHitModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_presentvertex_spirv,
            coin_vulkan_rt_presentvertex_spirv_count,
            this->presentVertexModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_presentfragment_spirv,
            coin_vulkan_rt_presentfragment_spirv_count,
            this->presentFragmentModule)) {
    return false;
  }
  return true;
}

bool
SoRTXRenderBackend::createDeviceLocalBuffer(VkDeviceSize size,
                                            VkBufferUsageFlags usage,
                                            VkBuffer & buffer,
                                            VkDeviceMemory & memory)
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
  VkMemoryAllocateFlagsInfo allocFlags {};
  allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = findMemoryType(this->physicalDevice, requirements,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  // Buffers carrying SHADER_DEVICE_ADDRESS_BIT must be allocated with the
  // device-address memory flag (VUID-VkMemoryAllocateInfo-flags-03339).
  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    ai.pNext = &allocFlags;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator, &memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(this->device, buffer, memory, 0);
  return true;
}

// Host-visible + host-coherent buffer (frame UBO, material buffer, instances,
// SBT, staging uploads).
bool
SoRTXRenderBackend::createHostVisibleBuffer(VkDeviceSize size,
                                            VkBufferUsageFlags usage,
                                            VkBuffer & buffer,
                                            VkDeviceMemory & memory)
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
  VkMemoryAllocateFlagsInfo allocFlags {};
  allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = findMemoryType(
    this->physicalDevice, requirements,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  // Buffers carrying SHADER_DEVICE_ADDRESS_BIT must be allocated with the
  // device-address memory flag (VUID-VkMemoryAllocateInfo-flags-03339).
  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    ai.pNext = &allocFlags;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator, &memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(this->device, buffer, memory, 0);
  return true;
}

VkDeviceAddress
SoRTXRenderBackend::getDeviceAddress(VkBuffer buffer)
{
  VkBufferDeviceAddressInfo info {};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  info.buffer = buffer;
  return vkGetBufferDeviceAddress(this->device, &info);
}

bool
SoRTXRenderBackend::createScratchBuffer(VkDeviceSize size)
{
  // The scratch device address must be aligned to
  // minAccelerationStructureScratchOffsetAlignment (queried in initialize()),
  // which VkMemoryRequirements of the buffer itself does not guarantee.
  // Overallocate by the alignment and expose the aligned address as
  // scratchAddress.
  const VkDeviceSize alignment = this->asScratchAlignment;
  const VkDeviceSize padded = size + alignment;
  if (this->scratchBuffer != VK_NULL_HANDLE && padded <= this->scratchSize) {
    return true;
  }
  // The old buffer must not be destroyed here: builds recorded earlier in
  // the active command buffer still reference its device address
  // (VUID-vkDestroyBuffer-buffer-00922).  Freeing it now invalidates those
  // references and faults the GPU when the BLAS/TLAS builds execute.
  // Destroy it after the submission completed instead.
  if (this->scratchBuffer != VK_NULL_HANDLE) {
    this->pendingStagingDestroys.emplace_back(this->scratchBuffer,
                                              this->scratchMemory);
    this->scratchBuffer = VK_NULL_HANDLE;
    this->scratchMemory = VK_NULL_HANDLE;
  }
  this->scratchSize = padded;
  if (!this->createDeviceLocalBuffer(
        padded, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        this->scratchBuffer, this->scratchMemory)) {
    this->scratchSize = 0;
    return false;
  }
  const VkDeviceAddress base = this->getDeviceAddress(this->scratchBuffer);
  const VkDeviceAddress offset = (alignment - (base % alignment)) % alignment;
  this->scratchAddress = base + offset;
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr,
            "[RTDBG] scratch: requiredAlignment=%llu base=0x%llx "
            "aligned=0x%llx offset=%llu size=%llu\n",
            static_cast<unsigned long long>(alignment),
            static_cast<unsigned long long>(base),
            static_cast<unsigned long long>(this->scratchAddress),
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(this->scratchSize));
  }
  return true;
}

bool
SoRTXRenderBackend::createStorageImage(uint32_t width, uint32_t height)
{
  if (this->storageImage != VK_NULL_HANDLE &&
      this->storageWidth == width && this->storageHeight == height) {
    return true;
  }
  if (this->storageImage != VK_NULL_HANDLE) {
    // The previous frame's submission may still sample this image; release
    // it after the next frame boundary instead of destroying it now.
    VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkImage image = this->storageImage;
    const VkImageView view = this->storageImageView;
    const VkDeviceMemory memory = this->storageImageMemory;
    this->deferDestroy([device, allocator, image, view, memory]() {
      vkDestroyImageView(device, view, allocator);
      vkDestroyImage(device, image, allocator);
      vkFreeMemory(device, memory, allocator);
    });
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageView = VK_NULL_HANDLE;
    this->storageImageMemory = VK_NULL_HANDLE;
  }
  this->storageWidth = width;
  this->storageHeight = height;

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.extent = {width, height, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(this->device, &ci, this->allocator,
                    &this->storageImage) != VK_SUCCESS) {
    this->storageWidth = 0;
    this->storageHeight = 0;
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(this->device, this->storageImage, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = findMemoryType(this->physicalDevice, requirements,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(this->device, &ai, this->allocator,
                       &this->storageImageMemory) != VK_SUCCESS) {
    vkDestroyImage(this->device, this->storageImage, this->allocator);
    this->storageImage = VK_NULL_HANDLE;
    this->storageWidth = 0;
    this->storageHeight = 0;
    return false;
  }
  vkBindImageMemory(this->device, this->storageImage, this->storageImageMemory,
                    0);

  VkImageViewCreateInfo vci {};
  vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vci.image = this->storageImage;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  vci.subresourceRange.layerCount = 1;
  vci.subresourceRange.levelCount = 1;
  if (vkCreateImageView(this->device, &vci, this->allocator,
                        &this->storageImageView) != VK_SUCCESS) {
    vkDestroyImage(this->device, this->storageImage, this->allocator);
    vkFreeMemory(this->device, this->storageImageMemory, this->allocator);
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageMemory = VK_NULL_HANDLE;
    this->storageWidth = 0;
    this->storageHeight = 0;
    return false;
  }

  // The image/view/sampler identity changed.  The previous sampler (if any)
  // may still be referenced by an in-flight present pass; release it at the
  // next frame boundary instead of leaking it.  The image/view/memory were
  // deferred-destroyed above.
  if (this->presentSampler != VK_NULL_HANDLE) {
    VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkSampler oldSampler = this->presentSampler;
    this->presentSampler = VK_NULL_HANDLE;
    this->deferDestroy([device, allocator, oldSampler]() {
      vkDestroySampler(device, oldSampler, allocator);
    });
  }

  VkSamplerCreateInfo sci {};
  sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sci.magFilter = VK_FILTER_NEAREST;
  sci.minFilter = VK_FILTER_NEAREST;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.maxLod = 0.0f;
  if (vkCreateSampler(this->device, &sci, this->allocator,
                      &this->presentSampler) != VK_SUCCESS) {
    // Unwind the image/view/memory created above so a later call retries
    // from scratch instead of early-outing on the cached dimensions with a
    // null sampler.
    vkDestroyImageView(this->device, this->storageImageView, this->allocator);
    vkDestroyImage(this->device, this->storageImage, this->allocator);
    vkFreeMemory(this->device, this->storageImageMemory, this->allocator);
    this->storageImageView = VK_NULL_HANDLE;
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageMemory = VK_NULL_HANDLE;
    this->storageWidth = 0;
    this->storageHeight = 0;
    return false;
  }
  // The image/view/sampler identity changed: mark the layout transition
  // pending and refresh both descriptor sets.
  this->storageImageNeedsLayoutInit = true;
  return this->updateDescriptors();
}

bool
SoRTXRenderBackend::ensureNormalPoolCapacity(VkDeviceSize bytes)
{
  if (this->normalPoolBuffer != VK_NULL_HANDLE &&
      this->normalPoolCapacity >= bytes) {
    return true;
  }
  // Grow-only pool: double until the requested size fits.  The new buffer
  // is created (and mapped) before the old one is released, so a failed
  // allocation leaves the previous pool intact and usable.  The old buffer
  // is only referenced by acceleration-structure-phase submissions, which
  // complete before the next pool resize can run (per-frame queue drain),
  // so releasing it here is safe.
  VkDeviceSize newCapacity = std::max<VkDeviceSize>(64 * 1024, bytes);
  while (newCapacity < this->normalPoolCapacity + bytes) {
    newCapacity *= 2;
  }
  VkBuffer newBuffer = VK_NULL_HANDLE;
  VkDeviceMemory newMemory = VK_NULL_HANDLE;
  void * newMapped = nullptr;
  if (!this->createHostVisibleBuffer(
        newCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        newBuffer, newMemory)) {
    return false;
  }
  if (vkMapMemory(this->device, newMemory, 0, newCapacity, 0,
                  &newMapped) != VK_SUCCESS) {
    vkDestroyBuffer(this->device, newBuffer, this->allocator);
    vkFreeMemory(this->device, newMemory, this->allocator);
    return false;
  }
  if (this->normalPoolBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->normalPoolBuffer, this->allocator);
    this->normalPoolBuffer = VK_NULL_HANDLE;
    vkFreeMemory(this->device, this->normalPoolMemory, this->allocator);
    this->normalPoolMemory = VK_NULL_HANDLE;
    this->normalPoolMapped = nullptr;
  }
  this->normalPoolCapacity = newCapacity;
  this->normalPoolBuffer = newBuffer;
  this->normalPoolMemory = newMemory;
  this->normalPoolMapped = newMapped;
  this->normalPoolUsed = 0;
  // The pool identity changed: refresh the descriptor sets.
  return this->updateDescriptors();
}

VkDeviceSize
SoRTXRenderBackend::appendTriangleNormals(const SoRenderCommand & command,
                                          RTXCachedGeometry & entry)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);
  const bool indexed = entry.indexCount > 0 && entry.idxKey != nullptr;
  const uint32_t triangleCount =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (triangleCount == 0) return 0;

  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(triangleCount) * 4 * sizeof(float);

  // Reuse the entry's existing pool slot when the triangle count is
  // unchanged; otherwise append (the pool grows over the session).
  const uint32_t existingOffset = entry.normalPoolOffset;
  const bool reuse = existingOffset != 0xFFFFFFFFu &&
    entry.normalCount == triangleCount &&
    (static_cast<VkDeviceSize>(existingOffset) * 16 + bytes) <=
      this->normalPoolUsed;
  if (!reuse) {
    if (!this->ensureNormalPoolCapacity(this->normalPoolUsed + bytes)) {
      this->emitError("appendTriangleNormals: pool allocation failed");
      return 0;
    }
    entry.normalPoolOffset =
      static_cast<uint32_t>(this->normalPoolUsed / (4 * sizeof(float)));
    this->normalPoolUsed += bytes;
  }
  entry.normalCount = triangleCount;

  // Object-space per-triangle geometric normals (flat shading).
  float * out = static_cast<float *>(this->normalPoolMapped) +
    static_cast<size_t>(entry.normalPoolOffset) * 4;
  const auto vertex = [&geometry, posStrideFloats](uint32_t i) {
    return geometry.positions + static_cast<size_t>(i) * posStrideFloats;
  };
  for (uint32_t t = 0; t < triangleCount; ++t) {
    const uint32_t i0 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 0] : t * 3 + 0;
    const uint32_t i1 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 1] : t * 3 + 1;
    const uint32_t i2 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 2] : t * 3 + 2;
    const float * p0 = vertex(i0);
    const float * p1 = vertex(i1);
    const float * p2 = vertex(i2);
    const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    float nx = e1[1] * e2[2] - e1[2] * e2[1];
    float ny = e1[2] * e2[0] - e1[0] * e2[2];
    float nz = e1[0] * e2[1] - e1[1] * e2[0];
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-12f) {
      nx /= len; ny /= len; nz /= len;
    }
    else {
      nx = 0.0f; ny = 0.0f; nz = 1.0f;
    }
    out[static_cast<size_t>(t) * 4 + 0] = nx;
    out[static_cast<size_t>(t) * 4 + 1] = ny;
    out[static_cast<size_t>(t) * 4 + 2] = nz;
    out[static_cast<size_t>(t) * 4 + 3] = 0.0f;
  }
  return bytes;
}

bool
SoRTXRenderBackend::createPathTracingBuffers(uint32_t width, uint32_t height)
{
  if (this->accumBuffer != VK_NULL_HANDLE &&
      this->ptBufferWidth == width && this->ptBufferHeight == height) {
    return true;
  }
  // Release the old buffers (deferred: the previous frame's submission may
  // still be executing); new ones are sized to the current viewport.
  if (this->accumBuffer != VK_NULL_HANDLE) {
    VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkBuffer accum = this->accumBuffer;
    const VkDeviceMemory accumMem = this->accumMemory;
    const VkBuffer normal = this->normalBuffer;
    const VkDeviceMemory normalMem = this->normalMemory;
    const VkBuffer position = this->positionBuffer;
    const VkDeviceMemory positionMem = this->positionMemory;
    const VkBuffer sumSq = this->sumSqBuffer;
    const VkDeviceMemory sumSqMem = this->sumSqMemory;
    const VkBuffer counter = this->activeCounterBuffer;
    const VkDeviceMemory counterMem = this->activeCounterMemory;
    this->deferDestroy([device, allocator, accum, accumMem, normal,
                        normalMem, position, positionMem, sumSq, sumSqMem,
                        counter, counterMem]() {
      vkDestroyBuffer(device, accum, allocator);
      vkFreeMemory(device, accumMem, allocator);
      vkDestroyBuffer(device, normal, allocator);
      vkFreeMemory(device, normalMem, allocator);
      vkDestroyBuffer(device, position, allocator);
      vkFreeMemory(device, positionMem, allocator);
      vkDestroyBuffer(device, sumSq, allocator);
      vkFreeMemory(device, sumSqMem, allocator);
      vkDestroyBuffer(device, counter, allocator);
      vkFreeMemory(device, counterMem, allocator);
    });
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = VK_NULL_HANDLE;
    this->positionBuffer = VK_NULL_HANDLE;
    this->positionMemory = VK_NULL_HANDLE;
    this->sumSqBuffer = VK_NULL_HANDLE;
    this->sumSqMemory = VK_NULL_HANDLE;
    this->activeCounterBuffer = VK_NULL_HANDLE;
    this->activeCounterMemory = VK_NULL_HANDLE;
    this->activeCounterMapped = nullptr;
  }
  this->ptBufferWidth = width;
  this->ptBufferHeight = height;
  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
  // The accumulation buffer doubles as a vkCmdFillBuffer target (fresh
  // progressive runs), so it also carries TRANSFER_DST.
  const VkBufferUsageFlags accumUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (!this->createDeviceLocalBuffer(bytes, accumUsage, this->accumBuffer,
                                     this->accumMemory)) {
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  if (!this->createDeviceLocalBuffer(bytes, usage, this->normalBuffer,
                                     this->normalMemory)) {
    // Unwind the partial success so a retry starts clean (and the handles
    // do not survive a subsequent early-out with inconsistent widths).
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  if (!this->createDeviceLocalBuffer(bytes, usage, this->positionBuffer,
                                     this->positionMemory)) {
    vkDestroyBuffer(this->device, this->normalBuffer, this->allocator);
    vkFreeMemory(this->device, this->normalMemory, this->allocator);
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = VK_NULL_HANDLE;
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  // Sums-of-squares (cleared via vkCmdFillBuffer like the accumulation
  // buffer) and the host-readable active-pixel counter (4 bytes; 16 keeps
  // the buffer comfortably above any minimum-alignment requirement).
  if (!this->createDeviceLocalBuffer(bytes, accumUsage, this->sumSqBuffer,
                                     this->sumSqMemory)) {
    vkDestroyBuffer(this->device, this->positionBuffer, this->allocator);
    vkFreeMemory(this->device, this->positionMemory, this->allocator);
    vkDestroyBuffer(this->device, this->normalBuffer, this->allocator);
    vkFreeMemory(this->device, this->normalMemory, this->allocator);
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->positionBuffer = VK_NULL_HANDLE;
    this->positionMemory = VK_NULL_HANDLE;
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = VK_NULL_HANDLE;
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  if (!this->createHostVisibleBuffer(
        16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        this->activeCounterBuffer, this->activeCounterMemory)) {
    vkDestroyBuffer(this->device, this->sumSqBuffer, this->allocator);
    vkFreeMemory(this->device, this->sumSqMemory, this->allocator);
    vkDestroyBuffer(this->device, this->positionBuffer, this->allocator);
    vkFreeMemory(this->device, this->positionMemory, this->allocator);
    vkDestroyBuffer(this->device, this->normalBuffer, this->allocator);
    vkFreeMemory(this->device, this->normalMemory, this->allocator);
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->sumSqBuffer = VK_NULL_HANDLE;
    this->sumSqMemory = VK_NULL_HANDLE;
    this->positionBuffer = VK_NULL_HANDLE;
    this->positionMemory = VK_NULL_HANDLE;
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = VK_NULL_HANDLE;
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  if (vkMapMemory(this->device, this->activeCounterMemory, 0,
                  VK_WHOLE_SIZE, 0, &this->activeCounterMapped) !=
      VK_SUCCESS) {
    this->activeCounterMapped = nullptr;
  }
  // Fresh buffers: refresh the descriptor sets so the new handles are
  // visible to the trace and present passes.
  return this->updateDescriptors();
}

bool
SoRTXRenderBackend::createFrameBuffer()
{
  if (this->frameBuffer != VK_NULL_HANDLE) return true;
  if (!this->createHostVisibleBuffer(
        sizeof(RTXFrameBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        this->frameBuffer, this->frameMemory)) {
    return false;
  }
  return vkMapMemory(this->device, this->frameMemory, 0,
                     sizeof(RTXFrameBlock), 0, &this->frameMapped) ==
    VK_SUCCESS;
}

bool
SoRTXRenderBackend::updateDescriptors()
{
  // Allocate the double-buffered pairs once (the layouts differ, so two
  // allocations of two sets each).
  for (int pair = 0; pair < 2; ++pair) {
    if (this->rtDescriptorSets[pair] != VK_NULL_HANDLE) continue;
    VkDescriptorSetLayout layout = this->rtSetLayout;
    VkDescriptorSetAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = this->descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(this->device, &ai,
                                 &this->rtDescriptorSets[pair]) !=
        VK_SUCCESS) {
      return false;
    }
  }
  for (int pair = 0; pair < 2; ++pair) {
    if (this->presentDescriptorSets[pair] != VK_NULL_HANDLE) continue;
    VkDescriptorSetLayout layout = this->presentSetLayout;
    VkDescriptorSetAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = this->descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(this->device, &ai,
                                 &this->presentDescriptorSets[pair]) !=
        VK_SUCCESS) {
      return false;
    }
  }
  const VkDescriptorSet rtSet = this->rtDescriptorSets[this->descriptorSetIndex];
  const VkDescriptorSet presentSet =
    this->presentDescriptorSets[this->descriptorSetIndex];

  VkDescriptorBufferInfo frameInfo {};
  frameInfo.buffer = this->frameBuffer;
  frameInfo.offset = 0;
  frameInfo.range = sizeof(RTXFrameBlock);

  VkDescriptorImageInfo storageInfo {};
  storageInfo.imageView = this->storageImageView;
  storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkDescriptorImageInfo presentInfo {};
  presentInfo.sampler = this->presentSampler;
  presentInfo.imageView = this->storageImageView;
  // The image stays in GENERAL layout for both the trace (storage) and
  // present (sampled) accesses; no in-render-pass transitions needed.
  presentInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkDescriptorBufferInfo materialInfo {};
  materialInfo.buffer = this->materialBuffer;
  materialInfo.offset = 0;
  materialInfo.range = VK_WHOLE_SIZE;

  // Path tracing buffers: accumulation (set 0, binding 4), world normal
  // G-buffer (binding 5) and world position/hit-distance G-buffer
  // (binding 6).  Written by the raygen shader.
  VkDescriptorBufferInfo accumInfo {};
  accumInfo.buffer = this->accumBuffer;
  accumInfo.offset = 0;
  accumInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo normalInfo {};
  normalInfo.buffer = this->normalBuffer;
  normalInfo.offset = 0;
  normalInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo positionInfo {};
  positionInfo.buffer = this->positionBuffer;
  positionInfo.offset = 0;
  positionInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo normalPoolInfo {};
  normalPoolInfo.buffer = this->normalPoolBuffer;
  normalPoolInfo.offset = 0;
  normalPoolInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo sumSqInfo {};
  sumSqInfo.buffer = this->sumSqBuffer;
  sumSqInfo.offset = 0;
  sumSqInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo counterInfo {};
  counterInfo.buffer = this->activeCounterBuffer;
  counterInfo.offset = 0;
  counterInfo.range = VK_WHOLE_SIZE;

  // Binding 0: the acceleration structure (TLAS) read by the raygen shader.
  // Only written once the TLAS exists; updateDescriptors() is re-invoked by
  // buildTlas() right after (re)creation so a null handle is never written
  // and the trace phase always observes a valid descriptor.
  VkWriteDescriptorSetAccelerationStructureKHR asWrite {};
  asWrite.sType =
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  asWrite.accelerationStructureCount = 1;
  VkAccelerationStructureKHR asHandle = this->tlas;
  asWrite.pAccelerationStructures = &asHandle;

  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(5);

  if (this->tlas != VK_NULL_HANDLE) {
    VkWriteDescriptorSet asBinding {};
    asBinding.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    asBinding.dstSet = rtSet;
    asBinding.dstBinding = 0;
    asBinding.descriptorCount = 1;
    asBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    asBinding.pNext = &asWrite;
    writes.push_back(asBinding);
  }

  VkWriteDescriptorSet storageWrite {};
  storageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  storageWrite.dstSet = rtSet;
  storageWrite.dstBinding = 1;
  storageWrite.descriptorCount = 1;
  storageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  storageWrite.pImageInfo = &storageInfo;
  if (this->storageImageView != VK_NULL_HANDLE) {
    writes.push_back(storageWrite);
  }

  VkWriteDescriptorSet frameWrite {};
  frameWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  frameWrite.dstSet = rtSet;
  frameWrite.dstBinding = 2;
  frameWrite.descriptorCount = 1;
  frameWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  frameWrite.pBufferInfo = &frameInfo;
  writes.push_back(frameWrite);

  if (this->materialBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet materialWrite {};
    materialWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    materialWrite.dstSet = rtSet;
    materialWrite.dstBinding = 3;
    materialWrite.descriptorCount = 1;
    materialWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    materialWrite.pBufferInfo = &materialInfo;
    writes.push_back(materialWrite);
  }

  if (this->accumBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet accumWrite {};
    accumWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    accumWrite.dstSet = rtSet;
    accumWrite.dstBinding = 4;
    accumWrite.descriptorCount = 1;
    accumWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    accumWrite.pBufferInfo = &accumInfo;
    writes.push_back(accumWrite);
  }
  if (this->normalBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet normalWrite {};
    normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    normalWrite.dstSet = rtSet;
    normalWrite.dstBinding = 5;
    normalWrite.descriptorCount = 1;
    normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    normalWrite.pBufferInfo = &normalInfo;
    writes.push_back(normalWrite);
  }
  if (this->positionBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet positionWrite {};
    positionWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    positionWrite.dstSet = rtSet;
    positionWrite.dstBinding = 6;
    positionWrite.descriptorCount = 1;
    positionWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    positionWrite.pBufferInfo = &positionInfo;
    writes.push_back(positionWrite);
  }
  if (this->normalPoolBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet poolWrite {};
    poolWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    poolWrite.dstSet = rtSet;
    poolWrite.dstBinding = 7;
    poolWrite.descriptorCount = 1;
    poolWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolWrite.pBufferInfo = &normalPoolInfo;
    writes.push_back(poolWrite);
  }
  if (this->sumSqBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet sumSqWrite {};
    sumSqWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    sumSqWrite.dstSet = rtSet;
    sumSqWrite.dstBinding = 8;
    sumSqWrite.descriptorCount = 1;
    sumSqWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sumSqWrite.pBufferInfo = &sumSqInfo;
    writes.push_back(sumSqWrite);
  }
  if (this->activeCounterBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet counterWrite {};
    counterWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    counterWrite.dstSet = rtSet;
    counterWrite.dstBinding = 9;
    counterWrite.descriptorCount = 1;
    counterWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    counterWrite.pBufferInfo = &counterInfo;
    writes.push_back(counterWrite);
  }

  VkWriteDescriptorSet presentWrite {};
  presentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  presentWrite.dstSet = presentSet;
  presentWrite.dstBinding = 1;
  presentWrite.descriptorCount = 1;
  presentWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  presentWrite.pImageInfo = &presentInfo;
  if (this->storageImageView != VK_NULL_HANDLE &&
      this->presentSampler != VK_NULL_HANDLE) {
    writes.push_back(presentWrite);
  }

  if (this->accumBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentAccumWrite {};
    presentAccumWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentAccumWrite.dstSet = presentSet;
    presentAccumWrite.dstBinding = 2;
    presentAccumWrite.descriptorCount = 1;
    presentAccumWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentAccumWrite.pBufferInfo = &accumInfo;
    writes.push_back(presentAccumWrite);
  }
  if (this->normalBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentNormalWrite {};
    presentNormalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentNormalWrite.dstSet = presentSet;
    presentNormalWrite.dstBinding = 3;
    presentNormalWrite.descriptorCount = 1;
    presentNormalWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentNormalWrite.pBufferInfo = &normalInfo;
    writes.push_back(presentNormalWrite);
  }
  if (this->positionBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentPositionWrite {};
    presentPositionWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentPositionWrite.dstSet = presentSet;
    presentPositionWrite.dstBinding = 4;
    presentPositionWrite.descriptorCount = 1;
    presentPositionWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentPositionWrite.pBufferInfo = &positionInfo;
    writes.push_back(presentPositionWrite);
  }

  vkUpdateDescriptorSets(this->device,
                         static_cast<uint32_t>(writes.size()), writes.data(),
                         0, nullptr);
  return true;
}

bool
SoRTXRenderBackend::createPipelines()
{
  VkPipelineLayoutCreateInfo layoutCI {};
  layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutCI.setLayoutCount = 1;
  layoutCI.pSetLayouts = &this->rtSetLayout;
  // The raygen receives its per-frame state (frame index, PT flags, bounce
  // budget) through a 16-byte push constant block.
  VkPushConstantRange raygenPush {};
  raygenPush.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  raygenPush.offset = 0;
  raygenPush.size = sizeof(RTXRaygenPush);
  layoutCI.pPushConstantRanges = &raygenPush;
  layoutCI.pushConstantRangeCount = 1;
  if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                             &this->rtPipelineLayout) != VK_SUCCESS) {
    return false;
  }

  layoutCI.pSetLayouts = &this->presentSetLayout;
  // The present shader receives width/height/denoiseOn/frameIndex via a
  // fragment push constant (the present pass must run inside the caller's
  // render pass, so a compute denoise pass cannot be dispatched there; the
  // edge-stopping filter lives in PresentFragment.glsl instead).
  VkPushConstantRange presentPush {};
  presentPush.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  presentPush.offset = 0;
  presentPush.size = 8 * sizeof(float);
  layoutCI.pPushConstantRanges = &presentPush;
  layoutCI.pushConstantRangeCount = 1;
  if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                             &this->presentPipelineLayout) != VK_SUCCESS) {
    return false;
  }

  // --- Ray tracing pipeline (five SBT groups) ----------------------------
  // Group layout: 0 = raygen, 1 = miss, 2 = shadow miss, 3 = closest hit,
  // 4 = shadow closest hit.  Primary rays use missIndex 0 and hit-group
  // record 0; shadow rays use missIndex 1 and hit-group record 1.
  VkPipelineShaderStageCreateInfo stages[SBT_GROUP_COUNT] {};
  const auto stage = [](VkShaderStageFlagBits flag, VkShaderModule module,
                        VkPipelineShaderStageCreateInfo & out) {
    out.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    out.stage = flag;
    out.module = module;
    out.pName = "main";
  };
  stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, this->raygenModule, stages[0]);
  stage(VK_SHADER_STAGE_MISS_BIT_KHR, this->missModule, stages[1]);
  stage(VK_SHADER_STAGE_MISS_BIT_KHR, this->shadowMissModule, stages[2]);
  stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, this->closestHitModule,
        stages[3]);
  stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, this->shadowClosestHitModule,
        stages[4]);

  VkRayTracingShaderGroupCreateInfoKHR groups[SBT_GROUP_COUNT] {};
  for (int i = 0; i < SBT_GROUP_COUNT; ++i) {
    groups[i].sType =
      VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[i].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[i].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
    if (i <= 2) {
      groups[i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      groups[i].generalShader = static_cast<uint32_t>(i);
    }
    else {
      groups[i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      groups[i].closestHitShader = static_cast<uint32_t>(i);
    }
  }

  VkRayTracingPipelineCreateInfoKHR ci {};
  ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  ci.stageCount = SBT_GROUP_COUNT;
  ci.pStages = stages;
  ci.groupCount = SBT_GROUP_COUNT;
  ci.pGroups = groups;
  ci.maxPipelineRayRecursionDepth = 2; // primary + one shadow level
  ci.layout = this->rtPipelineLayout;
  if (this->vkCreateRayTracingPipelinesKHR(
        this->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci,
        this->allocator, &this->rtPipeline) != VK_SUCCESS) {
    return false;
  }
  if (!this->createShaderBindingTable()) {
    return false;
  }

  // Ray-query compute pipeline (default dispatch mode): the same path
  // tracer compiled as a compute shader, driven by vkCmdDispatch.
  VkComputePipelineCreateInfo computeCI {};
  computeCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computeCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  computeCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  computeCI.stage.module = this->pathTraceModule;
  computeCI.stage.pName = "main";
  computeCI.layout = this->rtPipelineLayout;
  return vkCreateComputePipelines(this->device, VK_NULL_HANDLE, 1, &computeCI,
                                  this->allocator,
                                  &this->computePipeline) == VK_SUCCESS;
}

bool
SoRTXRenderBackend::createShaderBindingTable()
{
  // Five records: raygen, miss, shadow miss, closest hit, shadow closest
  // hit, each aligned to the driver's shader-group-handle alignment.  The
  // table is host-visible so the group handles can be copied in directly.
  // Extra base-alignment slack keeps the strided region device addresses
  // aligned to shaderGroupBaseAlignment (VUID-vkCmdTraceRaysKHR-03675).
  const VkDeviceSize baseAlignment = this->sbtGroupBaseAlignment;
  const VkDeviceSize tableSize =
    static_cast<VkDeviceSize>(this->sbtRecordSize) * SBT_GROUP_COUNT +
    baseAlignment;
  if (!this->createHostVisibleBuffer(
        tableSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        this->sbtBuffer, this->sbtMemory)) {
    return false;
  }

  // Fetch the group handles (one per pipeline group) and copy each into
  // its aligned record slot.
  const uint32_t handleSize = this->sbtGroupHandleSize;
  std::vector<uint8_t> handles(static_cast<size_t>(handleSize) *
                               SBT_GROUP_COUNT);
  if (this->vkGetRayTracingShaderGroupHandlesKHR(
        this->device, this->rtPipeline, 0, SBT_GROUP_COUNT,
        handles.size(), handles.data()) != VK_SUCCESS) {
    return false;
  }
  void * mapped = nullptr;
  if (vkMapMemory(this->device, this->sbtMemory, 0, tableSize, 0,
                  &mapped) != VK_SUCCESS) {
    return false;
  }
  const VkDeviceAddress rawBase = this->getDeviceAddress(this->sbtBuffer);
  const VkDeviceAddress alignedBase =
    (rawBase + baseAlignment - 1) / baseAlignment * baseAlignment;
  this->sbtBaseOffset = alignedBase - rawBase;
  for (int i = 0; i < SBT_GROUP_COUNT; ++i) {
    std::memcpy(static_cast<uint8_t *>(mapped) + this->sbtBaseOffset +
                  static_cast<size_t>(i) * this->sbtRecordSize,
                handles.data() + static_cast<size_t>(i) * handleSize,
                handleSize);
  }
  vkUnmapMemory(this->device, this->sbtMemory);

  // Strided device-address regions handed to vkCmdTraceRaysKHR.
  const VkDeviceSize stride = this->sbtRecordSize;
  this->raygenSbtRegion = {alignedBase + 0 * stride, stride, stride};
  this->missSbtRegion = {alignedBase + 1 * stride, stride, 2 * stride};
  this->hitSbtRegion = {alignedBase + 3 * stride, stride, 2 * stride};
  this->callableSbtRegion = {0, 0, 0};
  return true;
}

bool
SoRTXRenderBackend::createPresentPipeline(VkRenderPass renderPass,
                                           VkSampleCountFlagBits sampleCount)
{
  // The present pass renders into the swapchain/MSAA color attachment, so
  // the pipeline's rasterization sample count must match the render pass
  // (VUID-VkGraphicsPipelineCreateInfo-renderPass-06082).  Key the cache on
  // both the render pass and the sample count.
  if (this->presentPipeline != VK_NULL_HANDLE &&
      this->presentRenderPass == renderPass &&
      this->presentSampleCount == sampleCount) {
    return true;
  }
  if (this->presentPipeline != VK_NULL_HANDLE) {
    // Defer: a pending frame may still bind this pipeline.
    VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkPipeline pipeline = this->presentPipeline;
    this->deferDestroy([device, allocator, pipeline]() {
      vkDestroyPipeline(device, pipeline, allocator);
    });
    this->presentPipeline = VK_NULL_HANDLE;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = this->presentVertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = this->presentFragmentModule;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization.cullMode = VK_CULL_MODE_NONE;
  rasterization.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = sampleCount;
  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_FALSE;
  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
    VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
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
  ci.layout = this->presentPipelineLayout;
  ci.renderPass = renderPass;
  ci.subpass = 0;
  if (vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &ci,
                                this->allocator,
                                &this->presentPipeline) != VK_SUCCESS) {
    return false;
  }
  this->presentRenderPass = renderPass;
  this->presentSampleCount = sampleCount;
  return true;
}

// --- Geometry cache -------------------------------------------------------

RTXCachedGeometry &
SoRTXRenderBackend::getOrCreateCache(const SoRenderCommand * command)
{
  const auto found = this->commandToCache.find(command);
  if (found != this->commandToCache.end()) {
    return this->geometryCache[found->second];
  }
  const size_t index = this->geometryCache.size();
  this->geometryCache.emplace_back();
  this->geometryCache.back().commandKey = command;
  this->commandToCache[command] = index;
  return this->geometryCache.back();
}

void
SoRTXRenderBackend::destroyCacheEntry(RTXCachedGeometry & entry)
{
  if (entry.blas != VK_NULL_HANDLE) {
    vkDestroyAccelerationStructureKHR(this->device, entry.blas,
                                      this->allocator);
    entry.blas = VK_NULL_HANDLE;
  }
  if (entry.blasBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, entry.blasBuffer, this->allocator);
    entry.blasBuffer = VK_NULL_HANDLE;
  }
  if (entry.blasMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.blasMemory, this->allocator);
    entry.blasMemory = VK_NULL_HANDLE;
  }
  if (entry.vertexBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, entry.vertexBuffer, this->allocator);
    entry.vertexBuffer = VK_NULL_HANDLE;
  }
  if (entry.vertexMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.vertexMemory, this->allocator);
    entry.vertexMemory = VK_NULL_HANDLE;
  }
  if (entry.indexBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, entry.indexBuffer, this->allocator);
    entry.indexBuffer = VK_NULL_HANDLE;
  }
  if (entry.indexMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.indexMemory, this->allocator);
    entry.indexMemory = VK_NULL_HANDLE;
  }
  entry = RTXCachedGeometry();
}

void
SoRTXRenderBackend::deferDestroyCacheEntry(RTXCachedGeometry & entry)
{
  if (entry.blas == VK_NULL_HANDLE && entry.vertexBuffer == VK_NULL_HANDLE &&
      entry.indexBuffer == VK_NULL_HANDLE) {
    entry = RTXCachedGeometry();
    return;
  }
  VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  const PFN_vkDestroyAccelerationStructureKHR vkDestroyAS =
    this->vkDestroyAccelerationStructureKHR;
  const VkAccelerationStructureKHR blas = entry.blas;
  const VkBuffer blasBuffer = entry.blasBuffer;
  const VkDeviceMemory blasMemory = entry.blasMemory;
  const VkBuffer vertexBuffer = entry.vertexBuffer;
  const VkDeviceMemory vertexMemory = entry.vertexMemory;
  const VkBuffer indexBuffer = entry.indexBuffer;
  const VkDeviceMemory indexMemory = entry.indexMemory;
  this->deferDestroy([device, allocator, vkDestroyAS, blas, blasBuffer,
                      blasMemory, vertexBuffer, vertexMemory, indexBuffer,
                      indexMemory]() {
    if (blas != VK_NULL_HANDLE) {
      vkDestroyAS(device, blas, allocator);
    }
    if (blasBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, blasBuffer, allocator);
    }
    if (blasMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, blasMemory, allocator);
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
  entry = RTXCachedGeometry();
}

void
SoRTXRenderBackend::freePendingStagingDestroys()
{
  for (const auto & entry : this->pendingStagingDestroys) {
    if (entry.first != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, entry.first, this->allocator);
    }
    if (entry.second != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, entry.second, this->allocator);
    }
  }
  this->pendingStagingDestroys.clear();
}

void
SoRTXRenderBackend::flushPendingDestroys()
{
  const int batch = this->pendingDestroyIndex;
  this->pendingDestroyIndex = batch ^ 1;
  for (const auto & fn : this->pendingDestroys[batch]) {
    if (fn) fn();
  }
  this->pendingDestroys[batch].clear();
}

void
SoRTXRenderBackend::deferDestroy(std::function<void()> && fn)
{
  this->pendingDestroys[this->pendingDestroyIndex].push_back(std::move(fn));
}

void
SoRTXRenderBackend::invalidateCache()
{
  for (RTXCachedGeometry & entry : this->geometryCache) {
    this->destroyCacheEntry(entry);
  }
  this->geometryCache.clear();
  this->commandToCache.clear();
}

void
SoRTXRenderBackend::updateGeometryCache(const SoDrawList & drawlist)
{
  // The draw-list generation is a per-frame production counter, not a
  // scene-change signal, and the producer-owned geometry storage is a
  // per-frame arena whose pointers change every frame.  Cache invalidation
  // is therefore driven by the sampled content hash of each command's
  // vertex/index data: entries whose content is unchanged keep their BLAS;
  // only genuinely new geometry triggers a rebuild.
  this->cacheChanged = false;
  const uint32_t frame = ++this->cacheFrame;

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_TRANSPARENT ||
        command.pass == SO_RENDERPASS_OVERLAY) continue;
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) continue;
    if (geometry.topology != SO_TOPOLOGY_TRIANGLES) continue;

    const uint32_t vertexStride = geometry.vertexStride
      ? geometry.vertexStride : sizeof(float) * 3;
    const bool indexed = geometry.indexCount > 0 && geometry.indices != nullptr;

    // Cheap probe first: only run the full index hash when the change signal
    // disagrees with the cache entry.  The sampled signal costs a fraction of
    // the full hashGeometry() (which hashes every index up to 65536), so for
    // unchanged large CAD parts we reuse contentHash instead of re-walking
    // the whole index buffer every frame.
    const uint64_t signal = hashGeometrySignal(geometry, vertexStride, indexed);
    uint64_t hash = 0;
    RTXCachedGeometry * entryPtr = nullptr;

    const auto found = this->commandToCache.find(&command);
    if (found != this->commandToCache.end()) {
      RTXCachedGeometry & entry = this->geometryCache[found->second];
      hash = entry.contentHash;
      bool matches = entry.blas != VK_NULL_HANDLE &&
        entry.changeSignal == signal && entry.contentHash != 0;
      if (!matches) {
        hash = hashGeometry(geometry, vertexStride, indexed);
        matches = entry.blas != VK_NULL_HANDLE && entry.contentHash == hash;
      }
      if (!matches) {
        // Split the identity: position-only changes (same topology) refit
        // the existing BLAS in place; index/topology changes destroy and
        // rebuild.
        const uint64_t vertexHash = hashPositions(geometry, vertexStride);
        const uint64_t indexHash = indexed ? hashIndices(geometry) : 0;
        const bool topologyStable =
          entry.blas != VK_NULL_HANDLE &&
          entry.vertexCount == geometry.vertexCount &&
          entry.indexCount == geometry.indexCount &&
          entry.vertexStride == vertexStride &&
          ((entry.idxKey != nullptr) == indexed) &&
          entry.indexHash == indexHash;
        this->cacheChanged = true;
        if (topologyStable) {
          // In-place UPDATE build (see refitBlas()); keep buffers and BLAS.
          entry.refitPending = true;
          entry.posKey = geometry.positions;
          entry.idxKey = geometry.indices;
        }
        else {
          // Buffers/AS are rebuilt in recordAccelerationStructures (they
          // need a command buffer); only release old resources and record
          // the new identity here.  Destruction is deferred: a pending
          // frame may still reference the old BLAS.
          this->deferDestroyCacheEntry(entry);
          entry.posKey = geometry.positions;
          entry.idxKey = geometry.indices;
          entry.vertexCount = geometry.vertexCount;
          entry.indexCount = geometry.indexCount;
          entry.vertexStride = vertexStride;
        }
        entry.contentHash = hash;
        entry.changeSignal = signal;
        entry.vertexHash = vertexHash;
        entry.indexHash = indexHash;
      }
      entryPtr = &entry;
    }
    else {
      // The command pointer changed (draw-list storage reallocation or
      // reordering when objects are added/removed): instead of thrashing a
      // full BLAS rebuild, re-key the unclaimed entry whose content is
      // identical so the acceleration structure survives the pointer churn.
      hash = hashGeometry(geometry, vertexStride, indexed);
      RTXCachedGeometry * match = nullptr;
      for (RTXCachedGeometry & e : this->geometryCache) {
        if (e.cacheGeneration == frame) continue;
        if (e.blas != VK_NULL_HANDLE && e.contentHash == hash &&
            e.vertexCount == geometry.vertexCount &&
            e.indexCount == geometry.indexCount &&
            e.vertexStride == vertexStride &&
            ((e.idxKey != nullptr) == indexed)) {
          match = &e;
          break;
        }
      }
      if (match) {
        match->changeSignal = signal;
        this->commandToCache[&command] =
          static_cast<size_t>(match - this->geometryCache.data());
        entryPtr = match;
      }
      else {
        this->cacheChanged = true;
        RTXCachedGeometry & entry = this->getOrCreateCache(&command);
        entry.posKey = geometry.positions;
        entry.idxKey = geometry.indices;
        entry.vertexCount = geometry.vertexCount;
        entry.indexCount = geometry.indexCount;
        entry.vertexStride = vertexStride;
        entry.contentHash = hash;
        entry.changeSignal = signal;
        entry.vertexHash = hashPositions(geometry, vertexStride);
        entry.indexHash = indexed ? hashIndices(geometry) : 0;
        entryPtr = &entry;
      }
    }
    entryPtr->commandKey = &command;
    entryPtr->cacheGeneration = frame;
  }

  // Evict entries whose command disappeared from the draw list this frame
  // (their generation stamp is stale).  Command pointers live in a
  // per-frame arena, so the stamp -- not pointer identity -- decides
  // liveness; survivors rebuild the pointer map from their stored
  // commandKey.
  bool anyStale = false;
  for (RTXCachedGeometry & entry : this->geometryCache) {
    if (entry.cacheGeneration != frame) {
      this->deferDestroyCacheEntry(entry);
      anyStale = true;
    }
  }
  if (anyStale) {
    size_t write = 0;
    for (size_t idx = 0; idx < this->geometryCache.size(); ++idx) {
      if (this->geometryCache[idx].cacheGeneration == frame) {
        if (write != idx) {
          this->geometryCache[write] = std::move(this->geometryCache[idx]);
        }
        ++write;
      }
    }
    this->geometryCache.resize(write);
    this->commandToCache.clear();
    for (size_t idx = 0; idx < this->geometryCache.size(); ++idx) {
      this->commandToCache[this->geometryCache[idx].commandKey] = idx;
    }
  }
}

bool
SoRTXRenderBackend::buildBlas(RTXCachedGeometry & entry,
                              const SoRenderCommand & command,
                              VkCommandBuffer cmd)
{
  const SoGeometryDesc & geometry = command.geometry;
  const bool indexed = entry.indexCount > 0 && entry.idxKey != nullptr;
  const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);

  if (getenv("FC_VULKAN_RT_DEBUG")) {
    static uint32_t blasSeq = 0;
    fprintf(stderr,
            "[RTDBG] buildBlas #%u verts=%u idx=%u stride=%u indexed=%d "
            "pos=%p idxPtr=%p\n",
            blasSeq++, entry.vertexCount, entry.indexCount, entry.vertexStride,
            indexed ? 1 : 0, static_cast<const void *>(geometry.positions),
            static_cast<const void *>(geometry.indices));
  }

  // The path tracing compute shader shades flat faces from the object-space
  // triangle-normal pool; append this command's normals (the material
  // records pick up the offset afterwards in updateMaterials()).
  this->appendTriangleNormals(command, entry);

  // Position-only vertex buffer (tightly packed vec3) for the BLAS.
  std::vector<float> positions(static_cast<size_t>(entry.vertexCount) * 3);
  for (uint32_t i = 0; i < entry.vertexCount; ++i) {
    const float * p =
      geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    positions[static_cast<size_t>(i) * 3 + 0] = p[0];
    positions[static_cast<size_t>(i) * 3 + 1] = p[1];
    positions[static_cast<size_t>(i) * 3 + 2] = p[2];
  }
  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(entry.vertexCount) * 3 * sizeof(float);
  if (!this->createDeviceLocalBuffer(
        vertexBytes,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        entry.vertexBuffer, entry.vertexMemory)) {
    return false;
  }

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createHostVisibleBuffer(vertexBytes,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     staging, stagingMemory)) {
    return false;
  }
  void * mapped = nullptr;
  if (vkMapMemory(this->device, stagingMemory, 0, vertexBytes, 0, &mapped) !=
        VK_SUCCESS ||
      mapped == nullptr) {
    this->emitError("buildBlas: vkMapMemory (vertex staging) failed");
    vkDestroyBuffer(this->device, staging, this->allocator);
    vkFreeMemory(this->device, stagingMemory, this->allocator);
    return false;
  }
  std::memcpy(mapped, positions.data(), static_cast<size_t>(vertexBytes));
  vkUnmapMemory(this->device, stagingMemory);

  VkBuffer indexStaging = VK_NULL_HANDLE;
  VkDeviceMemory indexStagingMemory = VK_NULL_HANDLE;
  VkDeviceSize indexBytes = 0;
  if (indexed) {
    indexBytes =
      static_cast<VkDeviceSize>(entry.indexCount) * sizeof(uint32_t);
    if (!this->createDeviceLocalBuffer(
          indexBytes,
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          entry.indexBuffer, entry.indexMemory)) {
      vkDestroyBuffer(this->device, staging, this->allocator);
      vkFreeMemory(this->device, stagingMemory, this->allocator);
      return false;
    }
    if (!this->createHostVisibleBuffer(indexBytes,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       indexStaging, indexStagingMemory)) {
      vkDestroyBuffer(this->device, staging, this->allocator);
      vkFreeMemory(this->device, stagingMemory, this->allocator);
      return false;
    }
    void * imapped = nullptr;
    if (vkMapMemory(this->device, indexStagingMemory, 0, indexBytes, 0,
                    &imapped) != VK_SUCCESS ||
        imapped == nullptr) {
      this->emitError("buildBlas: vkMapMemory (index staging) failed");
      vkDestroyBuffer(this->device, staging, this->allocator);
      vkFreeMemory(this->device, stagingMemory, this->allocator);
      vkDestroyBuffer(this->device, indexStaging, this->allocator);
      vkFreeMemory(this->device, indexStagingMemory, this->allocator);
      return false;
    }
    std::memcpy(imapped, geometry.indices, static_cast<size_t>(indexBytes));
    vkUnmapMemory(this->device, indexStagingMemory);
  }

  VkBufferCopy vertexCopy {};
  vertexCopy.size = vertexBytes;
  vkCmdCopyBuffer(cmd, staging, entry.vertexBuffer, 1, &vertexCopy);
  if (indexed) {
    VkBufferCopy indexCopy {};
    indexCopy.size = indexBytes;
    vkCmdCopyBuffer(cmd, indexStaging, entry.indexBuffer, 1, &indexCopy);
  }
  VkMemoryBarrier copyBarrier {};
  copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  copyBarrier.dstAccessMask =
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &copyBarrier, 0, nullptr, 0, nullptr);

  // The staging buffers are referenced by the copy commands recorded above;
  // destroying them now would invalidate this command buffer.  Defer the
  // destruction until the submission completed (freePendingStagingDestroys).
  this->pendingStagingDestroys.emplace_back(staging, stagingMemory);
  if (indexStaging != VK_NULL_HANDLE) {
    this->pendingStagingDestroys.emplace_back(indexStaging, indexStagingMemory);
  }

  // --- Build the BLAS ----------------------------------------------------
  VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
  triangles.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress =
    this->getDeviceAddress(entry.vertexBuffer);
  triangles.vertexStride = 3 * sizeof(float);
  triangles.maxVertex = entry.vertexCount - 1;
  triangles.indexType =
    indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
  triangles.indexData.deviceAddress =
    indexed ? this->getDeviceAddress(entry.indexBuffer) : 0;

  VkAccelerationStructureGeometryKHR asGeometry {};
  asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeometry.geometry.triangles = triangles;
  asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  const uint32_t maxPrimitives =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (maxPrimitives == 0) return false;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
  buildInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  // ALLOW_UPDATE: lets position-only edits refit this BLAS in place (see
  // refitBlas()) instead of destroying and rebuilding it.
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &asGeometry;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
  sizeInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
    this->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo, &maxPrimitives, &sizeInfo);
  if (!this->createScratchBuffer(sizeInfo.buildScratchSize)) {
    return false;
  }

  if (!this->createDeviceLocalBuffer(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        entry.blasBuffer, entry.blasMemory)) {
    return false;
  }
  VkAccelerationStructureCreateInfoKHR asCI {};
  asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  asCI.buffer = entry.blasBuffer;
  asCI.size = sizeInfo.accelerationStructureSize;
  asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  if (vkCreateAccelerationStructureKHR(this->device, &asCI, this->allocator,
                                       &entry.blas) != VK_SUCCESS) {
    return false;
  }
  // Capture the BLAS device address now.  It is constant for the lifetime of
  // the BLAS, so the per-frame instance collection in buildTlas() reuses it
  // instead of calling vkGetAccelerationStructureDeviceAddressKHR every frame.
  entry.devAddr = 0;
  VkAccelerationStructureDeviceAddressInfoKHR devAddrInfo {};
  devAddrInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  devAddrInfo.accelerationStructure = entry.blas;
  entry.devAddr = vkGetAccelerationStructureDeviceAddressKHR(this->device,
                                                             &devAddrInfo);

  buildInfo.dstAccelerationStructure = entry.blas;
  buildInfo.scratchData.deviceAddress = this->scratchAddress;
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo {};
  rangeInfo.primitiveCount = maxPrimitives;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR * rangeInfos[] = {&rangeInfo};
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);

  VkMemoryBarrier blasBarrier {};
  blasBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  blasBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  blasBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &blasBarrier, 0, nullptr, 0, nullptr);
  return true;
}

bool
SoRTXRenderBackend::refitBlas(RTXCachedGeometry & entry,
                              const SoRenderCommand & command,
                              VkCommandBuffer cmd)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);

  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr,
            "[RTDBG] refitBlas verts=%u idx=%u stride=%u pos=%p\n",
            entry.vertexCount, entry.indexCount, entry.vertexStride,
            static_cast<const void *>(geometry.positions));
  }

  // Upload the new vertex positions into the EXISTING device buffers; the
  // index buffer and topology are unchanged (the refit precondition checked
  // in updateGeometryCache()).
  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(entry.vertexCount) * 3 * sizeof(float);

  // Moved vertices change the object-space flat normals: append a fresh
  // normal-pool record and let updateMaterials() pick up the new offset
  // (the pool is grow-only, matching the rebuild path).
  this->appendTriangleNormals(command, entry);
  std::vector<float> positions(static_cast<size_t>(entry.vertexCount) * 3);
  for (uint32_t i = 0; i < entry.vertexCount; ++i) {
    const float * p =
      geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    positions[static_cast<size_t>(i) * 3 + 0] = p[0];
    positions[static_cast<size_t>(i) * 3 + 1] = p[1];
    positions[static_cast<size_t>(i) * 3 + 2] = p[2];
  }

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createHostVisibleBuffer(vertexBytes,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     staging, stagingMemory)) {
    return false;
  }
  void * mapped = nullptr;
  if (vkMapMemory(this->device, stagingMemory, 0, vertexBytes, 0, &mapped) !=
        VK_SUCCESS ||
      mapped == nullptr) {
    this->emitError("refitBlas: vkMapMemory (vertex staging) failed");
    vkDestroyBuffer(this->device, staging, this->allocator);
    vkFreeMemory(this->device, stagingMemory, this->allocator);
    return false;
  }
  std::memcpy(mapped, positions.data(), static_cast<size_t>(vertexBytes));
  vkUnmapMemory(this->device, stagingMemory);

  VkBufferCopy vertexCopy {};
  vertexCopy.size = vertexBytes;
  vkCmdCopyBuffer(cmd, staging, entry.vertexBuffer, 1, &vertexCopy);
  VkMemoryBarrier copyBarrier {};
  copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  copyBarrier.dstAccessMask =
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &copyBarrier, 0, nullptr, 0, nullptr);
  this->pendingStagingDestroys.emplace_back(staging, stagingMemory);

  // --- In-place UPDATE build ---------------------------------------------
  VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
  triangles.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress =
    this->getDeviceAddress(entry.vertexBuffer);
  triangles.vertexStride = 3 * sizeof(float);
  triangles.maxVertex = entry.vertexCount - 1;
  const bool indexed = entry.indexCount > 0;
  triangles.indexType =
    indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
  triangles.indexData.deviceAddress =
    indexed ? this->getDeviceAddress(entry.indexBuffer) : 0;

  VkAccelerationStructureGeometryKHR asGeometry {};
  asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeometry.geometry.triangles = triangles;
  asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  const uint32_t maxPrimitives =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (maxPrimitives == 0) return false;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
  buildInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
  buildInfo.srcAccelerationStructure = entry.blas;
  buildInfo.dstAccelerationStructure = entry.blas;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &asGeometry;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
  sizeInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
    this->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo, &maxPrimitives, &sizeInfo);
  // createScratchBuffer() is grow-only; the update scratch size is bounded
  // by the build scratch size already allocated.
  if (!this->createScratchBuffer(sizeInfo.buildScratchSize)) {
    return false;
  }
  buildInfo.scratchData.deviceAddress = this->scratchAddress;

  VkAccelerationStructureBuildRangeInfoKHR rangeInfo {};
  rangeInfo.primitiveCount = maxPrimitives;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR * rangeInfos[] = {&rangeInfo};
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);

  VkMemoryBarrier blasBarrier {};
  blasBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  blasBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  blasBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &blasBarrier, 0, nullptr, 0, nullptr);

  entry.refitPending = false;
  return true;
}

bool
SoRTXRenderBackend::buildTlas(const SoDrawList & drawlist, VkCommandBuffer cmd)
{
  // Collect instance data for every cached geometry command (opaque only).
  // instanceCustomIndex is the draw-list command index so the closest-hit
  // shader can index the material buffer with gl_InstanceCustomIndexEXT.
  // Reuse a grow-only scratch vector instead of reallocating each frame.
  std::vector<VkAccelerationStructureInstanceKHR> & instances =
    this->instanceScratch;
  instances.clear();
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_TRANSPARENT ||
        command.pass == SO_RENDERPASS_OVERLAY) continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    const RTXCachedGeometry & entry = this->geometryCache[found->second];
    if (entry.blas == VK_NULL_HANDLE) continue;

    VkAccelerationStructureInstanceKHR instance {};
    // SbMatrix is row-major; VkTransformMatrixKHR is row-major 3x4.
    const SbMatrix & m = command.modelMatrix;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        instance.transform.matrix[r][c] = m[r][c];
      }
    }
    instance.instanceCustomIndex = static_cast<uint32_t>(i);
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = 0;
    // The BLAS device address is stable for the BLAS lifetime and was
    // captured at build time, so querying it here every frame is wasted work.
    // Fall back to a query only if the cached address is missing.
    if (entry.devAddr) {
      instance.accelerationStructureReference = entry.devAddr;
    }
    else {
      VkAccelerationStructureDeviceAddressInfoKHR addrInfo {};
      addrInfo.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addrInfo.accelerationStructure = entry.blas;
      instance.accelerationStructureReference =
        vkGetAccelerationStructureDeviceAddressKHR(this->device, &addrInfo);
    }
    instances.push_back(instance);
  }
  this->instanceCount = static_cast<uint32_t>(instances.size());

  if (getenv("FC_VULKAN_RT_DEBUG")) {
    static uint32_t debugFrame = 0;
    if ((debugFrame++ % 120) == 0) {
      fprintf(stderr,
              "[RTDBG] buildTlas: drawlist commands=%d instances=%zu "
              "cacheEntries=%zu mapEntries=%zu\n",
              drawlist.getNumCommands(), instances.size(),
              this->geometryCache.size(), this->commandToCache.size());
    }
  }

  // Instance buffer (host-visible; rebuilt every frame).  An empty scene
  // still builds a valid empty TLAS so the descriptor references a real
  // acceleration structure (raygen would otherwise read a null one).
  VkDeviceSize instanceBytes = 0;
  if (!instances.empty()) {
    instanceBytes =
      sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
    if (this->instanceBuffer == VK_NULL_HANDLE ||
        this->instanceBufferCapacity < instances.size()) {
      if (this->instanceBuffer != VK_NULL_HANDLE) {
        // Defer: a pending frame may still read the old instance buffer.
        VkDevice device = this->device;
        const VkAllocationCallbacks * allocator = this->allocator;
        const VkBuffer buffer = this->instanceBuffer;
        const VkDeviceMemory memory = this->instanceMemory;
        this->deferDestroy([device, allocator, buffer, memory]() {
          vkDestroyBuffer(device, buffer, allocator);
          vkFreeMemory(device, memory, allocator);
        });
        this->instanceBuffer = VK_NULL_HANDLE;
        this->instanceMemory = VK_NULL_HANDLE;
      }
      if (!this->createHostVisibleBuffer(
            instanceBytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            this->instanceBuffer, this->instanceMemory)) {
        return false;
      }
      this->instanceBufferCapacity = static_cast<uint32_t>(instances.size());
    }
    void * mapped = nullptr;
    if (vkMapMemory(this->device, this->instanceMemory, 0, instanceBytes, 0,
                    &mapped) != VK_SUCCESS) {
      return false;
    }
    std::memcpy(mapped, instances.data(), static_cast<size_t>(instanceBytes));
    vkUnmapMemory(this->device, this->instanceMemory);
  }

  // TLAS build sizes.
  VkAccelerationStructureGeometryInstancesDataKHR instancesData {};
  instancesData.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  instancesData.arrayOfPointers = VK_FALSE;
  if (this->instanceBuffer != VK_NULL_HANDLE) {
    instancesData.data.deviceAddress =
      this->getDeviceAddress(this->instanceBuffer);
  }

  VkAccelerationStructureGeometryKHR geometry {};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances = instancesData;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
  buildInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &geometry;
  buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
  sizeInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
    this->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo, &this->instanceCount, &sizeInfo);
  if (!this->createScratchBuffer(sizeInfo.buildScratchSize)) {
    return false;
  }

  // (Re)create the TLAS when it does not exist or is too small.
  if (this->tlas == VK_NULL_HANDLE ||
      this->tlasSize < sizeInfo.accelerationStructureSize) {
    if (this->tlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR(this->device, this->tlas,
                                        this->allocator);
      vkDestroyBuffer(this->device, this->tlasBuffer, this->allocator);
      vkFreeMemory(this->device, this->tlasMemory, this->allocator);
      this->tlas = VK_NULL_HANDLE;
      this->tlasBuffer = VK_NULL_HANDLE;
      this->tlasMemory = VK_NULL_HANDLE;
    }
    this->tlasSize = sizeInfo.accelerationStructureSize;
    if (!this->createDeviceLocalBuffer(
          this->tlasSize,
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
          this->tlasBuffer, this->tlasMemory)) {
      return false;
    }
    VkAccelerationStructureCreateInfoKHR asCI {};
    asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = this->tlasBuffer;
    asCI.size = this->tlasSize;
    asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(this->device, &asCI, this->allocator,
                                         &this->tlas) != VK_SUCCESS) {
      return false;
    }
    if (!this->updateDescriptors()) {
      return false;
    }
  }

  buildInfo.dstAccelerationStructure = this->tlas;
  buildInfo.scratchData.deviceAddress = this->scratchAddress;
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo {};
  rangeInfo.primitiveCount = this->instanceCount;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR * rangeInfos[] = {&rangeInfo};
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);
  return true;
}

// --- Material buffer ------------------------------------------------------

void
SoRTXRenderBackend::updateMaterials(const SoDrawList & drawlist)
{
  const int count = drawlist.getNumCommands();
  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(count) * sizeof(RTMaterial);
  if (bytes == 0) return;

  if (this->materialBuffer == VK_NULL_HANDLE ||
      bytes > this->materialBufferBytes) {
    if (this->materialBuffer != VK_NULL_HANDLE) {
      // Defer: a pending frame may still read the old material buffer.
      VkDevice device = this->device;
      const VkAllocationCallbacks * allocator = this->allocator;
      const VkBuffer buffer = this->materialBuffer;
      const VkDeviceMemory memory = this->materialMemory;
      void * mapped = this->materialMapped;
      this->deferDestroy([device, allocator, buffer, memory, mapped]() {
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, allocator);
        vkFreeMemory(device, memory, allocator);
      });
      this->materialBuffer = VK_NULL_HANDLE;
      this->materialMemory = VK_NULL_HANDLE;
      this->materialMapped = nullptr;
    }
    if (!this->createHostVisibleBuffer(
          bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          this->materialBuffer, this->materialMemory)) {
      this->emitError("updateMaterials: failed to create material buffer");
      return;
    }
    if (vkMapMemory(this->device, this->materialMemory, 0, bytes, 0,
                    &this->materialMapped) != VK_SUCCESS) {
      this->materialMapped = nullptr;
      return;
    }
    this->materialBufferBytes = bytes;
    if (!this->updateDescriptors()) {
      this->emitError("updateMaterials: descriptor update failed");
      return;
    }
  }
  this->materialCount = static_cast<uint32_t>(count);

  // Cache the PBR/lighting env overrides once for the whole frame instead of
  // calling envFlagEnabled()/getenv() inside the per-command loop below.
  // These flags never change mid-frame; they were recomputed identically for
  // every command on every frame.
  this->rtPbrEnabled = envFlagEnabled("FC_VULKAN_RT_PBR");
  this->rtMetalOverride = false;
  this->rtRoughOverride = false;
  this->rtMetalValue = 0.0f;
  this->rtRoughValue = 0.0f;
  if (const char * metalEnv = getenv("FC_VULKAN_RT_METAL")) {
    this->rtMetalOverride = true;
    this->rtMetalValue = strtof(metalEnv, nullptr);
  }
  if (const char * roughEnv = getenv("FC_VULKAN_RT_ROUGH")) {
    this->rtRoughOverride = true;
    this->rtRoughValue = strtof(roughEnv, nullptr);
  }

  // Reuse a grow-only scratch buffer instead of reallocating a fresh
  // std::vector<RTMaterial> from the heap every frame.
  std::vector<RTMaterial> & materials = this->materialScratch;
  materials.resize(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const SoMaterialData & material = command.material;
    RTMaterial & out = materials[static_cast<size_t>(i)];
    std::memset(&out, 0, sizeof(out));
    out.diffuse[0] = material.diffuse[0];
    out.diffuse[1] = material.diffuse[1];
    out.diffuse[2] = material.diffuse[2];
    out.diffuse[3] = material.diffuse[3];
    out.specular[0] = material.specular[0];
    out.specular[1] = material.specular[1];
    out.specular[2] = material.specular[2];
    out.specular[3] = 1.0f;
    out.emissive[0] = material.emissive[0];
    out.emissive[1] = material.emissive[1];
    out.emissive[2] = material.emissive[2];
    out.emissive[3] = 1.0f;
    out.params[0] = material.shininess;
    out.params[1] = material.twoSidedLighting ? 1.0f : 0.0f;
    out.params[3] =
      material.shadingModel == SO_SHADING_LEGACY_GOURAUD ? 1.0f : 0.0f;

    // Offset of this command's triangle normals in the normal pool (set by
    // appendTriangleNormals() during the BLAS build).
    const auto cacheFound = this->commandToCache.find(&command);
    if (cacheFound != this->commandToCache.end()) {
      const RTXCachedGeometry & entry = this->geometryCache[cacheFound->second];
      out.triangleData[0] = static_cast<float>(entry.normalPoolOffset);
      out.triangleData[1] = static_cast<float>(entry.normalCount);
      out.triangleData[2] = 0.0f;
      out.triangleData[3] = 0.0f;
    }

    // Optional PBR (metallic-roughness) parameters.  Off by default so
    // legacy Phong materials keep their exact current appearance; enable
    // with FC_VULKAN_RT_PBR=1 and optionally override the values with
    // FC_VULKAN_RT_METAL / FC_VULKAN_RT_ROUGH.  When disabled, usePbr stays
    // 0 and every shading path above falls back to the legacy model.
    out.pbr[0] = material.metalness;
    out.pbr[1] = material.roughness;
    out.pbr[2] = this->rtPbrEnabled ? 1.0f : 0.0f;
    out.pbr[3] = 0.0f;
    if (this->rtMetalOverride) {
      out.pbr[0] = this->rtMetalValue;
    }
    if (this->rtRoughOverride) {
      out.pbr[1] = this->rtRoughValue;
    }

    const SoLightingData * lighting =
      drawlist.getLighting(command.lightingHandle);
    static const SoLightingData emptyLighting;
    if (!lighting) lighting = &emptyLighting;

    // Fold the scene ambient into the material ambient (matches the raster
    // shader: litColor += ambientLight * materialAmbient).
    out.ambient[0] = lighting->ambient[0] * material.ambient[0];
    out.ambient[1] = lighting->ambient[1] * material.ambient[1];
    out.ambient[2] = lighting->ambient[2] * material.ambient[2];
    out.ambient[3] = 1.0f;

    const int lightCount = std::min<int>(
      static_cast<int>(lighting->lights.size()), MAX_SHADER_LIGHTS);
    out.params[2] = static_cast<float>(lightCount);
    for (int l = 0; l < lightCount; ++l) {
      const SoLightData & light = lighting->lights[static_cast<size_t>(l)];
      float * type = out.lightType + l * 4;
      type[0] = static_cast<float>(light.type);
      type[1] = type[2] = 0.0f;
      type[3] = 1.0f;
      float * color = out.lightColor + l * 4;
      color[0] = light.color[0];
      color[1] = light.color[1];
      color[2] = light.color[2];
      color[3] = 1.0f;
      float * direction = out.lightDirection + l * 4;
      direction[0] = light.direction[0];
      direction[1] = light.direction[1];
      direction[2] = light.direction[2];
      direction[3] = 1.0f;
      float * position = out.lightPosition + l * 4;
      position[0] = light.position[0];
      position[1] = light.position[1];
      position[2] = light.position[2];
      position[3] = 1.0f;
      float * attenuation = out.lightAttenuation + l * 4;
      attenuation[0] = light.attenuation[0];
      attenuation[1] = light.attenuation[1];
      attenuation[2] = light.attenuation[2];
      attenuation[3] = 1.0f;
      float * spot = out.lightSpot + l * 4;
      spot[0] = light.spotCutoffCos;
      spot[1] = light.spotExponent;
      spot[2] = 0.0f;
      spot[3] = 1.0f;
    }
  }
  if (this->materialMapped) {
    std::memcpy(this->materialMapped, materials.data(),
                static_cast<size_t>(bytes));
  }
}

// --- Frame recording ------------------------------------------------------

VkCommandBuffer
SoRTXRenderBackend::beginTransientCommandBuffer()
{
  // Persistent transient pool + one-shot command buffer for the AS phase,
  // allocated once instead of per frame.  The caller submits and waits the
  // buffer every frame; resetting it here is safe because the submission is
  // provably complete (vkQueueWaitIdle) by the time the next frame begins.
  if (this->transientPool == VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo pci {};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = this->queueFamilyIndex;
    if (vkCreateCommandPool(this->device, &pci, this->allocator,
                            &this->transientPool) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = this->transientPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(this->device, &ai,
                                 &this->transientCommandBuffer) !=
        VK_SUCCESS) {
      vkDestroyCommandPool(this->device, this->transientPool, this->allocator);
      this->transientPool = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
  }
  vkResetCommandBuffer(this->transientCommandBuffer, 0);
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(this->transientCommandBuffer, &bi) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  return this->transientCommandBuffer;
}

void
SoRTXRenderBackend::releaseTransientCommandBuffer()
{
  if (this->transientCommandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(this->device, this->transientPool, 1,
                         &this->transientCommandBuffer);
    this->transientCommandBuffer = VK_NULL_HANDLE;
  }
  if (this->transientPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->transientPool, this->allocator);
    this->transientPool = VK_NULL_HANDLE;
  }
}

namespace {

// Epsilon-aware float-array comparison: camera matrices may oscillate by
// tiny numerical amounts every frame without a real camera change, so a
// strict memcmp would never let the accumulation run.
bool
matricesNearlyEqual(const float * a, const float * b, size_t count)
{
  for (size_t i = 0; i < count; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    const float scale = std::max(std::fabs(a[i]), std::fabs(b[i]));
    if (d > 1e-4f * scale + 1e-7f) return false;
  }
  return true;
}

} // namespace

void
SoRTXRenderBackend::updatePathTracingState(const SoDrawList & drawlist,
                                           const SoRenderParams & params,
                                           const SoVulkanRenderTarget & target,
                                           VkCommandBuffer cmd)
{
  if (!this->createPathTracingBuffers(target.extent.width,
                                      target.extent.height)) {
    this->emitError("updatePathTracingState: failed to create path tracing buffers");
    return;
  }

  // Detect camera motion (view + projection) and scene changes; both reset
  // the progressive accumulation back to a live preview.  The scene-change
  // signal is the geometry-cache dirtiness computed by
  // updateGeometryCache() (the draw-list generation is a per-frame counter
  // and cannot be used for this).
  float viewMatrix[16];
  float projMatrix[16];
  SbMat viewValue;
  params.viewMatrix.getValue(viewValue);
  std::memcpy(viewMatrix, &viewValue[0][0], sizeof(viewMatrix));
  SbMat projValue;
  params.projMatrix.getValue(projValue);
  std::memcpy(projMatrix, &projValue[0][0], sizeof(projMatrix));
  const SbVec2s & vpSize = params.viewport.getViewportSizePixels();
  const bool viewChanged =
    !this->haveLastView ||
    !matricesNearlyEqual(viewMatrix, this->lastViewMatrix, 16) ||
    !matricesNearlyEqual(projMatrix, this->lastProjMatrix, 16) ||
    this->lastViewportWidth != static_cast<uint32_t>(vpSize[0]) ||
    this->lastViewportHeight != static_cast<uint32_t>(vpSize[1]);
  const bool sceneChanged = this->cacheChanged;

  if (!this->ptEnabled) {
    this->ptAccumulating = FALSE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
  }
  else if (this->ptStartLatch) {
    // The start flag: reset the accumulation and begin a fresh progressive
    // run, regardless of camera/scene changes.
    this->ptStartLatch = FALSE;
    this->ptAccumulating = TRUE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
  }
  else if (viewChanged || sceneChanged) {
    // Drop to the 1-spp live preview while the camera/scene is moving: the
    // accumulated rays are from the previous viewpoint and are useless.
    this->ptAccumulating = FALSE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
  }
  else if (this->ptAccumulating) {
    ++this->ptFrameIndex;
    // Converged: stop accumulating so the viewport can go idle instead of
    // tracing the same converged image forever.  Two stop conditions: the
    // hard sample cap, and the adaptive-sampling active-pixel fraction
    // (read back from the previous frame): once almost every pixel's
    // variance fell below the threshold, tracing more samples wastes GPU
    // time.  Saturate the idle counter so getPathTracingRefining() turns
    // the continuous-update request off.
    // Adaptive convergence only exists on the compute tracer (the SBT
    // raygen does not maintain the active-pixel counter).
    const bool adaptivelyConverged =
      !this->useSbtPipeline && this->ptAdaptiveEnabled &&
      this->ptFrameIndex >= this->ptAdaptiveMinSamples &&
      this->ptLastActiveFraction < this->ptAdaptiveStopFraction;
    if (this->ptFrameIndex >= this->ptMaxSamples || adaptivelyConverged) {
      this->ptAccumulating = FALSE;
      this->ptIdleFrames = this->ptSettleFrames;
    }
  }
  else if (this->ptIdleFrames < this->ptSettleFrames) {
    // Static but not accumulating (a move just stopped): count idle frames
    // and, once the camera has settled for a short window, auto-restart a
    // fresh accumulation so the view refines itself without an explicit
    // startPathTracing() call.
    ++this->ptIdleFrames;
    if (this->ptIdleFrames >= this->ptSettleFrames) {
      this->ptIdleFrames = 0;
      this->ptAccumulating = TRUE;
      this->ptFrameIndex = 0;
    }
  }
  // else: converged idle -- nothing to do until the camera or scene moves.

  if (getenv("FC_VULKAN_RT_DEBUG") && this->ptEnabled) {
    fprintf(stderr,
            "[RTDBG] ptState viewChanged=%d sceneChanged=%d accum=%d "
            "frameIndex=%u idle=%u\n",
            viewChanged ? 1 : 0, sceneChanged ? 1 : 0,
            this->ptAccumulating ? 1 : 0, this->ptFrameIndex,
            this->ptIdleFrames);
  }

  if (getenv("FC_VULKAN_PT_DEBUG")) {
    static uint32_t debugFrame = 0;
    if ((debugFrame++ % 30) == 0 || viewChanged || sceneChanged) {
      float maxViewDelta = 0.0f;
      int maxIdx = -1;
      for (int i = 0; i < 16; ++i) {
        const float d = std::fabs(viewMatrix[i] - this->lastViewMatrix[i]);
        if (d > maxViewDelta) { maxViewDelta = d; maxIdx = i; }
      }
      if (viewChanged && debugFrame < 40) {
        if (maxIdx >= 0) {
          fprintf(stderr,
                  "[PTDBG] f=%u view[%d] %.6f -> %.6f (t: %.3f,%.3f,%.3f)\n",
                  debugFrame - 1, maxIdx, this->lastViewMatrix[maxIdx],
                  viewMatrix[maxIdx], viewMatrix[12], viewMatrix[13],
                  viewMatrix[14]);
        }
        else {
          fprintf(stderr, "[PTDBG] f=%u view unchanged (viewport resize)\n",
                  debugFrame - 1);
        }
      }
      fprintf(stderr,
              "[PTDBG] f=%u latch=%d accum=%d frameIndex=%u "
              "viewChanged=%d sceneChanged=%d maxViewDelta=%g\n",
              debugFrame - 1, this->ptStartLatch ? 1 : 0,
              this->ptAccumulating ? 1 : 0, this->ptFrameIndex,
              viewChanged ? 1 : 0, sceneChanged ? 1 : 0, maxViewDelta);
    }
  }

  // Store the identity of this frame's view/scene for the next comparison.
  std::memcpy(this->lastViewMatrix, viewMatrix, sizeof(viewMatrix));
  std::memcpy(this->lastProjMatrix, projMatrix, sizeof(projMatrix));
  this->lastViewportWidth = static_cast<uint32_t>(vpSize[0]);
  this->lastViewportHeight = static_cast<uint32_t>(vpSize[1]);
  this->haveLastView = TRUE;
  this->cacheChanged = false;

  // A zero frame index means a fresh accumulation: clear the accumulation
  // and sums-of-squares buffers with a fill recorded here (still outside
  // any render pass).  This runs before the caller's render pass on the
  // same submission ordering, so the tracer observes the cleared buffers.
  if (this->ptEnabled && this->ptFrameIndex == 0) {
    vkCmdFillBuffer(cmd, this->accumBuffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, this->sumSqBuffer, 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier fillBarrier {};
    fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
  }
  // The active-pixel counter is per-frame: zero it before every traced
  // frame (the host reads it back after the submission's queue wait).
  if (this->ptEnabled && this->activeCounterBuffer != VK_NULL_HANDLE) {
    vkCmdFillBuffer(cmd, this->activeCounterBuffer, 0, VK_WHOLE_SIZE, 0);
    // Make the fill visible to the compute tracer's atomics.
    VkMemoryBarrier counterBarrier {};
    counterBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    counterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counterBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &counterBarrier, 0, nullptr, 0, nullptr);
  }
}

void
SoRTXRenderBackend::updateAdaptiveStats()
{
  // Called after the submission's queue wait: the host-visible counter
  // holds this frame's active-pixel count.
  uint32_t active = 0;
  if (this->activeCounterMapped && this->ptEnabled && this->ptAccumulating) {
    active = *static_cast<const uint32_t *>(this->activeCounterMapped);
  }
  this->ptLastActivePixels = active;
  const uint64_t total =
    static_cast<uint64_t>(this->ptBufferWidth) * this->ptBufferHeight;
  // Non-accumulating frames carry no meaningful count; report fully active
  // so a stale value can never prematurely stop the next progressive run.
  this->ptLastActiveFraction =
    (this->ptEnabled && this->ptAccumulating && total > 0)
      ? static_cast<float>(active) / static_cast<float>(total) : 1.0f;
  if (getenv("FC_VULKAN_RT_DEBUG") && this->ptEnabled) {
    fprintf(stderr,
            "[RTDBG] adaptive active=%u/%llu fraction=%.4f frameIndex=%u "
            "accum=%d self=%p buf=%ux%u\n",
            active, static_cast<unsigned long long>(total),
            this->ptLastActiveFraction, this->ptFrameIndex,
            this->ptAccumulating ? 1 : 0, static_cast<const void *>(this),
            this->ptBufferWidth, this->ptBufferHeight);
  }
}

bool
SoRTXRenderBackend::recordAccelerationStructures(
  const SoDrawList & drawlist, const SoRenderParams & params,
  const SoVulkanRenderTarget & target, VkCommandBuffer cmd)
{
  // Alternate the descriptor pair before any descriptor updates this frame:
  // the previous frame's sets may still be bound to a pending submission.
  this->descriptorSetIndex = (this->descriptorSetIndex + 1) & 1u;

  // The geometry cache update must run first: it computes cacheChanged,
  // which the path tracing state machine consumes as the scene-change
  // signal (see updatePathTracingState()).
  this->updateGeometryCache(drawlist);
  this->updatePathTracingState(drawlist, params, target, cmd);
  if (!this->createStorageImage(target.extent.width, target.extent.height)) {
    this->emitError("recordAccelerationStructures: failed to create storage image");
    return false;
  }

  // The storage image is written by the raygen shader and sampled by the
  // present shader, both in GENERAL layout.  Transition it once per
  // (re)creation here, outside the caller's render pass (the pass has no
  // subpass self-dependency, so layout transitions cannot be recorded
  // inside it).
  if (this->storageImageNeedsLayoutInit && this->storageImage != VK_NULL_HANDLE) {
    VkImageMemoryBarrier imageBarrier {};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = this->storageImage;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.layerCount = 1;
    imageBarrier.srcAccessMask = 0;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    this->storageImageNeedsLayoutInit = false;
  }

  // BLAS builds (lazy per command; only for entries without an AS) and
  // refits (topology-stable position edits).  The builds also (re)upload
  // the triangle-normal pool entries consumed by updateMaterials(), which
  // therefore runs after this loop.
  this->statBlasBuilt = 0;
  this->statBlasRefit = 0;
  this->statBlasReused = 0;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_TRANSPARENT ||
        command.pass == SO_RENDERPASS_OVERLAY) continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    RTXCachedGeometry & entry = this->geometryCache[found->second];
    if (entry.refitPending) {
      ++this->statBlasRefit;
      if (!this->refitBlas(entry, command, cmd)) {
        this->emitError("recordAccelerationStructures: failed to refit BLAS");
        return false;
      }
    }
    else if (entry.blas == VK_NULL_HANDLE) {
      ++this->statBlasBuilt;
      if (!this->buildBlas(entry, command, cmd)) {
        this->emitError("recordAccelerationStructures: failed to build BLAS");
        return false;
      }
    }
    else {
      ++this->statBlasReused;
    }
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr,
            "[RTDBG] blas built=%u refit=%u reused=%u cache=%zu\n",
            this->statBlasBuilt, this->statBlasRefit, this->statBlasReused,
            this->geometryCache.size());
  }

  this->updateMaterials(drawlist);

  // TLAS build (instances reference the BLASes built above).  The TLAS
  // handle may change here, so refresh the binding-0 descriptor before the
  // trace phase runs.
  if (!this->buildTlas(drawlist, cmd)) {
    this->emitError("recordAccelerationStructures: failed to build TLAS");
    return false;
  }
  if (!this->updateDescriptors()) {
    this->emitError("recordAccelerationStructures: descriptor update failed");
    return false;
  }

  // Barrier: BLAS/TLAS builds -> ray tracing shaders.  Recorded here, still
  // outside the render pass (acceleration-structure builds and buffer copies
  // are not allowed inside one).
  VkMemoryBarrier asBarrier {};
  asBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  asBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  asBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &asBarrier, 0, nullptr, 0, nullptr);

  // --- Frame uniform data (host-visible; no barrier needed) --------------
  if (this->frameMapped) {
    RTXFrameBlock frame {};    SbMat viewValue;
    params.viewMatrix.getValue(viewValue);
    std::memcpy(frame.view, &viewValue[0][0], sizeof(float) * 16);

    SbMatrix viewInverse = params.viewMatrix.inverse();
    SbMat viValue;
    viewInverse.getValue(viValue);
    std::memcpy(frame.viewInverse, &viValue[0][0], sizeof(float) * 16);

    SbMatrix projInverse = params.projMatrix.inverse();
    SbMat piValue;
    projInverse.getValue(piValue);
    std::memcpy(frame.projInverse, &piValue[0][0], sizeof(float) * 16);

    frame.cameraPos[0] = frame.viewInverse[12];
    frame.cameraPos[1] = frame.viewInverse[13];
    frame.cameraPos[2] = frame.viewInverse[14];
    frame.cameraPos[3] = 1.0f;

    const SbVec2s & vpSize = params.viewport.getViewportSizePixels();
    frame.viewport[0] = static_cast<float>(vpSize[0]);
    frame.viewport[1] = static_cast<float>(vpSize[1]);
    // Orthographic flag: Coin's ortho projection is affine (proj[3][3] == 1)
    // while the perspective projection has proj[3][3] == 0.  The compute
    // tracer picks parallel vs fanning primary rays from this bit.
    frame.viewport[2] = params.projMatrix[3][3] != 0.0f ? 1.0f : 0.0f;
    frame.viewport[3] = 0.0f;

    frame.bgTop[0] = params.backgroundTopColor[0];
    frame.bgTop[1] = params.backgroundTopColor[1];
    frame.bgTop[2] = params.backgroundTopColor[2];
    frame.bgTop[3] = 1.0f;
    frame.bgBottom[0] = params.backgroundBottomColor[0];
    frame.bgBottom[1] = params.backgroundBottomColor[1];
    frame.bgBottom[2] = params.backgroundBottomColor[2];
    frame.bgBottom[3] = 1.0f;

    frame.state[0] = static_cast<float>(this->ptFrameIndex);
    // 3.0 = debug constant fill (FC_VULKAN_RT_DEBUG_FILL); consumed by the
    // ray-query compute tracer (u_state.y > 2.5).
    frame.state[1] = envFlagEnabled("FC_VULKAN_RT_DEBUG_FILL")
      ? 3.0f : (this->ptEnabled ? 1.0f : 0.0f);
    frame.state[2] = this->ptAccumulating ? 1.0f : 0.0f;
    frame.state[3] = static_cast<float>(this->ptMaxBounces);

    // Adaptive sampling parameters; a zero threshold disables the early-out.
    frame.adaptive[0] = static_cast<float>(this->ptAdaptiveMinSamples);
    frame.adaptive[1] = this->ptAdaptiveEnabled
      ? this->ptAdaptiveThreshold : 0.0f;
    frame.adaptive[2] = 0.0f;
    frame.adaptive[3] = 0.0f;

    std::memcpy(this->frameMapped, &frame, sizeof(frame));

    if (getenv("FC_VULKAN_RT_DEBUG")) {
      static uint32_t debugFrame = 0;
      if ((debugFrame++ % 120) == 0) {
        fprintf(stderr,
                "[RTDBG] frame: cam=(%.2f,%.2f,%.2f) vp=(%.0fx%.0f) "
                "bgTop=(%.2f,%.2f,%.2f) bgBottom=(%.2f,%.2f,%.2f) "
                "state=(%.0f,%.0f,%.0f,%.0f)\n",
                frame.cameraPos[0], frame.cameraPos[1], frame.cameraPos[2],
                frame.viewport[0], frame.viewport[1],
                frame.bgTop[0], frame.bgTop[1], frame.bgTop[2],
                frame.bgBottom[0], frame.bgBottom[1], frame.bgBottom[2],
                frame.state[0], frame.state[1], frame.state[2],
                frame.state[3]);
      }
    }
  }

  // --- Trace --------------------------------------------------------------
  // The path tracer runs as a ray tracing pipeline dispatched with
  // vkCmdTraceRaysKHR outside the caller's render pass (ray tracing is not
  // allowed inside one).  The raygen receives its frame state through the
  // 16-byte push constant block; the descriptor set stays as updated after
  // the TLAS (re)build above.
  if (this->useSbtPipeline) {
    RTXRaygenPush raygenPush;
    raygenPush.frameIndex = this->ptFrameIndex;
    raygenPush.flags = (this->ptEnabled ? 1u : 0u) |
      (this->ptAccumulating ? 2u : 0u) |
      (envFlagEnabled("FC_VULKAN_RT_DEBUG_FILL") ? 4u : 0u);
    raygenPush.maxBounces = this->ptMaxBounces;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      this->rtPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            this->rtPipelineLayout, 0, 1,
                            &this->rtDescriptorSets[this->descriptorSetIndex],
                            0, nullptr);
    vkCmdPushConstants(cmd, this->rtPipelineLayout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0,
                       sizeof(raygenPush), &raygenPush);
    vkCmdTraceRaysKHR(cmd, &this->raygenSbtRegion, &this->missSbtRegion,
                      &this->hitSbtRegion, &this->callableSbtRegion,
                      this->storageWidth, this->storageHeight, 1);
  }
  else {
    // Ray-query compute dispatch (default): the state rides in the frame
    // UBO (u_state) and no push constants are needed.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      this->computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->rtPipelineLayout, 0, 1,
                            &this->rtDescriptorSets[this->descriptorSetIndex],
                            0, nullptr);
    vkCmdDispatch(cmd, (this->storageWidth + 7) / 8,
                  (this->storageHeight + 7) / 8, 1);
  }

  VkMemoryBarrier traceBarrier {};
  traceBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  traceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  traceBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1,
                       &traceBarrier, 0, nullptr, 0, nullptr);
  return true;
}

bool
SoRTXRenderBackend::recordTraceAndPresent(const SoRenderParams & params,
                                           const SoVulkanRenderTarget & target,
                                           VkCommandBuffer cmd,
                                           VkRenderPass renderPass)
{
   if (!this->createStorageImage(target.extent.width, target.extent.height)) {
    this->emitError("recordTraceAndPresent: failed to create storage image");
    return false;
  }
  if (!this->createPresentPipeline(renderPass, target.sampleCount)) {
    this->emitError("recordTraceAndPresent: failed to create present pipeline");
    return false;
  }

  // NOTE: descriptor sets are intentionally NOT updated here.  The AS phase
  // already refreshed them (storage image creation, PT buffers, TLAS build),
  // and updating a set that the trace bound above would invalidate this
  // still-recording command buffer (VUID-vkUpdateDescriptorSets-None-03047).

  // The trace itself was recorded in the acceleration-structure phase:
  // vkCmdTraceRaysKHR cannot be issued inside a render pass instance, and
  // the caller's pass also has no subpass self-dependency for layout
  // transitions.  Only the present pass (a plain fullscreen draw) is
  // recorded here.

  // Present fullscreen triangle into the swapchain color attachment.
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, this->presentPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          this->presentPipelineLayout, 0, 1,
                          &this->presentDescriptorSets[this->descriptorSetIndex],
                          0, nullptr);

  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  VkViewport viewport {};
  viewport.x = static_cast<float>(origin[0]);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(origin[1]) -
                                 static_cast<int32_t>(size[1]));
  viewport.width = static_cast<float>(size[0]);
  viewport.height = static_cast<float>(size[1]);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor {};
  scissor.offset = {0, 0};
  scissor.extent = target.extent;
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  // Present push constant: viewport size, denoiseOn (path tracing and the
  // denoise toggle), the progressive frame index for diagnostics, and the
  // viewport origin so the fragment shader indexes the accumulation buffers
  // relative to the viewport (not the framebuffer) when rendering into a
  // sub-rect.
  const float presentPush[8] = {
    static_cast<float>(size[0]),
    static_cast<float>(size[1]),
    (this->ptEnabled && this->ptDenoise) ? 1.0f : 0.0f,
    static_cast<float>(this->ptFrameIndex),
    static_cast<float>(origin[0]),
    static_cast<float>(origin[1]),
    0.0f,
    0.0f};
  vkCmdPushConstants(cmd, this->presentPipelineLayout,
                     VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(presentPush), presentPush);

  vkCmdDraw(cmd, 3, 1, 0, 0);
  return true;
}

// --- Lifecycle ------------------------------------------------------------

void
SoRTXRenderBackend::shutdown()
{
  if (!this->isInitialized()) return;

  vkQueueWaitIdle(this->queue);

  // The queue is idle: drain both deferred-destruction batches.
  this->flushPendingDestroys();
  this->flushPendingDestroys();

  this->invalidateCache();
  this->freePendingStagingDestroys();

  if (this->tlas != VK_NULL_HANDLE) {
    vkDestroyAccelerationStructureKHR(this->device, this->tlas,
                                      this->allocator);
    this->tlas = VK_NULL_HANDLE;
  }
  if (this->tlasBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->tlasBuffer, this->allocator);
    this->tlasBuffer = VK_NULL_HANDLE;
  }
  if (this->tlasMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->tlasMemory, this->allocator);
    this->tlasMemory = VK_NULL_HANDLE;
  }
  if (this->instanceBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->instanceBuffer, this->allocator);
    this->instanceBuffer = VK_NULL_HANDLE;
  }
  if (this->instanceMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->instanceMemory, this->allocator);
    this->instanceMemory = VK_NULL_HANDLE;
  }
  this->instanceBufferCapacity = 0;
  this->tlasSize = 0;
  if (this->scratchBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->scratchBuffer, this->allocator);
    this->scratchBuffer = VK_NULL_HANDLE;
  }
  if (this->scratchMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->scratchMemory, this->allocator);
    this->scratchMemory = VK_NULL_HANDLE;
  }
  this->scratchSize = 0;
  this->scratchAddress = 0;
  if (this->storageImage != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, this->storageImageView, this->allocator);
    vkDestroyImage(this->device, this->storageImage, this->allocator);
    vkFreeMemory(this->device, this->storageImageMemory, this->allocator);
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageView = VK_NULL_HANDLE;
    this->storageImageMemory = VK_NULL_HANDLE;
  }
  if (this->presentSampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->presentSampler, this->allocator);
    this->presentSampler = VK_NULL_HANDLE;
  }
  if (this->accumBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    this->accumBuffer = VK_NULL_HANDLE;
  }
  if (this->accumMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->accumMemory = VK_NULL_HANDLE;
  }
  if (this->normalBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->normalBuffer, this->allocator);
    this->normalBuffer = VK_NULL_HANDLE;
  }
  if (this->normalMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->normalMemory, this->allocator);
    this->normalMemory = VK_NULL_HANDLE;
  }
  if (this->positionBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->positionBuffer, this->allocator);
    this->positionBuffer = VK_NULL_HANDLE;
  }
  if (this->positionMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->positionMemory, this->allocator);
    this->positionMemory = VK_NULL_HANDLE;
  }
  this->ptBufferWidth = 0;
  this->ptBufferHeight = 0;
  this->ptAccumulating = FALSE;
  this->ptFrameIndex = 0;
  if (this->materialBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->materialBuffer, this->allocator);
    vkFreeMemory(this->device, this->materialMemory, this->allocator);
    this->materialBuffer = VK_NULL_HANDLE;
    this->materialMemory = VK_NULL_HANDLE;
    this->materialMapped = nullptr;
  }
  this->materialCount = 0;
  this->materialBufferBytes = 0;
  if (this->frameBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->frameBuffer, this->allocator);
    vkFreeMemory(this->device, this->frameMemory, this->allocator);
    this->frameBuffer = VK_NULL_HANDLE;
    this->frameMemory = VK_NULL_HANDLE;
    this->frameMapped = nullptr;
  }
  if (this->presentPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->presentPipeline, this->allocator);
    this->presentPipeline = VK_NULL_HANDLE;
  }
  if (this->rtPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->rtPipeline, this->allocator);
    this->rtPipeline = VK_NULL_HANDLE;
  }
  if (this->computePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->computePipeline, this->allocator);
    this->computePipeline = VK_NULL_HANDLE;
  }
  if (this->presentPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->presentPipelineLayout,
                            this->allocator);
    this->presentPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->rtPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->rtPipelineLayout,
                            this->allocator);
    this->rtPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->presentVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->presentVertexModule,
                          this->allocator);
    this->presentVertexModule = VK_NULL_HANDLE;
  }
  if (this->presentFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->presentFragmentModule,
                          this->allocator);
    this->presentFragmentModule = VK_NULL_HANDLE;
  }
  if (this->pathTraceModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->pathTraceModule,
                          this->allocator);
    this->pathTraceModule = VK_NULL_HANDLE;
  }
  if (this->raygenModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->raygenModule, this->allocator);
    this->raygenModule = VK_NULL_HANDLE;
  }
  if (this->missModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->missModule, this->allocator);
    this->missModule = VK_NULL_HANDLE;
  }
  if (this->shadowMissModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->shadowMissModule,
                          this->allocator);
    this->shadowMissModule = VK_NULL_HANDLE;
  }
  if (this->closestHitModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->closestHitModule,
                          this->allocator);
    this->closestHitModule = VK_NULL_HANDLE;
  }
  if (this->shadowClosestHitModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->shadowClosestHitModule,
                          this->allocator);
    this->shadowClosestHitModule = VK_NULL_HANDLE;
  }
  if (this->sbtBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->sbtBuffer, this->allocator);
    this->sbtBuffer = VK_NULL_HANDLE;
  }
  if (this->sbtMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->sbtMemory, this->allocator);
    this->sbtMemory = VK_NULL_HANDLE;
  }
  this->sbtRecordSize = 32;
  this->sbtBaseOffset = 0;
  if (this->normalPoolBuffer != VK_NULL_HANDLE) {
    this->normalPoolMapped = nullptr;
    vkDestroyBuffer(this->device, this->normalPoolBuffer, this->allocator);
    this->normalPoolBuffer = VK_NULL_HANDLE;
    vkFreeMemory(this->device, this->normalPoolMemory, this->allocator);
    this->normalPoolMemory = VK_NULL_HANDLE;
  }
  this->normalPoolCapacity = 0;
  this->normalPoolUsed = 0;
  if (this->descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(this->device, this->descriptorPool,
                            this->allocator);
    this->descriptorPool = VK_NULL_HANDLE;
  }
  if (this->rtSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->rtSetLayout,
                                 this->allocator);
    this->rtSetLayout = VK_NULL_HANDLE;
  }
  if (this->presentSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->presentSetLayout,
                                 this->allocator);
    this->presentSetLayout = VK_NULL_HANDLE;
  }
  if (this->offscreenFramebuffer != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(this->device, this->offscreenFramebuffer,
                         this->allocator);
    this->offscreenFramebuffer = VK_NULL_HANDLE;
  }
  if (this->offscreenRenderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(this->device, this->offscreenRenderPass,
                        this->allocator);
    this->offscreenRenderPass = VK_NULL_HANDLE;
  }
  this->releaseTransientCommandBuffer();
  this->offscreenColorImage = VK_NULL_HANDLE;
  this->offscreenColorView = VK_NULL_HANDLE;
  this->rtDescriptorSets[0] = VK_NULL_HANDLE;
  this->rtDescriptorSets[1] = VK_NULL_HANDLE;
  this->presentDescriptorSets[0] = VK_NULL_HANDLE;
  this->presentDescriptorSets[1] = VK_NULL_HANDLE;

  this->instance = VK_NULL_HANDLE;
  this->physicalDevice = VK_NULL_HANDLE;
  this->device = VK_NULL_HANDLE;
  this->queue = VK_NULL_HANDLE;
  this->allocator = nullptr;

  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

SbBool
SoRTXRenderBackend::render(const SoDrawList & drawlist,
                           const SoRenderParams & params)
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
  this->flushPendingDestroys();

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  // Offscreen path: single color attachment render pass + framebuffer.
  // The attachment is the swapchain/MSAA color image, so the render pass
  // and the present pipeline must both use the target's sample count
  // (VUID-VkFramebufferCreateInfo-renderPass-04553 and
  // VUID-VkGraphicsPipelineCreateInfo-renderPass-06082).  Both are cached
  // per target identity so this path does not recreate them every frame.
  const bool targetChanged =
    this->offscreenRenderPass == VK_NULL_HANDLE ||
    this->offscreenColorImage != target->colorImage ||
    this->offscreenColorView != target->colorImageView ||
    this->offscreenColorFormat != target->colorFormat ||
    this->offscreenSampleCount != target->sampleCount ||
    this->offscreenExtent.width != target->extent.width ||
    this->offscreenExtent.height != target->extent.height;
  if (targetChanged) {
    if (this->offscreenFramebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(this->device, this->offscreenFramebuffer,
                           this->allocator);
      this->offscreenFramebuffer = VK_NULL_HANDLE;
    }
    if (this->offscreenRenderPass != VK_NULL_HANDLE) {
      // The previous render() completed (it waits idle before returning),
      // so the old pass cannot be referenced by anything still pending.
      vkDestroyRenderPass(this->device, this->offscreenRenderPass,
                          this->allocator);
      this->offscreenRenderPass = VK_NULL_HANDLE;
    }

    VkAttachmentDescription attachment {};
    attachment.format = target->colorFormat;
    attachment.samples = target->sampleCount;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = target->colorLayout;
    attachment.finalLayout = target->colorLayout;

    VkAttachmentReference colorRef {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    VkRenderPassCreateInfo rpCI {};
    rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = 1;
    rpCI.pAttachments = &attachment;
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    if (vkCreateRenderPass(this->device, &rpCI, this->allocator,
                           &this->offscreenRenderPass) != VK_SUCCESS) {
      this->emitError("failed to create RT render pass");
      return FALSE;
    }

    VkFramebufferCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass = this->offscreenRenderPass;
    fci.attachmentCount = 1;
    fci.pAttachments = &target->colorImageView;
    fci.width = target->extent.width;
    fci.height = target->extent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(this->device, &fci, this->allocator,
                            &this->offscreenFramebuffer) != VK_SUCCESS) {
      vkDestroyRenderPass(this->device, this->offscreenRenderPass,
                          this->allocator);
      this->offscreenRenderPass = VK_NULL_HANDLE;
      this->emitError("failed to create RT framebuffer");
      return FALSE;
    }

    this->offscreenColorImage = target->colorImage;
    this->offscreenColorView = target->colorImageView;
    this->offscreenColorFormat = target->colorFormat;
    this->offscreenSampleCount = target->sampleCount;
    this->offscreenExtent = target->extent;
  }
  const VkRenderPass renderPass = this->offscreenRenderPass;
  const VkFramebuffer framebuffer = this->offscreenFramebuffer;

  // Persistent transient command buffer for the AS phase (BLAS/TLAS builds
  // and buffer copies), which is not allowed inside a render pass.  It is
  // recorded first, then the render pass begins and only the trace/present
  // work is recorded inside it.
  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    this->emitError("failed to allocate RT command buffer");
    return FALSE;
  }

  // Phase 1: acceleration structures (outside the render pass).
  const bool asOk =
    this->recordAccelerationStructures(drawlist, params, *target, cmd);
  bool traceOk = false;

  if (asOk) {
    VkRenderPassBeginInfo rpbi {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = framebuffer;
    rpbi.renderArea.offset = {0, 0};
    rpbi.renderArea.extent = target->extent;
    rpbi.clearValueCount = 0;
    rpbi.pClearValues = nullptr;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    // The attachment is loaded, not cleared at pass begin: the clear is
    // issued below scoped to the requested viewport region (matching the
    // raster backend's recordClear()).  Clearing the whole attachment here
    // would overwrite content outside a sub-region viewport.
    if (params.flags & SO_PARAM_CLEAR_WINDOW) {
      VkClearAttachment clear {};
      clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      clear.colorAttachment = 0;
      clear.clearValue.color.float32[0] = params.clearColor[0];
      clear.clearValue.color.float32[1] = params.clearColor[1];
      clear.clearValue.color.float32[2] = params.clearColor[2];
      clear.clearValue.color.float32[3] = params.clearColor[3];

      const SbVec2s & origin = params.viewport.getViewportOriginPixels();
      const SbVec2s & size = params.viewport.getViewportSizePixels();
      const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
      const int32_t y0 = std::max(
        0, static_cast<int32_t>(target->extent.height) -
             static_cast<int32_t>(origin[1]) -
             static_cast<int32_t>(size[1]));
      const int32_t x1 = std::min(static_cast<int32_t>(target->extent.width),
                                  static_cast<int32_t>(origin[0]) +
                                    static_cast<int32_t>(size[0]));
      const int32_t y1 = std::min(
        static_cast<int32_t>(target->extent.height),
        static_cast<int32_t>(target->extent.height) -
          static_cast<int32_t>(origin[1]));
      if (x1 > x0 && y1 > y0) {
        VkClearRect rect {};
        rect.rect.offset = {x0, y0};
        rect.rect.extent = {static_cast<uint32_t>(x1 - x0),
                            static_cast<uint32_t>(y1 - y0)};
        rect.baseArrayLayer = 0;
        rect.layerCount = 1;
        vkCmdClearAttachments(cmd, 1, &clear, 1, &rect);
      }
    }

    // Phase 2: trace + present (inside the render pass).
    traceOk =
      this->recordTraceAndPresent(params, *target, cmd, renderPass);

    vkCmdEndRenderPass(cmd);
  }
  vkEndCommandBuffer(cmd);

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  const VkResult submitResult = vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE);
  const VkResult waitResult = submitResult == VK_SUCCESS
    ? vkQueueWaitIdle(this->queue) : VK_SUCCESS;
  if (submitResult == VK_SUCCESS && waitResult == VK_SUCCESS) {
    this->updateAdaptiveStats();
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr, "[RTDBG] submit=%d wait=%d asOk=%d traceOk=%d\n",
            static_cast<int>(submitResult), static_cast<int>(waitResult),
            asOk ? 1 : 0, traceOk ? 1 : 0);
  }
  const bool submitted = submitResult == VK_SUCCESS && waitResult == VK_SUCCESS;

  // Staging buffers are only referenced by the private submission; release
  // them after it provably completed (or never ran).  The transient command
  // buffer stays pooled for the next frame.
  if (submitted) {
    this->freePendingStagingDestroys();
  }

  if (!asOk || !traceOk || !submitted) {
    this->emitError("render: RT frame failed");
    return FALSE;
  }
  return TRUE;
}

SbBool
SoRTXRenderBackend::renderExternal(const SoDrawList & drawlist,
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
    this->emitError("renderExternal called without command buffer/render pass");
    return FALSE;
  }
  this->debugValidateDrawList(drawlist);
  this->flushPendingDestroys();

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  // The caller's command buffer is already inside an active render pass, so
  // the acceleration-structure phase (BLAS/TLAS builds and buffer copies) is
  // recorded on the persistent transient command buffer which is submitted
  // and waited on here.  Queue submission is strictly ordered and the wait
  // makes the AS writes visible to the trace recorded below, so no explicit
  // synchronization with the caller's buffer is required.
  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    this->emitError("renderExternal: failed to allocate AS command buffer");
    return FALSE;
  }
  const bool asOk =
    this->recordAccelerationStructures(drawlist, params, *target, cmd);
  vkEndCommandBuffer(cmd);
  bool submitted = FALSE;
  if (asOk) {
    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    submitted = vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE) ==
      VK_SUCCESS && vkQueueWaitIdle(this->queue) == VK_SUCCESS;
  }
  if (submitted) {
    this->updateAdaptiveStats();
  }
  // Staging buffers are only referenced by the private submission; release
  // them after it provably completed (or never ran).  The command buffer
  // and pool are released in either case.
  if (submitted) {
    this->freePendingStagingDestroys();
  }

  if (!asOk || !submitted) {
    this->emitError("renderExternal: AS phase failed");
    return FALSE;
  }

  // The present pass is recorded into the caller's buffer (inside its
  // render pass); the trace ran in the AS phase above.  The descriptor set
  // was refreshed by recordAccelerationStructures() after the TLAS
  // (re)build, so binding 0 references the current TLAS.
  return this->recordTraceAndPresent(params, *target, commandBuffer,
                                     renderPass) ? TRUE : FALSE;
}
