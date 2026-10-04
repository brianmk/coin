// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendCore.cpp
//
// Initialization and lifecycle: ctor/dtor, getName() and overlay setters;
// setMaxFramesInFlight() ring resize; initialize() and its resource-create helpers
// (command pool, descriptor layouts/pool, frame buffers/fences, lighting and
// instance-model rings, white texture, pipeline layout, shader modules);
// frame-slot and deferred-destroy bookkeeping.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanDebug.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanShared.h"
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanDebugUtils.h"

#include <vk_mem_alloc.h>

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include "rendering/vulkan/visual/Fragment.spv.h"
#include "rendering/vulkan/visual/Vertex.spv.h"
#include "rendering/vulkan/visual/WideLineFragment.spv.h"
#include "rendering/vulkan/visual/WideLineInstancedVertex.spv.h"
#include "rendering/vulkan/visual/WideLineVertex.spv.h"
#include "rendering/vulkan/visual/SubPixelCull.spv.h"
#include "rendering/vulkan/visual/BackgroundVertex.spv.h"
#include "rendering/vulkan/visual/BackgroundFragment.spv.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

using namespace CoinVulkanDetail;

SoVulkanRenderBackend::SoVulkanRenderBackend()
{
  this->pendingDestroys.setBatchCount(this->maxFramesInFlight);
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

  // Resizing frees referenced command buffers/fences and orphans lower ring batches; wait first.
  if (this->isInitialized()) {
    this->waitForInFlightFrames();
  }

  this->maxFramesInFlight = count;
  this->pendingDestroys.setBatchCount(count);

  // Re-allocate per-frame-slot command buffers/fences (only valid while the queue is idle).
  if (this->isInitialized()) {
    this->releaseFrameResources();
    if (!this->allocateFrameResources()) {
      this->emitError(
        "setMaxFramesInFlight: failed to reallocate frame resources");
    }
    // The lighting UBO ring is sized maxFramesInFlight * slotsPerFrame; grow it to match
    // or the ring-offset math runs past the allocation.  swapLightingBuffer() waits and repoints.
    if (this->lightingBuffer != VK_NULL_HANDLE) {
      const VkDeviceSize totalBytes =
        static_cast<VkDeviceSize>(this->maxFramesInFlight) *
        this->uboSlotsPerFrame * this->uboSlotStride;
      VkBuffer newBuffer = VK_NULL_HANDLE;
      VmaAllocation newMemory = nullptr;
      void * newMapped = nullptr;
      if (!this->createMappedBuffer(totalBytes,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    newBuffer, newMemory, &newMapped)) {
        this->emitError(
          "setMaxFramesInFlight: failed to resize lighting UBO");
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
SoVulkanRenderBackend::setTessellationOverlay(SbBool enabled)
{
  this->tessellationOverlay = enabled;
}

void
SoVulkanRenderBackend::setEdgeColor(const SbColor4f & color)
{
  this->edgeColor = color;
}

void
SoVulkanRenderBackend::setSceneLights(const SoLightingData & lighting)
{
  this->sceneLighting = lighting;
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
  if (deviceContext->capsValid) {
    this->hasPipelineCreationFeedback =
      deviceContext->caps.pipelineCreationFeedback;
  }
  this->memProps.setDevice(this->physicalDevice);

  // Resolve synchronization2 once; null means not enabled, and helpers fall back to legacy.
  {
    SoVulkanShared::Sync2Dispatch & sync2 = SoVulkanShared::sync2Dispatch();
    sync2.cmdPipelineBarrier2 =
      SoVulkanShared::loadDispatch<PFN_vkCmdPipelineBarrier2KHR>(
        vkGetDeviceProcAddr(this->device, "vkCmdPipelineBarrier2KHR"));
    sync2.queueSubmit2 = SoVulkanShared::loadDispatch<PFN_vkQueueSubmit2KHR>(
      vkGetDeviceProcAddr(this->device, "vkQueueSubmit2KHR"));
    this->emitLog(sync2.cmdPipelineBarrier2 != nullptr
                    ? "synchronization2: enabled"
                    : "synchronization2: unavailable (legacy barriers)");
  }

  // Bind the render-pass/framebuffer cache; its deferred release goes into the frame ring.
  this->renderPasses.setDevice(this->device, this->allocator);
  this->renderPasses.setDeferredDestroy([this](std::function<void()> && fn) {
    this->deferDestroy(std::move(fn));
  });

  // VMA owns texture-image memory and sub-allocates from blocks, avoiding per-upload driver calls.
  {
    VmaAllocatorCreateInfo allocatorInfo {};
    allocatorInfo.physicalDevice = this->physicalDevice;
    allocatorInfo.device = this->device;
    allocatorInfo.instance = this->instance;
    // VMA selects its 1.1+/1.2 entry points (vkBindBufferMemory2,
    // vkGetBufferMemoryRequirements2, ...) from this; use the version the
    // device actually reports, capped at the 1.2 the backend is written
    // against, so a Vulkan 1.0/1.1 device is not handed 1.2 calls.
    const uint32_t deviceApiVersion = deviceContext->apiVersion != 0
      ? deviceContext->apiVersion
      : VK_API_VERSION_1_0;
    allocatorInfo.vulkanApiVersion =
      std::min(deviceApiVersion, static_cast<uint32_t>(VK_API_VERSION_1_2));
    allocatorInfo.pAllocationCallbacks = this->allocator;
    if (vmaCreateAllocator(&allocatorInfo, &this->vmaAllocator) != VK_SUCCESS) {
      this->emitError("SoVulkanRenderBackend: vmaCreateAllocator failed");
      return FALSE;
    }
  }

  // Worker count for the persistent record pool (also used by wide-line expansion, so
  // sized whenever cores exist); parallel recording (M1d) stays opt-in for identical output.
  // Worker 0 is the recording thread, 1..N-1 are spawned; capped at 8 (or config).
  {
    unsigned int hw = std::thread::hardware_concurrency();
    this->maxRecordWorkers = hw == 0 ? 1 : hw;
    if (this->maxRecordWorkers > 8) this->maxRecordWorkers = 8;
    const std::optional<unsigned int> & cap =
      SoVulkanConfig::get().concurrency.recordWorkerCap;
    if (cap && *cap >= 1 && *cap < this->maxRecordWorkers) {
      this->maxRecordWorkers = *cap;
    }
    this->parallelRecordEnabled =
      this->maxRecordWorkers > 1 &&
      SoVulkanConfig::get().concurrency.parallelRecord;
  }
  vkBackendTrace(0, "init.parallelConfig", "parallel=%d W=%u",
                 this->parallelRecordEnabled ? 1 : 0, this->maxRecordWorkers);

  // Vulkan cannot read back which features a created device enabled, so query what it
  // supports and rely on the app enabling exactly those (QuarterVulkanWidget::configureDeviceFeatures):
  // gates VK_POLYGON_MODE_LINE/POINT pipelines and 1/2-component upload formats.
  VkPhysicalDeviceFeatures supportedFeatures {};
  vkGetPhysicalDeviceFeatures(this->physicalDevice, &supportedFeatures);
  this->fillModeNonSolid = supportedFeatures.fillModeNonSolid ? true : false;

  const auto sampledOptimal = [this](const VkFormat fmt) {
    VkFormatProperties props {};
    vkGetPhysicalDeviceFormatProperties(this->physicalDevice, fmt, &props);
    return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
      ? true : false;
  };
  this->sampledR8 = sampledOptimal(VK_FORMAT_R8_UNORM);
  this->sampledR8G8 = sampledOptimal(VK_FORMAT_R8G8_UNORM);

  // Mark initialized BEFORE creating resources so a create*() failure still runs full shutdown().
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

  if (!this->createLightingConstBuffer()) {
    this->emitError("failed to create Vulkan lighting constant buffer");
    this->shutdown();
    return FALSE;
  }

  if (!this->createLightingDescriptorSet()) {
    this->emitError("failed to create Vulkan lighting descriptor set");
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

  if (!this->createPipelineCache()) {
    this->emitError("failed to create Vulkan pipeline cache");
    this->shutdown();
    return FALSE;
  }

  if (!this->createSubPixelCullPipeline()) {
    // Geometry LOD is optional: without compute the pipeline stays null and the pre-pass no-ops.
    this->emitLog("geometry-LOD compute pipeline unavailable; full draws only");
  }

  this->emitLog("initialized");
  return TRUE;
}

bool
SoVulkanRenderBackend::createPipelineCache()
{
  // The pipeline store owns the VkPipelineCache; bind device/allocator and route its logs.
  this->pipelines.setDevice(this->device, this->allocator);
  this->pipelines.setLogger(
    [this](const char * message) { this->emitLog(message); });
  return this->pipelines.initialize();
}

void
SoVulkanRenderBackend::setPipelineCachePath(const std::string & path)
{
  // Key the persisted cache to the compiled shaders: PipelineKey omits shader code, so a
  // rebuilt shader could otherwise be served a stale projection/push layout; a shader
  // change yields a new key and rejects the old blob.
  uint64_t shaderKey = 1469598103934665603ull; // FNV-1a offset basis
  const auto mix = [&shaderKey](const uint32_t * code, const size_t count) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(code);
    const size_t n = count * sizeof(uint32_t);
    for (size_t i = 0; i < n; ++i) {
      shaderKey ^= bytes[i];
      shaderKey *= 1099511628211ull; // FNV-1a prime
    }
  };
  mix(coin_vulkan_visual_vertex_spirv, coin_vulkan_visual_vertex_spirv_count);
  mix(coin_vulkan_visual_fragment_spirv,
      coin_vulkan_visual_fragment_spirv_count);
  mix(coin_vulkan_wide_line_vertex_spirv,
      coin_vulkan_wide_line_vertex_spirv_count);
  mix(coin_vulkan_wide_line_fragment_spirv,
      coin_vulkan_wide_line_fragment_spirv_count);
  mix(coin_vulkan_wide_line_instanced_vertex_spirv,
      coin_vulkan_wide_line_instanced_vertex_spirv_count);
  mix(coin_vulkan_background_vertex_spirv,
      coin_vulkan_background_vertex_spirv_count);
  mix(coin_vulkan_background_fragment_spirv,
      coin_vulkan_background_fragment_spirv_count);
  this->pipelines.setShaderKey(shaderKey);
  this->pipelines.setPath(path);
}

bool
SoVulkanRenderBackend::createCommandPool()
{
  SoVulkanDebugUtils::setDevice(this->device);
  SoVulkanDebugUtils::nameObject(this->device, VK_OBJECT_TYPE_DEVICE,
                                 reinterpret_cast<uint64_t>(this->device),
                                 "Coin raster VkDevice");
  VkCommandPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
             VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  ci.queueFamilyIndex = this->queueFamilyIndex;
  if (vkCreateCommandPool(this->device, &ci, this->allocator,
                          &this->commandPool) != VK_SUCCESS) {
    return false;
  }
  SoVulkanDebugUtils::nameObject(this->device, VK_OBJECT_TYPE_COMMAND_POOL,
                                 reinterpret_cast<uint64_t>(this->commandPool),
                                 "Coin raster primary command pool");
  // Secondary pools for M1c/M1d: one per worker (a shared pool would race its allocator).
  this->secondaryCommandPools.assign(this->maxRecordWorkers, VK_NULL_HANDLE);
  for (size_t i = 0; i < this->secondaryCommandPools.size(); ++i) {
    VkCommandPool & pool = this->secondaryCommandPools[i];
    if (vkCreateCommandPool(this->device, &ci, this->allocator, &pool) !=
        VK_SUCCESS) {
      return false;
    }
    char label[64];
    std::snprintf(label, sizeof(label),
                  "Coin raster secondary command pool %zu", i);
    SoVulkanDebugUtils::nameObject(this->device, VK_OBJECT_TYPE_COMMAND_POOL,
                                   reinterpret_cast<uint64_t>(pool), label);
  }
  return this->allocateFrameResources();
}

bool
SoVulkanRenderBackend::buildRecordPool()
{
  if (this->maxRecordWorkers <= 1) return true;
  if (this->recordWorkers.size() >= this->maxRecordWorkers - 1) return true;
  {
    std::lock_guard<std::mutex> lk(this->recordMutex);
    this->recordPoolStopped = false;
  }
  vkBackendTrace(0, "buildRecordPool.enter", "W=%u",
                 this->maxRecordWorkers);
  // Resume from the current pool size so a partially-built pool tops up, not duplicates.
  for (uint32_t w = static_cast<uint32_t>(this->recordWorkers.size()) + 1;
       w < this->maxRecordWorkers; ++w) {
    try {
      this->recordWorkers.emplace_back(
        &SoVulkanRenderBackend::recordJobWorker, this, static_cast<size_t>(w));
    }
    catch (const std::exception &) {
      // Spawn failed: fall back to serial recording and retire any spawned threads.
      this->shutdownRecordPool();
      this->parallelRecordEnabled = false;
      this->maxRecordWorkers = 1;
      return false;
    }
  }
  return true;
}

void
SoVulkanRenderBackend::shutdownRecordPool()
{
  if (this->recordWorkers.empty()) return;
  {
    std::lock_guard<std::mutex> lk(this->recordMutex);
    this->recordPoolStopped = true;
  }
  this->recordCvSpawn.notify_all();
  for (std::thread & t : this->recordWorkers) {
    if (t.joinable()) t.join();
  }
  this->recordWorkers.clear();
}

void
SoVulkanRenderBackend::recordJobWorker(const size_t workerIndex)
{
  uint32_t processed = 0;
  while (true) {
    uint32_t generation = 0;
    {
      std::unique_lock<std::mutex> lk(this->recordMutex);
      this->recordCvSpawn.wait(lk, [this, &processed] {
        return this->recordPoolStopped ||
               this->recordJobGeneration != processed;
      });
      if (this->recordPoolStopped) return;
      processed = this->recordJobGeneration;
      generation = processed;
    }
    vkBackendTrace(this->uboFrameIndex, "recordJobWorker.wake",
                   "w=%zu gen=%u", workerIndex, processed);
    if (workerIndex >= this->recordJobs.size()) continue;
    ParallelRecordJob & job = this->recordJobs[workerIndex];
    vkBackendTrace(this->uboFrameIndex, "recordJobWorker.record",
                   "w=%zu secondary=%p items=%zu", workerIndex,
                   reinterpret_cast<const void *>(job.secondary),
                   job.items.size());
    if (job.expandWideLines) {
      // Wide-line CPU expansion: no command buffer, just the quad computation; per-worker
      // cache entries and thread-local scratch make it race-free.
      if (job.wlineSplitPhase != 0) {
        // One command partitioned by segment range: owner-sized shared scratch, disjoint writes.
        this->expandWideLinesSplitRange(job.wlineSplitPhase,
                                        job.wlineSplitBegin, job.wlineSplitEnd);
        job.ok = true;
      }
      else if (!job.params) {
        job.ok = false;
      }
      else {
        for (const SoRenderCommand * command : job.wideLineCommands) {
          const auto found = this->commandToCache.find(command);
          if (found == this->commandToCache.end()) continue;
          VulkanCachedCommand & entry = this->gpuCache[found->second];
          this->expandWideLinesFor(entry, *command, *job.params,
                                   command->pass == SO_RENDERPASS_OVERLAY);
        }
        job.ok = true;
      }
    }
    else if (!job.ctx || !job.drawlist || !job.params || !job.target) {
      job.ok = false;
    }
    else {
      job.ok = this->recordSecondaryChunk(*job.ctx, *job.drawlist, *job.params,
                                          *job.target, job.renderPass,
                                          job.items, job.secondary,
                                          job.framebuffer);
    }
    vkBackendTrace(this->uboFrameIndex, "recordJobWorker.done",
                   "w=%zu ok=%d", workerIndex, job.ok ? 1 : 0);
    // The done count is the condition recordCvDone.wait() checks, so publish it under
    // recordMutex with the notify: incrementing outside can lose a notify after the
    // waiter's final predicate check (sleep forever).  Publish only for the current
    // generation, else a superseded completion makes the next join proceed early.
    {
      std::lock_guard<std::mutex> lk(this->recordMutex);
      if (this->recordJobGeneration == generation) {
        this->recordDoneCount.fetch_add(1);
        this->recordCvDone.notify_all();
      }
    }
  }
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

  // One secondary per in-flight slot per worker (M1c/M1d), from that worker's own pool,
  // indexed [slot * maxRecordWorkers + worker] (workerSecondary()'s layout); re-recorded
  // only after the slot fence is waited.
  const uint32_t secondaryCount =
    this->maxFramesInFlight * this->maxRecordWorkers;
  this->secondaryCommandBuffers.assign(secondaryCount, VK_NULL_HANDLE);
  for (uint32_t w = 0; w < this->maxRecordWorkers; ++w) {
    if (w >= this->secondaryCommandPools.size() ||
        this->secondaryCommandPools[w] == VK_NULL_HANDLE) {
      return false;
    }
    std::vector<VkCommandBuffer> perWorker(this->maxFramesInFlight,
                                           VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo sai {};
    sai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    sai.commandPool = this->secondaryCommandPools[w];
    sai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    sai.commandBufferCount = this->maxFramesInFlight;
    if (vkAllocateCommandBuffers(this->device, &sai, perWorker.data()) !=
        VK_SUCCESS) {
      return false;
    }
    for (uint32_t s = 0; s < this->maxFramesInFlight; ++s) {
      this->secondaryCommandBuffers[
        static_cast<size_t>(s) * this->maxRecordWorkers + w] = perWorker[s];
    }
  }

  // Per-worker record contexts + job slots, sized by maxRecordWorkers.
  this->workerRecordContexts.assign(this->maxRecordWorkers, VulkanRecordContext{});
  this->recordJobs.assign(this->maxRecordWorkers, ParallelRecordJob{});
  // The pool serves parallel recording and wide-line expansion, so build it if >1 worker.
  if (this->maxRecordWorkers > 1) {
    if (!this->buildRecordPool()) return false;
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
  // Caller must have made the queue idle, or waited pending fences, before this runs.
  for (VkCommandBuffer buffer : this->frameCommandBuffers) {
    if (buffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(this->device, this->commandPool, 1, &buffer);
    }
  }
  this->frameCommandBuffers.clear();
  for (size_t i = 0; i < this->secondaryCommandBuffers.size(); ++i) {
    VkCommandBuffer & buffer = this->secondaryCommandBuffers[i];
    if (buffer == VK_NULL_HANDLE) continue;
    const uint32_t w =
      static_cast<uint32_t>(i % this->maxRecordWorkers);
    if (w < this->secondaryCommandPools.size()) {
      vkFreeCommandBuffers(this->device, this->secondaryCommandPools[w], 1,
                           &buffer);
    }
  }
  this->secondaryCommandBuffers.clear();
  this->workerRecordContexts.clear();
  this->recordJobs.clear();
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

VkCommandBuffer
SoVulkanRenderBackend::currentSecondaryCommandBuffer()
{
  return this->workerSecondary(
    this->uboFrameIndex % std::max<uint32_t>(this->maxFramesInFlight, 1u), 0);
}

VkCommandBuffer
SoVulkanRenderBackend::workerSecondary(const uint32_t frameSlot,
                                       const uint32_t worker)
{
  if (this->secondaryCommandBuffers.empty() || this->maxRecordWorkers == 0) {
    return VK_NULL_HANDLE;
  }
  const size_t idx = static_cast<size_t>(frameSlot) * this->maxRecordWorkers +
                     worker;
  if (idx >= this->secondaryCommandBuffers.size()) return VK_NULL_HANDLE;
  return this->secondaryCommandBuffers[idx];
}

VkCommandBuffer
SoVulkanRenderBackend::parallelCurrentSecondary(const uint32_t worker)
{
  return this->workerSecondary(
    this->uboFrameIndex % std::max<uint32_t>(this->maxFramesInFlight, 1u),
    worker);
}

void
SoVulkanRenderBackend::waitForInFlightFrames()
{
  // Wait only fences that are actually pending: the external path never submits here, so
  // its fences are never signaled and waiting would block forever.  The current slot is
  // never pending (beginFrame() cleared it), so growLightingUbo() cannot deadlock.  Called
  // from growLightingUbo()/setMaxFramesInFlight(), both rewriting submitted-buffer resources.
  std::vector<VkFence> pending;
  for (size_t i = 0; i < this->frameFences.size(); ++i) {
    if (i < this->frameFencePending.size() && this->frameFencePending[i] &&
        this->frameFences[i] != VK_NULL_HANDLE) {
      pending.push_back(this->frameFences[i]);
    }
  }
  if (pending.empty()) return;
  const VkResult result = vkWaitForFences(
      this->device, static_cast<uint32_t>(pending.size()), pending.data(), VK_TRUE,
      UINT64_MAX);
  if (result != VK_SUCCESS) {
    SoDebugError::postWarning("SoVulkanRenderBackend::waitForInFlightFrames",
                              "vkWaitForFences failed (VkResult=%d)",
                              static_cast<int>(result));
  }
}

bool
SoVulkanRenderBackend::createDescriptorSetLayout()
{
  // Set 0: lighting constant ring (binding 0, UBO dynamic); both stages (Phong per fragment).
  VkDescriptorSetLayoutBinding lightingBinding {};
  lightingBinding.binding = 0;
  lightingBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  lightingBinding.descriptorCount = 1;
  lightingBinding.stageFlags =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  lightingBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo lightingCi {};
  lightingCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  lightingCi.bindingCount = 1;
  lightingCi.pBindings = &lightingBinding;
  if (vkCreateDescriptorSetLayout(this->device, &lightingCi, this->allocator,
                                  &this->lightingSetLayout) != VK_SUCCESS) {
    return false;
  }

  // Set 1: per-draw view/model/material UBO (binding 0, dynamic) + texture (binding 1).
  VkDescriptorSetLayoutBinding bindings[2] {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  bindings[0].descriptorCount = 1;
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
  poolSizes[0].descriptorCount = 2048;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = 2048;

  VkDescriptorPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  ci.maxSets = 2048;
  ci.poolSizeCount = 2;
  ci.pPoolSizes = poolSizes;

  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(this->device, &ci, this->allocator, &pool) !=
      VK_SUCCESS) {
    return false;
  }
  this->descriptorPool = pool;
  this->descriptorPools.push_back(pool);
  SoVulkanDebugUtils::nameObject(this->device, VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                 reinterpret_cast<uint64_t>(pool),
                                 "Coin raster descriptor pool");
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
  // Per-command slots in a ring sized for maxFramesInFlight frames: each draw binds its
  // slot with a dynamic offset, so the GPU reads that draw's block, not a shared one.
  VkPhysicalDeviceProperties deviceProps;
  vkGetPhysicalDeviceProperties(this->physicalDevice, &deviceProps);
  const VkDeviceSize alignment = std::max<VkDeviceSize>(
    1, deviceProps.limits.minUniformBufferOffsetAlignment);
  this->uboSlotStride =
    (sizeof(VulkanDrawUbo) + alignment - 1) / alignment * alignment;
  this->uboSlotsPerFrame = 4096;
  const VkDeviceSize totalBytes =
    static_cast<VkDeviceSize>(this->maxFramesInFlight) *
    static_cast<VkDeviceSize>(this->uboSlotsPerFrame) * this->uboSlotStride;
  if (!this->createMappedBuffer(totalBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                this->lightingBuffer, this->lightingMemory,
                                &this->lightingMapped)) {
    this->emitError("createLightingUniformBuffer: buffer create/map failed");
    return false;
  }
  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_BUFFER,
    reinterpret_cast<uint64_t>(this->lightingBuffer), "draw UBO ring");
  // The per-instance model-matrix ring parallels the lighting ring (same layout), so
  // pre-size it here; the per-draw path must never grow it (races parallel recording).
  if (!this->ensureInstanceModelRingCapacity()) {
    this->emitError("createLightingUniformBuffer: failed to size instance buffer");
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createLightingConstBuffer()
{
  // A few slots per in-flight frame, one per distinct lighting handle (typically one),
  // each holding one VulkanLightingUbo (ambient + 8 lights): writing once per handle per
  // frame makes shared lighting O(#handles), not O(#draws).
  VkPhysicalDeviceProperties deviceProps;
  vkGetPhysicalDeviceProperties(this->physicalDevice, &deviceProps);
  const VkDeviceSize alignment = std::max<VkDeviceSize>(
    1, deviceProps.limits.minUniformBufferOffsetAlignment);
  this->lightingConstStride =
    (sizeof(VulkanLightingUbo) + alignment - 1) / alignment * alignment;
  // Fixed 8-frame ring, 8 unique-handle slots per frame, independent of maxFramesInFlight
  // (so no resize path); safe while in-flight frames <= 8 (swapchains are 2-3 images).
  this->lightingConstMaxSlots = 8u * 8u;
  const VkDeviceSize totalBytes =
    static_cast<VkDeviceSize>(this->lightingConstMaxSlots) *
    this->lightingConstStride;
  if (!this->createMappedBuffer(totalBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                this->lightingConstBuffer,
                                this->lightingConstMemory,
                                &this->lightingConstMapped)) {
    this->emitError("createLightingConstBuffer: buffer create/map failed");
    return false;
  }
  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_BUFFER,
    reinterpret_cast<uint64_t>(this->lightingConstBuffer),
    "lighting UBO ring");
  return true;
}

bool
SoVulkanRenderBackend::createLightingDescriptorSet()
{
  VkDescriptorSetAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = this->descriptorPool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &this->lightingSetLayout;
  if (vkAllocateDescriptorSets(this->device, &ai,
                               &this->lightingDescriptorSet) != VK_SUCCESS) {
    return false;
  }
  ++this->descriptorSetCount;

  VkDescriptorBufferInfo bufferInfo {};
  bufferInfo.buffer = this->lightingConstBuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = this->lightingConstStride;

  VkWriteDescriptorSet write {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = this->lightingDescriptorSet;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  write.pBufferInfo = &bufferInfo;
  vkUpdateDescriptorSets(this->device, 1, &write, 0, nullptr);
  return true;
}

bool
SoVulkanRenderBackend::growLightingUbo(const uint32_t minSlots)
{
  uint32_t slots = 4096;
  while (slots < minSlots) slots <<= 1;
  if (slots <= this->uboSlotsPerFrame) return true;

  VkBuffer newBuffer = VK_NULL_HANDLE;
  VmaAllocation newMemory = nullptr;
  void * newMapped = nullptr;
  const VkDeviceSize totalBytes =
    static_cast<VkDeviceSize>(this->maxFramesInFlight) *
    static_cast<VkDeviceSize>(slots) * this->uboSlotStride;
  if (!this->createMappedBuffer(totalBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                newBuffer, newMemory, &newMapped)) {
    this->emitError("growLightingUbo: failed to allocate larger UBO");
    return false;
  }

  return this->swapLightingBuffer(newBuffer, newMemory, newMapped, slots);
}

bool
SoVulkanRenderBackend::swapLightingBuffer(VkBuffer newBuffer,
                                          VmaAllocation newMemory,
                                          void * newMapped,
                                          const uint32_t newSlotsPerFrame)
{
  const VkBuffer oldBuffer = this->lightingBuffer;
  const VmaAllocation oldMemory = this->lightingMemory;
  this->lightingBuffer = newBuffer;
  this->lightingMemory = newMemory;
  this->lightingMapped = newMapped;
  this->uboSlotsPerFrame = newSlotsPerFrame;

  // Grow the model-matrix ring to the same geometry so the per-draw path never grows it.
  if (!this->ensureInstanceModelRingCapacity()) {
    this->emitError("swapLightingBuffer: failed to size instance buffer");
  }

  // The old buffer may still be referenced by a pending frame: destroy it after the ring
  // wraps (flushPendingDestroys()); freeing the memory unmaps it, so no explicit unmap.
  this->deferDestroyBufferMemory(oldBuffer, oldMemory);

  // Every descriptor set captured the old buffer (binding 0).  Rewriting a set bound in an
  // already-submitted command buffer is a violation, so wait for all in-flight submissions
  // first (rare: ring growth or frame-count change); the current frame is not yet submitted.
  this->waitForInFlightFrames();
  std::vector<VkWriteDescriptorSet> writes;
  std::vector<VkDescriptorBufferInfo> bufferInfos;
  // Reserve up front: collect() stores &bufferInfos.back() in each write, and a later
  // push_back reallocation would dangle it (garbage VkBuffer -> driver fault).  Max set
  // count is the white set plus one per texture.
  const size_t maxSets = this->textureCache.size() + 1;
  writes.reserve(maxSets);
  bufferInfos.reserve(maxSets);
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
  // beginFrame() advanced the frame index; every render starts at slot zero of its ring
  // half, and the slot cursor lives in the per-recording VulkanRecordContext.
  return true;
}

void
SoVulkanRenderBackend::beginFrame()
{
  vkBackendTrace(this->uboFrameIndex, "beginFrame.enter",
                 "nextFrame=%u", this->uboFrameIndex + 1);
  // One frame boundary: advance the ring cursor, then on the own-queue path wait the slot
  // fence.  The slot was last used maxFramesInFlight frames ago, so its fence covers that
  // submission and the UBO half, command buffer and deferred resources are safe to reuse.
  // The external path never signals these fences (the caller owns submission), so no wait
  // occurs -- external correctness rests on the caller honoring setMaxFramesInFlight().
  this->uboFrameIndex++;
  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  if (slot < this->frameFencePending.size() &&
      this->frameFencePending[slot] &&
      this->frameFences[slot] != VK_NULL_HANDLE) {
    const VkResult waitResult = vkWaitForFences(
        this->device, 1, &this->frameFences[slot], VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS) {
      SoDebugError::postWarning("SoVulkanRenderBackend::beginFrame",
                                "vkWaitForFences failed (VkResult=%d)",
                                static_cast<int>(waitResult));
    }
    const VkResult resetResult =
        vkResetFences(this->device, 1, &this->frameFences[slot]);
    if (resetResult != VK_SUCCESS) {
      SoDebugError::postWarning("SoVulkanRenderBackend::beginFrame",
                                "vkResetFences failed (VkResult=%d)",
                                static_cast<int>(resetResult));
    }
    this->frameFencePending[slot] = 0;
  }
  // Dynamic state is per-recording: the previous frame's buffer may have left a different
  // pipeline/viewport/scissor bound, so forget it to force this frame's first apply*.
  this->resetBoundState(this->recordContext);
  this->flushPendingDestroys();
}

void
SoVulkanRenderBackend::cacheFrameMatrices(const SoRenderParams & params)
{
  // SbMatrix is exactly float[4][4]: copy the 16 floats once so every draw reuses them.
  std::memcpy(this->frameViewFloats, &params.viewMatrix[0][0],
              sizeof(float) * 16);
  std::memcpy(this->frameProjFloats, &params.projMatrix[0][0],
              sizeof(float) * 16);
  this->frameDpr = params.devicePixelRatio > 0.0f
    ? params.devicePixelRatio : 1.0f;
}

void
SoVulkanRenderBackend::flushPendingDestroys()
{
  if (this->pendingDestroys.empty()) return;
  this->pendingDestroys.flushAt(this->uboFrameIndex);
}

void
SoVulkanRenderBackend::flushAllPendingDestroys()
{
  this->pendingDestroys.flushAll();
}

void
SoVulkanRenderBackend::deferDestroy(std::function<void()> && fn)
{
  this->pendingDestroys.deferAt(this->uboFrameIndex, std::move(fn));
}

void
SoVulkanRenderBackend::deferDestroyCacheEntry(VulkanCachedCommand & entry)
{
  if (entry.vertexBuffer == VK_NULL_HANDLE &&
      entry.indexBuffer == VK_NULL_HANDLE &&
      entry.sharedBlockId == 0 &&
      entry.instancedLineBuffer == VK_NULL_HANDLE &&
      entry.subPixelSlots.empty() &&
      entry.wideLineBuffers.empty()) {
    entry = VulkanCachedCommand();
    return;
  }
  if (entry.sharedBlockId != 0) {
    const uint32_t sharedBlockId = entry.sharedBlockId;
    std::vector<VulkanCachedCommand::VulkanWideLineBuffer> wideLine =
      std::move(entry.wideLineBuffers);
    std::vector<VulkanCachedCommand::VulkanSubPixelSlot> subPixel =
      std::move(entry.subPixelSlots);
    const VkBuffer instancedLineBuffer = entry.instancedLineBuffer;
    const VmaAllocation instancedLineMemory = entry.instancedLineMemory;
    VmaAllocator vma = this->vmaAllocator;
    this->deferDestroy([vma, wideLine, subPixel, instancedLineBuffer,
                        instancedLineMemory]() mutable {
      for (VulkanCachedCommand::VulkanWideLineBuffer & slot : wideLine) {
        slot.destroy(vma);
      }
      for (VulkanCachedCommand::VulkanSubPixelSlot & slot : subPixel) {
        if (slot.indexBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, slot.indexBuffer, slot.indexMemory);
        }
        if (slot.indirectBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, slot.indirectBuffer, slot.indirectMemory);
        }
      }
      if (instancedLineBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, instancedLineBuffer, instancedLineMemory);
      }
    });
    this->deferReleaseGeometryBlock(sharedBlockId);
    entry = VulkanCachedCommand();
    return;
  }
  VmaAllocator vma = this->vmaAllocator;
  const VkBuffer vertexBuffer = entry.vertexBuffer;
  const VmaAllocation vertexMemory = entry.vertexMemory;
  const VkBuffer indexBuffer = entry.indexBuffer;
  const VmaAllocation indexMemory = entry.indexMemory;
  const VkBuffer instancedLineBuffer = entry.instancedLineBuffer;
  const VmaAllocation instancedLineMemory = entry.instancedLineMemory;
  std::vector<VulkanCachedCommand::VulkanWideLineBuffer> wideLine =
    std::move(entry.wideLineBuffers);
  std::vector<VulkanCachedCommand::VulkanSubPixelSlot> subPixel =
    std::move(entry.subPixelSlots);
  this->deferDestroy(
    [vma, vertexBuffer, vertexMemory, indexBuffer,
     indexMemory, instancedLineBuffer, instancedLineMemory, wideLine,
     subPixel]() mutable {
      for (VulkanCachedCommand::VulkanWideLineBuffer & slot : wideLine) {
        slot.destroy(vma);
      }
      for (VulkanCachedCommand::VulkanSubPixelSlot & slot : subPixel) {
        if (slot.indexBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, slot.indexBuffer, slot.indexMemory);
        }
        if (slot.indirectBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, slot.indirectBuffer, slot.indirectMemory);
        }
      }
      if (instancedLineBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, instancedLineBuffer, instancedLineMemory);
      }
      if (indexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, indexBuffer, indexMemory);
      }
      if (vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, vertexBuffer, vertexMemory);
      }
    });
  entry = VulkanCachedCommand();
}

void
SoVulkanRenderBackend::deferDestroyTextureEntry(VulkanCachedTexture & entry)
{
  // The set returns to its pool only after the batch ring wraps (vkFreeDescriptorSets on an
  // in-use set is a violation).  Pools are append-only, so the captured handle stays valid.
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
  const VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  const VmaAllocator vmaAllocator = this->vmaAllocator;
  const VkImage image = entry.image;
  const VmaAllocation allocation = entry.allocation;
  const VkImageView view = entry.view;
  // The sampler is shared (samplerCache) and released once at shutdown(), not here.  The
  // image + VMA allocation are freed together after the frame's submission (deferred ring).
  this->deferDestroy([device, allocator, vmaAllocator, image, allocation,
                      view]() {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, view, allocator);
    }
    if (image != VK_NULL_HANDLE) {
      vmaDestroyImage(vmaAllocator, image, allocation);
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
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (vmaCreateImage(this->vmaAllocator, &ci, &allocInfo, &this->whiteImage,
                     &this->whiteImageAllocation, nullptr) != VK_SUCCESS) {
    this->emitError("createWhiteTexture: vmaCreateImage failed");
    return false;
  }

  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingMemory = nullptr;
  if (!this->createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging,
                          stagingMemory, &white)) {
    return false;
  }

  // One-shot upload: stage -> image (white 1x1), UNDEFINED -> TRANSFER_DST ->
  // SHADER_READ_ONLY; the helper waits for the queue to idle so staging is safe to free.
  if (!SoVulkanShared::withOneShotSubmit(
        this->device, this->queue, this->commandPool, this->allocator,
        [this, staging](VkCommandBuffer uploadBuffer) {
          SoVulkanShared::imageTransition(
            uploadBuffer, this->whiteImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
          VkBufferImageCopy region {};
          region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          region.imageSubresource.layerCount = 1;
          region.imageExtent = {extent, extent, 1};
          vkCmdCopyBufferToImage(uploadBuffer, staging, this->whiteImage,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
          SoVulkanShared::imageTransition(
            uploadBuffer, this->whiteImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        })) {
    this->emitError("createWhiteTexture: one-shot upload failed");
    vmaDestroyBuffer(this->vmaAllocator, staging, stagingMemory);
    return false;
  }
  vmaDestroyBuffer(this->vmaAllocator, staging, stagingMemory);

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
  if (!this->createSampler(fallback.minFilter, fallback.magFilter,
                           fallback.wrapS, fallback.wrapT, this->whiteSampler)) {
    return false;
  }
  const bool allocated = this->allocateTextureDescriptorSet(
    this->whiteImageView, this->whiteSampler, this->whiteDescriptorSet);
  return allocated;
}

bool
SoVulkanRenderBackend::createPipelineLayout()
{
  // The visual push-constant block carries per-draw material/texture/line state (proj is
  // in the DrawBlock UBO); at 112 bytes it fits the 128-byte guaranteed minimum.
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

  // Set 0 = lighting constant, set 1 = per-draw UBO + texture.
  const VkDescriptorSetLayout setLayouts[2] = {
    this->lightingSetLayout,
    this->descriptorSetLayout,
  };
  const uint32_t setCount =
    (this->lightingSetLayout != VK_NULL_HANDLE &&
     this->descriptorSetLayout != VK_NULL_HANDLE)
      ? 2u : (this->descriptorSetLayout != VK_NULL_HANDLE ? 1u : 0u);

  VkPipelineLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = setCount;
  ci.pSetLayouts = setLayouts;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &range;
  return vkCreatePipelineLayout(this->device, &ci, this->allocator,
                                &this->pipelineLayout) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createShaderModule(const uint32_t * code, size_t count,
                                          VkShaderModule & module)
{
  VkShaderModuleCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = count * sizeof(uint32_t);
  ci.pCode = code;
  return vkCreateShaderModule(this->device, &ci, this->allocator, &module) ==
    VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createShaders(VkShaderModule & vertex,
                                     VkShaderModule & fragment)
{
  vertex = VK_NULL_HANDLE;
  fragment = VK_NULL_HANDLE;
  if (!this->createShaderModule(coin_vulkan_visual_vertex_spirv,
                                coin_vulkan_visual_vertex_spirv_count,
                                vertex)) {
    return false;
  }
  if (!this->createShaderModule(coin_vulkan_visual_fragment_spirv,
                                coin_vulkan_visual_fragment_spirv_count,
                                fragment)) {
    vkDestroyShaderModule(this->device, vertex, this->allocator);
    vertex = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createWideLineShaders()
{
  if (!this->createShaderModule(coin_vulkan_wide_line_vertex_spirv,
                                coin_vulkan_wide_line_vertex_spirv_count,
                                this->wideLineVertexModule)) {
    return false;
  }
  if (!this->createShaderModule(coin_vulkan_wide_line_fragment_spirv,
                                coin_vulkan_wide_line_fragment_spirv_count,
                                this->wideLineFragmentModule)) {
    vkDestroyShaderModule(this->device, this->wideLineVertexModule,
                          this->allocator);
    this->wideLineVertexModule = VK_NULL_HANDLE;
    return false;
  }
  if (!this->createShaderModule(
        coin_vulkan_wide_line_instanced_vertex_spirv,
        coin_vulkan_wide_line_instanced_vertex_spirv_count,
        this->wideLineInstancedVertexModule)) {
    vkDestroyShaderModule(this->device, this->wideLineFragmentModule,
                          this->allocator);
    vkDestroyShaderModule(this->device, this->wideLineVertexModule,
                          this->allocator);
    this->wideLineFragmentModule = VK_NULL_HANDLE;
    this->wideLineVertexModule = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createSubPixelCullPipeline()
{
  // Descriptor set 0: vertex buffer (0), original index buffer (1), compacted output index
  // buffer (2), indirect command (3).  Input descriptors bind the whole (possibly shared)
  // buffer; the shader applies per-command base offsets via push constants, so no
  // descriptor offset-alignment constraint applies.
  VkDescriptorSetLayoutBinding bindings[4] {};
  for (uint32_t i = 0; i < 4; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[i].pImmutableSamplers = nullptr;
  }
  VkDescriptorSetLayoutCreateInfo slci {};
  slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  slci.bindingCount = 4;
  slci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(this->device, &slci, this->allocator,
                                  &this->subPixelSetLayout) != VK_SUCCESS) {
    this->subPixelSetLayout = VK_NULL_HANDLE;
    return false;
  }

  // Push constants: mat4 mvp (64) + vec4 params (16) + vec4 offsets (16).
  constexpr VkPushConstantRange range {
    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 24
  };
  VkPipelineLayoutCreateInfo plci {};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &this->subPixelSetLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  if (vkCreatePipelineLayout(this->device, &plci, this->allocator,
                             &this->subPixelPipelineLayout) != VK_SUCCESS) {
    this->subPixelPipelineLayout = VK_NULL_HANDLE;
    return false;
  }

  if (!this->createShaderModule(
        coin_vulkan_geometry_lod_subpixel_cull_spirv,
        coin_vulkan_geometry_lod_subpixel_cull_spirv_count,
        this->subPixelCullModule)) {
    return false;
  }

  VkComputePipelineCreateInfo cpci {};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = this->subPixelCullModule;
  cpci.stage.pName = "main";
  cpci.layout = this->subPixelPipelineLayout;
  if (vkCreateComputePipelines(this->device, this->pipelines.handle(), 1,
                               &cpci, this->allocator,
                               &this->subPixelCullPipeline) != VK_SUCCESS) {
    this->subPixelCullPipeline = VK_NULL_HANDLE;
    return false;
  }

  // Dedicated append-only storage-buffer pool: sets are never freed (a frame may reference
  // them); when full, a fresh pool is appended.
  VkDescriptorPoolSize size {};
  size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  size.descriptorCount = 4096 * 4;
  VkDescriptorPoolCreateInfo dpci {};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 4096;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &size;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(this->device, &dpci, this->allocator,
                             &pool) != VK_SUCCESS) {
    return false;
  }
  this->subPixelDescriptorPools.push_back(pool);
  this->subPixelDescriptorSetCount = 0;
  SoVulkanDebugUtils::nameObject(this->device, VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                 reinterpret_cast<uint64_t>(pool),
                                 "Coin raster sub-pixel descriptor pool");

  // Cache the device's single-binding storage-buffer range limit so the pre-pass can
  // reject a command whose vertex/index buffer cannot legally be bound whole.
  VkPhysicalDeviceProperties props {};
  vkGetPhysicalDeviceProperties(this->physicalDevice, &props);
  this->subPixelMaxStorageRange = props.limits.maxStorageBufferRange;
  return true;
}

bool
SoVulkanRenderBackend::createBackgroundResources()
{
  if (!this->createShaderModule(coin_vulkan_background_vertex_spirv,
                                coin_vulkan_background_vertex_spirv_count,
                                this->backgroundVertexModule)) {
    return false;
  }
  if (!this->createShaderModule(coin_vulkan_background_fragment_spirv,
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
