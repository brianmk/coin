// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendCore.cpp
//
// SoVulkanRenderBackend initialization and lifecycle.  Provides:
//
//   - Constructor/destructor, getName() and the overlay setters
//   - setMaxFramesInFlight(): ring + lighting-UBO resize with a queue-wide
//     wait
//   - initialize() and its resource-create helpers: command pool, descriptor
//     set layout/pool, per-frame command buffers/fences, the lighting uniform
//     ring buffer (create/grow/swap), the white fallback texture, the
//     pipeline layout, and the visual / wide-line / background shader modules
//   - Frame-slot + deferred-destroy bookkeeping (beginFrame,
//     flushPendingDestroys, flushAllPendingDestroys,
//     deferDestroy[Cache|Texture]Entry, waitForInFlightFrames)

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanShared.h"
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanDebugUtils.h"

#include "vk_mem_alloc.h"

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

  // Resizing while submissions are still in flight would free command
  // buffers/fences that a pending submission references and orphan the ring
  // batches below the new size; wait for every pending submission first.
  if (this->isInitialized()) {
    this->waitForInFlightFrames();
  }

  const uint32_t previousCount = this->maxFramesInFlight;
  this->maxFramesInFlight = count;
  this->pendingDestroys.setBatchCount(count);

  // Re-allocate the per-frame-slot command buffers/fences at the new count.
  // Only valid while the queue is idle (guaranteed by the wait above).
  if (this->isInitialized()) {
    this->releaseFrameResources();
    if (!this->allocateFrameResources()) {
      // allocateFrameResources() cleaned up its partial allocations and left
      // the frame ring empty.  Roll back to the previous count and rebuild so
      // the frame loop still has command buffers/fences to record into; a
      // rollback that also fails disables the backend rather than letting
      // beginCommandBuffer() call vkBeginCommandBuffer(VK_NULL_HANDLE).
      this->emitError(
        "setMaxFramesInFlight: failed to reallocate frame resources");
      this->maxFramesInFlight = previousCount;
      this->pendingDestroys.setBatchCount(previousCount);
      if (!this->allocateFrameResources()) {
        this->emitError(
          "setMaxFramesInFlight: rollback failed; shutting down backend");
        this->shutdown();
        return;
      }
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
  // Resolve the synchronization2 entry points once for this device.  A null
  // pointer means the extension was not enabled; the shared barrier/submit
  // helpers then fall back to the legacy entry points.
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

  // Bind the render-pass/framebuffer cache to this device and hook its
  // deferred resource release into the frame ring: an old framebuffer is
  // destroyed a few frames after the submission that referenced it completes,
  // rather than synchronously (which would race a still-executing frame).
  this->renderPasses.setDevice(this->device, this->allocator);
  this->renderPasses.setDeferredDestroy([this](std::function<void()> && fn) {
    this->deferDestroy(std::move(fn));
  });

  // Vulkan Memory Allocator: owns the texture-image device memory.  The
  // allocator sub-allocates from large blocks, so per-texture creation does
  // not hit the driver (and maxMemoryAllocationCount) once per upload.
  {
    VmaAllocatorCreateInfo allocatorInfo {};
    allocatorInfo.physicalDevice = this->physicalDevice;
    allocatorInfo.device = this->device;
    allocatorInfo.instance = this->instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorInfo.pAllocationCallbacks = this->allocator;
    if (vmaCreateAllocator(&allocatorInfo, &this->vmaAllocator) != VK_SUCCESS) {
      this->emitError("SoVulkanRenderBackend: vmaCreateAllocator failed");
      return FALSE;
    }
  }

  // Worker count for the persistent record pool.  The pool is also used by the
  // parallel wide-line expansion pre-pass, so it is sized whenever the machine
  // has cores; parallel *recording* (M1d) stays opt-in so command-buffer
  // output is identical to the serial path unless enabled.  Worker 0 is the
  // recording thread, workers 1..N-1 are spawned.
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

  // Cache the device capabilities the backend relies on.  Vulkan has no API
  // to read back which features an already-created device enabled, so query
  // what the physical device *supports* and rely on the embedding
  // application enabling exactly the supported ones (see
  // QuarterVulkanWidget::configureDeviceFeatures).  This gates the
  // VK_POLYGON_MODE_LINE/POINT pipelines (fillModeNonSolid) and the 1/2-
  // component texture upload formats (optional sampled formats).
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
    // Geometry LOD is an optional acceleration: a device without compute (or
    // with too small a push-constant budget) simply keeps the full-draw path.
    // The pipeline stays null and the pre-pass no-ops.
    this->emitLog("geometry-LOD compute pipeline unavailable; full draws only");
  }

  this->emitLog("initialized");
  return TRUE;
}

bool
SoVulkanRenderBackend::createPipelineCache()
{
  // The pipeline store owns the VkPipelineCache handle and its persistence;
  // bind the device/allocator and route its messages through this backend's
  // log callback before creating the handle.
  this->pipelines.setDevice(this->device, this->allocator);
  this->pipelines.setLogger(
    [this](const char * message) { this->emitLog(message); });
  return this->pipelines.initialize();
}

void
SoVulkanRenderBackend::setPipelineCachePath(const std::string & path)
{
  // Key the persisted cache to the compiled shaders.  The pipeline-state key
  // (PipelineKey) does not capture shader code, so without this a rebuilt
  // shader would be served a pipeline compiled from the previous one -- e.g.
  // a stale projection/push-constant layout, which shows up as displaced
  // edges on the first frame.  A shader change yields a new key and the old
  // blob is rejected.
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
  // Secondary pools for the M1c/M1d opaque-pass re-record: same
  // transient/reset flags as the primary pool, secondary-level buffers.  One
  // pool per worker (worker 0 = the recording thread): VkCommandPool host
  // access is externally synchronized, so concurrent reset/begin/end of
  // buffers from a SHARED pool would race its internal allocator.
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
  // Resume from the current pool size so a partially-built pool (spawn failed
  // midway) tops up rather than spawning duplicate worker indices.
  for (uint32_t w = static_cast<uint32_t>(this->recordWorkers.size()) + 1;
       w < this->maxRecordWorkers; ++w) {
    try {
      this->recordWorkers.emplace_back(
        &SoVulkanRenderBackend::recordJobWorker, this, static_cast<size_t>(w));
    }
    catch (const std::exception &) {
      // Could not spawn: fall back to serial recording (worker 0 only) and
      // retire any threads already spawned.
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
      // Wide-line CPU expansion: no command buffer, just the per-command quad
      // computation.  Each command's cache entry is touched by exactly one
      // worker, and the scratch is thread-local, so this is race-free.
      if (job.wlineSplitPhase != 0) {
        // One command partitioned by segment range: the shared scratch is
        // owner-sized and every range writes disjoint slots.
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
    // The done count is the condition the recording thread waits on
    // (recordCvDone.wait), so it must be published under recordMutex together
    // with the notify.  Incrementing/notifying outside the lock races the
    // waiter's final predicate check: the notify can fire after the waiter
    // evaluated the predicate but before it blocks, and is lost, leaving the
    // recording thread asleep forever with the count already satisfied.
    //
    // Publish only while this worker's generation is still current.  If the
    // recording thread already dispatched the next job while this one ran, this
    // completion belongs to a superseded generation; counting it would let the
    // next join see a spurious done and proceed before every worker has run.
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
    // Drop whatever this call managed to allocate so a caller that continues
    // (setMaxFramesInFlight) cannot observe a half-populated frame ring.
    this->releaseFrameResources();
    return false;
  }

  // One secondary command buffer per in-flight slot per worker (M1c/M1d),
  // each allocated from that worker's own pool.  Recorded once per slot and
  // executed into that slot's primary, then re-recorded only after the slot's
  // fence is waited.  Indexed [slot * maxRecordWorkers + worker] (the layout
  // workerSecondary() reads), so allocate worker w's buffer into every
  // [slot][w] position.
  const uint32_t secondaryCount =
    this->maxFramesInFlight * this->maxRecordWorkers;
  this->secondaryCommandBuffers.assign(secondaryCount, VK_NULL_HANDLE);
  for (uint32_t w = 0; w < this->maxRecordWorkers; ++w) {
    if (w >= this->secondaryCommandPools.size() ||
        this->secondaryCommandPools[w] == VK_NULL_HANDLE) {
      this->releaseFrameResources();
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
      this->releaseFrameResources();
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
  // The pool serves both parallel recording and the wide-line expansion
  // pre-pass, so build it whenever there is more than one worker.
  if (this->maxRecordWorkers > 1) {
    if (!this->buildRecordPool()) {
      this->releaseFrameResources();
      return false;
    }
  }

  VkFenceCreateInfo fi {};
  fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (VkFence & fence : this->frameFences) {
    if (vkCreateFence(this->device, &fi, this->allocator, &fence) !=
        VK_SUCCESS) {
      this->releaseFrameResources();
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
  // Set 0: lighting constant ring (binding 0, UBO dynamic).  Visible to both
  // stages because lighting is evaluated per fragment (Phong).
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

  // Set 1: per-draw view/model/material UBO (binding 0, dynamic) plus the
  // embedded texture (binding 1).
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
  // Per-command slots in a ring buffer sized for maxFramesInFlight frames.
  // Each draw binds its slot with a dynamic offset, so the GPU reads the
  // uniform block that was recorded for that specific draw instead of a
  // shared buffer that later commands overwrite.
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
  // The per-instance model-matrix ring parallels the lighting UBO ring
  // (same slot layout), so pre-size it here; the per-draw path must never
  // grow (re-create) this buffer, which would race under parallel recording.
  if (!this->ensureInstanceModelRingCapacity()) {
    this->emitError("createLightingUniformBuffer: failed to size instance buffer");
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createLightingConstBuffer()
{
  // A few slots per in-flight frame, one per distinct lighting handle the
  // frame references (typically one).  Each slot holds one VulkanLightingUbo
  // (ambient + 8 lights); writing it once per handle per frame is what makes
  // the shared lighting setup cost O(#handles) instead of O(#draws).
  VkPhysicalDeviceProperties deviceProps;
  vkGetPhysicalDeviceProperties(this->physicalDevice, &deviceProps);
  const VkDeviceSize alignment = std::max<VkDeviceSize>(
    1, deviceProps.limits.minUniformBufferOffsetAlignment);
  this->lightingConstStride =
    (sizeof(VulkanLightingUbo) + alignment - 1) / alignment * alignment;
  // Fixed 8-frame ring with 8 unique-handle slots per frame.  This is
  // independent of maxFramesInFlight (which can change after init in
  // initSwapChainResources) so no resize path is needed; it is safe as long
  // as the in-flight frame count stays <= 8 (QVulkanWindow swapchains are
  // 2-3 images, +1 margin).
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

  // The ring was (re)sized to a new slot count or a new in-flight frame count;
  // grow the per-instance model-matrix ring to the same geometry so the
  // per-draw path never grows it (see ensureInstanceModelRingCapacity()).
  if (!this->ensureInstanceModelRingCapacity()) {
    this->emitError("swapLightingBuffer: failed to size instance buffer");
  }

  // The old buffer may still be referenced by a pending frame; destroy it
  // only after the batch ring wraps back around (flushPendingDestroys()).
  // Freeing the memory implicitly unmaps it, so no explicit unmap is needed.
  this->deferDestroyBufferMemory(oldBuffer, oldMemory);

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
  // Reserve up front: collect() stores &bufferInfos.back() into each write's
  // pBufferInfo.  Without a reservation a later push_back reallocates the
  // vector and turns those earlier pointers into dangling memory, so
  // vkUpdateDescriptorSets would read a garbage VkBuffer handle and fault in
  // the driver.  The max set count is the white set plus one per texture.
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
  // The frame index was advanced by beginFrame(); every render starts from
  // slot zero of its own ring half.  The slot cursor now lives in the
  // per-recording VulkanRecordContext (reset by its frame-start reset()).
  return true;
}

void
SoVulkanRenderBackend::beginFrame()
{
  vkBackendTrace(this->uboFrameIndex, "beginFrame.enter",
                 "nextFrame=%u", this->uboFrameIndex + 1);
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
    // A failed wait (e.g. VK_ERROR_DEVICE_LOST) must not be swallowed: the
    // slot's command buffer/resources may still be in flight, so leave the
    // pending flag set and report the fault rather than reusing them silently.
    const VkResult waitRes = vkWaitForFences(
      this->device, 1, &this->frameFences[slot], VK_TRUE, UINT64_MAX);
    if (waitRes != VK_SUCCESS) {
      this->emitError("beginFrame: vkWaitForFences failed on frame slot");
    }
    else if (vkResetFences(this->device, 1, &this->frameFences[slot]) !=
             VK_SUCCESS) {
      this->emitError("beginFrame: vkResetFences failed on frame slot");
    }
    else {
      this->frameFencePending[slot] = 0;
    }
  }
  // Dynamic state is per-recording: the previous frame's command buffer (the
  // other slot) may have left a different pipeline/viewport/scissor bound.
  // Forget it so this frame's first apply* emits, and so accidental reuse
  // from the slot's prior content cannot wrongly skip a needed change.
  this->resetBoundState(this->recordContext);
  this->flushPendingDestroys();
}

void
SoVulkanRenderBackend::cacheFrameMatrices(const SoRenderParams & params)
{
  // SbMatrix stores exactly float[4][4], so the raw storage IS the float
  // matrix: copying the 16 floats once per render lets every draw reuse the
  // result instead of re-converting per draw.
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
  const VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  const VmaAllocator vmaAllocator = this->vmaAllocator;
  const VkImage image = entry.image;
  const VmaAllocation allocation = entry.allocation;
  const VkImageView view = entry.view;
  // The sampler is shared (samplerCache), so it is NOT destroyed here; it is
  // released once at shutdown() after the texture cache has been emptied.  The
  // image and its VMA allocation are freed together, once the frame's
  // submission is complete (the deferred ring), so no in-flight reference is
  // aliased.
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

  // One-shot upload: stage buffer -> image (white 1x1), transitioning the image
  // UNDEFINED -> TRANSFER_DST -> SHADER_READ_ONLY.  The shared helper waits for
  // the queue to go idle so the staging buffer below is safe to destroy.
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
  // The visual push-constant block carries the per-draw material/texture/line
  // state (the projection matrix lives in the DrawBlock UBO).  At 112 bytes it
  // fits the 128-byte Vulkan guaranteed minimum, so even minimum-spec devices
  // (and the desktop-baseline device profiles) can create the pipeline.
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
  // Descriptor set 0: the command's vertex buffer (0), original index buffer
  // (1), compacted output index buffer (2) and the indirect command (3).  The
  // input descriptors bind the whole (possibly shared) buffer; the shader
  // applies the per-command base offset from its push constants, so no
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

  // Dedicated, append-only storage-buffer descriptor pool.  Sets are never
  // freed while a frame may reference them; when the pool fills a fresh one is
  // appended (mirroring descriptorPools).
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

  // Cache the device's single-binding storage-buffer range limit so the
  // pre-pass can reject a command whose vertex/index buffer cannot legally be
  // bound whole (see subPixelMaxStorageRange in the header).
  VkPhysicalDeviceProperties props {};
  vkGetPhysicalDeviceProperties(this->physicalDevice, &props);
  this->subPixelMaxStorageRange = props.limits.maxStorageBufferRange;
  return true;
}

bool
SoVulkanRenderBackend::createBackgroundResources()
{
  if (SoVulkanShared::envFlagEnabled("FC_VULKAN_BREADCRUMBS")) {
    fprintf(stderr, "[VK-TRACE] SoVulkanRenderBackend::createBackgroundResources enter\n");
  }
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
