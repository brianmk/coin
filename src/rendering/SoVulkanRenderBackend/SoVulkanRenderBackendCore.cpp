// src/rendering/SoVulkanRenderBackendCore.cpp

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include "rendering/vulkan/visual/Fragment.spv.h"
#include "rendering/vulkan/visual/Vertex.spv.h"
#include "rendering/vulkan/visual/WideLineFragment.spv.h"
#include "rendering/vulkan/visual/WideLineVertex.spv.h"
#include "rendering/vulkan/visual/BackgroundVertex.spv.h"
#include "rendering/vulkan/visual/BackgroundFragment.spv.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace CoinVulkanDetail;

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
