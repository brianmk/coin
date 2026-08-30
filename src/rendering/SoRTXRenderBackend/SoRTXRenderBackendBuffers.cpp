// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendBuffers.cpp

// Split from the original monolithic SoRTXRenderBackend.cpp.  Contains the
// member functions for the "Buffers" concern of the Vulkan RTX backend.

#include "rendering/SoRTXRenderBackend.h"
#include <Inventor/errors/SoDebugError.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>

using namespace SoRTXBackend;

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
SoRTXRenderBackend::createPathTracingBuffers(uint32_t width, uint32_t height)
{
  // Qt can present a transient 0x0 target while the window is being resized;
  // never size path-tracing (or denoiser) buffers to 0 or the allocation
  // degenerates to a zero-size device object (which nvidia returns
  // VK_ERROR_DEVICE_LOST for) and poisons the whole frame.
  if (width == 0 || height == 0) return true;
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
    const VkBuffer accumHist = this->accumHistoryBuffer;
    const VkDeviceMemory accumHistMem = this->accumHistoryMemory;
    const VkBuffer sumSqHist = this->sumSqHistoryBuffer;
    const VkDeviceMemory sumSqHistMem = this->sumSqHistoryMemory;
    const VkBuffer posHist = this->positionHistoryBuffer;
    const VkDeviceMemory posHistMem = this->positionHistoryMemory;
    const VkBuffer motion = this->motionBuffer;
    const VkDeviceMemory motionMem = this->motionMemory;
    this->deferDestroy([device, allocator, accum, accumMem, normal,
                        normalMem, position, positionMem, sumSq, sumSqMem,
                        counter, counterMem, accumHist, accumHistMem,
                        sumSqHist, sumSqHistMem, posHist, posHistMem,
                        motion, motionMem]() {
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
      vkDestroyBuffer(device, accumHist, allocator);
      vkFreeMemory(device, accumHistMem, allocator);
      vkDestroyBuffer(device, sumSqHist, allocator);
      vkFreeMemory(device, sumSqHistMem, allocator);
      vkDestroyBuffer(device, posHist, allocator);
      vkFreeMemory(device, posHistMem, allocator);
      vkDestroyBuffer(device, motion, allocator);
      vkFreeMemory(device, motionMem, allocator);
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
    this->accumHistoryBuffer = VK_NULL_HANDLE;
    this->accumHistoryMemory = VK_NULL_HANDLE;
    this->sumSqHistoryBuffer = VK_NULL_HANDLE;
    this->sumSqHistoryMemory = VK_NULL_HANDLE;
    this->positionHistoryBuffer = VK_NULL_HANDLE;
    this->positionHistoryMemory = VK_NULL_HANDLE;
    this->motionBuffer = VK_NULL_HANDLE;
    this->motionMemory = VK_NULL_HANDLE;
    this->ptHistoryValid = FALSE;
    this->ptReprojectFrame = FALSE;
  }
  this->ptBufferWidth = width;
  this->ptBufferHeight = height;
  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
  // The accumulation buffer doubles as a vkCmdFillBuffer target (fresh
  // progressive runs) and a denoiser-readback source, so it carries both
  // TRANSFER_DST and TRANSFER_SRC.
  const VkBufferUsageFlags accumUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  // The G-buffers (normal/position/albedo/sums-of-squares) are written by the
  // tracer and copied out by the denoiser readback, so they also carry
  // TRANSFER_SRC.
  const VkBufferUsageFlags usage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
  // Screen-space motion-vector G-buffer: written by the tracer, read by the
  // denoiser readback (same TRANSFER_SRC as the other guides).
  if (!this->createDeviceLocalBuffer(bytes, usage, this->motionBuffer,
                                     this->motionMemory)) {
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
    this->motionBuffer = VK_NULL_HANDLE;
    this->motionMemory = VK_NULL_HANDLE;
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
    vkDestroyBuffer(this->device, this->motionBuffer, this->allocator);
    vkFreeMemory(this->device, this->motionMemory, this->allocator);
    this->positionBuffer = VK_NULL_HANDLE;
    this->positionMemory = VK_NULL_HANDLE;
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = VK_NULL_HANDLE;
    this->motionBuffer = VK_NULL_HANDLE;
    this->motionMemory = VK_NULL_HANDLE;
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
  // Temporal reprojection history.  The accumulation and sums-of-squares
  // history buffers also carry TRANSFER_DST: after a swap they can become
  // the live fill targets for the next fresh run.
  if (!this->createDeviceLocalBuffer(bytes, accumUsage,
                                     this->accumHistoryBuffer,
                                     this->accumHistoryMemory) ||
      !this->createDeviceLocalBuffer(bytes, accumUsage,
                                     this->sumSqHistoryBuffer,
                                     this->sumSqHistoryMemory) ||
      !this->createDeviceLocalBuffer(bytes, usage,
                                     this->positionHistoryBuffer,
                                     this->positionHistoryMemory)) {
    vkDestroyBuffer(this->device, this->accumHistoryBuffer, this->allocator);
    vkFreeMemory(this->device, this->accumHistoryMemory, this->allocator);
    vkDestroyBuffer(this->device, this->sumSqHistoryBuffer, this->allocator);
    vkFreeMemory(this->device, this->sumSqHistoryMemory, this->allocator);
    vkDestroyBuffer(this->device, this->positionHistoryBuffer, this->allocator);
    vkFreeMemory(this->device, this->positionHistoryMemory, this->allocator);
    vkDestroyBuffer(this->device, this->activeCounterBuffer, this->allocator);
    vkFreeMemory(this->device, this->activeCounterMemory, this->allocator);
    vkDestroyBuffer(this->device, this->sumSqBuffer, this->allocator);
    vkFreeMemory(this->device, this->sumSqMemory, this->allocator);
    vkDestroyBuffer(this->device, this->positionBuffer, this->allocator);
    vkFreeMemory(this->device, this->positionMemory, this->allocator);
    vkDestroyBuffer(this->device, this->normalBuffer, this->allocator);
    vkFreeMemory(this->device, this->normalMemory, this->allocator);
    vkDestroyBuffer(this->device, this->motionBuffer, this->allocator);
    vkFreeMemory(this->device, this->motionMemory, this->allocator);
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->accumHistoryBuffer = VK_NULL_HANDLE;
    this->accumHistoryMemory = VK_NULL_HANDLE;
    this->sumSqHistoryBuffer = VK_NULL_HANDLE;
    this->sumSqHistoryMemory = VK_NULL_HANDLE;
    this->positionHistoryBuffer = VK_NULL_HANDLE;
    this->positionHistoryMemory = VK_NULL_HANDLE;
    this->activeCounterBuffer = VK_NULL_HANDLE;
    this->activeCounterMemory = VK_NULL_HANDLE;
    this->activeCounterMapped = nullptr;
    this->sumSqBuffer = VK_NULL_HANDLE;
    this->sumSqMemory = VK_NULL_HANDLE;
    this->positionBuffer = VK_NULL_HANDLE;
    this->positionMemory = VK_NULL_HANDLE;
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = VK_NULL_HANDLE;
    this->motionBuffer = VK_NULL_HANDLE;
    this->motionMemory = VK_NULL_HANDLE;
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  // Fresh buffers: refresh the descriptor sets so the new handles are
  // visible to the trace and present passes.
  if (!this->updateDescriptors()) return false;

  // Set up the denoiser backend for the new resolution: staging buffers for
  // the G-buffer readback plus the device-local denoised output, and the
  // OIDN/RTX/FSR device+filter themselves.  When a denoiser is active the
  // raygen writes an albedo G-buffer (binding 14) that createDenoiseBackend()
  // also allocates and uploads for the albedo guide.
  this->denoiseWidth = width;
  this->denoiseHeight = height;
  if (!this->createDenoiseBackend()) {
    this->emitError("createPathTracingBuffers: failed to create denoiser backend");
    if (this->queue != VK_NULL_HANDLE) {
      vkQueueWaitIdle(this->queue);
    }
    this->destroyDenoiser();

    if (this->activeCounterMapped != nullptr) {
      vkUnmapMemory(this->device, this->activeCounterMemory);
      this->activeCounterMapped = nullptr;
    }
    auto freeBuffer = [this](VkBuffer & buffer, VkDeviceMemory & memory) {
      if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(this->device, buffer, this->allocator);
        buffer = VK_NULL_HANDLE;
      }
      if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(this->device, memory, this->allocator);
        memory = VK_NULL_HANDLE;
      }
    };
    freeBuffer(this->accumBuffer, this->accumMemory);
    freeBuffer(this->normalBuffer, this->normalMemory);
    freeBuffer(this->positionBuffer, this->positionMemory);
    freeBuffer(this->sumSqBuffer, this->sumSqMemory);
    freeBuffer(this->activeCounterBuffer, this->activeCounterMemory);
    freeBuffer(this->accumHistoryBuffer, this->accumHistoryMemory);
    freeBuffer(this->sumSqHistoryBuffer, this->sumSqHistoryMemory);
    freeBuffer(this->positionHistoryBuffer, this->positionHistoryMemory);
    freeBuffer(this->motionBuffer, this->motionMemory);
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    this->ptHistoryValid = FALSE;
    this->ptReprojectFrame = FALSE;

    const VkDescriptorSet descriptorSets[] = {
      this->rtDescriptorSets[0], this->rtDescriptorSets[1],
      this->presentDescriptorSets[0], this->presentDescriptorSets[1]
    };
    size_t descriptorSetCount = 0;
    while (descriptorSetCount < sizeof(descriptorSets) /
                                  sizeof(descriptorSets[0]) &&
           descriptorSets[descriptorSetCount] != VK_NULL_HANDLE) {
      ++descriptorSetCount;
    }
    if (this->descriptorPool != VK_NULL_HANDLE && descriptorSetCount > 0) {
      vkFreeDescriptorSets(this->device, this->descriptorPool,
                           static_cast<uint32_t>(descriptorSetCount),
                           descriptorSets);
    }
    this->rtDescriptorSets[0] = VK_NULL_HANDLE;
    this->rtDescriptorSets[1] = VK_NULL_HANDLE;
    this->presentDescriptorSets[0] = VK_NULL_HANDLE;
    this->presentDescriptorSets[1] = VK_NULL_HANDLE;
    if (!this->updateDescriptors()) {
      this->emitError(
        "createPathTracingBuffers: failed to refresh descriptors after "
        "denoiser backend failure");
    }
    return false;
  }
  return true;
}
