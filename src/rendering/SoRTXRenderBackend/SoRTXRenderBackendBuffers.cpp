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
#include <rendering/SoVulkanConfig.h>
#include <rendering/SoVulkanDebugUtils.h>
#include <rendering/SoVulkanShared.h>

#include "vk_mem_alloc.h"

using namespace SoRTXBackend;

bool
SoRTXRenderBackend::createDeviceLocalBuffer(VkDeviceSize size,
                                            VkBufferUsageFlags usage,
                                            VkBuffer & buffer,
                                            VmaAllocation & memory)
{
  buffer = VK_NULL_HANDLE;
  memory = nullptr;
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  return true;
}

// Host-visible + host-coherent buffer (frame UBO, material buffer, instances,
// SBT, staging uploads).
bool
SoRTXRenderBackend::createHostVisibleBuffer(VkDeviceSize size,
                                            VkBufferUsageFlags usage,
                                            VkBuffer & buffer,
                                            VmaAllocation & memory,
                                            void ** mapped)
{
  buffer = VK_NULL_HANDLE;
  memory = nullptr;
  if (mapped) *mapped = nullptr;
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  // Host-visible RTX buffers are written (and some read back) through a
  // persistent map.  VMA_MEMORY_USAGE_AUTO requires an explicit host-access
  // flag; VMA forbids combining SEQUENTIAL_WRITE with RANDOM, and RANDOM
  // already covers the sequential-write case.  MAPPED establishes the
  // persistent mapping as part of the allocation, so callers must not call
  // vmaMapMemory (VMA asserts if a user mapping outlives the allocation).
  allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  if (mapped) *mapped = allocationInfo.pMappedData;
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
  if (SoVulkanConfig::get().rtxDebug.rtDebug) {
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
    const VmaAllocation memory = this->storageImageMemory;
    VmaAllocator vma = this->vmaAllocator;
    this->deferDestroy([vma, device, allocator, image, view, memory]() {
      vkDestroyImageView(device, view, allocator);
      vmaDestroyImage(vma, image, memory);
    });
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageView = VK_NULL_HANDLE;
    this->storageImageMemory = nullptr;
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
  // The storage image is read back (TRANSFER_SRC for the frame-dump and
  // denoiser readback), sampled by the present pass (SAMPLED), written by the
  // raygen (STORAGE), and -- on frames with no traceable geometry (tlas ==
  // VK_NULL_HANDLE) -- filled with the background colour via
  // vkCmdClearColorImage.  The clear requires TRANSFER_DST_BIT, which is the
  // only usage it does not already have.
  ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateImage(this->vmaAllocator, &ci, &allocInfo, &this->storageImage,
                     &this->storageImageMemory,
                     &allocationInfo) != VK_SUCCESS) {
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageMemory = nullptr;
    this->storageWidth = 0;
    this->storageHeight = 0;
    return false;
  }

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
    vmaDestroyImage(this->vmaAllocator, this->storageImage,
                    this->storageImageMemory);
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageMemory = nullptr;
    this->storageWidth = 0;
    this->storageHeight = 0;
    return false;
  }

  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_IMAGE,
    reinterpret_cast<uint64_t>(this->storageImage), "RT storage image");
  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_IMAGE_VIEW,
    reinterpret_cast<uint64_t>(this->storageImageView),
    "RT storage image view");

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
    vmaDestroyImage(this->vmaAllocator, this->storageImage,
                    this->storageImageMemory);
    this->storageImageView = VK_NULL_HANDLE;
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageMemory = nullptr;
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
      this->ptBufferWidth == width && this->ptBufferHeight == height &&
      !this->denoiseKindDirty) {
    // Same resolution and no denoiser change: nothing to rebuild.  A
    // denoiseKindDirty set by setDenoiserFilter()/setDenoiserScale() MUST fall
    // through here: otherwise the early return skips createDenoiseBackend()
    // below, the newly selected denoiser never builds its filter/staging, and
    // the view keeps the previous denoiser's denoisedBuffer (binding 5) on
    // screen -- the stale/black image a runtime OIDN<->RTX switch produced.
    return true;
  }
  // Release the old buffers (deferred: the previous frame's submission may
  // still be executing); new ones are sized to the current viewport.
  if (this->accumBuffer != VK_NULL_HANDLE) {
    VmaAllocator vma = this->vmaAllocator;
    const VkBuffer accum = this->accumBuffer;
    const VmaAllocation accumMem = this->accumMemory;
    const VkBuffer normal = this->normalBuffer;
    const VmaAllocation normalMem = this->normalMemory;
    const VkBuffer position = this->positionBuffer;
    const VmaAllocation positionMem = this->positionMemory;
    const VkBuffer sumSq = this->sumSqBuffer;
    const VmaAllocation sumSqMem = this->sumSqMemory;
    const VkBuffer counter = this->activeCounterBuffer;
    const VmaAllocation counterMem = this->activeCounterMemory;
    const VkBuffer accumHist = this->accumHistoryBuffer;
    const VmaAllocation accumHistMem = this->accumHistoryMemory;
    const VkBuffer sumSqHist = this->sumSqHistoryBuffer;
    const VmaAllocation sumSqHistMem = this->sumSqHistoryMemory;
    const VkBuffer posHist = this->positionHistoryBuffer;
    const VmaAllocation posHistMem = this->positionHistoryMemory;
    const VkBuffer motion = this->motionBuffer;
    const VmaAllocation motionMem = this->motionMemory;
    this->deferDestroy([vma, accum, accumMem, normal,
                        normalMem, position, positionMem, sumSq, sumSqMem,
                        counter, counterMem, accumHist, accumHistMem,
                        sumSqHist, sumSqHistMem, posHist, posHistMem,
                        motion, motionMem]() {
      vmaDestroyBuffer(vma, accum, accumMem);
      vmaDestroyBuffer(vma, normal, normalMem);
      vmaDestroyBuffer(vma, position, positionMem);
      vmaDestroyBuffer(vma, sumSq, sumSqMem);
      vmaDestroyBuffer(vma, counter, counterMem);
      vmaDestroyBuffer(vma, accumHist, accumHistMem);
      vmaDestroyBuffer(vma, sumSqHist, sumSqHistMem);
      vmaDestroyBuffer(vma, posHist, posHistMem);
      vmaDestroyBuffer(vma, motion, motionMem);
    });
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = nullptr;
    this->normalBuffer = VK_NULL_HANDLE;
    this->normalMemory = nullptr;
    this->positionBuffer = VK_NULL_HANDLE;
    this->positionMemory = nullptr;
    this->sumSqBuffer = VK_NULL_HANDLE;
    this->sumSqMemory = nullptr;
    this->activeCounterBuffer = VK_NULL_HANDLE;
    this->activeCounterMemory = nullptr;
    this->activeCounterMapped = nullptr;
    this->accumHistoryBuffer = VK_NULL_HANDLE;
    this->accumHistoryMemory = nullptr;
    this->sumSqHistoryBuffer = VK_NULL_HANDLE;
    this->sumSqHistoryMemory = nullptr;
    this->positionHistoryBuffer = VK_NULL_HANDLE;
    this->positionHistoryMemory = nullptr;
    this->motionBuffer = VK_NULL_HANDLE;
    this->motionMemory = nullptr;
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
    vmaDestroyBuffer(this->vmaAllocator, this->accumBuffer, this->accumMemory);
    this->accumBuffer = VK_NULL_HANDLE;
    this->accumMemory = VK_NULL_HANDLE;
    this->ptBufferWidth = 0;
    this->ptBufferHeight = 0;
    return false;
  }
  if (!this->createDeviceLocalBuffer(bytes, usage, this->positionBuffer,
                                     this->positionMemory)) {
    vmaDestroyBuffer(this->vmaAllocator, this->normalBuffer, this->normalMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->accumBuffer, this->accumMemory);
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
    vmaDestroyBuffer(this->vmaAllocator, this->positionBuffer, this->positionMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->normalBuffer, this->normalMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->accumBuffer, this->accumMemory);
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
    vmaDestroyBuffer(this->vmaAllocator, this->positionBuffer, this->positionMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->normalBuffer, this->normalMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->accumBuffer, this->accumMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->motionBuffer, this->motionMemory);
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
        this->activeCounterBuffer, this->activeCounterMemory,
        &this->activeCounterMapped)) {
    vmaDestroyBuffer(this->vmaAllocator, this->sumSqBuffer, this->sumSqMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->positionBuffer, this->positionMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->normalBuffer, this->normalMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->accumBuffer, this->accumMemory);
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
    vmaDestroyBuffer(this->vmaAllocator, this->accumHistoryBuffer, this->accumHistoryMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->sumSqHistoryBuffer, this->sumSqHistoryMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->positionHistoryBuffer, this->positionHistoryMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->activeCounterBuffer, this->activeCounterMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->sumSqBuffer, this->sumSqMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->positionBuffer, this->positionMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->normalBuffer, this->normalMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->motionBuffer, this->motionMemory);
    vmaDestroyBuffer(this->vmaAllocator, this->accumBuffer, this->accumMemory);
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
  //
  // denoiseWidth/Height are the DENOISER's internal resolution, not the full
  // viewport: a denoiseScale > 1 (setDenoiserScale) runs the filter at
  // reduced resolution and the present pass upscales the result.  The host
  // readback still stages the full-resolution G-buffers and the worker
  // downsamples them (see recordDenoiseReadback / updateDenoise); only the
  // host-side OIDN/FSR backends support scaling -- the RTX interop path stays
  // native (scale 1) because it reads the G-buffers device-to-device and
  // needs a GPU downsample that does not exist there.  The FSR/DNSR pass is
  // also device-local and native-resolution (it filters the full-res
  // G-buffers), so it stays scale 1 as well.
  const float scale = (this->denoiseKindPref != DenoiseRtx &&
                       this->denoiseKindPref != DenoiseFsr)
    ? this->denoiseScale : 1.0f;
  this->denoiseEffectiveScale = scale;
  this->denoiseWidth = std::max(1u, static_cast<uint32_t>(
    std::ceil(static_cast<double>(width) / (scale > 1.5f ? scale : 1.0f))));
  this->denoiseHeight = std::max(1u, static_cast<uint32_t>(
    std::ceil(static_cast<double>(height) / (scale > 1.5f ? scale : 1.0f))));
  if (!this->createDenoiseBackend()) {
    this->emitError("createPathTracingBuffers: failed to create denoiser backend");
    if (this->queue != VK_NULL_HANDLE) {
      vkQueueWaitIdle(this->queue);
    }
    this->destroyDenoiser();

    if (this->activeCounterMapped != nullptr) {
      vmaUnmapMemory(this->vmaAllocator, this->activeCounterMemory);
      this->activeCounterMapped = nullptr;
    }
    auto freeBuffer = [this](VkBuffer & buffer, VmaAllocation & memory) {
      if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(this->vmaAllocator, buffer, memory);
        buffer = VK_NULL_HANDLE;
        memory = nullptr;
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

    VkDescriptorSet descriptorSets[RTX_MAX_FRAMES_IN_FLIGHT * 2];
    uint32_t descriptorSetCount = 0;
    for (uint32_t i = 0; i < this->descriptorRingSize; ++i) {
      if (this->rtDescriptorSets[i] != VK_NULL_HANDLE) {
        descriptorSets[descriptorSetCount++] = this->rtDescriptorSets[i];
        this->rtDescriptorSets[i] = VK_NULL_HANDLE;
        this->rtSetValid[i] = false;
      }
      if (this->presentDescriptorSets[i] != VK_NULL_HANDLE) {
        descriptorSets[descriptorSetCount++] = this->presentDescriptorSets[i];
        this->presentDescriptorSets[i] = VK_NULL_HANDLE;
        this->presentSetValid[i] = false;
      }
    }
    if (this->descriptorPool != VK_NULL_HANDLE && descriptorSetCount > 0) {
      vkFreeDescriptorSets(this->device, this->descriptorPool,
                           descriptorSetCount, descriptorSets);
    }
    if (!this->updateDescriptors()) {
      this->emitError(
        "createPathTracingBuffers: failed to refresh descriptors after "
        "denoiser backend failure");
    }
    return false;
  }
  return true;
}
