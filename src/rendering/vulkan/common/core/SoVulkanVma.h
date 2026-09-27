// src/rendering/vulkan/common/core/SoVulkanVma.h
//
// Shared VMA-backed buffer creation for the raster and ray-tracing backends.
// One implementation replaces the VkBufferCreateInfo / VmaAllocationCreateInfo
// boilerplate (and the identical failure reporting) previously copied into
// every buffer factory.  Internal to Coin's Vulkan renderer (not installed,
// not public API); header-only so a translation unit that does not create
// buffers pays nothing for it.

#ifndef COIN_SOVULKANVMA_H
#define COIN_SOVULKANVMA_H

#include <string>

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

#include "rendering/vulkan/common/core/SoVulkanResult.h"
#include "rendering/vulkan/common/core/SoVulkanShared.h"

namespace SoVulkanShared {

// The parts of a VmaAllocation-backed VkBuffer that vary between call sites.
// sharingMode is always EXCLUSIVE and the memory usage is always
// VMA_MEMORY_USAGE_AUTO (the renderer's single policy), so neither is exposed.
struct VmaBufferDesc {
  VkDeviceSize size = 0;
  VkBufferUsageFlags usage = 0;
  VkMemoryPropertyFlags requiredFlags = 0;
  VmaAllocationCreateFlags flags = 0;
  VmaPool pool = VK_NULL_HANDLE;
  // Optional pNext for the VkBufferCreateInfo (e.g. an
  // VkExternalMemoryBufferCreateInfo that the CUDA interop buffers chain).
  const void * pNext = nullptr;
};

// Create `buffer`/`allocation` backed by VMA.  On success the optional
// `allocationInfo` receives VMA's report (persistent host pointer, device
// memory handle and size live there).  On any failure nothing is left
// allocated -- buffer and allocation are reset to null -- and a Result naming
// `context` and the failing VkResult is returned.
inline SoVulkan::Result
createVmaBuffer(VmaAllocator allocator, const VmaBufferDesc & desc,
                const char * context, VkBuffer & buffer,
                VmaAllocation & allocation,
                VmaAllocationInfo * allocationInfo = nullptr)
{
  buffer = VK_NULL_HANDLE;
  allocation = nullptr;

  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.pNext = desc.pNext;
  bci.size = desc.size;
  bci.usage = desc.usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = desc.requiredFlags;
  allocInfo.flags = desc.flags;
  allocInfo.pool = desc.pool;

  VmaAllocationInfo localInfo {};
  const VkResult res = vmaCreateBuffer(
    allocator, &bci, &allocInfo, &buffer, &allocation, &localInfo);
  if (res != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    allocation = nullptr;
    return SoVulkan::Result::error(std::string(context)
                                   + ": vmaCreateBuffer failed: "
                                   + vkResultName(res));
  }
  if (allocationInfo) {
    *allocationInfo = localInfo;
  }
  return SoVulkan::Result::ok();
}

} // namespace SoVulkanShared

#endif // COIN_SOVULKANVMA_H
