// src/rendering/SoVulkanStagingPool.cpp
#include "rendering/SoVulkanStagingPool.h"

#include "rendering/SoVulkanDebugUtils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

void
SoVulkanStagingPool::initialize(VkDevice device, VmaAllocator allocator)
{
  this->device_ = device;
  this->allocator_ = allocator;
}

bool
SoVulkanStagingPool::ensureCapacity(VkDeviceSize required)
{
  // The caller needs `required` MORE bytes at the current cursor: the pool
  // must fit cursor + required, not merely `required` (two mid-size uploads
  // in one frame would otherwise each pass the check individually yet
  // overrun the buffer end -- heap corruption downstream).
  if (this->buffer_ != VK_NULL_HANDLE &&
      this->capacity_ >= this->cursor_ + required) {
    return true;
  }
  // Grow: at least double the current capacity so a burst of uploads in a
  // frame amortizes a single reallocation instead of one per upload.
  VkDeviceSize newCapacity =
    std::max<VkDeviceSize>(this->cursor_ + required, this->capacity_ * 2);
  newCapacity = std::max<VkDeviceSize>(newCapacity, 256u * 1024u);

  VkBuffer newBuffer = VK_NULL_HANDLE;
  VmaAllocation newAllocation = nullptr;
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = newCapacity;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  // Ask VMA for the persistent host mapping up front: the staging pool is
  // written every frame, so a one-time map (no per-upload vkMapMemory) is the
  // whole point.  VMA_MEMORY_USAGE_AUTO requires an explicit host-access flag
  // whenever MAPPED is requested.
  allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->allocator_, &bci, &allocInfo, &newBuffer,
                      &newAllocation, &allocationInfo) != VK_SUCCESS) {
    return false;
  }
  void * newMapped = allocationInfo.pMappedData;
  if (newMapped == nullptr) {
    // Should not happen with VMA_ALLOCATION_CREATE_MAPPED_BIT on host-visible
    // memory, but do not leave a half-built pool behind.
    vmaDestroyBuffer(this->allocator_, newBuffer, newAllocation);
    return false;
  }
  // Preserve any bytes already staged in the old buffer (uploads prepared
  // earlier in this frame) before swapping it out.
  if (this->buffer_ != VK_NULL_HANDLE && this->mapped_ && this->cursor_ > 0) {
    std::memcpy(newMapped, this->mapped_, static_cast<size_t>(this->cursor_));
  }
  if (this->buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(this->allocator_, this->buffer_, this->allocation_);
  }
  this->buffer_ = newBuffer;
  this->allocation_ = newAllocation;
  this->mapped_ = newMapped;
  this->capacity_ = newCapacity;
  SoVulkanDebugUtils::nameObject(
    this->device_, VK_OBJECT_TYPE_BUFFER,
    reinterpret_cast<uint64_t>(this->buffer_), "texture staging pool");
  return true;
}

bool
SoVulkanStagingPool::stage(const void * src, VkDeviceSize bytes,
                           VkDeviceSize & offset)
{
  if (!this->ensureCapacity(bytes)) {
    return false;
  }
  offset = this->cursor_;
  std::memcpy(static_cast<unsigned char *>(this->mapped_) +
                static_cast<size_t>(offset),
              src, static_cast<size_t>(bytes));
  // Keep every upload 4-byte aligned so the next vkCmdCopyBufferToImage()
  // offset is a multiple of the copy alignment the driver requires.
  this->cursor_ += ((bytes + 3u) & ~(VkDeviceSize)3u);
  return true;
}

void
SoVulkanStagingPool::reset()
{
  this->cursor_ = 0;
}

void
SoVulkanStagingPool::destroy()
{
  if (this->buffer_ != VK_NULL_HANDLE) {
    // vmaDestroyBuffer releases the buffer, its memory and the persistent host
    // mapping together.
    vmaDestroyBuffer(this->allocator_, this->buffer_, this->allocation_);
    this->buffer_ = VK_NULL_HANDLE;
    this->allocation_ = nullptr;
    this->mapped_ = nullptr;
  }
  this->capacity_ = 0;
  this->cursor_ = 0;
}
