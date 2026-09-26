// src/rendering/vulkan/common/memory/SoVulkanStagingPool.h
//
// Persistent host-visible staging buffer that coalesces every pending texture
// upload of a frame into one buffer write (and, on the external fallback, one
// submit), instead of allocating a fresh transient staging buffer per pending
// upload per frame.  Grows on demand, is reused across frames, and is released
// by destroy().  Internal to Coin's Vulkan renderer (not installed, not public
// API).

#ifndef COIN_SOVULKANSTAGINGPOOL_H
#define COIN_SOVULKANSTAGINGPOOL_H

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

class SoVulkanStagingPool {
public:
  // Borrow the device and VMA allocator; no GPU resources are created here.
  void initialize(VkDevice device, VmaAllocator allocator);

  // Copy `bytes` from `src` into the pool at the current cursor and report the
  // byte offset the caller must pass to vkCmdCopyBufferToImage().  Grows the
  // pool (preserving already-staged bytes) if needed.  Returns false without
  // disturbing previously staged data if growth fails.
  bool stage(const void * src, VkDeviceSize bytes, VkDeviceSize & offset);

  // Rewind to the start for a new frame; the buffer is retained for reuse.
  void reset();

  // The buffer holding the staged bytes (VK_NULL_HANDLE before first stage()).
  VkBuffer buffer() const { return this->buffer_; }

  // Unmap and destroy the buffer + its memory.  Safe to call repeatedly.
  void destroy();

private:
  bool ensureCapacity(VkDeviceSize required);

  VkDevice device_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = nullptr;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = nullptr;
  void * mapped_ = nullptr;
  VkDeviceSize capacity_ = 0;
  // Running byte cursor for the current frame's staged uploads; reset by
  // reset() at the start of each flush/record pass.
  VkDeviceSize cursor_ = 0;
};

#endif // COIN_SOVULKANSTAGINGPOOL_H
