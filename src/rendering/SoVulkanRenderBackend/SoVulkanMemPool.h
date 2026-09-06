// src/rendering/SoVulkanRenderBackend/SoVulkanMemPool.h
//
// Minimal Vulkan device-memory sub-allocator.
//
// Every texture upload currently vkCreateImage -> vkGetImageMemoryRequirements
// -> vkAllocateMemory -> vkBindImageMemory, and its staging buffer similarly
// vkAllocateMemory/vkFreeMemory.  That is an OS/driver-level allocation per
// texture (and per upload), which is slow and counts against the device's
// maxMemoryAllocationCount.  The Khronos guidance for Vulkan memory is to
// create a few large allocations and sub-allocate from them.
//
// This pool sub-allocates ranges from a set of large VkDeviceMemory blocks,
// one pool per memory-type index.  A free-list tracks the unused ranges of
// each block so freed ranges can be reused; adjacent ranges are coalesced.
// Blocks grow geometrically (starting at blockSizeHint) and are released only
// at destroyAll().
//
// Correctness contract for the backend using this pool:
//   - alloc() returns (block, offset).  The caller binds exactly `size` bytes
//     at `offset` of a resource (buffer or image).  Ranges returned by two
//     alloc() calls never overlap, so two resources bound to the same block
//     are never live at the same offset.
//   - free() must be called only once the GPU can no longer reference the
//     resource (i.e. from the backend's deferred-destroy ring, which runs
//     maxFramesInFlight frames after the release was requested).  Reusing a
//     freed range before then could alias a still-executing submission.
//   - alloc() is not fully thread-safe for block creation (a new block
//     allocates device memory).  Keep the recording call off the worker-hot
//     path; the backend sizes the pool's blocks up front and only ever
//     returns free ranges there.

#ifndef COIN_SOVULKANMEMPOOL_H
#define COIN_SOVULKANMEMPOOL_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

class SoVulkanMemPool {
public:
  SoVulkanMemPool(VkDevice device, const VkAllocationCallbacks * allocator,
                  VkDeviceSize blockSizeHint = 4u * 1024u * 1024u);
  ~SoVulkanMemPool();
  SoVulkanMemPool(const SoVulkanMemPool &) = delete;
  SoVulkanMemPool & operator=(const SoVulkanMemPool &) = delete;

  // Allocate `size` bytes (rounded up to `alignment`, which must be a power of
  // two) from memory-type `memoryTypeIndex`.  On success returns true and sets
  // `block`/`offset`.  On failure returns false leaving block null.
  bool alloc(uint32_t memoryTypeIndex, VkDeviceSize size,
             VkDeviceSize alignment, VkDeviceMemory & block,
             VkDeviceSize & offset);

  // Return a range previously obtained from alloc() back to the pool.  The
  // caller guarantees the GPU can no longer reference the range (deferred
  // destroy).  Coalesces with adjacent free ranges.
  void free(VkDeviceMemory block, VkDeviceSize offset, VkDeviceSize size);

  // Release every block (shutdown).  The queue must be idle.
  void destroyAll();

  // Number of VkDeviceMemory blocks currently held (diagnostics / MEM_REPORT).
  std::size_t blockCount() const;

private:
  struct Block {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;
    // Free ranges, kept sorted by offset, non-overlapping, non-adjacent.
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> freeRanges;
  };
  struct BlockPool {
    std::vector<Block> blocks;
  };

  bool findBlock(uint32_t memoryTypeIndex, VkDeviceSize size,
                 VkDeviceSize alignment, VkDeviceMemory & outMemory,
                 VkDeviceSize & outOffset);
  bool grow(BlockPool & pool, uint32_t memoryTypeIndex, VkDeviceSize size,
            VkDeviceSize alignment, VkDeviceMemory & outMemory,
            VkDeviceSize & outOffset);

  static VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize alignment);

  VkDevice m_device = VK_NULL_HANDLE;
  const VkAllocationCallbacks * m_allocator = nullptr;
  VkDeviceSize m_blockSizeHint = 0;
  std::unordered_map<uint32_t, BlockPool> m_pools;
};

#endif // COIN_SOVULKANMEMPOOL_H
