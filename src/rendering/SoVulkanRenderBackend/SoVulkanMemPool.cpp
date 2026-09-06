// src/rendering/SoVulkanRenderBackend/SoVulkanMemPool.cpp
//
// Implementation of the device-memory sub-allocator (see SoVulkanMemPool.h).

#include "rendering/SoVulkanRenderBackend/SoVulkanMemPool.h"

#include <algorithm>
#include <cstddef>
#include <utility>

SoVulkanMemPool::SoVulkanMemPool(VkDevice device,
                                 const VkAllocationCallbacks * allocator,
                                 VkDeviceSize blockSizeHint)
  : m_device(device)
  , m_allocator(allocator)
  , m_blockSizeHint(blockSizeHint ? blockSizeHint : (4u * 1024u * 1024u))
{
}

SoVulkanMemPool::~SoVulkanMemPool()
{
  // The backend is expected to call destroyAll() at shutdown; if not (e.g. a
  // partially-failed initialize that never reached the ownership transfer),
  // release here so nothing leaks.  Device memory must not be freed while any
  // submission is in flight, but this destructor runs only at backend teardown
  // which waits for the queue.
  this->destroyAll();
}

VkDeviceSize
SoVulkanMemPool::alignUp(VkDeviceSize v, VkDeviceSize alignment)
{
  if (alignment <= 1) return v;
  return (v + alignment - 1) / alignment * alignment;
}

std::size_t
SoVulkanMemPool::blockCount() const
{
  std::size_t count = 0;
  for (const auto & kv : m_pools) count += kv.second.blocks.size();
  return count;
}

bool
SoVulkanMemPool::alloc(uint32_t memoryTypeIndex, VkDeviceSize size,
                       VkDeviceSize alignment, VkDeviceMemory & block,
                       VkDeviceSize & offset)
{
  block = VK_NULL_HANDLE;
  offset = 0;
  if (size == 0 || memoryTypeIndex >= 32) return false;
  const VkDeviceSize alignedSize = alignUp(size, alignment);
  if (alignedSize == 0) return false;

  BlockPool & pool = m_pools[memoryTypeIndex];
  if (this->findBlock(memoryTypeIndex, alignedSize, alignment, block, offset)) {
    return true;
  }
  return this->grow(pool, memoryTypeIndex, alignedSize, alignment, block,
                    offset);
}

bool
SoVulkanMemPool::findBlock(uint32_t memoryTypeIndex, VkDeviceSize size,
                           VkDeviceSize alignment, VkDeviceMemory & outMemory,
                           VkDeviceSize & outOffset)
{
  BlockPool & pool = m_pools[memoryTypeIndex];
  for (Block & b : pool.blocks) {
    // A free range must be large enough and begin on a boundary compatible
    // with `alignment` for this resource.
    auto it = b.freeRanges.begin();
    while (it != b.freeRanges.end()) {
      const VkDeviceSize start = alignUp(it->first, alignment);
      const VkDeviceSize end = it->first + it->second;
      if (start + size <= end) {
        outMemory = b.memory;
        outOffset = start;
        // Shrink the free range: keep the leading pad as a new free range,
        // and the trailing remainder (if any) as another.
        const VkDeviceSize before = start - it->first;
        const VkDeviceSize after = end - (start + size);
        if (before == 0 && after == 0) {
          b.freeRanges.erase(it);
        }
        else if (before == 0) {
          it->first = start + size;
          it->second = after;
        }
        else if (after == 0) {
          it->second = before;
        }
        else {
          // Split: shrink the current range to the leading pad, insert a new
          // trailing range right after it (they were adjacent, so keep
          // sorted order valid).
          it->second = before;
          b.freeRanges.insert(it + 1,
                              std::make_pair(start + size, after));
        }
        return true;
      }
      ++it;
    }
  }
  return false;
}

bool
SoVulkanMemPool::grow(BlockPool & pool, uint32_t memoryTypeIndex,
                      VkDeviceSize size, VkDeviceSize alignment,
                      VkDeviceMemory & outMemory, VkDeviceSize & outOffset)
{
  // Block size: at least `size`, at least the hint, else double the current
  // largest block so repeated small allocs amortise the block allocation.
  VkDeviceSize blockSize = std::max(m_blockSizeHint, size);
  for (const Block & b : pool.blocks) {
    if (b.capacity >= blockSize) {
      blockSize = b.capacity * 2;
      break;
    }
  }
  blockSize = alignUp(blockSize, alignment);

  // Allocate `blockSize` bytes from the memory type.  There is no fallback
  // memory type: the caller already picked it from the resource's requirements.
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = blockSize;
  ai.memoryTypeIndex = memoryTypeIndex;

  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (vkAllocateMemory(m_device, &ai, m_allocator, &memory) != VK_SUCCESS) {
    return false;
  }

  Block b;
  b.memory = memory;
  b.capacity = blockSize;
  // Consume the requested range up front (off=0), leaving the remainder free.
  if (size < blockSize) {
    b.freeRanges.emplace_back(size, blockSize - size);
  }
  pool.blocks.push_back(std::move(b));

  outMemory = memory;
  outOffset = 0;
  return true;
}

void
SoVulkanMemPool::free(VkDeviceMemory block, VkDeviceSize offset,
                      VkDeviceSize size)
{
  if (block == VK_NULL_HANDLE || size == 0) return;

  // Find the pool/block that owns `block`.
  for (auto & kv : m_pools) {
    BlockPool & pool = kv.second;
    for (Block & b : pool.blocks) {
      if (b.memory != block) continue;
      const VkDeviceSize end = offset + size;
      std::vector<std::pair<VkDeviceSize, VkDeviceSize>> & fr = b.freeRanges;
      // Insert keeping the list sorted and merge with any adjacent range.
      auto it = std::lower_bound(fr.begin(), fr.end(), offset,
                                 [](const std::pair<VkDeviceSize, VkDeviceSize> & r,
                                    VkDeviceSize off) { return r.first < off; });
      // Extend previous range if it touches ours.
      if (it != fr.begin()) {
        auto prev = it - 1;
        if (prev->first + prev->second == offset) {
          prev->second += size;
          // Merge with the next range too if they now are adjacent.
          if (it != fr.end() && prev->first + prev->second == it->first) {
            prev->second += it->second;
            fr.erase(it);
          }
          return;
        }
      }
      // Extend/merge at `it`.
      if (it != fr.end() && offset + size == it->first) {
        it->first = offset;
        it->second += size;
        return;
      }
      fr.insert(it, std::make_pair(offset, size));
      return;
    }
  }
}

void
SoVulkanMemPool::destroyAll()
{
  for (auto & kv : m_pools) {
    for (Block & b : kv.second.blocks) {
      if (b.memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, b.memory, m_allocator);
      }
    }
    kv.second.blocks.clear();
  }
  m_pools.clear();
}
