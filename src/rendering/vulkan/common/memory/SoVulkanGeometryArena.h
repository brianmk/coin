#ifndef COIN_SOVULKANGEOMETRYARENA_H
#define COIN_SOVULKANGEOMETRYARENA_H

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <functional>
#include <vector>

class SoVulkanBufferFactory;

// Round a geometry upload size up to the arena's allocation alignment.
VkDeviceSize alignGeometryUpload(VkDeviceSize bytes);

/*!
  \brief Bump-allocated vertex/index arena for the geometry cache.

  Geometry is packed into large shared host-visible blocks instead of one
  buffer per command, so a scene with many small meshes pays a handful of
  allocations rather than thousands.  Each command's vertex/index ranges are
  carved out of a block with allocate() and referenced by the cache entry
  (offset + shared block id); a block is released once the last entry that
  references it is evicted.

  Borrows the device handles and the shared buffer factory from the backend
  rather than owning them (the backend still uses the handles directly in many
  places), and routes block release through a deferred-destruction callback so
  an in-flight submission that may still read the block drains first.
*/
class SoVulkanGeometryArena {
public:
  struct Block {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    void * mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize used = 0;
    uint32_t refCount = 0;
  };

  // `buffers` must outlive the arena; `deferDestroy` is the backend's
  // deferred-destruction ring (see SoVulkanRenderBackend::deferDestroy).
  void initialize(VkDevice device, VmaAllocator vmaAllocator,
                  SoVulkanBufferFactory * buffers);
  void setDeferCallback(std::function<void(std::function<void()> &&)> cb);

  // Allocate a fresh (or recycled) block sized for `capacity`, returning its
  // 1-based id, or 0 on failure.
  uint32_t allocateBlock(VkDeviceSize capacity);
  // Carve `size` bytes out of `blockId`, returning the byte offset in `offset`.
  // Returns false when the block is full or invalid.
  bool allocate(uint32_t blockId, VkDeviceSize size, VkDeviceSize & offset);
  // Drop one reference from `blockId`; when it reaches zero the block is
  // unmapped, destroyed and returned to the free-id pool.
  void releaseBlock(uint32_t blockId);
  // releaseBlock() via the deferred-destruction ring (for a block that a
  // still-executing submission may read).
  void deferReleaseBlock(uint32_t blockId);
  // Release every block and reset the growth heuristic.  Called on shutdown.
  void destroyAll();

  // Mutable access for the upload path, which writes into the block's mapping.
  // Null when `blockId` is out of range or the block is empty.
  Block * block(uint32_t blockId);

private:
  void releaseBlockResources(Block & block);

  VkDevice device_ = VK_NULL_HANDLE;
  VmaAllocator vmaAllocator_ = nullptr;
  SoVulkanBufferFactory * buffers_ = nullptr;
  std::function<void(std::function<void()> &&)> deferDestroy_;

  std::vector<Block> blocks_;
  // Released blocks are kept as reusable ids so a geometry-change burst does
  // not grow blocks_ without bound; releaseBlock() returns an id here once its
  // refCount drops to 0.
  std::vector<uint32_t> freeBlockIds_;
  VkDeviceSize nextBlockCapacity_ = 256u * 1024u;
};

#endif // COIN_SOVULKANGEOMETRYARENA_H
