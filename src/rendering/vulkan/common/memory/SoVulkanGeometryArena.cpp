#include <rendering/vulkan/common/memory/SoVulkanGeometryArena.h>

#include <rendering/vulkan/common/memory/SoVulkanBufferFactory.h>

#include <algorithm>
#include <utility>

VkDeviceSize
alignGeometryUpload(VkDeviceSize bytes)
{
  const VkDeviceSize alignment = 64;
  return ((bytes + alignment - 1) / alignment) * alignment;
}

void
SoVulkanGeometryArena::initialize(VkDevice device, VmaAllocator vmaAllocator,
                                  SoVulkanBufferFactory * buffers)
{
  this->device_ = device;
  this->vmaAllocator_ = vmaAllocator;
  this->buffers_ = buffers;
}

void
SoVulkanGeometryArena::setDeferCallback(
    std::function<void(std::function<void()> &&)> cb)
{
  this->deferDestroy_ = std::move(cb);
}

uint32_t
SoVulkanGeometryArena::allocateBlock(VkDeviceSize capacity)
{
  capacity = alignGeometryUpload(std::max<VkDeviceSize>(capacity, 64));
  if (capacity == 0 || this->buffers_ == nullptr) {
    return 0;
  }

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation memory = nullptr;
  void * mapped = nullptr;
  if (!this->buffers_->createMapped(
        capacity,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        buffer, memory, &mapped)) {
    return 0;
  }

  Block block {};
  block.buffer = buffer;
  block.memory = memory;
  block.mapped = mapped;
  block.capacity = capacity;
  block.used = 0;
  block.refCount = 0;
  if (!this->freeBlockIds_.empty()) {
    const uint32_t recycled = this->freeBlockIds_.back();
    this->freeBlockIds_.pop_back();
    this->blocks_[recycled - 1] = block;
    this->nextBlockCapacity_ =
      std::min<VkDeviceSize>(this->nextBlockCapacity_ * 2u, 16u * 1024u * 1024u);
    return recycled;
  }
  this->blocks_.push_back(block);

  this->nextBlockCapacity_ =
    std::min<VkDeviceSize>(this->nextBlockCapacity_ * 2u, 16u * 1024u * 1024u);
  return static_cast<uint32_t>(this->blocks_.size());
}

bool
SoVulkanGeometryArena::allocate(uint32_t blockId, VkDeviceSize size,
                                VkDeviceSize & offset)
{
  offset = 0;
  if (blockId == 0 || blockId > this->blocks_.size() || size == 0) {
    return false;
  }
  Block & block = this->blocks_[blockId - 1];
  if (block.buffer == VK_NULL_HANDLE || block.mapped == nullptr) {
    return false;
  }
  const VkDeviceSize aligned = alignGeometryUpload(size);
  if (block.used + aligned > block.capacity) {
    return false;
  }
  offset = block.used;
  block.used += aligned;
  return true;
}

void
SoVulkanGeometryArena::releaseBlockResources(Block & block)
{
  if (block.buffer != VK_NULL_HANDLE) {
    // vmaDestroyBuffer releases the buffer, its memory and the persistent host
    // mapping together, so no explicit vkUnmapMemory is needed.
    vmaDestroyBuffer(this->vmaAllocator_, block.buffer, block.memory);
    block.buffer = VK_NULL_HANDLE;
  }
  block.memory = nullptr;
  block.mapped = nullptr;
  block.capacity = 0;
  block.used = 0;
  block.refCount = 0;
}

void
SoVulkanGeometryArena::releaseBlock(uint32_t blockId)
{
  if (blockId == 0 || blockId > this->blocks_.size()) {
    return;
  }
  Block & block = this->blocks_[blockId - 1];
  if (block.buffer == VK_NULL_HANDLE) {
    return;
  }
  if (block.refCount > 0) {
    --block.refCount;
  }
  if (block.refCount > 0) {
    return;
  }
  this->releaseBlockResources(block);
  this->freeBlockIds_.push_back(blockId);
}

void
SoVulkanGeometryArena::deferReleaseBlock(uint32_t blockId)
{
  if (blockId == 0) {
    return;
  }
  if (this->deferDestroy_) {
    this->deferDestroy_([this, blockId]() { this->releaseBlock(blockId); });
  }
  else {
    this->releaseBlock(blockId);
  }
}

void
SoVulkanGeometryArena::destroyAll()
{
  for (Block & block : this->blocks_) {
    this->releaseBlockResources(block);
  }
  this->blocks_.clear();
  this->freeBlockIds_.clear();
  this->nextBlockCapacity_ = 256u * 1024u;
}

SoVulkanGeometryArena::Block *
SoVulkanGeometryArena::block(uint32_t blockId)
{
  if (blockId == 0 || blockId > this->blocks_.size()) {
    return nullptr;
  }
  return &this->blocks_[blockId - 1];
}
