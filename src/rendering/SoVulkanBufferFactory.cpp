// src/rendering/SoVulkanBufferFactory.cpp
#include "rendering/SoVulkanBufferFactory.h"

#include "rendering/SoVulkanShared.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

void
SoVulkanBufferFactory::initialize(VkDevice device, VmaAllocator vmaAllocator,
                                  const VkAllocationCallbacks * allocator,
                                  VkQueue queue, VkCommandPool commandPool,
                                  ErrorSink emitError)
{
  this->device_ = device;
  this->vmaAllocator_ = vmaAllocator;
  this->allocator_ = allocator;
  this->queue_ = queue;
  this->commandPool_ = commandPool;
  this->emitError_ = std::move(emitError);
}

bool
SoVulkanBufferFactory::createWithProperties(
  const VkDeviceSize size, const VkBufferUsageFlags usage,
  const VkMemoryPropertyFlags desiredProperties, VkBuffer & buffer,
  VmaAllocation & memory, const void * data)
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
  allocInfo.requiredFlags = desiredProperties;
  if ((desiredProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    // HOST_VISIBLE memory is either filled once here or written per frame
    // through a persistent map; declare the sequential-write access VMA wants
    // and, for the one-time fill, request the mapping up front.
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    if (data) {
      allocInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
  }
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator_, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }

  if (data) {
    void * mapped = allocationInfo.pMappedData;
    const bool unmap = (mapped == nullptr);
    if (unmap &&
        vmaMapMemory(this->vmaAllocator_, memory, &mapped) != VK_SUCCESS) {
      if (this->emitError_) {
        this->emitError_("createBufferWithProperties: vmaMapMemory failed");
      }
      vmaDestroyBuffer(this->vmaAllocator_, buffer, memory);
      buffer = VK_NULL_HANDLE;
      memory = nullptr;
      return false;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    if (unmap) {
      vmaUnmapMemory(this->vmaAllocator_, memory);
    }
  }
  return true;
}

bool
SoVulkanBufferFactory::create(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkBuffer & buffer, VmaAllocation & memory,
                              const void * data)
{
  return this->createWithProperties(
    size, usage,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    buffer, memory, data);
}

bool
SoVulkanBufferFactory::createMapped(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkBuffer & buffer, VmaAllocation & memory,
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
  // VMA_MEMORY_USAGE_AUTO requires an explicit host-access flag whenever
  // MAPPED is requested.
  allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator_, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  if (allocationInfo.pMappedData == nullptr) {
    vmaDestroyBuffer(this->vmaAllocator_, buffer, memory);
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  if (mapped) *mapped = allocationInfo.pMappedData;
  return true;
}

bool
SoVulkanBufferFactory::createDeviceLocal(VkDeviceSize size,
                                         VkBufferUsageFlags usage,
                                         VkBuffer & buffer,
                                         VmaAllocation & memory,
                                         const void * data)
{
  // Retained static geometry is read by the GPU every frame, so it belongs in
  // device-local VRAM rather than host-visible memory.  `data` is copied from
  // a transient host-visible staging buffer with a one-shot transfer that is
  // fenced before this function returns.  The GPU then reads the mesh from
  // device memory instead of walking the PCIe/system bus every frame.
  //
  // This is only invoked from the geometry-change path (not steady-state), so
  // the synchronous transfer is acceptable.  On any failure the buffer/memory
  // are left null and the caller falls back to the host-visible create().
  buffer = VK_NULL_HANDLE;
  memory = nullptr;
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator_, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }

  if (!data) return true;

  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingMemory = nullptr;
  if (!this->create(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging,
                    stagingMemory, data)) {
    vmaDestroyBuffer(this->vmaAllocator_, buffer, memory);
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }

  // One-shot transfer command buffer.  The per-frame buffers are not yet begun
  // at this point (updateGeometryCache runs before beginCommandBuffer), so the
  // shared one-shot helper allocates a transient buffer from the command pool,
  // records the copy + barrier, submits, and drains the queue before returning.
  const bool ok = SoVulkanShared::withOneShotSubmit(
    this->device_, this->queue_, this->commandPool_, this->allocator_,
    [staging, buffer, size](VkCommandBuffer transfer) {
      VkBufferCopy copy {};
      copy.size = size;
      vkCmdCopyBuffer(transfer, staging, buffer, 1, &copy);
      // Make the device-local writes visible to a later vertex-input read.
      // The submit is drained before returning, but completion alone does not
      // establish a memory dependency for the buffer read as vertex/index
      // attributes in a later submit, so transition TRANSFER_WRITE ->
      // VERTEX_ATTRIBUTE/INDEX read explicitly.
      SoVulkanShared::bufferTransition(
        transfer, buffer, 0, size, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    });

  vmaDestroyBuffer(this->vmaAllocator_, staging, stagingMemory);

  if (!ok) {
    vmaDestroyBuffer(this->vmaAllocator_, buffer, memory);
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  return true;
}
