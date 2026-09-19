// src/rendering/SoVulkanBufferFactory.h
//
// Shared Vulkan buffer-creation helpers for the raster and ray-tracing
// backends.  Borrows the device / VMA allocator / queue / command-pool handles
// owned by the caller (it neither creates nor destroys them); owns only the
// creation logic.  Internal to Coin's Vulkan renderer (not installed, not
// public API).

#ifndef COIN_SOVULKANBUFFERFACTORY_H
#define COIN_SOVULKANBUFFERFACTORY_H

#include <functional>

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

class SoVulkanBufferFactory {
public:
  // Diagnostics sink for the rare internal failure (currently only the
  // one-time fill mapping in createWithProperties()).  May be empty.
  using ErrorSink = std::function<void(const char *)>;

  void initialize(VkDevice device, VmaAllocator vmaAllocator,
                  const VkAllocationCallbacks * allocator, VkQueue queue,
                  VkCommandPool commandPool, ErrorSink emitError = {});

  // HOST_VISIBLE | HOST_COHERENT buffer; fills it with `data` when non-null.
  // On failure buffer/allocation are left null.
  bool create(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer & buffer,
              VmaAllocation & memory, const void * data);

  // DEVICE_LOCAL buffer for retained static geometry.  Copies `data` through a
  // transient host-visible staging buffer with a one-shot, fenced transfer, so
  // it is only meant for the rare geometry-change path, never the steady-state
  // per-frame path.  On failure buffer/allocation are left null.
  bool createDeviceLocal(VkDeviceSize size, VkBufferUsageFlags usage,
                         VkBuffer & buffer, VmaAllocation & memory,
                         const void * data);

  // Buffer backed by memory with the desired properties.  When `data` is
  // non-null the host-visible contents are filled.  On failure buffer/
  // allocation are left null.
  bool createWithProperties(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags desiredProperties,
                            VkBuffer & buffer, VmaAllocation & memory,
                            const void * data = nullptr);

  // HOST_VISIBLE | HOST_COHERENT buffer with a persistent mapping established
  // in one step.  On any failure buffer/allocation are left null and *mapped
  // null, with nothing allocated.
  bool createMapped(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkBuffer & buffer, VmaAllocation & memory, void ** mapped);

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VmaAllocator vmaAllocator_ = nullptr;
  const VkAllocationCallbacks * allocator_ = nullptr;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  ErrorSink emitError_;
};

#endif // COIN_SOVULKANBUFFERFACTORY_H
