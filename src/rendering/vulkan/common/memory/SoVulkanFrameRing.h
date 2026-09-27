// src/rendering/vulkan/common/memory/SoVulkanFrameRing.h
//
// Owns the per-in-flight-frame primary command buffers and their fences.
// Borrows the device and command pool for allocate()/release(); it neither
// creates nor destroys them.  The own-queue path submits slot N's buffer and
// signals slot N's fence; the caller waits the fence before reusing the slot's
// UBO ring half, command buffer and deferred-destruction batch.  The external
// path records into the caller's buffer and never signals these fences, so
// pending() stays false there.  Internal to Coin's Vulkan renderer (not
// installed, not public API).

#ifndef COIN_SOVULKANFRAMERING_H
#define COIN_SOVULKANFRAMERING_H

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

class SoVulkanFrameRing {
public:
  // Allocate `count` primary command buffers + fences.  On failure the ring is
  // left empty (any partial allocation is freed), so the caller can safely
  // continue to release().
  bool allocate(VkDevice device, VkCommandPool commandPool,
                const VkAllocationCallbacks * allocator, uint32_t count);

  // Free every command buffer + fence.  The caller must have made the queue
  // idle (or waited the fences) first.
  void release(VkDevice device, VkCommandPool commandPool,
               const VkAllocationCallbacks * allocator);

  bool empty() const { return this->buffers_.empty(); }
  uint32_t count() const
  {
    return static_cast<uint32_t>(this->buffers_.size());
  }
  VkCommandBuffer buffer(uint32_t slot) const;
  VkFence fence(uint32_t slot) const;
  bool pending(uint32_t slot) const;
  void setPending(uint32_t slot, bool value);
  // Wait every fence whose pending flag is set (and which is non-null).
  // Returns false on a vkWaitForFences failure (typically device lost); the
  // caller must treat that as fatal rather than freeing resources that may
  // still be in flight.
  bool waitAll(VkDevice device) const;

private:
  std::vector<VkCommandBuffer> buffers_;
  std::vector<VkFence> fences_;
  std::vector<uint8_t> pending_;
};

#endif // COIN_SOVULKANFRAMERING_H
