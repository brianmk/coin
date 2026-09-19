// src/rendering/SoVulkanFrameRing.cpp
#include "rendering/SoVulkanFrameRing.h"

bool
SoVulkanFrameRing::allocate(VkDevice device, VkCommandPool commandPool,
                            const VkAllocationCallbacks * allocator,
                            uint32_t count)
{
  // Start from a clean ring so a failed allocation never leaves a
  // half-populated one behind.
  this->release(device, commandPool, allocator);
  if (commandPool == VK_NULL_HANDLE || count == 0) {
    return false;
  }

  this->buffers_.assign(count, VK_NULL_HANDLE);
  this->fences_.assign(count, VK_NULL_HANDLE);
  this->pending_.assign(count, 0);

  VkCommandBufferAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = count;
  if (vkAllocateCommandBuffers(device, &ai, this->buffers_.data()) !=
      VK_SUCCESS) {
    this->release(device, commandPool, allocator);
    return false;
  }

  VkFenceCreateInfo fi {};
  fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (uint32_t i = 0; i < count; ++i) {
    if (vkCreateFence(device, &fi, allocator, &this->fences_[i]) !=
        VK_SUCCESS) {
      this->release(device, commandPool, allocator);
      return false;
    }
  }
  return true;
}

void
SoVulkanFrameRing::release(VkDevice device, VkCommandPool commandPool,
                           const VkAllocationCallbacks * allocator)
{
  for (VkCommandBuffer buffer : this->buffers_) {
    if (buffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device, commandPool, 1, &buffer);
    }
  }
  this->buffers_.clear();
  for (VkFence fence : this->fences_) {
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(device, fence, allocator);
    }
  }
  this->fences_.clear();
  this->pending_.clear();
}

VkCommandBuffer
SoVulkanFrameRing::buffer(uint32_t slot) const
{
  if (slot >= this->buffers_.size()) return VK_NULL_HANDLE;
  return this->buffers_[slot];
}

VkFence
SoVulkanFrameRing::fence(uint32_t slot) const
{
  if (slot >= this->fences_.size()) return VK_NULL_HANDLE;
  return this->fences_[slot];
}

bool
SoVulkanFrameRing::pending(uint32_t slot) const
{
  return slot < this->pending_.size() && this->pending_[slot] != 0;
}

void
SoVulkanFrameRing::setPending(uint32_t slot, bool value)
{
  if (slot < this->pending_.size()) {
    this->pending_[slot] = value ? 1 : 0;
  }
}

void
SoVulkanFrameRing::waitAll(VkDevice device) const
{
  std::vector<VkFence> pending;
  for (size_t i = 0; i < this->fences_.size(); ++i) {
    if (i < this->pending_.size() && this->pending_[i] != 0 &&
        this->fences_[i] != VK_NULL_HANDLE) {
      pending.push_back(this->fences_[i]);
    }
  }
  if (pending.empty()) return;
  vkWaitForFences(device, static_cast<uint32_t>(pending.size()),
                  pending.data(), VK_TRUE, UINT64_MAX);
}
