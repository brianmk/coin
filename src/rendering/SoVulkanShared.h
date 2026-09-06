// src/rendering/SoVulkanShared.h
//
// Shared, internal Vulkan device-level primitives used by both the raster
// (SoVulkanRenderBackend) and ray-tracing (SoRTXRenderBackend) backends plus
// the orchestration layer.  This header is internal to Coin's Vulkan renderer
// (not installed, not public API).  Keep everything inline / POD so a
// translation unit that does not use a helper does not pull an out-of-line
// definition.

#ifndef COIN_SOVULKANSHARED_H
#define COIN_SOVULKANSHARED_H

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include <vulkan/vulkan.h>

namespace SoVulkanShared {

// Environment flags are enabled by presence, but honor the conventional
// "VAR=0"/"false"/"off" opt-out values.
inline bool
envFlagEnabled(const char * name)
{
  const char * value = std::getenv(name);
  if (value == nullptr) return false;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "off") != 0;
}

// Literal-name fast path: the per-call-site static resolves the flag once, so
// per-frame hot paths pay no getenv() at all.  Shared by both backends so the
// env-flag policy lives in one place.
#define COIN_VULKAN_ENV_FLAG(name) \
  ([] { static const bool coin_env_flag_cached = \
          SoVulkanShared::envFlagEnabled(name); \
        return coin_env_flag_cached; }())

// Cached physical-device memory properties picker.  vkGetPhysicalDeviceMemoryProperties
// is queried once per device (not per allocation); the raster and RT backends
// both route their memory-type search through it so the selection logic and its
// caching are singular.
class MemoryProperties {
public:
  MemoryProperties() = default;
  explicit MemoryProperties(VkPhysicalDevice device)
    : m_device(device) {}

  void setDevice(VkPhysicalDevice device)
  {
    if (device != m_device) {
      m_device = device;
      m_valid = false;
    }
  }
  VkPhysicalDevice device() const { return m_device; }

  // Cached physical-device memory properties (ensured once per device).
  const VkPhysicalDeviceMemoryProperties & properties() const
  {
    this->ensure();
    return m_props;
  }

  // Pick the first memory type matching `desired`, falling back to any type
  // the device offers for this resource.  Returns false only when no type is
  // usable (or no device is bound).
  bool pick(const VkMemoryRequirements & requirements,
            VkMemoryPropertyFlags desired,
            uint32_t & memoryTypeIndex) const
  {
    this->ensure();
    if (!m_valid) return false;
    for (uint32_t i = 0; i < m_props.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (m_props.memoryTypes[i].propertyFlags & desired) == desired) {
        memoryTypeIndex = i;
        return true;
      }
    }
    for (uint32_t i = 0; i < m_props.memoryTypeCount; ++i) {
      if (requirements.memoryTypeBits & (1u << i)) {
        memoryTypeIndex = i;
        return true;
      }
    }
    return false;
  }

private:
  void ensure() const
  {
    if (m_valid) return;
    if (m_device == VK_NULL_HANDLE) {
      m_props = {};
      return;
    }
    vkGetPhysicalDeviceMemoryProperties(m_device, &m_props);
    m_valid = true;
  }

  VkPhysicalDevice m_device = VK_NULL_HANDLE;
  mutable VkPhysicalDeviceMemoryProperties m_props {};
  mutable bool m_valid = false;
};

// Deferred-destruction batching for resources replaced while recording a
// frame.  A resource must not be destroyed while its owning submission may
// still reference it, so destroys are queued into the slot a few frames behind
// the producer and released once the reference is certainly drained.
//
// Two access styles are supported so both backends can use it without changing
// their frame model:
//   - ring-slot (raster backend): the caller passes an absolute frame index and
//     deferAt/flushAt mask by batchCount (the batch that is N frames old).
//   - current-slot (RT backend): defer() fills the current batch and the caller
//     toggles the index and flushes the batch it just vacated.
class PendingDestroys {
public:
  explicit PendingDestroys(uint32_t batchCount = 3)
    : m_batches(batchCount ? batchCount : 1) {}

  uint32_t batchCount() const { return static_cast<uint32_t>(m_batches.size()); }
  uint32_t index() const { return m_index; }
  void setIndex(uint32_t i) { m_index = i % m_batches.size(); }

  // Ring-slot style: caller supplies an absolute frame slot.
  void deferAt(uint32_t slot, std::function<void()> && fn)
  {
    m_batches[slot % m_batches.size()].push_back(std::move(fn));
  }
  void flushAt(uint32_t slot)
  {
    auto & b = m_batches[slot % m_batches.size()];
    for (auto & fn : b) { if (fn) fn(); }
    b.clear();
  }

  // Current-slot style (RT backend double-buffer).
  void defer(std::function<void()> && fn)
  {
    m_batches[m_index].push_back(std::move(fn));
  }
  std::vector<std::function<void()>> & batch(uint32_t i)
  {
    return m_batches[i % m_batches.size()];
  }
  const std::vector<std::function<void()>> & batch(uint32_t i) const
  {
    return m_batches[i % m_batches.size()];
  }

  // Regrow the batch ring.  On shrink, only the trailing batches (the ones a
  // new smaller ring no longer addresses) are flushed and emptied; the
  // retained batches keep their entries because their frames may still be in
  // flight.  Used when the caller's in-flight count changes.
  void setBatchCount(uint32_t count)
  {
    if (count == 0) count = 1;
    const uint32_t cur = this->batchCount();
    if (count == cur) return;
    if (count < cur) {
      for (uint32_t i = count; i < cur; ++i) {
        auto & b = m_batches[i];
        for (auto & fn : b) { if (fn) fn(); }
        b.clear();
      }
    }
    m_batches.resize(count);
    if (m_index >= count) m_index = 0;
  }

  bool empty() const
  {
    for (const auto & b : m_batches) { if (!b.empty()) return false; }
    return true;
  }

  void flushAll()
  {
    for (auto & b : m_batches) {
      for (auto & fn : b) { if (fn) fn(); }
      b.clear();
    }
  }

private:
  std::vector<std::vector<std::function<void()>>> m_batches;
  uint32_t m_index = 0;
};

// Memory-type picker for buffer allocation.  Given a resource's memory
// requirements and the desired property flags it returns the index of a
// compatible memory type.  Each backend supplies its own policy so the two
// search modes stay distinct: the raster backend uses exact-match (no fallback)
// and the RT backend uses best-effort fallback (MemoryProperties::pick).
using MemoryTypePicker =
  std::function<bool(const VkMemoryRequirements &, VkMemoryPropertyFlags, uint32_t &)>;

// Bind memory to an existing buffer after picking its type with `pick`.  Used
// for buffers whose VkBufferCreateInfo the caller builds itself (e.g. TRANSFER_DST
// staging, external memory) and for the raster backend's re-usable
// allocateBufferMemory path.  `requirements` are the buffer's memory
// requirements (queried by the caller) so the type index is selected against
// them without a redundant re-query.
inline bool
bindBufferMemory(VkDevice device, const VkAllocationCallbacks * allocator,
                 VkBuffer buffer, const VkMemoryRequirements & requirements,
                 VkMemoryPropertyFlags desired,
                 const MemoryTypePicker & pick, VkDeviceMemory & memory,
                 const void * allocPNext = nullptr)
{
  memory = VK_NULL_HANDLE;
  uint32_t memoryTypeIndex = 0;
  if (!pick(requirements, desired, memoryTypeIndex)) return false;

  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.pNext = allocPNext;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = memoryTypeIndex;
  if (vkAllocateMemory(device, &ai, allocator, &memory) != VK_SUCCESS) {
    memory = VK_NULL_HANDLE;
    return false;
  }
  if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
    vkFreeMemory(device, memory, allocator);
    memory = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

// Create a VkBuffer, then allocate and bind its memory.  `pick` selects the
// memory type (raster exact-match vs RT fallback).  `deviceAddress` sets
// VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT for SHADER_DEVICE_ADDRESS buffers
// (VUID-VkMemoryAllocateInfo-flags-03339).  On any failure nothing is left
// allocated and false is returned.
inline bool
createBufferAllocated(VkDevice device, const VkAllocationCallbacks * allocator,
                      VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags desired, bool deviceAddress,
                      const MemoryTypePicker & pick, VkBuffer & buffer,
                      VkDeviceMemory & memory)
{
  buffer = VK_NULL_HANDLE;
  memory = VK_NULL_HANDLE;
  VkBufferCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &ci, allocator, &buffer) != VK_SUCCESS) return false;

  // Buffers carrying SHADER_DEVICE_ADDRESS_BIT must be allocated with the
  // device-address memory flag (VUID-VkMemoryAllocateInfo-flags-03339).
  VkMemoryAllocateFlagsInfo allocFlags {};
  allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  const void * pNext = deviceAddress ? static_cast<const void *>(&allocFlags)
                                     : nullptr;

  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(device, buffer, &requirements);
  if (!bindBufferMemory(device, allocator, buffer, requirements, desired, pick,
                        memory, pNext)) {
    vkDestroyBuffer(device, buffer, allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

// Build an image memory barrier for a layout transition.  The subresource
// range defaults to the single mip / layer used by the bulk of the transition
// sites; pass levelCount/layerCount to cover a whole image.
inline VkImageMemoryBarrier
imageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
             VkAccessFlags srcMask, VkAccessFlags dstMask,
             VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
             uint32_t levelCount = 1, uint32_t layerCount = 1)
{
  VkImageMemoryBarrier b {};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.oldLayout = oldLayout;
  b.newLayout = newLayout;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange.aspectMask = aspect;
  b.subresourceRange.baseMipLevel = 0;
  b.subresourceRange.levelCount = levelCount;
  b.subresourceRange.baseArrayLayer = 0;
  b.subresourceRange.layerCount = layerCount;
  b.srcAccessMask = srcMask;
  b.dstAccessMask = dstMask;
  return b;
}

// Execute an image layout transition via a single pipeline image-memory barrier.
inline void
imageTransition(VkCommandBuffer cmd, VkImage image,
                VkImageLayout oldLayout, VkImageLayout newLayout,
                VkAccessFlags srcMask, VkAccessFlags dstMask,
                VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                uint32_t levelCount = 1, uint32_t layerCount = 1)
{
  VkImageMemoryBarrier b = imageBarrier(image, oldLayout, newLayout, srcMask,
                                        dstMask, aspect, levelCount, layerCount);
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Execute a buffer memory barrier to make a transfer/source region visible to a
// later access (e.g. TRANSFER_WRITE -> VERTEX_ATTRIBUTE/INDEX read).
inline void
bufferTransition(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset,
                 VkDeviceSize size, VkAccessFlags srcMask, VkAccessFlags dstMask,
                 VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
  VkBufferMemoryBarrier b {};
  b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  b.srcAccessMask = srcMask;
  b.dstAccessMask = dstMask;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer = buffer;
  b.offset = offset;
  b.size = size;
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &b, 0, nullptr);
}

// Run a small, non-render-pass command buffer on `queue` and wait until it has
// fully executed (vkQueueWaitIdle) before returning.  The buffer is allocated
// from `pool`, recorded by `record` between begin/end, submitted, and freed.
// On any Vulkan failure nothing is left allocated and false is returned.  Safe
// for setup / host-upload paths that must synchronously consume resources
// afterwards (the wait also retires any other in-flight work on the queue).
inline bool
withOneShotSubmit(VkDevice device, VkQueue queue, VkCommandPool pool,
                  const VkAllocationCallbacks * /*allocator*/,
                  const std::function<void(VkCommandBuffer)> & record)
{
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) return false;

  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  bool ok = vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS;
  if (ok && record) record(cmd);
  if (ok) ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
  if (ok) {
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    ok = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
  }
  // Retire any in-flight work on the queue so resources referenced by the
  // submission are safe to destroy synchronously on return.
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, pool, 1, &cmd);
  return ok;
}

} // namespace SoVulkanShared

#endif // COIN_SOVULKANSHARED_H
