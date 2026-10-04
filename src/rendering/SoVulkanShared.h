// Internal Vulkan device-level primitives shared by the raster backend
// (SoVulkanRenderBackend) and the orchestration layer. Not installed/public; keep
// inline/POD so a TU that does not use a helper pulls no out-of-line definition.

#ifndef COIN_SOVULKANSHARED_H
#define COIN_SOVULKANSHARED_H

#include <chrono>
#include "rendering/SoVulkanDebug.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include <Inventor/C/tidbits.h>
#include <Inventor/rendering/SoVulkanImageCopy.h>
#include "rendering/SoVulkanPlatform.h"
#include <vulkan/vulkan.h>

namespace SoVulkanShared {

// --- Environment access --------------------------------------------------
// Single choke point for every COIN_VULKAN_* / COIN_GUI_* lookup (opt-out policy stays auditable/singular).

// Raw value (or nullptr); any set value, including "0", counts as set. Use envFlagEnabled() for the "VAR=0"/"false"/"off" opt-out.
inline const char *
envString(const char * name)
{
  return coin_getenv(name);
}

// Presence-only test (any value, including "0"/"false"/"off").
inline bool
envSet(const char * name)
{
  return coin_getenv(name) != nullptr;
}

// Integer / float value with a default when the variable is unset or empty.
inline int
envInt(const char * name, int defaultValue = 0)
{
  const char * value = coin_getenv(name);
  return value ? std::atoi(value) : defaultValue;
}

inline float
envFloat(const char * name, float defaultValue = 0.0f)
{
  const char * value = coin_getenv(name);
  return value ? static_cast<float>(std::atof(value)) : defaultValue;
}

// Honor the "VAR=0"/"false"/"off" opt-out; a null value yields \a defaultValue, so a flag can default on unless explicitly disabled (retained-IR replay).
inline bool
envFlagEnabled(const char * name, bool defaultValue)
{
  const char * value = coin_getenv(name);
  if (value == nullptr) return defaultValue;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "off") != 0;
}

// Present-and-not-disabled, defaulting to off.
inline bool
envFlagEnabled(const char * name)
{
  return envFlagEnabled(name, false);
}

// --- Phase timing ---------------------------------------------------------
// Monotonic clock shared by the manager and the backends.

inline long
steadyNowUs()
{
  return (long)std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double
steadyNowMs()
{
  return steadyNowUs() * 0.001;
}

// Literal-name fast path: a per-call-site static resolves the flag once, so hot paths pay no getenv(); shared by both backends.
#define COIN_VULKAN_ENV_FLAG(name) \
  ([] { static const bool coin_env_flag_cached = \
          SoVulkanShared::envFlagEnabled(name); \
        return coin_env_flag_cached; }())

// Cached physical-device memory-properties picker: vkGetPhysicalDeviceMemoryProperties
// is queried once per device (not per allocation); both backends route through it.
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

  // Pick the first memory type exactly satisfying `desired` (no fallback); on failure
  // leaves memoryTypeIndex untouched and returns false. Raster policy: a resource that
  // cannot be placed in the requested class is a hard error, not a silent degradation.
  bool pickExact(const VkMemoryRequirements & requirements,
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
    return false;
  }

  // Pick the first memory type matching `desired`, falling back to any device type;
  // false only when none is usable (or no device bound). RT policy: best-effort fallback
  // keeps an allocation on usable memory rather than failing outright.
  bool pick(const VkMemoryRequirements & requirements,
            VkMemoryPropertyFlags desired,
            uint32_t & memoryTypeIndex) const
  {
    if (this->pickExact(requirements, desired, memoryTypeIndex)) return true;
    this->ensure();
    if (!m_valid) return false;
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

// Deferred-destruction batching for resources replaced while recording a frame: a
// resource must not be destroyed while its owning submission may still reference it,
// so destroys queue a few frames behind the producer and release once drained.
// Ring-slot (raster) = absolute frame index + deferAt/flushAt masked by batchCount;
// current-slot (RT) = defer() plus a caller-toggled index whose vacated batch is flushed.
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

  // Current-slot style: append to this slot's deferred-destroy batch.
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

  // Regrow the batch ring; on shrink only trailing batches a smaller ring no longer addresses are flushed (the rest may be in flight).
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

// Memory-type picker: given requirements + desired flags, returns a compatible index (raster exact-match, RT best-effort fallback).
using MemoryTypePicker =
  std::function<bool(const VkMemoryRequirements &, VkMemoryPropertyFlags, uint32_t &)>;

// Pick a type with `pick` and bind memory to an existing buffer (caller-built
// VkBufferCreateInfo: TRANSFER_DST staging, external memory, raster allocateBufferMemory).
// `requirements` are caller-queried so the type index needs no redundant re-query.
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

// Create a VkBuffer then allocate/bind memory. `pick` selects the memory type
// (raster exact vs RT fallback); deviceAddress sets VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
// for SHADER_DEVICE_ADDRESS buffers (VUID-VkMemoryAllocateInfo-flags-03339). On
// failure nothing is left allocated.
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

  // SHADER_DEVICE_ADDRESS buffers need the device-address alloc flag (VUID-VkMemoryAllocateInfo-flags-03339).
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

// Resolve a device entry point into its concrete dispatch type. A direct
// reinterpret_cast between incompatible function-pointer types is conditionally-
// supported and trips pedantic/32-bit compilers, so bit-copy through memcpy as the
// Vulkan loader docs recommend; static_assert guards pointer-width mismatches.
template <typename Fn>
inline Fn
loadDispatch(PFN_vkVoidFunction fn)
{
  static_assert(sizeof(Fn) == sizeof(fn),
                "Vulkan dispatch function pointer size mismatch");
  Fn result{};
  std::memcpy(&result, &fn, sizeof(result));
  return result;
}

// --- VK_KHR_synchronization2 dispatch -------------------------------------
// The device enables VK_KHR_synchronization2 (core in 1.3) when available; then barriers/
// submits use the *2 entry points, else legacy vkCmdPipelineBarrier/vkQueueSubmit. Pointers
// resolve once per device in backend initialize(); null cmdPipelineBarrier2 = legacy path.
// Only core stages/accesses are passed and their legacy/_2_ numeric values match, so masks
// widen with a plain cast (never ALL_COMMANDS/ALL_GRAPHICS in a barrier).
struct Sync2Dispatch {
  PFN_vkCmdPipelineBarrier2KHR cmdPipelineBarrier2 = nullptr;
  PFN_vkQueueSubmit2KHR queueSubmit2 = nullptr;
};

// Function-local static: one instance shared by every TU (C++11 inline semantics), so no out-of-line definition is needed.
inline Sync2Dispatch &
sync2Dispatch()
{
  static Sync2Dispatch dispatch;
  return dispatch;
}

// Emit a global memory barrier via synchronization2 when available, else the legacy pipeline barrier.
inline void
memoryBarrier(VkCommandBuffer cmd,
              VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
              VkAccessFlags srcMask, VkAccessFlags dstMask)
{
  Sync2Dispatch & d = sync2Dispatch();
  if (d.cmdPipelineBarrier2 != nullptr) {
    VkMemoryBarrier2 b {};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    b.srcStageMask = static_cast<VkPipelineStageFlags2>(srcStage);
    b.srcAccessMask = static_cast<VkAccessFlags2>(srcMask);
    b.dstStageMask = static_cast<VkPipelineStageFlags2>(dstStage);
    b.dstAccessMask = static_cast<VkAccessFlags2>(dstMask);
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &b;
    d.cmdPipelineBarrier2(cmd, &dep);
    return;
  }
  VkMemoryBarrier b {};
  b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = srcMask;
  b.dstAccessMask = dstMask;
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0,
                       nullptr);
}

// Build an image memory barrier for a layout transition; the subresource range defaults to one mip/layer (pass counts for a whole image).
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
  Sync2Dispatch & d = sync2Dispatch();
  if (d.cmdPipelineBarrier2 != nullptr) {
    VkImageMemoryBarrier2 b2 {};
    b2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b2.srcStageMask = static_cast<VkPipelineStageFlags2>(srcStage);
    b2.srcAccessMask = static_cast<VkAccessFlags2>(srcMask);
    b2.dstStageMask = static_cast<VkPipelineStageFlags2>(dstStage);
    b2.dstAccessMask = static_cast<VkAccessFlags2>(dstMask);
    b2.oldLayout = oldLayout;
    b2.newLayout = newLayout;
    b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.image = image;
    b2.subresourceRange = b.subresourceRange;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b2;
    d.cmdPipelineBarrier2(cmd, &dep);
    return;
  }
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Buffer barrier making a transfer/source region visible to a later access (e.g. TRANSFER_WRITE -> vertex/index read).
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
  Sync2Dispatch & d = sync2Dispatch();
  if (d.cmdPipelineBarrier2 != nullptr) {
    VkBufferMemoryBarrier2 b2 {};
    b2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    b2.srcStageMask = static_cast<VkPipelineStageFlags2>(srcStage);
    b2.srcAccessMask = static_cast<VkAccessFlags2>(srcMask);
    b2.dstStageMask = static_cast<VkPipelineStageFlags2>(dstStage);
    b2.dstAccessMask = static_cast<VkAccessFlags2>(dstMask);
    b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.buffer = buffer;
    b2.offset = offset;
    b2.size = size;
    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &b2;
    d.cmdPipelineBarrier2(cmd, &dep);
    return;
  }
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &b, 0, nullptr);
}

// Run a small non-render-pass command buffer on `queue` and vkQueueWaitIdle before
// returning: allocate from `pool`, record via `record`, submit, free. On failure
// nothing is left allocated. Safe for setup/host-upload paths that must synchronously
// consume resources (the wait also retires any other in-flight work on the queue).
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
  // Retire in-flight queue work so submission-referenced resources are safe to destroy here.
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, pool, 1, &cmd);
  return ok;
}

// Copy a whole RGBA VkImage (oldLayout) to a host-visible staging buffer via a one-shot
// submit, then hand mapped pixels to `consume`. Transitioned to TRANSFER_SRC_OPTIMAL and
// back to restoreLayout around the copy; `pick` selects staging memory. The single
// image-to-host primitive behind the debug frame dumps; false on failure, nothing allocated.
inline bool
dumpImageToHost(VkDevice device, VkQueue queue, VkCommandPool pool,
                const VkAllocationCallbacks * allocator, VkImage image,
                VkImageLayout oldLayout, VkImageLayout restoreLayout,
                uint32_t width, uint32_t height, const MemoryTypePicker & pick,
                const std::function<void(const void *)> & consume)
{
  if (width == 0 || height == 0) return false;
  const VkDeviceSize size =
    static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  if (!createBufferAllocated(device, allocator, size,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             false, pick, staging, stagingMem)) {
    return false;
  }

  const bool ok = withOneShotSubmit(
    device, queue, pool, allocator, [&](VkCommandBuffer cmd) {
      SoVulkanImageCopy::recordToBuffer(
        cmd, image, staging, oldLayout, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, restoreLayout,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, width, height);
    });

  if (ok) {
    void * mapped = nullptr;
    if (vkMapMemory(device, stagingMem, 0, size, 0, &mapped) == VK_SUCCESS &&
        mapped != nullptr) {
      if (consume) consume(mapped);
      vkUnmapMemory(device, stagingMem);
    }
  }

  vkDestroyBuffer(device, staging, allocator);
  vkFreeMemory(device, stagingMem, allocator);
  return ok;
}

} // namespace SoVulkanShared

#endif // COIN_SOVULKANSHARED_H
