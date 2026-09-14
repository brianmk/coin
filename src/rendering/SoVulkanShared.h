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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include <vulkan/vulkan.h>

namespace SoVulkanShared {

// --- Environment access --------------------------------------------------
// Single choke point for every FC_VULKAN_* / FC_GUI_* environment lookup in
// Coin's Vulkan renderer.  Routing all reads through here keeps the opt-out
// policy in one place and makes the flags auditable; previously the same
// policy was re-implemented (and in one case inverted) at each getenv() site.

// Raw value (or nullptr).  Presence semantics: a variable set to any value,
// including "0", counts as set.  Use envFlagEnabled() when the conventional
// "VAR=0"/"false"/"off" opt-out must be honored.
inline const char *
envString(const char * name)
{
  return std::getenv(name);
}

// Presence-only test (any value, including "0"/"false"/"off").
inline bool
envSet(const char * name)
{
  return std::getenv(name) != nullptr;
}

// Integer / float value with a default when the variable is unset or empty.
inline int
envInt(const char * name, int defaultValue = 0)
{
  const char * value = std::getenv(name);
  return value ? std::atoi(value) : defaultValue;
}

inline float
envFloat(const char * name, float defaultValue = 0.0f)
{
  const char * value = std::getenv(name);
  return value ? static_cast<float>(std::atof(value)) : defaultValue;
}

// Environment flags honor the conventional "VAR=0"/"false"/"off" opt-out
// values.  A null value yields \a defaultValue, so a flag can be on unless
// explicitly disabled (e.g. the retained-IR replay).
inline bool
envFlagEnabled(const char * name, bool defaultValue)
{
  const char * value = std::getenv(name);
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

// --- Breadcrumb / phase timing -------------------------------------------
// The fcprobe profile harness keys on the monotonic microsecond clock and the
// FC_GUI_OPEN_BREADCRUMB gate.  Both backends and the manager used to carry
// their own copies of these primitives (three steady_clock->us converters and
// two near-identical "since" printers); they live here so the time base and
// the gating policy are singular.

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

inline bool
breadcrumbsEnabled()
{
  static const bool enabled = envFlagEnabled("FC_GUI_OPEN_BREADCRUMB");
  return enabled;
}

// Emit "PREFIX <startUs> <phase> dur_us=<elapsed>" once a phase has exceeded
// `thresholdUs`, up to `logged` (a caller-owned counter, so each translation
// unit keeps its own log budget exactly as the per-file statics did).
inline void
breadcrumbSince(int & logged, const char * prefix, long startUs,
                long thresholdUs, const char * phase)
{
  if (!breadcrumbsEnabled()) return;
  const long now = steadyNowUs();
  if (logged < 30 && now - startUs >= thresholdUs) {
    ++logged;
    std::fprintf(stderr, "%s %ld %s dur_us=%ld\n", prefix, startUs, phase,
                 now - startUs);
    std::fflush(stderr);
  }
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

// Copy a whole RGBA VkImage (currently in `oldLayout`) into a host-visible
// staging buffer with a one-shot submit, then hand the mapped pixels to
// `consume`.  The image is transitioned to TRANSFER_SRC_OPTIMAL for the copy
// and restored to `restoreLayout` before the submit.  `pick` selects the
// staging memory type (the backend's policy).  Returns false on any Vulkan
// failure, leaving nothing allocated.  This is the single image-to-host
// primitive behind the debug frame dumps.
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
      imageTransition(cmd, image, oldLayout,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
      VkBufferImageCopy region {};
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.layerCount = 1;
      region.imageExtent = {width, height, 1};
      vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             staging, 1, &region);
      imageTransition(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      restoreLayout, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
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
