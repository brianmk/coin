// src/rendering/vulkan/common/timing/SoVulkanGpuTimers.cpp
#include "rendering/vulkan/common/timing/SoVulkanGpuTimers.h"

#include <cstdio>
#include <vector>

SoVulkanGpuTimers::~SoVulkanGpuTimers()
{
  this->shutdown();
}

void
SoVulkanGpuTimers::shutdown()
{
  if (this->queryPool != VK_NULL_HANDLE && this->device != VK_NULL_HANDLE) {
    vkDestroyQueryPool(this->device, this->queryPool, nullptr);
  }
  this->queryPool = VK_NULL_HANDLE;
}

bool
SoVulkanGpuTimers::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                              uint32_t queueFamilyIndex)
{
  if (this->queryPool != VK_NULL_HANDLE) {
    return true;
  }
  if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE) {
    return false;
  }

  VkPhysicalDeviceProperties props {};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  if (props.limits.timestampPeriod <= 0.0f) {
    std::fprintf(stderr,
                 "[RTDBG] gpuTiming disabled (device has no usable timestamps)\n");
    return false;
  }

  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount,
                                           nullptr);
  if (queueFamilyIndex >= familyCount) {
    return false;
  }
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount,
                                           families.data());
  if (families[queueFamilyIndex].timestampValidBits == 0) {
    std::fprintf(stderr,
                 "[RTDBG] gpuTiming disabled (queue family %u has no timestamps)\n",
                 queueFamilyIndex);
    return false;
  }

  VkQueryPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  ci.queryCount = kRingFrames * kMaxScopesPerFrame * 2;
  if (vkCreateQueryPool(device, &ci, nullptr, &this->queryPool) != VK_SUCCESS) {
    this->queryPool = VK_NULL_HANDLE;
    return false;
  }

  this->device = device;
  this->timestampPeriod = props.limits.timestampPeriod;
  this->timestampValidBits = families[queueFamilyIndex].timestampValidBits;
  return true;
}

void
SoVulkanGpuTimers::beginScope(VkCommandBuffer commandBuffer, const char * name)
{
  if (this->queryPool == VK_NULL_HANDLE ||
      commandBuffer == VK_NULL_HANDLE ||
      this->scopeCount >= kMaxScopesPerFrame) {
    // No begin was recorded, so the matching endScope() must not write either:
    // otherwise it would close the previous scope's pair a second time.
    this->scopePending = false;
    return;
  }
  const uint32_t slot = this->ringIndex;
  const uint32_t base = slot * kMaxScopesPerFrame * 2;
  if (this->scopeCount == 0) {
    // Reset this slot's query range before the first write: vkCmdWriteTimestamp
    // requires the query to be unavailable, and the slot's previous results
    // were read back kRingFrames ago.  Must run outside a render pass.
    vkCmdResetQueryPool(commandBuffer, this->queryPool, base,
                        kMaxScopesPerFrame * 2);
  }
  this->scopeNames[slot][this->scopeCount] = name;
  vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      this->queryPool, base + this->scopeCount * 2);
  ++this->scopeCount;
  this->scopePending = true;
}

void
SoVulkanGpuTimers::endScope(VkCommandBuffer commandBuffer)
{
  if (this->queryPool == VK_NULL_HANDLE || commandBuffer == VK_NULL_HANDLE ||
      !this->scopePending) {
    return;
  }
  const uint32_t slot = this->ringIndex;
  const uint32_t base = slot * kMaxScopesPerFrame * 2;
  vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      this->queryPool,
                      base + (this->scopeCount - 1) * 2 + 1);
  this->scopePending = false;
}

void
SoVulkanGpuTimers::endFrame()
{
  if (this->queryPool == VK_NULL_HANDLE) {
    return;
  }
  // An unmatched begin (no endScope this frame) must not leak into the next.
  this->scopePending = false;
  const uint32_t slot = this->ringIndex;
  this->slotScopeCount[slot] = this->scopeCount;
  this->scopeCount = 0;
  this->ringIndex = (this->ringIndex + 1) % kRingFrames;

  // After advancing, ringIndex is the oldest slot: its submission has had
  // kRingFrames-1 frames to complete, so the results are normally ready.
  const uint32_t readSlot = this->ringIndex;
  uint32_t count = this->slotScopeCount[readSlot];
  if (count > kMaxScopesPerFrame) {
    count = kMaxScopesPerFrame;
  }
  if (count == 0) {
    return;
  }

  uint64_t data[kMaxScopesPerFrame * 2] = {};
  const VkResult result = vkGetQueryPoolResults(
    this->device, this->queryPool, readSlot * kMaxScopesPerFrame * 2,
    count * 2, sizeof(data), data, sizeof(uint64_t),
    VK_QUERY_RESULT_64_BIT);
  if (result != VK_SUCCESS) {
    // VK_NOT_READY: the frame is still in flight; skip it rather than stall.
    return;
  }

  const uint64_t mask = this->timestampValidBits >= 64
    ? ~static_cast<uint64_t>(0)
    : ((static_cast<uint64_t>(1) << this->timestampValidBits) - 1);
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t begin = data[i * 2] & mask;
    const uint64_t end = data[i * 2 + 1] & mask;
    const double ms = end >= begin
      ? static_cast<double>(end - begin) *
          static_cast<double>(this->timestampPeriod) / 1.0e6
      : 0.0;
    std::fprintf(stderr, "[RTDBG] gpuTiming %s=%.3fms\n",
                 this->scopeNames[readSlot][i] ? this->scopeNames[readSlot][i]
                                               : "scope",
                 ms);
  }
  std::fflush(stderr);
}
