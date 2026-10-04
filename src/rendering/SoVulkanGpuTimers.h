// src/rendering/SoVulkanGpuTimers.h
//
// Per-pass GPU timestamps for the Vulkan renderer.  Internal, not public API.
// A VkQueryPool of VK_QUERY_TYPE_TIMESTAMP queries ringed over kRingFrames, so
// results are read only once complete (no pipeline stall).  Gated by
// diagnostics.gpuTimestamps (COIN_VULKAN_GPU_TIMING); disabled, or without device/
// queue-family timestamp support, every method is a no-op.  Callers bracket passes
// with beginScope()/endScope(), then endFrame(): it reads back the frame
// kRingFrames-1 old and prints "[RTDBG] gpuTiming <scope>=<ms>" (cf. cpuTimingRaster).

#ifndef COIN_SOVULKANGPUTIMERS_H
#define COIN_SOVULKANGPUTIMERS_H

#include <cstdint>

#include "rendering/SoVulkanPlatform.h"
#include <vulkan/vulkan.h>

class SoVulkanGpuTimers {
public:
  //! Max scopes per frame; excess scopes are dropped, not misattributed.
  static constexpr uint32_t kMaxScopesPerFrame = 16;
  //! Timestamp ring size; reading the oldest slot hides submission latency.
  static constexpr uint32_t kRingFrames = 4;

  SoVulkanGpuTimers() = default;
  ~SoVulkanGpuTimers();
  SoVulkanGpuTimers(const SoVulkanGpuTimers &) = delete;
  SoVulkanGpuTimers & operator=(const SoVulkanGpuTimers &) = delete;

  //! Create the query pool; false (stays disabled) if timestamps unsupported.
  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                  uint32_t queueFamilyIndex);
  bool initialized() const { return this->queryPool != VK_NULL_HANDLE; }

  //! Open/close a named scope on \a commandBuffer.  Flat, non-nested.
  //!
  //! vkCmdWriteTimestamp needs the scope's queries reset for this frame.  The
  //! own-queue path resets on the first beginScope() (before vkCmdBeginRenderPass);
  //! the caller-owned path cannot reset in-pass, so it calls resetSlot() first.
  void beginScope(VkCommandBuffer commandBuffer, const char * name);
  void endScope(VkCommandBuffer commandBuffer);

  //! Reset this frame's query range on \a commandBuffer; must be outside a pass.
  void resetSlot(VkCommandBuffer commandBuffer);
  //! True once resetSlot() ran this frame: the external path records scopes only then.
  bool slotReset() const { return this->slotResetForFrame; }

  //! Advance the ring and read back the oldest completed frame.
  void endFrame();

  //! Destroy the query pool (before the VkDevice; backend calls at shutdown).
  void shutdown();

private:
  VkDevice device = VK_NULL_HANDLE;
  VkQueryPool queryPool = VK_NULL_HANDLE;
  float timestampPeriod = 0.0f;
  uint32_t timestampValidBits = 0;

  uint32_t ringIndex = 0;
  uint32_t scopeCount = 0;
  //! A beginScope() without its endScope() yet, so an unmatched begin cannot leak.
  bool scopePending = false;
  //! This frame's query range was reset; no later begin() resets it again.
  bool slotResetForFrame = false;
  uint32_t slotScopeCount[kRingFrames] = {};
  const char * scopeNames[kRingFrames][kMaxScopesPerFrame] = {};
};

#endif // COIN_SOVULKANGPUTIMERS_H
