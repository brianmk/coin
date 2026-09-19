// src/rendering/SoVulkanGpuTimers.h
//
// Per-pass GPU timestamps for the Vulkan renderer.  Internal to Coin (not
// installed, not public API).
//
// Uses a VkQueryPool of VK_QUERY_TYPE_TIMESTAMP queries, ringed over a few
// frames so results are read back only once they are complete (no pipeline
// stall).  Gated by SoVulkanConfig::get().diagnostics.gpuTimestamps
// (FC_VULKAN_GPU_TIMING); when disabled, or when the device/queue family has no
// timestamp support, every method is a no-op.
//
// The caller brackets passes with beginScope()/endScope() while recording, then
// calls endFrame() once per submitted frame.  endFrame() reads back the frame
// kRingFrames-1 frames old and prints one "[RTDBG] gpuTiming <scope>=<ms>"
// line per scope, matching the existing cpuTimingRaster diagnostic style.

#ifndef COIN_SOVULKANGPUTIMERS_H
#define COIN_SOVULKANGPUTIMERS_H

#include <cstdint>

#include <vulkan/vulkan.h>

class SoVulkanGpuTimers {
public:
  //! Maximum scopes recorded per frame.  A frame that opens more is truncated
  //! (the extra scopes are dropped, not misattributed).
  static constexpr uint32_t kMaxScopesPerFrame = 16;
  //! Frames in the timestamp ring.  Reading back the oldest slot hides the
  //! latency between submission and query availability without waiting.
  static constexpr uint32_t kRingFrames = 4;

  SoVulkanGpuTimers() = default;
  ~SoVulkanGpuTimers();
  SoVulkanGpuTimers(const SoVulkanGpuTimers &) = delete;
  SoVulkanGpuTimers & operator=(const SoVulkanGpuTimers &) = delete;

  //! Create the query pool.  Returns false (and stays disabled) when the device
  //! or the given queue family cannot write timestamps.
  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                  uint32_t queueFamilyIndex);
  bool initialized() const { return this->queryPool != VK_NULL_HANDLE; }

  //! Open/close a named scope on \a commandBuffer.  Nesting is not supported:
  //! scopes are recorded as a flat sequence of begin/end pairs.
  //!
  //! Only valid on the own-queue path, where the slot's queries can be reset
  //! before vkCmdBeginRenderPass: vkCmdWriteTimestamp requires the query to be
  //! unavailable, and vkCmdResetQueryPool is illegal inside a render pass, so
  //! the caller-owned (external) path cannot use this yet.
  void beginScope(VkCommandBuffer commandBuffer, const char * name);
  void endScope(VkCommandBuffer commandBuffer);

  //! Advance the ring and read back the oldest completed frame.
  void endFrame();

  //! Destroy the query pool.  Must be called before the VkDevice is destroyed
  //! (the backend calls it from its own shutdown while the device is alive).
  void shutdown();

private:
  VkDevice device = VK_NULL_HANDLE;
  VkQueryPool queryPool = VK_NULL_HANDLE;
  float timestampPeriod = 0.0f;
  uint32_t timestampValidBits = 0;

  uint32_t ringIndex = 0;
  uint32_t scopeCount = 0;
  //! True between a recorded beginScope() and its matching endScope().  A
  //! dropped begin (max scopes reached) clears it, so the paired endScope() is
  //! a no-op and cannot overwrite the previous scope's end timestamp.
  bool scopePending = false;
  uint32_t slotScopeCount[kRingFrames] = {};
  const char * scopeNames[kRingFrames][kMaxScopesPerFrame] = {};
};

#endif // COIN_SOVULKANGPUTIMERS_H
