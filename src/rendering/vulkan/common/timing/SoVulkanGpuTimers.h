// src/rendering/vulkan/common/timing/SoVulkanGpuTimers.h
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
  //! vkCmdWriteTimestamp requires the scope's queries to be unavailable, i.e.
  //! reset for this frame.  On the own-queue (internal) path the first
  //! beginScope() records that reset itself, before vkCmdBeginRenderPass.  On
  //! the caller-owned (external) path vkCmdResetQueryPool is illegal inside the
  //! caller's already-begun render pass, so the caller records resetSlot() on
  //! its command buffer BEFORE vkCmdBeginRenderPass; the begin/end writes then
  //! proceed inside the pass.
  void beginScope(VkCommandBuffer commandBuffer, const char * name);
  void endScope(VkCommandBuffer commandBuffer);

  //! Reset the current frame's query range on \a commandBuffer.  Must be
  //! recorded outside a render pass.  Afterwards slotReset() is true until the
  //! next endFrame().  A no-op when timing is disabled; safe to call.
  void resetSlot(VkCommandBuffer commandBuffer);
  //! True once resetSlot() has run for the current frame and endFrame() has not
  //! yet advanced the ring.  The external path only records scopes when this is
  //! true: a beginScope() inside the caller's pass cannot reset the pool.
  bool slotReset() const { return this->slotResetForFrame; }

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
  //! True once this frame's query range has been reset (resetSlot() or the
  //! first beginScope()), so no later begin() resets it again.
  bool slotResetForFrame = false;
  uint32_t slotScopeCount[kRingFrames] = {};
  const char * scopeNames[kRingFrames][kMaxScopesPerFrame] = {};
};

#endif // COIN_SOVULKANGPUTIMERS_H
