// src/rendering/SoVulkanRenderBackend/SoVulkanRecordContext.h
//
// Per-recording command-buffer target plus the dedup dynamic-state/descriptor
// cache used while recording it.
//
// Recording remembers the state bound to skip redundant vkCmd* calls (see
// applyPipeline/applyViewportState/applyScissorState).  Backend-member caches
// limited recording to one thread; this context lets each (future) worker record
// with its own cache.  Descriptors/pipelines/rings stay backend-owned, read-only.

#ifndef COIN_SOVULKANRECORDCONTEXT_H
#define COIN_SOVULKANRECORDCONTEXT_H

#include <cstdint>

#include "rendering/SoVulkanPlatform.h"
#include <vulkan/vulkan.h>

struct VulkanRecordContext {
  // Command buffer being recorded: render() = backend's, renderExternal() = caller's.
  VkCommandBuffer buffer = VK_NULL_HANDLE;

  // Per-recording lighting/instance-model slot cursor, per worker so concurrent
  // recorders advance their own into disjoint (race-free) ring regions; planted at
  // an item's pre-assigned slotBase (M1b) before each item.
  uint32_t uboCmdIndex = 0;

  // Last dynamic state bound into `buffer` (see applyPipeline/applyViewportState/
  // applyScissorState); reset per frame, not per in-flight slot, so a reused slot's
  // prior content never suppresses a needed change.
  VkPipeline lastBoundPipeline = VK_NULL_HANDLE;
  VkViewport lastBoundViewport {};
  VkRect2D lastBoundScissor {};
  bool hasBoundViewport = false;
  bool hasBoundScissor = false;
  // Per-frame descriptor-bind caches: a frame usually shares one lighting handle
  // and one texture, avoiding an unordered_map lookup per draw; set 1 must still
  // re-bind every draw (its dynamic offset advances).
  uint32_t lastLightingHandle = UINT32_MAX;
  uint32_t lastLightingOffset = 0;
  uint32_t lastBoundLightingOffset = UINT32_MAX;
  VkDescriptorSet lastBoundTextureSet = VK_NULL_HANDLE;

  // Forget bound state (frame boundary); leaves `buffer` alone (set after reset).
  void reset()
  {
    uboCmdIndex = 0;
    lastBoundPipeline = VK_NULL_HANDLE;
    hasBoundViewport = false;
    hasBoundScissor = false;
    lastLightingHandle = UINT32_MAX;
    lastLightingOffset = 0;
    lastBoundLightingOffset = UINT32_MAX;
    lastBoundTextureSet = VK_NULL_HANDLE;
  }
};

#endif // COIN_SOVULKANRECORDCONTEXT_H
