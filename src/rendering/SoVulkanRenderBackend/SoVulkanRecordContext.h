// src/rendering/SoVulkanRenderBackend/SoVulkanRecordContext.h
//
// Per-recording command-buffer target plus the deduplicated dynamic-state
// cache used while recording it.
//
// Recording "remembers the state actually bound so far" to skip redundant
// vkCmd* calls (see applyPipeline/applyViewportState/applyScissorState and
// the descriptor-bind caches).  As long as that cache lives in backend
// members only one thread can ever record.  Lifting it into a small context
// struct lets each (future) worker thread record its own command buffer with
// its own dedup cache: every record* helper takes a `VulkanRecordContext &`
// and touches no other recording state.  Descriptors, pipelines and uniform
// rings stay backend-owned and read-only during recording.

#ifndef COIN_SOVULKANRECORDCONTEXT_H
#define COIN_SOVULKANRECORDCONTEXT_H

#include <cstdint>

#include <vulkan/vulkan.h>

struct VulkanRecordContext {
  // Command buffer being recorded.  Assigned by the render entry point
  // (the backend's own buffer in render(), the caller's buffer in
  // renderExternal()); cleared back to null when recording ends.
  VkCommandBuffer buffer = VK_NULL_HANDLE;

  // Per-recording lighting/instance-model slot cursor.  Moved out of a
  // backend member so each (future parallel) worker thread owns its own: the
  // record path plants this at an item's pre-assigned slotBase (M1b) before
  // each item, and recordDrawCommand/recordCommandBatch advance it, so
  // concurrent workers each advance their own cursor while writing disjoint
  // (race-free) ring regions.
  uint32_t uboCmdIndex = 0;

  // Last dynamic state actually bound into `buffer` (see applyPipeline /
  // applyViewportState / applyScissorState).  Reset once per frame, not per
  // in-flight slot: dynamic state is per-recording, so a reused slot's prior
  // content must never suppress a needed state change.
  VkPipeline lastBoundPipeline = VK_NULL_HANDLE;
  VkViewport lastBoundViewport {};
  VkRect2D lastBoundScissor {};
  bool hasBoundViewport = false;
  bool hasBoundScissor = false;
  // Per-frame descriptor-bind caches.  A frame typically shares one lighting
  // handle and one (usually the white) texture, so caching the resolved
  // set/offset avoids an unordered_map lookup per draw and lets set 0
  // re-bind only when the lighting handle actually changes.  Set 1 must
  // still re-bind every draw because its dynamic offset (the per-draw
  // view/model/material slot) advances each draw.
  uint32_t lastLightingHandle = UINT32_MAX;
  uint32_t lastLightingOffset = 0;
  uint32_t lastBoundLightingOffset = UINT32_MAX;
  VkDescriptorSet lastBoundTextureSet = VK_NULL_HANDLE;

  // Forget the bound state (frame boundary).  Deliberately leaves `buffer`
  // alone: the entry point assigns the buffer after the reset.
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
