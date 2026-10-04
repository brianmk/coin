// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendCommand.cpp
//
// Per-draw command recording: dynamic viewport/scissor state (with the Coin
// bottom-left -> Vulkan top-left Y-flip), clears, the lighting/material UBO
// write, the draw itself (plain, indexed or wide-line), and command-buffer
// begin/submit.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanDebug.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace CoinVulkanDetail;

namespace {

// Pack the per-draw push-constant block; shared by single-draw and batch recorders.
VulkanPushConstants
packPushConstants(const SoRenderCommand & command,
                  const VulkanCachedCommand & entry,
                  const float * uniformColorOverride,
                  const float dpr,
                  const float stippleFactor, const float stipplePatternBits,
                  const bool wideLine,
                  const float lineWidthPx = 0.0f,
                  const float viewportWidthPx = 0.0f,
                  const float viewportHeightPx = 0.0f)
{
  VulkanPushConstants push {};
  const SbVec4f & color = command.material.diffuse;
  const bool useOverrideColor = uniformColorOverride != nullptr;
  push.color[0] = useOverrideColor ? uniformColorOverride[0] : color[0];
  push.color[1] = useOverrideColor ? uniformColorOverride[1] : color[1];
  push.color[2] = useOverrideColor ? uniformColorOverride[2] : color[2];
  push.color[3] = useOverrideColor ? uniformColorOverride[3] : color[3];
  push.flags[0] = (entry.colorKey && !useOverrideColor) ? 1.0f : 0.0f;
  push.flags[1] =
    command.material.vertexColorAlphaIncludesOpacity ? 1.0f : 0.0f;
  const bool textured = command.material.texture.pixels &&
                        command.material.texture.width > 0 &&
                        command.material.texture.height > 0;
  push.flags[2] = (textured && !useOverrideColor) ? 1.0f : 0.0f;
  push.flags[3] = command.material.textureAlphaIncludesOpacity ? 1.0f : 0.0f;
  push.texParams[0] = static_cast<float>(command.material.texture.model);
  push.texParams[1] = static_cast<float>(command.state.alphaTest.function);
  push.texParams[2] = command.state.alphaTest.reference;
  push.texParams[3] =
    (command.material.flags & SO_MAT_IS_PIXEL_TEXT) ? 1.0f : 0.0f;
  const SbVec4f & blendColor = command.material.texture.blendColor;
  push.texBlend[0] = blendColor[0];
  push.texBlend[1] = blendColor[1];
  push.texBlend[2] = blendColor[2];
  push.texBlend[3] = blendColor[3];
  // Point size scaled by DPR (viewport is device px; SoDrawStyle is logical).
  push.pointSize = std::max(1.0f, command.state.raster.pointSize) * dpr;
  push.lineParams[0] = stippleFactor;
  // lineParams.y is shared: wide-line reads stipple bits, visual reads round-point.
  push.lineParams[1] = (wideLine && stipplePatternBits != 0.0f)
    ? stipplePatternBits
    : (command.state.raster.pointShape == SO_POINT_SHAPE_ROUND ? 1.0f : 0.0f);
  // Point primitives (Sketcher vertex dots) are round; no marker-bitmap rasterization here.
  if (command.geometry.topology == SO_TOPOLOGY_POINTS) {
    push.lineParams[1] = 1.0f;
  }
  push.lineParams[2] = wideLine ? 1.0f : 0.0f;
  push.lineParams[3] =
    command.geometry.topology == SO_TOPOLOGY_POINTS ? 1.0f : 0.0f;
  // GPU wide-line quad sizing: line width and viewport size in device pixels.
  push.lineGeom[0] = lineWidthPx;
  push.lineGeom[1] = viewportWidthPx;
  push.lineGeom[2] = viewportHeightPx;
  push.lineGeom[3] = dpr;
  return push;
}

} // namespace

void
SoVulkanRenderBackend::resetBoundState(VulkanRecordContext & ctx)
{
  ctx.reset();
}

void
SoVulkanRenderBackend::applyPipeline(const VkPipeline pipeline,
                                     VulkanRecordContext & ctx)
{
  if (pipeline == ctx.lastBoundPipeline) return;
  vkCmdBindPipeline(ctx.buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline);
  ctx.lastBoundPipeline = pipeline;
}

void
SoVulkanRenderBackend::applyViewportState(const VkViewport & viewport,
                                          VulkanRecordContext & ctx)
{
  if (ctx.hasBoundViewport &&
      std::memcmp(&ctx.lastBoundViewport, &viewport, sizeof(viewport)) == 0) {
    return;
  }
  vkCmdSetViewport(ctx.buffer, 0, 1, &viewport);
  ctx.lastBoundViewport = viewport;
  ctx.hasBoundViewport = true;
}

void
SoVulkanRenderBackend::applyScissorState(const VkRect2D & scissor,
                                         VulkanRecordContext & ctx)
{
  if (ctx.hasBoundScissor &&
      std::memcmp(&ctx.lastBoundScissor, &scissor, sizeof(scissor)) == 0) {
    return;
  }
  vkCmdSetScissor(ctx.buffer, 0, 1, &scissor);
  ctx.lastBoundScissor = scissor;
  ctx.hasBoundScissor = true;
}

void
SoVulkanRenderBackend::applyViewport(const SoRenderParams & params,
                                     const SoVulkanRenderTarget & target,
                                     VulkanRecordContext & ctx)
{
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();

  // Coin/GL viewport origin is bottom-left, Vulkan's top-left.  The vertex
  // shader flips Y in clip space, so re-anchor the rect to the top edge to cancel.
  VkViewport viewport {};
  viewport.x = static_cast<float>(origin[0]);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(origin[1]) -
                                 static_cast<int32_t>(size[1]));
  viewport.width = static_cast<float>(size[0]);
  viewport.height = static_cast<float>(size[1]);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  this->applyViewportState(viewport, ctx);

  // Clamp to the target so an off-screen viewport never clears outside the render area.
  this->applyScissorState(
    toVkRect(clampFlippedRect(origin[0], origin[1], size[0], size[1],
                              target.extent)),
    ctx);
}

// Apply a per-command viewport (from SoViewportRegionElement); commands without
// one keep the frame viewport from applyViewport().  Same Y-flip math.
void
SoVulkanRenderBackend::applyCommandViewport(const SoRenderCommand & command,
                                            const SoVulkanRenderTarget & target,
                                            VulkanRecordContext & ctx)
{
  const SoRasterState & raster = command.state.raster;
  if (!raster.viewportEnabled || raster.viewportWidth <= 0 ||
      raster.viewportHeight <= 0) {
    return;
  }
  VkViewport viewport {};
  viewport.x = static_cast<float>(raster.viewportX);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(raster.viewportY) -
                                 static_cast<int32_t>(raster.viewportHeight));
  viewport.width = static_cast<float>(raster.viewportWidth);
  viewport.height = static_cast<float>(raster.viewportHeight);
  // Depth range from SoDepthBufferElement; dynamic viewport state means no restore needed.
  viewport.minDepth =
    std::clamp(command.state.depth.range[0], 0.0f, 1.0f);
  viewport.maxDepth =
    std::clamp(command.state.depth.range[1], 0.0f, 1.0f);
  this->applyViewportState(viewport, ctx);

  // The per-command viewport also bounds the draw region; mirror applyViewport()'s clamp.
  this->applyScissorState(
    toVkRect(clampFlippedRect(raster.viewportX, raster.viewportY,
                              raster.viewportWidth, raster.viewportHeight,
                              target.extent)),
    ctx);
}

void
SoVulkanRenderBackend::applyScissor(const SoRenderCommand & command,
                                    const SoVulkanRenderTarget & target,
                                    VulkanRecordContext & ctx)
{
  VkRect2D scissor {};
  const SoRasterState & raster = command.state.raster;
  if (raster.scissorEnabled && raster.scissorWidth > 0 &&
      raster.scissorHeight > 0) {
    // Coin/GL scissors are bottom-left anchored; mirror the viewport Y-flip.
    const int32_t flippedY = static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(raster.scissorY) -
      static_cast<int32_t>(raster.scissorHeight);
    scissor.offset = {static_cast<int32_t>(raster.scissorX), flippedY};
    scissor.extent = {static_cast<uint32_t>(raster.scissorWidth),
                      static_cast<uint32_t>(raster.scissorHeight)};
  }
  else {
    scissor.offset = {0, 0};
    scissor.extent = target.extent;
  }
  this->applyScissorState(scissor, ctx);
}

bool
SoVulkanRenderBackend::isFullTargetClear(const SoRenderParams & params,
                                         const SoVulkanRenderTarget & target) const
{
  // Mirror recordClear()'s region math: a full-target clear means the clamped
  // viewport covers the whole attachment, so the render-pass loadOp can clear.
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  const FlippedRect r = clampFlippedRect(origin[0], origin[1], size[0], size[1],
                                         target.extent);
  return r.x0 == 0 && r.y0 == 0 &&
    r.x1 == static_cast<int32_t>(target.extent.width) &&
    r.y1 == static_cast<int32_t>(target.extent.height);
}

void
SoVulkanRenderBackend::recordClear(const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target,
                                   bool colorClearedByLoad,
                                   bool depthClearedByLoad,
                                   VulkanRecordContext & ctx)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;

  VkClearAttachment attachments[3];
  uint32_t attachmentCount = 0;

  if (params.flags & SO_PARAM_CLEAR_WINDOW) {
    // Full-target clear fast path already cleared via loadOp to clearColor, so
    // skip the matching vkCmdClearAttachments.
    if (!colorClearedByLoad) {
      const SbColor4f & color = params.clearColor;
      VkClearAttachment clear {};
      clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      clear.colorAttachment = 0;
      clear.clearValue.color.float32[0] = color[0];
      clear.clearValue.color.float32[1] = color[1];
      clear.clearValue.color.float32[2] = color[2];
      clear.clearValue.color.float32[3] = color[3];
      attachments[attachmentCount++] = clear;
    }
  }

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_DEPTH)) {
    if (!depthClearedByLoad) {
      VkClearAttachment clear {};
      clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      clear.colorAttachment = 0;
      clear.clearValue.depthStencil.depth = params.clearDepth;
      clear.clearValue.depthStencil.stencil = 0;
      attachments[attachmentCount++] = clear;
    }
  }

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_STENCIL)) {
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.depthStencil.depth = 0;
    clear.clearValue.depthStencil.stencil = params.clearStencil;
    attachments[attachmentCount++] = clear;
  }

  if (attachmentCount == 0) return;

  // Clear only the viewport region (Y-flipped); the full target would clobber other viewports.
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  const FlippedRect r = clampFlippedRect(origin[0], origin[1], size[0], size[1],
                                         target.extent);
  if (r.x1 <= r.x0 || r.y1 <= r.y0) return;

  VkClearRect rect {};
  rect.rect = toVkRect(r);
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(ctx.buffer, attachmentCount, attachments, 1,
                        &rect);
}

void
SoVulkanRenderBackend::recordOverlayDepthClear(const SoRenderCommand & command,
                                               const SoVulkanRenderTarget & target,
                                               VulkanRecordContext & ctx)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  if (!hasDepth) {
    return;
  }

  // Overlay rect is Coin/GL bottom-left; mirror applyScissor()'s Y-flip.
  const SoRasterState & raster = command.state.raster;
  const FlippedRect r = clampFlippedRect(raster.scissorX, raster.scissorY,
                                         raster.scissorWidth,
                                         raster.scissorHeight, target.extent);
  if (r.x1 <= r.x0 || r.y1 <= r.y0) {
    return;
  }

  VkClearAttachment attachment {};
  attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  attachment.colorAttachment = 0;
  attachment.clearValue.depthStencil.depth = 1.0f;
  attachment.clearValue.depthStencil.stencil = 0;

  VkClearRect rect {};
  rect.rect = toVkRect(r);
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(ctx.buffer, 1, &attachment, 1, &rect);
}

bool
SoVulkanRenderBackend::updateLightingSetup(const SoDrawList & drawlist)
{
  vkBackendTrace(this->uboFrameIndex, "updateLightingSetup.enter", "cmds=%d",
                 drawlist.getNumCommands());
  // Build one lighting constant block per distinct lightingHandle per frame
  // (into the fixed 8-slot ring), referenced by each draw via its dynamic offset.
  this->lightingSlotOffsets.clear();
  if (this->lightingConstMapped == nullptr || this->lightingConstStride == 0) {
    return false;
  }

  // Ring base for this frame (fixed 8-frame ring, 8 slots per frame).
  const uint32_t frameBase = (this->uboFrameIndex % 8u) * 8u;

  static const SoLightingData emptyLighting;
  uint32_t occupiedSlots = 0;

  // Setups are world-space but the shaders light in eye space, so pack through
  // the frame view (cacheFrameMatrices() ran first).
  SbMat frameViewMat;
  std::memcpy(frameViewMat, this->frameViewFloats, sizeof(float) * 16);
  const SbMatrix frameView(frameViewMat);

  // Seed slot 0 with empty lighting so the handle-0 fallback is always valid.
  {
    const VkDeviceSize offset =
      static_cast<VkDeviceSize>(frameBase + occupiedSlots) *
      this->lightingConstStride;
    VulkanLightingUbo * u = reinterpret_cast<VulkanLightingUbo *>(
      static_cast<char *>(this->lightingConstMapped) + offset);
    std::memset(u, 0, sizeof(VulkanLightingUbo));
    this->lightingSlotOffsets[0] = offset;
    occupiedSlots = 1;
  }

  // GL host pushed authoritative viewer lights: pack that single camera-anchored
  // set once and point every command at it (world-space, through the frame view).
  VkDeviceSize sceneLightOffset = 0;
  if (!this->sceneLighting.lights.empty()) {
    const uint32_t slot = std::min(occupiedSlots, 7u);
    const VkDeviceSize offset =
      static_cast<VkDeviceSize>(frameBase + slot) * this->lightingConstStride;
    VulkanLightingUbo * u = reinterpret_cast<VulkanLightingUbo *>(
      static_cast<char *>(this->lightingConstMapped) + offset);
    SoRenderIR::fillLightingBlock(*u, this->sceneLighting, &frameView);
    sceneLightOffset = offset;
    ++occupiedSlots;
  }

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const SoLightingHandle handle = command.lightingHandle;
    if (handle == 0 ||
        this->lightingSlotOffsets.find(handle) !=
          this->lightingSlotOffsets.end()) {
      continue;
    }
    if (sceneLightOffset != 0) {
      this->lightingSlotOffsets[handle] = sceneLightOffset;
      continue;
    }
    const SoLightingData * lighting = drawlist.getLighting(handle);
    if (!lighting) lighting = &emptyLighting;

    // Clamp to the per-frame slot budget (8); overflow reuses the last slot in bounds.
    const uint32_t slot = std::min(occupiedSlots, 7u);
    const VkDeviceSize offset =
      static_cast<VkDeviceSize>(frameBase + slot) * this->lightingConstStride;
    VulkanLightingUbo * u = reinterpret_cast<VulkanLightingUbo *>(
      static_cast<char *>(this->lightingConstMapped) + offset);
    SoRenderIR::fillLightingBlock(*u, *lighting, &frameView);
    this->lightingSlotOffsets[handle] = offset;
    ++occupiedSlots;
  }
  return true;
}

VkDeviceSize
SoVulkanRenderBackend::lightingOffsetFor(const SoRenderCommand & command) const
{
  const auto found = this->lightingSlotOffsets.find(command.lightingHandle);
  if (found != this->lightingSlotOffsets.end()) {
    return found->second;
  }
  // Unreachable in practice (updateLightingSetup() visits every command); fall
  // back to the seeded handle-0 empty slot.
  const auto zero = this->lightingSlotOffsets.find(0);
  if (zero != this->lightingSlotOffsets.end()) {
    return zero->second;
  }
  return 0;
}

void
SoVulkanRenderBackend::updateLightingUniforms(const SoDrawList & drawlist,
                                              const SoRenderCommand & command,
                                              const SoRenderParams & params,
                                              const VkDeviceSize uboOffset,
                                              const bool unlit,
                                              const float * projFloats)
{
  // Per-draw block: view/model/material + projection; the lighting constant block is in set 0.
  VulkanDrawUbo ubo {};
  if (projFloats) {
    std::memcpy(ubo.proj, projFloats, sizeof(float) * 16);
  }

  SbMat m;
  // Overlay geometry spanning the frame (selection highlight) is frame-camera
  // geometry: it must use the frame view/proj, not the scene camera's stale,
  // one-frame-lagging matrices.  Overlays with their own viewport keep theirs.
  const bool frameCameraOverlay = isFrameCameraOverlay(command, params);
  if (command.state.raster.scissorEnabled
      && command.pass == SO_RENDERPASS_OVERLAY && !frameCameraOverlay) {
    command.viewMatrix.getValue(m);
  }
  else {
    // Frame view, converted once per render (cacheFrameMatrices).
    std::memcpy(m, this->frameViewFloats, sizeof(float) * 16);
  }
  std::memcpy(ubo.view, &m[0][0], sizeof(float) * 16);
  command.modelMatrix.getValue(m);
  std::memcpy(ubo.model, &m[0][0], sizeof(float) * 16);

  const SoMaterialData & material = command.material;
  ubo.emissive[0] = material.emissive[0];
  ubo.emissive[1] = material.emissive[1];
  ubo.emissive[2] = material.emissive[2];
  ubo.emissive[3] = 1.0f;
  ubo.materialAmbient[0] = material.ambient[0];
  ubo.materialAmbient[1] = material.ambient[1];
  ubo.materialAmbient[2] = material.ambient[2];
  ubo.materialAmbient[3] = 1.0f;
  ubo.materialSpecular[0] = material.specular[0];
  ubo.materialSpecular[1] = material.specular[1];
  ubo.materialSpecular[2] = material.specular[2];
  ubo.materialSpecular[3] = 1.0f;
  ubo.materialParams[0] = material.shininess;
  ubo.materialParams[1] = material.twoSidedLighting ? 1.0f : 0.0f;
  ubo.materialParams[3] = unlit
    ? 0.0f
    : (material.shadingModel == SO_SHADING_LEGACY_GOURAUD ? 1.0f : 0.0f);

  // Light count: from scene lighting when the host pushed it, else the command's IR capture.
  int count = 0;
  if (!this->sceneLighting.lights.empty()) {
    count = this->sceneLighting.lightCount();
  }
  else {
    const SoLightingData * lighting =
      drawlist.getLighting(command.lightingHandle);
    static const SoLightingData emptyLighting;
    if (!lighting) lighting = &emptyLighting;
    count = lighting->lightCount();
  }
  ubo.materialParams[2] = static_cast<float>(count);

  if (this->lightingMapped && this->uboSlotStride > 0) {
    std::memcpy(static_cast<char *>(this->lightingMapped) + uboOffset,
                &ubo, sizeof(ubo));
  }
}



VkDeviceSize
SoVulkanRenderBackend::uboSlotOffset(const uint32_t slotIndex) const
{
  return ((this->uboFrameIndex % this->maxFramesInFlight) *
            this->uboSlotsPerFrame + slotIndex) * this->uboSlotStride;
}

void
SoVulkanRenderBackend::bindDrawDescriptors(const SoRenderCommand & command,
                                           const uint32_t uboDynamicOffset,
                                           const uint32_t slotIndex,
                                           VulkanRecordContext & ctx)
{
  // Bind set 0 (lighting constant, per-handle dynamic offset; written once by
  // updateLightingSetup()) and set 1 (per-draw UBO + texture, dynamic offset).
  VkDescriptorSet textureSet = this->resolveTextureSet(command);
  if (textureSet == VK_NULL_HANDLE) {
    textureSet = this->whiteDescriptorSet;
  }
  // Cache the lighting dynamic offset per handle: retained frames almost always
  // share ONE handle, so only the first new handle pays the lightingOffsetFor()
  // lookup.  Set 0 re-binds only when the handle changes; set 1 every draw.
  uint32_t lightingDynamicOffset = ctx.lastLightingOffset;
  if (command.lightingHandle != ctx.lastLightingHandle) {
    lightingDynamicOffset =
      static_cast<uint32_t>(this->lightingOffsetFor(command));
    ctx.lastLightingHandle = command.lightingHandle;
    ctx.lastLightingOffset = lightingDynamicOffset;
  }
  uint32_t bindingOffsets[2] = { lightingDynamicOffset, uboDynamicOffset };
  if (ctx.lastBoundLightingOffset != lightingDynamicOffset ||
      ctx.lastBoundTextureSet != textureSet) {
    // First draw of a new handle/texture set: bind both sets with their offsets at once.
    const VkDescriptorSet both[2] = {this->lightingDescriptorSet, textureSet};
    vkBackendTrace(this->uboFrameIndex, "draw.bindDescSets2",
                   "slot=%u", slotIndex);
    vkCmdBindDescriptorSets(ctx.buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            this->pipelineLayout, 0, 2, both, 2,
                            bindingOffsets);
    ctx.lastBoundLightingOffset = lightingDynamicOffset;
    ctx.lastBoundTextureSet = textureSet;
  }
  else {
    // Same handle + texture set: only the per-draw UBO offset advances, so
    // re-bind set 1 alone (set 0 stays bound from the last 2-set bind).
    vkBackendTrace(this->uboFrameIndex, "draw.bindDescSets1",
                   "slot=%u", slotIndex);
    vkCmdBindDescriptorSets(ctx.buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            this->pipelineLayout, 1, 1, &textureSet, 1,
                            &bindingOffsets[1]);
  }
}

void
SoVulkanRenderBackend::recordDrawCommand(const SoDrawList & drawlist,
                                         const SoRenderCommand & command,
                                         const SoVulkanRenderTarget & target,
                                         const SoRenderParams & params,
                                         VkRenderPass pass,
                                         const bool transparent,
                                         const int fillModeOverride,
                                         const float * uniformColorOverride,
                                         const bool overlayPass,
                                         VulkanRecordContext & ctx)
{
  vkBackendTrace(this->uboFrameIndex, "recordDrawCommand.enter",
                 "cmd=%p pass=%d slot=%u",
                 reinterpret_cast<const void *>(&command),
                 static_cast<int>(command.pass), ctx.uboCmdIndex);
  if (!command.geometry.positions || command.geometry.vertexCount == 0) {
    if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
      SoVulkanDebug::post("[VKBE] cmd %p pass=%d skip: no positions/verts\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) {
    if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
      SoVulkanDebug::post("[VKBE] cmd %p pass=%d skip: no gpu cache entry\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }
  if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS)) {
    const VulkanCachedCommand & entryTmp = this->gpuCache[found->second];
    SoVulkanDebug::post("[VKBE] line/point cmd=%p pass=%d topo=%d verts=%u "
            "diffuse=(%.2f,%.2f,%.2f,%.2f) colorKey=%d shading=%d "
            "lineWidth=%.2f pattern=0x%04x fillMode=%d\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            command.geometry.vertexCount,
            command.material.diffuse[0], command.material.diffuse[1],
            command.material.diffuse[2], command.material.diffuse[3],
            entryTmp.colorKey != nullptr ? 1 : 0,
            static_cast<int>(command.material.shadingModel),
            command.state.raster.lineWidth,
            static_cast<unsigned>(command.state.raster.linePattern),
            static_cast<int>(command.state.raster.fillMode));
  }
  VulkanCachedCommand & entry = this->gpuCache[found->second];
  if (entry.vertexBuffer == VK_NULL_HANDLE) {
    if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
      SoVulkanDebug::post("[VKBE] cmd %p pass=%d skip: vertexBuffer null\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }

  // Wide-line rendering mirrors GL: line width > 1 or a stipple pattern expands
  // each segment into a quad; overlay wireframe/point redraws stay plain lines.
  const bool useWideLine = isWideLine(command, fillModeOverride, this->interactionLodActive);
  const bool patternedLine = isPatternedLine(command);
  // Line stipple mirrors GL glLineStipple: each bit covers linePatternScaleFactor
  // screen pixels; the fragment shader tests bit floor(distance/factor) % 16.
  float stippleFactor = 0.0f;
  float stipplePatternBits = 0.0f;
  if (useWideLine && patternedLine) {
    stippleFactor = static_cast<float>(std::max(
      1, static_cast<int>(command.state.raster.linePatternScale)));
    const uint32_t patternBits =
      static_cast<uint32_t>(command.state.raster.linePattern & 0xFFFFu);
    std::memcpy(&stipplePatternBits, &patternBits, sizeof(patternBits));
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->getOrCreatePipeline(command, target, pass, pipeline, transparent,
                                 fillModeOverride, overlayPass, &entry) ||
      pipeline == VK_NULL_HANDLE) {
    if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
      SoVulkanDebug::post("[VKBE] cmd %p pass=%d skip: pipeline creation failed "
                      "(transparent=%d fillOverride=%d overlay=%d)\n",
              (const void*)&command, static_cast<int>(command.pass),
              transparent ? 1 : 0, fillModeOverride, overlayPass ? 1 : 0);
    }
    return;
  }
  if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
    static int drawn = 0;
    static int logged = 0;
    drawn++;
    if (logged++ < 200) {
      SoVulkanDebug::post("[VKBE] draw %d cmd=%p pass=%d verts=%u idx=%u topo=%d "
              "overlay=%d transparent=%d\n",
              drawn, (const void*)&command, static_cast<int>(command.pass),
              command.geometry.vertexCount, command.geometry.indexCount,
              static_cast<int>(command.geometry.topology),
              overlayPass ? 1 : 0, transparent ? 1 : 0);
    }
  }
  this->applyPipeline(pipeline, ctx);
  // Commands with their own viewport (SoViewportRegionElement) keep it; others use the frame viewport.
  this->applyCommandViewport(command, target, ctx);
  this->applyScissor(command, target, ctx);

  const uint32_t slotIndex = ctx.uboCmdIndex++;
  // prepareLightingSlots() pre-reserved worst-case slots, so only a missing pre-count trips this.
  if (slotIndex >= this->uboSlotsPerFrame) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      this->emitError(
        "lighting UBO slot overflow: buffer too small for draw list");
    }
    return;
  }
  const VkDeviceSize uboOffset = this->uboSlotOffset(slotIndex);
  const uint32_t uboDynamicOffset = static_cast<uint32_t>(uboOffset);

  this->bindDrawDescriptors(command, uboDynamicOffset, slotIndex, ctx);

  const VkDeviceSize vertexOffset = entry.vertexOffset;
  const bool indexed =
    entry.indexBuffer != VK_NULL_HANDLE && command.geometry.indexCount &&
    command.geometry.indices;
  // GPU geometry LOD: a frame-compacted command draws the compacted index buffer via indirect draw.
  const VulkanCachedCommand::VulkanSubPixelSlot * subPixel =
    (!useWideLine) ? this->subPixelSlotFor(entry) : nullptr;
  vkBackendTrace(this->uboFrameIndex, "draw.bindVbuf0",
                 "slot=%u vbuf=%p off=%llu", slotIndex,
                 reinterpret_cast<const void *>(entry.vertexBuffer),
                 static_cast<unsigned long long>(vertexOffset));
  vkCmdBindVertexBuffers(ctx.buffer, 0, 1, &entry.vertexBuffer,
                         &vertexOffset);
  if (!useWideLine && subPixel != nullptr) {
    vkBackendTrace(this->uboFrameIndex, "draw.bindIbufLod", "slot=%u",
                   slotIndex);
    vkCmdBindIndexBuffer(ctx.buffer, subPixel->indexBuffer, 0,
                         VK_INDEX_TYPE_UINT32);
  }
  else if (indexed && !useWideLine) {
    vkBackendTrace(this->uboFrameIndex, "draw.bindIbuf", "slot=%u", slotIndex);
    vkCmdBindIndexBuffer(ctx.buffer, entry.indexBuffer,
                         entry.indexOffset, VK_INDEX_TYPE_UINT32);
  }

  SbMat projValue;
  // Overlay with its own camera/viewport (NaviCube) uses its own projection;
  // overlay spanning the frame (selection highlight) shares the frame projection
  // in params, since the scene camera's near/far lag a frame during navigation.
  const bool frameCameraOverlay = isFrameCameraOverlay(command, params);
  if (overlayPass && !frameCameraOverlay) {
    command.projMatrix.getValue(projValue);
  }
  else {
    // Frame projection, converted once per render (cacheFrameMatrices).
    std::memcpy(projValue, this->frameProjFloats, sizeof(float) * 16);
  }
  // Projection lives in the DrawBlock UBO now, so write the UBO after projValue is resolved.
  this->updateLightingUniforms(drawlist, command, params, uboOffset,
                               uniformColorOverride != nullptr,
                               &projValue[0][0]);
  vkBackendTrace(this->uboFrameIndex, "draw.uboWrite", "slot=%u", slotIndex);
  // The GPU wide-line shader sizes quads from its rasterization viewport.  A
  // sub-viewport overlay (NaviCube) rasterizes into its own rect, so size from
  // the command's viewport there; the frame size would shrink quads to sub-pixel.
  SbVec2s lineViewportSize = params.viewport.getViewportSizePixels();
  const SoRasterState & lineRaster = command.state.raster;
  if (lineRaster.viewportEnabled && lineRaster.viewportWidth > 0
      && lineRaster.viewportHeight > 0) {
    lineViewportSize = SbVec2s(
      static_cast<short>(lineRaster.viewportWidth),
      static_cast<short>(lineRaster.viewportHeight));
  }
  const VulkanPushConstants push = packPushConstants(
    command, entry, uniformColorOverride, this->frameDpr,
    stippleFactor, stipplePatternBits, useWideLine,
    std::max(1.0f, command.state.raster.lineWidth) * this->frameDpr,
    static_cast<float>(lineViewportSize[0] > 0 ? lineViewportSize[0] : 1),
    static_cast<float>(lineViewportSize[1] > 0 ? lineViewportSize[1] : 1));

  vkBackendTrace(this->uboFrameIndex, "draw.pushConstants", "slot=%u",
                 slotIndex);
  vkCmdPushConstants(ctx.buffer, this->pipelineLayout,
                      VK_SHADER_STAGE_VERTEX_BIT |
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(push), &push);

  if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS ||
       s_debugPushCount++ < 40)) {
    uint32_t patternRaw = 0;
    std::memcpy(&patternRaw, &push.lineParams[1], sizeof(patternRaw));
    SoVulkanDebug::post("[PUSH] cmd=%p pass=%d topo=%d srcDiffuse=(%.2f,%.2f,%.2f,%.2f) "
            "override=%d pushColor=(%.2f,%.2f,%.2f,%.2f) flags=(%.0f,%.0f,%.0f,%.0f) "
            "lineParams=(%.2f,%.2f,%.2f,%.2f) pointSize=%.2f wideLine=%d stippleFactor=%.1f pattern=0x%04x patternRaw=0x%08x "
            "fillMode=%d fillModeOverride=%d overlayPass=%d transparent=%d vbuf=%p vertexCount=%u\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            command.material.diffuse[0], command.material.diffuse[1],
            command.material.diffuse[2], command.material.diffuse[3],
            uniformColorOverride != nullptr ? 1 : 0,
            push.color[0], push.color[1], push.color[2], push.color[3],
            push.flags[0], push.flags[1], push.flags[2], push.flags[3],
            push.lineParams[0], push.lineParams[1], push.lineParams[2],
            push.lineParams[3], static_cast<double>(push.pointSize),
            useWideLine ? 1 : 0, stippleFactor,
            static_cast<unsigned>(command.state.raster.linePattern), patternRaw,
            static_cast<int>(command.state.raster.fillMode),
            fillModeOverride, overlayPass ? 1 : 0, transparent ? 1 : 0,
            (const void*)entry.vertexBuffer,
            static_cast<unsigned>(command.geometry.vertexCount));
  }

  // GPU-instanced wide lines: the vertex shader expands each segment from a
  // static instance-rate endpoint stream (no CPU work / quad upload).  Falls
  // back to CPU expansion when the endpoint buffer is unavailable.
  const bool useInstancedWideLine = useWideLine &&
    isInstancedWideLine(command) && entry.instancedLineBuffer != VK_NULL_HANDLE;

  if (useWideLine && !useInstancedWideLine) {
    // CPU-side clip-space quad expansion (width and/or stipple), drawn as a triangle list.
    if (!this->expandWideLines(entry, command, params, projValue,
                               std::max(1.0f, command.state.raster.lineWidth) *
                                 this->frameDpr)) {
      return;
    }
    VkDeviceSize wideOffset = 0;
    const VulkanCachedCommand::VulkanWideLineBuffer & wslot =
      entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
    vkCmdBindVertexBuffers(ctx.buffer, 0, 1, &wslot.buffer,
                           &wideOffset);
  }

  // Bind the per-instance model matrix (binding 1, INSTANCE rate) for the visual
  // and GPU-instanced wide-line paths; both read the transform from the
  // attribute, written at the SAME ring element index as the draw UBO so the GPU
  // reads this draw's transform (a shared offset would collapse all onto the last).
  if (useInstancedWideLine || !useWideLine) {
    const VkDeviceSize instElement =
      static_cast<VkDeviceSize>((this->uboFrameIndex % this->maxFramesInFlight) *
        this->uboSlotsPerFrame + slotIndex);
    const VkDeviceSize instByteOffset = instElement * sizeof(float) * 16;
  // Defensive check only: the ring is pre-sized (ensureInstanceModelRingCapacity);
  // growing mid-record would race parallel workers.
    if (instByteOffset + sizeof(float) * 16 > this->instanceModelCapacity) {
      return;
    }
    SbMat mm;
    command.modelMatrix.getValue(mm);
    std::memcpy(static_cast<char *>(this->instanceModelMapped) + instByteOffset,
                &mm[0][0], sizeof(float) * 16);
    vkBackendTrace(this->uboFrameIndex, "draw.bindVbuf1", "slot=%u",
                   slotIndex);
    vkCmdBindVertexBuffers(ctx.buffer, 1, 1,
                           &this->instanceModelBuffer, &instByteOffset);
  }

  if (useInstancedWideLine) {
    VkDeviceSize zeroOffset = 0;
    vkCmdBindVertexBuffers(ctx.buffer, 0, 1,
                           &entry.instancedLineBuffer, &zeroOffset);
    vkBackendTrace(this->uboFrameIndex, "draw.wideLineInstanced",
                   "slot=%u segs=%u cmd=%p", slotIndex,
                   entry.instancedLineSegmentCount,
                   reinterpret_cast<const void *>(&command));
    // Six vertices per segment (two triangles); the shader picks the corner from gl_VertexIndex.
    vkCmdDraw(ctx.buffer, 6u, entry.instancedLineSegmentCount, 0, 0);
  }
  else if (useWideLine) {
    static int wldrawDiag = 0;
    if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG") && wldrawDiag++ < 40) {
      SoVulkanDebug::post("[WLINE2] DRAW cmd=%p wideLineVertexCount=%u pass=%d\n",
              (const void*)&command, entry.wideLineVertexCount,
              static_cast<int>(command.pass));
    }
    vkCmdDraw(ctx.buffer, entry.wideLineVertexCount, 1, 0, 0);
  }
  else {
    if (subPixel != nullptr) {
      vkBackendTrace(this->uboFrameIndex, "draw.drawIndexedIndirect",
                     "slot=%u buf=%p cmd=%p", slotIndex,
                     reinterpret_cast<const void *>(ctx.buffer),
                     reinterpret_cast<const void *>(&command));
      vkCmdDrawIndexedIndirect(ctx.buffer, subPixel->indirectBuffer, 0, 1,
                               sizeof(VkDrawIndexedIndirectCommand));
    }
    else if (indexed) {
      vkBackendTrace(this->uboFrameIndex, "draw.drawIndexed",
                     "slot=%u buf=%p ic=%u cmd=%p", slotIndex,
                     reinterpret_cast<const void *>(ctx.buffer),
                     command.geometry.indexCount,
                     reinterpret_cast<const void *>(&command));
      vkCmdDrawIndexed(ctx.buffer, command.geometry.indexCount, 1, 0,
                       0, 0);
    }
    else {
      vkBackendTrace(this->uboFrameIndex, "draw.draw",
                     "slot=%u buf=%p vc=%u ic=1 cmd=%p",
                     slotIndex, reinterpret_cast<const void *>(ctx.buffer),
                     command.geometry.vertexCount,
                     reinterpret_cast<const void *>(&command));
      vkCmdDraw(ctx.buffer, command.geometry.vertexCount, 1, 0, 0);
    }
  }
}

bool
SoVulkanRenderBackend::recordCommandBatch(const SoDrawList & drawlist,
                                          const SoRenderCommand * const * commands,
                                          int count,
                                          const SoVulkanRenderTarget & target,
                                          const SoRenderParams & params,
                                          VkRenderPass pass,
                                          const bool transparent,
                                          const int fillModeOverride,
                                          const float * uniformColorOverride,
                                          VulkanRecordContext & ctx)
{
  // The batch differs only by model matrix, so all commands share commands[0]'s state.
  vkBackendTrace(this->uboFrameIndex, "recordCommandBatch.enter",
                 "count=%d slot=%u", count, ctx.uboCmdIndex);
  const SoRenderCommand & command = *commands[0];
  if (count < 2 || !command.geometry.positions ||
      command.geometry.vertexCount == 0) {
    return false;
  }
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) return false;
  // Fragile: only guaranteed-correct side paths (pipeline + descriptor + push +
  // non-instanced geometry) batch; wide-line expands per command on the CPU.
  const VulkanCachedCommand & entryRef = this->gpuCache[found->second];
  if (entryRef.vertexBuffer == VK_NULL_HANDLE) return false;

  if (isWideLine(command, fillModeOverride, this->interactionLodActive)) {
    // Not batchable; caller falls back to per-command draws.
    return false;
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->getOrCreatePipeline(command, target, pass, pipeline, transparent,
                                 fillModeOverride, false) ||
      pipeline == VK_NULL_HANDLE) {
    return false;
  }
  this->applyPipeline(pipeline, ctx);
  this->applyCommandViewport(command, target, ctx);
  this->applyScissor(command, target, ctx);

  // Reserve `count` slots (base..base+N-1) for the batch's instance-model elements.
  const uint32_t slotIndex = ctx.uboCmdIndex;
  ctx.uboCmdIndex += count;
  const VkDeviceSize uboOffset = this->uboSlotOffset(slotIndex);
  const uint32_t uboDynamicOffset = static_cast<uint32_t>(uboOffset);

  // Set 0 (lighting) + set 1 (per-draw UBO + texture) are group-constant, so bind once.
  this->bindDrawDescriptors(command, uboDynamicOffset, slotIndex, ctx);

  // Binding 0 shares commands[0]'s geometry, so one bind serves every instance.
  const VkDeviceSize vertexOffset = entryRef.vertexOffset;
  vkCmdBindVertexBuffers(ctx.buffer, 0, 1, &entryRef.vertexBuffer,
                         &vertexOffset);
  const bool indexed =
    entryRef.indexBuffer != VK_NULL_HANDLE && command.geometry.indexCount &&
    command.geometry.indices;
  if (indexed) {
    vkCmdBindIndexBuffer(ctx.buffer, entryRef.indexBuffer,
                         entryRef.indexOffset, VK_INDEX_TYPE_UINT32);
  }

  this->updateLightingUniforms(drawlist, command, params, uboOffset,
                               uniformColorOverride != nullptr,
                               this->frameProjFloats);

  // Push constants are group-constant too; batches come only from main passes,
  // so wide-line/stipple fields are unused (mirrors recordDrawCommand's main-pass branch).
  const VulkanPushConstants push = packPushConstants(
    command, entryRef, uniformColorOverride,
    this->frameDpr, /*stippleFactor*/ 0.0f, /*stipplePatternBits*/ 0.0f,
    /*wideLine*/ false);

  vkCmdPushConstants(ctx.buffer, this->pipelineLayout,
                      VK_SHADER_STAGE_VERTEX_BIT |
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(push), &push);

  // Per-instance model matrices: commands[0..count) at consecutive ring slots.
  const VkDeviceSize instBase =
    static_cast<VkDeviceSize>((this->uboFrameIndex % this->maxFramesInFlight) *
      this->uboSlotsPerFrame + slotIndex);
  const VkDeviceSize instCountBytes = instBase * sizeof(float) * 16 +
    static_cast<VkDeviceSize>(count) * sizeof(float) * 16;
  // Defensive bounds check only; the ring is pre-sized and growing would race parallel workers.
  if (instCountBytes > this->instanceModelCapacity) {
    return false;
  }
  char * mapped = static_cast<char *>(this->instanceModelMapped);
  for (int k = 0; k < count; ++k) {
    SbMat mm;
    commands[k]->modelMatrix.getValue(mm);
    std::memcpy(mapped + (instBase + k) * sizeof(float) * 16,
                &mm[0][0], sizeof(float) * 16);
  }
  const VkDeviceSize instOffset = instBase * sizeof(float) * 16;
  vkCmdBindVertexBuffers(ctx.buffer, 1, 1,
                         &this->instanceModelBuffer, &instOffset);

  if (indexed) {
    vkCmdDrawIndexed(ctx.buffer, command.geometry.indexCount,
                     static_cast<uint32_t>(count), 0, 0, 0);
  }
  else {
    vkCmdDraw(ctx.buffer, command.geometry.vertexCount,
              static_cast<uint32_t>(count), 0, 0);
  }
  return true;
}

bool
SoVulkanRenderBackend::beginCommandBuffer()
{
  vkBackendTrace(this->uboFrameIndex, "beginCommandBuffer.enter",
                 "primary=%p", reinterpret_cast<const void *>(
                   this->currentCommandBuffer()));
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return vkBeginCommandBuffer(this->currentCommandBuffer(), &bi) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::endAndSubmit()
{
  vkBackendTrace(this->uboFrameIndex, "endAndSubmit.enter", "frame=%u",
                 this->uboFrameIndex);
  VkCommandBuffer cmd = this->currentCommandBuffer();
  const VkResult endRc = vkEndCommandBuffer(cmd);
  vkBackendTrace(this->uboFrameIndex, "endAndSubmit.endRc", "rc=%d",
                 static_cast<int>(endRc));
  if (endRc != VK_SUCCESS) return false;

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  const VkFence fence = this->frameFences[slot];
  const VkResult submitRc = vkQueueSubmit(this->queue, 1, &si, fence);
  vkBackendTrace(this->uboFrameIndex, "endAndSubmit.submitRc", "rc=%d",
                 static_cast<int>(submitRc));
  if (submitRc != VK_SUCCESS) {
    // Fence stays unsignaled (no signal requested), so beginFrame() would wait
    // forever on this slot.  Mark it not pending so the wait is skipped; a
    // submission failure (typically device loss) leaves the backend unusable.
    this->frameFencePending[slot] = 0;
    return false;
  }
  this->frameFencePending[slot] = 1;
  return true;
}
