// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendCommand.cpp
//
// Per-draw command recording.  Provides:
//
//   - applyViewport()/applyCommandViewport()/applyScissor(): dynamic state
//     (with the Coin bottom-left -> Vulkan top-left Y-flip)
//   - recordClear()/recordOverlayDepthClear(): emit clears
//   - updateLightingUniforms(): fill a lighting/material UBO slot
//   - recordDrawCommand(): bind the pipeline, descriptor set and viewports,
//     pack the push constants, and issue the draw (plain, indexed or
//     wide-line)
//   - beginCommandBuffer()/endAndSubmit()

#include "rendering/SoVulkanRenderBackend.h"
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

void
SoVulkanRenderBackend::applyViewport(const SoRenderParams & params,
                                     const SoVulkanRenderTarget & target)
{
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_MATRIX_DUMP") && s_debugFrame > 0
      && (s_debugFrame % 100 == 0)) {
    fprintf(stderr,
            "[VPRT] frame=%d origin=(%d,%d) size=(%d,%d) target=(%u,%u)\n",
            s_debugFrame, origin[0], origin[1], size[0], size[1],
            target.extent.width, target.extent.height);
  }

  // Coin/OpenGL viewport origins are bottom-left; Vulkan's are top-left.
  // The vertex shader flips Y in clip space, so the viewport rectangle must
  // be re-anchored to the top edge for the two to cancel out (and for
  // non-fullscreen viewports to land in the correct sub-region).
  VkViewport viewport {};
  viewport.x = static_cast<float>(origin[0]);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(origin[1]) -
                                 static_cast<int32_t>(size[1]));
  viewport.width = static_cast<float>(size[0]);
  viewport.height = static_cast<float>(size[1]);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(this->activeCommandBuffer, 0, 1, &viewport);

  // Clamp the clear region to the target so an off-screen viewport (origin
  // outside the target, or a size exceeding the extent) never generates a
  // clear outside the render area.
  const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(origin[1]) -
         static_cast<int32_t>(size[1]));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(origin[0]) +
                                static_cast<int32_t>(size[0]));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(origin[1]));
  VkRect2D scissor {};
  scissor.offset = {x0, y0};
  scissor.extent = {static_cast<uint32_t>(std::max(0, x1 - x0)),
                    static_cast<uint32_t>(std::max(0, y1 - y0))};
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

// Apply a per-command viewport (recorded by the IR producer from
// SoViewportRegionElement).  Draws that carry their own viewport render
// into that sub-region; commands without one keep the frame viewport set
// by applyViewport().  Same Y-flip math as applyViewport().
void
SoVulkanRenderBackend::applyCommandViewport(const SoRenderCommand & command,
                                            const SoVulkanRenderTarget & target)
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
  // Depth range from the retained SoDepthBufferElement state; GL applies
  // glDepthRange() per command and restores (0,1) after each draw.  The
  // viewport is dynamic state here, so each command gets its own range and
  // nothing needs restoring.  Clamp to the legal [0,1] window.
  viewport.minDepth =
    std::clamp(command.state.depth.range[0], 0.0f, 1.0f);
  viewport.maxDepth =
    std::clamp(command.state.depth.range[1], 0.0f, 1.0f);
  vkCmdSetViewport(this->activeCommandBuffer, 0, 1, &viewport);

  // The per-command viewport also bounds the draw region; mirror the
  // scissor clamp used by applyViewport().
  const int32_t x0 = std::max(0, static_cast<int32_t>(raster.viewportX));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(raster.viewportY) -
         static_cast<int32_t>(raster.viewportHeight));
  const int32_t x1 =
    std::min(static_cast<int32_t>(target.extent.width),
             static_cast<int32_t>(raster.viewportX) +
               static_cast<int32_t>(raster.viewportWidth));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(raster.viewportY));
  VkRect2D scissor {};
  scissor.offset = {x0, y0};
  scissor.extent = {static_cast<uint32_t>(std::max(0, x1 - x0)),
                    static_cast<uint32_t>(std::max(0, y1 - y0))};
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

void
SoVulkanRenderBackend::applyScissor(const SoRenderCommand & command,
                                    const SoVulkanRenderTarget & target)
{
  VkRect2D scissor {};
  const SoRasterState & raster = command.state.raster;
  if (raster.scissorEnabled && raster.scissorWidth > 0 &&
      raster.scissorHeight > 0) {
    // Coin/OpenGL scissors are anchored at the bottom-left; Vulkan's are
    // top-left.  Mirror the viewport math: flip the Y offset around the
    // target height so the region lands where the producer intends.
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
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

void
SoVulkanRenderBackend::recordClear(const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;

  VkClearAttachment attachments[3];
  uint32_t attachmentCount = 0;

  if (params.flags & SO_PARAM_CLEAR_WINDOW) {
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

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_DEPTH)) {
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.depthStencil.depth = params.clearDepth;
    clear.clearValue.depthStencil.stencil = 0;
    attachments[attachmentCount++] = clear;
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

  // Clear only the requested viewport region (Y-flipped into Vulkan
  // coordinates like applyViewport()).  Clearing the whole target would
  // overwrite other viewports or the backing image outside the viewport.
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(origin[1]) -
         static_cast<int32_t>(size[1]));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(origin[0]) +
                                static_cast<int32_t>(size[0]));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(origin[1]));
  if (x1 <= x0 || y1 <= y0) return;

  VkClearRect rect {};
  rect.rect.offset = {x0, y0};
  rect.rect.extent = {static_cast<uint32_t>(x1 - x0),
                      static_cast<uint32_t>(y1 - y0)};
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(this->activeCommandBuffer, attachmentCount, attachments, 1,
                        &rect);
}

void
SoVulkanRenderBackend::recordOverlayDepthClear(const SoRenderCommand & command,
                                               const SoVulkanRenderTarget & target)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  if (!hasDepth) {
    return;
  }

  // The overlay rect is stored in Coin/OpenGL (bottom-left) coordinates by
  // the producer; mirror the Y-flip applied by applyScissor().
  const SoRasterState & raster = command.state.raster;
  const int32_t x0 = std::max(0, static_cast<int32_t>(raster.scissorX));
  const int32_t y0 = std::max(
    0, static_cast<int32_t>(target.extent.height) -
         static_cast<int32_t>(raster.scissorY) -
         static_cast<int32_t>(raster.scissorHeight));
  const int32_t x1 = std::min(static_cast<int32_t>(target.extent.width),
                              static_cast<int32_t>(raster.scissorX) +
                                static_cast<int32_t>(raster.scissorWidth));
  const int32_t y1 = std::min(
    static_cast<int32_t>(target.extent.height),
    static_cast<int32_t>(target.extent.height) -
      static_cast<int32_t>(raster.scissorY));
  if (x1 <= x0 || y1 <= y0) {
    return;
  }

  VkClearAttachment attachment {};
  attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  attachment.colorAttachment = 0;
  attachment.clearValue.depthStencil.depth = 1.0f;
  attachment.clearValue.depthStencil.stencil = 0;

  VkClearRect rect {};
  rect.rect.offset = {x0, y0};
  rect.rect.extent = {static_cast<uint32_t>(x1 - x0),
                      static_cast<uint32_t>(y1 - y0)};
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(this->activeCommandBuffer, 1, &attachment, 1, &rect);
}

void
SoVulkanRenderBackend::updateLightingUniforms(const SoDrawList & drawlist,
                                              const SoRenderCommand & command,
                                              const SoRenderParams & params,
                                              const VkDeviceSize uboOffset,
                                              const bool unlit)
{
  VulkanVisualUbo ubo {};

  SbMat m;
  // Overlay-pass geometry that spans the whole frame viewport (the
  // selection/preselection highlight) is frame-camera geometry: it must be
  // projected with the frame camera matrices, not with the scene camera's
  // own recorded matrices (whose near/far fields are stale and which lag
  // one frame behind during navigation, making the highlight rotate at a
  // different speed and z-fight the coplanar base).  Overlays that carry
  // their own viewport (the navigation cube sub-scene) keep their own
  // camera.
  const SbVec2s frameSize = params.viewport.getViewportSizePixels();
  const bool frameCameraOverlay =
    command.pass == SO_RENDERPASS_OVERLAY
    && command.state.raster.viewportWidth == frameSize[0]
    && command.state.raster.viewportHeight == frameSize[1];
  if (command.state.raster.scissorEnabled
      && command.pass == SO_RENDERPASS_OVERLAY && !frameCameraOverlay) {
    command.viewMatrix.getValue(m);
  }
  else {
    params.viewMatrix.getValue(m);
  }
  std::memcpy(ubo.view, &m[0][0], sizeof(float) * 16);
  command.modelMatrix.getValue(m);
  std::memcpy(ubo.model, &m[0][0], sizeof(float) * 16);
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_CLIP_DEBUG")) {
    static int uboLog = 0;
    if (uboLog++ < 6) {
      fprintf(stderr, "[UBO] cmd pass=%d verts=%u model00=%.3f m11=%.3f m22=%.3f "
                      "trans=(%.3f,%.3f,%.3f) view33=%.3f\n",
              static_cast<int>(command.pass),
              static_cast<unsigned>(command.geometry.vertexCount),
              ubo.model[0], ubo.model[5], ubo.model[10],
              ubo.model[12], ubo.model[13], ubo.model[14],
              ubo.view[15]);
    }
  }

  const SoMaterialData & material = command.material;
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS)) {
    fprintf(stderr,
            "[UBO] cmd=%p pass=%d topo=%d diffuse=(%.2f,%.2f,%.2f,%.2f) "
            "emissive=(%.2f,%.2f,%.2f) ambient=(%.2f,%.2f,%.2f) "
            "specular=(%.2f,%.2f,%.2f) shading=%d unlit=%d\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            material.diffuse[0], material.diffuse[1], material.diffuse[2],
            material.diffuse[3],
            material.emissive[0], material.emissive[1], material.emissive[2],
            material.ambient[0], material.ambient[1], material.ambient[2],
            material.specular[0], material.specular[1], material.specular[2],
            static_cast<int>(material.shadingModel), unlit ? 1 : 0);
  }
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

  const SoLightingData * lighting = drawlist.getLighting(command.lightingHandle);
  static const SoLightingData emptyLighting;
  if (!lighting) {
    lighting = &emptyLighting;
    if (command.lightingHandle != 0) {
      static std::once_flag invalidHandleWarning;
      std::call_once(invalidHandleWarning, []() {
        SoDebugError::postWarning(
          "SoVulkanRenderBackend::updateLightingUniforms",
          "Draw command references missing lighting data; no headlight is "
          "synthesized.");
      });
    }
  }

  ubo.ambientLight[0] = lighting->ambient[0];
  ubo.ambientLight[1] = lighting->ambient[1];
  ubo.ambientLight[2] = lighting->ambient[2];
  ubo.ambientLight[3] = 1.0f;

  const int count = std::min<int>(
    static_cast<int>(lighting->lights.size()), MAX_SHADER_LIGHTS);
  if (static_cast<int>(lighting->lights.size()) > MAX_SHADER_LIGHTS) {
    static std::once_flag lightLimitWarning;
    std::call_once(lightLimitWarning, []() {
      SoDebugError::postWarning(
        "SoVulkanRenderBackend::updateLightingUniforms",
        "The Visual program supports eight lights; additional retained "
        "lights are ignored by this executor.");
    });
  }
  ubo.materialParams[2] = static_cast<float>(count);

  for (int i = 0; i < count; ++i) {
    const SoLightData & light = lighting->lights[static_cast<size_t>(i)];
    float * type = ubo.lightType + i * 4;
    type[0] = static_cast<float>(light.type);
    type[1] = type[2] = 0.0f;
    type[3] = 1.0f;

    float * color = ubo.lightColor + i * 4;
    color[0] = light.color[0];
    color[1] = light.color[1];
    color[2] = light.color[2];
    color[3] = 1.0f;

    // SoLightData::direction and ::position are already expressed in the VIEW
    // (eye) space of the scene camera: fillLightingFromState() derives them via
    // SoLightElement::getMatrix(), which stores the light's model*view matrix
    // (see SoDirectionalLight::GLRender), so a directional light's "toward the
    // light" vector and a point/spot light's position arrive in eye space.  The
    // vertex stage outputs the normal and eye position in the SAME eye space
    // (this u_view), so they must be used verbatim.  Re-applying the view matrix
    // here would rotate the light a second time and make a static world light
    // chase the camera as it orbits.
    float * direction = ubo.lightDirection + i * 4;
    direction[0] = light.direction[0];
    direction[1] = light.direction[1];
    direction[2] = light.direction[2];
    direction[3] = 1.0f;

    float * position = ubo.lightPosition + i * 4;
    position[0] = light.position[0];
    position[1] = light.position[1];
    position[2] = light.position[2];
    position[3] = 1.0f;

    float * attenuation = ubo.lightAttenuation + i * 4;
    attenuation[0] = light.attenuation[0];
    attenuation[1] = light.attenuation[1];
    attenuation[2] = light.attenuation[2];
    attenuation[3] = 1.0f;

    float * spot = ubo.lightSpotParams + i * 4;
    spot[0] = light.spotCutoffCos;
    spot[1] = light.spotExponent;
    spot[2] = 0.0f;
    spot[3] = 1.0f;
  }

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_LIGHT_DEBUG") && count > 0 &&
      s_lightLog++ < 6) {
    fprintf(stderr,
            "[LIT] cmd=%p pass=%d topo=%d verts=%u lightCount=%d "
            "ambient=(%.3f,%.3f,%.3f) matAmb=(%.3f,%.3f,%.3f) "
            "shininess=%.2f\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            static_cast<unsigned>(command.geometry.vertexCount), count,
            lighting->ambient[0], lighting->ambient[1], lighting->ambient[2],
            material.ambient[0], material.ambient[1], material.ambient[2],
            material.shininess);
    for (int li = 0; li < count && li < 3; ++li) {
      const SoLightData & l = lighting->lights[static_cast<size_t>(li)];
      fprintf(stderr,
              "[LIT]   light%d type=%d color=(%.2f,%.2f,%.2f) "
              "dir=(%.3f,%.3f,%.3f) pos=(%.2f,%.2f,%.2f)\n",
              li, static_cast<int>(l.type), l.color[0], l.color[1],
              l.color[2], l.direction[0], l.direction[1], l.direction[2],
              l.position[0], l.position[1], l.position[2]);
    }
  }

  if (this->lightingMapped && this->uboSlotStride > 0) {
    std::memcpy(static_cast<char *>(this->lightingMapped) + uboOffset,
                &ubo, sizeof(ubo));
  }

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_MATRIX_DUMP") && s_debugFrame > 0
      && (s_debugFrame % 100 == 0) && s_dumpCmdCount <= 4) {
    fprintf(stderr,
            "[LGT] frame=%d cmd#%d ambient=(%.2f,%.2f,%.2f) lights=%d\n",
            s_debugFrame, s_dumpCmdCount - 1, lighting->ambient[0],
            lighting->ambient[1], lighting->ambient[2], count);
    for (int i = 0; i < count && i < 3; ++i) {
      const SoLightData & light = lighting->lights[static_cast<size_t>(i)];
      fprintf(stderr,
              "[LGT]   light%d type=%.0f dir=(%.3f,%.3f,%.3f) "
              "pos=(%.3f,%.3f,%.3f) color=(%.2f,%.2f,%.2f)\n",
              i, static_cast<float>(light.type),
              light.direction[0], light.direction[1], light.direction[2],
              light.position[0], light.position[1], light.position[2],
              light.color[0], light.color[1], light.color[2]);
    }
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
                                         const bool overlayPass)
{
  if (!command.geometry.positions || command.geometry.vertexCount == 0) {
    if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: no positions/verts\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) {
    if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: no gpu cache entry\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS)) {
    const VulkanCachedCommand & entryTmp = this->gpuCache[found->second];
    fprintf(stderr,
            "[VKBE] line/point cmd=%p pass=%d topo=%d verts=%u "
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
    if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: vertexBuffer null\n",
              (const void*)&command, static_cast<int>(command.pass));
    }
    return;
  }

  // Wide-line rendering mirrors the GL wide-line path: line width > 1 or a
  // stipple pattern expands each segment into a quad.  The overlay
  // wireframe/point redraws keep the plain line path.
  const bool lineTopology = command.geometry.topology == SO_TOPOLOGY_LINES ||
    command.geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  const bool patternedLine =
    command.state.raster.linePattern != 0xFFFF &&
    command.state.raster.linePattern != 0;
  const bool useWideLine =
    lineTopology && fillModeOverride < 0 &&
    (command.state.raster.lineWidth > 1.0f || patternedLine);
  // Line stipple mirrors classic GL (glLineStipple): each pattern bit
  // covers linePatternScaleFactor PIXELS in screen space.  The fragment
  // shader tests the bit selected by floor(distance / factor) % 16.
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
                                 fillModeOverride, overlayPass) ||
      pipeline == VK_NULL_HANDLE) {
    if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
      fprintf(stderr, "[VKBE] cmd %p pass=%d skip: pipeline creation failed "
                      "(transparent=%d fillOverride=%d overlay=%d)\n",
              (const void*)&command, static_cast<int>(command.pass),
              transparent ? 1 : 0, fillModeOverride, overlayPass ? 1 : 0);
    }
    return;
  }
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
    static int drawn = 0;
    static int logged = 0;
    drawn++;
    if (logged++ < 200) {
      fprintf(stderr,
              "[VKBE] draw %d cmd=%p pass=%d verts=%u idx=%u topo=%d "
              "overlay=%d transparent=%d\n",
              drawn, (const void*)&command, static_cast<int>(command.pass),
              command.geometry.vertexCount, command.geometry.indexCount,
              static_cast<int>(command.geometry.topology),
              overlayPass ? 1 : 0, transparent ? 1 : 0);
    }
  }
  vkCmdBindPipeline(this->activeCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline);
  // Commands carrying their own viewport (SoViewportRegionElement) render
  // into that sub-region; otherwise the frame viewport from applyViewport()
  // stays active.
  this->applyCommandViewport(command, target);
  this->applyScissor(command, target);

  const uint32_t slotIndex = this->uboCmdIndex++;
  // prepareLightingSlots() reserves a worst-case slot count before any
  // recording, so this can only trip if a future recording path forgets to
  // pre-count.  Guard at runtime regardless: the mapped UBO write below
  // would otherwise run past the allocation.
  if (slotIndex >= this->uboSlotsPerFrame) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      this->emitError(
        "lighting UBO slot overflow: buffer too small for draw list");
    }
    return;
  }
  const VkDeviceSize uboOffset =
    ((this->uboFrameIndex % this->maxFramesInFlight) *
       this->uboSlotsPerFrame + slotIndex) *
    this->uboSlotStride;
  const uint32_t uboDynamicOffset = static_cast<uint32_t>(uboOffset);

  VkDescriptorSet textureSet = this->resolveTextureSet(command);
  if (textureSet != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(this->activeCommandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            this->pipelineLayout, 0, 1, &textureSet, 1,
                            &uboDynamicOffset);
  }

  const VkDeviceSize vertexOffset = entry.vertexOffset;
  vkCmdBindVertexBuffers(this->activeCommandBuffer, 0, 1, &entry.vertexBuffer,
                         &vertexOffset);
  const bool indexed =
    entry.indexBuffer != VK_NULL_HANDLE && command.geometry.indexCount &&
    command.geometry.indices;
  if (indexed && !useWideLine) {
    vkCmdBindIndexBuffer(this->activeCommandBuffer, entry.indexBuffer,
                         entry.indexOffset, VK_INDEX_TYPE_UINT32);
  }

  this->updateLightingUniforms(drawlist, command, params, uboOffset,
                               uniformColorOverride != nullptr);

  VulkanPushConstants push {};
  SbMat projValue;
  // Overlay-pass geometry that carries its own camera and viewport (the
  // navigation cube sub-scene) uses its own projection; overlay geometry
  // that spans the whole frame viewport (the selection/preselection
  // highlight) is frame-camera geometry and must share the frame projection
  // in params, otherwise it is projected through the scene camera's stale
  // near/far fields and lags behind navigation (see updateLightingUniforms).
  const SbVec2s frameSize = params.viewport.getViewportSizePixels();
  const bool frameCameraOverlay =
    command.pass == SO_RENDERPASS_OVERLAY
    && command.state.raster.viewportWidth == frameSize[0]
    && command.state.raster.viewportHeight == frameSize[1];
  if (overlayPass && !frameCameraOverlay) {
    command.projMatrix.getValue(projValue);
  }
  else {
    params.projMatrix.getValue(projValue);
  }
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_OVERLAY_CAM_DEBUG")
      && command.pass == SO_RENDERPASS_OVERLAY
      && command.state.raster.scissorEnabled
      && command.state.raster.scissorWidth > 800) {
    SbMat cv, pv, cp, pp;
    command.viewMatrix.getValue(cv);
    params.viewMatrix.getValue(pv);
    command.projMatrix.getValue(cp);
    params.projMatrix.getValue(pp);
    static int ovcamLog = 0;
    if (ovcamLog++ < 6) {
      fprintf(stderr,
              "[OVCAM-FULL] pass=%d frameCam=%d scissor=%d,%d %dx%d viewport=%d,%d %dx%d\n"
              "  cmdView:\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "  parView:\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "  cmdProj:\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "  parProj:\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n"
              "    [%.4f %.4f %.4f %.4f]\n    [%.4f %.4f %.4f %.4f]\n",
              static_cast<int>(command.pass),
              frameCameraOverlay ? 1 : 0,
              command.state.raster.scissorX, command.state.raster.scissorY,
              command.state.raster.scissorWidth, command.state.raster.scissorHeight,
              command.state.raster.viewportX, command.state.raster.viewportY,
              command.state.raster.viewportWidth, command.state.raster.viewportHeight,
              cv[0][0], cv[0][1], cv[0][2], cv[0][3],
              cv[1][0], cv[1][1], cv[1][2], cv[1][3],
              cv[2][0], cv[2][1], cv[2][2], cv[2][3],
              cv[3][0], cv[3][1], cv[3][2], cv[3][3],
              pv[0][0], pv[0][1], pv[0][2], pv[0][3],
              pv[1][0], pv[1][1], pv[1][2], pv[1][3],
              pv[2][0], pv[2][1], pv[2][2], pv[2][3],
              pv[3][0], pv[3][1], pv[3][2], pv[3][3],
              cp[0][0], cp[0][1], cp[0][2], cp[0][3],
              cp[1][0], cp[1][1], cp[1][2], cp[1][3],
              cp[2][0], cp[2][1], cp[2][2], cp[2][3],
              cp[3][0], cp[3][1], cp[3][2], cp[3][3],
              pp[0][0], pp[0][1], pp[0][2], pp[0][3],
              pp[1][0], pp[1][1], pp[1][2], pp[1][3],
              pp[2][0], pp[2][1], pp[2][2], pp[2][3],
              pp[3][0], pp[3][1], pp[3][2], pp[3][3]);
    }
  }
  std::memcpy(push.proj, &projValue[0][0], sizeof(float) * 16);
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
  push.flags[3] = command.material.textureAlphaIncludesOpacity
                    ? 1.0f : 0.0f;
  push.texParams[0] =
    static_cast<float>(command.material.texture.model);
  push.texParams[1] =
    static_cast<float>(command.state.alphaTest.function);
  push.texParams[2] = command.state.alphaTest.reference;
  push.texParams[3] =
    (command.material.flags & SO_MAT_IS_PIXEL_TEXT) ? 1.0f : 0.0f;
  const SbVec4f & blendColor = command.material.texture.blendColor;
  push.texBlend[0] = blendColor[0];
  push.texBlend[1] = blendColor[1];
  push.texBlend[2] = blendColor[2];
  push.texBlend[3] = blendColor[3];
  // Point size from the retained state (SoDrawStyle::pointSize via
  // SoPointSizeElement); GL multiplies by the device pixel ratio because
  // its viewport is in device pixels -- Vulkan viewports are too, so the
  // same value applies directly.  Applies to point primitives and to
  // VK_POLYGON_MODE_POINT (the wireframe/points overlay).
  // The GL path scales width/size by the device-pixel ratio because the
  // viewport is in device pixels while SoDrawStyle width/point-size are
  // logical points (so a 2pt line is 2*dpr device px on a scaled display).
  // The Vulkan viewport is device pixels too, so apply the same ratio here
  // for parity; otherwise lines/points render 1/dpr too thin on a fractional
  // (e.g. 1.25 / 1.5 / 2.0) scaling display.
  const float dpr = params.devicePixelRatio > 0.0f
    ? params.devicePixelRatio : 1.0f;
  push.pointSize = std::max(1.0f, command.state.raster.pointSize) * dpr;
  push.lineParams[0] = push.lineParams[1] = push.lineParams[2] = 0.0f;
  push.lineParams[3] = 0.0f;
  push.lineParams[0] = stippleFactor;
  // Slot y serves two masters: the wide-line shader reads the stipple
  // pattern bits here, the visual shader reads the round-point flag.  A
  // draw never reaches both shaders, so the slot is safe to share.
  push.lineParams[1] = (useWideLine && patternedLine)
    ? stipplePatternBits
    : (command.state.raster.pointShape == SO_POINT_SHAPE_ROUND ? 1.0f : 0.0f);
  // Point primitives (e.g. Sketcher vertex "dots" via SoMarkerSet, whose
  // CIRCLE_FILLED marker is the intended glyph) render as round dots: the
  // IR/Vulkan path has no marker-bitmap rasterization, so a filled round
  // point (fragment-shader round discard) reproduces the dot appearance
  // instead of a featureless (often sub-pixel) square.  Nothing scene-side
  // sets SO_POINT_SHAPE_ROUND (SoRenderIR forces SQUARE in the blend state),
  // so default every point primitive to round.
  if (command.geometry.topology == SO_TOPOLOGY_POINTS) {
    push.lineParams[1] = 1.0f;
  }
  push.lineParams[2] = useWideLine ? 1.0f : 0.0f;
  push.lineParams[3] = command.geometry.topology == SO_TOPOLOGY_POINTS
    ? 1.0f : 0.0f;

  vkCmdPushConstants(this->activeCommandBuffer, this->pipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG") &&
      (command.geometry.topology == SO_TOPOLOGY_LINES ||
       command.geometry.topology == SO_TOPOLOGY_LINE_STRIP ||
       command.geometry.topology == SO_TOPOLOGY_POINTS ||
       s_debugPushCount++ < 40)) {
    uint32_t patternRaw = 0;
    std::memcpy(&patternRaw, &push.lineParams[1], sizeof(patternRaw));
    fprintf(stderr,
            "[PUSH] cmd=%p pass=%d topo=%d srcDiffuse=(%.2f,%.2f,%.2f,%.2f) "
            "override=%d pushColor=(%.2f,%.2f,%.2f,%.2f) flags=(%.0f,%.0f,%.0f,%.0f) "
            "lineParams=(%.2f,%.2f,%.2f,%.2f) pointSize=%.2f wideLine=%d stippleFactor=%.1f pattern=0x%04x patternRaw=0x%08x "
            "fillMode=%d fillModeOverride=%d overlayPass=%d transparent=%d vbuf=%p vertexCount=%u\n",
            (const void*)&command, static_cast<int>(command.pass),
            static_cast<int>(command.geometry.topology),
            color[0], color[1], color[2], color[3],
            useOverrideColor ? 1 : 0,
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

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_MATRIX_DUMP") && s_debugFrame > 0
      && (s_debugFrame % 100 == 0) && s_dumpCmdCount < 12) {
    s_dumpCmdCount++;
    SbMat mm;
    command.modelMatrix.getValue(mm);
    SbMat vm;
    if (command.state.raster.scissorEnabled
        && command.pass == SO_RENDERPASS_OVERLAY) {
      command.viewMatrix.getValue(vm);
    }
    else {
      params.viewMatrix.getValue(vm);
    }
    fprintf(stderr,
            "[MATX] frame=%d cmd#%d pass=%d verts=%u overlay=%d "
            "scissor=%d model=\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n",
            s_debugFrame, s_dumpCmdCount - 1, static_cast<int>(command.pass),
            command.geometry.vertexCount, overlayPass ? 1 : 0,
            command.state.raster.scissorEnabled ? 1 : 0,
            mm[0][0], mm[0][1], mm[0][2], mm[0][3],
            mm[1][0], mm[1][1], mm[1][2], mm[1][3],
            mm[2][0], mm[2][1], mm[2][2], mm[2][3],
            mm[3][0], mm[3][1], mm[3][2], mm[3][3]);
    fprintf(stderr,
            "[MATX]   view=\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "[MATX]   proj=\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n"
            "  [%.4f %.4f %.4f %.4f]\n",
            vm[0][0], vm[0][1], vm[0][2], vm[0][3],
            vm[1][0], vm[1][1], vm[1][2], vm[1][3],
            vm[2][0], vm[2][1], vm[2][2], vm[2][3],
            vm[3][0], vm[3][1], vm[3][2], vm[3][3],
            projValue[0][0], projValue[0][1], projValue[0][2], projValue[0][3],
            projValue[1][0], projValue[1][1], projValue[1][2], projValue[1][3],
            projValue[2][0], projValue[2][1], projValue[2][2], projValue[2][3],
            projValue[3][0], projValue[3][1], projValue[3][2], projValue[3][3]);
  }

  if (useWideLine) {
    // CPU-side quad expansion in clip space (line width and/or stipple);
    // the wide-line pipeline draws it as a triangle list.  Binds here so
    // the projection matrix (projValue) is already resolved.
    if (!this->expandWideLines(entry, command, params, projValue,
                               std::max(1.0f, command.state.raster.lineWidth) * dpr)) {
      return;
    }
    VkDeviceSize wideOffset = 0;
    const VulkanCachedCommand::VulkanWideLineBuffer & wslot =
      entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
    vkCmdBindVertexBuffers(this->activeCommandBuffer, 0, 1, &wslot.buffer,
                           &wideOffset);
  }

  if (useWideLine) {
    static int wldrawDiag = 0;
    if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG") && wldrawDiag++ < 40) {
      fprintf(stderr, "[WLINE2] DRAW cmd=%p wideLineVertexCount=%u pass=%d\n",
              (const void*)&command, entry.wideLineVertexCount,
              static_cast<int>(command.pass));
    }
    vkCmdDraw(this->activeCommandBuffer, entry.wideLineVertexCount, 1, 0, 0);
  }
  else if (indexed) {
    vkCmdDrawIndexed(this->activeCommandBuffer, command.geometry.indexCount, 1, 0,
                     0, 0);
  }
  else {
    vkCmdDraw(this->activeCommandBuffer, command.geometry.vertexCount, 1, 0, 0);
  }
}

bool
SoVulkanRenderBackend::beginCommandBuffer()
{
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return vkBeginCommandBuffer(this->currentCommandBuffer(), &bi) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::endAndSubmit()
{
  VkCommandBuffer cmd = this->currentCommandBuffer();
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  const VkFence fence = this->frameFences[slot];
  if (vkQueueSubmit(this->queue, 1, &si, fence) != VK_SUCCESS) {
    // The fence stays unsignaled (no signal was requested), so beginFrame()
    // would wait forever on the next reuse of this slot.  Signal it by
    // submitting nothing and relying on the next frame's failure path, but
    // mark the slot as not pending so the wait is skipped; a submission
    // failure (typically device loss) leaves the backend unusable anyway.
    this->frameFencePending[slot] = 0;
    return false;
  }
  this->frameFencePending[slot] = 1;
  return true;
}
