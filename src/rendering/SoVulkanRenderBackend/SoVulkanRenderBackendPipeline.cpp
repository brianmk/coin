// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendPipeline.cpp
//
// Graphics pipeline and render-pass management.  Provides:
//
//   - getOrCreatePipeline(): build + cache an immutable VkPipeline per unique
//     retained state (topology, fill/cull, depth, blend, stencil, sample
//     count, wide-line) and translate it into the Vulkan state structs
//   - Background-gradient pipeline + recordBackground()
//   - Render-pass create/cache path (createRenderPass, renderPassIdentity,
//     getOrCreateRenderPass)

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

bool
SoVulkanRenderBackend::createBackgroundPipeline(
  const SoVulkanRenderTarget & target,
  VkRenderPass renderPass,
  VkPipeline & pipeline)
{
  BackgroundPipelineKey key;
  key.renderPass = renderPass;
  key.sampleCount = target.sampleCount;
  const auto found = this->backgroundPipelineCache.find(key);
  if (found != this->backgroundPipelineCache.end()) {
    pipeline = found->second;
    return pipeline != VK_NULL_HANDLE;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = this->backgroundVertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = this->backgroundFragmentModule;
  stages[1].pName = "main";

  // Fullscreen triangle: no vertex inputs.
  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 0;
  vertexInput.vertexAttributeDescriptionCount = 0;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  rasterization.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization.cullMode = VK_CULL_MODE_NONE;
  rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType =
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = target.sampleCount;

  // The gradient fills the whole viewport and writes no depth so geometry
  // drawn afterwards is unaffected.
  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_FALSE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType =
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.logicOpEnable = VK_FALSE;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkGraphicsPipelineCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci.stageCount = 2;
  ci.pStages = stages;
  ci.pVertexInputState = &vertexInput;
  ci.pInputAssemblyState = &inputAssembly;
  ci.pViewportState = &viewportState;
  ci.pRasterizationState = &rasterization;
  ci.pMultisampleState = &multisample;
  ci.pDepthStencilState = &depthStencil;
  ci.pColorBlendState = &colorBlend;
  ci.pDynamicState = &dynamicState;
  ci.layout = this->backgroundPipelineLayout;
  ci.renderPass = renderPass;
  ci.subpass = 0;

  VkPipeline created = VK_NULL_HANDLE;
  const VkResult result =
    vkCreateGraphicsPipelines(this->device, this->pipelineCacheHandle, 1, &ci,
                              this->allocator, &created);
  if (result != VK_SUCCESS) {
    this->emitError("failed to create Vulkan background pipeline");
    this->backgroundPipelineCache[key] = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->backgroundPipelineCache[key] = created;
  pipeline = created;
  return true;
}

void
SoVulkanRenderBackend::recordBackground(const SoRenderParams & params,
                                        const SoVulkanRenderTarget & target,
                                        VkRenderPass renderPass)
{
  if (!params.backgroundGradient) {
    return;
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->createBackgroundPipeline(target, renderPass, pipeline) ||
      pipeline == VK_NULL_HANDLE) {
    return;
  }

  // The gradient covers exactly the viewport region (same Y-flip math as
  // applyViewport()); geometry drawn afterwards restores its own viewport.
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
  const int32_t w = std::max(0, x1 - x0);
  const int32_t h = std::max(0, y1 - y0);
  if (w == 0 || h == 0) return;

  VkViewport viewport {};
  viewport.x = static_cast<float>(x0);
  viewport.y = static_cast<float>(y0);
  viewport.width = static_cast<float>(w);
  viewport.height = static_cast<float>(h);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  this->applyViewportState(viewport);

  VkRect2D scissor {};
  scissor.offset = {x0, y0};
  scissor.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
  this->applyScissorState(scissor);

  this->applyPipeline(pipeline);

  VulkanBackgroundPush push {};
  push.topColor[0] = params.backgroundTopColor[0];
  push.topColor[1] = params.backgroundTopColor[1];
  push.topColor[2] = params.backgroundTopColor[2];
  push.topColor[3] = params.backgroundTopColor[3];
  push.bottomColor[0] = params.backgroundBottomColor[0];
  push.bottomColor[1] = params.backgroundBottomColor[1];
  push.bottomColor[2] = params.backgroundBottomColor[2];
  push.bottomColor[3] = params.backgroundBottomColor[3];
  push.viewport[0] = static_cast<float>(w);
  push.viewport[1] = static_cast<float>(h);
  push.viewport[2] = static_cast<float>(x0);
  push.viewport[3] = static_cast<float>(y0);
  vkCmdPushConstants(this->activeCommandBuffer, this->backgroundPipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);

  vkCmdDraw(this->activeCommandBuffer, 3, 1, 0, 0);
}

bool
SoVulkanRenderBackend::createRenderPass(const SoVulkanRenderTarget & target,
                                        VkRenderPass & pass)
{
  VkAttachmentDescription attachments[2];
  uint32_t attachmentCount = 1;

  attachments[0].flags = 0;
  attachments[0].format = target.colorFormat;
  attachments[0].samples = target.sampleCount;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = target.colorLayout;
  attachments[0].finalLayout = target.colorLayout;

  VkAttachmentReference colorRef {};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef {};
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  if (hasDepth) {
    attachments[1].flags = 0;
    attachments[1].format = target.depthFormat;
    attachments[1].samples = target.sampleCount;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = target.depthLayout;
    attachments[1].finalLayout = target.depthLayout;
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachmentCount = 2;
  }

  VkSubpassDescription subpass {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

  VkRenderPassCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = attachmentCount;
  ci.pAttachments = attachments;
  ci.subpassCount = 1;
  ci.pSubpasses = &subpass;
  ci.dependencyCount = 0;
  ci.pDependencies = nullptr;

  return vkCreateRenderPass(this->device, &ci, this->allocator, &pass) ==
         VK_SUCCESS;
}

SoVulkanRenderBackend::RenderPassIdentity
SoVulkanRenderBackend::renderPassIdentity(const SoVulkanRenderTarget & target) const
{
  RenderPassIdentity identity;
  identity.colorFormat = target.colorFormat;
  identity.sampleCount = target.sampleCount;
  identity.colorLayout = target.colorLayout;
  // createRenderPass() only adds a depth attachment when a depth view is
  // present, so a configured-but-viewless depth format must not be part of
  // the identity.
  identity.depthFormat =
    (target.depthImageView != VK_NULL_HANDLE) ? target.depthFormat
                                              : VK_FORMAT_UNDEFINED;
  identity.depthLayout = target.depthLayout;
  return identity;
}

VkRenderPass
SoVulkanRenderBackend::getOrCreateRenderPass(const SoVulkanRenderTarget & target)
{
  const RenderPassIdentity identity = this->renderPassIdentity(target);
  const auto found = this->renderPassCache.find(identity);
  if (found != this->renderPassCache.end()) return found->second;

  VkRenderPass pass = VK_NULL_HANDLE;
  if (!this->createRenderPass(target, pass)) return VK_NULL_HANDLE;
  this->renderPassCache.emplace(identity, pass);
  return pass;
}

bool
SoVulkanRenderBackend::getOrCreatePipeline(const SoRenderCommand & command,
                                           const SoVulkanRenderTarget & target,
                                           VkRenderPass pass,
                                           VkPipeline & pipeline,
                                           const bool transparent,
                                           const int fillModeOverride,
                                           const bool overlayPass)
{
  // Pipelines are immutable in Vulkan.  Key the cache on every retained
  // state value that changes the created pipeline so commands of different
  // topology, fill mode, depth/blend state, or sample count never reuse an
  // incompatible pipeline.  Shading model, vertex-color, texture, and
  // lighting remain uniform/push-constant concerns in this milestone and do
  // not need to participate in the key yet.
  const bool blending = transparent || command.state.blend.enabled ||
                        command.material.diffuse[3] < 0.999f;
  const bool overlay = fillModeOverride >= 0;
  // SoPolygonOffsetElement contributes an explicit depth bias captured into
  // the raster state.  Selection/overlay faces use it to pull themselves in
  // front of the coplanar base geometry (GL glPolygonOffset semantics).
  // Respect it in the key so selection overlays stop z-fighting with the
  // geometry underneath them.
  const bool polygonOffset =
    command.state.raster.polygonOffsetFactor != 0.0f ||
    command.state.raster.polygonOffsetUnits != 0.0f;
  const bool depthBias = overlay || polygonOffset;
  // GL polygon-offset units map ~1:1 onto Vulkan's depthBiasConstantFactor,
  // but the two differ in how `r` (the minimum resolvable depth step) is
  // derived: GL uses the (fixed-point, 24-bit) depth range while this backend
  // commonly owns a float (D32_SFLOAT) depth attachment, whose resolvable
  // step is far finer.  A GL-sized offset therefore leaves the coplanar
  // selection/hover overlay Z-fighting with the base (a dark seam along the
  // face boundary).  Scale the GL decal up so the overlay wins the depth test
  // decisively; the slope factor keeps it from detaching at grazing,
  // silhouette edges.
  constexpr float kDecalScale = 512.0f;
  const float kUseDecal = COIN_VULKAN_ENV_FLAG("FC_VULKAN_RASTER_DECAL")
    ? kDecalScale : 1.0f;
  const float depthBiasConstant = polygonOffset
    ? command.state.raster.polygonOffsetUnits * kUseDecal
    : (overlay ? -0.5f : 0.0f);
  const float depthBiasSlope = polygonOffset
    ? command.state.raster.polygonOffsetFactor * kUseDecal
    : (overlay ? -0.5f : 0.0f);
  PipelineKey key;
  // Wide-line rendering (line width > 1 and/or a stipple pattern) expands
  // segments into quads on the CPU and draws them with the wide-line
  // pipeline as triangle lists, mirroring the GL wide-line geometry shader.
  // The overlay wireframe redraw stays on the plain line path.
  const bool lineTopology = command.geometry.topology == SO_TOPOLOGY_LINES ||
    command.geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  const bool patternedLine =
    command.state.raster.linePattern != 0xFFFF &&
    command.state.raster.linePattern != 0;
  const bool useWideLine =
    lineTopology && fillModeOverride < 0 &&
    (command.state.raster.lineWidth > 1.0f || patternedLine);
  key.wideLine = useWideLine;
  key.renderPass = pass;
  key.topology = command.geometry.topology;
  key.fillMode = overlay ? static_cast<uint8_t>(fillModeOverride)
                          : command.state.raster.fillMode;
  key.cullMode = overlay ? 0 : command.state.raster.cullMode;
  key.ccwFrontFace = command.state.raster.ccwFrontFace;
  key.depthTestEnable = command.state.depth.enabled || overlay;
  // Overlay-pass geometry (e.g. the navigation cube) draws last into its own
  // viewport and keeps depth writes so it can self-occlude correctly; the
  // wireframe/point redraw overlays deliberately disable depth writes.
  key.depthWriteEnable = overlayPass
    ? command.state.depth.writeEnabled
    : (!transparent && !overlay && command.state.depth.writeEnabled);
  key.depthFunction = overlayPass ? static_cast<uint8_t>(command.state.depth.func)
                                  : (overlay ? static_cast<uint8_t>(SO_DEPTH_LEQUAL)
                                             : command.state.depth.func);
  key.depthBiasEnable = depthBias;
  key.depthBiasConstantFactor = depthBiasConstant;
  key.depthBiasSlopeFactor = depthBiasSlope;
  key.blendEnable = blending;
  key.sampleCount = target.sampleCount;
  if (blending) {
    key.blendSrcRGB = command.state.blend.srcRGBFactor;
    key.blendDstRGB = command.state.blend.dstRGBFactor;
    key.blendSrcAlpha = command.state.blend.srcAlphaFactor;
    key.blendDstAlpha = command.state.blend.dstAlphaFactor;
    key.blendEquationRGB = command.state.blend.rgbEquation;
    key.blendEquationAlpha = command.state.blend.alphaEquation;
  }
  const SoStencilState & stencil = command.state.stencil;
  key.stencilEnable = stencil.enabled;
  if (stencil.enabled) {
    key.stencilFunction = stencil.function;
    key.stencilReference = stencil.reference;
    key.stencilCompareMask = stencil.compareMask;
    key.stencilWriteMask = stencil.writeMask;
    key.stencilFailOp = stencil.failOp;
    key.stencilZFailOp = stencil.zfailOp;
    key.stencilZPassOp = stencil.zpassOp;
  }

  // Per-command fast path: an unchanged command (same retained state -> the
  // same PipelineKey) re-resolves to the same VkPipeline without paying the
  // unordered_map lookup (key hash + bucket walk + equality) every frame.
  // The key that produced the last resolved handle is stored verbatim on the
  // geometry-cache entry, and the (cheap field-by-field, hash-free) equality
  // below decides the hit.  The backing entry is destroyed together with the
  // pipeline cache in invalidateCache(), so the cached handle can never
  // dangle.
  VulkanCachedCommand * cacheEntry = nullptr;
  const auto cmdEntry = this->commandToCache.find(&command);
  if (cmdEntry != this->commandToCache.end()) {
    cacheEntry = &this->gpuCache[cmdEntry->second];
    if (cacheEntry->hasResolvedPipeline &&
        cacheEntry->resolvedKey == key) {
      pipeline = cacheEntry->resolvedPipeline;
      return pipeline != VK_NULL_HANDLE;
    }
  }

  const auto found = this->pipelineCache.find(key);
  if (found != this->pipelineCache.end()) {
    if (cacheEntry) {
      cacheEntry->resolvedKey = key;
      cacheEntry->resolvedPipeline = found->second;
      cacheEntry->hasResolvedPipeline = true;
    }
    pipeline = found->second;
    return pipeline != VK_NULL_HANDLE;
  }

  // VK_POLYGON_MODE_LINE and VK_POLYGON_MODE_POINT require the
  // fillModeNonSolid feature to be enabled at device creation.  The
  // embedding application enables it only when the hardware advertises it,
  // so creating such a pipeline without the feature is a spec violation
  // (VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507) and can
  // make vkCreateGraphicsPipelines fail or hang drivers.  Refuse the
  // pipeline instead; the failure is cached under the key so the warning is
  // emitted once and every later lookup of the same state cheaply returns
  // false.
  if (!this->fillModeNonSolid && !key.wideLine &&
      (key.fillMode == SoDrawStyleElement::LINES ||
       key.fillMode == SoDrawStyleElement::POINTS)) {
    this->emitError(
      "Vulkan backend: the device does not support the fillModeNonSolid "
      "feature; wireframe and point fill modes cannot be rendered");
    this->pipelineCache[key] = VK_NULL_HANDLE;
    if (cacheEntry) {
      cacheEntry->resolvedKey = key;
      cacheEntry->resolvedPipeline = VK_NULL_HANDLE;
      cacheEntry->hasResolvedPipeline = true;
    }
    pipeline = VK_NULL_HANDLE;
    return false;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = key.wideLine ? this->wideLineVertexModule
                                  : this->vertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = key.wideLine ? this->wideLineFragmentModule
                                  : this->fragmentModule;
  stages[1].pName = "main";

  // Binding 0: the interleaved position/normal/color/texcoord stream.  The
  // wide-line path substitutes its own 36-byte clip-space layout at binding 0.
  VkVertexInputBindingDescription binding[2] {};
  binding[0].binding = 0;
  binding[0].stride = key.wideLine ? 36u : VULKAN_VERTEX_STRIDE;
  binding[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  // Binding 1 (visual pipelines only): the per-instance model matrix, four
  // R32G32B32A32 rows advanced per instance (rate INSTANCE).  This lets a
  // group of commands sharing geometry/material but differing only by model
  // matrix be drawn as one instanced vkCmdDraw.  A one-element instance buffer
  // is bound for ordinary non-instanced draws, so every visual draw carries
  // the attribute.  Wide-line pipelines keep their own layout (no instancing).
  if (!key.wideLine) {
    binding[1].binding = 1;
    binding[1].stride = sizeof(float) * 16; // mat4, 4 x vec4
    binding[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
  }

  VkVertexInputAttributeDescription attributes[8] {};
  attributes[0].location = 0;
  attributes[0].binding = 0;
  attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset = 0;
  attributes[1].location = 1;
  attributes[1].binding = 0;
  attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset = 12;
  attributes[2].location = 2;
  attributes[2].binding = 0;
  attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[2].offset = 24;
  attributes[3].location = 3;
  attributes[3].binding = 0;
  attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
  attributes[3].offset = 40;
  // Instance model matrix rows (binding 1, rate INSTANCE).
  attributes[4].location = 4;
  attributes[4].binding = 1;
  attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[4].offset = 0;
  attributes[5].location = 5;
  attributes[5].binding = 1;
  attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[5].offset = 16;
  attributes[6].location = 6;
  attributes[6].binding = 1;
  attributes[6].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[6].offset = 32;
  attributes[7].location = 7;
  attributes[7].binding = 1;
  attributes[7].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[7].offset = 48;

  // Wide-line layout: clip-space position (0), color (16), polyline
  // distance (32).
  VkVertexInputAttributeDescription wideLineAttributes[3] {};
  wideLineAttributes[0].location = 0;
  wideLineAttributes[0].binding = 0;
  wideLineAttributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  wideLineAttributes[0].offset = 0;
  wideLineAttributes[1].location = 2;
  wideLineAttributes[1].binding = 0;
  wideLineAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  wideLineAttributes[1].offset = 16;
  wideLineAttributes[2].location = 4;
  wideLineAttributes[2].binding = 0;
  wideLineAttributes[2].format = VK_FORMAT_R32_SFLOAT;
  wideLineAttributes[2].offset = 32;

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = key.wideLine ? 1u : 2u;
  vertexInput.pVertexBindingDescriptions = binding;
  vertexInput.vertexAttributeDescriptionCount = key.wideLine ? 3u : 8u;
  vertexInput.pVertexAttributeDescriptions =
    key.wideLine ? wideLineAttributes : attributes;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = key.wideLine
    ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    : topologyToVk(command.geometry.topology);
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  const uint8_t fillMode = fillModeOverride >= 0
                             ? static_cast<uint8_t>(fillModeOverride)
                             : command.state.raster.fillMode;
  // The overlay fill mode passed in by recordFrame() uses SoDrawStyleElement
  // style values, and the retained IR stores the same encoding (see
  // SoRenderIR::fillRenderStateFromState): FILLED=0, LINES=1, POINTS=2.
  //
  // Wide lines expand each segment into FILLED quads (drawn as a triangle
  // list), so the polygon mode must be FILL regardless of the underlying
  // draw style.  Using the inherited LINES mode here rasterizes the quad's
  // edges as hairline wireframe instead of the solid line, which makes the
  // expanded quads (a few pixels wide) effectively invisible -- the
  // "wide lines don't render" symptom.
  rasterization.polygonMode =
    key.wideLine ? VK_POLYGON_MODE_FILL
    : fillMode == SoDrawStyleElement::LINES ? VK_POLYGON_MODE_LINE
    : fillMode == SoDrawStyleElement::POINTS ? VK_POLYGON_MODE_POINT
    : VK_POLYGON_MODE_FILL;
  // The vertex shader flips Y to match Coin's bottom-left origin; that
  // reflection reverses screen winding, so the Vulkan front face is the
  // inverse of the GL vertex ordering captured in the IR.  Back-face
  // culling matches GL: only shapes declaring an explicit winding plus
  // SOLID shape type cull (ccwFrontFace/cullMode above).  FreeCAD BRep
  // tessellations declare COUNTERCLOCKWISE/SOLID, so closed parts cull
  // back faces here exactly like the GL pipeline does.
  rasterization.cullMode =
    key.wideLine || !key.cullMode ? VK_CULL_MODE_NONE
                                  : VK_CULL_MODE_BACK_BIT;
  rasterization.frontFace = key.ccwFrontFace
    ? VK_FRONT_FACE_CLOCKWISE
    : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;
  // Depth bias: wireframe/point overlays pull toward the camera so they pass
  // the depth test against coplanar filled geometry; selection/overlay faces
  // carry an explicit SoPolygonOffsetElement captured into the raster state.
  rasterization.depthBiasEnable = depthBias ? VK_TRUE : VK_FALSE;
  rasterization.depthBiasConstantFactor = depthBiasConstant;
  rasterization.depthBiasSlopeFactor = depthBiasSlope;

  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType =
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = target.sampleCount;

  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable =
    (command.state.depth.enabled || overlay) ? VK_TRUE : VK_FALSE;
  depthStencil.depthWriteEnable =
    (!transparent && !overlay && command.state.depth.writeEnabled)
      ? VK_TRUE : VK_FALSE;
  depthStencil.depthCompareOp = overlay
    ? VK_COMPARE_OP_LESS_OR_EQUAL
    : depthFunctionToVk(command.state.depth.func);
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = stencil.enabled ? VK_TRUE : VK_FALSE;
  VkStencilOpState stencilState {};
  if (stencil.enabled) {
    stencilState.failOp = stencilOpToVk(stencil.failOp);
    stencilState.passOp = stencilOpToVk(stencil.zpassOp);
    stencilState.depthFailOp = stencilOpToVk(stencil.zfailOp);
    stencilState.compareOp = stencilFunctionToVk(stencil.function);
    stencilState.compareMask = stencil.compareMask;
    stencilState.writeMask = stencil.writeMask;
    stencilState.reference = stencil.reference;
  }
  depthStencil.front = stencilState;
  depthStencil.back = stencilState;

  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendAttachment.blendEnable = blending ? VK_TRUE : VK_FALSE;
  if (command.state.blend.enabled) {
    blendAttachment.srcColorBlendFactor =
      blendFactorToVk(command.state.blend.srcRGBFactor);
    blendAttachment.dstColorBlendFactor =
      blendFactorToVk(command.state.blend.dstRGBFactor);
    blendAttachment.colorBlendOp =
      blendEquationToVk(command.state.blend.rgbEquation);
    blendAttachment.srcAlphaBlendFactor =
      blendFactorToVk(command.state.blend.srcAlphaFactor);
    blendAttachment.dstAlphaBlendFactor =
      blendFactorToVk(command.state.blend.dstAlphaFactor);
    blendAttachment.alphaBlendOp =
      blendEquationToVk(command.state.blend.alphaEquation);
  }
  else {
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  }

  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType =
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.logicOpEnable = VK_FALSE;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkGraphicsPipelineCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci.stageCount = 2;
  ci.pStages = stages;
  ci.pVertexInputState = &vertexInput;
  ci.pInputAssemblyState = &inputAssembly;
  ci.pViewportState = &viewportState;
  ci.pRasterizationState = &rasterization;
  ci.pMultisampleState = &multisample;
  ci.pDepthStencilState = &depthStencil;
  ci.pColorBlendState = &colorBlend;
  ci.pDynamicState = &dynamicState;
  ci.layout = this->pipelineLayout;
  ci.renderPass = pass;
  ci.subpass = 0;

  VkPipeline created = VK_NULL_HANDLE;
  const VkResult result =
    vkCreateGraphicsPipelines(this->device, this->pipelineCacheHandle, 1, &ci,
                              this->allocator, &created);
  if (result != VK_SUCCESS) {
    this->emitError("failed to create Vulkan graphics pipeline");
    this->pipelineCache[key] = VK_NULL_HANDLE;
    if (cacheEntry) {
      cacheEntry->resolvedKey = key;
      cacheEntry->resolvedPipeline = VK_NULL_HANDLE;
      cacheEntry->hasResolvedPipeline = true;
    }
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->pipelineCache[key] = created;
  if (cacheEntry) {
    cacheEntry->resolvedKey = key;
    cacheEntry->resolvedPipeline = created;
    cacheEntry->hasResolvedPipeline = true;
  }
  pipeline = created;
  return true;
}
