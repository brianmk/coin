// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendPipeline.cpp
//
// Graphics pipeline and render-pass management.  getOrCreatePipeline() builds
// and caches an immutable VkPipeline per unique retained state (topology,
// fill/cull, depth, blend, stencil, sample count, wide-line); recordBackground()
// draws the gradient.  Render-pass/framebuffer cache: SoVulkanRenderPassCache.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanDebug.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanConfig.h"

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

VkPipeline
SoVulkanRenderBackend::createGraphicsPipeline(
  VkPipelineLayout layout, VkRenderPass renderPass,
  const VkPipelineShaderStageCreateInfo stages[2],
  const VkPipelineVertexInputStateCreateInfo & vertexInput,
  const VkPipelineInputAssemblyStateCreateInfo & inputAssembly,
  const VkPipelineRasterizationStateCreateInfo & rasterization,
  VkSampleCountFlagBits sampleCount,
  const VkPipelineDepthStencilStateCreateInfo & depthStencil,
  const VkPipelineColorBlendAttachmentState & blendAttachment)
{
  // Fixed shared state: dynamic viewport/scissor, one color attachment, no logic op.
  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = sampleCount;

  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.logicOpEnable = VK_FALSE;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
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
  ci.layout = layout;
  ci.renderPass = renderPass;
  ci.subpass = 0;

  const bool wantFeedback =
    this->hasPipelineCreationFeedback &&
    SoVulkanConfig::get().diagnostics.pipelineFeedback;
  VkPipelineCreationFeedbackEXT feedback {};
  VkPipelineCreationFeedbackCreateInfoEXT feedbackInfo {};
  if (wantFeedback) {
    feedbackInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT;
    feedbackInfo.pPipelineCreationFeedback = &feedback;
    feedbackInfo.pipelineStageCreationFeedbackCount = 0;
    ci.pNext = &feedbackInfo;
  }

  VkPipeline created = VK_NULL_HANDLE;
  if (vkCreateGraphicsPipelines(this->device, this->pipelines.handle(), 1,
                                &ci, this->allocator, &created) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  if (wantFeedback) {
    const bool cacheHit =
      (feedback.flags &
       VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT_EXT) != 0;
    SoVulkanDebug::post("[RTDBG] pipelineFeedback raster cacheHit=%d creation=%.3fus\n",
                 cacheHit ? 1 : 0,
                 static_cast<double>(feedback.duration) * 1.0e-3);
  }
  return created;
}

bool
SoVulkanRenderBackend::createBackgroundPipeline(
  const SoVulkanRenderTarget & target,
  VkRenderPass renderPass,
  VkPipeline & pipeline)
{
  BackgroundPipelineKey key;
  key.renderPass = renderPass;
  key.sampleCount = target.sampleCount;
  if (this->pipelines.findBackground(key, pipeline)) {
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

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  rasterization.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization.cullMode = VK_CULL_MODE_NONE;
  rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;

  // Gradient fills the viewport and writes no depth, so later geometry is unaffected.
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

  const VkPipeline created = this->createGraphicsPipeline(
    this->backgroundPipelineLayout, renderPass, stages, vertexInput,
    inputAssembly, rasterization, target.sampleCount, depthStencil,
    blendAttachment);
  if (created == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan background pipeline");
    this->pipelines.storeBackground(key, VK_NULL_HANDLE);
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->pipelines.storeBackground(key, created);
  pipeline = created;
  return true;
}

void
SoVulkanRenderBackend::recordBackground(const SoRenderParams & params,
                                        const SoVulkanRenderTarget & target,
                                        VkRenderPass renderPass,
                                        VulkanRecordContext & ctx)
{
  if (!params.backgroundGradient) {
    return;
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->createBackgroundPipeline(target, renderPass, pipeline) ||
      pipeline == VK_NULL_HANDLE) {
    return;
  }

  // Gradient covers exactly the viewport (same Y-flip as applyViewport()).
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  const VkRect2D rect = toVkRect(clampFlippedRect(
    origin[0], origin[1], size[0], size[1], target.extent));
  if (rect.extent.width == 0 || rect.extent.height == 0) return;

  VkViewport viewport {};
  viewport.x = static_cast<float>(rect.offset.x);
  viewport.y = static_cast<float>(rect.offset.y);
  viewport.width = static_cast<float>(rect.extent.width);
  viewport.height = static_cast<float>(rect.extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  this->applyViewportState(viewport, ctx);

  this->applyScissorState(rect, ctx);

  this->applyPipeline(pipeline, ctx);

  VulkanBackgroundPush push {};
  push.topColor[0] = params.backgroundTopColor[0];
  push.topColor[1] = params.backgroundTopColor[1];
  push.topColor[2] = params.backgroundTopColor[2];
  push.topColor[3] = params.backgroundTopColor[3];
  push.bottomColor[0] = params.backgroundBottomColor[0];
  push.bottomColor[1] = params.backgroundBottomColor[1];
  push.bottomColor[2] = params.backgroundBottomColor[2];
  push.bottomColor[3] = params.backgroundBottomColor[3];
  push.viewport[0] = static_cast<float>(rect.extent.width);
  push.viewport[1] = static_cast<float>(rect.extent.height);
  push.viewport[2] = static_cast<float>(rect.offset.x);
  push.viewport[3] = static_cast<float>(rect.offset.y);
  vkCmdPushConstants(ctx.buffer, this->backgroundPipelineLayout,
                      VK_SHADER_STAGE_VERTEX_BIT |
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(push), &push);

  vkCmdDraw(ctx.buffer, 3, 1, 0, 0);
}

bool
SoVulkanRenderBackend::getOrCreatePipeline(const SoRenderCommand & command,
                                           const SoVulkanRenderTarget & target,
                                           VkRenderPass pass,
                                           VkPipeline & pipeline,
                                           const bool transparent,
                                           const int fillModeOverride,
                                           const bool overlayPass,
                                           VulkanCachedCommand * cacheEntry)
{
  if (vkBackendTraceEnabled()) {
    static std::atomic<int> n(0);
    if (n.fetch_add(1) < 32) {
      vkBackendTrace(this->uboFrameIndex, "getOrCreatePipeline.enter",
                     "call=%d", n.load());
    }
  }
  // Pipelines are immutable: key on every retained state value that changes the
  // created pipeline (topology, fill, depth/blend, sample count) so incompatible
  // states never share one.  Shading/texture/lighting are uniforms, not keyed yet.
  const bool blending = transparent || command.state.blend.enabled ||
                        command.material.diffuse[3] < 0.999f;
  const bool overlay = fillModeOverride >= 0;
  // SoPolygonOffsetElement's captured depth bias pulls selection/overlay faces
  // in front of coplanar base geometry (GL glPolygonOffset semantics); key it.
  const bool polygonOffset =
    command.state.raster.polygonOffsetFactor != 0.0f ||
    command.state.raster.polygonOffsetUnits != 0.0f;
  const bool depthBias = overlay || polygonOffset;
  // GL offset units map ~1:1 to depthBiasConstantFactor, but the min-resolvable-
  // depth step differs (GL 24-bit fixed vs float D32 here), so a GL-sized offset
  // still Z-fights.  Scale the decal up; slope factor keeps grazing edges attached.
  constexpr float kDecalScale = 512.0f;
  const float kUseDecal = COIN_VULKAN_ENV_FLAG("COIN_VULKAN_RASTER_DECAL")
    ? kDecalScale : 1.0f;
  const float depthBiasConstant = polygonOffset
    ? command.state.raster.polygonOffsetUnits * kUseDecal
    : (overlay ? -0.5f : 0.0f);
  const float depthBiasSlope = polygonOffset
    ? command.state.raster.polygonOffsetFactor * kUseDecal
    : (overlay ? -0.5f : 0.0f);
  PipelineKey key;
  // Wide-line (width > 1 and/or stipple) draws each segment as a quad: eligible
  // commands expand on the GPU (key.wideLineInstanced), the rest on the CPU;
  // the overlay wireframe redraw stays on the plain line path.
  key.wideLine = isWideLine(command, fillModeOverride, this->interactionLodActive);
  // GPU-instanced variant when eligible AND its instance endpoint buffer exists.
  // The record path and expansion pre-pass use the identical test, so key == draw.
  key.wideLineInstanced = key.wideLine && isInstancedWideLine(command) &&
    cacheEntry != nullptr &&
    cacheEntry->instancedLineBuffer != VK_NULL_HANDLE;
  key.renderPass = pass;
  key.topology = command.geometry.topology;
  key.fillMode = overlay ? static_cast<uint8_t>(fillModeOverride)
                          : command.state.raster.fillMode;
  key.cullMode = overlay ? 0 : command.state.raster.cullMode;
  key.ccwFrontFace = command.state.raster.ccwFrontFace;
  key.depthTestEnable = command.state.depth.enabled || overlay;
  // Overlay-pass geometry (e.g. navigation cube) draws last, keeps depth writes
  // to self-occlude; wireframe/point redraws deliberately disable depth writes.
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

  // Per-command fast path: an unchanged command reuses its resolved VkPipeline
  // without the map lookup.  The key is stored on the geometry cache entry
  // (destroyed with the pipeline cache, so never dangling); callers that already
  // resolved it (recordDrawCommand/recordCommandBatch) pass it in to skip the lookup.
  VulkanCachedCommand * entry = cacheEntry;
  if (entry == nullptr) {
    const auto cmdEntry = this->commandToCache.find(&command);
    if (cmdEntry != this->commandToCache.end()) {
      entry = &this->gpuCache[cmdEntry->second];
    }
  }
  if (entry && entry->hasResolvedPipeline && entry->resolvedKey == key) {
    pipeline = entry->resolvedPipeline;
    return pipeline != VK_NULL_HANDLE;
  }

  if (this->pipelines.find(key, pipeline)) {
    if (entry) {
      entry->resolvedKey = key;
      entry->resolvedPipeline = pipeline;
      entry->hasResolvedPipeline = true;
    }
    return pipeline != VK_NULL_HANDLE;
  }

  // VK_POLYGON_MODE_LINE/POINT need the fillModeNonSolid device feature; without
  // it creation is a spec violation (VUID-VkPipelineRasterizationStateCreateInfo-
  // polygonMode-01507) and can fail or hang drivers.  Refuse, caching the failure
  // so the warning emits once and later lookups return false cheaply.
  if (!this->fillModeNonSolid && !key.wideLine &&
      (key.fillMode == SoDrawStyleElement::LINES ||
       key.fillMode == SoDrawStyleElement::POINTS)) {
    this->emitError(
      "Vulkan backend: the device does not support the fillModeNonSolid "
      "feature; wireframe and point fill modes cannot be rendered");
    this->pipelines.store(key, VK_NULL_HANDLE);
    if (entry) {
      entry->resolvedKey = key;
      entry->resolvedPipeline = VK_NULL_HANDLE;
      entry->hasResolvedPipeline = true;
    }
    pipeline = VK_NULL_HANDLE;
    return false;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = key.wideLineInstanced ? this->wideLineInstancedVertexModule
                    : key.wideLine ? this->wideLineVertexModule
                                   : this->vertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = key.wideLine ? this->wideLineFragmentModule
                                  : this->fragmentModule;
  stages[1].pName = "main";

  // Binding 0: interleaved position/normal/color/texcoord (wide-line substitutes
  // its own 36-byte clip-space layout).
  VkVertexInputBindingDescription binding[2] {};
  binding[0].binding = 0;
  // Instanced wide lines read one segment/instance from binding 0 (p0,p1,c0,c1);
  // CPU-expanded reads the 36-byte quad stream; visual reads the interleaved vertex.
  binding[0].stride = key.wideLineInstanced ? sizeof(float) * 16
                     : key.wideLine ? 36u : VULKAN_VERTEX_STRIDE;
  binding[0].inputRate = key.wideLineInstanced
    ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
  // Binding 1: per-instance model matrix (4 R32G32B32A32 rows, rate INSTANCE),
  // used by visual + GPU-instanced wide-line; CPU-expanded uses one binding only.
  if (!key.wideLine || key.wideLineInstanced) {
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
  attributes[2].format = VK_FORMAT_R8G8B8A8_UNORM;
  attributes[2].offset = 24;
  attributes[3].location = 3;
  attributes[3].binding = 0;
  attributes[3].format = VK_FORMAT_R16G16_SFLOAT;
  attributes[3].offset = 28;
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

  // Wide-line layout: clip position@0, color@16, polyline distance@32.
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

  // GPU-instanced layout: segment endpoints/colors at binding 0 (loc 0..3),
  // per-instance model matrix at binding 1 (loc 4..7, matching the visual pass).
  VkVertexInputAttributeDescription instancedLineAttributes[8] {};
  instancedLineAttributes[0] = { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
  instancedLineAttributes[1] = { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
  instancedLineAttributes[2] = { 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 32 };
  instancedLineAttributes[3] = { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 48 };
  instancedLineAttributes[4] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
  instancedLineAttributes[5] = { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
  instancedLineAttributes[6] = { 6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 };
  instancedLineAttributes[7] = { 7, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48 };

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.pVertexBindingDescriptions = binding;
  if (key.wideLineInstanced) {
    vertexInput.vertexBindingDescriptionCount = 2u;
    vertexInput.vertexAttributeDescriptionCount = 8u;
    vertexInput.pVertexAttributeDescriptions = instancedLineAttributes;
  }
  else if (key.wideLine) {
    vertexInput.vertexBindingDescriptionCount = 1u;
    vertexInput.vertexAttributeDescriptionCount = 3u;
    vertexInput.pVertexAttributeDescriptions = wideLineAttributes;
  }
  else {
    vertexInput.vertexBindingDescriptionCount = 2u;
    vertexInput.vertexAttributeDescriptionCount = 8u;
    vertexInput.pVertexAttributeDescriptions = attributes;
  }

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = key.wideLine
    ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    : topologyToVk(command.geometry.topology);
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  const uint8_t fillMode = fillModeOverride >= 0
                             ? static_cast<uint8_t>(fillModeOverride)
                             : command.state.raster.fillMode;
  // SoDrawStyleElement values == retained IR encoding (FILLED=0, LINES=1,
  // POINTS=2).  Wide lines expand to FILLED quads (triangle list), so polygonMode
  // must be FILL -- LINES rasterizes hairline edges and the widened quads vanish.
  rasterization.polygonMode =
    key.wideLine ? VK_POLYGON_MODE_FILL
    : fillMode == SoDrawStyleElement::LINES ? VK_POLYGON_MODE_LINE
    : fillMode == SoDrawStyleElement::POINTS ? VK_POLYGON_MODE_POINT
    : VK_POLYGON_MODE_FILL;
  // Vertex shader flips Y for Coin's bottom-left origin; that reflection reverses
  // winding, so Vulkan frontFace inverts GL's IR order.  Back-face culling matches
  // GL: only explicit winding + SOLID shapes cull (FreeCAD BRep is CCW/SOLID).
  rasterization.cullMode =
    key.wideLine || !key.cullMode ? VK_CULL_MODE_NONE
                                  : VK_CULL_MODE_BACK_BIT;
  rasterization.frontFace = key.ccwFrontFace
    ? VK_FRONT_FACE_CLOCKWISE
    : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;
  // Depth bias: wireframe/point overlays pull toward the camera to beat coplanar
  // filled geometry; selection faces carry SoPolygonOffsetElement bias.
  rasterization.depthBiasEnable = depthBias ? VK_TRUE : VK_FALSE;
  rasterization.depthBiasConstantFactor = depthBiasConstant;
  rasterization.depthBiasSlopeFactor = depthBiasSlope;

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

  const VkPipeline created = this->createGraphicsPipeline(
    this->pipelineLayout, pass, stages, vertexInput, inputAssembly,
    rasterization, target.sampleCount, depthStencil, blendAttachment);
  if (created == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan graphics pipeline");
    this->pipelines.store(key, VK_NULL_HANDLE);
    if (entry) {
      entry->resolvedKey = key;
      entry->resolvedPipeline = VK_NULL_HANDLE;
      entry->hasResolvedPipeline = true;
    }
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->pipelines.store(key, created);
  if (entry) {
    entry->resolvedKey = key;
    entry->resolvedPipeline = created;
    entry->hasResolvedPipeline = true;
  }
  pipeline = created;
  return true;
}
