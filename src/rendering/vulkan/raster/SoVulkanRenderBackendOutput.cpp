// src/rendering/vulkan/raster/SoVulkanRenderBackendOutput.cpp
//
// HDR output pass for the raster backend.
//
// renderExternalHdr() renders the scene into a backend-owned RGBA16F
// intermediate (display-referred sRGB, as the visual shaders write it) and then
// presents it into the caller's FP16 swapchain framebuffer as scRGB
// extended-linear sRGB.  Doing the display transform once, after all geometry
// and transparency have blended, keeps it in one place; the sRGB->linear decode
// and the diffuse-white gain happen there too (see
// data/shaders/vulkan/output/OutputFragment.glsl).
//
// The SDR path is unchanged: when HDR is off the manager uses renderExternal()
// and the scene is drawn directly into the caller's framebuffer, byte-identical
// to the pre-HDR renderer.

#include "rendering/vulkan/raster/SoVulkanRenderBackend.h"
#include "rendering/vulkan/raster/SoVulkanRenderBackendP.h"
#include "rendering/vulkan/raster/SoVulkanRenderPassCache.h"
#include "rendering/vulkan/common/core/SoVulkanShared.h"

#include "rendering/vulkan/generated/shaders/output/OutputFragment.spv.h"
#include "rendering/vulkan/generated/shaders/output/OutputVertex.spv.h"

#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace CoinVulkanDetail;

void
SoVulkanRenderBackend::setHdrOutput(SbBool enabled, float exposure, int toneMap)
{
  this->hdrOutput = enabled != FALSE;
  if (exposure > 0.0f) {
    this->hdrExposure = exposure;
  }
  this->hdrToneMap = toneMap;
}

// Release the per-frame intermediate images/depth/framebuffers and the
// offscreen render pass.  Destruction is deferred through the frame ring (like
// every other resource replaced mid-session) so an in-flight submission that
// still references them drains first.  The output descriptor sets are owned by
// the append-only output descriptor pools and freed with them in
// destroyHdrOutputResources(), not here.
void
SoVulkanRenderBackend::releaseHdrIntermediate()
{
  const VkDevice dev = this->device;
  const VkAllocationCallbacks * alloc = this->allocator;
  VmaAllocator vma = this->vmaAllocator;
  for (HdrFrame & f : this->hdrFrames) {
    const VkImage color = f.colorImage;
    const VmaAllocation colorMem = f.colorMemory;
    const VkImageView colorView = f.colorView;
    const VkImage depth = f.depthImage;
    const VmaAllocation depthMem = f.depthMemory;
    const VkImageView depthView = f.depthView;
    const VkFramebuffer framebuffer = f.framebuffer;
    f = HdrFrame {};
    if (color == VK_NULL_HANDLE && depth == VK_NULL_HANDLE &&
        framebuffer == VK_NULL_HANDLE && colorView == VK_NULL_HANDLE &&
        depthView == VK_NULL_HANDLE) {
      continue;
    }
    this->deferDestroy([dev, alloc, vma, color, colorMem, colorView, depth,
                        depthMem, depthView, framebuffer]() {
      if (framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(dev, framebuffer, alloc);
      }
      if (colorView != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, colorView, alloc);
      }
      if (color != VK_NULL_HANDLE) {
        vmaDestroyImage(vma, color, colorMem);
      }
      if (depthView != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, depthView, alloc);
      }
      if (depth != VK_NULL_HANDLE) {
        vmaDestroyImage(vma, depth, depthMem);
      }
    });
  }
  this->hdrFrames.clear();
  this->hdrExtent = {0, 0};
  this->hdrRenderPass = VK_NULL_HANDLE;
  this->hdrPasses.destroyAll();
}

void
SoVulkanRenderBackend::destroyHdrOutputResources()
{
  for (auto & it : this->outputPipelines) {
    if (it.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, it.second, this->allocator);
    }
  }
  this->outputPipelines.clear();
  if (this->outputPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->outputPipelineLayout,
                            this->allocator);
    this->outputPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->outputSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->outputSetLayout,
                                 this->allocator);
    this->outputSetLayout = VK_NULL_HANDLE;
  }
  for (VkDescriptorPool pool : this->outputDescriptorPools) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(this->device, pool, this->allocator);
    }
  }
  this->outputDescriptorPools.clear();
  this->outputDescriptorSetCount = 0;
  if (this->outputSampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->outputSampler, this->allocator);
    this->outputSampler = VK_NULL_HANDLE;
  }
  if (this->outputVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->outputVertexModule,
                          this->allocator);
    this->outputVertexModule = VK_NULL_HANDLE;
  }
  if (this->outputFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->outputFragmentModule,
                          this->allocator);
    this->outputFragmentModule = VK_NULL_HANDLE;
  }
  this->releaseHdrIntermediate();
}

bool
SoVulkanRenderBackend::ensureHdrIntermediate(
    const SoVulkanRenderTarget & outputTarget, SoVulkanRenderTarget & outTarget,
    VkRenderPass & outPass, VkFramebuffer & outFramebuffer,
    VkDescriptorSet & outDescriptorSet)
{
  // One-time output-pass objects: descriptor set layout, pipeline layout and
  // sampler.  The pipeline itself is created per output render pass in
  // ensureOutputPipeline().
  if (this->outputSetLayout == VK_NULL_HANDLE) {
    VkDescriptorSetLayoutBinding binding {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 1;
    ci.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(this->device, &ci, this->allocator,
                                    &this->outputSetLayout) != VK_SUCCESS) {
      this->emitError("ensureHdrIntermediate: descriptor set layout failed");
      return false;
    }
  }
  if (this->outputPipelineLayout == VK_NULL_HANDLE) {
    VkPushConstantRange range {};
    range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    range.offset = 0;
    range.size = 16;
    VkPipelineLayoutCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &this->outputSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(this->device, &ci, this->allocator,
                               &this->outputPipelineLayout) != VK_SUCCESS) {
      this->emitError("ensureHdrIntermediate: pipeline layout failed");
      return false;
    }
  }
  if (this->outputSampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(this->device, &si, this->allocator,
                        &this->outputSampler) != VK_SUCCESS) {
      this->emitError("ensureHdrIntermediate: sampler failed");
      return false;
    }
  }

  const VkExtent2D extent = outputTarget.extent;
  if (extent.width == 0 || extent.height == 0) {
    this->emitError("ensureHdrIntermediate: zero extent");
    return false;
  }

  // Ring depth: one intermediate per in-flight frame, with one extra slot of
  // margin over QVulkanWindow's swapchain-images-in-flight (the embedding sets
  // maxFramesInFlight accordingly; see QuarterVulkanWidget).
  const uint32_t slotCount = std::max(1u, this->maxFramesInFlight);
  const bool rebuild = this->hdrFrames.size() != slotCount ||
                       this->hdrExtent.width != extent.width ||
                       this->hdrExtent.height != extent.height ||
                       this->hdrRenderPass == VK_NULL_HANDLE;
  if (rebuild) {
    this->releaseHdrIntermediate();
    this->hdrFrames.resize(slotCount);
    this->hdrExtent = extent;

    const VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    auto createImage = [this, &extent](VkFormat format, VkImageUsageFlags usage,
                                       VkImage & image, VmaAllocation & mem) {
      VkImageCreateInfo ci {};
      ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      ci.imageType = VK_IMAGE_TYPE_2D;
      ci.format = format;
      ci.extent = {extent.width, extent.height, 1};
      ci.mipLevels = 1;
      ci.arrayLayers = 1;
      ci.samples = VK_SAMPLE_COUNT_1_BIT;
      ci.tiling = VK_IMAGE_TILING_OPTIMAL;
      ci.usage = usage;
      ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      VmaAllocationCreateInfo ai {};
      ai.usage = VMA_MEMORY_USAGE_AUTO;
      ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      return vmaCreateImage(this->vmaAllocator, &ci, &ai, &image, &mem,
                            nullptr) == VK_SUCCESS;
    };

    // Build every ring slot's images/views and the output descriptor set that
    // samples its color view.  A set is never freed while a frame may reference
    // it, so each slot gets its own from the append-only pool.
    for (HdrFrame & f : this->hdrFrames) {
      if (!createImage(colorFormat,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT,
                       f.colorImage, f.colorMemory)) {
        this->emitError("ensureHdrIntermediate: color image failed");
        this->releaseHdrIntermediate();
        return false;
      }
      f.colorView = createImageView(this->device, f.colorImage, colorFormat,
                                    VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
      if (f.colorView == VK_NULL_HANDLE) {
        this->emitError("ensureHdrIntermediate: color view failed");
        this->releaseHdrIntermediate();
        return false;
      }
      if (!createImage(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       f.depthImage, f.depthMemory)) {
        this->emitError("ensureHdrIntermediate: depth image failed");
        this->releaseHdrIntermediate();
        return false;
      }
      f.depthView = createImageView(this->device, f.depthImage, depthFormat,
                                    VK_IMAGE_ASPECT_DEPTH_BIT, this->allocator);
      if (f.depthView == VK_NULL_HANDLE) {
        this->emitError("ensureHdrIntermediate: depth view failed");
        this->releaseHdrIntermediate();
        return false;
      }
      if (!this->ensureOutputDescriptorSet(f.colorView,
                                           f.outputDescriptorSet)) {
        this->releaseHdrIntermediate();
        return false;
      }
    }

    // Shared offscreen render pass (identical for every slot: same formats,
    // sample count and load ops).  The pass identity depends on whether a depth
    // attachment is present, so derive it from a fully-populated slot target.
    SoVulkanRenderTarget desc {};
    desc.colorImage = this->hdrFrames[0].colorImage;
    desc.colorImageView = this->hdrFrames[0].colorView;
    desc.colorFormat = colorFormat;
    desc.colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    desc.depthImage = this->hdrFrames[0].depthImage;
    desc.depthImageView = this->hdrFrames[0].depthView;
    desc.depthFormat = depthFormat;
    desc.depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    desc.extent = extent;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    this->hdrRenderPass = this->hdrPasses.getOrCreateRenderPass(
      desc, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR);
    if (this->hdrRenderPass == VK_NULL_HANDLE) {
      this->emitError("ensureHdrIntermediate: offscreen render pass failed");
      this->releaseHdrIntermediate();
      return false;
    }

    // One framebuffer per slot (created directly, not through hdrPasses, which
    // caches a single "current" framebuffer and would thrash every frame).
    for (HdrFrame & f : this->hdrFrames) {
      VkImageView attachments[2] = {f.colorView, f.depthView};
      VkFramebufferCreateInfo ci {};
      ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      ci.renderPass = this->hdrRenderPass;
      ci.attachmentCount = 2;
      ci.pAttachments = attachments;
      ci.width = extent.width;
      ci.height = extent.height;
      ci.layers = 1;
      if (vkCreateFramebuffer(this->device, &ci, this->allocator,
                              &f.framebuffer) != VK_SUCCESS) {
        this->emitError("ensureHdrIntermediate: framebuffer failed");
        this->releaseHdrIntermediate();
        return false;
      }
    }
  }

  const uint32_t slot = this->uboFrameIndex % slotCount;
  if (slot >= this->hdrFrames.size()) {
    this->emitError("ensureHdrIntermediate: ring slot out of range");
    return false;
  }
  HdrFrame & frame = this->hdrFrames[slot];

  outTarget = SoVulkanRenderTarget {};
  outTarget.colorImage = frame.colorImage;
  outTarget.colorImageView = frame.colorView;
  outTarget.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  outTarget.colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  outTarget.depthImage = frame.depthImage;
  outTarget.depthImageView = frame.depthView;
  outTarget.depthFormat = VK_FORMAT_D32_SFLOAT;
  outTarget.depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  outTarget.extent = extent;
  outTarget.sampleCount = VK_SAMPLE_COUNT_1_BIT;

  outPass = this->hdrRenderPass;
  outFramebuffer = frame.framebuffer;
  outDescriptorSet = frame.outputDescriptorSet;
  return outFramebuffer != VK_NULL_HANDLE;
}

bool
SoVulkanRenderBackend::ensureOutputDescriptorSet(VkImageView source,
                                                 VkDescriptorSet & outSet)
{
  if (this->outputDescriptorSetCount == 0
      || this->outputDescriptorPools.empty()) {
    VkDescriptorPoolSize size {};
    size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    size.descriptorCount = 4;
    VkDescriptorPoolCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 4;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(this->device, &ci, this->allocator, &pool)
        != VK_SUCCESS) {
      this->emitError("ensureOutputDescriptorSet: pool failed");
      return false;
    }
    this->outputDescriptorPools.push_back(pool);
    this->outputDescriptorSetCount = 4;
  }
  VkDescriptorSetAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = this->outputDescriptorPools.back();
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &this->outputSetLayout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(this->device, &ai, &set) != VK_SUCCESS) {
    this->emitError("ensureOutputDescriptorSet: allocate failed");
    return false;
  }
  --this->outputDescriptorSetCount;

  VkDescriptorImageInfo imageInfo {};
  imageInfo.sampler = this->outputSampler;
  imageInfo.imageView = source;
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkWriteDescriptorSet write {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &imageInfo;
  vkUpdateDescriptorSets(this->device, 1, &write, 0, nullptr);
  outSet = set;
  return true;
}

bool
SoVulkanRenderBackend::ensureOutputPipeline(VkRenderPass outputPass,
                                            VkPipeline & outPipeline)
{
  auto found = this->outputPipelines.find(outputPass);
  if (found != this->outputPipelines.end()) {
    outPipeline = found->second;
    return outPipeline != VK_NULL_HANDLE;
  }
  if (this->outputVertexModule == VK_NULL_HANDLE) {
    if (!this->createShaderModule(coin_vulkan_output_vertex_spirv,
                                  coin_vulkan_output_vertex_spirv_count,
                                  this->outputVertexModule)) {
      this->emitError("ensureOutputPipeline: vertex shader failed");
      return false;
    }
  }
  if (this->outputFragmentModule == VK_NULL_HANDLE) {
    if (!this->createShaderModule(coin_vulkan_output_fragment_spirv,
                                  coin_vulkan_output_fragment_spirv_count,
                                  this->outputFragmentModule)) {
      this->emitError("ensureOutputPipeline: fragment shader failed");
      return false;
    }
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = this->outputVertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = this->outputFragmentModule;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization.cullMode = VK_CULL_MODE_NONE;
  rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization.lineWidth = 1.0f;

  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_FALSE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState blend {};
  blend.blendEnable = VK_FALSE;
  blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
    | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  // The output pass is the caller's swapchain pass; its sample count is the
  // output target's (the pipeline must match it).
  const VkSampleCountFlagBits samples =
    this->hdrOutputPassSampleCount != 0 ? this->hdrOutputPassSampleCount
                                        : VK_SAMPLE_COUNT_1_BIT;
  const VkPipeline pipeline = this->createGraphicsPipeline(
    this->outputPipelineLayout, outputPass, stages, vertexInput, inputAssembly,
    rasterization, samples, depthStencil, blend);
  if (pipeline == VK_NULL_HANDLE) {
    this->emitError("ensureOutputPipeline: pipeline creation failed");
    return false;
  }
  this->outputPipelines.emplace(outputPass, pipeline);
  outPipeline = pipeline;
  return true;
}

SbBool
SoVulkanRenderBackend::renderExternalHdr(const SoDrawList & drawlist,
                                         const SoRenderParams & params,
                                         VkCommandBuffer commandBuffer,
                                         VkRenderPass outputPass,
                                         VkFramebuffer outputFramebuffer)
{
  const SoVulkanRenderTarget * outputTarget = this->validateRenderTarget(params);
  if (outputTarget == nullptr) {
    return FALSE;
  }

  SoVulkanRenderTarget hdrTarget;
  VkRenderPass hdrPass = VK_NULL_HANDLE;
  VkFramebuffer hdrFramebuffer = VK_NULL_HANDLE;
  VkDescriptorSet hdrDescriptorSet = VK_NULL_HANDLE;
  if (!this->ensureHdrIntermediate(*outputTarget, hdrTarget, hdrPass,
                                   hdrFramebuffer, hdrDescriptorSet)) {
    return FALSE;
  }

  // The scene must render into the offscreen target (single-sample RGBA16F),
  // not the caller's swapchain target: the frame plan resolves the target and
  // the pipeline sample count from params.renderTarget, so override it for the
  // scene pass.  The output pass below uses the caller's target explicitly.
  SoRenderParams hdrParams = params;
  hdrParams.renderTarget = &hdrTarget;

  // Per-frame setup + texture/LOD pre-pass.  Vulkan forbids transfer/compute
  // inside a render pass and the caller has not begun one, so the pre-pass is
  // submitted before the offscreen pass.
  ExternalFrameTiming timing;
  FramePlan plan;
  if (!this->prepareExternalFrame(drawlist, hdrParams, commandBuffer, hdrPass,
                                  "renderExternalHdr", /*overlaysOnly*/ false,
                                  /*reserveCompositeSlots*/ false, plan,
                                  &timing)) {
    return FALSE;
  }
  // The offscreen pass clears via its loadOp, so recordClear() must skip the
  // redundant vkCmdClearAttachments (recordClear consults the backend's
  // per-frame clear flags).
  this->frameColorClearedByLoad_ = true;
  this->frameDepthClearedByLoad_ = true;

  // Record the frame's texture uploads and sub-pixel compaction dispatches
  // into the caller's command buffer, ahead of the offscreen pass.  This path
  // begins its own offscreen pass below, so unlike the SDR external path
  // (renderExternal(), which cannot touch the caller's already-begun pass and
  // must use the beginExternalPrepass() transient buffer) the transfer and
  // compute commands can be recorded inline here.  Without this the staged
  // uploads stay pending and are discarded next frame, leaving textured
  // geometry on the white fallback descriptor, and the geometry LOD never runs.
  if (this->textureCache.hasPendingUploads()) {
    this->textureCache.recordPendingInto(commandBuffer);
    this->textureCache.finalizePending();
  }
  if (this->externalGeometryLodActive(hdrParams)) {
    this->recordGeometryLodPrepass(commandBuffer, drawlist, hdrParams);
  }

  // --- Offscreen pass: scene -> linear RGBA16F ---------------------------
  // The intermediate starts UNDEFINED and the previous frame's output pass
  // left it SHADER_READ_ONLY, while the offscreen render pass declares
  // COLOR_ATTACHMENT / DEPTH_STENCIL initial layouts.  Transition explicitly;
  // oldLayout UNDEFINED is always legal and discards the previous contents,
  // which the CLEAR loadOp rewrites anyway.
  SoVulkanShared::imageTransition(
    commandBuffer, hdrTarget.colorImage, VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  SoVulkanShared::imageTransition(
    commandBuffer, hdrTarget.depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0,
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = hdrPass;
  rpbi.framebuffer = hdrFramebuffer;
  rpbi.renderArea.offset = {0, 0};
  rpbi.renderArea.extent = hdrTarget.extent;
  VkClearValue clearValues[2];
  clearValues[0].color.float32[0] = params.clearColor[0];
  clearValues[0].color.float32[1] = params.clearColor[1];
  clearValues[0].color.float32[2] = params.clearColor[2];
  clearValues[0].color.float32[3] = params.clearColor[3];
  clearValues[1].depthStencil.depth = params.clearDepth;
  clearValues[1].depthStencil.stencil = 0;
  rpbi.clearValueCount = 2;
  rpbi.pClearValues = clearValues;
  vkCmdBeginRenderPass(commandBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

  this->recordContext.buffer = commandBuffer;
  const bool recorded = this->recordFramePlan(drawlist, hdrParams, plan,
                                              hdrPass, hdrFramebuffer,
                                              this->recordContext);
  this->recordContext.buffer = VK_NULL_HANDLE;
  vkCmdEndRenderPass(commandBuffer);

  // --- Output pass: RGBA16F -> swapchain (scRGB extended-linear or clamp) --
  SoVulkanShared::imageTransition(
    commandBuffer, hdrTarget.colorImage,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

  this->hdrOutputPassSampleCount = outputTarget->sampleCount;
  VkPipeline outputPipeline = VK_NULL_HANDLE;
  if (!this->ensureOutputPipeline(outputPass, outputPipeline)) {
    return FALSE;
  }

  VkRenderPassBeginInfo outBegin {};
  outBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  outBegin.renderPass = outputPass;
  outBegin.framebuffer = outputFramebuffer;
  outBegin.renderArea.offset = {0, 0};
  outBegin.renderArea.extent = outputTarget->extent;
  // The caller's swapchain pass (QVulkanWindow's default render pass) is
  // created with LOAD_OP_CLEAR on the color, depth and (with MSAA) the MSAA
  // color attachments, so the begin info must carry one clear value per
  // cleared attachment even though the fullscreen triangle overwrites the
  // color (VUID-VkRenderPassBeginInfo-clearValueCount-00902).
  VkClearValue outClear[3] {};
  outClear[0].color.float32[0] = params.clearColor[0];
  outClear[0].color.float32[1] = params.clearColor[1];
  outClear[0].color.float32[2] = params.clearColor[2];
  outClear[0].color.float32[3] = params.clearColor[3];
  outClear[1].depthStencil.depth = 1.0f;
  outClear[1].depthStencil.stencil = 0;
  const bool outMultisample = outputTarget->sampleCount > VK_SAMPLE_COUNT_1_BIT;
  if (outMultisample) {
    outClear[2] = outClear[0];
  }
  outBegin.clearValueCount = outMultisample ? 3u : 2u;
  outBegin.pClearValues = outClear;
  vkCmdBeginRenderPass(commandBuffer, &outBegin, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    outputPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          this->outputPipelineLayout, 0, 1,
                          &hdrDescriptorSet, 0, nullptr);
  const float push[4] = {this->hdrOutput ? 1.0f : 0.0f, this->hdrExposure,
                         static_cast<float>(this->hdrToneMap), 0.0f};
  vkCmdPushConstants(commandBuffer, this->outputPipelineLayout,
                     VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push);

  VkViewport viewport {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(outputTarget->extent.width);
  viewport.height = static_cast<float>(outputTarget->extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
  VkRect2D scissor {};
  scissor.offset = {0, 0};
  scissor.extent = outputTarget->extent;
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  vkCmdDraw(commandBuffer, 3, 1, 0, 0);
  vkCmdEndRenderPass(commandBuffer);

  return recorded ? TRUE : FALSE;
}
