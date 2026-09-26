// src/rendering/vulkan/raytracing/denoise/SoRTXRenderBackendDnsr.cpp

// AMD FidelityFX DNSR denoiser backend - the "dnsr" path-tracing denoiser
// slot.  The MIT shader sources are vendored under
// src/rendering/third_party/fidelityfx/dnsr/ and ported to Vulkan GLSL compute
// in data/shaders/vulkan/rt/denoise/ (DnsrCommon.glsl, DnsrPrefilter.glsl,
// DnsrResolveTemporal.glsl; see DnsrCommon.glsl for the provenance note).
//
// Two device-local GPU passes over the path tracer's G-buffers (no host
// staging):
//   1. DnsrPrefilter       - the DNSR 15-tap edge-stopping spatial filter.
//   2. DnsrResolveTemporal - motion-vector reprojection + disocclusion
//                           rejection + AABB-clipped accumulation against the
//                           previous frame's history.
// The body is compiled out when COIN_BUILD_DNSR_DENOISER=0, in which case the
// declarations in SoRTXRenderBackend.h are still present (mirroring the
// OIDN/RTX pattern) so the dispatch in SoRTXRenderBackendDenoise.cpp stays
// type-correct and simply degrades to OIDN at runtime.

#include "rendering/vulkan/raytracing/rtx/SoRTXRenderBackend.h"
#include "rendering/vulkan/common/core/SoVulkanConfig.h"
#include <Inventor/errors/SoDebugError.h>
#include <cstdio>
#include "rendering/vulkan/generated/shaders/rt/denoise/DnsrPrefilter.spv.h"
#include "rendering/vulkan/generated/shaders/rt/denoise/DnsrResolveTemporal.spv.h"
#include <rendering/vulkan/raytracing/rtx/SoRTXRenderBackendP.h>

#include "vk_mem_alloc.h"

using namespace SoRTXBackend;

#if COIN_BUILD_DNSR_DENOISER

namespace {

// One storage-buffer write descriptor.
VkWriteDescriptorSet dnsrWrite(VkDescriptorSet set, uint32_t binding,
                              const VkDescriptorBufferInfo * info)
{
  VkWriteDescriptorSet w {};
  w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w.dstSet = set;
  w.dstBinding = binding;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo = info;
  return w;
}

} // namespace

bool
SoRTXRenderBackend::createDnsrPipeline()
{
  if (this->dnsrPipelineReady) {
    return true;
  }
  if (this->device == VK_NULL_HANDLE || this->descriptorPool == VK_NULL_HANDLE) {
    return false;
  }
  if (this->accumBuffer == VK_NULL_HANDLE || this->sumSqBuffer == VK_NULL_HANDLE ||
      this->normalBuffer == VK_NULL_HANDLE || this->positionBuffer == VK_NULL_HANDLE ||
      this->motionBuffer == VK_NULL_HANDLE || this->denoisedBuffer == VK_NULL_HANDLE) {
    this->emitError("DNSR denoiser: G-buffers not ready");
    return false;
  }

  const VkDeviceSize imageBytes =
    static_cast<VkDeviceSize>(this->ptBufferWidth) * this->ptBufferHeight * 16;
  const VkBufferUsageFlags bufUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  checkDnsrLayout();
  checkDnsrTemporalLayout();

  // --- Intermediate + history buffers -------------------------------------
  auto makeBuffer = [&](VkBuffer & buf, VmaAllocation & mem, const char * what) {
    if (!this->createDeviceLocalBuffer(imageBytes, bufUsage, buf, mem)) {
      this->emitError(what);
      return false;
    }
    return true;
  };
  if (!makeBuffer(this->dnsrPrefilterRad, this->dnsrPrefilterRadMem,
                  "DNSR denoiser: failed to create prefilter radiance buffer") ||
      !makeBuffer(this->dnsrPrefilterVar, this->dnsrPrefilterVarMem,
                  "DNSR denoiser: failed to create prefilter variance buffer") ||
      !makeBuffer(this->dnsrHistRad[0], this->dnsrHistRadMem[0],
                  "DNSR denoiser: failed to create history radiance buffer") ||
      !makeBuffer(this->dnsrHistRad[1], this->dnsrHistRadMem[1],
                  "DNSR denoiser: failed to create history radiance buffer") ||
      !makeBuffer(this->dnsrHistVar[0], this->dnsrHistVarMem[0],
                  "DNSR denoiser: failed to create history variance buffer") ||
      !makeBuffer(this->dnsrHistVar[1], this->dnsrHistVarMem[1],
                  "DNSR denoiser: failed to create history variance buffer") ||
      !makeBuffer(this->dnsrHistPos, this->dnsrHistPosMem,
                  "DNSR denoiser: failed to create history position buffer") ||
      !makeBuffer(this->dnsrHistNrm, this->dnsrHistNrmMem,
                  "DNSR denoiser: failed to create history normal buffer")) {
    return false;
  }
  this->dnsrHistIndex = 0;

  // --- Prefilter (spatial) stage ------------------------------------------
  {
    VkDescriptorSetLayoutBinding bindings[6] {};
    for (uint32_t b = 0; b < 6; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setCI {};
    setCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setCI.bindingCount = 6;
    setCI.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(this->device, &setCI, this->allocator,
                                    &this->dnsrSetLayout) != VK_SUCCESS) {
      this->dnsrSetLayout = VK_NULL_HANDLE;
      this->emitError("DNSR denoiser: failed to create prefilter set layout");
      return false;
    }
    VkPushConstantRange push {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(DnsrPush);
    VkPipelineLayoutCreateInfo layoutCI {};
    layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts = &this->dnsrSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                               &this->dnsrPipelineLayout) != VK_SUCCESS) {
      this->dnsrPipelineLayout = VK_NULL_HANDLE;
      this->emitError("DNSR denoiser: failed to create prefilter pipeline layout");
      return false;
    }
    VkShaderModuleCreateInfo smCI {};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = coin_vulkan_rt_dnsrprefilter_spirv_count * sizeof(uint32_t);
    smCI.pCode = coin_vulkan_rt_dnsrprefilter_spirv;
    const VkResult prefilterModuleRes = vkCreateShaderModule(
      this->device, &smCI, this->allocator, &this->dnsrModule);
    if (prefilterModuleRes != VK_SUCCESS) {
      this->dnsrModule = VK_NULL_HANDLE;
      this->emitError(
        ("DNSR denoiser: failed to create prefilter module: "
         + SoVulkanShared::vkResultName(prefilterModuleRes)).c_str());
      return false;
    }
    VkComputePipelineCreateInfo computeCI {};
    computeCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeCI.stage.module = this->dnsrModule;
    computeCI.stage.pName = "main";
    computeCI.layout = this->dnsrPipelineLayout;
    const VkResult prefilterRes = vkCreateComputePipelines(
      this->device, VK_NULL_HANDLE, 1, &computeCI, this->allocator,
      &this->dnsrPipeline);
    if (prefilterRes != VK_SUCCESS) {
      this->dnsrPipeline = VK_NULL_HANDLE;
      this->emitError(
        ("DNSR denoiser: failed to create prefilter pipeline: "
         + SoVulkanShared::vkResultName(prefilterRes)).c_str());
      return false;
    }
    VkDescriptorSetAllocateInfo allocCI {};
    allocCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocCI.descriptorPool = this->descriptorPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts = &this->dnsrSetLayout;
    if (vkAllocateDescriptorSets(this->device, &allocCI,
                                 &this->dnsrDescriptorSet) != VK_SUCCESS) {
      this->dnsrDescriptorSet = VK_NULL_HANDLE;
      this->emitError("DNSR denoiser: failed to allocate prefilter set");
      return false;
    }
    VkDescriptorBufferInfo infos[6] = {};
    infos[0].buffer = this->accumBuffer;
    infos[1].buffer = this->sumSqBuffer;
    infos[2].buffer = this->normalBuffer;
    infos[3].buffer = this->positionBuffer;
    infos[4].buffer = this->dnsrPrefilterRad;
    infos[5].buffer = this->dnsrPrefilterVar;
    VkWriteDescriptorSet writes[6] {};
    for (uint32_t b = 0; b < 6; ++b) {
      infos[b].range = VK_WHOLE_SIZE;
      writes[b] = dnsrWrite(this->dnsrDescriptorSet, b, &infos[b]);
    }
    vkUpdateDescriptorSets(this->device, 6, writes, 0, nullptr);
  }

  // --- Temporal reproject + resolve stage ---------------------------------
  {
    VkDescriptorSetLayoutBinding bindings[12] {};
    for (uint32_t b = 0; b < 12; ++b) {
      bindings[b].binding = b;
      bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[b].descriptorCount = 1;
      bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo setCI {};
    setCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setCI.bindingCount = 12;
    setCI.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(this->device, &setCI, this->allocator,
                                    &this->dnsrTemporalSetLayout) != VK_SUCCESS) {
      this->dnsrTemporalSetLayout = VK_NULL_HANDLE;
      this->emitError("DNSR denoiser: failed to create temporal set layout");
      return false;
    }
    VkPushConstantRange push {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(DnsrTemporalPush);
    VkPipelineLayoutCreateInfo layoutCI {};
    layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts = &this->dnsrTemporalSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                               &this->dnsrTemporalPipelineLayout) != VK_SUCCESS) {
      this->dnsrTemporalPipelineLayout = VK_NULL_HANDLE;
      this->emitError("DNSR denoiser: failed to create temporal pipeline layout");
      return false;
    }
    VkShaderModuleCreateInfo smCI {};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize =
      coin_vulkan_rt_dnsrresolvetemporal_spirv_count * sizeof(uint32_t);
    smCI.pCode = coin_vulkan_rt_dnsrresolvetemporal_spirv;
    const VkResult temporalModuleRes = vkCreateShaderModule(
      this->device, &smCI, this->allocator, &this->dnsrTemporalModule);
    if (temporalModuleRes != VK_SUCCESS) {
      this->dnsrTemporalModule = VK_NULL_HANDLE;
      this->emitError(
        ("DNSR denoiser: failed to create temporal module: "
         + SoVulkanShared::vkResultName(temporalModuleRes)).c_str());
      return false;
    }
    VkComputePipelineCreateInfo computeCI {};
    computeCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeCI.stage.module = this->dnsrTemporalModule;
    computeCI.stage.pName = "main";
    computeCI.layout = this->dnsrTemporalPipelineLayout;
    const VkResult temporalRes = vkCreateComputePipelines(
      this->device, VK_NULL_HANDLE, 1, &computeCI, this->allocator,
      &this->dnsrTemporalPipeline);
    if (temporalRes != VK_SUCCESS) {
      this->dnsrTemporalPipeline = VK_NULL_HANDLE;
      this->emitError(
        ("DNSR denoiser: failed to create temporal pipeline: "
         + SoVulkanShared::vkResultName(temporalRes)).c_str());
      return false;
    }
    VkDescriptorSetAllocateInfo allocCI {};
    allocCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocCI.descriptorPool = this->descriptorPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts = &this->dnsrTemporalSetLayout;
    if (vkAllocateDescriptorSets(this->device, &allocCI,
                                 &this->dnsrTemporalDescriptorSet) != VK_SUCCESS) {
      this->dnsrTemporalDescriptorSet = VK_NULL_HANDLE;
      this->emitError("DNSR denoiser: failed to allocate temporal set");
      return false;
    }
  }

  // Seed the history so the first temporal pass reprojects against something
  // valid instead of uninitialized device memory.  Position/normal can be
  // copied from the current G-buffers, but the accumulated history
  // (radiance/variance/sample-count) has no valid prior value: the resolve
  // blends at accumSpeed = 1/max(historyNum, 1), so a garbage sample count
  // would make the very first DNSR frame blend almost entirely from garbage.
  // Zero-fill all three so the first frame sees historyNum = 0 and the current
  // prefiltered frame wins outright.
  {
    VkCommandBuffer cmd = this->beginTransientCommandBuffer();
    if (cmd == VK_NULL_HANDLE) {
      this->emitError(
        "DNSR denoiser: could not allocate the history-seed command buffer");
      return false;
    }
    for (int i = 0; i < 2; ++i) {
      vkCmdFillBuffer(cmd, this->dnsrHistRad[i], 0, imageBytes, 0);
      vkCmdFillBuffer(cmd, this->dnsrHistVar[i], 0, imageBytes, 0);
    }
    VkBufferCopy cp {0, 0, imageBytes};
    vkCmdCopyBuffer(cmd, this->positionBuffer, this->dnsrHistPos, 1, &cp);
    vkCmdCopyBuffer(cmd, this->normalBuffer, this->dnsrHistNrm, 1, &cp);
    SoVulkanShared::memoryBarrier(
      cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT);
    // The seed is not optional: uninitialized history makes the first DNSR
    // frames blend garbage, so a failure here must fail setup (and let the
    // caller degrade to OIDN) instead of proceeding with a blank history.
    const VkResult endRes = vkEndCommandBuffer(cmd);
    if (endRes != VK_SUCCESS) {
      this->emitError(
        ("DNSR denoiser: history seed vkEndCommandBuffer failed: "
         + SoVulkanShared::vkResultName(endRes)).c_str());
      return false;
    }
    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    const VkResult submitRes =
      vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE);
    if (submitRes != VK_SUCCESS) {
      this->emitError(
        ("DNSR denoiser: history seed vkQueueSubmit failed: "
         + SoVulkanShared::vkResultName(submitRes)).c_str());
      return false;
    }
    if (!this->drainQueue("DNSR denoiser: history seed")) return false;
  }

  this->dnsrPipelineReady = true;
  if (SoVulkanConfig::get().rtxDebug.denoiserDebug) {
    fprintf(stderr, "[DENOISE] DNSR (prefilter+temporal) pipeline ready\n");
  }
  return true;
}

void
SoRTXRenderBackend::destroyDnsrResources()
{
  this->dnsrPipelineReady = false;
  auto freeBuf = [this](VkBuffer & buf, VmaAllocation & mem) {
    if (buf != VK_NULL_HANDLE) {
      VmaAllocator vma = this->vmaAllocator;
      VkBuffer b = buf;
      VmaAllocation m = mem;
      buf = VK_NULL_HANDLE;
      mem = VK_NULL_HANDLE;
      this->deferDestroy([vma, b, m]() { vmaDestroyBuffer(vma, b, m); });
    }
  };
  freeBuf(this->dnsrPrefilterRad, this->dnsrPrefilterRadMem);
  freeBuf(this->dnsrPrefilterVar, this->dnsrPrefilterVarMem);
  for (int i = 0; i < 2; ++i) {
    freeBuf(this->dnsrHistRad[i], this->dnsrHistRadMem[i]);
    freeBuf(this->dnsrHistVar[i], this->dnsrHistVarMem[i]);
  }
  freeBuf(this->dnsrHistPos, this->dnsrHistPosMem);
  freeBuf(this->dnsrHistNrm, this->dnsrHistNrmMem);

  this->freeDescriptorSet(this->descriptorPool, this->dnsrTemporalDescriptorSet,
                          "destroyDnsrResources (temporal)");
  this->dnsrTemporalDescriptorSet = VK_NULL_HANDLE;
  this->freeDescriptorSet(this->descriptorPool, this->dnsrDescriptorSet,
                          "destroyDnsrResources (prefilter)");
  this->dnsrDescriptorSet = VK_NULL_HANDLE;
  if (this->dnsrPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->dnsrPipeline, this->allocator);
    this->dnsrPipeline = VK_NULL_HANDLE;
  }
  if (this->dnsrTemporalPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->dnsrTemporalPipeline, this->allocator);
    this->dnsrTemporalPipeline = VK_NULL_HANDLE;
  }
  if (this->dnsrPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->dnsrPipelineLayout, this->allocator);
    this->dnsrPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->dnsrTemporalPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->dnsrTemporalPipelineLayout,
                            this->allocator);
    this->dnsrTemporalPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->dnsrModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->dnsrModule, this->allocator);
    this->dnsrModule = VK_NULL_HANDLE;
  }
  if (this->dnsrTemporalModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->dnsrTemporalModule, this->allocator);
    this->dnsrTemporalModule = VK_NULL_HANDLE;
  }
  if (this->dnsrSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->dnsrSetLayout, this->allocator);
    this->dnsrSetLayout = VK_NULL_HANDLE;
  }
  if (this->dnsrTemporalSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->dnsrTemporalSetLayout,
                                 this->allocator);
    this->dnsrTemporalSetLayout = VK_NULL_HANDLE;
  }
  this->dnsrHistIndex = 0;
}

bool
SoRTXRenderBackend::dispatchDnsrDenoise(uint32_t w, uint32_t h)
{
  if (!this->dnsrPipelineReady || this->dnsrDescriptorSet == VK_NULL_HANDLE ||
      this->dnsrTemporalDescriptorSet == VK_NULL_HANDLE) {
    return false;
  }
  if (w == 0 || h == 0) {
    return false;
  }

  const int read = this->dnsrHistIndex;
  const int write = 1 - read;

  // Bind the temporal set for this frame's history ping-pong slots.
  {
    VkDescriptorBufferInfo infos[12] = {};
    infos[0].buffer = this->dnsrPrefilterRad;
    infos[1].buffer = this->dnsrPrefilterVar;
    infos[2].buffer = this->positionBuffer;
    infos[3].buffer = this->normalBuffer;
    infos[4].buffer = this->motionBuffer;
    infos[5].buffer = this->dnsrHistRad[read];
    infos[6].buffer = this->dnsrHistVar[read];
    infos[7].buffer = this->dnsrHistPos;
    infos[8].buffer = this->dnsrHistNrm;
    infos[9].buffer = this->denoisedBuffer;
    infos[10].buffer = this->dnsrHistRad[write];
    infos[11].buffer = this->dnsrHistVar[write];
    VkWriteDescriptorSet writes[12] {};
    for (uint32_t b = 0; b < 12; ++b) {
      infos[b].range = VK_WHOLE_SIZE;
      writes[b] = dnsrWrite(this->dnsrTemporalDescriptorSet, b, &infos[b]);
    }
    vkUpdateDescriptorSets(this->device, 12, writes, 0, nullptr);
  }

  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    this->emitError("DNSR denoiser: failed to begin command buffer");
    return false;
  }

  // G-buffer writes (ray tracing) -> prefilter reads.
  SoVulkanShared::memoryBarrier(
    cmd,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT);

  // Pass 1: spatial prefilter.
  {
    DnsrPush push;
    push.screen[0] = w;
    push.screen[1] = h;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->dnsrPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->dnsrPipelineLayout, 0, 1,
                            &this->dnsrDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, this->dnsrPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
  }

  // Prefilter writes -> temporal reads.
  SoVulkanShared::memoryBarrier(
    cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT);

  // Pass 2: temporal reproject + resolve.
  {
    DnsrTemporalPush push;
    push.cameraPos[0] = this->dnsrCameraPos[0];
    push.cameraPos[1] = this->dnsrCameraPos[1];
    push.cameraPos[2] = this->dnsrCameraPos[2];
    push.cameraPos[3] = 1.0f;
    push.screen[0] = w;
    push.screen[1] = h;
    push.maxSamples = static_cast<float>(this->ptMaxSamples);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      this->dnsrTemporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->dnsrTemporalPipelineLayout, 0, 1,
                            &this->dnsrTemporalDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, this->dnsrTemporalPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
  }

  // Copy the current G-buffers into the history position/normal buffers for
  // the next frame's reprojection.
  {
    const VkDeviceSize imageBytes =
      static_cast<VkDeviceSize>(w) * h * 16;
    VkBufferCopy cp {0, 0, imageBytes};
    vkCmdCopyBuffer(cmd, this->positionBuffer, this->dnsrHistPos, 1, &cp);
    vkCmdCopyBuffer(cmd, this->normalBuffer, this->dnsrHistNrm, 1, &cp);
  }

  // Denoised output -> present fragment shader.
  SoVulkanShared::memoryBarrier(
    cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    this->emitError("DNSR denoiser: failed to end command buffer");
    return false;
  }

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  if (vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
    this->emitError("DNSR denoiser: vkQueueSubmit failed");
    return false;
  }
  if (!this->drainQueue("DNSR denoiser")) return false;

  // The slot just written becomes the next frame's history source.
  this->dnsrHistIndex = write;
  if (SoVulkanConfig::get().rtxDebug.denoiseTiming) {
    fprintf(stderr, "[DENOISE] DNSR (prefilter+temporal) dispatched (%ux%u)\n",
            w, h);
  }
  return true;
}

#else // !COIN_BUILD_DNSR_DENOISER

// The declarations are unconditional in the header, so provide inert
// definitions when the backend is not built.  The denoise dispatch never
// reaches these (the "dnsr" selection degrades to OIDN at resolve time), but
// the symbols must exist for a non-DNSR build.
bool
SoRTXRenderBackend::createDnsrPipeline()
{
  return false;
}

void
SoRTXRenderBackend::destroyDnsrResources()
{
}

bool
SoRTXRenderBackend::dispatchDnsrDenoise(uint32_t, uint32_t)
{
  return false;
}

#endif // COIN_BUILD_DNSR_DENOISER
