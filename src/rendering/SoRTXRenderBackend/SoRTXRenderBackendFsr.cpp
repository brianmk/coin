// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendFsr.cpp

// AMD FidelityFX DNSR denoiser backend - the "fsr" path-tracing denoiser
// slot.  The MIT shader sources are vendored under
// src/rendering/third_party/fidelityfx/dnsr/ and ported to Vulkan GLSL compute
// in data/shaders/vulkan/rt/ (FsrCommon.glsl, FsrPrefilter.glsl,
// FsrResolveTemporal.glsl; see FsrCommon.glsl for the provenance note).
//
// Two device-local GPU passes over the path tracer's G-buffers (no host
// staging):
//   1. FsrPrefilter       - the DNSR 15-tap edge-stopping spatial filter.
//   2. FsrResolveTemporal - motion-vector reprojection + disocclusion
//                           rejection + AABB-clipped accumulation against the
//                           previous frame's history.
// The body is compiled out when COIN_BUILD_FSR_DENOISER=0, in which case the
// declarations in SoRTXRenderBackend.h are still present (mirroring the
// OIDN/RTX pattern) so the dispatch in SoRTXRenderBackendDenoise.cpp stays
// type-correct and simply degrades to OIDN at runtime.

#include "rendering/SoRTXRenderBackend.h"
#include "rendering/SoVulkanConfig.h"
#include <Inventor/errors/SoDebugError.h>
#include <cstdio>
#include "rendering/vulkan/rt/FsrPrefilter.spv.h"
#include "rendering/vulkan/rt/FsrResolveTemporal.spv.h"
#include <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>

#include "vk_mem_alloc.h"

using namespace SoRTXBackend;

#if COIN_BUILD_FSR_DENOISER

namespace {

// One storage-buffer write descriptor.
VkWriteDescriptorSet fsrWrite(VkDescriptorSet set, uint32_t binding,
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
SoRTXRenderBackend::createFsrPipeline()
{
  if (this->fsrPipelineReady) {
    return true;
  }
  if (this->device == VK_NULL_HANDLE || this->descriptorPool == VK_NULL_HANDLE) {
    return false;
  }
  if (this->accumBuffer == VK_NULL_HANDLE || this->sumSqBuffer == VK_NULL_HANDLE ||
      this->normalBuffer == VK_NULL_HANDLE || this->positionBuffer == VK_NULL_HANDLE ||
      this->motionBuffer == VK_NULL_HANDLE || this->denoisedBuffer == VK_NULL_HANDLE) {
    this->emitError("FSR denoiser: G-buffers not ready");
    return false;
  }

  const VkDeviceSize imageBytes =
    static_cast<VkDeviceSize>(this->ptBufferWidth) * this->ptBufferHeight * 16;
  const VkBufferUsageFlags bufUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  checkFsrLayout();
  checkFsrTemporalLayout();

  // --- Intermediate + history buffers -------------------------------------
  auto makeBuffer = [&](VkBuffer & buf, VmaAllocation & mem, const char * what) {
    if (!this->createDeviceLocalBuffer(imageBytes, bufUsage, buf, mem)) {
      this->emitError(what);
      return false;
    }
    return true;
  };
  if (!makeBuffer(this->fsrPrefilterRad, this->fsrPrefilterRadMem,
                  "FSR denoiser: failed to create prefilter radiance buffer") ||
      !makeBuffer(this->fsrPrefilterVar, this->fsrPrefilterVarMem,
                  "FSR denoiser: failed to create prefilter variance buffer") ||
      !makeBuffer(this->fsrHistRad[0], this->fsrHistRadMem[0],
                  "FSR denoiser: failed to create history radiance buffer") ||
      !makeBuffer(this->fsrHistRad[1], this->fsrHistRadMem[1],
                  "FSR denoiser: failed to create history radiance buffer") ||
      !makeBuffer(this->fsrHistVar[0], this->fsrHistVarMem[0],
                  "FSR denoiser: failed to create history variance buffer") ||
      !makeBuffer(this->fsrHistVar[1], this->fsrHistVarMem[1],
                  "FSR denoiser: failed to create history variance buffer") ||
      !makeBuffer(this->fsrHistPos, this->fsrHistPosMem,
                  "FSR denoiser: failed to create history position buffer") ||
      !makeBuffer(this->fsrHistNrm, this->fsrHistNrmMem,
                  "FSR denoiser: failed to create history normal buffer")) {
    return false;
  }
  this->fsrHistIndex = 0;

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
                                    &this->fsrSetLayout) != VK_SUCCESS) {
      this->fsrSetLayout = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create prefilter set layout");
      return false;
    }
    VkPushConstantRange push {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(FsrPush);
    VkPipelineLayoutCreateInfo layoutCI {};
    layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts = &this->fsrSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                               &this->fsrPipelineLayout) != VK_SUCCESS) {
      this->fsrPipelineLayout = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create prefilter pipeline layout");
      return false;
    }
    VkShaderModuleCreateInfo smCI {};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = coin_vulkan_rt_fsrprefilter_spirv_count * sizeof(uint32_t);
    smCI.pCode = coin_vulkan_rt_fsrprefilter_spirv;
    if (vkCreateShaderModule(this->device, &smCI, this->allocator,
                             &this->fsrModule) != VK_SUCCESS) {
      this->fsrModule = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create prefilter module");
      return false;
    }
    VkComputePipelineCreateInfo computeCI {};
    computeCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeCI.stage.module = this->fsrModule;
    computeCI.stage.pName = "main";
    computeCI.layout = this->fsrPipelineLayout;
    if (vkCreateComputePipelines(this->device, VK_NULL_HANDLE, 1, &computeCI,
                                 this->allocator, &this->fsrPipeline) !=
        VK_SUCCESS) {
      this->fsrPipeline = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create prefilter pipeline");
      return false;
    }
    VkDescriptorSetAllocateInfo allocCI {};
    allocCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocCI.descriptorPool = this->descriptorPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts = &this->fsrSetLayout;
    if (vkAllocateDescriptorSets(this->device, &allocCI,
                                 &this->fsrDescriptorSet) != VK_SUCCESS) {
      this->fsrDescriptorSet = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to allocate prefilter set");
      return false;
    }
    VkDescriptorBufferInfo infos[6] = {};
    infos[0].buffer = this->accumBuffer;
    infos[1].buffer = this->sumSqBuffer;
    infos[2].buffer = this->normalBuffer;
    infos[3].buffer = this->positionBuffer;
    infos[4].buffer = this->fsrPrefilterRad;
    infos[5].buffer = this->fsrPrefilterVar;
    VkWriteDescriptorSet writes[6] {};
    for (uint32_t b = 0; b < 6; ++b) {
      infos[b].range = VK_WHOLE_SIZE;
      writes[b] = fsrWrite(this->fsrDescriptorSet, b, &infos[b]);
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
                                    &this->fsrTemporalSetLayout) != VK_SUCCESS) {
      this->fsrTemporalSetLayout = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create temporal set layout");
      return false;
    }
    VkPushConstantRange push {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(FsrTemporalPush);
    VkPipelineLayoutCreateInfo layoutCI {};
    layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts = &this->fsrTemporalSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                               &this->fsrTemporalPipelineLayout) != VK_SUCCESS) {
      this->fsrTemporalPipelineLayout = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create temporal pipeline layout");
      return false;
    }
    VkShaderModuleCreateInfo smCI {};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize =
      coin_vulkan_rt_fsrresolvetemporal_spirv_count * sizeof(uint32_t);
    smCI.pCode = coin_vulkan_rt_fsrresolvetemporal_spirv;
    if (vkCreateShaderModule(this->device, &smCI, this->allocator,
                             &this->fsrTemporalModule) != VK_SUCCESS) {
      this->fsrTemporalModule = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create temporal module");
      return false;
    }
    VkComputePipelineCreateInfo computeCI {};
    computeCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeCI.stage.module = this->fsrTemporalModule;
    computeCI.stage.pName = "main";
    computeCI.layout = this->fsrTemporalPipelineLayout;
    if (vkCreateComputePipelines(this->device, VK_NULL_HANDLE, 1, &computeCI,
                                 this->allocator, &this->fsrTemporalPipeline) !=
        VK_SUCCESS) {
      this->fsrTemporalPipeline = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to create temporal pipeline");
      return false;
    }
    VkDescriptorSetAllocateInfo allocCI {};
    allocCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocCI.descriptorPool = this->descriptorPool;
    allocCI.descriptorSetCount = 1;
    allocCI.pSetLayouts = &this->fsrTemporalSetLayout;
    if (vkAllocateDescriptorSets(this->device, &allocCI,
                                 &this->fsrTemporalDescriptorSet) != VK_SUCCESS) {
      this->fsrTemporalDescriptorSet = VK_NULL_HANDLE;
      this->emitError("FSR denoiser: failed to allocate temporal set");
      return false;
    }
  }

  // Seed the history so the first temporal pass reprojects against something
  // valid instead of uninitialized device memory.  Position/normal can be
  // copied from the current G-buffers, but the accumulated history
  // (radiance/variance/sample-count) has no valid prior value: the resolve
  // blends at accumSpeed = 1/max(historyNum, 1), so a garbage sample count
  // would make the very first FSR frame blend almost entirely from garbage.
  // Zero-fill all three so the first frame sees historyNum = 0 and the current
  // prefiltered frame wins outright.
  {
    VkCommandBuffer cmd = this->beginTransientCommandBuffer();
    if (cmd != VK_NULL_HANDLE) {
      for (int i = 0; i < 2; ++i) {
        vkCmdFillBuffer(cmd, this->fsrHistRad[i], 0, imageBytes, 0);
        vkCmdFillBuffer(cmd, this->fsrHistVar[i], 0, imageBytes, 0);
      }
      VkBufferCopy cp {0, 0, imageBytes};
      vkCmdCopyBuffer(cmd, this->positionBuffer, this->fsrHistPos, 1, &cp);
      vkCmdCopyBuffer(cmd, this->normalBuffer, this->fsrHistNrm, 1, &cp);
      SoVulkanShared::memoryBarrier(
        cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT);
      if (vkEndCommandBuffer(cmd) == VK_SUCCESS) {
        VkSubmitInfo si {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(this->queue);
      }
    }
  }

  this->fsrPipelineReady = true;
  if (SoVulkanConfig::get().rtxDebug.denoiserDebug) {
    fprintf(stderr, "[DENOISE] FSR (DNSR prefilter+temporal) pipeline ready\n");
  }
  return true;
}

void
SoRTXRenderBackend::destroyFsrResources()
{
  this->fsrPipelineReady = false;
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
  freeBuf(this->fsrPrefilterRad, this->fsrPrefilterRadMem);
  freeBuf(this->fsrPrefilterVar, this->fsrPrefilterVarMem);
  for (int i = 0; i < 2; ++i) {
    freeBuf(this->fsrHistRad[i], this->fsrHistRadMem[i]);
    freeBuf(this->fsrHistVar[i], this->fsrHistVarMem[i]);
  }
  freeBuf(this->fsrHistPos, this->fsrHistPosMem);
  freeBuf(this->fsrHistNrm, this->fsrHistNrmMem);

  if (this->fsrTemporalDescriptorSet != VK_NULL_HANDLE &&
      this->descriptorPool != VK_NULL_HANDLE) {
    vkFreeDescriptorSets(this->device, this->descriptorPool, 1,
                         &this->fsrTemporalDescriptorSet);
  }
  this->fsrTemporalDescriptorSet = VK_NULL_HANDLE;
  if (this->fsrDescriptorSet != VK_NULL_HANDLE &&
      this->descriptorPool != VK_NULL_HANDLE) {
    vkFreeDescriptorSets(this->device, this->descriptorPool, 1,
                         &this->fsrDescriptorSet);
  }
  this->fsrDescriptorSet = VK_NULL_HANDLE;
  if (this->fsrPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->fsrPipeline, this->allocator);
    this->fsrPipeline = VK_NULL_HANDLE;
  }
  if (this->fsrTemporalPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->fsrTemporalPipeline, this->allocator);
    this->fsrTemporalPipeline = VK_NULL_HANDLE;
  }
  if (this->fsrPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->fsrPipelineLayout, this->allocator);
    this->fsrPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->fsrTemporalPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->fsrTemporalPipelineLayout,
                            this->allocator);
    this->fsrTemporalPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->fsrModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->fsrModule, this->allocator);
    this->fsrModule = VK_NULL_HANDLE;
  }
  if (this->fsrTemporalModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->fsrTemporalModule, this->allocator);
    this->fsrTemporalModule = VK_NULL_HANDLE;
  }
  if (this->fsrSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->fsrSetLayout, this->allocator);
    this->fsrSetLayout = VK_NULL_HANDLE;
  }
  if (this->fsrTemporalSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->fsrTemporalSetLayout,
                                 this->allocator);
    this->fsrTemporalSetLayout = VK_NULL_HANDLE;
  }
  this->fsrHistIndex = 0;
}

bool
SoRTXRenderBackend::dispatchFsrDenoise(uint32_t w, uint32_t h)
{
  if (!this->fsrPipelineReady || this->fsrDescriptorSet == VK_NULL_HANDLE ||
      this->fsrTemporalDescriptorSet == VK_NULL_HANDLE) {
    return false;
  }
  if (w == 0 || h == 0) {
    return false;
  }

  const int read = this->fsrHistIndex;
  const int write = 1 - read;

  // Bind the temporal set for this frame's history ping-pong slots.
  {
    VkDescriptorBufferInfo infos[12] = {};
    infos[0].buffer = this->fsrPrefilterRad;
    infos[1].buffer = this->fsrPrefilterVar;
    infos[2].buffer = this->positionBuffer;
    infos[3].buffer = this->normalBuffer;
    infos[4].buffer = this->motionBuffer;
    infos[5].buffer = this->fsrHistRad[read];
    infos[6].buffer = this->fsrHistVar[read];
    infos[7].buffer = this->fsrHistPos;
    infos[8].buffer = this->fsrHistNrm;
    infos[9].buffer = this->denoisedBuffer;
    infos[10].buffer = this->fsrHistRad[write];
    infos[11].buffer = this->fsrHistVar[write];
    VkWriteDescriptorSet writes[12] {};
    for (uint32_t b = 0; b < 12; ++b) {
      infos[b].range = VK_WHOLE_SIZE;
      writes[b] = fsrWrite(this->fsrTemporalDescriptorSet, b, &infos[b]);
    }
    vkUpdateDescriptorSets(this->device, 12, writes, 0, nullptr);
  }

  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    this->emitError("FSR denoiser: failed to begin command buffer");
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
    FsrPush push;
    push.screen[0] = w;
    push.screen[1] = h;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->fsrPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->fsrPipelineLayout, 0, 1,
                            &this->fsrDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, this->fsrPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
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
    FsrTemporalPush push;
    push.cameraPos[0] = this->fsrCameraPos[0];
    push.cameraPos[1] = this->fsrCameraPos[1];
    push.cameraPos[2] = this->fsrCameraPos[2];
    push.cameraPos[3] = 1.0f;
    push.screen[0] = w;
    push.screen[1] = h;
    push.maxSamples = static_cast<float>(this->ptMaxSamples);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      this->fsrTemporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->fsrTemporalPipelineLayout, 0, 1,
                            &this->fsrTemporalDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, this->fsrTemporalPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
  }

  // Copy the current G-buffers into the history position/normal buffers for
  // the next frame's reprojection.
  {
    const VkDeviceSize imageBytes =
      static_cast<VkDeviceSize>(w) * h * 16;
    VkBufferCopy cp {0, 0, imageBytes};
    vkCmdCopyBuffer(cmd, this->positionBuffer, this->fsrHistPos, 1, &cp);
    vkCmdCopyBuffer(cmd, this->normalBuffer, this->fsrHistNrm, 1, &cp);
  }

  // Denoised output -> present fragment shader.
  SoVulkanShared::memoryBarrier(
    cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    this->emitError("FSR denoiser: failed to end command buffer");
    return false;
  }

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  if (vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
    this->emitError("FSR denoiser: vkQueueSubmit failed");
    return false;
  }
  if (vkQueueWaitIdle(this->queue) != VK_SUCCESS) {
    this->emitError("FSR denoiser: vkQueueWaitIdle failed");
    return false;
  }

  // The slot just written becomes the next frame's history source.
  this->fsrHistIndex = write;
  if (SoVulkanConfig::get().rtxDebug.denoiseTiming) {
    fprintf(stderr, "[DENOISE] FSR (DNSR prefilter+temporal) dispatched (%ux%u)\n",
            w, h);
  }
  return true;
}

#else // !COIN_BUILD_FSR_DENOISER

// The declarations are unconditional in the header, so provide inert
// definitions when the backend is not built.  The denoise dispatch never
// reaches these (the "fsr" selection degrades to OIDN at resolve time), but
// the symbols must exist for a non-FSR build.
bool
SoRTXRenderBackend::createFsrPipeline()
{
  return false;
}

void
SoRTXRenderBackend::destroyFsrResources()
{
}

bool
SoRTXRenderBackend::dispatchFsrDenoise(uint32_t, uint32_t)
{
  return false;
}

#endif // COIN_BUILD_FSR_DENOISER
