// src/rendering/vulkan/raytracing/rtx/SoRTXRenderBackendPick.cpp

// GPU picking for the Vulkan/RTX viewport: a single-ray ray-query against the
// frame's TLAS, used to replace the O(triangles) CPU SoRayPickAction on hover
// and selection.  Vulkan/RTX only -- the GL renderer and the raster Vulkan
// backend keep the CPU path and never touch this file.

#include "rendering/vulkan/raytracing/rtx/SoRTXRenderBackend.h"
#include <Inventor/errors/SoDebugError.h>
#include <cstdint>
#include <cstring>

#include "vk_mem_alloc.h"

#include "rendering/vulkan/generated/shaders/rt/Pick.spv.h"

namespace {

// std430 mirror of PickHit in Pick.glsl (two vec4 + one uvec4 = 48 bytes).
struct PickHitGpu {
  float data0[4];     // x = t, y = hit flag
  float worldPos[4];  // xyz = world-space hit position
  uint32_t ids[4];    // x = instance/command index, y = primitive id
};

// Push constant block of Pick.glsl: origin(vec4) + direction(vec4) + tMax,
// padded to the block alignment (16).
struct PickPush {
  float origin[4];
  float direction[4];
  float tMax;
  float pad[3];
};

static_assert(sizeof(PickHitGpu) == 48, "Pick.glsl hit record layout drifted");
static_assert(sizeof(PickPush) == 48, "Pick.glsl push block layout drifted");

}  // namespace

bool
SoRTXRenderBackend::createPickResources()
{
  if (this->pickResourcesReady) {
    return true;
  }
  if (this->device == VK_NULL_HANDLE) {
    return false;
  }

  VkShaderModuleCreateInfo sci {};
  sci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  sci.codeSize = coin_vulkan_rt_pick_spirv_count * sizeof(uint32_t);
  sci.pCode = coin_vulkan_rt_pick_spirv;
  const VkResult smRes = vkCreateShaderModule(
    this->device, &sci, this->allocator, &this->pickModule);
  if (smRes != VK_SUCCESS) {
    this->emitError(("createPickResources: vkCreateShaderModule failed: "
                     + SoVulkanShared::vkResultName(smRes)).c_str());
    this->destroyPickResources();
    return false;
  }

  // Set 0: binding 0 = TLAS, binding 1 = host-readable hit record.
  VkDescriptorSetLayoutBinding bindings[2] {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo lci {};
  lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  lci.bindingCount = 2;
  lci.pBindings = bindings;
  VkDescriptorBindingFlags pickFlags[2] = {};
  VkDescriptorSetLayoutBindingFlagsCreateInfo pickFlagsCI {};
  if (this->hasUpdateAfterBind) {
    // Binding 0 is the TLAS: acceleration-structure update-after-bind is a
    // separate feature (VkPhysicalDeviceAccelerationStructureFeaturesKHR) that
    // is not requested, so only the result-buffer binding is flagged
    // (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingAccelerationStructureUpdateAfterBind-03570).
    pickFlags[0] = 0;
    pickFlags[1] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    pickFlagsCI.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    pickFlagsCI.bindingCount = 2;
    pickFlagsCI.pBindingFlags = pickFlags;
    lci.pNext = &pickFlagsCI;
    lci.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  }
  if (vkCreateDescriptorSetLayout(this->device, &lci, this->allocator,
                                  &this->pickSetLayout) != VK_SUCCESS) {
    this->destroyPickResources();
    return false;
  }

  VkPushConstantRange pushRange {};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(PickPush);

  VkPipelineLayoutCreateInfo plci {};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &this->pickSetLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pushRange;
  if (vkCreatePipelineLayout(this->device, &plci, this->allocator,
                             &this->pickPipelineLayout) != VK_SUCCESS) {
    this->destroyPickResources();
    return false;
  }

  VkComputePipelineCreateInfo cpci {};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = this->pickModule;
  cpci.stage.pName = "main";
  cpci.layout = this->pickPipelineLayout;
  const VkResult createRes = vkCreateComputePipelines(
    this->device, VK_NULL_HANDLE, 1, &cpci, this->allocator,
    &this->pickPipeline);
  if (createRes != VK_SUCCESS) {
    this->emitError(("createPickResources: vkCreateComputePipelines failed: "
                     + SoVulkanShared::vkResultName(createRes)).c_str());
    this->destroyPickResources();
    return false;
  }

  if (!this->createHostVisibleBuffer(
        sizeof(PickHitGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        this->pickResultBuffer, this->pickResultMemory,
        &this->pickResultMapped)) {
    this->destroyPickResources();
    return false;
  }

  VkDescriptorSetAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = this->descriptorPool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &this->pickSetLayout;
  if (vkAllocateDescriptorSets(this->device, &ai,
                               &this->pickDescriptorSet) != VK_SUCCESS) {
    this->pickDescriptorSet = VK_NULL_HANDLE;
    this->destroyPickResources();
    return false;
  }

  VkCommandPoolCreateInfo pci {};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = this->queueFamilyIndex;
  if (vkCreateCommandPool(this->device, &pci, this->allocator,
                          &this->pickCommandPool) != VK_SUCCESS) {
    this->destroyPickResources();
    return false;
  }
  VkCommandBufferAllocateInfo cbai {};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = this->pickCommandPool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(this->device, &cbai,
                               &this->pickCommandBuffer) != VK_SUCCESS) {
    this->pickCommandBuffer = VK_NULL_HANDLE;
    this->destroyPickResources();
    return false;
  }

  VkFenceCreateInfo fci {};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  // Signaled so the first vkResetFences() before submit is valid.
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  if (vkCreateFence(this->device, &fci, this->allocator,
                    &this->pickFence) != VK_SUCCESS) {
    this->destroyPickResources();
    return false;
  }

  this->pickResourcesReady = true;
  return true;
}

void
SoRTXRenderBackend::destroyPickResources()
{
  if (this->device != VK_NULL_HANDLE) {
    // The persistent mapping comes from VMA_ALLOCATION_CREATE_MAPPED_BIT, so
    // there is no vmaMapMemory to balance before vmaDestroyBuffer.
    this->pickResultMapped = nullptr;
    if (this->pickResultBuffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(this->vmaAllocator, this->pickResultBuffer, this->pickResultMemory);
      this->pickResultBuffer = VK_NULL_HANDLE;
    }
    if (this->pickResultMemory != VK_NULL_HANDLE) {
      this->pickResultMemory = VK_NULL_HANDLE;
    }
    if (this->pickFence != VK_NULL_HANDLE) {
      vkDestroyFence(this->device, this->pickFence, this->allocator);
      this->pickFence = VK_NULL_HANDLE;
    }
    if (this->pickCommandBuffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(this->device, this->pickCommandPool, 1,
                           &this->pickCommandBuffer);
      this->pickCommandBuffer = VK_NULL_HANDLE;
    }
    if (this->pickCommandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(this->device, this->pickCommandPool,
                           this->allocator);
      this->pickCommandPool = VK_NULL_HANDLE;
    }
    // The descriptor set is owned by the pool; just drop the handle.
    this->pickDescriptorSet = VK_NULL_HANDLE;
    if (this->pickPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, this->pickPipeline, this->allocator);
      this->pickPipeline = VK_NULL_HANDLE;
    }
    if (this->pickPipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(this->device, this->pickPipelineLayout,
                              this->allocator);
      this->pickPipelineLayout = VK_NULL_HANDLE;
    }
    if (this->pickSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(this->device, this->pickSetLayout,
                                   this->allocator);
      this->pickSetLayout = VK_NULL_HANDLE;
    }
    if (this->pickModule != VK_NULL_HANDLE) {
      vkDestroyShaderModule(this->device, this->pickModule, this->allocator);
      this->pickModule = VK_NULL_HANDLE;
    }
  }
  this->pickResourcesReady = false;
}

bool
SoRTXRenderBackend::pickRay(const float origin[3], const float direction[3],
                            float tMax, RTPickHit & out)
{
  out = RTPickHit {};
  if (!this->isInitialized() || this->device == VK_NULL_HANDLE ||
      this->queue == VK_NULL_HANDLE) {
    return false;
  }
  // No TLAS yet means no frame has rendered; the caller keeps CPU picking.
  if (this->tlas == VK_NULL_HANDLE) {
    return false;
  }
  if (!this->createPickResources()) {
    return false;
  }

  // Point the descriptor at the current TLAS and result buffer.  The TLAS
  // handle can change when buildTlas() reallocates it, so this is rewritten on
  // every call (two writes on a 48-byte buffer is negligible).
  VkWriteDescriptorSetAccelerationStructureKHR asWrite {};
  asWrite.sType =
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  asWrite.accelerationStructureCount = 1;
  VkAccelerationStructureKHR asHandle = this->tlas;
  asWrite.pAccelerationStructures = &asHandle;

  VkDescriptorBufferInfo bufInfo {};
  bufInfo.buffer = this->pickResultBuffer;
  bufInfo.offset = 0;
  bufInfo.range = sizeof(PickHitGpu);

  VkWriteDescriptorSet writes[2] {};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = this->pickDescriptorSet;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  writes[0].pNext = &asWrite;
  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = this->pickDescriptorSet;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &bufInfo;
  vkUpdateDescriptorSets(this->device, 2, writes, 0, nullptr);

  // A trace that never runs (early-out below) must read as "no hit".
  std::memset(this->pickResultMapped, 0, sizeof(PickHitGpu));

  // Both resets must succeed.  If vkResetFences fails while the fence is still
  // signaled from the previous pick, the wait below passes immediately and the
  // stale result is returned, so fail loudly here instead.
  const VkResult resetFenceRes =
    vkResetFences(this->device, 1, &this->pickFence);
  if (resetFenceRes != VK_SUCCESS) {
    this->emitError(
      ("pickRay: vkResetFences failed: "
       + SoVulkanShared::vkResultName(resetFenceRes)).c_str());
    return false;
  }
  const VkResult resetCmdRes = vkResetCommandBuffer(this->pickCommandBuffer, 0);
  if (resetCmdRes != VK_SUCCESS) {
    this->emitError(
      ("pickRay: vkResetCommandBuffer failed: "
       + SoVulkanShared::vkResultName(resetCmdRes)).c_str());
    return false;
  }

  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  const VkResult beginRes = vkBeginCommandBuffer(this->pickCommandBuffer, &bi);
  if (beginRes != VK_SUCCESS) {
    this->emitError(("pickRay: vkBeginCommandBuffer failed: "
                     + SoVulkanShared::vkResultName(beginRes)).c_str());
    return false;
  }

  vkCmdBindPipeline(this->pickCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    this->pickPipeline);
  vkCmdBindDescriptorSets(this->pickCommandBuffer,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          this->pickPipelineLayout, 0, 1,
                          &this->pickDescriptorSet, 0, nullptr);

  PickPush push {};
  push.origin[0] = origin[0];
  push.origin[1] = origin[1];
  push.origin[2] = origin[2];
  push.direction[0] = direction[0];
  push.direction[1] = direction[1];
  push.direction[2] = direction[2];
  push.tMax = tMax;
  vkCmdPushConstants(this->pickCommandBuffer, this->pickPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PickPush), &push);
  vkCmdDispatch(this->pickCommandBuffer, 1, 1, 1);

  // Make the shader's write visible to the host read after the fence wait.
  SoVulkanShared::memoryBarrier(this->pickCommandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_HOST_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_HOST_READ_BIT);

  const VkResult endRes = vkEndCommandBuffer(this->pickCommandBuffer);
  if (endRes != VK_SUCCESS) {
    this->emitError(("pickRay: vkEndCommandBuffer failed: "
                     + SoVulkanShared::vkResultName(endRes)).c_str());
    return false;
  }

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &this->pickCommandBuffer;
  const VkResult submitRes = vkQueueSubmit(this->queue, 1, &si, this->pickFence);
  if (submitRes != VK_SUCCESS) {
    this->emitError(("pickRay: vkQueueSubmit failed: "
                     + SoVulkanShared::vkResultName(submitRes)).c_str());
    return false;
  }
  const VkResult waitRes = vkWaitForFences(this->device, 1, &this->pickFence,
                                           VK_TRUE, UINT64_MAX);
  if (waitRes != VK_SUCCESS) {
    this->emitError(("pickRay: vkWaitForFences failed: "
                     + SoVulkanShared::vkResultName(waitRes)).c_str());
    return false;
  }

  const PickHitGpu * gpu =
    static_cast<const PickHitGpu *>(this->pickResultMapped);
  out.hit = gpu->data0[1] > 0.5f;
  out.t = gpu->data0[0];
  out.worldPos[0] = gpu->worldPos[0];
  out.worldPos[1] = gpu->worldPos[1];
  out.worldPos[2] = gpu->worldPos[2];
  out.commandIndex = gpu->ids[0];
  out.primitiveId = gpu->ids[1];
  // Resolve the command index to the producer identity captured when the TLAS
  // was built (the per-frame draw list is already gone by now).
  if (out.hit && out.commandIndex < this->pickCommandInfo.size()) {
    const RTPickCommandInfo & info =
      this->pickCommandInfo[static_cast<size_t>(out.commandIndex)];
    out.userData = info.userData;
    out.primitiveOffset = info.primitiveOffset;
  }
  return true;
}
