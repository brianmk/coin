// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendPipeline.cpp

// Split from the original monolithic SoRTXRenderBackend.cpp.  Contains the
// member functions for the "Pipeline" concern of the Vulkan RTX backend.

#include "rendering/SoRTXRenderBackend.h"
#include <Inventor/errors/SoDebugError.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include "rendering/vulkan/rt/PathTrace.spv.h"
#include "rendering/vulkan/rt/Raygen.spv.h"
#include "rendering/vulkan/rt/Miss.spv.h"
#include "rendering/vulkan/rt/ShadowMiss.spv.h"
#include "rendering/vulkan/rt/ClosestHit.spv.h"
#include "rendering/vulkan/rt/ShadowClosestHit.spv.h"
#include "rendering/vulkan/rt/PresentVertex.spv.h"
#include "rendering/vulkan/rt/PresentFragment.spv.h"
#include <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>

using namespace SoRTXBackend;

bool
SoRTXRenderBackend::createDescriptorSetLayout()
{
  // Ray tracing descriptor set: bindings 0-7 (see Raygen.glsl and
  // ClosestHit.glsl).  Stage flags mirror the consumers: the raygen traces
  // rays and writes the image/accum/G-buffers, the miss shader samples the
  // frame UBO, and the closest-hit shader reads materials, the frame UBO
  // and the triangle-normal pool.
  VkDescriptorSetLayoutBinding bindings[15] {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[3].binding = 3;
  bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT;

  // Path tracing: accumulation buffer, first-bounce G-buffers (written by
  // the raygen) and the triangle-normal pool (read by the closest hit).
  for (uint32_t b = 4; b <= 6; ++b) {
    bindings[b].binding = b;
    bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[b].descriptorCount = 1;
    bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
      VK_SHADER_STAGE_COMPUTE_BIT;
  }
  bindings[7].binding = 7;
  bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[7].descriptorCount = 1;
  bindings[7].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  // Adaptive sampling: per-pixel sums-of-squares and the active-pixel
  // counter (compute-tracer only).
  bindings[8].binding = 8;
  bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[8].descriptorCount = 1;
  bindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[9].binding = 9;
  bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[9].descriptorCount = 1;
  bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // Temporal reprojection history: accumulation, sums-of-squares and world
  // positions of the previous traced frame (compute-tracer only).
  for (uint32_t b = 10; b <= 12; ++b) {
    bindings[b].binding = b;
    bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[b].descriptorCount = 1;
    bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  // Emissive-triangle pool for NEE (compute-tracer only).
  bindings[13].binding = 13;
  bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[13].descriptorCount = 1;
  bindings[13].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // Albedo G-buffer (binding 14): written by the raygen and fed to the
  // denoiser as a guide.  Only bound while a denoiser backend is active.
  bindings[14].binding = 14;
  bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[14].descriptorCount = 1;
  bindings[14].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 15;
  ci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(this->device, &ci, this->allocator,
                                  &this->rtSetLayout) != VK_SUCCESS) {
    return false;
  }

  // Present descriptor set: combined image sampler at binding 1 (the raw
  // traced image for the preview mode) plus the accumulation and G-buffer
  // storage buffers at bindings 2-4 (the denoising path tracing path) and
  // the denoiser output at binding 5 (sampled when a denoiser backend has
  // produced a result for the current frame).  Binding 6 is the traced
  // camera's view/projection (world->view->clip) so the present pass can
  // write scene depth for the raster composite overlay's edge occlusion.
  VkDescriptorSetLayoutBinding presentBindings[6] {};
  presentBindings[0].binding = 1;
  presentBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  presentBindings[0].descriptorCount = 1;
  presentBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  for (uint32_t b = 2; b <= 5; ++b) {
    presentBindings[b - 1].binding = b;
    presentBindings[b - 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentBindings[b - 1].descriptorCount = 1;
    presentBindings[b - 1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  presentBindings[5].binding = 6;
  presentBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  presentBindings[5].descriptorCount = 1;
  presentBindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo pci {};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  pci.bindingCount = 6;
  pci.pBindings = presentBindings;
  return vkCreateDescriptorSetLayout(this->device, &pci, this->allocator,
                                     &this->presentSetLayout) == VK_SUCCESS;
}

bool
SoRTXRenderBackend::createDescriptorPool()
{
  VkDescriptorPoolSize sizes[5] {};
  sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  sizes[0].descriptorCount = 2;
  sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  sizes[1].descriptorCount = 2;
  sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sizes[2].descriptorCount = 2;
  sizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sizes[3].descriptorCount = 2;
  sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sizes[4].descriptorCount = 48;

  VkDescriptorPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  ci.maxSets = 4;
  ci.poolSizeCount = 5;
  ci.pPoolSizes = sizes;
  return vkCreateDescriptorPool(this->device, &ci, this->allocator,
                                &this->descriptorPool) == VK_SUCCESS;
}

bool
SoRTXRenderBackend::createShaderModules()
{
  auto load = [this](const uint32_t * code, size_t count,
                     VkShaderModule & module) {
    VkShaderModuleCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = count * sizeof(uint32_t);
    ci.pCode = code;
    return vkCreateShaderModule(this->device, &ci, this->allocator,
                                &module) == VK_SUCCESS;
  };
  if (!load(coin_vulkan_rt_pathtrace_spirv,
            coin_vulkan_rt_pathtrace_spirv_count, this->pathTraceModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_raygen_spirv,
            coin_vulkan_rt_raygen_spirv_count, this->raygenModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_miss_spirv,
            coin_vulkan_rt_miss_spirv_count, this->missModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_shadowmiss_spirv,
            coin_vulkan_rt_shadowmiss_spirv_count, this->shadowMissModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_closesthit_spirv,
            coin_vulkan_rt_closesthit_spirv_count, this->closestHitModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_shadowclosesthit_spirv,
            coin_vulkan_rt_shadowclosesthit_spirv_count,
            this->shadowClosestHitModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_presentvertex_spirv,
            coin_vulkan_rt_presentvertex_spirv_count,
            this->presentVertexModule)) {
    return false;
  }
  if (!load(coin_vulkan_rt_presentfragment_spirv,
            coin_vulkan_rt_presentfragment_spirv_count,
            this->presentFragmentModule)) {
    return false;
  }
  return true;
}

bool
SoRTXRenderBackend::createFrameBuffer()
{
  if (this->presentFrameBuffer != VK_NULL_HANDLE) {
    return this->frameBuffer != VK_NULL_HANDLE;
  }
  if (this->frameBuffer == VK_NULL_HANDLE) {
    if (!this->createHostVisibleBuffer(
          sizeof(RTXFrameBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          this->frameBuffer, this->frameMemory)) {
      return false;
    }
    if (vkMapMemory(this->device, this->frameMemory, 0,
                    sizeof(RTXFrameBlock), 0, &this->frameMapped) !=
        VK_SUCCESS) {
      return false;
    }
  }
  // Compact present frame block: world->view (mat4) followed by view->clip
  // (mat4), exactly matching the PresentFrame std140 block in
  // PresentFragment.glsl (two mat4, offsets 0 and 64).
  if (!this->createHostVisibleBuffer(
        2 * sizeof(float) * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        this->presentFrameBuffer, this->presentFrameMemory)) {
    return false;
  }
  return vkMapMemory(this->device, this->presentFrameMemory, 0,
                     2 * sizeof(float) * 16, 0, &this->presentFrameMapped) ==
    VK_SUCCESS;
}

bool
SoRTXRenderBackend::updateDescriptors()
{
  // Allocate the double-buffered pairs once (the layouts differ, so two
  // allocations of two sets each).
  for (int pair = 0; pair < 2; ++pair) {
    if (this->rtDescriptorSets[pair] != VK_NULL_HANDLE) continue;
    VkDescriptorSetLayout layout = this->rtSetLayout;
    VkDescriptorSetAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = this->descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(this->device, &ai,
                                 &this->rtDescriptorSets[pair]) !=
        VK_SUCCESS) {
      return false;
    }
  }
  for (int pair = 0; pair < 2; ++pair) {
    if (this->presentDescriptorSets[pair] != VK_NULL_HANDLE) continue;
    VkDescriptorSetLayout layout = this->presentSetLayout;
    VkDescriptorSetAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = this->descriptorPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(this->device, &ai,
                                 &this->presentDescriptorSets[pair]) !=
        VK_SUCCESS) {
      return false;
    }
  }
  const VkDescriptorSet rtSet = this->rtDescriptorSets[this->descriptorSetIndex];
  const VkDescriptorSet presentSet =
    this->presentDescriptorSets[this->descriptorSetIndex];

  VkDescriptorBufferInfo frameInfo {};
  frameInfo.buffer = this->frameBuffer;
  frameInfo.offset = 0;
  frameInfo.range = sizeof(RTXFrameBlock);

  VkDescriptorImageInfo storageInfo {};
  storageInfo.imageView = this->storageImageView;
  storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkDescriptorImageInfo presentInfo {};
  presentInfo.sampler = this->presentSampler;
  presentInfo.imageView = this->storageImageView;
  // The image stays in GENERAL layout for both the trace (storage) and
  // present (sampled) accesses; no in-render-pass transitions needed.
  presentInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkDescriptorBufferInfo materialInfo {};
  materialInfo.buffer = this->materialBuffer;
  materialInfo.offset = 0;
  materialInfo.range = VK_WHOLE_SIZE;

  // Path tracing buffers: accumulation (set 0, binding 4), world normal
  // G-buffer (binding 5) and world position/hit-distance G-buffer
  // (binding 6).  Written by the raygen shader.
  VkDescriptorBufferInfo accumInfo {};
  accumInfo.buffer = this->accumBuffer;
  accumInfo.offset = 0;
  accumInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo normalInfo {};
  normalInfo.buffer = this->normalBuffer;
  normalInfo.offset = 0;
  normalInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo positionInfo {};
  positionInfo.buffer = this->positionBuffer;
  positionInfo.offset = 0;
  positionInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo normalPoolInfo {};
  normalPoolInfo.buffer = this->normalPoolBuffer;
  normalPoolInfo.offset = 0;
  normalPoolInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo sumSqInfo {};
  sumSqInfo.buffer = this->sumSqBuffer;
  sumSqInfo.offset = 0;
  sumSqInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo counterInfo {};
  counterInfo.buffer = this->activeCounterBuffer;
  counterInfo.offset = 0;
  counterInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo accumHistInfo {};
  accumHistInfo.buffer = this->accumHistoryBuffer;
  accumHistInfo.offset = 0;
  accumHistInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo sumSqHistInfo {};
  sumSqHistInfo.buffer = this->sumSqHistoryBuffer;
  sumSqHistInfo.offset = 0;
  sumSqHistInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo posHistInfo {};
  posHistInfo.buffer = this->positionHistoryBuffer;
  posHistInfo.offset = 0;
  posHistInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo neePoolInfo {};
  // The RT set layout always declares binding 13, but scenes without
  // emissive geometry never allocate the NEE pool.  Bind a valid zero-count
  // placeholder instead of leaving the set entry uninitialized; the NEE
  // shaders only read this buffer when the uniform block reports a non-zero
  // triangle count.
  neePoolInfo.buffer = this->neePoolBuffer != VK_NULL_HANDLE
                         ? this->neePoolBuffer
                         : this->activeCounterBuffer;
  neePoolInfo.offset = 0;
  neePoolInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo albedoInfo {};
  albedoInfo.buffer = this->albedoBuffer;
  albedoInfo.offset = 0;
  albedoInfo.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo denoisedInfo {};
  denoisedInfo.buffer = this->denoisedBuffer;
  denoisedInfo.offset = 0;
  denoisedInfo.range = VK_WHOLE_SIZE;

  // Binding 0: the acceleration structure (TLAS) read by the raygen shader.
  // Only written once the TLAS exists; updateDescriptors() is re-invoked by
  // buildTlas() right after (re)creation so a null handle is never written
  // and the trace phase always observes a valid descriptor.
  VkWriteDescriptorSetAccelerationStructureKHR asWrite {};
  asWrite.sType =
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  asWrite.accelerationStructureCount = 1;
  VkAccelerationStructureKHR asHandle = this->tlas;
  asWrite.pAccelerationStructures = &asHandle;

  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(5);

  if (this->tlas != VK_NULL_HANDLE) {
    VkWriteDescriptorSet asBinding {};
    asBinding.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    asBinding.dstSet = rtSet;
    asBinding.dstBinding = 0;
    asBinding.descriptorCount = 1;
    asBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    asBinding.pNext = &asWrite;
    writes.push_back(asBinding);
  }

  VkWriteDescriptorSet storageWrite {};
  storageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  storageWrite.dstSet = rtSet;
  storageWrite.dstBinding = 1;
  storageWrite.descriptorCount = 1;
  storageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  storageWrite.pImageInfo = &storageInfo;
  if (this->storageImageView != VK_NULL_HANDLE) {
    writes.push_back(storageWrite);
  }

  VkWriteDescriptorSet frameWrite {};
  frameWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  frameWrite.dstSet = rtSet;
  frameWrite.dstBinding = 2;
  frameWrite.descriptorCount = 1;
  frameWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  frameWrite.pBufferInfo = &frameInfo;
  writes.push_back(frameWrite);

  if (this->materialBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet materialWrite {};
    materialWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    materialWrite.dstSet = rtSet;
    materialWrite.dstBinding = 3;
    materialWrite.descriptorCount = 1;
    materialWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    materialWrite.pBufferInfo = &materialInfo;
    writes.push_back(materialWrite);
  }

  if (this->accumBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet accumWrite {};
    accumWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    accumWrite.dstSet = rtSet;
    accumWrite.dstBinding = 4;
    accumWrite.descriptorCount = 1;
    accumWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    accumWrite.pBufferInfo = &accumInfo;
    writes.push_back(accumWrite);
  }
  if (this->normalBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet normalWrite {};
    normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    normalWrite.dstSet = rtSet;
    normalWrite.dstBinding = 5;
    normalWrite.descriptorCount = 1;
    normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    normalWrite.pBufferInfo = &normalInfo;
    writes.push_back(normalWrite);
  }
  if (this->positionBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet positionWrite {};
    positionWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    positionWrite.dstSet = rtSet;
    positionWrite.dstBinding = 6;
    positionWrite.descriptorCount = 1;
    positionWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    positionWrite.pBufferInfo = &positionInfo;
    writes.push_back(positionWrite);
  }
  if (this->normalPoolBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet poolWrite {};
    poolWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    poolWrite.dstSet = rtSet;
    poolWrite.dstBinding = 7;
    poolWrite.descriptorCount = 1;
    poolWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolWrite.pBufferInfo = &normalPoolInfo;
    writes.push_back(poolWrite);
  }
  if (this->sumSqBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet sumSqWrite {};
    sumSqWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    sumSqWrite.dstSet = rtSet;
    sumSqWrite.dstBinding = 8;
    sumSqWrite.descriptorCount = 1;
    sumSqWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sumSqWrite.pBufferInfo = &sumSqInfo;
    writes.push_back(sumSqWrite);
  }
  if (this->activeCounterBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet counterWrite {};
    counterWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    counterWrite.dstSet = rtSet;
    counterWrite.dstBinding = 9;
    counterWrite.descriptorCount = 1;
    counterWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    counterWrite.pBufferInfo = &counterInfo;
    writes.push_back(counterWrite);
  }
  if (this->accumHistoryBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet accumHistWrite {};
    accumHistWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    accumHistWrite.dstSet = rtSet;
    accumHistWrite.dstBinding = 10;
    accumHistWrite.descriptorCount = 1;
    accumHistWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    accumHistWrite.pBufferInfo = &accumHistInfo;
    writes.push_back(accumHistWrite);
  }
  if (this->sumSqHistoryBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet sumSqHistWrite {};
    sumSqHistWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    sumSqHistWrite.dstSet = rtSet;
    sumSqHistWrite.dstBinding = 11;
    sumSqHistWrite.descriptorCount = 1;
    sumSqHistWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sumSqHistWrite.pBufferInfo = &sumSqHistInfo;
    writes.push_back(sumSqHistWrite);
  }
  if (this->positionHistoryBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet posHistWrite {};
    posHistWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    posHistWrite.dstSet = rtSet;
    posHistWrite.dstBinding = 12;
    posHistWrite.descriptorCount = 1;
    posHistWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    posHistWrite.pBufferInfo = &posHistInfo;
    writes.push_back(posHistWrite);
  }
  if (neePoolInfo.buffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet neePoolWrite {};
    neePoolWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    neePoolWrite.dstSet = rtSet;
    neePoolWrite.dstBinding = 13;
    neePoolWrite.descriptorCount = 1;
    neePoolWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    neePoolWrite.pBufferInfo = &neePoolInfo;
    writes.push_back(neePoolWrite);
  }
  if (this->albedoBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet albedoWrite {};
    albedoWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    albedoWrite.dstSet = rtSet;
    albedoWrite.dstBinding = 14;
    albedoWrite.descriptorCount = 1;
    albedoWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    albedoWrite.pBufferInfo = &albedoInfo;
    writes.push_back(albedoWrite);
  }

  VkWriteDescriptorSet presentWrite {};
  presentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  presentWrite.dstSet = presentSet;
  presentWrite.dstBinding = 1;
  presentWrite.descriptorCount = 1;
  presentWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  presentWrite.pImageInfo = &presentInfo;
  if (this->storageImageView != VK_NULL_HANDLE &&
      this->presentSampler != VK_NULL_HANDLE) {
    writes.push_back(presentWrite);
  }

  if (this->accumBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentAccumWrite {};
    presentAccumWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentAccumWrite.dstSet = presentSet;
    presentAccumWrite.dstBinding = 2;
    presentAccumWrite.descriptorCount = 1;
    presentAccumWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentAccumWrite.pBufferInfo = &accumInfo;
    writes.push_back(presentAccumWrite);
  }
  if (this->normalBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentNormalWrite {};
    presentNormalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentNormalWrite.dstSet = presentSet;
    presentNormalWrite.dstBinding = 3;
    presentNormalWrite.descriptorCount = 1;
    presentNormalWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentNormalWrite.pBufferInfo = &normalInfo;
    writes.push_back(presentNormalWrite);
  }
  if (this->positionBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentPositionWrite {};
    presentPositionWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentPositionWrite.dstSet = presentSet;
    presentPositionWrite.dstBinding = 4;
    presentPositionWrite.descriptorCount = 1;
    presentPositionWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentPositionWrite.pBufferInfo = &positionInfo;
    writes.push_back(presentPositionWrite);
  }
  if (this->denoisedBuffer != VK_NULL_HANDLE) {
    VkWriteDescriptorSet presentDenoisedWrite {};
    presentDenoisedWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentDenoisedWrite.dstSet = presentSet;
    presentDenoisedWrite.dstBinding = 5;
    presentDenoisedWrite.descriptorCount = 1;
    presentDenoisedWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    presentDenoisedWrite.pBufferInfo = &denoisedInfo;
    writes.push_back(presentDenoisedWrite);
  }
  if (this->presentFrameBuffer != VK_NULL_HANDLE) {
    VkDescriptorBufferInfo presentFrameInfo {};
    presentFrameInfo.buffer = this->presentFrameBuffer;
    presentFrameInfo.offset = 0;
    presentFrameInfo.range = 2 * sizeof(float) * 16;
    VkWriteDescriptorSet presentFrameWrite {};
    presentFrameWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    presentFrameWrite.dstSet = presentSet;
    presentFrameWrite.dstBinding = 6;
    presentFrameWrite.descriptorCount = 1;
    presentFrameWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    presentFrameWrite.pBufferInfo = &presentFrameInfo;
    writes.push_back(presentFrameWrite);
  }

  vkUpdateDescriptorSets(this->device,
                         static_cast<uint32_t>(writes.size()), writes.data(),
                         0, nullptr);
  return true;
}

bool
SoRTXRenderBackend::createPipelines()
{
  VkPipelineLayoutCreateInfo layoutCI {};
  layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutCI.setLayoutCount = 1;
  layoutCI.pSetLayouts = &this->rtSetLayout;
  // The raygen receives its per-frame state (frame index, PT flags, bounce
  // budget) through a 16-byte push constant block.
  VkPushConstantRange raygenPush {};
  raygenPush.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  raygenPush.offset = 0;
  raygenPush.size = sizeof(RTXRaygenPush);
  layoutCI.pPushConstantRanges = &raygenPush;
  layoutCI.pushConstantRangeCount = 1;
  if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                             &this->rtPipelineLayout) != VK_SUCCESS) {
    return false;
  }

  layoutCI.pSetLayouts = &this->presentSetLayout;
  // The present shader receives width/height/denoiseOn/frameIndex via
  // u_present, the viewport origin via u_origin and the denoiser
  // flag/scale via u_denoise (the present pass must run inside the caller's
  // render pass, so a compute denoise pass cannot be dispatched there; the
  // edge-stopping filter lives in PresentFragment.glsl instead).
  VkPushConstantRange presentPush {};
  presentPush.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  presentPush.offset = 0;
  presentPush.size = 12 * sizeof(float);
  layoutCI.pPushConstantRanges = &presentPush;
  layoutCI.pushConstantRangeCount = 1;
  if (vkCreatePipelineLayout(this->device, &layoutCI, this->allocator,
                             &this->presentPipelineLayout) != VK_SUCCESS) {
    return false;
  }

  // --- Ray tracing pipeline (five SBT groups) ----------------------------
  // Group layout: 0 = raygen, 1 = miss, 2 = shadow miss, 3 = closest hit,
  // 4 = shadow closest hit.  Primary rays use missIndex 0 and hit-group
  // record 0; shadow rays use missIndex 1 and hit-group record 1.
  VkPipelineShaderStageCreateInfo stages[SBT_GROUP_COUNT] {};
  const auto stage = [](VkShaderStageFlagBits flag, VkShaderModule module,
                        VkPipelineShaderStageCreateInfo & out) {
    out.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    out.stage = flag;
    out.module = module;
    out.pName = "main";
  };
  stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, this->raygenModule, stages[0]);
  stage(VK_SHADER_STAGE_MISS_BIT_KHR, this->missModule, stages[1]);
  stage(VK_SHADER_STAGE_MISS_BIT_KHR, this->shadowMissModule, stages[2]);
  stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, this->closestHitModule,
        stages[3]);
  stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, this->shadowClosestHitModule,
        stages[4]);

  VkRayTracingShaderGroupCreateInfoKHR groups[SBT_GROUP_COUNT] {};
  for (int i = 0; i < SBT_GROUP_COUNT; ++i) {
    groups[i].sType =
      VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[i].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[i].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
    if (i <= 2) {
      groups[i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      groups[i].generalShader = static_cast<uint32_t>(i);
    }
    else {
      groups[i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      groups[i].closestHitShader = static_cast<uint32_t>(i);
    }
  }

  VkRayTracingPipelineCreateInfoKHR ci {};
  ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  ci.stageCount = SBT_GROUP_COUNT;
  ci.pStages = stages;
  ci.groupCount = SBT_GROUP_COUNT;
  ci.pGroups = groups;
  ci.maxPipelineRayRecursionDepth = 2; // primary + one shadow level
  ci.layout = this->rtPipelineLayout;
  if (this->vkCreateRayTracingPipelinesKHR(
        this->device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci,
        this->allocator, &this->rtPipeline) != VK_SUCCESS) {
    return false;
  }
  if (!this->createShaderBindingTable()) {
    return false;
  }

  // Ray-query compute pipeline (default dispatch mode): the same path
  // tracer compiled as a compute shader, driven by vkCmdDispatch.
  VkComputePipelineCreateInfo computeCI {};
  computeCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computeCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  computeCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  computeCI.stage.module = this->pathTraceModule;
  computeCI.stage.pName = "main";
  computeCI.layout = this->rtPipelineLayout;
  return vkCreateComputePipelines(this->device, VK_NULL_HANDLE, 1, &computeCI,
                                  this->allocator,
                                  &this->computePipeline) == VK_SUCCESS;
}

bool
SoRTXRenderBackend::createShaderBindingTable()
{
  // Five records: raygen, miss, shadow miss, closest hit, shadow closest
  // hit, each aligned to the driver's shader-group-handle alignment.  The
  // table is host-visible so the group handles can be copied in directly.
  // Extra base-alignment slack keeps the strided region device addresses
  // aligned to shaderGroupBaseAlignment (VUID-vkCmdTraceRaysKHR-03675).
  const VkDeviceSize baseAlignment = this->sbtGroupBaseAlignment;
  const VkDeviceSize tableSize =
    static_cast<VkDeviceSize>(this->sbtRecordSize) * SBT_GROUP_COUNT +
    baseAlignment;
  if (!this->createHostVisibleBuffer(
        tableSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        this->sbtBuffer, this->sbtMemory)) {
    return false;
  }

  // Fetch the group handles (one per pipeline group) and copy each into
  // its aligned record slot.
  const uint32_t handleSize = this->sbtGroupHandleSize;
  std::vector<uint8_t> handles(static_cast<size_t>(handleSize) *
                               SBT_GROUP_COUNT);
  if (this->vkGetRayTracingShaderGroupHandlesKHR(
        this->device, this->rtPipeline, 0, SBT_GROUP_COUNT,
        handles.size(), handles.data()) != VK_SUCCESS) {
    return false;
  }
  void * mapped = nullptr;
  if (vkMapMemory(this->device, this->sbtMemory, 0, tableSize, 0,
                  &mapped) != VK_SUCCESS) {
    return false;
  }
  const VkDeviceAddress rawBase = this->getDeviceAddress(this->sbtBuffer);
  const VkDeviceAddress alignedBase =
    (rawBase + baseAlignment - 1) / baseAlignment * baseAlignment;
  this->sbtBaseOffset = alignedBase - rawBase;
  for (int i = 0; i < SBT_GROUP_COUNT; ++i) {
    std::memcpy(static_cast<uint8_t *>(mapped) + this->sbtBaseOffset +
                  static_cast<size_t>(i) * this->sbtRecordSize,
                handles.data() + static_cast<size_t>(i) * handleSize,
                handleSize);
  }
  vkUnmapMemory(this->device, this->sbtMemory);

  // Strided device-address regions handed to vkCmdTraceRaysKHR.
  const VkDeviceSize stride = this->sbtRecordSize;
  this->raygenSbtRegion = {alignedBase + 0 * stride, stride, stride};
  this->missSbtRegion = {alignedBase + 1 * stride, stride, 2 * stride};
  this->hitSbtRegion = {alignedBase + 3 * stride, stride, 2 * stride};
  this->callableSbtRegion = {0, 0, 0};
  return true;
}

bool
SoRTXRenderBackend::createPresentPipeline(VkRenderPass renderPass,
                                           VkSampleCountFlagBits sampleCount)
{
  // The present pass renders into the swapchain/MSAA color attachment, so
  // the pipeline's rasterization sample count must match the render pass
  // (VUID-VkGraphicsPipelineCreateInfo-renderPass-06082).  Key the cache on
  // both the render pass and the sample count.
  if (this->presentPipeline != VK_NULL_HANDLE &&
      this->presentRenderPass == renderPass &&
      this->presentSampleCount == sampleCount) {
    return true;
  }
  if (this->presentPipeline != VK_NULL_HANDLE) {
    // Defer: a pending frame may still bind this pipeline.
    VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkPipeline pipeline = this->presentPipeline;
    this->deferDestroy([device, allocator, pipeline]() {
      vkDestroyPipeline(device, pipeline, allocator);
    });
    this->presentPipeline = VK_NULL_HANDLE;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = this->presentVertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = this->presentFragmentModule;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization.cullMode = VK_CULL_MODE_NONE;
  rasterization.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = sampleCount;
  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  // The present pass writes the scene depth from the first-bounce hit
  // position (PresentFragment.glsl) so the raster composite overlay can
  // depth-test BRep edge lines against the traced surface, occluding hidden
  // edges.  Depth test stays off (it is a fullscreen present), but depth
  // write is enabled for the composite to see.
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
    VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
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
  ci.layout = this->presentPipelineLayout;
  ci.renderPass = renderPass;
  ci.subpass = 0;
  if (vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &ci,
                                this->allocator,
                                &this->presentPipeline) != VK_SUCCESS) {
    return false;
  }
  this->presentRenderPass = renderPass;
  this->presentSampleCount = sampleCount;
  return true;
}
