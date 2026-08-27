// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendTexture.cpp
//
// Texture cache and upload path.  Provides:
//
//   - createSampler()
//   - prepare/record/finalize staging-to-image uploads for changed textures
//     (prepareTextureUpload, recordTextureUpload, finalizeTexture,
//     recordPendingTextureUploads, finalizePendingTextureUploads)
//   - flushPendingTextureUploadsExternal() on the external-command-buffer path
//   - ensureDescriptorPoolSpace(): grow the descriptor pool
//   - allocateTextureDescriptorSet() / resolveTextureSet(): bind the
//     descriptor set a draw uses

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

// --- Texture cache --------------------------------------------------------

void
SoVulkanRenderBackend::destroyTextureEntry(VulkanCachedTexture & entry)
{
  if (entry.descriptorSet != VK_NULL_HANDLE) {
    if (entry.descriptorPool != VK_NULL_HANDLE) {
      vkFreeDescriptorSets(this->device, entry.descriptorPool, 1,
                           &entry.descriptorSet);
    }
    if (this->descriptorSetCount > 0) --this->descriptorSetCount;
    entry.descriptorSet = VK_NULL_HANDLE;
  }
  if (entry.sampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, entry.sampler, this->allocator);
    entry.sampler = VK_NULL_HANDLE;
  }
  if (entry.view != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, entry.view, this->allocator);
    entry.view = VK_NULL_HANDLE;
  }
  if (entry.image != VK_NULL_HANDLE) {
    vkDestroyImage(this->device, entry.image, this->allocator);
    entry.image = VK_NULL_HANDLE;
  }
  if (entry.memory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.memory, this->allocator);
    entry.memory = VK_NULL_HANDLE;
  }
  entry = VulkanCachedTexture();
}

void
SoVulkanRenderBackend::invalidateTextureCache()
{
  for (VulkanCachedTexture & entry : this->textureCache) {
    this->destroyTextureEntry(entry);
  }
  this->textureCache.clear();
  this->commandToTexture.clear();
}

VulkanCachedTexture &
SoVulkanRenderBackend::getOrCreateTexture(const SoRenderCommand * command)
{
  const auto found = this->commandToTexture.find(command);
  if (found != this->commandToTexture.end()) {
    return this->textureCache[found->second];
  }
  const size_t index = this->textureCache.size();
  this->textureCache.emplace_back();
  this->textureCache.back().commandKey = command;
  this->commandToTexture[command] = index;
  return this->textureCache.back();
}

bool
SoVulkanRenderBackend::createSampler(const SoTextureData & texture,
                                     VkSampler & sampler)
{
  VkSamplerCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  ci.magFilter = textureFilterToVk(texture.magFilter);
  ci.minFilter = textureFilterToVk(texture.minFilter);
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  ci.addressModeU = textureWrapToVk(texture.wrapS);
  ci.addressModeV = textureWrapToVk(texture.wrapT);
  ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.mipLodBias = 0.0f;
  ci.anisotropyEnable = VK_FALSE;
  ci.maxAnisotropy = 1.0f;
  ci.compareEnable = VK_FALSE;
  ci.minLod = 0.0f;
  ci.maxLod = 0.0f;
  ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  ci.unnormalizedCoordinates = VK_FALSE;
  return vkCreateSampler(this->device, &ci, this->allocator, &sampler) ==
         VK_SUCCESS;
}

bool
SoVulkanRenderBackend::prepareTextureUpload(VulkanCachedTexture & entry,
                                            const SoTextureData & texture,
                                            VkBuffer & staging,
                                            VkDeviceMemory & stagingMemory)
{
  // VK_FORMAT_R8G8B8_UNORM is not guaranteed to be sampleable, so expand
  // 3-component (RGB) textures to 4-component RGBA on the host.  Other
  // component counts map directly.
  const int components =
    (texture.numComponents == 3) ? 4 : texture.numComponents;
  if (components < 1 || components > 4) {
    this->emitError("prepareTextureUpload: unsupported component count");
    return false;
  }
  const VkFormat format = (texture.numComponents == 3)
    ? VK_FORMAT_R8G8B8A8_UNORM : textureFormatToVk(texture.numComponents);
  const VkDeviceSize byteSize =
    static_cast<VkDeviceSize>(texture.width) * texture.height * components;

  std::vector<unsigned char> converted;
  const unsigned char * uploadPixels = texture.pixels;
  if (texture.numComponents == 3) {
    const size_t pixelCount =
      static_cast<size_t>(texture.width) * texture.height;
    converted.resize(pixelCount * 4);
    for (size_t i = 0; i < pixelCount; ++i) {
      converted[i * 4 + 0] = texture.pixels[i * 3 + 0];
      converted[i * 4 + 1] = texture.pixels[i * 3 + 1];
      converted[i * 4 + 2] = texture.pixels[i * 3 + 2];
      converted[i * 4 + 3] = 255;
    }
    uploadPixels = converted.data();
  }

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {static_cast<uint32_t>(texture.width),
               static_cast<uint32_t>(texture.height), 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(this->device, &ci, this->allocator, &entry.image) !=
      VK_SUCCESS) {
    this->emitError("prepareTextureUpload: vkCreateImage failed");
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(this->device, entry.image, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = 0;
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &props);
  bool found = false;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      ai.memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    this->emitError("prepareTextureUpload: no device-local memory type");
    this->destroyTextureEntry(entry);
    return false;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator, &entry.memory) !=
      VK_SUCCESS) {
    this->emitError("prepareTextureUpload: vkAllocateMemory failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  vkBindImageMemory(this->device, entry.image, entry.memory, 0);

  if (!this->createBuffer(byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMemory, uploadPixels)) {
    this->emitError("prepareTextureUpload: staging buffer creation failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  return true;
}

void
SoVulkanRenderBackend::recordTextureUpload(
  VkCommandBuffer commandBuffer,
  const VulkanCachedTexture & entry,
  const SoTextureData & texture,
  VkBuffer staging)
{
  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = entry.image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {static_cast<uint32_t>(texture.width),
                        static_cast<uint32_t>(texture.height), 1};
  vkCmdCopyBufferToImage(commandBuffer, staging, entry.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);
}

bool
SoVulkanRenderBackend::finalizeTexture(VulkanCachedTexture & entry,
                                       const SoTextureData & texture)
{
  // The image format matches what prepareTextureUpload() created (RGB
  // textures are expanded to RGBA there).
  const VkFormat format = (texture.numComponents == 3)
    ? VK_FORMAT_R8G8B8A8_UNORM : textureFormatToVk(texture.numComponents);
  entry.view = createImageView(this->device, entry.image, format,
                               VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (entry.view == VK_NULL_HANDLE ||
      !this->createSampler(texture, entry.sampler) ||
      !this->allocateTextureDescriptorSet(entry.view, entry.sampler,
                                          entry.descriptorSet)) {
    this->emitError("finalizeTexture: view/sampler/descriptor creation failed");
    // Leave the entry half-initialized for the caller to dispose of.  On the
    // own-queue path the image is already referenced by recorded copies in
    // an unsubmitted command buffer, so the caller must defer the
    // destruction rather than destroy synchronously.
    return false;
  }
  entry.descriptorPool = this->descriptorPool;
  return true;
}

bool
SoVulkanRenderBackend::recordPendingTextureUploads()
{
  // Own-queue path: record the copies into the frame command buffer, ahead
  // of the render pass that samples them.  No separate submit is needed, so
  // no extra queue drain per frame.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    this->recordTextureUpload(this->currentCommandBuffer(),
                              this->textureCache[upload.index],
                              *upload.texture, upload.staging);
  }
  return true;
}

void
SoVulkanRenderBackend::finalizePendingTextureUploads()
{
  // Own-queue path: create the views/samplers/descriptor sets (bound by the
  // draws recorded below), stamp the content identity, and defer the staging
  // buffers to the frame's deferred-destruction batch.  Staging buffers are
  // referenced by the just-recorded submission, so they are released only
  // after the slot fence signals.
  VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    VulkanCachedTexture & texEntry = this->textureCache[upload.index];
    if (this->finalizeTexture(texEntry, *upload.texture)) {
      const SoTextureData & texture = *upload.texture;
      texEntry.pixelsKey = texture.pixels;
      texEntry.width = texture.width;
      texEntry.height = texture.height;
      texEntry.numComponents = texture.numComponents;
      texEntry.minFilter = texture.minFilter;
      texEntry.magFilter = texture.magFilter;
      texEntry.wrapS = texture.wrapS;
      texEntry.wrapT = texture.wrapT;
      texEntry.model = texture.model;
      texEntry.contentHash = hashTextureContent(texture);
    }
    else {
      // The entry's image is referenced by the recorded copies, so the
      // half-initialized resources must be destroyed through the deferred
      // ring, not synchronously.  Keys stay unstamped so the next frame
      // retries the upload.
      this->deferDestroyTextureEntry(texEntry);
    }
    if (upload.staging != VK_NULL_HANDLE) {
      const VkBuffer staging = upload.staging;
      const VkDeviceMemory stagingMemory = upload.stagingMemory;
      this->deferDestroy([device, allocator, staging, stagingMemory]() {
        if (staging != VK_NULL_HANDLE) {
          vkDestroyBuffer(device, staging, allocator);
        }
        if (stagingMemory != VK_NULL_HANDLE) {
          vkFreeMemory(device, stagingMemory, allocator);
        }
      });
    }
  }
  this->pendingUploads.clear();
}

bool
SoVulkanRenderBackend::flushPendingTextureUploadsExternal()
{
  if (this->pendingUploads.empty()) return true;

  // External path: the caller owns the frame command buffer and is already
  // inside a render pass, so the copies cannot be merged into it.  Record
  // them into one transient command buffer, submit, and wait.  The wait also
  // retires any in-flight frames submitted by the external caller, which
  // keeps ring-buffer reuse in renderExternal() safe even when the caller
  // pipelines more frames than maxFramesInFlight.
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer uploadBuffer = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(this->device, &allocInfo, &uploadBuffer) !=
      VK_SUCCESS) {
    this->emitError(
      "flushPendingTextureUploadsExternal: failed to allocate upload "
      "command buffer");
    goto fail;
  }

  {
    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(uploadBuffer, &bi) != VK_SUCCESS) {
      this->emitError(
        "flushPendingTextureUploadsExternal: failed to begin upload buffer");
      vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
      goto fail;
    }
  }

  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    this->recordTextureUpload(uploadBuffer, this->textureCache[upload.index],
                              *upload.texture, upload.staging);
  }

  if (vkEndCommandBuffer(uploadBuffer) != VK_SUCCESS) {
    this->emitError(
      "flushPendingTextureUploadsExternal: failed to end upload buffer");
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
    goto fail;
  }

  {
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &uploadBuffer;
    const VkResult submitResult =
      vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(this->queue);
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
    if (submitResult != VK_SUCCESS) {
      this->emitError("flushPendingTextureUploadsExternal: vkQueueSubmit "
                      "failed");
      goto fail;
    }
  }

  // Staging buffers are no longer referenced by the completed submission.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, upload.staging, this->allocator);
      vkFreeMemory(this->device, upload.stagingMemory, this->allocator);
    }
  }

  // Host-side completion (views/samplers/descriptor sets) and content
  // identity stamping.  The queue is idle here, so synchronous destruction
  // on failure is safe.  A failure leaves the content keys unstamped, so the
  // next frame retries the upload.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    VulkanCachedTexture & texEntry = this->textureCache[upload.index];
    if (this->finalizeTexture(texEntry, *upload.texture)) {
      const SoTextureData & texture = *upload.texture;
      texEntry.pixelsKey = texture.pixels;
      texEntry.width = texture.width;
      texEntry.height = texture.height;
      texEntry.numComponents = texture.numComponents;
      texEntry.minFilter = texture.minFilter;
      texEntry.magFilter = texture.magFilter;
      texEntry.wrapS = texture.wrapS;
      texEntry.wrapT = texture.wrapT;
      texEntry.model = texture.model;
      texEntry.contentHash = hashTextureContent(texture);
    }
    else {
      this->destroyTextureEntry(texEntry);
    }
  }
  this->pendingUploads.clear();
  return true;

fail:
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, upload.staging, this->allocator);
      vkFreeMemory(this->device, upload.stagingMemory, this->allocator);
    }
    if (upload.index < this->textureCache.size()) {
      // Reset the half-initialized entry so the next frame retries cleanly.
      this->destroyTextureEntry(this->textureCache[upload.index]);
    }
  }
  this->pendingUploads.clear();
  return false;
}

bool
SoVulkanRenderBackend::ensureDescriptorPoolSpace()
{
  // Each pool is sized for 1024 sets.  Textures accumulate per unique
  // command until the cache is invalidated (scene change, backend re-init),
  // so long-lived scenes with many distinct textures can exhaust the active
  // pool.  Resetting a pool wholesale would invalidate every set allocated
  // from it -- including sets referenced by frames the caller still has in
  // flight -- so instead a fresh pool is appended and becomes current.
  // Sets live in whatever pool allocated them and are freed back to that
  // pool (or destroyed with it at shutdown); never reset.
  if (this->descriptorSetCount < 1000) {
    return true;
  }
  this->descriptorSetCount = 0;
  return this->createDescriptorPool();
}

VkDescriptorSet
SoVulkanRenderBackend::resolveTextureSet(const SoRenderCommand & command)
{
  const auto found = this->commandToTexture.find(&command);
  if (found != this->commandToTexture.end() &&
      this->textureCache[found->second].descriptorSet != VK_NULL_HANDLE) {
    return this->textureCache[found->second].descriptorSet;
  }
  return this->whiteDescriptorSet;
}
