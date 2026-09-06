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
#include "rendering/SoVulkanShared.h"

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
SoVulkanRenderBackend::releaseMemory(VkDeviceMemory memory, VkDeviceSize size,
                                    VkDeviceSize offset)
{
  if (memory == VK_NULL_HANDLE) return;
  if (this->usingMemPool() && size > 0) {
    // Return the range to the sub-allocator.  Safe here because the callers
    // defer this free: destroyTextureEntry() flushes through the deferred ring
    // (see deferDestroyTextureEntry), so the GPU can no longer reference the
    // range when this runs.
    this->memPool->free(memory, offset, size);
  }
  else {
    vkFreeMemory(this->device, memory, this->allocator);
  }
}

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
    // Shared sampler owned by samplerCache; released at shutdown(), not here.
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
    this->releaseMemory(entry.memory, entry.memorySize, entry.memoryOffset);
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

SoVulkanRenderBackend::SamplerKey
SoVulkanRenderBackend::samplerKey(SoTextureFilter minFilter,
                                  SoTextureFilter magFilter,
                                  SoTextureWrap wrapS, SoTextureWrap wrapT)
{
  return static_cast<SamplerKey>(
    (static_cast<uint8_t>(minFilter) & 0x3u) |
    ((static_cast<uint8_t>(magFilter) & 0x3u) << 2u) |
    ((static_cast<uint8_t>(wrapS) & 0x3u) << 4u) |
    ((static_cast<uint8_t>(wrapT) & 0x3u) << 6u));
}

VkSampler
SoVulkanRenderBackend::cachedSampler(SoTextureFilter minFilter,
                                     SoTextureFilter magFilter,
                                     SoTextureWrap wrapS, SoTextureWrap wrapT)
{
  const SamplerKey key = samplerKey(minFilter, magFilter, wrapS, wrapT);
  const auto found = this->samplerCache.find(key);
  if (found != this->samplerCache.end()) return found->second;
  VkSampler sampler = VK_NULL_HANDLE;
  if (this->createSampler(minFilter, magFilter, wrapS, wrapT, sampler)) {
    this->samplerCache.emplace(key, sampler);
  }
  return sampler;
}

bool
SoVulkanRenderBackend::createSampler(SoTextureFilter minFilter,
                                     SoTextureFilter magFilter,
                                     SoTextureWrap wrapS, SoTextureWrap wrapT,
                                     VkSampler & sampler)
{
  VkSamplerCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  ci.magFilter = textureFilterToVk(magFilter);
  ci.minFilter = textureFilterToVk(minFilter);
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  ci.addressModeU = textureWrapToVk(wrapS);
  ci.addressModeV = textureWrapToVk(wrapT);
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

VkFormat
SoVulkanRenderBackend::effectiveTextureFormat(const int numComponents) const
{
  // VK_FORMAT_R8_UNORM and VK_FORMAT_R8G8_UNORM are not core-required
  // sampled formats, so expand 1- and 2-component textures to
  // VK_FORMAT_R8G8B8A8_UNORM (a required format) when the device lacks
  // SAMPLED_IMAGE support; the same host-side expansion the 3-component
  // path always applies for the optional VK_FORMAT_R8G8B8_UNORM.  Component
  // counts that map directly are unchanged.
  if (numComponents == 3) return VK_FORMAT_R8G8B8A8_UNORM;
  if (numComponents == 1 && !this->sampledR8) return VK_FORMAT_R8G8B8A8_UNORM;
  if (numComponents == 2 && !this->sampledR8G8) return VK_FORMAT_R8G8B8A8_UNORM;
  return textureFormatToVk(numComponents);
}

bool
SoVulkanRenderBackend::ensureStagingPoolSize(VkDeviceSize required)
{
  // The caller needs `required` MORE bytes at the current cursor: the pool
  // must fit cursor + required, not merely `required` (two mid-size uploads
  // in one frame would otherwise each pass the check individually yet
  // overrun the buffer end -- heap corruption downstream).
  if (this->stagingPoolBuffer != VK_NULL_HANDLE &&
      this->stagingPoolCapacity >= this->stagingPoolCursor + required) {
    return true;
  }
  // Grow: at least double the current capacity so a burst of uploads in a
  // frame amortizes a single reallocation instead of one per upload.
  VkDeviceSize newCapacity =
    std::max<VkDeviceSize>(this->stagingPoolCursor + required,
                           this->stagingPoolCapacity * 2);
  newCapacity = std::max<VkDeviceSize>(newCapacity, 256u * 1024u);

  VkBuffer newBuffer = VK_NULL_HANDLE;
  VkDeviceMemory newMemory = VK_NULL_HANDLE;
  if (!SoVulkanShared::createBufferAllocated(
        this->device, this->allocator, newCapacity,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        /*deviceAddress*/ false,
        [this](const VkMemoryRequirements & req, VkMemoryPropertyFlags desired,
               uint32_t & memoryTypeIndex) {
          return this->selectMemoryType(req, desired, memoryTypeIndex);
        }, newBuffer, newMemory)) {
    return false;
  }
  void * newMapped = nullptr;
  if (vkMapMemory(this->device, newMemory, 0, newCapacity, 0, &newMapped) !=
      VK_SUCCESS) {
    vkDestroyBuffer(this->device, newBuffer, this->allocator);
    vkFreeMemory(this->device, newMemory, this->allocator);
    return false;
  }
  // Preserve any bytes already staged in the old buffer (uploads prepared
  // earlier in this frame) before swapping it out.
  if (this->stagingPoolBuffer != VK_NULL_HANDLE && this->stagingPoolMapped &&
      this->stagingPoolCursor > 0) {
    std::memcpy(newMapped, this->stagingPoolMapped,
                static_cast<size_t>(this->stagingPoolCursor));
  }
  if (this->stagingPoolBuffer != VK_NULL_HANDLE) {
    if (this->stagingPoolMapped != nullptr) {
      vkUnmapMemory(this->device, this->stagingPoolMemory);
    }
    vkDestroyBuffer(this->device, this->stagingPoolBuffer, this->allocator);
    vkFreeMemory(this->device, this->stagingPoolMemory, this->allocator);
  }
  this->stagingPoolBuffer = newBuffer;
  this->stagingPoolMemory = newMemory;
  this->stagingPoolMapped = newMapped;
  this->stagingPoolCapacity = newCapacity;
  return true;
}

bool
SoVulkanRenderBackend::prepareTextureUpload(VulkanCachedTexture & entry,
                                            const SoTextureData & texture,
                                            VkDeviceSize & stagingOffset,
                                            VkDeviceSize & stagingBytes)
{
  if (texture.numComponents < 1 || texture.numComponents > 4) {
    this->emitError("prepareTextureUpload: unsupported component count");
    return false;
  }
  const VkFormat format = this->effectiveTextureFormat(texture.numComponents);
  const bool expandToRgba = (format == VK_FORMAT_R8G8B8A8_UNORM &&
                             texture.numComponents < 4);
  const int components = expandToRgba ? 4 : texture.numComponents;
  const VkDeviceSize byteSize =
    static_cast<VkDeviceSize>(texture.width) * texture.height * components;

  // The expanded upload must sample identically to the native format it
  // replaces, so the extra channels take the values the hardware would have
  // produced: R8 -> (r,0,0,1), R8G8 -> (r,g,0,1), RGB -> (r,g,b,1).
  std::vector<unsigned char> converted;
  const unsigned char * uploadPixels = texture.pixels;
  if (expandToRgba) {
    const size_t pixelCount =
      static_cast<size_t>(texture.width) * texture.height;
    converted.resize(pixelCount * 4);
    for (size_t i = 0; i < pixelCount; ++i) {
      const unsigned char * src =
        texture.pixels + i * static_cast<size_t>(texture.numComponents);
      converted[i * 4 + 0] = src[0];
      converted[i * 4 + 1] = texture.numComponents >= 2 ? src[1] : 0;
      converted[i * 4 + 2] = texture.numComponents == 3 ? src[2] : 0;
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
  entry.memorySize = requirements.size;
  // Stage the pixels into the shared staging pool.  The pool is host-visible
  // and reused across frames (grown on demand), so all pending uploads of a
  // frame coalesce into one buffer -- one allocation, one cleanup surface --
  // rather than a fresh per-upload staging buffer.  st_offset is the byte
  // offset (grown monotonically within the frame) where these pixels land.
  if (!this->ensureStagingPoolSize(byteSize)) {
    this->emitError("prepareTextureUpload: staging pool growth failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  stagingOffset = this->stagingPoolCursor;
  stagingBytes = byteSize;
  unsigned char * dst = static_cast<unsigned char *>(this->stagingPoolMapped) +
                        static_cast<size_t>(stagingOffset);
  std::memcpy(dst, uploadPixels, static_cast<size_t>(byteSize));
  this->stagingPoolCursor += ((byteSize + 3u) & ~(VkDeviceSize)3u);

  if (this->usingMemPool()) {
    // Sub-allocate the image memory from the pool; falls back to a standalone
    // allocation if the pool cannot fit/grow the range.
    uint32_t poolTypeIndex = 0;
    if (this->selectMemoryType(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               poolTypeIndex)) {
      VkDeviceSize offset = 0;
      if (this->memPool->alloc(poolTypeIndex, requirements.size,
                               requirements.alignment, entry.memory, offset)) {
      entry.memoryOffset = offset;
      const VkResult bindRes = vkBindImageMemory(
        this->device, entry.image, entry.memory, offset);
      if (bindRes == VK_SUCCESS) {
        return true;
      }
        // Binding failed: return the range to the pool and fall through to the
        // legacy path so the upload can still proceed (at a cost of correctness
        // pressure only in the failure case).
        this->releaseMemory(entry.memory, entry.memorySize, entry.memoryOffset);
        entry.memory = VK_NULL_HANDLE;
      }
    }
  }
  uint32_t memoryTypeIndex = 0;
  if (!this->selectMemoryType(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              memoryTypeIndex)) {
    this->emitError("prepareTextureUpload: no device-local memory type");
    this->destroyTextureEntry(entry);
    return false;
  }
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = memoryTypeIndex;
  if (vkAllocateMemory(this->device, &ai, this->allocator, &entry.memory) !=
      VK_SUCCESS) {
    this->emitError("prepareTextureUpload: vkAllocateMemory failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  entry.memoryOffset = 0;
  vkBindImageMemory(this->device, entry.image, entry.memory, 0);

  return true;
}

void
SoVulkanRenderBackend::recordTextureUpload(
  VkCommandBuffer commandBuffer,
  const VulkanCachedTexture & entry,
  const SoTextureData & texture,
  VkBuffer staging,
  VkDeviceSize stagingOffset)
{
  SoVulkanShared::imageTransition(
    commandBuffer, entry.image,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    0, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkBufferImageCopy region {};
  region.bufferOffset = stagingOffset;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {static_cast<uint32_t>(texture.width),
                        static_cast<uint32_t>(texture.height), 1};
  vkCmdCopyBufferToImage(commandBuffer, staging, entry.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  SoVulkanShared::imageTransition(
    commandBuffer, entry.image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

bool
SoVulkanRenderBackend::finalizeTexture(VulkanCachedTexture & entry,
                                       const SoTextureData & texture)
{
  // The image format matches what prepareTextureUpload() created (RGB and
  // unsupported R/RG textures are expanded to RGBA there).
  const VkFormat format = this->effectiveTextureFormat(texture.numComponents);
  entry.view = createImageView(this->device, entry.image, format,
                               VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (entry.view == VK_NULL_HANDLE ||
      // Shared sampler: entries with identical filter/wrap state reuse one
      // VkSampler from samplerCache instead of creating one per texture entry.
      (entry.sampler = this->cachedSampler(texture.minFilter, texture.magFilter,
                                           texture.wrapS, texture.wrapT)) ==
        VK_NULL_HANDLE ||
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
                              *upload.texture, this->stagingPoolBuffer,
                              upload.stagingOffset);
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
  }
  this->pendingUploads.clear();
}

bool
SoVulkanRenderBackend::flushPendingTextureUploadsExternal()
{
  if (this->pendingUploads.empty()) return true;

  // External path: the caller owns the frame command buffer and is already
  // inside a render pass, so the copies cannot be merged into it.  All
  // pending uploads were staged into the single shared staging pool buffer
  // at their recording offsets, so one submit copies every pending texture.
  // The wait also retires any in-flight frames submitted by the external
  // caller, which keeps the staging pool free for reuse even when the caller
  // pipelines more frames than maxFramesInFlight.
  if (!SoVulkanShared::withOneShotSubmit(
        this->device, this->queue, this->commandPool, this->allocator,
        [this](VkCommandBuffer uploadBuffer) {
          for (const PendingTextureUpload & upload : this->pendingUploads) {
            if (upload.index >= this->textureCache.size()) continue;
            this->recordTextureUpload(uploadBuffer,
                                      this->textureCache[upload.index],
                                      *upload.texture,
                                      this->stagingPoolBuffer,
                                      upload.stagingOffset);
          }
        })) {
    this->emitError(
      "flushPendingTextureUploadsExternal: one-shot upload failed");
    goto fail;
  }

  // Host-side completion (views/samplers/descriptor sets) and content
  // identity stamping.  The queue is idle here, so the shared staging pool
  // is free to be reused by the next frame.  A failure leaves the content
  // keys unstamped, so the next frame retries the upload.
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
  // Fast path for the overwhelmingly common untextured case: most retained
  // commands (default CAD surfaces, edges, points) carry no texture, so fall
  // straight through to the white set without touching the commandToTexture
  // unordered_map (a hash + bucket walk per draw otherwise).
  const SoTextureData & tex = command.material.texture;
  if (!tex.pixels || tex.width == 0 || tex.height == 0 ||
      tex.numComponents == 0) {
    return this->whiteDescriptorSet;
  }
  const auto found = this->commandToTexture.find(&command);
  if (found != this->commandToTexture.end() &&
      this->textureCache[found->second].descriptorSet != VK_NULL_HANDLE) {
    return this->textureCache[found->second].descriptorSet;
  }
  return this->whiteDescriptorSet;
}
