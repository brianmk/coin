// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendTexture.cpp
//
// Texture cache and upload path: sampler cache, staging-to-image uploads for
// changed textures (own-queue + external pre-pass), descriptor-pool growth and
// per-draw descriptor binding.  flushPendingTextureUploadsExternal() is the
// one-shot fallback when the external pre-pass cannot allocate its transient
// command buffer.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanDebugUtils.h"
#include "rendering/SoVulkanShared.h"

#include <vk_mem_alloc.h>

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

namespace {

// Record an uploaded texture's identity on the cache entry: pixel pointer,
// dimensions/components, sampler state, content hash (both completion paths).
void stampTextureContent(VulkanCachedTexture & entry,
                         const SoTextureData & texture)
{
  entry.pixelsKey = texture.pixels;
  entry.width = texture.width;
  entry.height = texture.height;
  entry.numComponents = texture.numComponents;
  entry.minFilter = texture.minFilter;
  entry.magFilter = texture.magFilter;
  entry.wrapS = texture.wrapS;
  entry.wrapT = texture.wrapT;
  entry.model = texture.model;
  entry.contentHash = hashTextureContent(texture);
}

} // namespace

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
    vmaDestroyImage(this->vmaAllocator, entry.image, entry.allocation);
    entry.image = VK_NULL_HANDLE;
    entry.allocation = nullptr;
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
  // R8/R8G8 aren't core-required sampled formats: expand 1-/2-component
  // textures to R8G8B8A8_UNORM unless SAMPLED_IMAGE is supported (RGB always).
  if (numComponents == 3) return VK_FORMAT_R8G8B8A8_UNORM;
  if (numComponents == 1 && !this->sampledR8) return VK_FORMAT_R8G8B8A8_UNORM;
  if (numComponents == 2 && !this->sampledR8G8) return VK_FORMAT_R8G8B8A8_UNORM;
  return textureFormatToVk(numComponents);
}

bool
SoVulkanRenderBackend::ensureStagingPoolSize(VkDeviceSize required)
{
  // Caller needs `required` MORE bytes at the cursor: fit cursor + required, not
  // merely `required`, or a second mid-frame upload overruns the buffer end.
  if (this->stagingPoolBuffer != VK_NULL_HANDLE &&
      this->stagingPoolCapacity >= this->stagingPoolCursor + required) {
    return true;
  }
  // Grow to at least double, amortizing a frame's upload burst into one realloc.
  VkDeviceSize newCapacity =
    std::max<VkDeviceSize>(this->stagingPoolCursor + required,
                           this->stagingPoolCapacity * 2);
  newCapacity = std::max<VkDeviceSize>(newCapacity, 256u * 1024u);

  VkBuffer newBuffer = VK_NULL_HANDLE;
  VmaAllocation newAllocation = nullptr;
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = newCapacity;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  // Persistent host mapping up front (written every frame; no per-upload
  // vkMapMemory).  VMA_MEMORY_USAGE_AUTO needs an explicit host-access flag.
  allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator, &bci, &allocInfo, &newBuffer,
                      &newAllocation, &allocationInfo) != VK_SUCCESS) {
    return false;
  }
  void * newMapped = allocationInfo.pMappedData;
  if (newMapped == nullptr) {
    // Unexpected with MAPPED_BIT on host-visible memory; don't leak a pool.
    vmaDestroyBuffer(this->vmaAllocator, newBuffer, newAllocation);
    return false;
  }
  // Preserve bytes already staged this frame before swapping buffers.
  if (this->stagingPoolBuffer != VK_NULL_HANDLE && this->stagingPoolMapped &&
      this->stagingPoolCursor > 0) {
    std::memcpy(newMapped, this->stagingPoolMapped,
                static_cast<size_t>(this->stagingPoolCursor));
  }
  if (this->stagingPoolBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(this->vmaAllocator, this->stagingPoolBuffer,
                     this->stagingPoolAllocation);
  }
  this->stagingPoolBuffer = newBuffer;
  this->stagingPoolAllocation = newAllocation;
  this->stagingPoolMapped = newMapped;
  this->stagingPoolCapacity = newCapacity;
  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_BUFFER,
    reinterpret_cast<uint64_t>(this->stagingPoolBuffer), "texture staging pool");
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

  // Expanded uploads sample as native: R8->(r,0,0,1), R8G8->(r,g,0,1), RGB->(r,g,b,1).
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
  // VMA owns the image + device memory: one vmaCreateImage creates, allocates
  // and binds, sub-allocating large blocks (device-local).
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (vmaCreateImage(this->vmaAllocator, &ci, &allocInfo, &entry.image,
                     &entry.allocation, nullptr) != VK_SUCCESS) {
    this->emitError("prepareTextureUpload: vmaCreateImage failed");
    return false;
  }

  // Stage into the shared host-visible staging pool (reused across frames,
  // grown on demand): a frame's uploads coalesce into one buffer.
  // stagingOffset is the frame-monotonic byte offset where these pixels land.
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
  // Must match prepareTextureUpload()'s image (RGB and unsupported R/RG -> RGBA).
  const VkFormat format = this->effectiveTextureFormat(texture.numComponents);
  entry.view = createImageView(this->device, entry.image, format,
                               VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (entry.view == VK_NULL_HANDLE ||
      // Shared sampler: identical filter/wrap state reuses one cached VkSampler.
      (entry.sampler = this->cachedSampler(texture.minFilter, texture.magFilter,
                                           texture.wrapS, texture.wrapT)) ==
        VK_NULL_HANDLE ||
      !this->allocateTextureDescriptorSet(entry.view, entry.sampler,
                                          entry.descriptorSet)) {
    this->emitError("finalizeTexture: view/sampler/descriptor creation failed");
    // Leave the entry half-initialized: on the own-queue path recorded copies
    // still reference the image, so the caller must destroy it via the ring.
    return false;
  }
  entry.descriptorPool = this->descriptorPool;
  return true;
}

void
SoVulkanRenderBackend::recordPendingTextureUploadsInto(VkCommandBuffer commandBuffer)
{
  // Record the copies into the caller's command buffer (which must not be in a
  // render pass).  The caller owns submission and must order them before the
  // sampling draws (own-queue: same frame buffer; external: submit/waits first).
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    this->recordTextureUpload(commandBuffer, this->textureCache[upload.index],
                              *upload.texture, this->stagingPoolBuffer,
                              upload.stagingOffset);
  }
}

bool
SoVulkanRenderBackend::recordPendingTextureUploads()
{
  // Own-queue path: copies go into the frame buffer ahead of the pass, no submit.
  this->recordPendingTextureUploadsInto(this->currentCommandBuffer());
  return true;
}

void
SoVulkanRenderBackend::finalizePendingTextureUploads()
{
  // Own-queue path: create views/samplers/descriptor sets, stamp content, and
  // defer staging buffers to the frame's deferred-destruction batch (released
  // only after the slot fence signals).
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    VulkanCachedTexture & texEntry = this->textureCache[upload.index];
    if (this->finalizeTexture(texEntry, *upload.texture)) {
      stampTextureContent(texEntry, *upload.texture);
    }
    else {
      // The image is referenced by recorded copies, so defer destruction of the
      // half-initialized resources.  Keys stay unstamped to retry next frame.
      this->deferDestroyTextureEntry(texEntry);
    }
  }
  this->pendingUploads.clear();
}

SoVulkan::Result
SoVulkanRenderBackend::flushPendingTextureUploadsExternal()
{
  if (this->pendingUploads.empty()) return SoVulkan::Result::ok();

  // One-shot fallback for the external path, used only when
  // beginExternalPrepass() could not allocate its transient buffer (which
  // otherwise carries the copies at no extra submission).  The caller is inside
  // a render pass and owns the frame buffer, so copies cannot merge there; all
  // pending uploads sit in the shared staging pool, so one submit copies them
  // all.  The wait also retires in-flight external frames, keeping the pool
  // reusable even when the caller pipelines more frames than maxFramesInFlight.
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
    // Batch failure means none completed and every entry is half-initialized;
    // reset all so the next frame retries.  (Deduped in prepareGeometryTextures.)
    for (const PendingTextureUpload & upload : this->pendingUploads) {
      if (upload.index < this->textureCache.size()) {
        this->destroyTextureEntry(this->textureCache[upload.index]);
      }
    }
    this->pendingUploads.clear();
    return SoVulkan::Result::error("one-shot texture upload failed");
  }

  // Host-side completion (views/samplers/descriptor sets) and content stamping;
  // on failure keys stay unstamped so the next frame retries.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index >= this->textureCache.size()) continue;
    VulkanCachedTexture & texEntry = this->textureCache[upload.index];
    if (this->finalizeTexture(texEntry, *upload.texture)) {
      stampTextureContent(texEntry, *upload.texture);
    }
    else {
      this->destroyTextureEntry(texEntry);
    }
  }
  this->pendingUploads.clear();
  return SoVulkan::Result::ok();
}

bool
SoVulkanRenderBackend::ensureDescriptorPoolSpace()
{
  // Each pool holds 1024 sets; textures accumulate per unique command until the
  // cache is invalidated, so texture-heavy scenes can exhaust the active pool.
  // Resetting would invalidate allocated sets (incl. in-flight caller frames),
  // so append a fresh pool.  Sets are freed to their allocating pool; never reset.
  if (this->descriptorSetCount < 1000) {
    return true;
  }
  this->descriptorSetCount = 0;
  return this->createDescriptorPool();
}

VkDescriptorSet
SoVulkanRenderBackend::resolveTextureSet(const SoRenderCommand & command)
{
  // Fast path for the common untextured case: return the white set, no lookup.
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
