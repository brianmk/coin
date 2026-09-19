// src/rendering/SoVulkanTextureCache.cpp
#include "rendering/SoVulkanTextureCache.h"

#include "rendering/SoRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanShared.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

using namespace CoinVulkanDetail;

namespace {

// Record the identity of an uploaded texture on the cache entry: the pixel
// pointer, dimensions/components, sampler state and the sampled content hash.
// Both upload-completion paths (own-queue and external) stamped the same ten
// fields by hand.
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

bool
SoVulkanTextureCache::initialize(const SoVulkanTextureCacheContext & context)
{
  this->context_ = context;
  this->stagingPool_.initialize(context.device, context.vmaAllocator);
  return this->createWhiteTexture();
}

void
SoVulkanTextureCache::destroy()
{
  this->invalidate();
  this->stagingPool_.destroy();
  this->whiteSampler_ = VK_NULL_HANDLE;
  if (this->whiteImageView_ != VK_NULL_HANDLE) {
    vkDestroyImageView(this->context_.device, this->whiteImageView_,
                       this->context_.allocator);
    this->whiteImageView_ = VK_NULL_HANDLE;
  }
  if (this->whiteImage_ != VK_NULL_HANDLE) {
    vmaDestroyImage(this->context_.vmaAllocator, this->whiteImage_,
                    this->whiteImageAllocation_);
    this->whiteImage_ = VK_NULL_HANDLE;
    this->whiteImageAllocation_ = nullptr;
  }
  this->whiteDescriptorSet_ = VK_NULL_HANDLE;
}

void
SoVulkanTextureCache::invalidate()
{
  for (VulkanCachedTexture & entry : this->entries_) {
    this->destroyEntry(entry);
  }
  this->entries_.clear();
  this->commandToIndex_.clear();
}

VulkanCachedTexture &
SoVulkanTextureCache::getOrCreate(const SoRenderCommand * command)
{
  const auto found = this->commandToIndex_.find(command);
  if (found != this->commandToIndex_.end()) {
    return this->entries_[found->second];
  }
  const size_t index = this->entries_.size();
  this->entries_.emplace_back();
  this->entries_.back().commandKey = command;
  this->commandToIndex_[command] = index;
  return this->entries_.back();
}

void
SoVulkanTextureCache::destroyEntry(VulkanCachedTexture & entry)
{
  if (entry.descriptorSet != VK_NULL_HANDLE) {
    if (this->context_.freeDescriptorSet) {
      this->context_.freeDescriptorSet(entry.descriptorSet,
                                       entry.descriptorPool);
    }
    entry.descriptorSet = VK_NULL_HANDLE;
  }
  // Shared sampler owned by the sampler cache; released at destroy(), not here.
  entry.sampler = VK_NULL_HANDLE;
  if (entry.view != VK_NULL_HANDLE) {
    vkDestroyImageView(this->context_.device, entry.view,
                       this->context_.allocator);
    entry.view = VK_NULL_HANDLE;
  }
  if (entry.image != VK_NULL_HANDLE) {
    vmaDestroyImage(this->context_.vmaAllocator, entry.image,
                    entry.allocation);
    entry.image = VK_NULL_HANDLE;
    entry.allocation = nullptr;
  }
  entry = VulkanCachedTexture();
}

bool
SoVulkanTextureCache::prepareCommand(const SoRenderCommand & command,
                                     uint32_t generation)
{
  const SoTextureData & texture = command.material.texture;
  if (!(texture.pixels && texture.width > 0 && texture.height > 0)) {
    return false;
  }
  VulkanCachedTexture & entry = this->getOrCreate(&command);
  // Texture pixels come from per-frame action storage (arena), which may
  // rewrite the same pointer in place, so the content hash is always
  // re-verified -- pointer identity alone is not sound for textures.
  const bool matches = entry.image != VK_NULL_HANDLE &&
    entry.pixelsKey == texture.pixels &&
    entry.width == texture.width &&
    entry.height == texture.height &&
    entry.numComponents == texture.numComponents &&
    entry.minFilter == texture.minFilter &&
    entry.magFilter == texture.magFilter &&
    entry.wrapS == texture.wrapS &&
    entry.wrapT == texture.wrapT &&
    entry.model == texture.model &&
    entry.contentHash == hashTextureContent(texture);
  bool staged = false;
  if (!matches) {
    this->context_.deferDestroyEntry(entry);
    // A command that appears twice in one draw list would otherwise prepare two
    // uploads for the same entry (leaking the first image); the first pending
    // upload for this index wins.
    bool alreadyPending = false;
    for (const PendingUpload & prior : this->pendingUploads_) {
      if (prior.index == this->commandToIndex_[&command]) {
        alreadyPending = true;
        break;
      }
    }
    if (!alreadyPending) {
      PendingUpload upload;
      upload.command = &command;
      upload.index = this->commandToIndex_[&command];
      upload.texture = &texture;
      // Report the attempt (matching the caller's breadcrumb count) even if the
      // staging allocation failed; on failure the entry is reset by
      // prepareUpload() and the content keys stay unstamped so the next frame
      // retries.
      staged = true;
      if (this->prepareUpload(entry, texture, upload.stagingOffset,
                              upload.stagingBytes)) {
        this->pendingUploads_.push_back(upload);
      }
    }
  }
  entry.commandKey = &command;
  entry.cacheGeneration = generation;
  return staged;
}

void
SoVulkanTextureCache::sweep(uint32_t generation)
{
  // Evict entries not visited this generation: their command has disappeared
  // from the draw list.  Destruction is deferred: a pending frame may still
  // reference the evicted images.  Entries surviving eviction keep their index
  // identity, so rebuild the pointer map from the stored commandKey.
  bool anyStale = false;
  for (VulkanCachedTexture & entry : this->entries_) {
    if (entry.cacheGeneration != generation) {
      this->context_.deferDestroyEntry(entry);
      anyStale = true;
    }
  }
  if (anyStale) {
    size_t write = 0;
    for (size_t idx = 0; idx < this->entries_.size(); ++idx) {
      if (this->entries_[idx].cacheGeneration == generation) {
        if (write != idx) this->entries_[write] = std::move(this->entries_[idx]);
        ++write;
      }
    }
    this->entries_.resize(write);
    this->commandToIndex_.clear();
    for (size_t idx = 0; idx < this->entries_.size(); ++idx) {
      this->commandToIndex_[this->entries_[idx].commandKey] = idx;
    }
  }

  // Compaction reindexes the cache, so the upload indices captured during
  // prepareCommand() are stale.  Re-resolve each pending upload through its
  // command pointer; entries just prepared carry the current generation and
  // survive the sweep.
  for (PendingUpload & upload : this->pendingUploads_) {
    const auto it = this->commandToIndex_.find(upload.command);
    if (it != this->commandToIndex_.end()) {
      upload.index = it->second;
    }
    else {
      upload.index = std::numeric_limits<size_t>::max();
    }
  }
}

void
SoVulkanTextureCache::resetStaging()
{
  this->stagingPool_.reset();
}

VkFormat
SoVulkanTextureCache::effectiveFormat(const int numComponents) const
{
  // VK_FORMAT_R8_UNORM and VK_FORMAT_R8G8_UNORM are not core-required sampled
  // formats, so expand 1- and 2-component textures to VK_FORMAT_R8G8B8A8_UNORM
  // (a required format) when the device lacks SAMPLED_IMAGE support; the same
  // host-side expansion the 3-component path always applies for the optional
  // VK_FORMAT_R8G8B8_UNORM.  Component counts that map directly are unchanged.
  if (numComponents == 3) return VK_FORMAT_R8G8B8A8_UNORM;
  if (numComponents == 1 && !this->context_.sampledR8) {
    return VK_FORMAT_R8G8B8A8_UNORM;
  }
  if (numComponents == 2 && !this->context_.sampledR8G8) {
    return VK_FORMAT_R8G8B8A8_UNORM;
  }
  return textureFormatToVk(numComponents);
}

bool
SoVulkanTextureCache::prepareUpload(VulkanCachedTexture & entry,
                                    const SoTextureData & texture,
                                    VkDeviceSize & stagingOffset,
                                    VkDeviceSize & stagingBytes)
{
  if (texture.numComponents < 1 || texture.numComponents > 4) {
    if (this->context_.emitError) {
      this->context_.emitError(
        "prepareTextureUpload: unsupported component count");
    }
    return false;
  }
  const VkFormat format = this->effectiveFormat(texture.numComponents);
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
  // The image and its device memory are owned by the VMA allocator: a single
  // vmaCreateImage creates, allocates and binds, sub-allocating from large
  // blocks rather than hitting the driver (and maxMemoryAllocationCount) per
  // upload.  Device-local, matching the old selectMemoryType() requirement.
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (vmaCreateImage(this->context_.vmaAllocator, &ci, &allocInfo, &entry.image,
                     &entry.allocation, nullptr) != VK_SUCCESS) {
    if (this->context_.emitError) {
      this->context_.emitError("prepareTextureUpload: vmaCreateImage failed");
    }
    return false;
  }

  // Stage the pixels into the shared staging pool.  The pool is host-visible
  // and reused across frames (grown on demand), so all pending uploads of a
  // frame coalesce into one buffer -- one allocation, one cleanup surface --
  // rather than a fresh per-upload staging buffer.
  if (!this->stagingPool_.stage(uploadPixels, byteSize, stagingOffset)) {
    if (this->context_.emitError) {
      this->context_.emitError(
        "prepareTextureUpload: staging pool growth failed");
    }
    this->destroyEntry(entry);
    return false;
  }
  stagingBytes = byteSize;
  return true;
}

void
SoVulkanTextureCache::recordUpload(VkCommandBuffer commandBuffer,
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
SoVulkanTextureCache::finalizeEntry(VulkanCachedTexture & entry,
                                    const SoTextureData & texture)
{
  // The image format matches what prepareUpload() created (RGB and unsupported
  // R/RG textures are expanded to RGBA there).
  const VkFormat format = this->effectiveFormat(texture.numComponents);
  entry.view = createImageView(this->context_.device, entry.image, format,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               this->context_.allocator);
  if (entry.view == VK_NULL_HANDLE ||
      (entry.sampler = this->context_.samplerCache->get(
         texture.minFilter, texture.magFilter, texture.wrapS, texture.wrapT)) ==
        VK_NULL_HANDLE ||
      !this->context_.allocateDescriptorSet(entry.view, entry.sampler,
                                            entry.descriptorSet)) {
    if (this->context_.emitError) {
      this->context_.emitError(
        "finalizeTexture: view/sampler/descriptor creation failed");
    }
    // Leave the entry half-initialized for the caller to dispose of.  On the
    // own-queue path the image is already referenced by recorded copies in an
    // unsubmitted command buffer, so the caller must defer the destruction
    // rather than destroy synchronously.
    return false;
  }
  return true;
}

void
SoVulkanTextureCache::recordPendingInto(VkCommandBuffer commandBuffer)
{
  // Record the copies into the caller's command buffer (which must not be
  // inside a render pass).  The caller owns submission, so it is responsible
  // for ordering them ahead of the draws that sample the images.
  for (const PendingUpload & upload : this->pendingUploads_) {
    if (upload.index >= this->entries_.size()) continue;
    this->recordUpload(commandBuffer, this->entries_[upload.index],
                       *upload.texture, this->stagingPool_.buffer(),
                       upload.stagingOffset);
  }
}

bool
SoVulkanTextureCache::recordPending()
{
  // Own-queue path: record the copies into the frame command buffer, ahead of
  // the render pass that samples them.  No separate submit is needed, so no
  // extra queue drain per frame.  A null command buffer means the frame ring
  // could not be allocated; report it so the caller's error branch is not dead.
  VkCommandBuffer cmd =
    this->context_.currentCommandBuffer ? this->context_.currentCommandBuffer()
                                        : VK_NULL_HANDLE;
  if (cmd == VK_NULL_HANDLE) return false;
  this->recordPendingInto(cmd);
  return true;
}

void
SoVulkanTextureCache::finalizePending()
{
  // Own-queue path: create the views/samplers/descriptor sets (bound by the
  // draws recorded below), stamp the content identity, and record the
  // finalized indices so the external pre-pass can un-stamp them if its submit
  // fails.
  this->finalizedIndices_.clear();
  for (const PendingUpload & upload : this->pendingUploads_) {
    if (upload.index >= this->entries_.size()) continue;
    VulkanCachedTexture & texEntry = this->entries_[upload.index];
    if (this->finalizeEntry(texEntry, *upload.texture)) {
      stampTextureContent(texEntry, *upload.texture);
      this->finalizedIndices_.push_back(upload.index);
    }
    else {
      // The entry's image is referenced by the recorded copies, so the
      // half-initialized resources must be destroyed through the deferred
      // ring, not synchronously.  Keys stay unstamped so the next frame
      // retries the upload.
      this->context_.deferDestroyEntry(texEntry);
    }
  }
  this->pendingUploads_.clear();
}

SoVulkan::Result
SoVulkanTextureCache::flushExternal()
{
  if (this->pendingUploads_.empty()) return SoVulkan::Result::ok();

  // One-shot fallback for the external path, used only when
  // beginExternalPrepass() could not allocate its transient command buffer.
  // All pending uploads were staged into the single shared staging pool buffer
  // at their recording offsets, so one submit copies every pending texture.
  if (!SoVulkanShared::withOneShotSubmit(
        this->context_.device, this->context_.queue,
        this->context_.commandPool, this->context_.allocator,
        [this](VkCommandBuffer uploadBuffer) {
          for (const PendingUpload & upload : this->pendingUploads_) {
            if (upload.index >= this->entries_.size()) continue;
            this->recordUpload(uploadBuffer, this->entries_[upload.index],
                               *upload.texture, this->stagingPool_.buffer(),
                               upload.stagingOffset);
          }
        })) {
    // The one-shot submit copies the whole batch, so a failure means no upload
    // in it completed -- every pending entry is half-initialized.  Reset them
    // all so the next frame retries cleanly.
    for (const PendingUpload & upload : this->pendingUploads_) {
      if (upload.index < this->entries_.size()) {
        this->destroyEntry(this->entries_[upload.index]);
      }
    }
    this->pendingUploads_.clear();
    return SoVulkan::Result::error("one-shot texture upload failed");
  }

  // Host-side completion (views/samplers/descriptor sets) and content identity
  // stamping.  The queue is idle here, so the shared staging pool is free to be
  // reused by the next frame.  A failure leaves the content keys unstamped, so
  // the next frame retries the upload.
  for (const PendingUpload & upload : this->pendingUploads_) {
    if (upload.index >= this->entries_.size()) continue;
    VulkanCachedTexture & texEntry = this->entries_[upload.index];
    if (this->finalizeEntry(texEntry, *upload.texture)) {
      stampTextureContent(texEntry, *upload.texture);
    }
    else {
      this->destroyEntry(texEntry);
    }
  }
  this->pendingUploads_.clear();
  return SoVulkan::Result::ok();
}

void
SoVulkanTextureCache::discardPending()
{
  this->pendingUploads_.clear();
}

void
SoVulkanTextureCache::unStampFinalized()
{
  // The copies in the external pre-pass never completed, so the entries
  // finalizePending() stamped still hold their (empty) images.  Un-stamp them
  // so prepareCommand() re-prepares the upload next frame instead of sampling
  // the empty image forever.
  for (const size_t index : this->finalizedIndices_) {
    if (index < this->entries_.size()) {
      this->entries_[index].pixelsKey = nullptr;
    }
  }
  this->finalizedIndices_.clear();
}

VkDescriptorSet
SoVulkanTextureCache::resolve(const SoRenderCommand & command) const
{
  // Fast path for the overwhelmingly common untextured case: most retained
  // commands (default CAD surfaces, edges, points) carry no texture, so fall
  // straight through to the white set without touching the commandToIndex_
  // map (a hash + bucket walk per draw otherwise).
  const SoTextureData & tex = command.material.texture;
  if (!tex.pixels || tex.width == 0 || tex.height == 0 ||
      tex.numComponents == 0) {
    return this->whiteDescriptorSet_;
  }
  const auto found = this->commandToIndex_.find(&command);
  if (found != this->commandToIndex_.end() &&
      this->entries_[found->second].descriptorSet != VK_NULL_HANDLE) {
    return this->entries_[found->second].descriptorSet;
  }
  return this->whiteDescriptorSet_;
}

bool
SoVulkanTextureCache::createWhiteTexture()
{
  const uint8_t white = 255;
  const uint32_t extent = 1;

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.extent = {extent, extent, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (vmaCreateImage(this->context_.vmaAllocator, &ci, &allocInfo,
                     &this->whiteImage_, &this->whiteImageAllocation_,
                     nullptr) != VK_SUCCESS) {
    if (this->context_.emitError) {
      this->context_.emitError("createWhiteTexture: vmaCreateImage failed");
    }
    return false;
  }

  // Stage the 4 white bytes through the shared pool, then copy them with a
  // one-shot submit (which drains the queue before returning).
  VkDeviceSize stagingOffset = 0;
  if (!this->stagingPool_.stage(&white, 4, stagingOffset)) {
    return false;
  }
  if (!SoVulkanShared::withOneShotSubmit(
        this->context_.device, this->context_.queue,
        this->context_.commandPool, this->context_.allocator,
        [this, stagingOffset](VkCommandBuffer uploadBuffer) {
          SoVulkanShared::imageTransition(
            uploadBuffer, this->whiteImage_,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
          VkBufferImageCopy region {};
          region.bufferOffset = stagingOffset;
          region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          region.imageSubresource.layerCount = 1;
          region.imageExtent = {extent, extent, 1};
          vkCmdCopyBufferToImage(uploadBuffer, this->stagingPool_.buffer(),
                                 this->whiteImage_,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                 &region);
          SoVulkanShared::imageTransition(
            uploadBuffer, this->whiteImage_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        })) {
    if (this->context_.emitError) {
      this->context_.emitError("createWhiteTexture: one-shot upload failed");
    }
    return false;
  }

  this->whiteImageView_ =
    createImageView(this->context_.device, this->whiteImage_,
                    VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT,
                    this->context_.allocator);
  if (this->whiteImageView_ == VK_NULL_HANDLE) {
    return false;
  }

  SoTextureData fallback;
  fallback.minFilter = SO_TEXTURE_FILTER_NEAREST;
  fallback.magFilter = SO_TEXTURE_FILTER_NEAREST;
  fallback.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  fallback.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  this->whiteSampler_ = this->context_.samplerCache->get(
    fallback.minFilter, fallback.magFilter, fallback.wrapS, fallback.wrapT);
  if (this->whiteSampler_ == VK_NULL_HANDLE) {
    return false;
  }
  return this->context_.allocateDescriptorSet(
    this->whiteImageView_, this->whiteSampler_, this->whiteDescriptorSet_);
}
