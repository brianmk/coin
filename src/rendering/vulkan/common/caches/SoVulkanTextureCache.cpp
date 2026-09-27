// src/rendering/vulkan/common/caches/SoVulkanTextureCache.cpp
#include "rendering/vulkan/common/caches/SoVulkanTextureCache.h"

#include "rendering/backend/SoRenderBackend.h"
#include "rendering/vulkan/raster/SoVulkanRenderBackendP.h"
#include "rendering/vulkan/common/core/SoVulkanShared.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

using namespace CoinVulkanDetail;

namespace {

// A texture channel is usable when it carries a decodable image.
bool
hasUsableTexture(const SoTextureData & texture)
{
  return texture.pixels && texture.width > 0 && texture.height > 0 &&
    texture.numComponents > 0;
}

VkImage &
channelImage(VulkanCachedTexture & entry, const int slot)
{
  return slot == VULKAN_TEX_CHANNEL_DIFFUSE ? entry.image
                                            : entry.maps[slot - 1].image;
}

VmaAllocation &
channelAllocation(VulkanCachedTexture & entry, const int slot)
{
  return slot == VULKAN_TEX_CHANNEL_DIFFUSE ? entry.allocation
                                            : entry.maps[slot - 1].allocation;
}

VkImageView &
channelView(VulkanCachedTexture & entry, const int slot)
{
  return slot == VULKAN_TEX_CHANNEL_DIFFUSE ? entry.view
                                            : entry.maps[slot - 1].view;
}

// Record the identity of an uploaded texture on a cache channel: the pixel
// pointer, dimensions/components and the sampled content hash.
void
stampChannel(VulkanCachedTexture & entry, const int slot,
             const SoTextureData & texture)
{
  if (slot == VULKAN_TEX_CHANNEL_DIFFUSE) {
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
    return;
  }
  VulkanCachedMap & map = entry.maps[slot - 1];
  map.pixelsKey = texture.pixels;
  map.width = texture.width;
  map.height = texture.height;
  map.numComponents = texture.numComponents;
  map.minFilter = texture.minFilter;
  map.magFilter = texture.magFilter;
  map.wrapS = texture.wrapS;
  map.wrapT = texture.wrapT;
  map.contentHash = hashTextureContent(texture);
}

// Clear the content identity stamped by stampChannel() so a subsequent
// prepareCommand() re-stages the upload.
void
clearChannelStamps(VulkanCachedTexture & entry)
{
  entry.pixelsKey = nullptr;
  entry.contentHash = 0;
  for (VulkanCachedMap & map : entry.maps) {
    map.pixelsKey = nullptr;
    map.contentHash = 0;
  }
}

// True when the cached channel still matches the texture it was uploaded from.
// A channel that is absent from the command must also be absent from the cache
// (otherwise the entry is rebuilt to drop the stale image).
bool
channelMatches(const VulkanCachedTexture & entry, const int slot,
               const SoTextureData & texture)
{
  const bool usable = hasUsableTexture(texture);
  const unsigned char * pixelsKey;
  int width, height, numComponents;
  SoTextureFilter minFilter, magFilter;
  SoTextureWrap wrapS, wrapT;
  uint64_t contentHash;
  bool present;
  if (slot == VULKAN_TEX_CHANNEL_DIFFUSE) {
    present = entry.image != VK_NULL_HANDLE;
    pixelsKey = entry.pixelsKey;
    width = entry.width;
    height = entry.height;
    numComponents = entry.numComponents;
    minFilter = entry.minFilter;
    magFilter = entry.magFilter;
    wrapS = entry.wrapS;
    wrapT = entry.wrapT;
    contentHash = entry.contentHash;
  }
  else {
    const VulkanCachedMap & map = entry.maps[slot - 1];
    present = map.image != VK_NULL_HANDLE;
    pixelsKey = map.pixelsKey;
    width = map.width;
    height = map.height;
    numComponents = map.numComponents;
    minFilter = map.minFilter;
    magFilter = map.magFilter;
    wrapS = map.wrapS;
    wrapT = map.wrapT;
    contentHash = map.contentHash;
  }
  if (!usable) {
    return !present;
  }
  if (!present) {
    return false;
  }
  return pixelsKey == texture.pixels && width == texture.width &&
    height == texture.height && numComponents == texture.numComponents &&
    minFilter == texture.minFilter && magFilter == texture.magFilter &&
    wrapS == texture.wrapS && wrapT == texture.wrapT &&
    contentHash == hashTextureContent(texture);
}

bool
entryHasAnyChannel(const VulkanCachedTexture & entry)
{
  if (entry.image != VK_NULL_HANDLE || entry.view != VK_NULL_HANDLE) {
    return true;
  }
  for (const VulkanCachedMap & map : entry.maps) {
    if (map.image != VK_NULL_HANDLE || map.view != VK_NULL_HANDLE) {
      return true;
    }
  }
  return entry.mapDescriptorSet != VK_NULL_HANDLE;
}

} // namespace

bool
SoVulkanTextureCache::initialize(const SoVulkanTextureCacheContext & context)
{
  this->context_ = context;
  this->stagingPool_.initialize(context.device, context.vmaAllocator);
  if (!this->createWhiteTexture()) {
    return false;
  }
  if (!this->createDefaultMapImages()) {
    return false;
  }
  return this->createDefaultMapSet();
}

void
SoVulkanTextureCache::destroy()
{
  this->invalidate();
  this->stagingPool_.destroy();
  this->whiteSampler_ = VK_NULL_HANDLE;
  const auto destroyImage = [this](VkImage & image, VmaAllocation & allocation,
                                   VkImageView & view) {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(this->context_.device, view, this->context_.allocator);
      view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE) {
      vmaDestroyImage(this->context_.vmaAllocator, image, allocation);
      image = VK_NULL_HANDLE;
      allocation = nullptr;
    }
  };
  destroyImage(this->whiteImage_, this->whiteImageAllocation_,
               this->whiteImageView_);
  destroyImage(this->flatNormalImage_, this->flatNormalImageAllocation_,
               this->flatNormalImageView_);
  destroyImage(this->blackImage_, this->blackImageAllocation_,
               this->blackImageView_);
  this->whiteDescriptorSet_ = VK_NULL_HANDLE;
  this->defaultMapDescriptorSet_ = VK_NULL_HANDLE;
  this->defaultMapDescriptorPool_ = VK_NULL_HANDLE;
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
  if (entry.mapDescriptorSet != VK_NULL_HANDLE) {
    if (this->context_.freeDescriptorSet) {
      this->context_.freeDescriptorSet(entry.mapDescriptorSet,
                                       entry.mapDescriptorPool);
    }
    entry.mapDescriptorSet = VK_NULL_HANDLE;
  }
  if (entry.descriptorSet != VK_NULL_HANDLE) {
    if (this->context_.freeDescriptorSet) {
      this->context_.freeDescriptorSet(entry.descriptorSet,
                                       entry.descriptorPool);
    }
    entry.descriptorSet = VK_NULL_HANDLE;
  }
  // Shared samplers owned by the sampler cache; released at destroy().
  entry.sampler = VK_NULL_HANDLE;
  const auto destroyImage = [this](VkImage & image, VmaAllocation & allocation,
                                   VkImageView & view) {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(this->context_.device, view, this->context_.allocator);
      view = VK_NULL_HANDLE;
    }
    if (image != VK_NULL_HANDLE) {
      vmaDestroyImage(this->context_.vmaAllocator, image, allocation);
      image = VK_NULL_HANDLE;
      allocation = nullptr;
    }
  };
  destroyImage(entry.image, entry.allocation, entry.view);
  for (VulkanCachedMap & map : entry.maps) {
    map.sampler = VK_NULL_HANDLE;
    destroyImage(map.image, map.allocation, map.view);
  }
  entry = VulkanCachedTexture();
}

bool
SoVulkanTextureCache::prepareCommand(const SoRenderCommand & command,
                                     const uint32_t generation)
{
  const SoMaterialData & material = command.material;
  const SoTextureData * channels[VULKAN_TEX_CHANNEL_COUNT] = {
    &material.texture,
    &material.roughnessTexture,
    &material.normalTexture,
    &material.emissiveTexture,
  };

  bool anyUsable = false;
  for (int slot = 0; slot < VULKAN_TEX_CHANNEL_COUNT; ++slot) {
    if (hasUsableTexture(*channels[slot])) {
      anyUsable = true;
      break;
    }
  }

  const auto found = this->commandToIndex_.find(&command);
  if (!anyUsable) {
    // Nothing to upload.  If a previous frame cached content for this command
    // (maps/material removed), drop it so the draw falls back to the white /
    // default-map sets.
    if (found != this->commandToIndex_.end()) {
      VulkanCachedTexture & entry = this->entries_[found->second];
      if (entryHasAnyChannel(entry)) {
        this->context_.deferDestroyEntry(entry);
      }
      entry.commandKey = &command;
      entry.cacheGeneration = generation;
    }
    return false;
  }

  VulkanCachedTexture & entry = this->getOrCreate(&command);

  bool changed = false;
  for (int slot = 0; slot < VULKAN_TEX_CHANNEL_COUNT; ++slot) {
    if (!channelMatches(entry, slot, *channels[slot])) {
      changed = true;
      break;
    }
  }
  if (!changed) {
    entry.commandKey = &command;
    entry.cacheGeneration = generation;
    return false;
  }

  // The channel bundle changed: drop the whole entry (base texture + maps +
  // set 1 + set 2) and re-stage every usable channel.  Material changes are
  // infrequent, so rebuilding the bundle is preferable to per-channel
  // bookkeeping.
  this->context_.deferDestroyEntry(entry);

  const size_t entryIndex = this->commandToIndex_[&command];
  // A command that appears twice in one draw list would otherwise prepare two
  // uploads for the same entry (leaking the first image); the first pending
  // upload for this index wins.
  bool alreadyPending = false;
  for (const PendingUpload & prior : this->pendingUploads_) {
    if (prior.index == entryIndex) {
      alreadyPending = true;
      break;
    }
  }

  bool staged = false;
  if (!alreadyPending) {
    for (int slot = 0; slot < VULKAN_TEX_CHANNEL_COUNT; ++slot) {
      const SoTextureData & texture = *channels[slot];
      if (!hasUsableTexture(texture)) {
        continue;
      }
      PendingUpload upload;
      upload.command = &command;
      upload.index = entryIndex;
      upload.slot = slot;
      upload.texture = &texture;
      // Report the attempt even if the staging allocation failed; on failure
      // the channel is reset by prepareUpload() and the content keys stay
      // unstamped so the next frame retries.
      staged = true;
      if (this->prepareUpload(channelImage(entry, slot),
                              channelAllocation(entry, slot), texture,
                              upload.stagingOffset, upload.stagingBytes)) {
        this->pendingUploads_.push_back(upload);
      }
    }
  }
  entry.commandKey = &command;
  entry.cacheGeneration = generation;
  return staged;
}

void
SoVulkanTextureCache::sweep(const uint32_t generation)
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
SoVulkanTextureCache::prepareUpload(VkImage & image, VmaAllocation & allocation,
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
  if (vmaCreateImage(this->context_.vmaAllocator, &ci, &allocInfo, &image,
                     &allocation, nullptr) != VK_SUCCESS) {
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
    if (allocation != nullptr) {
      vmaDestroyImage(this->context_.vmaAllocator, image, allocation);
    }
    image = VK_NULL_HANDLE;
    allocation = nullptr;
    return false;
  }
  stagingBytes = byteSize;
  return true;
}

void
SoVulkanTextureCache::recordUpload(VkCommandBuffer commandBuffer,
                                   const VkImage image,
                                   const SoTextureData & texture,
                                   VkBuffer staging,
                                   VkDeviceSize stagingOffset)
{
  SoVulkanShared::imageTransition(
    commandBuffer, image,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    0, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkBufferImageCopy region {};
  region.bufferOffset = stagingOffset;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {static_cast<uint32_t>(texture.width),
                        static_cast<uint32_t>(texture.height), 1};
  vkCmdCopyBufferToImage(commandBuffer, staging, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  SoVulkanShared::imageTransition(
    commandBuffer, image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

bool
SoVulkanTextureCache::finalizeChannel(VulkanCachedTexture & entry,
                                      const int slot,
                                      const SoTextureData & texture)
{
  // The image format matches what prepareUpload() created (RGB and unsupported
  // R/RG textures are expanded to RGBA there).
  const VkFormat format = this->effectiveFormat(texture.numComponents);
  VkImageView & view = channelView(entry, slot);
  view = createImageView(this->context_.device, channelImage(entry, slot),
                         format, VK_IMAGE_ASPECT_COLOR_BIT,
                         this->context_.allocator);
  if (view == VK_NULL_HANDLE) {
    if (this->context_.emitError) {
      this->context_.emitError(
        "finalizeTexture: view creation failed");
    }
    return false;
  }
  const VkSampler sampler = this->context_.samplerCache->get(
    texture.minFilter, texture.magFilter, texture.wrapS, texture.wrapT);
  if (sampler == VK_NULL_HANDLE) {
    if (this->context_.emitError) {
      this->context_.emitError("finalizeTexture: sampler creation failed");
    }
    return false;
  }
  if (slot == VULKAN_TEX_CHANNEL_DIFFUSE) {
    entry.sampler = sampler;
    if (!this->context_.allocateDescriptorSet(view, sampler,
                                              entry.descriptorSet)) {
      if (this->context_.emitError) {
        this->context_.emitError(
          "finalizeTexture: descriptor creation failed");
      }
      return false;
    }
  }
  else {
    entry.maps[slot - 1].sampler = sampler;
  }
  stampChannel(entry, slot, texture);
  return true;
}

bool
SoVulkanTextureCache::finalizeMapSet(VulkanCachedTexture & entry)
{
  bool anyMap = false;
  for (const VulkanCachedMap & map : entry.maps) {
    if (map.view != VK_NULL_HANDLE) {
      anyMap = true;
      break;
    }
  }
  if (!anyMap) {
    // No PBR maps on this command: release any previous set and let
    // resolveMaps() fall back to the shared default bundle.
    if (entry.mapDescriptorSet != VK_NULL_HANDLE) {
      if (this->context_.freeDescriptorSet) {
        this->context_.freeDescriptorSet(entry.mapDescriptorSet,
                                         entry.mapDescriptorPool);
      }
      entry.mapDescriptorSet = VK_NULL_HANDLE;
      entry.mapDescriptorPool = VK_NULL_HANDLE;
    }
    return true;
  }

  const VkImageView views[3] = {
    entry.maps[0].view ? entry.maps[0].view : this->whiteImageView_,
    entry.maps[1].view ? entry.maps[1].view : this->flatNormalImageView_,
    entry.maps[2].view ? entry.maps[2].view : this->blackImageView_,
  };
  const VkSampler samplers[3] = {
    entry.maps[0].sampler ? entry.maps[0].sampler : this->whiteSampler_,
    entry.maps[1].sampler ? entry.maps[1].sampler : this->whiteSampler_,
    entry.maps[2].sampler ? entry.maps[2].sampler : this->whiteSampler_,
  };

  if (entry.mapDescriptorSet != VK_NULL_HANDLE) {
    if (this->context_.freeDescriptorSet) {
      this->context_.freeDescriptorSet(entry.mapDescriptorSet,
                                       entry.mapDescriptorPool);
    }
    entry.mapDescriptorSet = VK_NULL_HANDLE;
    entry.mapDescriptorPool = VK_NULL_HANDLE;
  }
  if (!this->context_.allocateMapDescriptorSet(views, samplers,
                                               entry.mapDescriptorSet,
                                               entry.mapDescriptorPool)) {
    if (this->context_.emitError) {
      this->context_.emitError("finalizeMapSet: descriptor creation failed");
    }
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
    this->recordUpload(commandBuffer,
                       channelImage(this->entries_[upload.index], upload.slot),
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
  // draws recorded below), build the per-entry map bundle, stamp the content
  // identity, and record the finalized indices so the external pre-pass can
  // un-stamp them if its submit fails.
  std::vector<size_t> touched;
  std::vector<size_t> failed;
  for (const PendingUpload & upload : this->pendingUploads_) {
    if (upload.index >= this->entries_.size()) continue;
    VulkanCachedTexture & entry = this->entries_[upload.index];
    if (!this->finalizeChannel(entry, upload.slot, *upload.texture)) {
      failed.push_back(upload.index);
      continue;
    }
    if (std::find(touched.begin(), touched.end(), upload.index) ==
        touched.end()) {
      touched.push_back(upload.index);
    }
  }
  // The image of a half-initialized entry is referenced by the recorded
  // copies, so destroy it through the deferred ring, not synchronously.
  for (const size_t index : failed) {
    if (index < this->entries_.size()) {
      this->context_.deferDestroyEntry(this->entries_[index]);
    }
  }
  this->finalizedIndices_.clear();
  for (const size_t index : touched) {
    if (std::find(failed.begin(), failed.end(), index) != failed.end()) {
      continue;
    }
    VulkanCachedTexture & entry = this->entries_[index];
    if (this->finalizeMapSet(entry)) {
      this->finalizedIndices_.push_back(index);
    }
    else {
      clearChannelStamps(entry);
      this->context_.deferDestroyEntry(entry);
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
  const SoVulkan::Result uploadResult = SoVulkanShared::withOneShotSubmit(
        this->context_.device, this->context_.queue,
        this->context_.commandPool, this->context_.allocator,
        [this](VkCommandBuffer uploadBuffer) {
          for (const PendingUpload & upload : this->pendingUploads_) {
            if (upload.index >= this->entries_.size()) continue;
            this->recordUpload(
              uploadBuffer,
              channelImage(this->entries_[upload.index], upload.slot),
              *upload.texture, this->stagingPool_.buffer(),
              upload.stagingOffset);
          }
        });
  if (!uploadResult.isOk()) {
    // The one-shot submit copies the whole batch, so a failure means no upload
    // in it completed -- every pending entry is half-initialized.  Reset them
    // all so the next frame retries cleanly.
    for (const PendingUpload & upload : this->pendingUploads_) {
      if (upload.index < this->entries_.size()) {
        this->destroyEntry(this->entries_[upload.index]);
      }
    }
    this->pendingUploads_.clear();
    return SoVulkan::Result::error(
      "one-shot texture upload failed: " + uploadResult.message());
  }

  // Host-side completion (views/samplers/descriptor sets) and content identity
  // stamping.  The queue is idle here, so the shared staging pool is free to be
  // reused by the next frame.  A failure leaves the content keys unstamped, so
  // the next frame retries the upload.
  std::vector<size_t> touched;
  std::vector<size_t> failed;
  for (const PendingUpload & upload : this->pendingUploads_) {
    if (upload.index >= this->entries_.size()) continue;
    VulkanCachedTexture & entry = this->entries_[upload.index];
    if (!this->finalizeChannel(entry, upload.slot, *upload.texture)) {
      failed.push_back(upload.index);
      continue;
    }
    if (std::find(touched.begin(), touched.end(), upload.index) ==
        touched.end()) {
      touched.push_back(upload.index);
    }
  }
  for (const size_t index : failed) {
    if (index < this->entries_.size()) {
      this->destroyEntry(this->entries_[index]);
    }
  }
  for (const size_t index : touched) {
    if (std::find(failed.begin(), failed.end(), index) != failed.end()) {
      continue;
    }
    VulkanCachedTexture & entry = this->entries_[index];
    if (!this->finalizeMapSet(entry)) {
      clearChannelStamps(entry);
      this->destroyEntry(entry);
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
      clearChannelStamps(this->entries_[index]);
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

VkDescriptorSet
SoVulkanTextureCache::resolveMaps(const SoRenderCommand & command) const
{
  const SoMaterialData & material = command.material;
  if (!hasUsableTexture(material.roughnessTexture) &&
      !hasUsableTexture(material.normalTexture) &&
      !hasUsableTexture(material.emissiveTexture)) {
    return this->defaultMapDescriptorSet_;
  }
  const auto found = this->commandToIndex_.find(&command);
  if (found != this->commandToIndex_.end() &&
      this->entries_[found->second].mapDescriptorSet != VK_NULL_HANDLE) {
    return this->entries_[found->second].mapDescriptorSet;
  }
  return this->defaultMapDescriptorSet_;
}

bool
SoVulkanTextureCache::createSolidImage(const uint8_t rgba[4],
                                       VkImage & image,
                                       VmaAllocation & allocation,
                                       VkImageView & view)
{
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
  if (vmaCreateImage(this->context_.vmaAllocator, &ci, &allocInfo, &image,
                     &allocation, nullptr) != VK_SUCCESS) {
    if (this->context_.emitError) {
      this->context_.emitError("createSolidImage: vmaCreateImage failed");
    }
    return false;
  }

  // Stage the 4 bytes through the shared pool, then copy them with a one-shot
  // submit (which drains the queue before returning).
  VkDeviceSize stagingOffset = 0;
  if (!this->stagingPool_.stage(rgba, 4, stagingOffset)) {
    return false;
  }
  const SoVulkan::Result uploadResult = SoVulkanShared::withOneShotSubmit(
        this->context_.device, this->context_.queue,
        this->context_.commandPool, this->context_.allocator,
        [this, &image, stagingOffset](VkCommandBuffer uploadBuffer) {
          SoVulkanShared::imageTransition(
            uploadBuffer, image,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
          VkBufferImageCopy region {};
          region.bufferOffset = stagingOffset;
          region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          region.imageSubresource.layerCount = 1;
          region.imageExtent = {extent, extent, 1};
          vkCmdCopyBufferToImage(uploadBuffer, this->stagingPool_.buffer(),
                                 image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                 &region);
          SoVulkanShared::imageTransition(
            uploadBuffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        });
  if (!uploadResult.isOk()) {
    if (this->context_.emitError) {
      this->context_.emitError(
        ("createSolidImage: one-shot upload failed: "
         + uploadResult.message()).c_str());
    }
    return false;
  }

  view = createImageView(this->context_.device, image,
                         VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT,
                         this->context_.allocator);
  return view != VK_NULL_HANDLE;
}

bool
SoVulkanTextureCache::createWhiteTexture()
{
  const uint8_t white[4] = {255, 255, 255, 255};
  if (!this->createSolidImage(white, this->whiteImage_,
                              this->whiteImageAllocation_,
                              this->whiteImageView_)) {
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

bool
SoVulkanTextureCache::createDefaultMapImages()
{
  // Flat normal as an 8-bit RGB tangent-space normal: (0.5, 0.5, 1.0).
  const uint8_t flatNormal[4] = {128, 128, 255, 255};
  if (!this->createSolidImage(flatNormal, this->flatNormalImage_,
                              this->flatNormalImageAllocation_,
                              this->flatNormalImageView_)) {
    return false;
  }
  const uint8_t black[4] = {0, 0, 0, 255};
  return this->createSolidImage(black, this->blackImage_,
                                this->blackImageAllocation_,
                                this->blackImageView_);
}

bool
SoVulkanTextureCache::createDefaultMapSet()
{
  const VkImageView views[3] = {
    this->whiteImageView_,
    this->flatNormalImageView_,
    this->blackImageView_,
  };
  const VkSampler samplers[3] = {
    this->whiteSampler_,
    this->whiteSampler_,
    this->whiteSampler_,
  };
  return this->context_.allocateMapDescriptorSet(views, samplers,
                                                 this->defaultMapDescriptorSet_,
                                                 this->defaultMapDescriptorPool_);
}
