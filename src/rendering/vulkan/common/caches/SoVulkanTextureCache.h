// src/rendering/vulkan/common/caches/SoVulkanTextureCache.h
//
// Owns the raster backend's per-command GPU texture entries and the
// staging->image upload path: entry lookup/content-change detection, staging
// through a shared host-visible pool, one-shot or frame-command-buffer copies,
// view/sampler/descriptor finalization, eviction, and the 1x1 white fallback.
//
// Borrows the device/VMA/queue/command-pool and the sampler cache from the
// backend, and calls back into it for the two things it does not own: the
// shared set-1/set-2 descriptor pools (allocate/free) and deferred destruction
// (the submission that may still reference an image/set must drain first).
// This keeps the god class from owning texture state while the descriptor
// pools stay shared with the lighting set.
//
// Each entry owns up to four channels: the base-color texture (descriptor set
// 1) and three optional PBR maps -- roughness, normal and emissive (descriptor
// set 2).  Absent maps bind a per-command default bundle (white roughness,
// flat normal, black emissive) so the shader can always sample all three.
//
// Internal to Coin's Vulkan renderer (not installed, not public API).

#ifndef COIN_SOVULKANTEXTURECACHE_H
#define COIN_SOVULKANTEXTURECACHE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include <vk_mem_alloc.h>

#include <Inventor/rendering/SoRenderIR.h>

#include "rendering/vulkan/common/core/SoVulkanResult.h"
#include "rendering/vulkan/common/caches/SoVulkanSamplerCache.h"
#include "rendering/vulkan/common/memory/SoVulkanStagingPool.h"

// Channel indices within a cached entry: 0 base color, 1 roughness, 2 normal,
// 3 emissive.  The PBR maps are stored in a VulkanCachedMap each.
enum : int {
  VULKAN_TEX_CHANNEL_DIFFUSE = 0,
  VULKAN_TEX_CHANNEL_ROUGHNESS = 1,
  VULKAN_TEX_CHANNEL_NORMAL = 2,
  VULKAN_TEX_CHANNEL_EMISSIVE = 3,
  VULKAN_TEX_CHANNEL_COUNT = 4
};

/*! \brief Cached GPU texture for one optional PBR map channel. */
struct VulkanCachedMap {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  // Shared sampler owned by the sampler cache (not destroyed here).
  VkSampler sampler = VK_NULL_HANDLE;

  // Identity of the last upload.  A null pixelsKey means "no content stamped"
  // (upload not yet known to have completed).
  const unsigned char * pixelsKey = nullptr;
  int width = 0;
  int height = 0;
  int numComponents = 0;
  SoTextureFilter minFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureFilter magFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureWrap wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureWrap wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  // Content hash of the uploaded pixels (sampled): pixel-pointer identity
  // alone cannot detect in-place edits, which would serve stale textures.
  uint64_t contentHash = 0;
};

/*! \brief Cached GPU texture for one retained command's texture bundle. */
struct VulkanCachedTexture {
  VkImage image = VK_NULL_HANDLE;
  // Backing device memory, owned by the VMA allocator.  The image and its
  // allocation are created and destroyed together (vmaCreateImage /
  // vmaDestroyImage); no separate VkDeviceMemory handle is kept.
  VmaAllocation allocation = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  // Pool the descriptor set was allocated from (pools are append-only, so
  // the set must be returned to this pool, not the currently active one).
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

  // Optional secondary PBR maps (roughness/normal/emissive), described by
  // VULKAN_TEX_CHANNEL_* - 1 into this array.  They are bound together through
  // descriptor set 2.
  VulkanCachedMap maps[3];
  VkDescriptorSet mapDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorPool mapDescriptorPool = VK_NULL_HANDLE;

  // Command that last touched this entry (per-frame arena pointer; used
  // only as an identity key for map rebuilds after cache eviction).
  const SoRenderCommand * commandKey = nullptr;

  // Identity of the last upload.
  const unsigned char * pixelsKey = nullptr;
  int width = 0;
  int height = 0;
  int numComponents = 0;
  SoTextureFilter minFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureFilter magFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureWrap wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureWrap wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureModel model = SO_TEXTURE_MODEL_MODULATE;
  uint32_t cacheGeneration = 0;
  // Content hash of the uploaded pixels (sampled): pixel-pointer identity
  // alone cannot detect in-place edits, which would serve stale textures.
  uint64_t contentHash = 0;
};

//! Borrowed handles + backend callbacks the texture cache does not own.
struct SoVulkanTextureCacheContext {
  VkDevice device = VK_NULL_HANDLE;
  VmaAllocator vmaAllocator = nullptr;
  const VkAllocationCallbacks * allocator = nullptr;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  bool sampledR8 = false;
  bool sampledR8G8 = false;
  SoVulkanSamplerCache * samplerCache = nullptr;
  // Allocate/free the shared set-1 descriptor (draw UBO + texture).
  std::function<bool(VkImageView, VkSampler, VkDescriptorSet &)>
    allocateDescriptorSet;
  // Allocate a set-2 descriptor binding the three PBR maps (roughness, normal,
  // emissive) and report the pool it came from so the cache can free it.
  std::function<bool(const VkImageView[3], const VkSampler[3],
                     VkDescriptorSet &, VkDescriptorPool &)>
    allocateMapDescriptorSet;
  std::function<void(VkDescriptorSet, VkDescriptorPool)> freeDescriptorSet;
  // Defer destruction of an entry's image/view/set to the frame's ring.
  std::function<void(VulkanCachedTexture &)> deferDestroyEntry;
  std::function<VkCommandBuffer()> currentCommandBuffer;
  std::function<void(const char *)> emitError;
};

class SoVulkanTextureCache {
public:
  // One texture waiting for its GPU-side upload (staging copy).
  struct PendingUpload {
    size_t index = 0;
    int slot = VULKAN_TEX_CHANNEL_DIFFUSE;
    const SoRenderCommand * command = nullptr;
    const SoTextureData * texture = nullptr;
    VkDeviceSize stagingOffset = 0;
    VkDeviceSize stagingBytes = 0;
  };

  // Create the white fallback texture and the default map bundle.  `context`
  // is retained.
  bool initialize(const SoVulkanTextureCacheContext & context);
  // Destroy every entry, the staging pool and the fallback textures.  The
  // device must still be valid.
  void destroy();

  // Drop every cached entry (scene change / backend re-init).
  void invalidate();

  // Per-command maintenance during the geometry-cache update: look up/create
  // the entry, detect a content change, defer-destroy the stale image and
  // stage a new upload.  Returns true when a new upload was staged.
  bool prepareCommand(const SoRenderCommand & command, uint32_t generation);
  // Evict entries not visited this generation, then re-resolve the pending
  // upload indices against the compacted cache.
  void sweep(uint32_t generation);
  // Rewind the staging pool for a new frame.
  void resetStaging();

  // Resolve the set-1 descriptor (draw UBO + base texture) for a command.
  VkDescriptorSet resolve(const SoRenderCommand & command) const;
  // Resolve the set-2 descriptor (the three PBR maps) for a command, falling
  // back to the default bundle when the command carries no maps.
  VkDescriptorSet resolveMaps(const SoRenderCommand & command) const;
  VkDescriptorSet whiteDescriptorSet() const
  {
    return this->whiteDescriptorSet_;
  }
  VkDescriptorSet defaultMapDescriptorSet() const
  {
    return this->defaultMapDescriptorSet_;
  }

  // Upload plumbing.  recordPendingInto() records the copies into a caller
  // command buffer (must not be inside a render pass); finalizePending()
  // creates the views/samplers/sets and stamps content identity; flushExternal
  // is the one-shot fallback for the external pre-pass.
  void recordPendingInto(VkCommandBuffer commandBuffer);
  bool recordPending();
  void finalizePending();
  SoVulkan::Result flushExternal();
  bool hasPendingUploads() const { return !this->pendingUploads_.empty(); }
  // Drop the pending list without finalizing (aborted frame).
  void discardPending();
  // Un-stamp the entries finalized last frame (external pre-pass failure).
  void unStampFinalized();

  std::vector<VulkanCachedTexture> & entries() { return this->entries_; }
  const std::vector<VulkanCachedTexture> & entries() const
  {
    return this->entries_;
  }

private:
  VulkanCachedTexture & getOrCreate(const SoRenderCommand * command);
  void destroyEntry(VulkanCachedTexture & entry);
  VkFormat effectiveFormat(int numComponents) const;
  bool prepareUpload(VkImage & image, VmaAllocation & allocation,
                     const SoTextureData & texture,
                     VkDeviceSize & stagingOffset, VkDeviceSize & stagingBytes);
  void recordUpload(VkCommandBuffer commandBuffer, VkImage image,
                    const SoTextureData & texture, VkBuffer staging,
                    VkDeviceSize stagingOffset);
  bool finalizeChannel(VulkanCachedTexture & entry, int slot,
                       const SoTextureData & texture);
  // (Re)allocate the entry's set-2 descriptor from its current map views,
  // substituting the default views/samplers for absent maps.  Frees any
  // previous set.
  bool finalizeMapSet(VulkanCachedTexture & entry);
  bool createWhiteTexture();
  bool createDefaultMapImages();
  bool createDefaultMapSet();
  // Create a 1x1 device-local image with the given RGBA and upload it through
  // the shared staging pool (one-shot submit), leaving `view` created.
  bool createSolidImage(const uint8_t rgba[4], VkImage & image,
                        VmaAllocation & allocation, VkImageView & view);

  SoVulkanTextureCacheContext context_;
  std::vector<VulkanCachedTexture> entries_;
  std::unordered_map<const SoRenderCommand *, size_t> commandToIndex_;
  std::vector<PendingUpload> pendingUploads_;
  // Entry indices finalized (content-stamped) by the last finalizePending();
  // the external pre-pass un-stamps them if its submit fails.
  std::vector<size_t> finalizedIndices_;
  SoVulkanStagingPool stagingPool_;

  VkImage whiteImage_ = VK_NULL_HANDLE;
  VmaAllocation whiteImageAllocation_ = nullptr;
  VkImageView whiteImageView_ = VK_NULL_HANDLE;
  VkSampler whiteSampler_ = VK_NULL_HANDLE;
  VkDescriptorSet whiteDescriptorSet_ = VK_NULL_HANDLE;

  // 1x1 defaults for the optional PBR maps: white roughness (no-op multiply),
  // flat normal (0.5,0.5,1.0) and black emissive (no-op add).
  VkImage flatNormalImage_ = VK_NULL_HANDLE;
  VmaAllocation flatNormalImageAllocation_ = nullptr;
  VkImageView flatNormalImageView_ = VK_NULL_HANDLE;
  VkImage blackImage_ = VK_NULL_HANDLE;
  VmaAllocation blackImageAllocation_ = nullptr;
  VkImageView blackImageView_ = VK_NULL_HANDLE;
  VkDescriptorSet defaultMapDescriptorSet_ = VK_NULL_HANDLE;
  VkDescriptorPool defaultMapDescriptorPool_ = VK_NULL_HANDLE;
};

#endif // COIN_SOVULKANTEXTURECACHE_H
