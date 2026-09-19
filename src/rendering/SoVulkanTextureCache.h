// src/rendering/SoVulkanTextureCache.h
//
// Owns the raster backend's per-command GPU texture entries and the
// staging->image upload path: entry lookup/content-change detection, staging
// through a shared host-visible pool, one-shot or frame-command-buffer copies,
// view/sampler/descriptor finalization, eviction, and the 1x1 white fallback.
//
// Borrows the device/VMA/queue/command-pool and the sampler cache from the
// backend, and calls back into it for the two things it does not own: the
// shared set-1 descriptor pool (allocate/free) and deferred destruction (the
// submission that may still reference an image/set must drain first).  This
// keeps the god class from owning texture state while the descriptor pool
// stays shared with the lighting set.
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

#include "rendering/SoVulkanResult.h"
#include "rendering/SoVulkanSamplerCache.h"
#include "rendering/SoVulkanStagingPool.h"

/*! \brief Cached GPU texture for one retained command's SoTextureData. */
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
    const SoRenderCommand * command = nullptr;
    const SoTextureData * texture = nullptr;
    VkDeviceSize stagingOffset = 0;
    VkDeviceSize stagingBytes = 0;
  };

  // Create the white fallback texture.  `context` is retained.
  bool initialize(const SoVulkanTextureCacheContext & context);
  // Destroy every entry, the staging pool and the white fallback.  The device
  // must still be valid.
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

  VkDescriptorSet resolve(const SoRenderCommand & command) const;
  VkDescriptorSet whiteDescriptorSet() const
  {
    return this->whiteDescriptorSet_;
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
  bool prepareUpload(VulkanCachedTexture & entry, const SoTextureData & texture,
                     VkDeviceSize & stagingOffset, VkDeviceSize & stagingBytes);
  void recordUpload(VkCommandBuffer commandBuffer,
                    const VulkanCachedTexture & entry,
                    const SoTextureData & texture, VkBuffer staging,
                    VkDeviceSize stagingOffset);
  bool finalizeEntry(VulkanCachedTexture & entry,
                     const SoTextureData & texture);
  bool createWhiteTexture();

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
};

#endif // COIN_SOVULKANTEXTURECACHE_H
