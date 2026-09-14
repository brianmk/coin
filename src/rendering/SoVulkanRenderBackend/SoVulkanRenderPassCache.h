// src/rendering/SoVulkanRenderBackend/SoVulkanRenderPassCache.h
//
// Render-pass and framebuffer cache for the Vulkan render backend.
//
// Vulkan render passes and framebuffers are expensive, immutable objects.  A
// scene is drawn every frame into a swapchain whose images cycle, so this
// cache creates a VkRenderPass per unique attachment identity (color/depth
// format, sample count, image layouts, and the color/depth load ops) and a
// VkFramebuffer per unique target identity (image views + extent + render
// pass).  Pipelines are keyed on the render-pass handle (see PipelineKey), so
// reusing the same pass across targets that differ only in their images keeps
// the pipeline cache warm as well.
//
// This is a private helper of the raster backend (not public API): the owning
// SoVulkanRenderBackend creates it, binds a device/allocator at
// initialize(), and queries the current pass/framebuffer while recording each
// frame.  Deferred destruction (releasing an old framebuffer a few frames
// after the submissions that referenced it have completed) is supplied by the
// backend through setDeferredDestroy() so this class stays free of any
// backend/ring dependencies.

#ifndef COIN_SOVULKANRENDERPASSCACHE_H
#define COIN_SOVULKANRENDERPASSCACHE_H

#include <cstdint>
#include <functional>
#include <unordered_map>

#include <vulkan/vulkan.h>

#include <Inventor/rendering/SoVulkanRenderTarget.h>

class SoVulkanRenderPassCache {
public:
  SoVulkanRenderPassCache();
  ~SoVulkanRenderPassCache();
  SoVulkanRenderPassCache(const SoVulkanRenderPassCache &) = delete;
  SoVulkanRenderPassCache & operator=(const SoVulkanRenderPassCache &) = delete;

  //! Bind the device/allocator used to create and destroy render passes and
  //! framebuffers.  Called once from the owning backend's initialize().
  void setDevice(VkDevice device, const VkAllocationCallbacks * allocator);

  /*!
    \brief Register the deferred-destruction mechanism.

    \a fn queues a GPU resource's destruction a few frames behind the producer
    (the backend's frame ring), so a resource is never destroyed while a
    submission that references it may still be in flight.  When unset, replaced
    framebuffers are freed synchronously instead of deferred.
  */
  void setDeferredDestroy(std::function<void(std::function<void()> &&)> fn);

  /*!
    \brief Recompute + cache the render pass for the target's attachment
    identity and the requested load ops.

    Returns the cached VkRenderPass (also available from
    currentRenderPass()), or VK_NULL_HANDLE when creation fails.
  */
  VkRenderPass getOrCreateRenderPass(const SoVulkanRenderTarget & target,
                                     VkAttachmentLoadOp colorLoadOp,
                                     VkAttachmentLoadOp depthLoadOp);

  /*!
    \brief Ensure the cached framebuffer matches the current target/render
    pass.

    Recreates the framebuffer (deferring the old one) whenever the target's
    image views, extent or render pass change, as swapchain targets cycle
    their images every frame.  The current framebuffer is available from
    framebuffer().
  */
  bool ensureFramebuffer(const SoVulkanRenderTarget * target,
                         VkRenderPass renderPass);

  //! Destroy the cached render passes and framebuffer (shutdown).  Idempotent.
  void destroyAll();

  // --- per-frame state queried by the owning backend ----------------------

  //! Render pass cached for the current frame.
  VkRenderPass currentRenderPass() const { return renderPass_; }

  //! Framebuffer cached for the current target identity.
  VkFramebuffer framebuffer() const { return framebuffer_; }

  //! Whether the current render pass clears the color attachment via its loadOp.
  bool colorClearedByLoad() const { return colorCleared_; }

  //! Whether the current render pass clears the depth attachment via its loadOp.
  bool depthClearedByLoad() const { return depthCleared_; }

  //! Set the current pass's clear-by-load flags (per frame / external pass).
  void setClearedByLoad(bool color, bool depth)
  {
    colorCleared_ = color;
    depthCleared_ = depth;
  }

private:
  // Immutable render-pass identity.  Created once per unique combination and
  // reused across targets that share the definition (so the pipeline cache
  // keyed on the render-pass handle stays warm).
  struct RenderPassIdentity {
    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkImageLayout depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // Load ops distinguish a render pass that clears its attachments at begin
    // (full-target clear fast path) from one that loads them and clears via
    // vkCmdClearAttachments.  Two passes that differ only in loadOp must not
    // share a cache entry.
    VkAttachmentLoadOp colorLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkAttachmentLoadOp depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

    bool operator==(const RenderPassIdentity & other) const
    {
      return colorFormat == other.colorFormat &&
        depthFormat == other.depthFormat &&
        sampleCount == other.sampleCount &&
        colorLayout == other.colorLayout &&
        depthLayout == other.depthLayout &&
        colorLoadOp == other.colorLoadOp &&
        depthLoadOp == other.depthLoadOp;
    }
  };

  struct RenderPassIdentityHash {
    size_t operator()(const RenderPassIdentity & key) const
    {
      size_t hash = std::hash<uint32_t>()(
        static_cast<uint32_t>(key.colorFormat));
      hash = SoVulkanRenderPassCache::hashCombine(
        hash, std::hash<uint32_t>()(static_cast<uint32_t>(key.depthFormat)));
      hash = SoVulkanRenderPassCache::hashCombine(
        hash, std::hash<uint32_t>()(static_cast<uint32_t>(key.sampleCount)));
      hash = SoVulkanRenderPassCache::hashCombine(
        hash, std::hash<uint32_t>()(static_cast<uint32_t>(key.colorLayout)));
      hash = SoVulkanRenderPassCache::hashCombine(
        hash, std::hash<uint32_t>()(static_cast<uint32_t>(key.depthLayout)));
      hash = SoVulkanRenderPassCache::hashCombine(
        hash, std::hash<uint32_t>()(static_cast<uint32_t>(key.colorLoadOp)));
      hash = SoVulkanRenderPassCache::hashCombine(
        hash, std::hash<uint32_t>()(static_cast<uint32_t>(key.depthLoadOp)));
      return hash;
    }
  };
  //! Shared combine step for the hand-rolled hash functors above.
  static size_t hashCombine(size_t hash, size_t value)
  {
    return hash ^ (value + 0x9e3779b9 + (hash << 6) + (hash >> 2));
  }

  //! Derive the immutable render-pass identity from a target's attachments.
  RenderPassIdentity identityFor(const SoVulkanRenderTarget & target) const;
  //! Native Vulkan render-pass creation (no caching).
  bool createRenderPass(const SoVulkanRenderTarget & target,
                        VkAttachmentLoadOp colorLoadOp,
                        VkAttachmentLoadOp depthLoadOp,
                        VkRenderPass & pass);

  VkDevice device_ = VK_NULL_HANDLE;
  const VkAllocationCallbacks * allocator_ = nullptr;
  //! Deferred-release hook (backend frame ring); empty = synchronous free.
  std::function<void(std::function<void()> &&)> deferDestroyFn_;

  std::unordered_map<RenderPassIdentity, VkRenderPass, RenderPassIdentityHash>
    passCache_;
  //! Render pass used by the current frame (looked up from passCache_).
  VkRenderPass renderPass_ = VK_NULL_HANDLE;
  bool colorCleared_ = false;
  bool depthCleared_ = false;

  //! Framebuffer cached for the current target identity (image views +
  //! extent + render pass).
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
  VkRenderPass framebufferPass_ = VK_NULL_HANDLE;
  VkImage framebufferColorImage_ = VK_NULL_HANDLE;
  VkImageView framebufferColorView_ = VK_NULL_HANDLE;
  VkImage framebufferDepthImage_ = VK_NULL_HANDLE;
  VkImageView framebufferDepthView_ = VK_NULL_HANDLE;
  VkExtent2D framebufferExtent_ = {0, 0};
};

#endif // COIN_SOVULKANRENDERPASSCACHE_H
