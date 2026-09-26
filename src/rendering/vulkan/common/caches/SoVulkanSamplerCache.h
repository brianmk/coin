// src/rendering/vulkan/common/caches/SoVulkanSamplerCache.h
//
// Caches VkSampler objects by their packed filter/wrap state so textures that
// share sampling state reuse one sampler instead of creating one per texture
// entry.  Internal to Coin (not installed, not public API).
//
// The cache borrows the device and allocation callbacks: the owner must call
// destroyAll() while the device is still valid, before it is destroyed.
//
// This is the first collaborator extracted from SoVulkanRenderBackend (see the
// backend decomposition in docs/vulkan/ARCHITECTURE.md): it owns sampler state
// and its lifecycle, and exposes a narrow get()/destroyAll() contract instead
// of leaving that state entangled with the backend's texture cache.

#ifndef COIN_SOVULKANSAMPLERCACHE_H
#define COIN_SOVULKANSAMPLERCACHE_H

#include <Inventor/rendering/SoRenderIR.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_map>

class SoVulkanSamplerCache {
public:
  SoVulkanSamplerCache() = default;
  SoVulkanSamplerCache(const SoVulkanSamplerCache &) = delete;
  SoVulkanSamplerCache & operator=(const SoVulkanSamplerCache &) = delete;

  //! Bind the borrowed device and allocator.  Safe to call again after a
  //! device change (the owner must destroyAll() first).
  void initialize(VkDevice device, const VkAllocationCallbacks * allocator);

  //! Cached sampler for the given state, created on first use.  Returns
  //! VK_NULL_HANDLE when creation fails; the caller keeps its fallback path.
  VkSampler get(SoTextureFilter minFilter, SoTextureFilter magFilter,
                SoTextureWrap wrapS, SoTextureWrap wrapT);

  //! Destroy every cached sampler.  The device must still be valid.
  void destroyAll();

private:
  // Packed sampler-state key: min | mag << 2 | wrapS << 4 | wrapT << 6.
  typedef uint8_t Key;
  static Key makeKey(SoTextureFilter minFilter, SoTextureFilter magFilter,
                     SoTextureWrap wrapS, SoTextureWrap wrapT);
  bool create(SoTextureFilter minFilter, SoTextureFilter magFilter,
              SoTextureWrap wrapS, SoTextureWrap wrapT, VkSampler & sampler);

  VkDevice device = VK_NULL_HANDLE;
  const VkAllocationCallbacks * allocator = nullptr;
  std::unordered_map<Key, VkSampler> samplers;
};

#endif // COIN_SOVULKANSAMPLERCACHE_H
