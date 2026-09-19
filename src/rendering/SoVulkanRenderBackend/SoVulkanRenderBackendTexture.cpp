// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendTexture.cpp
//
// The texture-entry cache and the staging->image upload path live in the
// SoVulkanTextureCache collaborator (src/rendering/SoVulkanTextureCache.{h,cpp}).
// What remains here is descriptor-pool growth, which is shared with the
// lighting descriptor set and therefore stays on the backend.

#include "rendering/SoVulkanRenderBackend.h"

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
