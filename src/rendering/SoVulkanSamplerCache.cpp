// src/rendering/SoVulkanSamplerCache.cpp
#include "rendering/SoVulkanSamplerCache.h"

namespace {

VkFilter textureFilterToVk(const SoTextureFilter filter)
{
  switch (filter) {
  case SO_TEXTURE_FILTER_NEAREST:
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    return VK_FILTER_NEAREST;
  case SO_TEXTURE_FILTER_LINEAR:
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
  default:
    return VK_FILTER_LINEAR;
  }
}

VkSamplerAddressMode textureWrapToVk(const SoTextureWrap wrap)
{
  switch (wrap) {
  case SO_TEXTURE_WRAP_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case SO_TEXTURE_WRAP_CLAMP_TO_EDGE:
  case SO_TEXTURE_WRAP_CLAMP_TO_BORDER:
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

} // namespace

void
SoVulkanSamplerCache::initialize(VkDevice device,
                                 const VkAllocationCallbacks * allocator)
{
  this->device = device;
  this->allocator = allocator;
}

SoVulkanSamplerCache::Key
SoVulkanSamplerCache::makeKey(SoTextureFilter minFilter,
                              SoTextureFilter magFilter,
                              SoTextureWrap wrapS, SoTextureWrap wrapT)
{
  return static_cast<Key>(
    (static_cast<uint8_t>(minFilter) & 0x3u) |
    ((static_cast<uint8_t>(magFilter) & 0x3u) << 2u) |
    ((static_cast<uint8_t>(wrapS) & 0x3u) << 4u) |
    ((static_cast<uint8_t>(wrapT) & 0x3u) << 6u));
}

bool
SoVulkanSamplerCache::create(SoTextureFilter minFilter,
                             SoTextureFilter magFilter, SoTextureWrap wrapS,
                             SoTextureWrap wrapT, VkSampler & sampler)
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

VkSampler
SoVulkanSamplerCache::get(SoTextureFilter minFilter,
                          SoTextureFilter magFilter, SoTextureWrap wrapS,
                          SoTextureWrap wrapT)
{
  const Key key = makeKey(minFilter, magFilter, wrapS, wrapT);
  const auto found = this->samplers.find(key);
  if (found != this->samplers.end()) {
    return found->second;
  }
  VkSampler sampler = VK_NULL_HANDLE;
  if (this->create(minFilter, magFilter, wrapS, wrapT, sampler)) {
    this->samplers.emplace(key, sampler);
  }
  return sampler;
}

void
SoVulkanSamplerCache::destroyAll()
{
  if (this->device == VK_NULL_HANDLE) {
    this->samplers.clear();
    return;
  }
  for (const auto & entry : this->samplers) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroySampler(this->device, entry.second, this->allocator);
    }
  }
  this->samplers.clear();
}
