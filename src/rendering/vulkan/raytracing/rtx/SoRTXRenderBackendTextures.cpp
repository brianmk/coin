// src/rendering/vulkan/raytracing/rtx/SoRTXRenderBackendTextures.cpp

// Split from the original monolithic SoRTXRenderBackend.cpp.  Contains the
// member functions for the "Material textures" concern of the Vulkan RTX
// backend: the per-triangle UV pool and the sampler2DArray that holds every
// distinct material texture.  The raster backend's SoVulkanTextureCache is the
// eventual home for this storage; v1 keeps a self-contained store here so the
// path tracer can sample the same embedded textures the raster backend does.

#include "rendering/vulkan/raytracing/rtx/SoRTXRenderBackend.h"
#include "rendering/vulkan/common/core/SoVulkanShared.h"
#include "rendering/vulkan/common/core/SoVulkanDebugUtils.h"
#include <Inventor/errors/SoDebugError.h>
#include <rendering/vulkan/raytracing/rtx/SoRTXRenderBackendP.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include "vk_mem_alloc.h"

using namespace SoRTXBackend;

namespace {

constexpr uint32_t RT_TEXTURE_MAX_EXTENT = 2048;
constexpr uint64_t FNV_OFFSET = 1469598103934665603ull;
constexpr uint64_t FNV_PRIME = 1099511628211ull;

uint64_t fnvBytes(const unsigned char * data, size_t count, uint64_t h)
{
  for (size_t i = 0; i < count; ++i) {
    h ^= static_cast<uint64_t>(data[i]);
    h *= FNV_PRIME;
  }
  return h;
}

uint64_t fnvValue(uint64_t value, uint64_t h)
{
  return fnvBytes(reinterpret_cast<const unsigned char *>(&value), sizeof(value), h);
}

uint32_t nextPowerOfTwo(uint32_t value)
{
  uint32_t p = 1;
  while (p < value && p < RT_TEXTURE_MAX_EXTENT) {
    p <<= 1;
  }
  return p;
}

bool texturePresent(const SoTextureData & texture)
{
  return texture.pixels && texture.width > 0 && texture.height > 0 &&
    texture.numComponents > 0;
}

}  // namespace

bool
SoRTXRenderBackend::ensureUvPoolCapacity(VkDeviceSize bytes)
{
  return this->ensurePoolCapacity(bytes, this->uvPoolBuffer,
                                  this->uvPoolMemory, this->uvPoolMapped,
                                  this->uvPoolCapacity, this->uvPoolUsed, true);
}

bool
SoRTXRenderBackend::ensureTangentPoolCapacity(VkDeviceSize bytes)
{
  return this->ensurePoolCapacity(bytes, this->tangentPoolBuffer,
                                  this->tangentPoolMemory,
                                  this->tangentPoolMapped,
                                  this->tangentPoolCapacity,
                                  this->tangentPoolUsed, true);
}

VkDeviceSize
SoRTXRenderBackend::appendTriangleUvs(const SoRenderCommand & command,
                                      RTXCachedGeometry & entry)
{
  const SoGeometryDesc & geometry = command.geometry;
  const bool indexed = entry.indexCount > 0 && entry.idxKey != nullptr;
  const uint32_t triangleCount =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (triangleCount == 0 || geometry.texcoords == nullptr) {
    entry.uvCount = 0;
    return 0;
  }

  const uint32_t texcoordStrideFloats =
    (geometry.texcoordStride ? geometry.texcoordStride : sizeof(float) * 4) /
    sizeof(float);
  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(triangleCount) * 3 * 4 * sizeof(float);

  const uint32_t existingOffset = entry.uvPoolOffset;
  const bool reuse = existingOffset != 0xFFFFFFFFu &&
    entry.uvCount == triangleCount &&
    (static_cast<VkDeviceSize>(existingOffset) * 16 + bytes) <= this->uvPoolUsed;
  if (!reuse) {
    if (!this->ensureUvPoolCapacity(this->uvPoolUsed + bytes) ||
        !this->ensureTangentPoolCapacity(this->tangentPoolUsed + bytes)) {
      this->emitError("appendTriangleUvs: pool allocation failed");
      return 0;
    }
    entry.uvPoolOffset =
      static_cast<uint32_t>(this->uvPoolUsed / (4 * sizeof(float)));
    // The UV and tangent pools are appended in lockstep, so the shader indexes
    // both with the same RTMaterial::textureData.x offset.
    this->uvPoolUsed += bytes;
    this->tangentPoolUsed += bytes;
  }
  entry.uvCount = triangleCount;

  float * out = static_cast<float *>(this->uvPoolMapped) +
    static_cast<size_t>(entry.uvPoolOffset) * 4;
  for (uint32_t t = 0; t < triangleCount; ++t) {
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t vertexIndex = indexed
        ? geometry.indices[static_cast<size_t>(t) * 3 + k]
        : t * 3 + k;
      const float * tc = geometry.texcoords +
        static_cast<size_t>(vertexIndex) * texcoordStrideFloats;
      float * dst = out + (static_cast<size_t>(t) * 3 + k) * 4;
      dst[0] = tc[0];
      dst[1] = tc[1];
      dst[2] = 0.0f;
      dst[3] = 0.0f;
    }
  }

  // Per-triangle face tangent (from positions + UVs) for tangent-space normal
  // mapping, written to the tangent pool at the same offset.  A flat tangent is
  // stored for all three vertices; w carries the bitangent sign.  A mesh with
  // no positions (or a degenerate UV mapping) gets a zero tangent and the
  // shader keeps the geometric normal.
  float * tout = static_cast<float *>(this->tangentPoolMapped) +
    static_cast<size_t>(entry.uvPoolOffset) * 4;
  const uint32_t posStrideFloats =
    (entry.vertexStride ? entry.vertexStride : sizeof(float) * 3) / sizeof(float);
  const auto position = [&geometry, posStrideFloats](uint32_t i) {
    return geometry.positions + static_cast<size_t>(i) * posStrideFloats;
  };
  for (uint32_t t = 0; t < triangleCount; ++t) {
    const uint32_t i0 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 0]
                                : t * 3 + 0;
    const uint32_t i1 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 1]
                                : t * 3 + 1;
    const uint32_t i2 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 2]
                                : t * 3 + 2;
    float tx = 0.0f, ty = 0.0f, tz = 0.0f, sign = 1.0f;
    if (geometry.positions != nullptr) {
      const float * p0 = position(i0);
      const float * p1 = position(i1);
      const float * p2 = position(i2);
      const float * tc0 = geometry.texcoords +
        static_cast<size_t>(i0) * texcoordStrideFloats;
      const float * tc1 = geometry.texcoords +
        static_cast<size_t>(i1) * texcoordStrideFloats;
      const float * tc2 = geometry.texcoords +
        static_cast<size_t>(i2) * texcoordStrideFloats;
      const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
      const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      const float du1 = tc1[0] - tc0[0];
      const float dv1 = tc1[1] - tc0[1];
      const float du2 = tc2[0] - tc0[0];
      const float dv2 = tc2[1] - tc0[1];
      const float det = du1 * dv2 - du2 * dv1;
      // Face normal = cross(e1, e2); used for the bitangent sign and as the
      // fallback basis when the UV mapping is degenerate on this triangle.
      float nx = e1[1] * e2[2] - e1[2] * e2[1];
      float ny = e1[2] * e2[0] - e1[0] * e2[2];
      float nz = e1[0] * e2[1] - e1[1] * e2[0];
      const float nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (nlen > 1e-12f) {
        nx /= nlen; ny /= nlen; nz /= nlen;
      }
      if (std::fabs(det) > 1e-12f) {
        const float r = 1.0f / det;
        tx = (e1[0] * dv2 - e2[0] * dv1) * r;
        ty = (e1[1] * dv2 - e2[1] * dv1) * r;
        tz = (e1[2] * dv2 - e2[2] * dv1) * r;
        const float bx = (e2[0] * du1 - e1[0] * du2) * r;
        const float by = (e2[1] * du1 - e1[1] * du2) * r;
        const float bz = (e2[2] * du1 - e1[2] * du2) * r;
        // bitangent sign = sign(dot(cross(N, T), B)).
        const float cx = ny * tz - nz * ty;
        const float cy = nz * tx - nx * tz;
        const float cz = nx * ty - ny * tx;
        sign = (cx * bx + cy * by + cz * bz) < 0.0f ? -1.0f : 1.0f;
        const float len = std::sqrt(tx * tx + ty * ty + tz * tz);
        if (len > 1e-12f) {
          tx /= len; ty /= len; tz /= len;
        }
        else {
          tx = ty = tz = 0.0f;
        }
      }
      // Degenerate UVs (e.g. a planar texture projection on faces parallel to
      // the projection axis): the UV gradient is unusable, so build an
      // arbitrary but valid tangent perpendicular to the face normal.  This
      // keeps the TBN well-defined so a normal map still perturbs the shading.
      if (tx == 0.0f && ty == 0.0f && tz == 0.0f && nlen > 1e-12f) {
        if (std::fabs(nz) < 0.9f) {
          tx = -ny; ty = nx; tz = 0.0f;
        }
        else {
          tx = 0.0f; ty = -nz; tz = ny;
        }
        const float len = std::sqrt(tx * tx + ty * ty + tz * tz);
        if (len > 1e-12f) {
          tx /= len; ty /= len; tz /= len;
        }
        sign = 1.0f;
      }
    }
    for (uint32_t k = 0; k < 3; ++k) {
      float * dst = tout + (static_cast<size_t>(t) * 3 + k) * 4;
      dst[0] = tx; dst[1] = ty; dst[2] = tz; dst[3] = sign;
    }
  }
  return bytes;
}

void
SoRTXRenderBackend::destroyTextureArray()
{
  if (this->textureArrayView != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, this->textureArrayView, this->allocator);
    this->textureArrayView = VK_NULL_HANDLE;
  }
  if (this->textureArrayImage != VK_NULL_HANDLE) {
    vmaDestroyImage(this->vmaAllocator, this->textureArrayImage,
                    this->textureArrayMemory);
    this->textureArrayImage = VK_NULL_HANDLE;
    this->textureArrayMemory = nullptr;
  }
  if (this->textureArraySampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->textureArraySampler, this->allocator);
    this->textureArraySampler = VK_NULL_HANDLE;
  }
  this->textureArrayLayers = 0;
  this->textureArrayExtent = 0;
  this->textureSetHash = 0;
  this->textureArrayCommandCount = 0;
}

bool
SoRTXRenderBackend::ensureTextureArray(uint32_t layers, uint32_t extent)
{
  if (layers == 0) layers = 1;
  if (extent == 0) extent = 1;
  if (this->textureArrayImage != VK_NULL_HANDLE &&
      this->textureArrayLayers == layers &&
      this->textureArrayExtent == extent) {
    return true;
  }

  if (this->textureArrayImage != VK_NULL_HANDLE) {
    VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkImage image = this->textureArrayImage;
    const VkImageView view = this->textureArrayView;
    const VmaAllocation memory = this->textureArrayMemory;
    VmaAllocator vma = this->vmaAllocator;
    this->deferDestroy([vma, device, allocator, image, view, memory]() {
      vkDestroyImageView(device, view, allocator);
      vmaDestroyImage(vma, image, memory);
    });
    this->textureArrayImage = VK_NULL_HANDLE;
    this->textureArrayView = VK_NULL_HANDLE;
    this->textureArrayMemory = nullptr;
  }

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.extent = {extent, extent, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = layers;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (vmaCreateImage(this->vmaAllocator, &ci, &allocInfo,
                     &this->textureArrayImage, &this->textureArrayMemory,
                     nullptr) != VK_SUCCESS) {
    this->textureArrayImage = VK_NULL_HANDLE;
    this->textureArrayMemory = nullptr;
    return false;
  }

  VkImageViewCreateInfo vci {};
  vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vci.image = this->textureArrayImage;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  vci.subresourceRange.layerCount = layers;
  vci.subresourceRange.levelCount = 1;
  if (vkCreateImageView(this->device, &vci, this->allocator,
                        &this->textureArrayView) != VK_SUCCESS) {
    vmaDestroyImage(this->vmaAllocator, this->textureArrayImage,
                    this->textureArrayMemory);
    this->textureArrayImage = VK_NULL_HANDLE;
    this->textureArrayMemory = nullptr;
    return false;
  }

  if (this->textureArraySampler == VK_NULL_HANDLE) {
    VkSamplerCreateInfo sci {};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(this->device, &sci, this->allocator,
                        &this->textureArraySampler) != VK_SUCCESS) {
      this->textureArraySampler = VK_NULL_HANDLE;
      return false;
    }
  }

  this->textureArrayLayers = layers;
  this->textureArrayExtent = extent;

  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_IMAGE,
    reinterpret_cast<uint64_t>(this->textureArrayImage), "RT material textures");
  SoVulkanDebugUtils::nameObject(
    this->device, VK_OBJECT_TYPE_IMAGE_VIEW,
    reinterpret_cast<uint64_t>(this->textureArrayView),
    "RT material textures view");
  return true;
}

bool
SoRTXRenderBackend::uploadMaterialTextures(const SoDrawList & drawlist,
                                           VkCommandBuffer cmd)
{
  const uint32_t commandCount = drawlist.getNumCommands();

  // Resolve the base-colour texture of every command to a texture-array layer
  // (content-deduplicated) and build a scene signature over the resolved
  // texture contents.  The IR copies texture pixels into a per-frame arena, so
  // a pointer key would force a re-upload every frame; the content hash keeps
  // an unchanged scene cached while still detecting an actual texture edit.
  struct LayerSource {
    const SoTextureData * texture;
    uint64_t contentHash;
  };
  std::vector<LayerSource> layers;
  std::unordered_map<uint64_t, uint32_t> byContent;
  std::vector<int32_t> commandLayers(static_cast<size_t>(commandCount) * 4, -1);

  uint64_t signature = FNV_OFFSET;
  signature = fnvValue(commandCount, signature);

  // Resolve one embedded texture to a texture-array layer, content-deduplicated.
  // The IR copies texture pixels into a per-frame arena, so a pointer key would
  // force a re-upload every frame; the content hash keeps an unchanged scene
  // cached while still detecting an actual texture edit.
  const auto resolveLayer = [&](const SoTextureData & texture) -> int32_t {
    if (!texturePresent(texture)) {
      return -1;
    }
    const size_t byteCount = static_cast<size_t>(texture.width) *
      static_cast<size_t>(texture.height) *
      static_cast<size_t>(texture.numComponents);
    const uint64_t contentHash =
      fnvBytes(texture.pixels, byteCount, FNV_OFFSET);
    uint64_t key = contentHash;
    key = fnvValue(static_cast<uint64_t>(texture.width), key);
    key = fnvValue(static_cast<uint64_t>(texture.height), key);
    key = fnvValue(static_cast<uint64_t>(texture.numComponents), key);

    uint32_t layer;
    const auto found = byContent.find(key);
    if (found == byContent.end()) {
      layer = static_cast<uint32_t>(layers.size());
      byContent.emplace(key, layer);
      layers.push_back({&texture, contentHash});
    }
    else {
      layer = found->second;
    }
    signature = fnvValue(contentHash, signature);
    signature = fnvValue(static_cast<uint64_t>(texture.width), signature);
    signature = fnvValue(static_cast<uint64_t>(texture.height), signature);
    signature = fnvValue(static_cast<uint64_t>(texture.numComponents), signature);
    return static_cast<int32_t>(layer);
  };

  for (uint32_t i = 0; i < commandCount; ++i) {
    const SoMaterialData & material = drawlist.getCommand(i).material;
    commandLayers[static_cast<size_t>(i) * 4 + 0] =
      resolveLayer(material.texture);
    commandLayers[static_cast<size_t>(i) * 4 + 1] =
      resolveLayer(material.roughnessTexture);
    commandLayers[static_cast<size_t>(i) * 4 + 2] =
      resolveLayer(material.normalTexture);
    commandLayers[static_cast<size_t>(i) * 4 + 3] =
      resolveLayer(material.emissiveTexture);
    // Mix the resolved channels so a scene that only reorders which map sits in
    // which channel is not mistaken for the cached layout.
    for (int channel = 0; channel < 4; ++channel) {
      signature = fnvValue(
        static_cast<uint64_t>(
          commandLayers[static_cast<size_t>(i) * 4 + channel] + 1),
        signature);
    }
  }

  this->commandTextureLayers.swap(commandLayers);
  this->textureArrayCommandCount = commandCount;

  if (signature == this->textureSetHash &&
      this->textureArrayImage != VK_NULL_HANDLE &&
      this->textureArrayLayers >= (layers.empty() ? 1u : layers.size())) {
    return true;
  }

  // Pick a common square extent (power of two) for every layer.
  uint32_t extent = 1;
  for (const LayerSource & source : layers) {
    extent = std::max(extent,
      static_cast<uint32_t>(std::max(source.texture->width,
                                     source.texture->height)));
  }
  extent = std::min(nextPowerOfTwo(extent), RT_TEXTURE_MAX_EXTENT);
  if (extent == 0) extent = 1;
  const uint32_t layerCount =
    layers.empty() ? 1u : static_cast<uint32_t>(layers.size());

  if (!this->ensureTextureArray(layerCount, extent)) {
    this->emitError("uploadMaterialTextures: texture array allocation failed");
    return false;
  }

  const VkDeviceSize layerBytes =
    static_cast<VkDeviceSize>(extent) * extent * 4;
  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingMemory = nullptr;
  void * stagingMapped = nullptr;
  if (!this->createHostVisibleBuffer(layerBytes * layerCount,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     staging, stagingMemory, &stagingMapped)) {
    this->emitError("uploadMaterialTextures: staging allocation failed");
    return false;
  }

  // Resample each source texture into an RGBA8 layer (nearest neighbour; the
  // RT preview does not need the full mip chain).  A scene without textures
  // still gets a single white layer so binding 17 is always valid.
  unsigned char * dst = static_cast<unsigned char *>(stagingMapped);
  if (layers.empty()) {
    std::memset(dst, 0xff, static_cast<size_t>(layerBytes));
  }
  else {
    for (uint32_t l = 0; l < layerCount; ++l) {
      const SoTextureData & texture = *layers[l].texture;
      unsigned char * layer = dst + static_cast<size_t>(l) * layerBytes;
      for (uint32_t y = 0; y < extent; ++y) {
        const uint32_t sy =
          std::min(static_cast<uint32_t>(texture.height) - 1u,
                   static_cast<uint32_t>(static_cast<uint64_t>(y) *
                                         texture.height / extent));
        for (uint32_t x = 0; x < extent; ++x) {
          const uint32_t sx =
            std::min(static_cast<uint32_t>(texture.width) - 1u,
                     static_cast<uint32_t>(static_cast<uint64_t>(x) *
                                           texture.width / extent));
          const unsigned char * src = texture.pixels +
            (static_cast<size_t>(sy) * texture.width + sx) *
              texture.numComponents;
          unsigned char * out =
            layer + (static_cast<size_t>(y) * extent + x) * 4;
          switch (texture.numComponents) {
          case 1:
            out[0] = out[1] = out[2] = src[0];
            out[3] = 255;
            break;
          case 2:
            out[0] = out[1] = out[2] = src[0];
            out[3] = src[1];
            break;
          case 3:
            out[0] = src[0]; out[1] = src[1]; out[2] = src[2];
            out[3] = 255;
            break;
          default:
            out[0] = src[0]; out[1] = src[1]; out[2] = src[2];
            out[3] = src[3];
            break;
          }
        }
      }
    }
  }

  SoVulkanShared::imageTransition(
    cmd, this->textureArrayImage, VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_IMAGE_ASPECT_COLOR_BIT, 1, layerCount);
  for (uint32_t l = 0; l < layerCount; ++l) {
    VkBufferImageCopy region {};
    region.bufferOffset = static_cast<VkDeviceSize>(l) * layerBytes;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = l;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {extent, extent, 1};
    vkCmdCopyBufferToImage(cmd, staging, this->textureArrayImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }
  SoVulkanShared::imageTransition(
    cmd, this->textureArrayImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    VK_IMAGE_ASPECT_COLOR_BIT, 1, layerCount);

  this->deferDestroy([staging, stagingMemory, vma = this->vmaAllocator]() {
    vmaDestroyBuffer(vma, staging, stagingMemory);
  });

  this->textureSetHash = signature;
  return true;
}
