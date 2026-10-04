// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendGeometry.cpp
//
// Retained geometry cache: createBuffer()/createBufferDeviceLocal() (host-visible
// or device-local via staging + one-shot transfer); uploadGeometry() repacks
// interleaved position/normal/color/texcoord; getOrCreateCache()/invalidateCache()/
// updateGeometryCache() run the per-command cache (content-key/hash detection,
// per-frame eviction).

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanShared.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include <vk_mem_alloc.h>

#include <algorithm>
#include <chrono>
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

VkDeviceSize alignGeometryUpload(VkDeviceSize bytes)
{
  const VkDeviceSize alignment = 64;
  return ((bytes + alignment - 1) / alignment) * alignment;
}

// Encode a float to binary16; CAD texcoords are in [0,1] so this is exact.
static inline uint16_t floatToHalf(float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t exp = (bits >> 23) & 0xFFu;
  uint32_t mant = bits & 0x7FFFFFu;

  if (exp == 0xFFu) {
    // Inf/NaN: saturate to inf with the sign, preserving NaN payload roughly.
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0));
  }
  // The single exponent is biased by 127; the half by 15.  Shift the excess.
  int32_t halfExp = static_cast<int32_t>(exp) - 127 + 15;
  if (halfExp >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (halfExp <= 0) {
    if (halfExp < -10) {
      return static_cast<uint16_t>(sign);
    }
    // Subnormal: shift the mantissa into the denormal range.
    mant = (mant | 0x800000u) >> (1 - halfExp);
    return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
  }
  // Normalized value.  Round to nearest even on the 13 dropped mantissa bits.
  return static_cast<uint16_t>(sign | (halfExp << 10) |
                               ((mant + 0x1000u) >> 13));
}

// True when updateGeometryCache() should visit a command.  Overlay-only renders
// visit SO_RENDERPASS_OVERLAY plus untraced non-triangle residue; full renders
// visit every command with usable geometry.
bool shouldProcessGeometry(const SoRenderCommand & command,
                           const bool overlaysOnly)
{
  const bool isResidual =
    command.geometry.topology != SO_TOPOLOGY_TRIANGLES &&
    command.pass != SO_RENDERPASS_OVERLAY;
  if (overlaysOnly && command.pass != SO_RENDERPASS_OVERLAY && !isResidual) {
    return false;
  }
  const SoGeometryDesc & geometry = command.geometry;
  return geometry.positions && geometry.vertexCount != 0 &&
    geometry.vertexCount <= MAX_VERTEX_COUNT;
}

// Record an uploaded stream's identity on the cache entry: producer-owned
// pointers, counts/strides, and the sampled content hash.
void stampGeometryKeys(VulkanCachedCommand & entry,
                       const SoGeometryDesc & geometry,
                       const uint32_t vertexCount,
                       const uint32_t vertexStride)
{
  entry.posKey = geometry.positions;
  entry.normalKey = geometry.normals;
  entry.colorKey = geometry.colors;
  entry.texcoordKey = geometry.texcoords;
  entry.idxKey = geometry.indices;
  entry.vertexCount = vertexCount;
  entry.indexCount = geometry.indexCount;
  entry.vertexStride = vertexStride;
  entry.texcoordStride = geometry.texcoordStride;
  entry.normalCount = geometry.normalCount;
  entry.contentHash = hashGeometryContent(geometry);
}

void packInterleavedVertices(const SoGeometryDesc & geometry, uint8_t * vertices)
{
  const uint32_t vertexCount = geometry.vertexCount;
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t normalStrideFloats =
    (geometry.normals ? posStrideFloats : 0);
  const uint32_t texcoordStride = geometry.texcoordStride
    ? geometry.texcoordStride : sizeof(float) * 4;
  const uint32_t texcoordStrideFloats = texcoordStride / sizeof(float);

  for (uint32_t i = 0; i < vertexCount; ++i) {
    // 32-byte vertex: pos f32 @0, normal f32 @12, color R8G8B8A8_UNORM @24, uv R16G16_SFLOAT @28.
    uint8_t * out = vertices + static_cast<size_t>(i) * VULKAN_VERTEX_STRIDE;

    const float * pos = geometry.positions +
      static_cast<size_t>(i) * posStrideFloats;
    std::memcpy(out + 0, pos, 3 * sizeof(float));

    float n[3] = {0.0f, 0.0f, 1.0f};
    if (geometry.normals && i < geometry.normalCount) {
      const float * normal = geometry.normals +
        static_cast<size_t>(i) * normalStrideFloats;
      n[0] = normal[0]; n[1] = normal[1]; n[2] = normal[2];
    }
    std::memcpy(out + 12, n, 3 * sizeof(float));

    float c[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (geometry.colors) {
      const float * color = geometry.colors + static_cast<size_t>(i) * 4;
      c[0] = color[0]; c[1] = color[1]; c[2] = color[2]; c[3] = color[3];
    }
    // Clamp to [0,1] then quantize to 8-bit UNORM, matching the VkFormat.
    for (int k = 0; k < 4; ++k) {
      float v = c[k] < 0.0f ? 0.0f : (c[k] > 1.0f ? 1.0f : c[k]);
      out[24 + k] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }

    float uv[2] = {0.0f, 0.0f};
    if (geometry.texcoords) {
      const float * tex = geometry.texcoords +
        static_cast<size_t>(i) * texcoordStrideFloats;
      uv[0] = tex[0]; uv[1] = tex[1];
    }
    const uint16_t halfU = floatToHalf(uv[0]);
    const uint16_t halfV = floatToHalf(uv[1]);
    std::memcpy(out + 28, &halfU, sizeof(halfU));
    std::memcpy(out + 30, &halfV, sizeof(halfV));
  }
}

} // namespace

// --- Geometry cache -------------------------------------------------------

VulkanCachedCommand &
SoVulkanRenderBackend::getOrCreateCache(const SoRenderCommand * command)
{
  const auto found = this->commandToCache.find(command);
  if (found != this->commandToCache.end()) {
    return this->gpuCache[found->second];
  }
  const size_t index = this->gpuCache.size();
  this->gpuCache.emplace_back();
  this->gpuCache.back().commandKey = command;
  this->commandToCache[command] = index;
  return this->gpuCache.back();
}

const VulkanCachedCommand *
SoVulkanRenderBackend::findCachedDrawable(
  const SoRenderCommand & command) const
{
  if (!command.geometry.positions || command.geometry.vertexCount == 0) {
    return nullptr;
  }
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) return nullptr;
  const VulkanCachedCommand & entry = this->gpuCache[found->second];
  if (entry.vertexBuffer == VK_NULL_HANDLE) return nullptr;
  return &entry;
}

bool
SoVulkanRenderBackend::createBufferWithProperties(const VkDeviceSize size,
                                                  const VkBufferUsageFlags usage,
                                                  const VkMemoryPropertyFlags desiredProperties,
                                                  VkBuffer & buffer,
                                                  VmaAllocation & memory,
                                                  const void * data)
{
  buffer = VK_NULL_HANDLE;
  memory = nullptr;

  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = desiredProperties;
  if ((desiredProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    // Declare the sequential-write host access VMA wants; map up front if filling once.
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    if (data) {
      allocInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
  }
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }

  if (data) {
    void * mapped = allocationInfo.pMappedData;
    const bool unmap = (mapped == nullptr);
    if (unmap &&
        vmaMapMemory(this->vmaAllocator, memory, &mapped) != VK_SUCCESS) {
      this->emitError("createBufferWithProperties: vmaMapMemory failed");
      vmaDestroyBuffer(this->vmaAllocator, buffer, memory);
      buffer = VK_NULL_HANDLE;
      memory = nullptr;
      return false;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    if (unmap) {
      vmaUnmapMemory(this->vmaAllocator, memory);
    }
  }
  return true;
}

bool
SoVulkanRenderBackend::createBuffer(VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkBuffer & buffer,
                                    VmaAllocation & memory,
                                    const void * data)
{
  return this->createBufferWithProperties(
    size, usage,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    buffer, memory, data);
}

bool
SoVulkanRenderBackend::createMappedBuffer(VkDeviceSize size,
                                          VkBufferUsageFlags usage,
                                          VkBuffer & buffer,
                                          VmaAllocation & memory,
                                          void ** mapped)
{
  buffer = VK_NULL_HANDLE;
  memory = nullptr;
  if (mapped) *mapped = nullptr;

  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  // VMA_MEMORY_USAGE_AUTO requires an explicit host-access flag with MAPPED.
  allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  if (allocationInfo.pMappedData == nullptr) {
    vmaDestroyBuffer(this->vmaAllocator, buffer, memory);
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  if (mapped) *mapped = allocationInfo.pMappedData;
  return true;
}

void
SoVulkanRenderBackend::deferDestroyBufferMemory(VkBuffer buffer,
                                                VmaAllocation memory)
{
  if (buffer == VK_NULL_HANDLE && memory == nullptr) return;
  const VmaAllocator vma = this->vmaAllocator;
  this->deferDestroy([vma, buffer, memory]() {
    if (buffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(vma, buffer, memory);
    }
  });
}

bool
SoVulkanRenderBackend::createBufferDeviceLocal(VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VkBuffer & buffer,
                                               VmaAllocation & memory,
                                               const void * data)
{
  // Retained static geometry is GPU-read every frame, so it belongs in
  // device-local VRAM; `data` is staged via a transient host-visible buffer and
  // a one-shot transfer fenced before return (acceptable: only reached on the
  // geometry-change path).  On failure the caller falls back to host-visible.
  buffer = VK_NULL_HANDLE;
  memory = nullptr;
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  VmaAllocationInfo allocationInfo {};
  if (vmaCreateBuffer(this->vmaAllocator, &bci, &allocInfo, &buffer, &memory,
                      &allocationInfo) != VK_SUCCESS) {
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }

  if (!data) return true;

  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingMemory = nullptr;
  if (!this->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMemory, data)) {
    vmaDestroyBuffer(this->vmaAllocator, buffer, memory);
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }

  // Per-frame buffers are not yet begun (updateGeometryCache runs before
  // beginCommandBuffer), so the shared one-shot helper records, submits, drains.
  const bool ok = SoVulkanShared::withOneShotSubmit(
    this->device, this->queue, this->commandPool, this->allocator,
    [staging, buffer, size](VkCommandBuffer transfer) {
      VkBufferCopy copy {};
      copy.size = size;
      vkCmdCopyBuffer(transfer, staging, buffer, 1, &copy);
      // Completion alone gives no memory dependency for a later vertex/index
      // read, so explicitly transition TRANSFER_WRITE -> VERTEX_ATTRIBUTE/INDEX.
      SoVulkanShared::bufferTransition(
        transfer, buffer, 0, size, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    });

  vmaDestroyBuffer(this->vmaAllocator, staging, stagingMemory);

  if (!ok) {
    vmaDestroyBuffer(this->vmaAllocator, buffer, memory);
    buffer = VK_NULL_HANDLE;
    memory = nullptr;
    return false;
  }
  return true;
}

void
VulkanCachedCommand::VulkanWideLineBuffer::destroy(VmaAllocator allocator)
{
  if (buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator, buffer, memory);
  }
  buffer = VK_NULL_HANDLE;
  memory = nullptr;
  mapped = nullptr;
  size = 0;
}

void
SoVulkanRenderBackend::uploadGeometry(VulkanCachedCommand & entry,
                                      const SoRenderCommand & command)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;

  // Pack vertices with deterministic defaults for absent streams.  uploadScratch
  // is reused (resize() preserves capacity), so the heap is touched only on the
  // largest upload; the lock spans packing plus the synchronous uploads below.
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;

  std::lock_guard<std::mutex> scratchLock(this->uploadScratchMutex);
  this->uploadScratch.resize(static_cast<size_t>(vertexCount) * VULKAN_VERTEX_STRIDE);
  uint8_t * const vertices = this->uploadScratch.data();
  packInterleavedVertices(geometry, vertices);

  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(vertexCount) * VULKAN_VERTEX_STRIDE;
  // Retained geometry goes to device-local VRAM (staged via one-shot transfer;
  // falls back to host-visible).  Non-retained per-frame geometry (overlays,
  // highlights, text) stays host-visible to avoid the transfer stall per re-upload.
  bool vertexCreated = false;
  if (geometry.retained) {
    vertexCreated = this->createBufferDeviceLocal(vertexBytes,
                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  entry.vertexBuffer,
                                                  entry.vertexMemory, vertices);
  }
  if (!vertexCreated) {
    vertexCreated = this->createBuffer(vertexBytes,
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       entry.vertexBuffer, entry.vertexMemory,
                                       vertices);
  }
  if (!vertexCreated) {
    this->emitError("uploadGeometry: failed to create vertex buffer");
    return;
  }

  if (geometry.indexCount && geometry.indices) {
    const VkDeviceSize indexBytes =
      static_cast<VkDeviceSize>(geometry.indexCount) * sizeof(uint32_t);
    bool indexCreated = false;
    if (geometry.retained) {
      indexCreated = this->createBufferDeviceLocal(indexBytes,
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    entry.indexBuffer,
                                                    entry.indexMemory,
                                                    geometry.indices);
    }
    if (!indexCreated) {
      indexCreated = this->createBuffer(indexBytes,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        entry.indexBuffer, entry.indexMemory,
                                        geometry.indices);
    }
    if (!indexCreated) {
      this->emitError("uploadGeometry: failed to create index buffer");
      if (entry.vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(this->vmaAllocator, entry.vertexBuffer,
                         entry.vertexMemory);
        entry.vertexBuffer = VK_NULL_HANDLE;
        entry.vertexMemory = nullptr;
      }
      return;
    }
  }

  stampGeometryKeys(entry, geometry, vertexCount, posStride);
  entry.vertexOffset = 0;
  entry.indexOffset = 0;
  entry.sharedBlockId = 0;
}

bool
SoVulkanRenderBackend::uploadGeometryShared(VulkanCachedCommand & entry,
                                            const SoRenderCommand & command,
                                            uint32_t blockId)
{
  if (blockId == 0 || blockId > this->geometryBlocks.size()) {
    return false;
  }
  VulkanGeometryBlock & block = this->geometryBlocks[blockId - 1];
  if (block.buffer == VK_NULL_HANDLE || block.mapped == nullptr) {
    return false;
  }

  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;

  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(vertexCount) * VULKAN_VERTEX_STRIDE;
  const VkDeviceSize indexBytes =
    (geometry.indexCount && geometry.indices)
      ? static_cast<VkDeviceSize>(geometry.indexCount) * sizeof(uint32_t)
      : 0;

  VkDeviceSize vertexOffset = 0;
  VkDeviceSize indexOffset = 0;
  if (!this->allocateGeometryArena(blockId, vertexBytes, vertexOffset)) {
    return false;
  }
  if (indexBytes != 0 &&
      !this->allocateGeometryArena(blockId, indexBytes, indexOffset)) {
    return false;
  }

  char * base = reinterpret_cast<char*>(block.mapped);
  packInterleavedVertices(geometry,
    reinterpret_cast<uint8_t*>(base + vertexOffset));
  if (indexBytes != 0) {
    std::memcpy(base + indexOffset, geometry.indices,
      static_cast<size_t>(indexBytes));
  }

  entry.vertexBuffer = block.buffer;
  entry.vertexMemory = VK_NULL_HANDLE;
  entry.indexBuffer = indexBytes != 0 ? block.buffer : VK_NULL_HANDLE;
  entry.indexMemory = VK_NULL_HANDLE;
  entry.vertexOffset = vertexOffset;
  entry.indexOffset = indexOffset;
  entry.sharedBlockId = blockId;
  ++block.refCount;

  stampGeometryKeys(entry, geometry, vertexCount, posStride);
  return true;
}

void
SoVulkanRenderBackend::destroyCacheEntry(VulkanCachedCommand & entry)
{
  if (entry.sharedBlockId != 0) {
    this->releaseGeometryBlock(entry.sharedBlockId);
  }
  else {
    if (entry.indexBuffer) {
      vmaDestroyBuffer(this->vmaAllocator, entry.indexBuffer,
                       entry.indexMemory);
      entry.indexBuffer = VK_NULL_HANDLE;
      entry.indexMemory = nullptr;
    }
    if (entry.vertexBuffer) {
      vmaDestroyBuffer(this->vmaAllocator, entry.vertexBuffer,
                       entry.vertexMemory);
      entry.vertexBuffer = VK_NULL_HANDLE;
      entry.vertexMemory = nullptr;
    }
  }
  for (VulkanCachedCommand::VulkanWideLineBuffer & slot : entry.wideLineBuffers) {
    slot.destroy(this->vmaAllocator);
  }
  entry.wideLineBuffers.clear();
  // Instanced wide-line endpoint buffer.  deferDestroyCacheEntry released it on
  // the deferred path; synchronous invalidateCache() must too, or it leaks past
  // vkDestroyDevice (VUID-vkDestroyDevice-device-05137).
  if (entry.instancedLineBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(this->vmaAllocator, entry.instancedLineBuffer,
                     entry.instancedLineMemory);
    entry.instancedLineBuffer = VK_NULL_HANDLE;
    entry.instancedLineMemory = nullptr;
  }
  this->destroySubPixelResources(entry);
  entry = VulkanCachedCommand();
}

void
SoVulkanRenderBackend::invalidateCache()
{
  for (VulkanCachedCommand & entry : this->gpuCache) {
    this->destroyCacheEntry(entry);
  }
  this->gpuCache.clear();
  this->commandToCache.clear();
  this->invalidateTextureCache();
}

uint32_t
SoVulkanRenderBackend::allocateGeometryBlock(VkDeviceSize capacity)
{
  capacity = alignGeometryUpload(std::max<VkDeviceSize>(capacity, 64));
  if (capacity == 0) {
    return 0;
  }

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation memory = nullptr;
  void * mapped = nullptr;
  if (!this->createMappedBuffer(
        capacity,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        buffer, memory, &mapped)) {
    return 0;
  }

  VulkanGeometryBlock block {};
  block.buffer = buffer;
  block.memory = memory;
  block.mapped = mapped;
  block.capacity = capacity;
  block.used = 0;
  block.refCount = 0;
  if (!this->freeGeometryBlockIds.empty()) {
    const uint32_t recycled = this->freeGeometryBlockIds.back();
    this->freeGeometryBlockIds.pop_back();
    this->geometryBlocks[recycled - 1] = block;
    this->nextGeometryBlockCapacity =
      std::min<VkDeviceSize>(this->nextGeometryBlockCapacity * 2u, 16u * 1024u * 1024u);
    return recycled;
  }
  this->geometryBlocks.push_back(block);

  this->nextGeometryBlockCapacity =
    std::min<VkDeviceSize>(this->nextGeometryBlockCapacity * 2u, 16u * 1024u * 1024u);
  return static_cast<uint32_t>(this->geometryBlocks.size());
}

bool
SoVulkanRenderBackend::allocateGeometryArena(uint32_t blockId, VkDeviceSize size,
                                             VkDeviceSize & offset)
{
  offset = 0;
  if (blockId == 0 || blockId > this->geometryBlocks.size() || size == 0) {
    return false;
  }
  VulkanGeometryBlock & block = this->geometryBlocks[blockId - 1];
  if (block.buffer == VK_NULL_HANDLE || block.mapped == nullptr) {
    return false;
  }
  const VkDeviceSize aligned = alignGeometryUpload(size);
  if (block.used + aligned > block.capacity) {
    return false;
  }
  offset = block.used;
  block.used += aligned;
  return true;
}

void
SoVulkanRenderBackend::releaseGeometryBlockResources(VulkanGeometryBlock & block)
{
  if (block.buffer != VK_NULL_HANDLE) {
    // vmaDestroyBuffer frees the buffer, memory and persistent mapping together.
    vmaDestroyBuffer(this->vmaAllocator, block.buffer, block.memory);
    block.buffer = VK_NULL_HANDLE;
  }
  block.memory = nullptr;
  block.mapped = nullptr;
  block.capacity = 0;
  block.used = 0;
  block.refCount = 0;
}

void
SoVulkanRenderBackend::releaseGeometryBlock(uint32_t blockId)
{
  if (blockId == 0 || blockId > this->geometryBlocks.size()) {
    return;
  }
  VulkanGeometryBlock & block = this->geometryBlocks[blockId - 1];
  if (block.buffer == VK_NULL_HANDLE) {
    return;
  }
  if (block.refCount > 0) {
    --block.refCount;
  }
  if (block.refCount > 0) {
    return;
  }
  this->releaseGeometryBlockResources(block);
  this->freeGeometryBlockIds.push_back(blockId);
}

void
SoVulkanRenderBackend::deferReleaseGeometryBlock(uint32_t blockId)
{
  if (blockId == 0) {
    return;
  }
  this->deferDestroy([this, blockId]() {
    this->releaseGeometryBlock(blockId);
  });
}

void
SoVulkanRenderBackend::destroyAllGeometryBlocks()
{
  for (VulkanGeometryBlock & block : this->geometryBlocks) {
    this->releaseGeometryBlockResources(block);
  }
  this->geometryBlocks.clear();
  this->freeGeometryBlockIds.clear();
  this->nextGeometryBlockCapacity = 256u * 1024u;
}

void
SoVulkanRenderBackend::updateGeometryCache(const SoDrawList & drawlist,
                                           const bool overlaysOnly,
                                           const bool geometryContentUnchanged)
{
  // Frame boundary handled by beginFrame() at the entry point.  Drop uploads left
  // by a frame that aborted before flush/finalize; the shared cross-frame staging
  // pool needs no teardown, and the next frame re-stages from scratch.
  this->pendingUploads.clear();

  const uint32_t generation = drawlist.getGeneration();

  // Overlay-composite mode (another renderer owns the scene): the sweep must
  // release triangle commands this backend no longer visits, but replayed
  // (camera-only) frames do not advance the retained list's generation, so use a
  // dedicated epoch.
  const bool compositeSweep = overlaysOnly && this->overlayCompositeMode;
  if (compositeSweep) {
    ++this->overlayCompositeEpoch;
  }

  this->needsGeometryScratch.assign(
    static_cast<size_t>(std::max(0, drawlist.getNumCommands())), 0);
  // Start texture staging at the shared pool front so uploads coalesce (see prepareTextureUpload).
  this->stagingPoolCursor = 0;
  std::vector<uint8_t> & needsGeometry = this->needsGeometryScratch;
  int retainedUploads = 0;
  VkDeviceSize retainedUploadBytes = 0;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (!shouldProcessGeometry(command, overlaysOnly)) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;

    VulkanCachedCommand & entry = this->getOrCreateCache(&command);
    const uint32_t vertexStride = geometry.vertexStride
      ? geometry.vertexStride : sizeof(float) * 3;
    const bool identityMatches = entry.vertexBuffer != VK_NULL_HANDLE &&
      entry.posKey == geometry.positions &&
      entry.normalKey == geometry.normals &&
      entry.colorKey == geometry.colors &&
      entry.texcoordKey == geometry.texcoords &&
      entry.idxKey == geometry.indices &&
      entry.vertexCount == geometry.vertexCount &&
      entry.indexCount == geometry.indexCount &&
      entry.normalCount == geometry.normalCount &&
      entry.vertexStride == vertexStride &&
      entry.texcoordStride == geometry.texcoordStride;
    // Change detection.  Retained geometry (SoGeometryDesc::retained) guarantees
    // pointers change exactly when content does (rebuild reallocates), so pointer
    // identity suffices and the FNV walk is skipped; replayed frames
    // (geometryContentUnchanged) are bit-identical, also skipped.  Otherwise
    // (arena streams rewriting the same pointer in place) use the content hash.
    const bool pointerIdentitySufficient =
      geometry.retained || geometryContentUnchanged;
    const bool geometryMatches = identityMatches &&
      (pointerIdentitySufficient ||
       entry.contentHash == hashGeometryContent(geometry));
    if (!geometryMatches) {
      needsGeometry[static_cast<size_t>(i)] = 1;
      if (geometry.retained) {
        ++retainedUploads;
        retainedUploadBytes += alignGeometryUpload(
          static_cast<VkDeviceSize>(geometry.vertexCount) *
            VULKAN_VERTEX_STRIDE);
        if (geometry.indexCount && geometry.indices) {
          retainedUploadBytes += alignGeometryUpload(
            static_cast<VkDeviceSize>(geometry.indexCount) *
              sizeof(uint32_t));
        }
      }
    }
  }

  uint32_t sharedBlockId = 0;
  int sharedUploads = 0;
  constexpr VkDeviceSize VK_GEOMETRY_BATCH_HOST_LIMIT =
    static_cast<VkDeviceSize>(8u * 1024u * 1024u);
  if (retainedUploads > 1 && retainedUploadBytes > 0 &&
      retainedUploadBytes <= VK_GEOMETRY_BATCH_HOST_LIMIT) {
    sharedBlockId = this->allocateGeometryBlock(retainedUploadBytes + 4096u);
  }

  // Ensure the descriptor pool holds one set per distinct texture before any
  // allocation; growth never invalidates existing sets, so this is always safe.
  if (!this->ensureDescriptorPoolSpace()) {
    this->emitError("updateGeometryCache: failed to grow descriptor pool");
    // Continue with the current pool; failures fall back to the white texture.
  }

  this->pendingUploads.clear();

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    // Overlay-only renders (RT compositing) upload SO_RENDERPASS_OVERLAY commands
    // (nav cube, axis cross, highlights) plus untraced non-triangle residue (Brep
    // edges, markers, polylines); skipping already-traced triangles keeps it cheap.
    if (!shouldProcessGeometry(command, overlaysOnly)) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;

    VulkanCachedCommand & entry = this->getOrCreateCache(&command);
    // The draw-list generation changes every frame (clear() bumps it), so it is
    // only an eviction visit stamp, never a re-upload signal; re-uploads come from
    // the producer-owned content keys (content hash re-verified for in-place edits).
    if (needsGeometry[static_cast<size_t>(i)]) {
      this->deferDestroyCacheEntry(entry);
      bool uploadedShared = false;
      if (geometry.retained && sharedBlockId != 0) {
        uploadedShared =
          this->uploadGeometryShared(entry, command, sharedBlockId);
        if (uploadedShared) {
          ++sharedUploads;
        }
      }
      if (!uploadedShared) {
        this->uploadGeometry(entry, command);
      }
    }
    entry.commandKey = &command;
    entry.cacheGeneration = generation;
    if (compositeSweep) {
      entry.compositeEpoch = this->overlayCompositeEpoch;
    }

    const SoTextureData & texture = command.material.texture;
    if (texture.pixels && texture.width > 0 && texture.height > 0) {
      VulkanCachedTexture & texEntry = this->getOrCreateTexture(&command);
      // Texture pixels live in per-frame arena storage that may rewrite the same
      // pointer in place, so the content hash is always re-verified.
      const bool textureMatches = texEntry.image != VK_NULL_HANDLE &&
        texEntry.pixelsKey == texture.pixels &&
        texEntry.width == texture.width &&
        texEntry.height == texture.height &&
        texEntry.numComponents == texture.numComponents &&
        texEntry.minFilter == texture.minFilter &&
        texEntry.magFilter == texture.magFilter &&
        texEntry.wrapS == texture.wrapS &&
        texEntry.wrapT == texture.wrapT &&
        texEntry.model == texture.model &&
        texEntry.contentHash == hashTextureContent(texture);
      if (!textureMatches) {
        this->deferDestroyTextureEntry(texEntry);
        // A command appearing twice would prepare two uploads for the same entry
        // (leaking the first image); the first pending upload wins.
        bool alreadyPending = false;
        for (const PendingTextureUpload & prior : this->pendingUploads) {
          if (prior.index == this->commandToTexture[&command]) {
            alreadyPending = true;
            break;
          }
        }
        if (!alreadyPending) {
          PendingTextureUpload upload;
          upload.command = &command;
          upload.index = this->commandToTexture[&command];
          upload.texture = &texture;
          if (this->prepareTextureUpload(texEntry, texture, upload.stagingOffset,
                                         upload.stagingBytes)) {
            this->pendingUploads.push_back(upload);
          }
          // Failure reset the entry; unstamped content keys make the next frame retry.
        }
      }
      texEntry.commandKey = &command;
      texEntry.cacheGeneration = generation;
    }
  }

  if (sharedBlockId != 0 && sharedUploads == 0) {
    this->releaseGeometryBlock(sharedBlockId);
  }

  // Evict entries not visited this frame (command gone, or its arena pointer no
  // longer in this frame); survivors keep index identity, so rebuild the pointer
  // maps from commandKey.  Destruction is deferred.  A transient overlay-only
  // render skips the sweep (it would evict the whole cache); overlay-composite
  // mode (another renderer owns the scene) runs it to release geometry that
  // renderer already owns.
  if (!overlaysOnly || this->overlayCompositeMode) {
    // `stale` decides whether to drop an entry: full/transient-overlay renders key
    // on the draw-list generation, an overlay-composite pass on the composite epoch.
    const auto evictStale = [&](auto & cache, auto destroyEntry,
                                auto & indexMap, auto stale) {
      bool anyStale = false;
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        if (stale(cache[idx])) {
          destroyEntry(cache[idx]);
          anyStale = true;
        }
      }
      if (!anyStale) return;
      size_t write = 0;
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        if (!stale(cache[idx])) {
          if (write != idx) cache[write] = std::move(cache[idx]);
          ++write;
        }
      }
      cache.resize(write);
      indexMap.clear();
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        indexMap[cache[idx].commandKey] = idx;
      }
    };
    if (compositeSweep) {
      // Release triangle entries not visited this pass (only overlays and
      // residue are stamped), so the owning renderer's copy is the only one.
      evictStale(this->gpuCache,
                 [this](VulkanCachedCommand & entry) {
                   this->deferDestroyCacheEntry(entry);
                 },
                 this->commandToCache,
                 [this](const VulkanCachedCommand & entry) {
                   return entry.compositeEpoch != this->overlayCompositeEpoch;
                 });
    }
    else {
      evictStale(this->gpuCache,
                 [this](VulkanCachedCommand & entry) {
                   this->deferDestroyCacheEntry(entry);
                 },
                 this->commandToCache,
                 [generation](const VulkanCachedCommand & entry) {
                   return entry.cacheGeneration != generation;
                 });
    }
    evictStale(this->textureCache,
               [this](VulkanCachedTexture & entry) {
                 this->deferDestroyTextureEntry(entry);
               },
               this->commandToTexture,
               [generation](const VulkanCachedTexture & entry) {
                 return entry.cacheGeneration != generation;
               });

    // Eviction compacted/reindexed the texture cache, so captured upload indices
    // are stale; re-resolve via the command pointer (just-prepared entries survive).
    for (PendingTextureUpload & upload : this->pendingUploads) {
      const auto it = this->commandToTexture.find(upload.command);
      if (it != this->commandToTexture.end()) {
        upload.index = it->second;
      }
      else {
        upload.index = std::numeric_limits<size_t>::max();
      }
    }
  }
}
