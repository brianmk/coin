// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendGeometry.cpp
//
// Retained geometry cache.  Provides:
//
//   - createBuffer()/createBufferDeviceLocal(): host-visible or device-local
//     Vulkan buffers (the latter via a transient staging copy + one-shot
//     transfer)
//   - uploadGeometry(): repack the interleaved vertex layout (position +
//     normal + color + texcoord) into the reusable uploadScratch vector
//   - getOrCreateCache(), invalidateCache(), updateGeometryCache(): drive the
//     per-command GPU cache, re-uploading on a content-key/content-hash change
//     and evicting stale entries each frame

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanShared.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

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

long vkGeometryBreadcrumbNowUs()
{
  return (long)std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool vkGeometryBreadcrumbEnabled()
{
  static const bool enabled = std::getenv("FC_GUI_OPEN_BREADCRUMB") != nullptr;
  return enabled;
}

VkDeviceSize alignGeometryUpload(VkDeviceSize bytes)
{
  const VkDeviceSize alignment = 64;
  return ((bytes + alignment - 1) / alignment) * alignment;
}

void packInterleavedVertices(const SoGeometryDesc & geometry, float * vertices)
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
    float * out = vertices + static_cast<size_t>(i) * 12;

    const float * pos = geometry.positions +
      static_cast<size_t>(i) * posStrideFloats;
    out[0] = pos[0];
    out[1] = pos[1];
    out[2] = pos[2];

    if (geometry.normals && i < geometry.normalCount) {
      const float * normal = geometry.normals +
        static_cast<size_t>(i) * normalStrideFloats;
      out[3] = normal[0];
      out[4] = normal[1];
      out[5] = normal[2];
    }
    else {
      out[3] = 0.0f;
      out[4] = 0.0f;
      out[5] = 1.0f;
    }

    if (geometry.colors) {
      const float * color = geometry.colors + static_cast<size_t>(i) * 4;
      out[6] = color[0];
      out[7] = color[1];
      out[8] = color[2];
      out[9] = color[3];
    }
    else {
      out[6] = 1.0f;
      out[7] = 1.0f;
      out[8] = 1.0f;
      out[9] = 1.0f;
    }

    if (geometry.texcoords) {
      const float * uv = geometry.texcoords +
        static_cast<size_t>(i) * texcoordStrideFloats;
      out[10] = uv[0];
      out[11] = uv[1];
    }
    else {
      out[10] = 0.0f;
      out[11] = 0.0f;
    }
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

bool
SoVulkanRenderBackend::selectMemoryType(const VkMemoryRequirements & requirements,
                                        const VkMemoryPropertyFlags desired,
                                        uint32_t & memoryTypeIndex)
{
  const VkPhysicalDeviceMemoryProperties & props = this->memProps.properties();
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags & desired) == desired) {
      memoryTypeIndex = i;
      return true;
    }
  }
  memoryTypeIndex = 0;
  return false;
}

bool
SoVulkanRenderBackend::allocateBufferMemory(VkBuffer buffer,
                                            const VkMemoryRequirements & requirements,
                                            const VkMemoryPropertyFlags desiredProperties,
                                            VkDeviceMemory & memory)
{
  // Memory-type policy stays with this backend (exact-match, no fallback);
  // only the allocate+bind boilerplate is shared.
  return SoVulkanShared::bindBufferMemory(
    this->device, this->allocator, buffer, requirements, desiredProperties,
    [this](const VkMemoryRequirements & req, VkMemoryPropertyFlags desired,
           uint32_t & memoryTypeIndex) {
      return this->selectMemoryType(req, desired, memoryTypeIndex);
    }, memory);
}

bool
SoVulkanRenderBackend::createBufferWithProperties(const VkDeviceSize size,
                                                  const VkBufferUsageFlags usage,
                                                  const VkMemoryPropertyFlags desiredProperties,
                                                  VkBuffer & buffer,
                                                  VkDeviceMemory & memory,
                                                  const void * data)
{
  buffer = VK_NULL_HANDLE;
  memory = VK_NULL_HANDLE;
  if (!SoVulkanShared::createBufferAllocated(
        this->device, this->allocator, size, usage, desiredProperties,
        /*deviceAddress*/ false,
        [this](const VkMemoryRequirements & req, VkMemoryPropertyFlags desired,
               uint32_t & memoryTypeIndex) {
          return this->selectMemoryType(req, desired, memoryTypeIndex);
        }, buffer, memory)) {
    return false;
  }

  if (data) {
    void * mapped = nullptr;
    if (vkMapMemory(this->device, memory, 0, size, 0, &mapped) != VK_SUCCESS) {
      this->emitError("createBufferWithProperties: vkMapMemory failed");
      vkDestroyBuffer(this->device, buffer, this->allocator);
      vkFreeMemory(this->device, memory, this->allocator);
      buffer = VK_NULL_HANDLE;
      memory = VK_NULL_HANDLE;
      return false;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(this->device, memory);
  }
  return true;
}

bool
SoVulkanRenderBackend::createBuffer(VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkBuffer & buffer,
                                    VkDeviceMemory & memory,
                                    const void * data)
{
  return this->createBufferWithProperties(
    size, usage,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    buffer, memory, data);
}

bool
SoVulkanRenderBackend::createBufferDeviceLocal(VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VkBuffer & buffer,
                                               VkDeviceMemory & memory,
                                               const void * data)
{
  // Retained static geometry is read by the GPU every frame, so it belongs in
  // device-local VRAM rather than host-visible memory.  `data` is copied from
  // a transient host-visible staging buffer with a one-shot transfer that is
  // fenced before this function returns.  The GPU then reads the mesh from
  // device memory instead of walking the PCIe/system bus every frame.
  //
  // This is only invoked from the geometry-change path (not steady-state), so
  // the synchronous transfer is acceptable.  On any failure the buffer/memory
  // are left null and the caller falls back to the host-visible createBuffer().
  if (!SoVulkanShared::createBufferAllocated(
        this->device, this->allocator, size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        /*deviceAddress*/ false,
        [this](const VkMemoryRequirements & req, VkMemoryPropertyFlags desired,
               uint32_t & memoryTypeIndex) {
          return this->selectMemoryType(req, desired, memoryTypeIndex);
        }, buffer, memory)) {
    return false;
  }

  if (!data) return true;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMemory, data)) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    vkFreeMemory(this->device, memory, this->allocator);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    return false;
  }

  // One-shot transfer command buffer.  The per-frame buffers are not yet begun
  // at this point (updateGeometryCache runs before beginCommandBuffer), so we
  // allocate a transient buffer from the shared pool and fence-wait it.
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer transfer = VK_NULL_HANDLE;
  bool ok = vkAllocateCommandBuffers(this->device, &allocInfo, &transfer) ==
    VK_SUCCESS;

  if (ok) {
    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ok = vkBeginCommandBuffer(transfer, &bi) == VK_SUCCESS;
  }
  if (ok) {
    VkBufferCopy copy {};
    copy.size = size;
    vkCmdCopyBuffer(transfer, staging, buffer, 1, &copy);
    // Make the device-local writes visible to a later vertex-input read.  The
    // transfer is fenced so the copy has executed, but fence completion alone
    // does not establish a memory dependency for the buffer being read as
    // vertex/index attributes in a subsequent submit.  This barrier transitions
    // it from TRANSFER_WRITE to VERTEX_ATTRIBUTE/INDEX read.
    SoVulkanShared::bufferTransition(
      transfer, buffer, 0, size, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    ok = vkEndCommandBuffer(transfer) == VK_SUCCESS;
  }

  VkFence fence = VK_NULL_HANDLE;
  if (ok) {
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ok = vkCreateFence(this->device, &fci, this->allocator, &fence) == VK_SUCCESS;
  }
  if (ok) {
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transfer;
    ok = vkQueueSubmit(this->queue, 1, &submit, fence) == VK_SUCCESS;
  }
  if (ok) {
    ok = vkWaitForFences(this->device, 1, &fence, VK_TRUE, UINT64_MAX) ==
      VK_SUCCESS;
  }
  if (fence != VK_NULL_HANDLE) {
    vkDestroyFence(this->device, fence, this->allocator);
  }
  if (transfer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &transfer);
  }
  vkDestroyBuffer(this->device, staging, this->allocator);
  vkFreeMemory(this->device, stagingMemory, this->allocator);

  if (!ok) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    vkFreeMemory(this->device, memory, this->allocator);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

void
SoVulkanRenderBackend::uploadGeometry(VulkanCachedCommand & entry,
                                      const SoRenderCommand & command)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;

  // Pack interleaved vertices with deterministic defaults for absent streams.
  // The buffer is a reusable member scratch vector: resize() preserves
  // capacity, so the heap is only touched on the first (largest) upload that
  // reaches this size rather than on every geometry change.  The lock spans the
  // packing plus the synchronous uploads below, which read `vertices`.
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;

  std::lock_guard<std::mutex> scratchLock(this->uploadScratchMutex);
  this->uploadScratch.resize(static_cast<size_t>(vertexCount) * 12);
  float * const vertices = this->uploadScratch.data();
  packInterleavedVertices(geometry, vertices);

  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(vertexCount) * VULKAN_VERTEX_STRIDE;
  // Cached static geometry (retained) lives in device-local VRAM so the GPU
  // does not read large meshes across the PCIe/system bus every frame.  The
  // upload is staged through a transient one-shot transfer; if device-local is
  // unavailable or the copy fails, fall back to the host-visible path so
  // rendering still works.  Non-retained geometry (per-frame overlays,
  // highlights, text, images -- which rewrite every frame) stays host-visible
  // so its frequent re-uploads never take the synchronous transfer stall.
  bool vertexCreated = false;
  if (geometry.retained) {
    vertexCreated = this->createBufferDeviceLocal(vertexBytes,
                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                  entry.vertexBuffer,
                                                  entry.vertexMemory, vertices);
  }
  if (!vertexCreated) {
    vertexCreated = this->createBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                    entry.indexBuffer,
                                                    entry.indexMemory,
                                                    geometry.indices);
    }
    if (!indexCreated) {
      indexCreated = this->createBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                        entry.indexBuffer, entry.indexMemory,
                                        geometry.indices);
    }
    if (!indexCreated) {
      this->emitError("uploadGeometry: failed to create index buffer");
      if (entry.vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(this->device, entry.vertexBuffer, this->allocator);
        entry.vertexBuffer = VK_NULL_HANDLE;
      }
      if (entry.vertexMemory != VK_NULL_HANDLE) {
        vkFreeMemory(this->device, entry.vertexMemory, this->allocator);
        entry.vertexMemory = VK_NULL_HANDLE;
      }
      return;
    }
  }

  entry.posKey = geometry.positions;
  entry.normalKey = geometry.normals;
  entry.colorKey = geometry.colors;
  entry.texcoordKey = geometry.texcoords;
  entry.idxKey = geometry.indices;
  entry.vertexCount = vertexCount;
  entry.indexCount = geometry.indexCount;
  entry.vertexStride = posStride;
  entry.texcoordStride = geometry.texcoordStride;
  entry.normalCount = geometry.normalCount;
  entry.contentHash = hashGeometryContent(geometry);
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
    reinterpret_cast<float*>(base + vertexOffset));
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

  entry.posKey = geometry.positions;
  entry.normalKey = geometry.normals;
  entry.colorKey = geometry.colors;
  entry.texcoordKey = geometry.texcoords;
  entry.idxKey = geometry.indices;
  entry.vertexCount = vertexCount;
  entry.indexCount = geometry.indexCount;
  entry.vertexStride = posStride;
  entry.texcoordStride = geometry.texcoordStride;
  entry.normalCount = geometry.normalCount;
  entry.contentHash = hashGeometryContent(geometry);
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
      vkDestroyBuffer(this->device, entry.indexBuffer, this->allocator);
      entry.indexBuffer = VK_NULL_HANDLE;
    }
    if (entry.indexMemory) {
      vkFreeMemory(this->device, entry.indexMemory, this->allocator);
      entry.indexMemory = VK_NULL_HANDLE;
    }
    if (entry.vertexBuffer) {
      vkDestroyBuffer(this->device, entry.vertexBuffer, this->allocator);
      entry.vertexBuffer = VK_NULL_HANDLE;
    }
    if (entry.vertexMemory) {
      vkFreeMemory(this->device, entry.vertexMemory, this->allocator);
      entry.vertexMemory = VK_NULL_HANDLE;
    }
  }
  for (const VulkanCachedCommand::VulkanWideLineBuffer & slot : entry.wideLineBuffers) {
    if (slot.buffer) {
      vkDestroyBuffer(this->device, slot.buffer, this->allocator);
    }
    if (slot.memory) {
      vkFreeMemory(this->device, slot.memory, this->allocator);
    }
  }
  entry.wideLineBuffers.clear();
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

  VkBufferCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = capacity;
  ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer = VK_NULL_HANDLE;
  if (vkCreateBuffer(this->device, &ci, this->allocator, &buffer) != VK_SUCCESS) {
    return 0;
  }

  VkMemoryRequirements requirements {};
  vkGetBufferMemoryRequirements(this->device, buffer, &requirements);

  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (!this->allocateBufferMemory(buffer, requirements,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  memory)) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    return 0;
  }

  void * mapped = nullptr;
  const VkDeviceSize allocationSize = requirements.size;
  if (vkMapMemory(this->device, memory, 0, allocationSize, 0, &mapped) != VK_SUCCESS ||
      mapped == nullptr) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    vkFreeMemory(this->device, memory, this->allocator);
    return 0;
  }

  VulkanGeometryBlock block {};
  block.buffer = buffer;
  block.memory = memory;
  block.mapped = mapped;
  block.capacity = allocationSize;
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
  if (block.mapped != nullptr) {
    vkUnmapMemory(this->device, block.memory);
    block.mapped = nullptr;
  }
  if (block.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, block.buffer, this->allocator);
    block.buffer = VK_NULL_HANDLE;
  }
  if (block.memory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, block.memory, this->allocator);
    block.memory = VK_NULL_HANDLE;
  }
  block.capacity = 0;
  block.used = 0;
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
    if (block.mapped != nullptr) {
      vkUnmapMemory(this->device, block.memory);
      block.mapped = nullptr;
    }
    if (block.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, block.buffer, this->allocator);
      block.buffer = VK_NULL_HANDLE;
    }
    if (block.memory != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, block.memory, this->allocator);
      block.memory = VK_NULL_HANDLE;
    }
    block.capacity = 0;
    block.used = 0;
    block.refCount = 0;
  }
  this->geometryBlocks.clear();
  this->freeGeometryBlockIds.clear();
  this->nextGeometryBlockCapacity = 256u * 1024u;
}

void
SoVulkanRenderBackend::updateGeometryCache(const SoDrawList & drawlist,
                                           const bool overlaysOnly)
{
  // The frame boundary was handled by beginFrame() at the entry point.

  // Release uploads left over from a frame that aborted between the cache
  // update and the flush/finalize step (e.g. a failed framebuffer create).
  // Their copies were never recorded, but deferring the staging destruction
  // is uniformly safe and keeps a single cleanup path.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging == VK_NULL_HANDLE) continue;
    const VkDevice device = this->device;
    const VkAllocationCallbacks * allocator = this->allocator;
    const VkBuffer staging = upload.staging;
    const VkDeviceMemory stagingMemory = upload.stagingMemory;
    this->deferDestroy([device, allocator, staging, stagingMemory]() {
      if (staging != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, staging, allocator);
      }
      if (stagingMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, stagingMemory, allocator);
      }
    });
  }
  this->pendingUploads.clear();

  const long cacheBcStart = vkGeometryBreadcrumbEnabled() ? vkGeometryBreadcrumbNowUs() : 0;
  int bcCommands = 0;
  int bcGeometryUploads = 0;
  int bcTexturePrepares = 0;
  size_t bcVertices = 0;
  size_t bcIndices = 0;

  const uint32_t generation = drawlist.getGeneration();

  this->needsGeometryScratch.assign(
    static_cast<size_t>(std::max(0, drawlist.getNumCommands())), 0);
  std::vector<uint8_t> & needsGeometry = this->needsGeometryScratch;
  int retainedUploads = 0;
  VkDeviceSize retainedUploadBytes = 0;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const bool isResidual =
      command.geometry.topology != SO_TOPOLOGY_TRIANGLES &&
      command.pass != SO_RENDERPASS_OVERLAY;
    if (overlaysOnly && command.pass != SO_RENDERPASS_OVERLAY &&
        !isResidual) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) {
      continue;
    }

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
    const bool geometryMatches = identityMatches &&
      entry.contentHash == hashGeometryContent(geometry);
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

  // Make sure the descriptor pool can hold one set per distinct texture in
  // this frame before any allocation happens.  Pool growth never
  // invalidates existing sets, so this is safe regardless of recording
  // state.
  if (!this->ensureDescriptorPoolSpace()) {
    this->emitError("updateGeometryCache: failed to grow descriptor pool");
    // Continue with the current pool: allocateTextureDescriptorSet()
    // failures fall back to the white texture per command.
  }

  this->pendingUploads.clear();

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    // Overlay-only renders (ray-tracing compositing) draw SO_RENDERPASS_OVERLAY
    // commands (nav cube, axis cross, selection/hover highlights) plus the
    // non-triangle residue the RT backend did not trace (BRep edge lines,
    // point markers, polylines): those must be uploaded so the composite can
    // rasterize them onto the traced surface.  Pure triangle geometry is
    // already traced and skipping it here keeps the composite cheap.
    const bool isResidual =
      command.geometry.topology != SO_TOPOLOGY_TRIANGLES &&
      command.pass != SO_RENDERPASS_OVERLAY;
    if (overlaysOnly && command.pass != SO_RENDERPASS_OVERLAY &&
        !isResidual) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) {
      continue;
    }
    ++bcCommands;
    bcVertices += geometry.vertexCount;
    bcIndices += geometry.indexCount;

    VulkanCachedCommand & entry = this->getOrCreateCache(&command);
    // The draw-list generation changes every frame (clear() bumps it), so
    // it is only a visit stamp for cache eviction below -- never a signal
    // to re-upload.  Re-uploads are driven purely by the producer-owned
    // content keys.
    //
    // Content (not just pointer identity) is always re-verified, because a
    // producer may edit retained buffers in place (same pointer, new data);
    // the sampled content hash is bounded and cheap, so it is authoritative
    // for every command, retained or per-frame.
    if (needsGeometry[static_cast<size_t>(i)]) {
      ++bcGeometryUploads;
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

    const SoTextureData & texture = command.material.texture;
    if (texture.pixels && texture.width > 0 && texture.height > 0) {
      VulkanCachedTexture & texEntry = this->getOrCreateTexture(&command);
      // Texture pixels come from per-frame action storage (arena), which may
      // rewrite the same pointer in place, so the content hash is always
      // re-verified -- pointer identity alone is not sound for textures.
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
        // A command that appears twice in one draw list would otherwise
        // prepare two uploads for the same entry (leaking the first image);
        // the first pending upload for this index wins.
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
          ++bcTexturePrepares;
          if (this->prepareTextureUpload(texEntry, texture, upload.staging,
                                         upload.stagingMemory)) {
            this->pendingUploads.push_back(upload);
          }
          // On failure the entry was reset by prepareTextureUpload();
          // leaving the content keys unstamped makes the next frame retry.
        }
      }
      texEntry.commandKey = &command;
      texEntry.cacheGeneration = generation;
    }
  }

  if (sharedBlockId != 0 && sharedUploads == 0) {
    this->releaseGeometryBlock(sharedBlockId);
  }

  // Evict entries that were not visited this frame: their command has
  // disappeared from the draw list (or its pointer is no longer part of
  // this frame's arena).  Entries surviving eviction keep their index
  // identity, so rebuild the pointer maps from the stored commandKey.
  // Destruction is deferred: a pending frame may still reference the
  // evicted buffers/images.
  // Overlay-only renders skip the sweep: their traversal deliberately
  // visits only overlay commands, so a sweep here would evict the entire
  // scene cache and force a full re-upload on the next full render.
  if (!overlaysOnly) {
    const auto evictStale = [&](auto & cache, auto destroyEntry,
                                auto & indexMap) {
      bool anyStale = false;
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        if (cache[idx].cacheGeneration != generation) {
          destroyEntry(cache[idx]);
          anyStale = true;
        }
      }
      if (!anyStale) return;
      size_t write = 0;
      for (size_t idx = 0; idx < cache.size(); ++idx) {
        if (cache[idx].cacheGeneration == generation) {
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
    evictStale(this->gpuCache,
               [this](VulkanCachedCommand & entry) {
                 this->deferDestroyCacheEntry(entry);
               },
               this->commandToCache);
    evictStale(this->textureCache,
               [this](VulkanCachedTexture & entry) {
                 this->deferDestroyTextureEntry(entry);
               },
               this->commandToTexture);

    // Eviction compacts the texture cache and reindexes it, so the upload
    // indices captured above are stale.  Re-resolve each pending upload
    // through its command pointer; entries that were just prepared carry the
    // current generation and survive the sweep.
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

  if (cacheBcStart) {
    static int logged = 0;
    const long now = vkGeometryBreadcrumbNowUs();
    const long dur = now - cacheBcStart;
    if (logged < 20 && (dur >= 5000 || bcGeometryUploads > 0)) {
      ++logged;
      std::fprintf(stderr,
                   "[VKGEOMCACHE] %ld updateGeometryCache dur_us=%ld commands=%d "
                   "uploads=%d textures=%d vertices=%zu indices=%zu\n",
                   cacheBcStart, dur, bcCommands, bcGeometryUploads,
                   bcTexturePrepares, bcVertices, bcIndices);
      std::fflush(stderr);
    }
  }
}
