#include <rendering/SoVulkanGeometryCache.h>

// SoRenderBackend.h defines SoRenderParams, which SoVulkanRenderBackendP.h's
// inline helpers use; it must precede P.h.
#include <rendering/SoRenderBackend.h>
#include <rendering/SoVulkanBufferFactory.h>
#include <rendering/SoVulkanGeometryArena.h>
#include <rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h>

#include <cstring>
#include <utility>

using namespace CoinVulkanDetail;

namespace {

// Encode a float to a half (binary16).  Ranges outside the finite half range
// clamp to +/-inf; CAD texcoords are in [0,1] so this is exact in practice.
inline uint16_t floatToHalf(float value)
{
  // IEEE 754 single -> binary16 by narrowing the exponent/mantissa.
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
    // Overflow -> inf.
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (halfExp <= 0) {
    // Subnormal / zero underflow.
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

// Record the identity of an uploaded geometry stream on the cache entry: the
// producer-owned pointers, counts/strides, and the sampled content hash.  The
// two upload paths (per-command and shared-arena) stamped the same eleven
// fields by hand.
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
    // 32-byte interleaved vertex: vec3 position f32 @0, vec3 normal f32 @12,
    // vec4 color R8G8B8A8_UNORM @24, vec2 texcoord R16G16_SFLOAT @28.
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
SoVulkanGeometryCache::initialize(VmaAllocator vmaAllocator,
                                  SoVulkanBufferFactory * buffers,
                                  SoVulkanGeometryArena * arena)
{
  this->vmaAllocator_ = vmaAllocator;
  this->buffers_ = buffers;
  this->arena_ = arena;
}

void
SoVulkanGeometryCache::setDeferCallback(
    std::function<void(std::function<void()> &&)> cb)
{
  this->deferDestroy_ = std::move(cb);
}

void
SoVulkanGeometryCache::setErrorSink(std::function<void(const char *)> sink)
{
  this->emitError_ = std::move(sink);
}

VulkanCachedCommand &
SoVulkanGeometryCache::getOrCreate(const SoRenderCommand * command)
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
SoVulkanGeometryCache::markVisited(VulkanCachedCommand & entry,
                                   const SoRenderCommand * command,
                                   const uint32_t generation,
                                   const bool composite,
                                   const uint32_t compositeEpoch)
{
  entry.commandKey = command;
  entry.cacheGeneration = generation;
  if (composite) {
    entry.compositeEpoch = compositeEpoch;
  }
}

VulkanCachedCommand *
SoVulkanGeometryCache::find(const SoRenderCommand * command)
{
  const auto found = this->commandToIndex_.find(command);
  if (found == this->commandToIndex_.end()) return nullptr;
  return &this->entries_[found->second];
}

const VulkanCachedCommand *
SoVulkanGeometryCache::find(const SoRenderCommand * command) const
{
  const auto found = this->commandToIndex_.find(command);
  if (found == this->commandToIndex_.end()) return nullptr;
  return &this->entries_[found->second];
}

const VulkanCachedCommand *
SoVulkanGeometryCache::findDrawable(const SoRenderCommand & command) const
{
  if (!command.geometry.positions || command.geometry.vertexCount == 0) {
    return nullptr;
  }
  const VulkanCachedCommand * entry = this->find(&command);
  if (entry == nullptr || entry->vertexBuffer == VK_NULL_HANDLE) return nullptr;
  return entry;
}

void
SoVulkanGeometryCache::upload(VulkanCachedCommand & entry,
                              const SoRenderCommand & command)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;

  // Pack interleaved vertices with deterministic defaults for absent streams.
  // The buffer is a reusable scratch vector: resize() preserves capacity, so
  // the heap is only touched on the first (largest) upload that reaches this
  // size rather than on every geometry change.  The lock spans the packing
  // plus the synchronous uploads below, which read `vertices`.
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;

  std::lock_guard<std::mutex> scratchLock(this->uploadScratchMutex_);
  // Byte-sized scratch: 32 bytes per packed vertex.
  this->uploadScratch_.resize(static_cast<size_t>(vertexCount) * VULKAN_VERTEX_STRIDE);
  uint8_t * const vertices = this->uploadScratch_.data();
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
    vertexCreated = this->buffers_->createDeviceLocal(vertexBytes,
                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  entry.vertexBuffer,
                                                  entry.vertexMemory, vertices);
  }
  if (!vertexCreated) {
    vertexCreated = this->buffers_->create(vertexBytes,
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       entry.vertexBuffer, entry.vertexMemory,
                                       vertices);
  }
  if (!vertexCreated) {
    if (this->emitError_) {
      this->emitError_("uploadGeometry: failed to create vertex buffer");
    }
    return;
  }

  if (geometry.indexCount && geometry.indices) {
    const VkDeviceSize indexBytes =
      static_cast<VkDeviceSize>(geometry.indexCount) * sizeof(uint32_t);
    bool indexCreated = false;
    if (geometry.retained) {
      indexCreated = this->buffers_->createDeviceLocal(indexBytes,
                                                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   entry.indexBuffer,
                                                   entry.indexMemory,
                                                   geometry.indices);
    }
    if (!indexCreated) {
      indexCreated = this->buffers_->create(indexBytes,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        entry.indexBuffer, entry.indexMemory,
                                        geometry.indices);
    }
    if (!indexCreated) {
      if (this->emitError_) {
        this->emitError_("uploadGeometry: failed to create index buffer");
      }
      if (entry.vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(this->vmaAllocator_, entry.vertexBuffer,
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
SoVulkanGeometryCache::uploadShared(VulkanCachedCommand & entry,
                                    const SoRenderCommand & command,
                                    uint32_t blockId)
{
  auto * blockPtr = this->arena_->block(blockId);
  if (blockPtr == nullptr) {
    return false;
  }
  SoVulkanGeometryArena::Block & block = *blockPtr;
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
  if (!this->arena_->allocate(blockId, vertexBytes, vertexOffset)) {
    return false;
  }
  if (indexBytes != 0 &&
      !this->arena_->allocate(blockId, indexBytes, indexOffset)) {
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
SoVulkanGeometryCache::destroySubPixelResources(VulkanCachedCommand & entry)
{
  for (VulkanCachedCommand::VulkanSubPixelSlot & s : entry.subPixelSlots) {
    if (s.indexBuffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(this->vmaAllocator_, s.indexBuffer, s.indexMemory);
      s.indexBuffer = VK_NULL_HANDLE;
      s.indexMemory = nullptr;
    }
    if (s.indirectBuffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(this->vmaAllocator_, s.indirectBuffer, s.indirectMemory);
      s.indirectBuffer = VK_NULL_HANDLE;
      s.indirectMemory = nullptr;
    }
  }
  entry.subPixelSlots.clear();
  entry.subPixelHash = 0;
}

void
SoVulkanGeometryCache::deferDestroySubPixelResources(VulkanCachedCommand & entry)
{
  if (entry.subPixelSlots.empty()) return;
  std::vector<VulkanCachedCommand::VulkanSubPixelSlot> slots =
    std::move(entry.subPixelSlots);
  VmaAllocator vma = this->vmaAllocator_;
  if (this->deferDestroy_) {
    this->deferDestroy_([vma, slots]() mutable {
      for (VulkanCachedCommand::VulkanSubPixelSlot & s : slots) {
        if (s.indexBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, s.indexBuffer, s.indexMemory);
        }
        if (s.indirectBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, s.indirectBuffer, s.indirectMemory);
        }
      }
    });
  }
  entry.subPixelHash = 0;
}

void
SoVulkanGeometryCache::destroyEntry(VulkanCachedCommand & entry)
{
  if (entry.sharedBlockId != 0) {
    this->arena_->releaseBlock(entry.sharedBlockId);
  }
  else {
    if (entry.indexBuffer) {
      vmaDestroyBuffer(this->vmaAllocator_, entry.indexBuffer,
                       entry.indexMemory);
      entry.indexBuffer = VK_NULL_HANDLE;
      entry.indexMemory = nullptr;
    }
    if (entry.vertexBuffer) {
      vmaDestroyBuffer(this->vmaAllocator_, entry.vertexBuffer,
                       entry.vertexMemory);
      entry.vertexBuffer = VK_NULL_HANDLE;
      entry.vertexMemory = nullptr;
    }
  }
  for (VulkanCachedCommand::VulkanWideLineBuffer & slot : entry.wideLineBuffers) {
    slot.destroy(this->vmaAllocator_);
  }
  entry.wideLineBuffers.clear();
  // GPU-instanced wide-line endpoint buffer.  The deferred destroy path
  // (deferDestroyEntry) already released this; the synchronous path used by
  // invalidate() must too, or every instanced line command leaks its buffer +
  // memory past vkDestroyDevice (VUID-vkDestroyDevice-device-05137).
  if (entry.instancedLineBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(this->vmaAllocator_, entry.instancedLineBuffer,
                     entry.instancedLineMemory);
    entry.instancedLineBuffer = VK_NULL_HANDLE;
    entry.instancedLineMemory = nullptr;
  }
  this->destroySubPixelResources(entry);
  entry = VulkanCachedCommand();
}

void
SoVulkanGeometryCache::deferDestroyEntry(VulkanCachedCommand & entry)
{
  if (entry.vertexBuffer == VK_NULL_HANDLE &&
      entry.indexBuffer == VK_NULL_HANDLE &&
      entry.sharedBlockId == 0 &&
      entry.instancedLineBuffer == VK_NULL_HANDLE &&
      entry.subPixelSlots.empty() &&
      entry.wideLineBuffers.empty()) {
    entry = VulkanCachedCommand();
    return;
  }
  if (entry.sharedBlockId != 0) {
    const uint32_t sharedBlockId = entry.sharedBlockId;
    std::vector<VulkanCachedCommand::VulkanWideLineBuffer> wideLine =
      std::move(entry.wideLineBuffers);
    std::vector<VulkanCachedCommand::VulkanSubPixelSlot> subPixel =
      std::move(entry.subPixelSlots);
    const VkBuffer instancedLineBuffer = entry.instancedLineBuffer;
    const VmaAllocation instancedLineMemory = entry.instancedLineMemory;
    VmaAllocator vma = this->vmaAllocator_;
    if (this->deferDestroy_) {
      this->deferDestroy_([vma, wideLine, subPixel, instancedLineBuffer,
                           instancedLineMemory]() mutable {
        for (VulkanCachedCommand::VulkanWideLineBuffer & slot : wideLine) {
          slot.destroy(vma);
        }
        for (VulkanCachedCommand::VulkanSubPixelSlot & slot : subPixel) {
          if (slot.indexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(vma, slot.indexBuffer, slot.indexMemory);
          }
          if (slot.indirectBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(vma, slot.indirectBuffer, slot.indirectMemory);
          }
        }
        if (instancedLineBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, instancedLineBuffer, instancedLineMemory);
        }
      });
    }
    this->arena_->deferReleaseBlock(sharedBlockId);
    entry = VulkanCachedCommand();
    return;
  }
  VmaAllocator vma = this->vmaAllocator_;
  const VkBuffer vertexBuffer = entry.vertexBuffer;
  const VmaAllocation vertexMemory = entry.vertexMemory;
  const VkBuffer indexBuffer = entry.indexBuffer;
  const VmaAllocation indexMemory = entry.indexMemory;
  const VkBuffer instancedLineBuffer = entry.instancedLineBuffer;
  const VmaAllocation instancedLineMemory = entry.instancedLineMemory;
  std::vector<VulkanCachedCommand::VulkanWideLineBuffer> wideLine =
    std::move(entry.wideLineBuffers);
  std::vector<VulkanCachedCommand::VulkanSubPixelSlot> subPixel =
    std::move(entry.subPixelSlots);
  if (this->deferDestroy_) {
    this->deferDestroy_(
      [vma, vertexBuffer, vertexMemory, indexBuffer,
       indexMemory, instancedLineBuffer, instancedLineMemory, wideLine,
       subPixel]() mutable {
        for (VulkanCachedCommand::VulkanWideLineBuffer & slot : wideLine) {
          slot.destroy(vma);
        }
        for (VulkanCachedCommand::VulkanSubPixelSlot & slot : subPixel) {
          if (slot.indexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(vma, slot.indexBuffer, slot.indexMemory);
          }
          if (slot.indirectBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(vma, slot.indirectBuffer, slot.indirectMemory);
          }
        }
        if (instancedLineBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, instancedLineBuffer, instancedLineMemory);
        }
        if (indexBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, indexBuffer, indexMemory);
        }
        if (vertexBuffer != VK_NULL_HANDLE) {
          vmaDestroyBuffer(vma, vertexBuffer, vertexMemory);
        }
      });
  }
  entry = VulkanCachedCommand();
}

void
SoVulkanGeometryCache::invalidate()
{
  for (VulkanCachedCommand & entry : this->entries_) {
    this->destroyEntry(entry);
  }
  this->entries_.clear();
  this->commandToIndex_.clear();
}

void
SoVulkanGeometryCache::sweep(uint32_t generation)
{
  auto stale = [generation](const VulkanCachedCommand & entry) {
    return entry.cacheGeneration != generation;
  };
  bool anyStale = false;
  for (size_t idx = 0; idx < this->entries_.size(); ++idx) {
    if (stale(this->entries_[idx])) {
      this->deferDestroyEntry(this->entries_[idx]);
      anyStale = true;
    }
  }
  if (!anyStale) return;
  size_t write = 0;
  for (size_t idx = 0; idx < this->entries_.size(); ++idx) {
    if (!stale(this->entries_[idx])) {
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

void
SoVulkanGeometryCache::sweepComposite(uint32_t epoch)
{
  auto stale = [epoch](const VulkanCachedCommand & entry) {
    return entry.compositeEpoch != epoch;
  };
  bool anyStale = false;
  for (size_t idx = 0; idx < this->entries_.size(); ++idx) {
    if (stale(this->entries_[idx])) {
      this->deferDestroyEntry(this->entries_[idx]);
      anyStale = true;
    }
  }
  if (!anyStale) return;
  size_t write = 0;
  for (size_t idx = 0; idx < this->entries_.size(); ++idx) {
    if (!stale(this->entries_[idx])) {
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
