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

#include "vk_mem_alloc.h"

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
  return SoVulkanShared::steadyNowUs();
}

bool vkGeometryBreadcrumbEnabled()
{
  return SoVulkanShared::breadcrumbsEnabled();
}

// Encode a float to a half (binary16).  Ranges outside the finite half range
// clamp to +/-inf; CAD texcoords are in [0,1] so this is exact in practice.
static inline uint16_t floatToHalf(float value)
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

// True when updateGeometryCache() should visit a command.  On an overlay-only
// render only SO_RENDERPASS_OVERLAY commands and the non-triangle residual
// geometry the RT backend did not trace are in scope; a full render visits
// every command with usable geometry.  Both cache passes shared this filter.
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
    geometry.vertexCount <= static_cast<uint32_t>(maxVertexCount());
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

  // Pack interleaved vertices with deterministic defaults for absent streams.
  // The buffer is a reusable member scratch vector: resize() preserves
  // capacity, so the heap is only touched on the first (largest) upload that
  // reaches this size rather than on every geometry change.  The lock spans the
  // packing plus the synchronous uploads below, which read `vertices`.
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;

  std::lock_guard<std::mutex> scratchLock(this->uploadScratchMutex);
  // Byte-sized scratch: 32 bytes per packed vertex.
  this->uploadScratch.resize(static_cast<size_t>(vertexCount) * VULKAN_VERTEX_STRIDE);
  uint8_t * const vertices = this->uploadScratch.data();
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
    vertexCreated = this->buffers.createDeviceLocal(vertexBytes,
                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  entry.vertexBuffer,
                                                  entry.vertexMemory, vertices);
  }
  if (!vertexCreated) {
    vertexCreated = this->buffers.create(vertexBytes,
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
      indexCreated = this->buffers.createDeviceLocal(indexBytes,
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                    entry.indexBuffer,
                                                    entry.indexMemory,
                                                    geometry.indices);
    }
    if (!indexCreated) {
      indexCreated = this->buffers.create(indexBytes,
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
  auto * blockPtr = this->geometryArena.block(blockId);
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
  if (!this->geometryArena.allocate(blockId, vertexBytes, vertexOffset)) {
    return false;
  }
  if (indexBytes != 0 &&
      !this->geometryArena.allocate(blockId, indexBytes, indexOffset)) {
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
    this->geometryArena.releaseBlock(entry.sharedBlockId);
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
  // GPU-instanced wide-line endpoint buffer.  The deferred destroy path
  // (deferDestroyCacheEntry) already released this; the synchronous path used
  // by invalidateCache() must too, or every instanced line command leaks its
  // buffer + memory past vkDestroyDevice (VUID-vkDestroyDevice-device-05137).
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
  this->textureCache.invalidate();
}

void
SoVulkanRenderBackend::updateGeometryCache(const SoDrawList & drawlist,
                                           const bool overlaysOnly,
                                           const bool geometryContentUnchanged)
{
  // The frame boundary was handled by beginFrame() at the entry point.

  // Release uploads left over from a frame that aborted between the cache
  // update and the flush/finalize step (e.g. a failed framebuffer create).
  // Their copies were never recorded.  The staged pixels live in the shared
  // staging pool (which persists across frames), so there is no per-upload
  // staging buffer to defer-destroy; just drop the pending list so the next
  // frame re-stages from scratch.
  this->textureCache.discardPending();

  const long cacheBcStart = vkGeometryBreadcrumbEnabled() ? vkGeometryBreadcrumbNowUs() : 0;
  int bcCommands = 0;
  int bcGeometryUploads = 0;
  int bcTexturePrepares = 0;
  size_t bcVertices = 0;
  size_t bcIndices = 0;

  const uint32_t generation = drawlist.getGeneration();

  // Overlay-composite mode (ray tracing active): the sweep must release the
  // traced triangle commands this backend no longer visits, but the draw-list
  // generation cannot key that sweep -- on a replayed (camera-only) frame the
  // retained list is not cleared, so the generation does not change and the
  // stale triangle entries would survive.  Use a dedicated epoch bumped once
  // per composite pass instead, stamped on every entry this pass visits.
  const bool compositeSweep = overlaysOnly && this->overlayCompositeMode;
  if (compositeSweep) {
    ++this->overlayCompositeEpoch;
  }

  this->needsGeometryScratch.assign(
    static_cast<size_t>(std::max(0, drawlist.getNumCommands())), 0);
  // Start this frame's texture staging at the front of the shared staging
  // pool so pending uploads coalesce into one buffer (see SoVulkanTextureCache).
  this->textureCache.resetStaging();
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
    // Change detection:
    //  - Retained geometry (SoGeometryDesc::retained): the producer guarantees
    //    the stream pointers change exactly when the content changes (shape
    //    tessellation reallocates the buffers on rebuild), so pointer/count
    //    identity alone is a correct change detector.  A per-frame content hash
    //    is redundant and doing it defeats the retained contract those
    //    producers rely on.  Skip the FNV walk entirely here.
    //  - Replayed frames (geometryContentUnchanged): no traversal ran, so
    //    pointer-identical geometry is bit-identical; also skip.
    //  - Otherwise (per-frame arena streams that rewrite the same pointer in
    //    place, e.g. per-vertex colors): fall back to the sampled content hash
    //    to catch in-place edits.
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
    sharedBlockId = this->geometryArena.allocateBlock(retainedUploadBytes + 4096u);
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

  this->textureCache.discardPending();

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    // Overlay-only renders (ray-tracing compositing) draw SO_RENDERPASS_OVERLAY
    // commands (nav cube, axis cross, selection/hover highlights) plus the
    // non-triangle residue the RT backend did not trace (Brep edge lines,
    // point markers, polylines): those must be uploaded so the composite can
    // rasterize them onto the traced surface.  Pure triangle geometry is
    // already traced and skipping it here keeps the composite cheap.
    if (!shouldProcessGeometry(command, overlaysOnly)) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;
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
    if (compositeSweep) {
      entry.compositeEpoch = this->overlayCompositeEpoch;
    }

    // Texture lookup/content-change detection, defer-destroy of the stale
    // image and staging of a new upload all live in the cache.
    if (this->textureCache.prepareCommand(command, generation)) {
      ++bcTexturePrepares;
    }
  }

  if (sharedBlockId != 0 && sharedUploads == 0) {
    this->geometryArena.releaseBlock(sharedBlockId);
  }

  // Evict entries that were not visited this frame: their command has
  // disappeared from the draw list (or its pointer is no longer part of
  // this frame's arena).  Entries surviving eviction keep their index
  // identity, so rebuild the pointer maps from the stored commandKey.
  // Destruction is deferred: a pending frame may still reference the
  // evicted buffers/images.
  //
  // A transient overlay-only render skips the sweep: its traversal
  // deliberately visits only overlay commands, so a sweep would evict the
  // entire scene cache and force a full re-upload on the next full render.
  // In overlay-composite mode (ray tracing active, so this backend never
  // performs a full render again) the sweep DOES run: it releases the traced
  // triangle geometry the RT backend already owns, instead of holding a
  // second resident copy for the lifetime of the RT session.
  if (!overlaysOnly || this->overlayCompositeMode) {
    // `stale` decides whether an entry is dropped.  A full render and a
    // transient overlay render key on the draw-list generation; an
    // overlay-composite pass keys on the composite epoch (see above) because
    // the retained list's generation does not advance on replayed frames.
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
      // Release every traced triangle entry the composite pass did not visit
      // (only overlays and the non-triangle residue are stamped this pass), so
      // the RT backend's own copy is the only resident one.
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
    // Evict unvisited texture entries and re-resolve the pending-upload
    // indices against the compacted cache (the sweep owns both).
    this->textureCache.sweep(generation);
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
