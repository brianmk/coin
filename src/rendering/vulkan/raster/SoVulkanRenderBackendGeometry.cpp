// src/rendering/vulkan/raster/SoVulkanRenderBackendGeometry.cpp
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

#include "rendering/vulkan/raster/SoVulkanRenderBackend.h"
#include "rendering/vulkan/raster/SoVulkanRenderBackendP.h"
#include "rendering/vulkan/common/core/SoVulkanShared.h"

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

} // namespace

// --- Geometry cache -------------------------------------------------------

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
SoVulkanRenderBackend::invalidateCache()
{
  this->geometryCache.invalidate();
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

  this->geometryCache.needsGeometryScratch().assign(
    static_cast<size_t>(std::max(0, drawlist.getNumCommands())), 0);
  // Start this frame's texture staging at the front of the shared staging
  // pool so pending uploads coalesce into one buffer (see SoVulkanTextureCache).
  this->textureCache.resetStaging();
  std::vector<uint8_t> & needsGeometry = this->geometryCache.needsGeometryScratch();
  int retainedUploads = 0;
  VkDeviceSize retainedUploadBytes = 0;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (!shouldProcessGeometry(command, overlaysOnly)) {
      continue;
    }
    const SoGeometryDesc & geometry = command.geometry;

    VulkanCachedCommand & entry = this->geometryCache.getOrCreate(&command);
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
    //  - Replayed frames (geometryContentUnchanged): no traversal ran, so
    //    pointer-identical geometry is bit-identical; skip the FNV walk.
    //  - Otherwise: always fall back to the sampled content hash.  This covers
    //    both per-frame arena streams that rewrite the same pointer in place
    //    (e.g. per-vertex colors) AND retained buffers that a producer edits in
    //    place.  The retained producer is NOT guaranteed to reallocate on
    //    rebuild: FreeCAD's SoBrepEdgeSet updates its coordinate/edge fields in
    //    place (same pointer, same counts, new data), so pointer/count identity
    //    alone reported the geometry unchanged after a Part::Box edit -- the
    //    vertex buffer was not re-uploaded and, because the wide-line instance
    //    buffer is keyed on this entry's content hash, the object's wide edges
    //    stayed at the pre-edit geometry.  The hash is bounded (sampled), and it
    //    now runs only on re-traverse frames, since a replayed frame skips it.
    const bool pointerIdentitySufficient = geometryContentUnchanged;
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

    VulkanCachedCommand & entry = this->geometryCache.getOrCreate(&command);
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
      this->geometryCache.deferDestroyEntry(entry);
      bool uploadedShared = false;
      if (geometry.retained && sharedBlockId != 0) {
        uploadedShared =
          this->geometryCache.uploadShared(entry, command, sharedBlockId);
        if (uploadedShared) {
          ++sharedUploads;
        }
      }
      if (!uploadedShared) {
        this->geometryCache.upload(entry, command);
      }
    }
    this->geometryCache.markVisited(entry, &command, generation, compositeSweep,
                                    this->overlayCompositeEpoch);

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
    if (compositeSweep) {
      // Release every traced triangle entry the composite pass did not visit
      // (only overlays and the non-triangle residue are stamped this pass), so
      // the RT backend's own copy is the only resident one.
      this->geometryCache.sweepComposite(this->overlayCompositeEpoch);
    }
    else {
      this->geometryCache.sweep(generation);
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
