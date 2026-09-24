#ifndef COIN_SOVULKANGEOMETRYCACHE_H
#define COIN_SOVULKANGEOMETRYCACHE_H

#include <rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.h>
#include <rendering/SoVulkanGeometryArena.h>

#include <Inventor/rendering/SoRenderIR.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

class SoVulkanBufferFactory;

/*!
  \brief Cached GPU geometry for one retained SoRenderCommand.

  Vulkan buffers are packed per command: one interleaved vertex buffer (fixed
  32-byte stride: position f32x3 + normal f32x3 + color R8G8B8A8 + texcoord
  R16G16) and one optional uint32 index buffer.  Unlike the GL backend, the
  Vulkan backend always uses the same vertex layout so a single static
  vertex-input state can be shared by every pipeline.
*/
struct VulkanCachedCommand {
  VkBuffer vertexBuffer = VK_NULL_HANDLE;
  VmaAllocation vertexMemory = nullptr;
  VkBuffer indexBuffer = VK_NULL_HANDLE;
  VmaAllocation indexMemory = nullptr;
  VkDeviceSize vertexOffset = 0;
  VkDeviceSize indexOffset = 0;
  uint32_t sharedBlockId = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;

  // GPU-instanced wide-line endpoint stream (object space, static per
  // geometry): four vec4 per segment (p0, p1, c0, c1).  Built once when the
  // geometry is first seen and reused until the content hash changes; the
  // vertex shader expands each segment on the GPU, so no per-frame CPU work
  // or quad upload happens for this command.  A null buffer means the build
  // failed (or the command is not eligible) and the CPU expansion is used.
  VkBuffer instancedLineBuffer = VK_NULL_HANDLE;
  VmaAllocation instancedLineMemory = nullptr;
  uint32_t instancedLineSegmentCount = 0;
  uint64_t instancedLineHash = 0;

  // GPU sub-pixel geometry LOD (raster only).  While the camera moves the
  // backend compacts an eligible triangle command's index list on the GPU,
  // dropping triangles whose projected screen area falls below a threshold,
  // and draws the survivors with vkCmdDrawIndexedIndirect.  One slot per
  // in-flight frame: the compacted index buffer is rewritten every frame, so
  // it must not alias a buffer a still-executing frame may read.  Slots are
  // built lazily by the pre-pass and kept until the geometry content changes.
  struct VulkanSubPixelSlot {
    VkBuffer indexBuffer = VK_NULL_HANDLE;      // compacted indices
    VmaAllocation indexMemory = nullptr;
    VkBuffer indirectBuffer = VK_NULL_HANDLE;   // VkDrawIndexedIndirectCommand
    VmaAllocation indirectMemory = nullptr;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    uint32_t maxIndices = 0;                    // capacity of indexBuffer
    // Frame ordinal this slot was compacted for (0 = not ready).  The draw
    // path uses the slot only when it matches the frame being recorded.
    uint64_t readyFrame = 0;
  };
  std::vector<VulkanSubPixelSlot> subPixelSlots;
  // Content hash of the geometry the slots were built from; a change rebuilds
  // them (mirrors instancedLineHash).
  uint64_t subPixelHash = 0;
  //! Geometry-LOD fallback diagnostics.  The oversized-cap and storage-range
  //! conditions are per command and geometry-static, so they are latched on
  //! the cache entry: thread-safe (no shared function-local static) and a
  //! second, distinct offending command still gets its own warning instead of
  //! being suppressed by the first.
  bool warnedGeomLodCap = false;
  bool warnedGeomLodRange = false;

  // CPU-expanded wide-line quads (per-frame content; line width > 1 or a
  // stipple pattern).  One host-visible scratch buffer per in-flight frame
  // slot: the quad expansion is a function of the projection matrix, so the
  // content is rewritten every frame, and the rewrite for frame N +
  // maxFramesInFlight may not clobber data a still-executing frame N reads.
  // The slot for the current frame index is selected by beginFrame(), which
  // waits the slot's fence before the slot is reused.
  struct VulkanWideLineBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    //! Persistent host mapping of `memory` (VK_MEMORY_PROPERTY_HOST_VISIBLE |
    //! HOST_COHERENT), established once at (re)creation and kept alive so the
    //! steady-state per-frame update is a plain memcpy instead of a per-command
    //! vkMapMemory/vkUnmapMemory pair.  map/unmap dominates the wide-line cost
    //! on line-heavy scenes (each call ~50us; a dozen visible edge commands per
    //! frame is ~1ms+).  Cleared to null whenever `memory` is destroyed; freeing
    //! memory implicitly unmaps, so no explicit unmap is needed at teardown.
    void * mapped = nullptr;
    VkDeviceSize size = 0;
    //! Fingerprint of the geometry/view/proj/width/viewport that produced the
    //! quads in this slot.  A match means the buffer already holds the exact
    //! quads for the current frame, so expandWideLines() skips the expansion
    //! -- the dominant per-frame CPU cost for line/edge-heavy scenes on a
    //! retained draw list with an unchanged camera.
    uint64_t expandFingerprint = 0;
    uint32_t expandVertexCount = 0;

    // Release the slot's buffer + memory and reset it to the empty state.
    // Singular teardown used by both the synchronous and the deferred cache
    // destroy paths.  Defined out-of-line in SoVulkanRenderBackendGeometry.cpp
    // because vmaDestroyBuffer needs the full VMA API, which this header
    // deliberately does not include.
    void destroy(VmaAllocator allocator);
  };
  std::vector<VulkanWideLineBuffer> wideLineBuffers;
  uint32_t wideLineVertexCount = 0;

  // Command that last touched this entry (per-frame arena pointer; used
  // only as an identity key for map rebuilds after cache eviction).
  const SoRenderCommand * commandKey = nullptr;

  // Identity keys mirroring the producer-owned storage of the last upload.
  const float * posKey = nullptr;
  const float * normalKey = nullptr;
  const float * colorKey = nullptr;
  const float * texcoordKey = nullptr;
  const uint32_t * idxKey = nullptr;
  uint32_t vertexStride = 0;
  uint32_t texcoordStride = 0;
  uint32_t normalCount = 0;
  uint32_t cacheGeneration = 0;
  // Visit stamp for the overlay-composite sweep.  While ray tracing owns the
  // scene, overlays-only frames cannot key eviction on the draw-list
  // generation (a replayed retained list never advances it), so this epoch --
  // bumped once per composite pass -- marks the entries the pass visited;
  // everything else (the traced triangle commands) is released.
  uint32_t compositeEpoch = 0;
  // Content hash of the uploaded streams: pointer identity alone cannot
  // detect in-place edits (the per-frame arena hands out the same pointers
  // for unchanged layouts), which would otherwise serve stale geometry.
  uint64_t contentHash = 0;

  // Pipeline-resolution fast path (getOrCreatePipeline()).  The exact
  // PipelineKey resolved for this command last is stored verbatim, plus the
  // handle it produced.  A match (cheap field-by-field equality, no hashing)
  // skips rebuilding the key and the SoVulkanPipelineCache map lookup for
  // unchanged commands.  The entry lives and dies with the
  // geometry cache, which invalidateCache() clears together with the
  // pipeline cache, so these fields never outlive the handles they name.
  PipelineKey resolvedKey;
  VkPipeline resolvedPipeline = VK_NULL_HANDLE;
  bool hasResolvedPipeline = false;
};

/*!
  \brief Per-command GPU geometry cache.

  Owns the retained-geometry cache entries (`VulkanCachedCommand`), their
  lookup map, the upload paths and the eviction sweep that were inline in
  SoVulkanRenderBackend.  Borrows the device/VMA handles, the shared buffer
  factory and the geometry arena from the backend, and routes destruction
  through the deferred-destruction ring so a still-executing submission that
  may read a buffer drains first.
*/
class SoVulkanGeometryCache {
public:
  void initialize(VmaAllocator vmaAllocator, SoVulkanBufferFactory * buffers,
                  SoVulkanGeometryArena * arena);
  void setDeferCallback(std::function<void(std::function<void()> &&)> cb);
  void setErrorSink(std::function<void(const char *)> sink);

  // --- Lookup / lifetime ------------------------------------------------
  VulkanCachedCommand & getOrCreate(const SoRenderCommand * command);
  VulkanCachedCommand * find(const SoRenderCommand * command);
  const VulkanCachedCommand * find(const SoRenderCommand * command) const;
  // Drawable lookup: null when the command has no uploaded vertex buffer.
  const VulkanCachedCommand * findDrawable(const SoRenderCommand & command) const;

  std::vector<VulkanCachedCommand> & entries() { return entries_; }
  const std::vector<VulkanCachedCommand> & entries() const { return entries_; }

  // Release one entry's GPU resources (shared block, per-command buffers,
  // wide-line/instanced-line/sub-pixel buffers) and reset it.
  void destroyEntry(VulkanCachedCommand & entry);
  // destroyEntry() through the deferred-destruction ring.
  void deferDestroyEntry(VulkanCachedCommand & entry);
  // Record that the entry was visited by the current frame: store the command
  // identity used to rebuild the lookup map after eviction and stamp the visit
  // so the matching sweep keeps it.  `composite` selects the overlay-composite
  // epoch stamp (used while ray tracing owns the scene) instead of the
  // draw-list generation.
  void markVisited(VulkanCachedCommand & entry, const SoRenderCommand * command,
                   uint32_t generation, bool composite,
                   uint32_t compositeEpoch);
  // Drop every entry (synchronously); the backend's invalidateCache() calls
  // this next to SoVulkanTextureCache::invalidate().
  void invalidate();

  // --- Upload -----------------------------------------------------------
  void upload(VulkanCachedCommand & entry, const SoRenderCommand & command);
  bool uploadShared(VulkanCachedCommand & entry, const SoRenderCommand & command,
                    uint32_t blockId);

  // --- Eviction sweep ---------------------------------------------------
  // Drop entries whose visit stamp does not match (a full render / transient
  // overlay keys on the draw-list generation; an overlay-composite pass keys
  // on the composite epoch).  Destruction is deferred; surviving entries keep
  // their index identity and the lookup map is rebuilt from commandKey.
  void sweep(uint32_t generation);
  void sweepComposite(uint32_t epoch);

  // Release one entry's sub-pixel buffers (and descriptor set) -- synchronously
  // or through the deferred ring.  Called by the backend's geometry-LOD
  // pre-pass when a command's compaction slots are rebuilt.
  void destroySubPixelResources(VulkanCachedCommand & entry);
  void deferDestroySubPixelResources(VulkanCachedCommand & entry);

  // Scratch reused by the backend's updateGeometryCache() loop.
  std::vector<uint8_t> & needsGeometryScratch() { return needsGeometryScratch_; }

private:
  VmaAllocator vmaAllocator_ = nullptr;
  SoVulkanBufferFactory * buffers_ = nullptr;
  SoVulkanGeometryArena * arena_ = nullptr;
  std::function<void(std::function<void()> &&)> deferDestroy_;
  std::function<void(const char *)> emitError_;

  std::vector<VulkanCachedCommand> entries_;
  std::unordered_map<const SoRenderCommand *, size_t> commandToIndex_;

  // Reusable scratch packing buffer for upload().  Stores the 32-byte packed
  // layout (see VULKAN_VERTEX_STRIDE), so it is byte-addressed.  Guarded by
  // uploadScratchMutex_ because a shared shape may be uploaded from more than
  // one thread.
  std::vector<uint8_t> uploadScratch_;
  std::mutex uploadScratchMutex_;
  // Per-draw "needs geometry upload" flags for updateGeometryCache().
  std::vector<uint8_t> needsGeometryScratch_;
};

#endif // COIN_SOVULKANGEOMETRYCACHE_H
