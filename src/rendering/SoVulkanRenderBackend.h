// src/rendering/SoVulkanRenderBackend.h

#ifndef COIN_SOVULKANRENDERBACKEND_H
#define COIN_SOVULKANRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include "rendering/SoVulkanShared.h"
#include "rendering/SoVulkanResult.h"
// VMA opaque handles only; vk_mem_alloc.h is included by the allocating .cpp
// files.  VK_DEFINE_HANDLE matches VMA's typedef, so either order is safe.
#ifndef AMD_VULKAN_MEMORY_ALLOCATOR_H
VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)
#endif
#include "rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRecordContext.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderPassCache.h"
#include "rendering/SoVulkanGpuTimers.h"

#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// PipelineKey and the SoVulkanPipelineCache store live in SoVulkanPipelineCache.h.

/*!
  \brief Cached GPU geometry for one retained SoRenderCommand.

  One interleaved vertex buffer (fixed 48-byte stride: position + normal +
  color + texcoord) and one optional uint32 index buffer.  The layout is fixed
  so a single static vertex-input state is shared by every pipeline.
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

  // GPU-instanced wide-line endpoint stream (object space, static): four vec4
  // per segment (p0, p1, c0, c1) per content hash; null => CPU expansion.
  VkBuffer instancedLineBuffer = VK_NULL_HANDLE;
  VmaAllocation instancedLineMemory = nullptr;
  uint32_t instancedLineSegmentCount = 0;
  uint64_t instancedLineHash = 0;

  // GPU sub-pixel geometry LOD (raster only): compact an eligible triangle
  // command's indices on the GPU (drop triangles below a projected-area
  // threshold) and draw with vkCmdDrawIndexedIndirect.  One slot per in-flight
  // frame; built lazily by the pre-pass until the content changes.
  struct VulkanSubPixelSlot {
    VkBuffer indexBuffer = VK_NULL_HANDLE;      // compacted indices
    VmaAllocation indexMemory = nullptr;
    VkBuffer indirectBuffer = VK_NULL_HANDLE;   // VkDrawIndexedIndirectCommand
    VmaAllocation indirectMemory = nullptr;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    uint32_t maxIndices = 0;                    // capacity of indexBuffer
    // Frame ordinal this slot was compacted for (0 = not ready).
    uint64_t readyFrame = 0;
  };
  std::vector<VulkanSubPixelSlot> subPixelSlots;
  // Content hash of the geometry the slots were built from; changes rebuild.
  uint64_t subPixelHash = 0;
  //! Geometry-LOD fallback diagnostics, latched per entry (thread-safe, and
  //! each offending command warns once rather than being suppressed).
  bool warnedGeomLodCap = false;
  bool warnedGeomLodRange = false;

  // CPU-expanded wide-line quads (line width > 1 or stipple): one host-visible
  // buffer per in-flight frame slot, rewritten every frame off the projection;
  // beginFrame() waits the fence, so N + maxFramesInFlight never clobbers N.
  struct VulkanWideLineBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    //! Persistent HOST_VISIBLE|HOST_COHERENT mapping set at (re)creation so
    //! per-frame updates are a plain memcpy (map/unmap ~50us/call dominates
    //! wide-line cost); nulled when `memory` is destroyed (freeing unmaps).
    void * mapped = nullptr;
    VkDeviceSize size = 0;
    //! Fingerprint of geometry/view/proj/width/viewport; a match means the slot
    //! already holds this frame's quads, so the (dominant CPU) expansion is skipped.
    uint64_t expandFingerprint = 0;
    uint32_t expandVertexCount = 0;

    // Release the slot's buffer + memory and reset it.  Out-of-line in
    // SoVulkanRenderBackendGeometry.cpp (vmaDestroyBuffer needs the full VMA API).
    void destroy(VmaAllocator allocator);
  };
  std::vector<VulkanWideLineBuffer> wideLineBuffers;
  uint32_t wideLineVertexCount = 0;

  // Command that last touched this entry (arena pointer; map-rebuild key).
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
  // Visit stamp for the overlay-composite sweep: a replayed retained list never
  // advances the draw-list generation, so this epoch marks visited entries.
  uint32_t compositeEpoch = 0;
  // Content hash of the uploaded streams: pointer identity misses in-place edits.
  uint64_t contentHash = 0;
  // Retained-IR build id of the uploaded streams (SoGeometryDesc::retainedGeneration).  Pointer
  // identity alone misses a rebuild that reused the freed streams' address, which the allocator
  // routinely does; this changes whenever the producer tessellated anew.
  uint64_t geometryGeneration = 0;

  // Pipeline-resolution fast path (getOrCreatePipeline()): last PipelineKey +
  // handle stored verbatim; a field match skips rebuild + cache lookup.
  PipelineKey resolvedKey;
  VkPipeline resolvedPipeline = VK_NULL_HANDLE;
  bool hasResolvedPipeline = false;
};

/*! \brief Cached GPU texture for one retained command's SoTextureData. */
struct VulkanCachedTexture {
  VkImage image = VK_NULL_HANDLE;
  // Backing device memory owned by VMA; image + allocation destroyed together.
  VmaAllocation allocation = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  // Owning pool of the set (append-only: return here, not to the active pool).
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

  // Command that last touched this entry (arena pointer; map-rebuild key).
  const SoRenderCommand * commandKey = nullptr;

  // Identity of the last upload.
  const unsigned char * pixelsKey = nullptr;
  int width = 0;
  int height = 0;
  int numComponents = 0;
  SoTextureFilter minFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureFilter magFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureWrap wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureWrap wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureModel model = SO_TEXTURE_MODEL_MODULATE;
  uint32_t cacheGeneration = 0;
  // Content hash of the uploaded pixels: pointer identity misses in-place edits.
  uint64_t contentHash = 0;
};

/*! \brief Minimal Vulkan executor for retained DrawList IR. */
class SoVulkanRenderBackend : public SoRenderBackend {
public:
  SoVulkanRenderBackend();
  ~SoVulkanRenderBackend() override;

  const char * getName() const override;
  SbBool initialize(const SoRenderBackendInitParams & params) override;
  void shutdown() override;
  SbBool render(const SoDrawList & drawlist,
                const SoRenderParams & params) override;

  /*!
    \brief Record the draw list into a caller-owned command buffer/render pass.

    The caller has already begun \a commandBuffer and begun \a renderPass with
    a compatible framebuffer, and owns submission/presentation; this path does
    not begin/end either or submit to the queue.  Used by embedding surfaces
    such as QVulkanWindow.

    The per-frame setup, pending texture uploads and GPU geometry-LOD pre-pass
    are handled internally, so this is the single external entry point.
    Vulkan forbids transfer/compute inside a render pass, so the pre-pass is
    recorded into a backend-owned transient buffer before the frame draws and
    submitted after (see beginExternalPrepass()/submitExternalPrepass()); the
    caller's pass sees the uploaded textures and compacted geometry, and CPU
    recording overlaps the previous GPU frame.
  */
  SbBool renderExternal(const SoDrawList & drawlist,
                        const SoRenderParams & params,
                        VkCommandBuffer commandBuffer,
                        VkRenderPass renderPass,
                        VkFramebuffer framebuffer);

  /*!
    \brief Reset the current frame's GPU-timestamp queries on the caller's
    command buffer.

    Only meaningful under COIN_VULKAN_GPU_TIMING; an external embedder must call
    this before vkCmdBeginRenderPass (vkCmdResetQueryPool is illegal inside a
    pass).  A no-op when timing is disabled.
  */
  void resetExternalGpuQueries(VkCommandBuffer commandBuffer);

  /*!
    \brief Record only the overlay pass (e.g. the navigation cube) into a
    caller-owned command buffer/render pass.  Used when another renderer draws
    the scene and this backend rasterizes only the overlays on top.
  */
  SbBool renderExternalOverlay(const SoDrawList & drawlist,
                               const SoRenderParams & params,
                               VkCommandBuffer commandBuffer,
                               VkRenderPass renderPass);

  /*!
    \brief Composite only the overlay pass (e.g. the navigation cube) into
    the render target.

    Offscreen counterpart of renderExternalOverlay(): one-shot render into
    params.renderTarget touching only SO_RENDERPASS_OVERLAY commands, layerable
    over a previously rendered frame.  No-op if there are none.
  */
  SbBool renderOverlaysOnly(const SoDrawList & drawlist,
                            const SoRenderParams & params);

  /*!
    \brief Declare that this backend only composites overlays and residual
    geometry on top of a frame drawn by another renderer.

    While that composite mode is active the manager drives this backend through
    renderExternalOverlay()/renderOverlaysOnly() only, so updateGeometryCache()
    also sweeps stale entries on overlays-only frames, evicting triangle
    commands this backend no longer visits.  Must stay false for a backend that
    also does full raster renders, whose cache must survive an overlay pass.
  */
  void setOverlayCompositeMode(SbBool enabled);

  /*!
    \brief Declare how many recorded frames the caller may keep in flight.

    Drives the deferred-destruction batch count and lighting UBO ring size:
    resources replaced while recording frame N are released when frame
    N + \a count begins.  Concurrent submitters must set this before the first
    render call.  Defaults to 3 (QVulkanWindow's default concurrency).
  */
  void setMaxFramesInFlight(uint32_t count);

  /*!
    \brief Path of a persistent (on-disk) Vulkan pipeline cache.

    When non-empty, initialize() loads it and shutdown() writes the driver's
    cache blob back, so lazily-created pipeline variants survive a restart.
    Advisory: a missing/corrupt/stale file is rejected in favor of an empty
    cache.  The app owns the path and directory; set it before initialize().
  */
  void setPipelineCachePath(const std::string & path);

  /*!
    \brief Configure Vulkan-only display overlays (wireframe/point edges and
    color).  Deliberately backend state, not SoRenderParams, so GL never sees them.
  */
  void setWireframeOverlay(SbBool enabled);
  void setPointsOverlay(SbBool enabled);
  void setTessellationOverlay(SbBool enabled);
  void setEdgeColor(const SbColor4f & color);

  /*!
    \brief Provide the authoritative viewer lighting (GL host -> raster backend).

    \a lighting is the camera-anchored world-space light set (headlight,
    backlight, fill) plus intensity-scaled ambient.  A non-empty list overrides
    the per-command IR capture, matching Coin GL; empty restores the
    per-command IR lighting.
  */
  void setSceneLights(const SoLightingData & lighting);

private:
  // --- Initialization helpers -------------------------------------------
  bool createCommandPool();
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createLightingUniformBuffer();
  bool createLightingConstBuffer();
  bool createLightingDescriptorSet();
  bool createPipelineLayout();
  bool createShaders(VkShaderModule & vertexModule,
                     VkShaderModule & fragmentModule);
  bool createWideLineShaders();
  bool createSubPixelCullPipeline();
  bool createBackgroundResources();
  bool createPipelineCache();
  // Wrap a SPIR-V blob in a VkShaderModule (shared helper).
  bool createShaderModule(const uint32_t * code, size_t count,
                          VkShaderModule & module);
  bool createBackgroundPipeline(const SoVulkanRenderTarget & target,
                                VkRenderPass renderPass,
                                VkPipeline & pipeline);
  // Create a graphics pipeline from the caller's varying state (stages/vertex
  // input/input assembly/raster/depth-stencil/blend); fixed shared state here.
  VkPipeline createGraphicsPipeline(
    VkPipelineLayout layout, VkRenderPass renderPass,
    const VkPipelineShaderStageCreateInfo stages[2],
    const VkPipelineVertexInputStateCreateInfo & vertexInput,
    const VkPipelineInputAssemblyStateCreateInfo & inputAssembly,
    const VkPipelineRasterizationStateCreateInfo & rasterization,
    VkSampleCountFlagBits sampleCount,
    const VkPipelineDepthStencilStateCreateInfo & depthStencil,
    const VkPipelineColorBlendAttachmentState & blendAttachment);
  void recordBackground(const SoRenderParams & params,
                        const SoVulkanRenderTarget & target,
                        VkRenderPass renderPass,
                        VulkanRecordContext & ctx);
  bool getOrCreatePipeline(const SoRenderCommand & command,
                           const SoVulkanRenderTarget & target,
                           VkRenderPass renderPass,
                           VkPipeline & pipeline,
                           bool transparent,
                           int fillModeOverride = -1,
                           bool overlayPass = false,
                           VulkanCachedCommand * cacheEntry = nullptr);

  // --- Per-draw lighting ------------------------------------------------
  // Write each distinct SoLightingHandle's constant block into the current
  // frame's lighting ring and build lightingSlotOffsets; before recording draws.
  bool updateLightingSetup(const SoDrawList & drawlist);
  void updateLightingUniforms(const SoDrawList & drawlist,
                              const SoRenderCommand & command,
                              const SoRenderParams & params,
                              VkDeviceSize uboOffset,
                              bool unlit = false,
                              const float * projFloats = nullptr);
  // Dynamic byte offset into the lighting ring for a command's handle (0 if none).
  VkDeviceSize lightingOffsetFor(const SoRenderCommand & command) const;

  // --- Geometry cache ---------------------------------------------------
  void invalidateCache();
  // \a geometryContentUnchanged: replay-frame geometry is bit-identical, skip re-hash.
  void updateGeometryCache(const SoDrawList & drawlist, bool overlaysOnly = false,
                           bool geometryContentUnchanged = false);
  VulkanCachedCommand & getOrCreateCache(const SoRenderCommand * command);
  // GPU cache entry for a drawable command, or nullptr if absent/geometryless.
  const VulkanCachedCommand * findCachedDrawable(
    const SoRenderCommand & command) const;
  void uploadGeometry(VulkanCachedCommand & entry,
                      const SoRenderCommand & command);
  bool uploadGeometryShared(VulkanCachedCommand & entry,
                            const SoRenderCommand & command,
                            uint32_t blockId);
  void destroyCacheEntry(VulkanCachedCommand & entry);

  struct VulkanGeometryBlock {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    void * mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize used = 0;
    uint32_t refCount = 0;
  };

  uint32_t allocateGeometryBlock(VkDeviceSize capacity);
  bool allocateGeometryArena(uint32_t blockId, VkDeviceSize size,
                             VkDeviceSize & offset);
  // Destroy a geometry block's buffer + memory and reset it to empty.
  void releaseGeometryBlockResources(VulkanGeometryBlock & block);
  void releaseGeometryBlock(uint32_t blockId);
  void deferReleaseGeometryBlock(uint32_t blockId);
  void destroyAllGeometryBlocks();

  // --- Texture cache ----------------------------------------------------
  bool createWhiteTexture();
  void invalidateTextureCache();
  void destroyTextureEntry(VulkanCachedTexture & entry);
  VulkanCachedTexture & getOrCreateTexture(const SoRenderCommand * command);

  // One texture awaiting its GPU-side staging copy.  Host side (image, memory,
  // staging) prepared up front; copies go into the frame command buffer
  // (own-queue) or one transient buffer (external) per frame.  The command
  // pointer is kept so a post-eviction index can be re-resolved before use.
  struct PendingTextureUpload {
    size_t index = 0;
    const SoRenderCommand * command = nullptr;
    const SoTextureData * texture = nullptr;
    // Byte offset into the shared staging pool where the pixels were staged.
    VkDeviceSize stagingOffset = 0;
    VkDeviceSize stagingBytes = 0;
  };

  // One resolvable draw unit produced by buildWorkItems(): it pre-assigns each
  // item a disjoint `slotBase` (first of `count` UBO/instance-model ring slots).
  // Workers only read these items.  `commands` is one pointer (count == 1) or
  // `count` pointers sharing geometry/material (one instanced draw).
  struct VulkanWorkItem {
    const SoRenderCommand * single = nullptr;     // count == 1 draw
    const SoRenderCommand * const * commands = nullptr; // count > 1 batch array
    int count = 1;
    uint32_t slotBase = 0;
    bool transparent = false;
    bool recordToSecondary = false; // opaque pass → secondary cmd buffer (M1c)
    bool overlayPass = false;      // SO_RENDERPASS_OVERLAY screen-space draw
    int fillModeOverride = -1;     // wireframe/point redraw fill mode, or -1
    const float * uniformColorOverride = nullptr;
    // Pre-resolved state (filled by buildWorkItems()) so the record path avoids
    // re-walking the pipeline cache / texture-set map / lighting map.
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet textureSet = VK_NULL_HANDLE;
    uint32_t lightingDynamicOffset = 0;
  };

  bool buildWorkItems(const SoDrawList & drawlist,
                      const SoRenderParams & params,
                      bool wireframeOverlay, bool pointsOverlay,
                      bool tessellationOverlay,
                      const float * overlayColor,
                      VkRenderPass renderPass,
                      std::vector<VulkanWorkItem> & out);
  // Record one pre-assigned work item into ctx.buffer (single draw or a batch).
  void recordWorkItem(const SoDrawList & drawlist,
                      const SoRenderParams & params,
                      const SoVulkanRenderTarget & target,
                      VkRenderPass renderPass,
                      const VulkanWorkItem & item,
                      VulkanRecordContext & ctx);
  // M1d pool: build persistent worker threads, partition+dispatch, join.
  bool buildRecordPool();
  void shutdownRecordPool();
  void recordJobWorker(size_t workerIndex);
  bool recordSecondaryChunk(VulkanRecordContext & ctx,
                            const SoDrawList & drawlist,
                            const SoRenderParams & params,
                            const SoVulkanRenderTarget & target,
                            VkRenderPass renderPass,
                            const std::vector<const VulkanWorkItem *> & items,
                            VkCommandBuffer secondary,
                            VkFramebuffer framebuffer);
  VkCommandBuffer workerSecondary(uint32_t frameSlot, uint32_t worker);
  VkCommandBuffer parallelCurrentSecondary(uint32_t worker);
  bool prepareTextureUpload(VulkanCachedTexture & entry,
                            const SoTextureData & texture,
                            VkDeviceSize & stagingOffset,
                            VkDeviceSize & stagingBytes);
  void recordTextureUpload(VkCommandBuffer commandBuffer,
                           const VulkanCachedTexture & entry,
                           const SoTextureData & texture,
                           VkBuffer staging,
                           VkDeviceSize stagingOffset);
  bool finalizeTexture(VulkanCachedTexture & entry,
                       const SoTextureData & texture);
  // Record pending staging -> image copies into \a commandBuffer (must not be
  // inside a render pass) and finalize host-side resources (view, sampler,
  // descriptor set, content stamp) so draws can bind them.
  void recordPendingTextureUploadsInto(VkCommandBuffer commandBuffer);
  bool recordPendingTextureUploads();
  void finalizePendingTextureUploads();
  // Legacy external fallback: submit copies in a dedicated buffer, drain the queue.
  SoVulkan::Result flushPendingTextureUploadsExternal();
  bool createSampler(SoTextureFilter minFilter, SoTextureFilter magFilter,
                     SoTextureWrap wrapS, SoTextureWrap wrapT,
                     VkSampler & sampler);
  // Format for an N-component SoTextureData image.  R8/R8G8/R8G8B8_UNORM are
  // optional sampleable formats, so without SAMPLED_IMAGE support the upload
  // expands to R8G8B8A8_UNORM on the host with matching channel values.
  VkFormat effectiveTextureFormat(const int numComponents) const;
  bool allocateTextureDescriptorSet(VkImageView view, VkSampler sampler,
                                    VkDescriptorSet & set);
  bool ensureDescriptorPoolSpace();
  VkDescriptorSet resolveTextureSet(const SoRenderCommand & command);

  // --- Render recording ---------------------------------------------------
  // Every record* helper touches only its VulkanRecordContext, so worker threads
  // can each record their own buffer.
  bool beginCommandBuffer();
  VkCommandBuffer currentCommandBuffer();
  VkCommandBuffer currentSecondaryCommandBuffer();
  void recordClear(const SoRenderParams & params,
                   const SoVulkanRenderTarget & target,
                   bool colorClearedByLoad,
                   bool depthClearedByLoad,
                   VulkanRecordContext & ctx);
  // True if the clear region covers the whole target (CLEAR loadOps, no vkCmdClear).
  bool isFullTargetClear(const SoRenderParams & params,
                         const SoVulkanRenderTarget & target) const;
  // Byte offset of a per-draw UBO/instance-model ring slot within the frame's half.
  VkDeviceSize uboSlotOffset(uint32_t slotIndex) const;
  // Resolve/bind set 0 (lighting constant) and set 1 (per-draw UBO + texture)
  // for one draw.  Set 0 re-binds only when the lighting offset changes; set 1
  // always advances with the UBO offset.
  void bindDrawDescriptors(const SoRenderCommand & command,
                           uint32_t uboDynamicOffset,
                           uint32_t slotIndex,
                           VulkanRecordContext & ctx);
  void recordDrawCommand(const SoDrawList & drawlist,
                         const SoRenderCommand & command,
                         const SoVulkanRenderTarget & target,
                         const SoRenderParams & params,
                         VkRenderPass renderPass,
                         bool transparent,
                         int fillModeOverride,
                         const float * uniformColorOverride,
                         bool overlayPass,
                         VulkanRecordContext & ctx);
  // Instanced batch: draw `count` commands sharing geometry/material/state that
  // differ only by model matrix as ONE vkCmdDraw(instanceCount=count).  Returns
  // false if the first can't be drawn or any command is not batchable.
  bool recordCommandBatch(const SoDrawList & drawlist,
                           const SoRenderCommand * const * commands, int count,
                           const SoVulkanRenderTarget & target,
                           const SoRenderParams & params,
                           VkRenderPass renderPass,
                           bool transparent,
                           int fillModeOverride,
                           const float * uniformColorOverride,
                           VulkanRecordContext & ctx);
  bool expandWideLines(VulkanCachedCommand & entry,
                       const SoRenderCommand & command,
                       const SoRenderParams & params,
                       const SbMat & proj,
                       float lineWidth);
  // Ensure per-command wide-line quad buffers are worst-case sized on the calling
  // thread before workers fill them (VMA/deferred-ring are not thread-safe).
  void prepareWideLineBuffers(const SoDrawList & drawlist);
  // Build (once per content hash) the object-space instance endpoint stream:
  // four vec4 per segment (p0, p1, c0, c1); false => caller uses CPU expansion.
  bool buildInstancedLineBuffer(VulkanCachedCommand & entry,
                                const SoRenderCommand & command);

  // --- GPU sub-pixel geometry LOD (raster) ------------------------------
  // True for an indexed triangle list in a non-overlay pass (other paths unchanged).
  static bool isSubPixelEligible(const SoRenderCommand & command);
  // Sub-pixel slot for the current frame, or nullptr (caller draws full geometry).
  const VulkanCachedCommand::VulkanSubPixelSlot * subPixelSlotFor(
    const VulkanCachedCommand & entry) const;
  // Lazily create/size one slot's buffer + descriptor set; content-hash rebuild.
  bool ensureSubPixelSlot(VulkanCachedCommand & entry,
                          const SoRenderCommand & command, uint32_t slot);
  // Release every slot's buffers/descriptor set (cache eviction / teardown).
  void destroySubPixelResources(VulkanCachedCommand & entry);
  // Deferred variant: buffers may be referenced by an in-flight indirect draw.
  void deferDestroySubPixelResources(VulkanCachedCommand & entry);
  // Allocate a storage-buffer descriptor set, appending a pool when exhausted.
  bool allocateSubPixelDescriptorSet(VkDescriptorSet & set);
  // Record the frame's compaction dispatches into \a cb (outside a render pass;
  // trailing barrier orders writes against later draws).  0 => skip the buffer.
  uint32_t recordGeometryLodPrepass(VkCommandBuffer cb,
                                    const SoDrawList & drawlist,
                                    const SoRenderParams & params);
  // True when the geometry-LOD pre-pass would record this frame.
  bool externalGeometryLodActive(const SoRenderParams & params) const;

  // Per-external-frame timing for the [RTDBG] cpuTimingRaster line.
  struct ExternalFrameTiming {
    double setupMs = 0.0;
    double geomMs = 0.0;
    // Texture copies + host-side finalize recorded into the pre-pass.
    double texMs = 0.0;
    // Geometry-LOD compaction dispatches recorded into the pre-pass.
    double lodRecordMs = 0.0;
    // Pre-pass submit + host wait (the transient submit's queue drain).
    double lodMs = 0.0;
  };

  // Run the external pre-pass for a caller-owned, already-begun render pass.
  // Vulkan forbids transfer/compute inside a pass, so pending texture copies
  // and (when \a lod) sub-pixel compaction go into a backend-owned transient
  // buffer.  Recorded before recordFrame() so draws see finalized resources;
  // submitted by submitExternalPrepass() after, overlapping CPU recording with
  // the previous GPU frame.  Returns VK_NULL_HANDLE when nothing to record or
  // the buffer could not be prepared; pending uploads then must fall back to
  // flushPendingTextureUploadsExternal().
  VkCommandBuffer beginExternalPrepass(const SoDrawList & drawlist,
                                       const SoRenderParams & params,
                                       bool lod,
                                       ExternalFrameTiming * timing);
  // Submit the transient buffer and wait so its writes are visible; frees it.
  void submitExternalPrepass(VkCommandBuffer commandBuffer,
                             ExternalFrameTiming * timing);

  // Resolve the projection for a command's wide-line quads (own for a
  // self-camera overlay, else the frame's), matching recordDrawCommand()'s key.
  void resolveCommandProj(const SoRenderCommand & command,
                          const SoRenderParams & params,
                          bool overlayPass,
                          SbMat & out) const;
  // Expand one command's wide lines with the record path's projection/width.
  bool expandWideLinesFor(VulkanCachedCommand & entry,
                          const SoRenderCommand & command,
                          const SoRenderParams & params,
                          bool overlayPass);
  // Expand every wide-line command across the pool before the serial record pass,
  // so recordDrawCommand() only binds cached quads (embarrassingly parallel).
  void expandWideLinesParallel(const SoDrawList & drawlist,
                               const SoRenderParams & params);
  // Expand ONE large non-stippled LINE_LIST command by partitioning its segments
  // across the pool (the whole-command round-robin leaves one command serial).
  bool expandWideLinesSplit(VulkanCachedCommand & entry,
                            const SoRenderCommand & command,
                            const SoRenderParams & params,
                            const SbMat & proj,
                            float lineWidth);
  // Dispatch one split phase over [0, count): owner runs range 0 inline, workers
  // run the rest, then owner joins.  `phase` goes to every non-owner job.
  void dispatchWideLineSplit(int phase, uint32_t count,
                             const SoRenderParams & params);
  // Execute one split phase for segments [begin, end); concurrent-safe (disjoint
  // ranges; per-segment ph-1 or prefix-summed ph-2 slots).
  void expandWideLinesSplitRange(int phase, uint32_t begin, uint32_t end);
  bool endAndSubmit();
  void applyViewport(const SoRenderParams & params,
                     const SoVulkanRenderTarget & target,
                     VulkanRecordContext & ctx);
  void applyCommandViewport(const SoRenderCommand & command,
                            const SoVulkanRenderTarget & target,
                            VulkanRecordContext & ctx);
  void applyScissor(const SoRenderCommand & command,
                    const SoVulkanRenderTarget & target,
                    VulkanRecordContext & ctx);
  // Deduplicated dynamic-state emitters: record the value last set in `ctx` and
  // skip the vkCmd* when unchanged (only the submitted value is remembered).
  void applyPipeline(VkPipeline pipeline, VulkanRecordContext & ctx);
  void applyViewportState(const VkViewport & viewport,
                          VulkanRecordContext & ctx);
  void applyScissorState(const VkRect2D & scissor, VulkanRecordContext & ctx);
  void resetBoundState(VulkanRecordContext & ctx);
  void recordOverlayDepthClear(const SoRenderCommand & command,
                               const SoVulkanRenderTarget & target,
                               VulkanRecordContext & ctx);
  void recordOverlayBlock(const SoDrawList & drawlist,
                          const SoRenderParams & params,
                          const SoVulkanRenderTarget & target,
                          VkRenderPass renderPass,
                          VulkanRecordContext & ctx);
  void recordTracedComposite(const SoDrawList & drawlist,
                             const SoRenderParams & params,
                             const SoVulkanRenderTarget & target,
                             VkRenderPass renderPass,
                             VulkanRecordContext & ctx);
  SbBool renderInternal(const SoDrawList & drawlist,
                        const SoRenderParams & params,
                        bool overlaysOnly);
  bool recordFrame(const SoDrawList & drawlist,
                   const SoRenderParams & params,
                   const SoVulkanRenderTarget & target,
                   VkRenderPass renderPass,
                   VulkanRecordContext & ctx,
                   VkFramebuffer inheritFramebuffer);

  // Shared prologue of renderExternal()/renderExternalOverlay(): validate the
  // target, advance the frame and stage changed textures into pendingUploads.
  // `reserveCompositeSlots` reserves countCompositeCommands() lighting slots;
  // non-null `timing` gets setup/geom durations for [RTDBG].
  const SoVulkanRenderTarget * prepareExternalFrame(
      const SoDrawList & drawlist, const SoRenderParams & params,
      VkCommandBuffer commandBuffer, VkRenderPass renderPass,
      const char * caller, bool overlaysOnly, bool reserveCompositeSlots,
      ExternalFrameTiming * timing);
  // Validate params.renderTarget (non-null, image views, non-zero extent).
  const SoVulkanRenderTarget * validateRenderTarget(
      const SoRenderParams & params) const;

  // --- Vulkan resource helpers -------------------------------------------
  // Create a buffer + VMA allocation; fill via a one-time mapping when `data` set.
  bool createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkBuffer & buffer,
                    VmaAllocation & memory,
                    const void * data);
  // Device-local variant for retained static geometry: transient staging buffer
  // + one-shot transfer, waits the copy.  Only for the rare geometry-change path.
  bool createBufferDeviceLocal(VkDeviceSize size,
                               VkBufferUsageFlags usage,
                               VkBuffer & buffer,
                               VmaAllocation & memory,
                               const void * data);
  // Create a buffer backed by memory with the desired properties (fill `data`).
  bool createBufferWithProperties(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags desiredProperties,
                                  VkBuffer & buffer, VmaAllocation & memory,
                                  const void * data = nullptr);
  // Create a HOST_VISIBLE|HOST_COHERENT buffer with its persistent mapping; on
  // failure all outputs are null.  Shared by every per-frame UBO/ring buffer.
  bool createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                          VkBuffer & buffer, VmaAllocation & memory,
                          void ** mapped);
  // Defer buffer + allocation destruction; null handles are ignored.
  void deferDestroyBufferMemory(VkBuffer buffer, VmaAllocation memory);
  // Ensure the model-matrix buffer holds >= `bytes`; grows via the deferred ring.
  bool ensureInstanceModelBuffer(VkDeviceSize bytes);
  // Size the model-matrix buffer to the full ring, parallel to the lighting-UBO
  // ring, so the per-draw path never grows it (a vkMapMemory/vkDestroyBuffer
  // per draw would race once workers record in parallel).
  bool ensureInstanceModelRingCapacity();
  bool growLightingUbo(uint32_t minSlots);
  bool swapLightingBuffer(VkBuffer newBuffer, VmaAllocation newMemory,
                          void * newMapped, uint32_t newSlotsPerFrame);
  bool prepareLightingSlots(uint32_t neededDraws);
  void beginFrame();
  void flushPendingDestroys();
  void flushAllPendingDestroys();
  void deferDestroy(std::function<void()> && fn);
  void deferDestroyCacheEntry(VulkanCachedCommand & entry);
  void deferDestroyTextureEntry(VulkanCachedTexture & entry);
  void waitForInFlightFrames();
  bool allocateFrameResources();
  void releaseFrameResources();

  // --- Owned device ------------------------------------------------------
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  const VkAllocationCallbacks * allocator = nullptr;
  // Vulkan Memory Allocator; created in initialize(), destroyed at shutdown().
  VmaAllocator vmaAllocator = nullptr;
  // Cached physical-device memory-properties picker (bound in initialize()).
  SoVulkanShared::MemoryProperties memProps;

  // --- Device capabilities (probed once in initialize()) -----------------
  // fillModeNonSolid gates the wireframe/points overlay and LINES/POINTS
  // pipelines (VK_POLYGON_MODE_LINE/POINT).  Enabled only when supported (see
  // QuarterVulkanWidget::configureDeviceFeatures); otherwise refused.
  bool fillModeNonSolid = false;
  // SAMPLED_IMAGE support for R8_UNORM / R8G8_UNORM (R8G8B8A8_UNORM is required).
  bool sampledR8 = false;
  bool sampledR8G8 = false;
  // VK_EXT_pipeline_creation_feedback; gates the COIN_VULKAN_PIPELINE_FEEDBACK log.
  bool hasPipelineCreationFeedback = false;

  // Per-pass GPU timestamps (COIN_VULKAN_GPU_TIMING), lazily initialized.
  SoVulkanGpuTimers gpuTimers;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  // One command buffer and fence per in-flight frame slot.  The own-queue path
  // submits slot N's buffer/fence (beginFrame() waits it before reuse); the
  // external path never signals them, so frameFencePending stays false.
  std::vector<VkCommandBuffer> frameCommandBuffers;
  std::vector<VkFence> frameFences;
  std::vector<uint8_t> frameFencePending;
  // Fence for the external pre-pass submit (texture copies + geometry-LOD
  // compaction).  The caller owns its render-pass submit, so no semaphore can be
  // threaded through; this fence waits only that submit, not the whole queue
  // (vkQueueWaitIdle would also wait on the caller's acquire/present).
  VkFence externalPrepassFence = VK_NULL_HANDLE;
  // Secondary command buffers (M1c/M1d): one per in-flight slot per worker,
  // recording the render-order-independent opaque pass inside an already-begun
  // pass (RENDER_PASS_CONTINUE), then vkCmdExecuteCommands()'d into the slot's
  // primary.  Kept per-slot so a secondary is never reset while its primary is
  // pending; RESET_COMMAND_BUFFER_BIT lets it re-record after the fence.
  //
  // ONE POOL PER WORKER: VkCommandPool host access is externally synchronized,
  // so concurrent vkReset/vkBegin/vkEnd from a SHARED pool (M1d workers + main)
  // would race the pool allocator.  Pool w owns only worker w's secondaries.
  std::vector<VkCommandPool> secondaryCommandPools;
  // Indexed [slot * maxRecordWorkers + worker]; never resets a pending secondary.
  std::vector<VkCommandBuffer> secondaryCommandBuffers;
  // M1d parallel recording: the worklist is partitioned into disjoint chunks, each
  // recorded by one worker into its own secondary (M1b pre-assigns disjoint slots,
  // so mapped UBO writes are race-free).  Joined in shutdown().
  bool parallelRecordEnabled = false;
  uint32_t maxRecordWorkers = 1;
  // Interaction LOD: wide lines draw as 1px GPU lines instead of CPU quads.
  bool interactionLodActive = false;
  std::vector<std::thread> recordWorkers;
  std::vector<VulkanRecordContext> workerRecordContexts;
  struct ParallelRecordJob {
    const SoDrawList * drawlist = nullptr;
    const SoRenderParams * params = nullptr;
    const SoVulkanRenderTarget * target = nullptr;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    // This worker's chunk: pointers into workItemsScratch (alive throughout).
    std::vector<const VulkanWorkItem *> items;
    VkCommandBuffer secondary = VK_NULL_HANDLE;
    VulkanRecordContext * ctx = nullptr;
    bool ok = false;
    // Wide-line expansion dispatch: expand `wideLineCommands` instead of a chunk.
    bool expandWideLines = false;
    std::vector<const SoRenderCommand *> wideLineCommands;
    // Intra-command split: expand segments [wlineSplitBegin, wlineSplitEnd) of
    // wlineSplitCtx's command.  1 = clip+validity, 2 = emit quads.  Owner-thread.
    int wlineSplitPhase = 0;
    uint32_t wlineSplitBegin = 0;
    uint32_t wlineSplitEnd = 0;
  };
  std::vector<ParallelRecordJob> recordJobs;
  std::mutex recordMutex;
  std::condition_variable recordCvSpawn, recordCvDone;
  // Monotonic dispatch generation: workers re-wait until a NEW generation arrives,
  // consuming each dispatch once (a plain ready flag would double-process).
  uint32_t recordJobGeneration = 0;
  std::atomic<uint32_t> recordDoneCount {0};
  bool recordPoolStopped = false;
  // Per-recording command-buffer target + dedup state for the current call
  // (backend's own buffer in render(), caller's in renderExternal()).
  VulkanRecordContext recordContext;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  // Set-0 layout: lighting constant ring; lightingDescriptorSet comes from it.
  VkDescriptorSetLayout lightingSetLayout = VK_NULL_HANDLE;
  // Descriptor pools, append-only while frames are in flight (live sets stay valid).
  std::vector<VkDescriptorPool> descriptorPools;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  // Sets allocated from the active descriptorPool (white set plus textures).
  uint32_t descriptorSetCount = 0;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  // Per-draw view/model/material UBO (set 1, binding 0, dynamic offset): one ring
  // buffer of maxFramesInFlight frames of per-command slots.  Distinct slots + a
  // ring half per frame keep uniform data stable while the GPU runs async.
  VkBuffer lightingBuffer = VK_NULL_HANDLE;
  VmaAllocation lightingMemory = nullptr;
  void * lightingMapped = nullptr;
  VkDeviceSize uboSlotStride = 0;
  uint32_t uboSlotsPerFrame = 0;
  uint32_t uboFrameIndex = 0;

  // Host-visible, persistently-mapped per-instance model matrices (binding 1,
  // rate INSTANCE): commands sharing geometry but differing by model matrix draw
  // as one vkCmdDraw(instanceCount=N).  Grows on demand, freed in shutdown().
  VkBuffer instanceModelBuffer = VK_NULL_HANDLE;
  VmaAllocation instanceModelMemory = nullptr;
  void * instanceModelMapped = nullptr;
  VkDeviceSize instanceModelCapacity = 0;
  // Per-frame pre-converted camera matrices (double -> float): cacheFrameMatrices()
  // converts once per render instead of per draw in updateLightingUniforms().
  float frameViewFloats[16] = {};
  float frameProjFloats[16] = {};
  // Hoisted device-pixel ratio, read once by cacheFrameMatrices() instead of per draw.
  float frameDpr = 1.0f;
  void cacheFrameMatrices(const SoRenderParams & params);

  // Lighting constant ring (set 0, binding 0, dynamic offset): one slot per
  // distinct SoLightingHandle, written once per frame and shared via dynamic offset.
  VkBuffer lightingConstBuffer = VK_NULL_HANDLE;
  VmaAllocation lightingConstMemory = nullptr;
  void * lightingConstMapped = nullptr;
  VkDeviceSize lightingConstStride = 0;
  uint32_t lightingConstMaxSlots = 0;
  // The single set-0 descriptor (lighting UBO), bound with set 1 on every draw.
  VkDescriptorSet lightingDescriptorSet = VK_NULL_HANDLE;
  // Handle -> byte offset into the current frame's lighting ring.
  std::unordered_map<SoLightingHandle, VkDeviceSize> lightingSlotOffsets;

  // Authoritative viewer lighting (setSceneLights()).  When non-empty,
  // updateLightingSetup() uses this camera-anchored set instead of per-command IR
  // SoLightingData, matching the RT path.
  SoLightingData sceneLighting;

  // Resources replaced during recording are destroyed maxFramesInFlight frames
  // later, once referencing submissions have completed (batch B flushed then).
  uint32_t maxFramesInFlight = 3;
  SoVulkanShared::PendingDestroys pendingDestroys;

  // Vulkan-only display overlays (shaded-with-edges / show-vertices).
  SbBool wireframeOverlay = FALSE;
  SbBool pointsOverlay = FALSE;
  // Debug overlay: re-draw triangles in polygon-LINES (see buildWorkItems()).
  SbBool tessellationOverlay = FALSE;
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);

  // Texture uploads gathered by updateGeometryCache().  Own-queue path records
  // into the frame command buffer; external path into the pre-pass buffer.  Indices
  // are re-resolved from command pointers after eviction compacts the cache.
  std::vector<PendingTextureUpload> pendingUploads;

  // Persistent host-visible staging buffer coalescing a frame's texture uploads
  // into one write/one submit instead of per-upload allocations.  Grows on demand,
  // reused across frames; its mapping owns the staged pixels.
  VkBuffer stagingPoolBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingPoolAllocation = nullptr;
  void * stagingPoolMapped = nullptr;
  VkDeviceSize stagingPoolCapacity = 0;
  // Byte cursor into stagingPoolBuffer for the current frame; reset each flush.
  VkDeviceSize stagingPoolCursor = 0;
  bool ensureStagingPoolSize(VkDeviceSize required);

  // Texture binding (set 0, binding 1); 1x1 white fallback when untextured.
  VkImage whiteImage = VK_NULL_HANDLE;
  VmaAllocation whiteImageAllocation = nullptr;
  VkImageView whiteImageView = VK_NULL_HANDLE;
  VkSampler whiteSampler = VK_NULL_HANDLE;
  VkDescriptorSet whiteDescriptorSet = VK_NULL_HANDLE;

  VkShaderModule vertexModule = VK_NULL_HANDLE;
  VkShaderModule fragmentModule = VK_NULL_HANDLE;

  // Wide-line (line width > 1 and/or stippled line pattern) pipeline shaders.
  VkShaderModule wideLineVertexModule = VK_NULL_HANDLE;
  VkShaderModule wideLineFragmentModule = VK_NULL_HANDLE;
  // GPU-instanced wide-line vertex shader (no CPU quads); shares the fragment.
  VkShaderModule wideLineInstancedVertexModule = VK_NULL_HANDLE;

  // GPU sub-pixel geometry LOD resources (raster only): a compute pipeline that
  // compacts triangle indices while the camera moves.  Per-slot buffers live in
  // the geometry cache entries.
  VkShaderModule subPixelCullModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout subPixelSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout subPixelPipelineLayout = VK_NULL_HANDLE;
  VkPipeline subPixelCullPipeline = VK_NULL_HANDLE;
  // Append-only storage-buffer descriptor pools (sets never freed while in flight).
  std::vector<VkDescriptorPool> subPixelDescriptorPools;
  uint32_t subPixelDescriptorSetCount = 0;
  // Largest storage-buffer range for one binding (maxStorageBufferRange), cached
  // at pipeline build.  Larger vertex buffers force the pre-pass to skip the
  // command and fall back to the full draw; 0 = unknown, do not guard.
  uint64_t subPixelMaxStorageRange = 0;

  // Background gradient resources (no descriptor sets; push constants only).
  VkShaderModule backgroundVertexModule = VK_NULL_HANDLE;
  VkShaderModule backgroundFragmentModule = VK_NULL_HANDLE;
  VkPipelineLayout backgroundPipelineLayout = VK_NULL_HANDLE;

  // Render passes cached by VkRenderPassCreateInfo identity (formats, sample
  // count, layouts, load ops); pipelines keyed on the pass handle, so reusing a
  // pass across targets differing only in images keeps the pipeline cache warm
  // (notably cycling swapchain targets).  The per-target framebuffer is recreated
  // on target change while the pass survives.  All in SoVulkanRenderPassCache.
  SoVulkanRenderPassCache renderPasses;

  // Pipeline store keyed by retained pipeline-affecting state (pipelines are
  // immutable).  Key types live in SoVulkanPipelineCache.
  SoVulkanPipelineCache pipelines;

  // Wide-line expansion scratch is thread_local inside expandWideLines() because
  // expansion runs on the parallel workers (SoVulkanRenderBackendWideLine.cpp).
  // The vector below collects commands for the dispatch (main thread only).
  std::vector<const SoRenderCommand *> wlineExpandScratch;
  // Thread driving the expansion pre-pass, keeping wide-line diagnostics on one thread.
  std::thread::id wlineOwnerThread;

  // Intra-command wide-line split: expandWideLines()' thread_local scratch is
  // per-worker, but the split path partitions ONE command, so its phases share
  // these owner-sized buffers (phase 1 clip/valid, phase 2 quads).  Sized once
  // per command; owner-thread only.
  struct WideLineSplitCtx {
    const SoRenderCommand * command = nullptr;
    const SoGeometryDesc * geometry = nullptr;
    SbMat mvp;
    float vpWidth = 1.0f;
    float vpHeight = 1.0f;
    float lineWidth = 1.0f;
    float nearEps = 1.0e-5f;
    // Phase 2 writes quads straight into the slot's mapped buffer (avoids a
    // full-size memcpy -- hundreds of MB for a million segments).
    float * outBase = nullptr;
    uint32_t posStrideFloats = 3;
    uint32_t segmentCount = 0;
  };
  WideLineSplitCtx wlineSplitCtx;
  std::vector<float> wlineSplitClip;      // vertexCount * 4
  std::vector<uint8_t> wlineSplitValid;   // segmentCount
  std::vector<size_t> wlineSplitOffsets;  // segmentCount (float index into quads)
  // Commands routed to the split path for the current frame (owner thread).
  std::vector<const SoRenderCommand *> wlineSplitScratch;

  std::vector<VulkanCachedCommand> gpuCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  std::vector<VulkanCachedTexture> textureCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToTexture;

  // True while this backend only composites overlays/residual geometry over a
  // frame drawn by another renderer (setOverlayCompositeMode()); lets
  // updateGeometryCache() sweep stale entries so that renderer's triangles
  // aren't kept resident here too.
  bool overlayCompositeMode = false;
  // Monotonic visit stamp for the overlay-composite sweep (compositeEpoch).
  uint32_t overlayCompositeEpoch = 0;

  // Reusable batch-key bucket map for recordFrame()'s opaque batching (clear()+reuse).
  std::unordered_map<uint64_t, std::vector<const SoRenderCommand *>> batchBucketScratch;
  // Separate bucket map for the wireframe/point overlay pass.  Must be distinct from
  // batchBucketScratch: opaque work items hold pointers into the latter's vectors, so
  // clearing it for the overlay would dangle them.
  std::unordered_map<uint64_t, std::vector<const SoRenderCommand *>> overlayBatchBucketScratch;
  // Reusable worklist from buildWorkItems() (M1b/M1c/M1d), refilled per frame.
  std::vector<VulkanWorkItem> workItemsScratch;

  // Reusable scratch for recordFrame()'s secondary/parallel record paths.
  std::vector<const VulkanWorkItem *> opaqueItemsScratch;
  std::vector<std::pair<uint64_t, const VulkanWorkItem *>> heaviestScratch;
  std::vector<uint64_t> loadScratch;
  std::vector<VkCommandBuffer> executeScratch;

  // Packed sampler-state key: minFilter | magFilter << 2 | wrapS << 4 | wrapT << 6.
  typedef uint8_t SamplerKey;
  static SamplerKey samplerKey(SoTextureFilter minFilter,
                               SoTextureFilter magFilter,
                               SoTextureWrap wrapS, SoTextureWrap wrapT);
  // Sampler cache: textures sharing filter/wrap state reuse one VkSampler.
  std::unordered_map<SamplerKey, VkSampler> samplerCache;
  VkSampler cachedSampler(SoTextureFilter minFilter, SoTextureFilter magFilter,
                          SoTextureWrap wrapS, SoTextureWrap wrapT);

  std::vector<VulkanGeometryBlock> geometryBlocks;
  // Released blocks kept as reusable ids to bound geometryBlocks growth.
  std::vector<uint32_t> freeGeometryBlockIds;
  VkDeviceSize nextGeometryBlockCapacity = 256u * 1024u;

  // Reusable packing buffer for uploadGeometry() (32-byte layout, VULKAN_VERTEX_STRIDE).
  std::vector<uint8_t> uploadScratch;
  // Reusable per-draw "needs geometry upload" flags for updateGeometryCache().
  std::vector<uint8_t> needsGeometryScratch;
  // Guards uploadScratch against concurrent packing from multiple render threads.
  std::mutex uploadScratchMutex;
};

#endif // COIN_SOVULKANRENDERBACKEND_H
