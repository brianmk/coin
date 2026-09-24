// src/rendering/SoVulkanRenderBackend.h

#ifndef COIN_SOVULKANRENDERBACKEND_H
#define COIN_SOVULKANRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include "rendering/SoVulkanShared.h"
#include "rendering/SoVulkanResult.h"
// Vulkan Memory Allocator handles.  Only the opaque handle types are needed in
// this header; the full API lives in third_party/vma/vk_mem_alloc.h, included
// by the .cpp files that allocate.  VK_DEFINE_HANDLE produces the same typedef
// VMA does, so either include order is safe.
#ifndef AMD_VULKAN_MEMORY_ALLOCATOR_H
VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)
#endif
#include "rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRecordContext.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderPassCache.h"
#include "rendering/SoVulkanGpuTimers.h"
#include "rendering/SoVulkanSamplerCache.h"
#include "rendering/SoVulkanStagingPool.h"
#include "rendering/SoVulkanBufferFactory.h"
#include "rendering/SoVulkanFrameRing.h"
#include "rendering/SoVulkanTextureCache.h"
#include "rendering/SoVulkanGeometryArena.h"
#include "rendering/SoVulkanGeometryCache.h"

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

// PipelineKey / PipelineKeyHash / BackgroundPipelineKey* and the
// SoVulkanPipelineCache store live in SoVulkanPipelineCache.h (included
// above).  VulkanCachedCommand and the geometry cache store live in
// SoVulkanGeometryCache.h (included above).

// VulkanCachedTexture is defined in SoVulkanTextureCache.h (included above).

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

    The caller must already have begun \a commandBuffer and started
    \a renderPass with a compatible framebuffer; it also owns submission and
    presentation.  This path does not begin/end the command buffer, begin/end
    the render pass, create a framebuffer, or submit to the queue.  It is used
    by embedding surfaces such as QVulkanWindow whose command buffer and render
    pass lifecycle are managed by the window system.

    The per-frame setup, the pending texture uploads and the GPU geometry-LOD
    pre-pass are handled internally, so this is the single external entry
    point: the caller never coordinates a separate prepare step.  Because
    Vulkan forbids transfer and compute commands inside a render pass, the
    pre-pass is recorded into a backend-owned transient command buffer before
    the frame draws and submitted after them (see beginExternalPrepass()/
    submitExternalPrepass()), so the caller's pass sees the uploaded textures
    and the compacted geometry, and the CPU recording overlaps the previous
    GPU frame.
  */
  SbBool renderExternal(const SoDrawList & drawlist,
                        const SoRenderParams & params,
                        VkCommandBuffer commandBuffer,
                        VkRenderPass renderPass,
                        VkFramebuffer framebuffer);

  /*!
    \brief Record only the overlay pass (e.g. the navigation cube) into a
    caller-owned command buffer/render pass.

    Used in ray-tracing mode where the scene is traced by the RT backend but
    screen-space overlays are still rasterized on top.
  */
  SbBool renderExternalOverlay(const SoDrawList & drawlist,
                               const SoRenderParams & params,
                               VkCommandBuffer commandBuffer,
                               VkRenderPass renderPass);

  /*!
    \brief HDR output variant of renderExternal() for the raster path.

    Renders the scene into a backend-owned linear RGBA16F intermediate and then
    presents it into the caller's \a outputPass / \a outputFramebuffer with the
    exposure + transfer-function transform (SMPTE ST 2084 / PQ).  Doing the
    encode once, after all geometry and transparency have blended in linear
    light, is what makes the output color-correct; see
    data/shaders/vulkan/output/OutputFragment.glsl.

    Unlike renderExternal(), the caller must NOT have begun a render pass: this
    method owns the whole pass lifecycle (offscreen pass, barrier, output pass).
    The caller still owns the command buffer's begin/end and the submission.

    The intermediate is recreated when the target extent changes; the output
    pipeline is cached per \a outputPass (the swapchain render pass changes with
    the color format).
  */
  SbBool renderExternalHdr(const SoDrawList & drawlist,
                           const SoRenderParams & params,
                           VkCommandBuffer commandBuffer,
                           VkRenderPass outputPass,
                           VkFramebuffer outputFramebuffer);

  //! Enable/disable the HDR output transform (see renderExternalHdr()).
  //! \a exposure is the linear scale mapping scene-white to the PQ peak
  //! (0.02 ~= 200 cd/m^2 reference white) and \a toneMap selects the
  //! tone-mapping operator (0 = clip, 1 = Reinhard, 2 = ACES, 3 = Hable).
  void setHdrOutput(SbBool enabled, float exposure, int toneMap);

  /*!
    \brief Composite only the overlay pass (e.g. the navigation cube) into
    the render target.

    Offscreen counterpart of renderExternalOverlay(): runs a complete
    one-shot render into params.renderTarget without touching anything but
    SO_RENDERPASS_OVERLAY commands, so it can be layered on top of a
    previously rendered (e.g. ray-traced) frame.  Returns TRUE with no work
    if the draw list contains no overlay commands.
  */
  SbBool renderOverlaysOnly(const SoDrawList & drawlist,
                            const SoRenderParams & params);

  /*!
    \brief Declare that this backend only composites overlays and residual
    geometry on top of a ray-traced frame.

    While ray tracing is active the manager drives this backend through
    renderExternalOverlay()/renderOverlaysOnly() only; the RT backend owns the
    scene's triangle geometry.  In that state updateGeometryCache() runs its
    stale-entry sweep on overlays-only frames too, evicting the traced triangle
    commands this backend no longer visits, so the scene meshes are not held
    resident a second time alongside the RT backend's copy.  The manager sets
    this while ray tracing is active and clears it when it is not; it must stay
    false for a backend that also performs full raster renders, whose cache has
    to survive an interleaved overlay pass.
  */
  void setOverlayCompositeMode(SbBool enabled);

  /*!
    \brief Declare how many recorded frames the caller may keep in flight.

    Drives the deferred-destruction batch count and the lighting UBO ring
    size: resources replaced while recording frame N are only released when
    frame N + \a count begins, by which time every submission that could
    still reference them has completed.  Callers that submit frames
    concurrently must set this to their maximum in-flight frame count before
    the first render call.  Defaults to 3 (QVulkanWindow's default
    concurrency).
  */
  void setMaxFramesInFlight(uint32_t count);

  /*!
    \brief Path of a persistent (on-disk) Vulkan pipeline cache.

    When non-empty, initialize() loads the file's bytes as the initial
    pipeline-cache data and shutdown() writes the driver's cache blob back, so
    the many lazily-created pipeline variants survive a process restart.  The
    file is advisory: a missing, corrupt or stale (different device/driver)
    file is rejected by the implementation and an empty cache is created
    instead.  The embedding application owns the path and its directory, since
    Coin has no window-system or user-cache knowledge.  Set it before
    initialize().
  */
  void setPipelineCachePath(const std::string & path);

  /*!
    \brief Configure Vulkan-only display overlays.

    These toggle the wireframe/point edge overlays and their color.  They are
    deliberately backend state rather than SoRenderParams fields, so the
    OpenGL backend never sees them.
  */
  void setWireframeOverlay(SbBool enabled);
  void setPointsOverlay(SbBool enabled);
  void setTessellationOverlay(SbBool enabled);
  void setEdgeColor(const SbColor4f & color);

  /*!
    \brief Provide the authoritative viewer lighting (GL host -> raster backend).

    \a lighting is the camera-anchored world-space viewer light set (headlight,
    backlight, fill light) plus the intensity-scaled scene ambient.  When its
    light list is non-empty the executor uses this single set for every command
    instead of the per-command IR capture, so the raster Vulkan path lights the
    scene exactly like the RT backend (and follows the camera like Coin GL).
    Passing an empty light list restores the per-command IR lighting.
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
  // Wrap a SPIR-V blob in a VkShaderModule.  The three shader-pair creators
  // used to define the same create-module lambda each.
  bool createShaderModule(const uint32_t * code, size_t count,
                          VkShaderModule & module);
  bool createBackgroundPipeline(const SoVulkanRenderTarget & target,
                                VkRenderPass renderPass,
                                VkPipeline & pipeline);
  // Assemble and create a graphics pipeline from the caller-supplied varying
  // state (stages/vertex input/input assembly/raster/depth-stencil/blend
  // attachment).  The fixed state shared by every pipeline (dynamic viewport+
  // scissor, multisample, color-blend wrapper, create info) lives here; the
  // visual and background pipelines both route through it.
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
  // Write each distinct SoLightingHandle's constant block once into the
  // lighting ring for the current frame, and build lightingSlotOffsets.
  // Must run before recording draws; called at the start of renderInternal /
  // renderExternal after beginFrame().
  bool updateLightingSetup(const SoDrawList & drawlist);
  void updateLightingUniforms(const SoDrawList & drawlist,
                              const SoRenderCommand & command,
                              const SoRenderParams & params,
                              VkDeviceSize uboOffset,
                              bool unlit = false,
                              const float * projFloats = nullptr);
  // Dynamic byte offset into the lighting ring for a command's handle (0 if
  // the command references no lighting).  Uses the frame-local
  // lightingSlotOffsets built by updateLightingSetup().
  VkDeviceSize lightingOffsetFor(const SoRenderCommand & command) const;

  // --- Geometry cache ---------------------------------------------------
  void invalidateCache();
  // \a geometryContentUnchanged (SoRenderParams::geometryContentUnchanged):
  // on a retained-IR replay frame the main geometry content is bit-identical
  // to the previous frame, so for commands whose pointer identity already
  // matches the cache the sampled content re-hash is skipped.
  void updateGeometryCache(const SoDrawList & drawlist, bool overlaysOnly = false,
                           bool geometryContentUnchanged = false);

  // --- Geometry arena ---------------------------------------------------
  // The shared vertex/index blocks the geometry cache carves per-command
  // ranges out of live in the SoVulkanGeometryArena collaborator
  // (this->geometryArena); see SoVulkanGeometryArena.h.

  // --- Texture cache ----------------------------------------------------
  // Texture state and the staging->image upload path live in the
  // SoVulkanTextureCache collaborator (this->textureCache); see
  // SoVulkanTextureCache.h.  The shared set-1 descriptor pool and the
  // deferred-destruction ring stay here and are handed to it as callbacks.

  // One resolvable draw unit of the frame's worklist, produced by
  // buildWorkItems().  buildWorkItems() walks the sorted order, buckets
  // opaque depth-tested commands, and pre-assigns each item a disjoint
  // `slotBase` (the first of `count` consecutive lighting-UBO / instance-model
  // ring slots it consumes).  Worker threads later record from these read-only
  // items, so nothing here mutates shared backend state.  `commands` is either
  // a single pointer (count == 1, recorded as one draw) or an array of `count`
  // pointers sharing geometry/material (recorded as one instanced draw).
  struct VulkanWorkItem {
    const SoRenderCommand * single = nullptr;     // count == 1 draw
    const SoRenderCommand * const * commands = nullptr; // count > 1 batch array
    int count = 1;
    uint32_t slotBase = 0;
    bool transparent = false;
    bool recordToSecondary = false; // opaque pass → secondary cmd buffer (M1c)
    int fillModeOverride = -1;     // wireframe/point redraw fill mode, or -1
    const float * uniformColorOverride = nullptr;
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
  // Allocate a set-1 descriptor (draw UBO + texture) from the shared pool.
  bool allocateTextureDescriptorSet(VkImageView view, VkSampler sampler,
                                    VkDescriptorSet & set);
  bool ensureDescriptorPoolSpace();

  // --- Render recording ---------------------------------------------------
  // Every record* helper below takes the VulkanRecordContext it records
  // into and touches no other recording state, so (future) worker threads
  // can each record their own command buffer with their own dedup cache.
  bool beginCommandBuffer();
  VkCommandBuffer currentCommandBuffer();
  VkCommandBuffer currentSecondaryCommandBuffer();
  void recordClear(const SoRenderParams & params,
                   const SoVulkanRenderTarget & target,
                   bool colorClearedByLoad,
                   bool depthClearedByLoad,
                   VulkanRecordContext & ctx);
  // Returns true when the frame's clear region (derived from params.viewport)
  // covers the whole target, so the render pass can use a CLEAR color/depth
  // loadOp instead of a vkCmdClearAttachments region clear.
  bool isFullTargetClear(const SoRenderParams & params,
                         const SoVulkanRenderTarget & target) const;
  // Byte offset of a per-draw lighting-UBO / instance-model ring slot within
  // the current frame's ring half.  Both the single-draw and batch recorders
  // derived this from their slot index identically.
  VkDeviceSize uboSlotOffset(uint32_t slotIndex) const;
  // Resolve and bind descriptor set 0 (lighting constant, dynamic offset) and
  // set 1 (per-draw UBO + texture, dynamic offset) for one draw.  Set 0
  // re-binds only when the lighting handle's offset actually changes; set 1
  // always advances with the per-draw UBO offset.  Shared by the single-draw
  // and batch recorders.
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
  // Instanced batch: draw `count` commands that share geometry/material/state
  // and differ only by model matrix as ONE vkCmdDraw(instanceCount=count).
  // `commands` is an array of pointers into the draw list.  Returns false if
  // the first command cannot be drawn (missing cache entry) or any command is
  // not batchable.
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
  // Ensure the per-command wide-line quad buffers exist and are large enough
  // for the worst case, on the calling (recording) thread, before any parallel
  // record worker fills them.  Device-memory allocation and the deferred
  // destroy ring are not thread-safe, so expandWideLines() must never grow a
  // buffer inside a worker.
  void prepareWideLineBuffers(const SoDrawList & drawlist);
  // Build (once per content hash) the object-space instance endpoint stream
  // for a GPU-instanced wide line: four vec4 per segment (p0, p1, c0, c1).
  // Runs on the recording thread; returns false if the command is not
  // eligible or the buffer could not be created, so the caller falls back to
  // the CPU quad expansion.
  bool buildInstancedLineBuffer(VulkanCachedCommand & entry,
                                const SoRenderCommand & command);

  // --- GPU sub-pixel geometry LOD (raster) ------------------------------
  // True when a command can be compacted: an indexed triangle list in a
  // non-overlay pass.  Line/point/overlay geometry keeps its existing path.
  static bool isSubPixelEligible(const SoRenderCommand & command);
  // The sub-pixel slot for the current frame, or nullptr when the command was
  // not compacted this frame (the caller then draws the full geometry).
  const VulkanCachedCommand::VulkanSubPixelSlot * subPixelSlotFor(
    const VulkanCachedCommand & entry) const;
  // Lazily create (and size) the compacted index / indirect buffers and the
  // storage-buffer descriptor set for one slot of one command.  Rebuilt when
  // the geometry content hash changes.  Returns false to fall back to a full
  // draw.
  bool ensureSubPixelSlot(VulkanCachedCommand & entry,
                          const SoRenderCommand & command, uint32_t slot);
  // Allocate a storage-buffer descriptor set from the sub-pixel pool,
  // appending a fresh pool when the active one is exhausted.
  bool allocateSubPixelDescriptorSet(VkDescriptorSet & set);
  // Record the compaction dispatches for the frame into \a cb.  Must run
  // outside a render pass; a trailing memory barrier orders the writes
  // against the indirect/index reads of the subsequent draws.
  void recordGeometryLodPrepass(VkCommandBuffer cb,
                                const SoDrawList & drawlist,
                                const SoRenderParams & params);
  // True when the geometry-LOD pre-pass would record anything this frame:
  // interaction LOD (or the verification override) is engaged, the feature is
  // enabled and the compaction pipeline exists.  Split out so the external
  // pre-pass can decide whether it needs a transient command buffer at all.
  bool externalGeometryLodActive(const SoRenderParams & params) const;

  // Timing breakdown of one external frame, filled by prepareExternalFrame()
  // and the pre-pass helpers for the [RTDBG] cpuTimingRaster line.
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

  // Run the external pre-pass for a frame whose caller owns (and has already
  // begun) its render pass.  Vulkan forbids transfer and compute commands
  // inside a render pass, so both the pending texture copies (buffer -> image)
  // and -- when \a lod and externalGeometryLodActive() hold -- the sub-pixel
  // compaction dispatches are recorded into a backend-owned transient command
  // buffer.  Recording happens before recordFrame() so the draw path sees the
  // finalized textures and the compacted slots; the buffer is submitted by
  // submitExternalPrepass() after recordFrame(), which overlaps the CPU frame
  // recording with the previous GPU frame instead of stalling before it.
  //
  // Returns VK_NULL_HANDLE when there is nothing to record, or when the
  // transient buffer could not be prepared.  When it returns null while
  // texture uploads were pending, the caller must fall back to
  // SoVulkanTextureCache::flushExternal() so the textures still upload.
  VkCommandBuffer beginExternalPrepass(const SoDrawList & drawlist,
                                       const SoRenderParams & params,
                                       bool lod,
                                       ExternalFrameTiming * timing);
  // Submit the transient buffer from beginExternalPrepass() and wait for it to
  // complete, so the copies and the compacted writes are visible before the
  // caller submits its pass.  Frees the buffer.  No-op (returns true) on
  // VK_NULL_HANDLE.  Returns false when the submit or the host wait failed; the
  // finalized texture entries are then un-stamped so the next frame re-uploads
  // them instead of sampling never-uploaded images forever.
  bool submitExternalPrepass(VkCommandBuffer commandBuffer,
                             ExternalFrameTiming * timing);

  // Resolve the projection a command's wide-line quads must use (its own for a
  // self-camera overlay, else the frame projection), matching
  // recordDrawCommand()'s push-constant path exactly so the expansion cache
  // key is identical whether the expansion runs in the pre-pass or inline.
  void resolveCommandProj(const SoRenderCommand & command,
                          const SoRenderParams & params,
                          bool overlayPass,
                          SbMat & out) const;
  // Expand one command's wide lines with the same projection/width the record
  // path uses.  Safe to call from a worker thread (thread-local scratch, and
  // prepareWideLineBuffers() has already sized the target buffer).
  bool expandWideLinesFor(VulkanCachedCommand & entry,
                          const SoRenderCommand & command,
                          const SoRenderParams & params,
                          bool overlayPass);
  // Expand every wide-line command of the frame across the worker pool before
  // the (serial) record pass, so recordDrawCommand() only binds the cached
  // quads.  This is the CPU-heavy part of line rendering and is embarrassingly
  // parallel; the command-buffer recording itself stays single-threaded.
  void expandWideLinesParallel(const SoDrawList & drawlist,
                               const SoRenderParams & params);
  // Expand ONE large non-stippled LINE_LIST command by partitioning its
  // segments across the worker pool.  The whole-command round-robin above
  // leaves a single dominant command (a Voronoi lattice's edge set, say)
  // serial; this splits that command's segments instead.  Runs on the owner
  // thread and joins each phase before returning.
  bool expandWideLinesSplit(VulkanCachedCommand & entry,
                            const SoRenderCommand & command,
                            const SoRenderParams & params,
                            const SbMat & proj,
                            float lineWidth);
  // Dispatch one split phase over [0, count): the owner runs range 0 inline,
  // workers 1..W-1 run the rest, then the owner joins.  `phase` is passed to
  // every non-owner job.
  void dispatchWideLineSplit(int phase, uint32_t count,
                             const SoRenderParams & params);
  // Execute one split phase for segments [begin, end) using wlineSplitCtx.
  // Safe to call concurrently: ranges are disjoint and every write goes to a
  // per-segment slot (phase 1) or a prefix-summed slot (phase 2).
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
  // Deduplicated dynamic-state emitters.  Each records the value last set
  // into `ctx` and skips the vkCmd* call when the incoming value is already
  // active, so an opaque run of draws sharing a pipeline/viewport pays one
  // state change instead of one per draw.  Only the value actually submitted
  // to the command buffer is remembered, so early-return paths (e.g. a
  // command with no per-command viewport) leave the remembered state as the
  // real current binding.
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

  // Per-frame plan shared by every render entry point.  beginFramePlan()
  // resolves the target and the frame kind once, so the internal and external
  // paths stop re-deriving (and re-validating) the same setup, and
  // recordFramePlan() dispatches the record step from the same decision.
  struct FramePlan {
    const SoVulkanRenderTarget * target = nullptr;
    bool overlaysOnly = false;
  };

  // Shared frame prologue of renderInternal()/renderExternal()/
  // renderExternalOverlay(): validate the target, cache the frame matrices,
  // advance the frame boundary, write the lighting setup, update the geometry
  // cache and -- when `reserveCompositeSlots` -- reserve the
  // countCompositeCommands() lighting slots the overlay path consumes (the
  // full path reserves inside recordFrame()).  Fills `plan`; returns false
  // after emitting the caller-specific error.  A non-null `timing` receives
  // the setup/geom sub-phase durations for the [RTDBG] line (tex/lod are filled
  // by beginExternalPrepass()).
  bool beginFramePlan(const SoDrawList & drawlist,
                      const SoRenderParams & params, const char * caller,
                      bool overlaysOnly, bool reserveCompositeSlots,
                      FramePlan & plan, ExternalFrameTiming * timing);

  // Record the frame described by `plan`: the composite (traced overlay +
  // overlay block) recorders for an overlays-only frame, the full opaque/
  // transparent frame otherwise.  Returns false only when the full record
  // failed.
  bool recordFramePlan(const SoDrawList & drawlist,
                       const SoRenderParams & params, const FramePlan & plan,
                       VkRenderPass renderPass, VkFramebuffer framebuffer,
                       VulkanRecordContext & ctx);

  // Shared prologue of renderExternal()/renderExternalOverlay(): validate the
  // caller-owned command buffer/render pass, run beginFramePlan() and mark the
  // caller's LOAD render pass as not cleared-by-load (recordClear() must emit
  // vkCmdClearAttachments).  Fills `plan`; returns false after emitting the
  // caller-specific error.
  bool prepareExternalFrame(
      const SoDrawList & drawlist, const SoRenderParams & params,
      VkCommandBuffer commandBuffer, VkRenderPass renderPass,
      const char * caller, bool overlaysOnly, bool reserveCompositeSlots,
      FramePlan & plan, ExternalFrameTiming * timing);
  // Validate params.renderTarget (non-null, image views set, non-zero extent),
  // emitting "invalid Vulkan render target" on failure.  Used by every render
  // entry point.
  const SoVulkanRenderTarget * validateRenderTarget(
      const SoRenderParams & params) const;

  // --- HDR output pass (raster path) -------------------------------------
  // Ensure the per-frame linear RGBA16F intermediate ring matches `outputTarget`'s
  // extent and is ready to render into; fills outTarget/outPass/outFramebuffer
  // for the current frame's ring slot and outDescriptorSet with the set that
  // samples it.  Recreates the images (and their render pass/framebuffers) when
  // the extent or the in-flight-frame count changes.
  bool ensureHdrIntermediate(const SoVulkanRenderTarget & outputTarget,
                             SoVulkanRenderTarget & outTarget,
                             VkRenderPass & outPass,
                             VkFramebuffer & outFramebuffer,
                             VkDescriptorSet & outDescriptorSet);
  // Lazily create the output pipeline for `outputPass` (keyed on the pass,
  // which encodes the swapchain color format + sample count).
  bool ensureOutputPipeline(VkRenderPass outputPass, VkPipeline & outPipeline);
  // Allocate/refresh the descriptor set bound to the intermediate image view.
  bool ensureOutputDescriptorSet(VkImageView source, VkDescriptorSet & outSet);
  // Release the intermediate image + depth and the offscreen pass/framebuffer
  // (deferred through the frame ring).  Keeps the pipelines/layouts/sampler.
  void releaseHdrIntermediate();
  // Release the intermediate image/pipeline/descriptor resources (device must
  // still be valid).
  void destroyHdrOutputResources();

  // --- Vulkan resource helpers -------------------------------------------
  // Buffer creation lives in the SoVulkanBufferFactory collaborator (this->
  // buffers); see SoVulkanBufferFactory.h.
  // Defer destruction of a buffer + its allocation to the deferred-destruction
  // ring (the submission that may still reference it must drain first).  Null
  // handles are ignored, so callers need not pre-check.
  void deferDestroyBufferMemory(VkBuffer buffer, VmaAllocation memory);
  // Ensure the per-instance model-matrix buffer holds at least `bytes`
  // (HOST_VISIBLE | HOST_COHERENT, persistently mapped).  Recreates + remaps
  // on growth; the old buffer is released through the deferred ring.
  bool ensureInstanceModelBuffer(VkDeviceSize bytes);
  // Size the per-instance model-matrix buffer to the full ring
  // (maxFramesInFlight * uboSlotsPerFrame * sizeof(float[16])), matching the
  // lighting-UBO ring the instance element index parallels
  // ((frameIndex % maxFramesInFlight) * uboSlotsPerFrame + slotIndex).  Called
  // at the same sites that create/resize the UBO ring so the per-draw path
  // never grows (or worse, re-creates) the buffer -- a per-draw vkMapMemory /
  // vkDestroyBuffer would otherwise race once workers record in parallel.
  bool ensureInstanceModelRingCapacity();
  bool growLightingUbo(uint32_t minSlots);
  bool swapLightingBuffer(VkBuffer newBuffer, VmaAllocation newMemory,
                          void * newMapped, uint32_t newSlotsPerFrame);
  // Replace the lighting constant ring buffer (the set-0 UBO) after a
  // maxFramesInFlight change, repointing its descriptor set at the new buffer
  // and deferring the old one.  Waits for in-flight submissions first, for the
  // same reason swapLightingBuffer() does.
  bool swapLightingConstBuffer(VkBuffer newBuffer, VmaAllocation newMemory,
                               void * newMapped);
  bool prepareLightingSlots(uint32_t neededDraws);
  void beginFrame();
  void flushPendingDestroys();
  void flushAllPendingDestroys();
  void deferDestroy(std::function<void()> && fn);
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
  // Vulkan Memory Allocator, created in initialize() and destroyed at
  // shutdown().  Owns the texture-image device memory.
  VmaAllocator vmaAllocator = nullptr;
  // Shared buffer-creation helpers, borrowing the handles above.  Initialized
  // in initialize() once the command pool exists.
  SoVulkanBufferFactory buffers;

  // --- Device capabilities (probed once in initialize()) -----------------
  // VkPhysicalDeviceFeatures::fillModeNonSolid gates the wireframe/points
  // overlay and the LINES/POINTS draw-style pipelines, which use
  // VK_POLYGON_MODE_LINE/POINT.  The embedding application enables the
  // feature only when the hardware supports it (see
  // QuarterVulkanWidget::configureDeviceFeatures), so without it creating
  // those pipelines would violate the spec; getOrCreatePipeline() refuses
  // them instead.
  bool fillModeNonSolid = false;
  // SAMPLED_IMAGE support for the two optional texture formats the 1- and
  // 2-component upload paths pick (VK_FORMAT_R8_UNORM / VK_FORMAT_R8G8_
  // UNORM).  VK_FORMAT_R8G8B8A8_UNORM (the fallback) is a required format.
  bool sampledR8 = false;
  bool sampledR8G8 = false;
  // VK_EXT_pipeline_creation_feedback enabled by the app; gates the optional
  // pipeline-cache-hit / creation-cost log (FC_VULKAN_PIPELINE_FEEDBACK).
  bool hasPipelineCreationFeedback = false;

  // Per-pass GPU timestamps (FC_VULKAN_GPU_TIMING).  Lazily initialized on the
  // first instrumented frame; a no-op when disabled or unsupported.
  SoVulkanGpuTimers gpuTimers;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  // One command buffer and fence per in-flight frame slot (this->frameRing).
  // The own-queue path (render()) submits slot N's buffer and signals slot N's
  // fence; beginFrame() waits the fence before reusing the slot's UBO ring
  // half, command buffer, and deferred-destruction batch.  The external path
  // records into the caller's command buffer and never signals these fences;
  // the ring's pending flag stays false for those slots so beginFrame() never
  // waits on them.
  SoVulkanFrameRing frameRing;
  // Secondary command buffers for M1c/M1d: one per in-flight frame slot per
  // worker, used to record the render-order-independent opaque pass inside an
  // already-begun render pass (RENDER_PASS_CONTINUE), then
  // vkCmdExecuteCommands()'d into the slot's primary buffer.  Kept per-slot
  // so a secondary is never reset while a primary that executed it is still
  // pending; the pool's RESET_COMMAND_BUFFER_BIT lets each be re-recorded
  // after its slot's fence is waited.
  //
  // ONE POOL PER WORKER: VkCommandPool host access is externally
  // synchronized per the Vulkan spec, so concurrent vkReset/vkBegin/vkEnd of
  // buffers allocated from a SHARED pool (M1d workers + main) races the
  // pool's internal allocator.  Pool w owns only worker w's secondaries.
  std::vector<VkCommandPool> secondaryCommandPools;
  // Secondaries are indexed [slot * maxRecordWorkers + worker], so a worker
  // always records into its own buffer of the current in-flight slot and a
  // secondary is never reset while a primary that executed it is pending.
  std::vector<VkCommandBuffer> secondaryCommandBuffers;
  // M1d parallel recording.  The opaque worklist is partitioned into disjoint
  // chunks, each recorded by one worker thread into its own secondary buffer
  // (slots are pre-assigned disjoint by M1b, so the mapped UBO/instance-model
  // writes are race-free).  The pool is persistent; worker 0 is the recording
  // thread and workers 1..N-1 are spawned threads, all joined in shutdown().
  bool parallelRecordEnabled = false;
  uint32_t maxRecordWorkers = 1;
  // True when the device was created with VK_EXT_nested_command_buffer and
  // nestedCommandBufferRendering, so a subpass may begin with
  // VK_SUBPASS_CONTENTS_INLINE_AND_SECONDARY_COMMAND_BUFFERS_EXT and both
  // record inline commands and execute secondaries.  False (the default, and
  // the offscreen/test case where the caps are unavailable) forces plain
  // INLINE contents and disables secondary recording.
  bool nestedCommandBufferEnabled = false;
  // Interaction LOD for this frame (see SoRenderParams::interactionLod): wide
  // lines are drawn as plain 1px GPU lines instead of being expanded into
  // quads on the CPU.  Latched from the frame params at the top of recordFrame
  // so every isWideLine() decision within the frame agrees.
  bool interactionLodActive = false;
  std::vector<std::thread> recordWorkers;
  std::vector<VulkanRecordContext> workerRecordContexts;
  struct ParallelRecordJob {
    const SoDrawList * drawlist = nullptr;
    const SoRenderParams * params = nullptr;
    const SoVulkanRenderTarget * target = nullptr;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    // This worker's chunk: pointers into the frame's worklist (workItemsScratch
    // is a backend member that stays alive for the whole recording).
    std::vector<const VulkanWorkItem *> items;
    VkCommandBuffer secondary = VK_NULL_HANDLE;
    VulkanRecordContext * ctx = nullptr;
    bool ok = false;
    // Wide-line expansion dispatch: when true the worker expands
    // `wideLineCommands` instead of recording a secondary chunk.  Reuses the
    // same pool/wakeup machinery; the two dispatches never overlap (the
    // expansion pre-pass is joined before recording starts).
    bool expandWideLines = false;
    std::vector<const SoRenderCommand *> wideLineCommands;
    // Intra-command split: when non-zero the worker expands segments
    // [wlineSplitBegin, wlineSplitEnd) of the single command in
    // wlineSplitCtx instead of `wideLineCommands`.  1 = clip transform +
    // validity, 2 = emit quads.  These dispatches originate on the owner
    // thread only, so a worker never re-enters the pool.
    int wlineSplitPhase = 0;
    uint32_t wlineSplitBegin = 0;
    uint32_t wlineSplitEnd = 0;
  };
  std::vector<ParallelRecordJob> recordJobs;
  std::mutex recordMutex;
  std::condition_variable recordCvSpawn, recordCvDone;
  // Monotonic dispatch generation: main increments it under recordMutex when
  // publishing a frame's jobs; each worker remembers the generation it already
  // processed and re-waits until a NEW one arrives, so every worker consumes
  // each dispatch exactly once (a plain ready flag would let a finished worker
  // immediately re-process the same frame while main is still waiting).
  uint32_t recordJobGeneration = 0;
  std::atomic<uint32_t> recordDoneCount {0};
  bool recordPoolStopped = false;
  // Per-recording command-buffer target + dedup state for the current
  // render()/renderExternal() call.  This names the backend's own buffer in
  // render(), or the caller's buffer in renderExternal().  Kept as one
  // member (single-threaded recording); worker threads get their own
  // VulkanRecordContext instances.
  VulkanRecordContext recordContext;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  // Set-0 layout: the lighting constant ring (binding 0, UBO dynamic).  The
  // single lightingDescriptorSet is allocated from it; the per-draw set-1
  // descriptor (draw UBO + texture) uses descriptorSetLayout.
  VkDescriptorSetLayout lightingSetLayout = VK_NULL_HANDLE;
  // Descriptor pools.  Sets are never reset wholesale while frames may be
  // in flight: when the active pool nears its capacity a fresh pool is
  // appended and becomes current, leaving every live set valid.  All pools
  // (and with them all outstanding sets) are destroyed at shutdown.
  std::vector<VkDescriptorPool> descriptorPools;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  // Sets allocated from the currently active descriptorPool (white fallback
  // set plus one per cached texture).
  uint32_t descriptorSetCount = 0;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  // Per-draw view/model/material uniform buffer (set 1, binding 0, dynamic
  // offset).  One large ring buffer holds maxFramesInFlight frames' worth of
  // per-command slots; every draw binds its own slot through the descriptor's
  // dynamic offset.  The GPU executes asynchronously, so a single shared
  // buffer rewritten per command would make every draw read the last
  // command's matrices; distinct slots per command and a ring half per
  // in-flight frame keep each draw's uniform data stable until its frame
  // completes.
  VkBuffer lightingBuffer = VK_NULL_HANDLE;
  VmaAllocation lightingMemory = nullptr;
  void * lightingMapped = nullptr;
  VkDeviceSize uboSlotStride = 0;
  uint32_t uboSlotsPerFrame = 0;
  uint32_t uboFrameIndex = 0;

  // Host-visible, persistently-mapped buffer holding the per-instance model
  // matrices for instanced drawing (binding 1, rate INSTANCE).  A group of
  // commands sharing geometry/material and differing only by model matrix is
  // drawn with a single vkCmdDraw(instanceCount=N); the N model matrices are
  // memcpy'd here and the attribute reads them per instance.  A one-element
  // buffer serves ordinary non-instanced draws.  Grows on demand; freed in
  // shutdown().
  VkBuffer instanceModelBuffer = VK_NULL_HANDLE;
  VmaAllocation instanceModelMemory = nullptr;
  void * instanceModelMapped = nullptr;
  VkDeviceSize instanceModelCapacity = 0;
  // Per-frame pre-conversion of the frame camera matrices (double -> float).
  // The main pass draws every command with the frame view/projection (only
  // scissor overlays carry their own matrices), so converting once per render
  // via cacheFrameMatrices() avoids a 16-value double -> float conversion in
  // updateLightingUniforms() and recordDrawCommand() on every draw.
  float frameViewFloats[16] = {};
  float frameProjFloats[16] = {};
  // Hoisted device-pixel ratio: cacheFrameMatrices() computes the frame's
  // scalar once so per-command push-constant code reads a member instead of
  // re-evaluating params.devicePixelRatio (a branch + member access) per draw.
  float frameDpr = 1.0f;
  void cacheFrameMatrices(const SoRenderParams & params);

  // Unique SoLightingHandle slots per in-flight frame in the lighting
  // constant ring.  updateLightingSetup() clamps the distinct-handle count to
  // this; the ring depth is maxFramesInFlight, so the total slot count is
  // maxFramesInFlight * kLightingConstSlotsPerFrame.  The depth must track
  // maxFramesInFlight or a frame's ring base would alias another frame's.
  static constexpr uint32_t kLightingConstSlotsPerFrame = 8;

  // Lighting constant ring (set 0, binding 0, dynamic offset).  Holds a few
  // slots per in-flight frame, one per distinct SoLightingHandle the frame
  // references.  Each unique block is written once per frame (updateLightingSetup);
  // every draw that shares a handle binds the same slot through its dynamic
  // offset, so the 8-light setup is computed once, not per draw.
  VkBuffer lightingConstBuffer = VK_NULL_HANDLE;
  VmaAllocation lightingConstMemory = nullptr;
  void * lightingConstMapped = nullptr;
  VkDeviceSize lightingConstStride = 0;
  uint32_t lightingConstMaxSlots = 0;
  // The single set-0 descriptor (lighting UBO).  Bound on every draw together
  // with the per-draw/texture set-1 descriptor.
  VkDescriptorSet lightingDescriptorSet = VK_NULL_HANDLE;
  // Handle -> byte offset into the lighting ring for the current frame.  Built
  // by updateLightingSetup() before recording and consumed by
  // updateLightingUniforms()/recordDrawCommand().
  std::unordered_map<SoLightingHandle, VkDeviceSize> lightingSlotOffsets;

  // Authoritative viewer lighting pushed by the GL host via setSceneLights().
  // When its light list is non-empty, updateLightingSetup() uses this
  // camera-anchored world-space set for every command instead of the
  // per-command IR SoLightingData, so the raster path matches the RT path and
  // the GL viewer's view-relative lights.
  SoLightingData sceneLighting;

  // Resources replaced during recording are destroyed maxFramesInFlight
  // frames later, after the submissions that still reference them have
  // certainly completed.  Batch B holds the destroys deferred during the
  // frame recording of batch B's frame; it is flushed at the start of the
  // frame with the same ring index, N frames later.
  uint32_t maxFramesInFlight = 3;
  SoVulkanShared::PendingDestroys pendingDestroys;

  // Vulkan-only display overlays (shaded-with-edges / show-vertices).
  // Configured through the manager; never part of the shared render params.
  SbBool wireframeOverlay = FALSE;
  SbBool pointsOverlay = FALSE;
  // Debug overlay: re-draw the triangle commands in polygon-LINES mode so
  // the raw tessellation (triangle edges) is visible on top of the shaded
  // geometry.  Distinct from the wireframe/edge overlay, which draws only
  // the true B-Rep feature-edge line commands.
  // Debug overlay: re-draw the triangle commands in polygon-LINES so the raw
  // tessellation is visible (see buildWorkItems()).
  SbBool tessellationOverlay = FALSE;
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);

  // Per-command texture entries, the staging pool and the white fallback
  // texture.  Uploads gathered during updateGeometryCache() are recorded into
  // the frame command buffer (own-queue path) or the external pre-pass's
  // transient buffer; eviction and index re-resolution happen inside the
  // cache's sweep().
  SoVulkanTextureCache textureCache;

  VkShaderModule vertexModule = VK_NULL_HANDLE;
  VkShaderModule fragmentModule = VK_NULL_HANDLE;

  // Wide-line (line width > 1 and/or stippled line pattern) pipeline shaders.
  VkShaderModule wideLineVertexModule = VK_NULL_HANDLE;
  VkShaderModule wideLineFragmentModule = VK_NULL_HANDLE;
  // GPU-instanced wide-line vertex shader (no CPU quad expansion); shares the
  // fragment shader above.  Used for main-pass, non-stippled line commands.
  VkShaderModule wideLineInstancedVertexModule = VK_NULL_HANDLE;

  // GPU sub-pixel geometry LOD resources (raster only).  The compute pipeline
  // compacts eligible triangle commands' indices while the camera moves; the
  // pipeline/descriptor layout live for the backend's lifetime, the per-slot
  // buffers live in the geometry cache entries.
  VkShaderModule subPixelCullModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout subPixelSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout subPixelPipelineLayout = VK_NULL_HANDLE;
  VkPipeline subPixelCullPipeline = VK_NULL_HANDLE;
  // Append-only storage-buffer descriptor pools.  A set is never freed while
  // a frame may reference it, so when the active pool fills a fresh one is
  // appended (mirroring descriptorPools); all are destroyed at shutdown.
  std::vector<VkDescriptorPool> subPixelDescriptorPools;
  uint32_t subPixelDescriptorSetCount = 0;
  // Largest storage-buffer range the device accepts for one binding
  // (VkPhysicalDeviceLimits::maxStorageBufferRange), cached when the pipeline
  // is built.  A huge CAD mesh's vertex buffer can exceed it, in which case
  // the pre-pass must skip the command (the descriptor update would otherwise
  // be invalid) and the full draw is used.  0 means "unknown / do not guard".
  uint64_t subPixelMaxStorageRange = 0;

  // Background gradient resources (no descriptor sets; push constants only).
  VkShaderModule backgroundVertexModule = VK_NULL_HANDLE;
  VkShaderModule backgroundFragmentModule = VK_NULL_HANDLE;
  VkPipelineLayout backgroundPipelineLayout = VK_NULL_HANDLE;

  // Render passes are cached by their VkRenderPassCreateInfo identity
  // (color/depth format, sample count, image layouts) plus the color/depth
  // load ops.  Pipelines are keyed on the render-pass handle (see
  // PipelineKey), so reusing the same pass across targets that differ only in
  // their images/extent keeps the pipeline cache warm -- in particular for
  // swapchain targets whose images cycle every frame.  The cache also keeps the
  // per-target framebuffer, recreated on any target change while the render
  // pass itself survives.  All of that state lives in the owned
  // SoVulkanRenderPassCache so the backend records a frame with the current
  // pass/framebuffer without duplicating the cache bookkeeping.
  SoVulkanRenderPassCache renderPasses;

  // Per-frame clear-by-load flags for the pass being recorded.  Set by each
  // frame entry point (recordFrame(), prepareExternalFrame(),
  // renderExternalHdr()) before recordClear() consults them.  These describe
  // the *current frame*, not any cached pass, so they live on the backend
  // rather than on the render-pass cache.
  bool frameColorClearedByLoad_ = false;
  bool frameDepthClearedByLoad_ = false;

  // --- HDR output pass (raster path) -------------------------------------
  // When hdrOutput is set, renderExternalHdr() renders the scene into a linear
  // RGBA16F intermediate (hdrColorImage) and presents it into the caller's
  // swapchain framebuffer with the exposure + PQ transform.  The intermediate
  // render pass/framebuffer are created through a dedicated cache so they do
  // not disturb renderPasses (which the internal path owns).
  bool hdrOutput = false;
  float hdrExposure = 0.02f;
  // Tone-mapping operator applied before the PQ encode (0 = clip, 1 = Reinhard,
  // 2 = ACES, 3 = Hable); see OutputFragment.glsl.
  int hdrToneMap = 1;
  SoVulkanRenderPassCache hdrPasses;
  // One ring slot per in-flight frame: each owns the offscreen RGBA16F
  // color/depth images, the framebuffer binding them, and the output
  // descriptor set that samples the color view.  Ringing is required because
  // QVulkanWindow keeps several frames in flight; without it, frame N+1's
  // offscreen pass (which discards and rewrites the intermediate) could race
  // frame N's output pass, which is still sampling it.
  struct HdrFrame {
    VkImage colorImage = VK_NULL_HANDLE;
    VmaAllocation colorMemory = nullptr;
    VkImageView colorView = VK_NULL_HANDLE;
    VkImage depthImage = VK_NULL_HANDLE;
    VmaAllocation depthMemory = nullptr;
    VkImageView depthView = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkDescriptorSet outputDescriptorSet = VK_NULL_HANDLE;
  };
  std::vector<HdrFrame> hdrFrames;
  VkRenderPass hdrRenderPass = VK_NULL_HANDLE;
  VkExtent2D hdrExtent {0, 0};
  // Output-pass shaders and pipeline, keyed per output render pass (the pass
  // changes with the swapchain color format).
  VkShaderModule outputVertexModule = VK_NULL_HANDLE;
  VkShaderModule outputFragmentModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout outputSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout outputPipelineLayout = VK_NULL_HANDLE;
  std::unordered_map<VkRenderPass, VkPipeline> outputPipelines;
  // Append-only descriptor pools (a set is never freed while a frame may
  // reference it), mirroring descriptorPools/subPixelDescriptorPools.
  std::vector<VkDescriptorPool> outputDescriptorPools;
  uint32_t outputDescriptorSetCount = 0;
  VkSampler outputSampler = VK_NULL_HANDLE;
  // Sample count of the caller's output pass, set before ensureOutputPipeline()
  // so the output pipeline matches the pass's MSAA state.
  VkSampleCountFlagBits hdrOutputPassSampleCount = VK_SAMPLE_COUNT_1_BIT;

  // Pipeline store: keyed by the retained state that affects the created
  // pipeline.  Vulkan pipelines are immutable, so every topology/fill/depth/
  // blend/sample-count combination gets its own entry.  The key types and the
  // persistent VkPipelineCache handle live in SoVulkanPipelineCache.
  SoVulkanPipelineCache pipelines;

  // The wide-line quad-expansion scratch (clip cache, per-vertex distance,
  // quad vertices) lives in thread_local vectors inside expandWideLines(): the
  // expansion runs on the parallel record workers, so a single shared scratch
  // would race.  See SoVulkanRenderBackendWideLine.cpp.
  //
  // Commands collected for the parallel expansion dispatch (main thread only;
  // cleared and reused, never reallocated in steady state).
  std::vector<const SoRenderCommand *> wlineExpandScratch;
  // Thread that drives the expansion pre-pass (the recording/GUI thread).
  // Used to keep the wide-line diagnostics on that one thread: interleaved
  // worker-thread debug writes are pure noise (and the diagnostics are the
  // only place the expansion touches shared I/O).
  std::thread::id wlineOwnerThread;

  // Intra-command wide-line split.  The thread_local scratch inside
  // expandWideLines() is per-worker because each worker owns whole commands;
  // the split path partitions ONE command, so its phases share these
  // owner-sized buffers: phase 1 writes disjoint clip/valid ranges, the owner
  // prefix-sums the quad offsets, phase 2 writes disjoint quad ranges.  Sized
  // once per command and reused across frames.  Only touched on the owner
  // thread (and read/written by workers during a joined dispatch).
  struct WideLineSplitCtx {
    const SoRenderCommand * command = nullptr;
    const SoGeometryDesc * geometry = nullptr;
    SbMat mvp;
    float vpWidth = 1.0f;
    float vpHeight = 1.0f;
    float lineWidth = 1.0f;
    float nearEps = 1.0e-5f;
    // Phase 2 writes the compacted quads straight into the slot's mapped
    // buffer (host-visible), so no intermediate quad array + full-size memcpy
    // is needed -- that copy is hundreds of MB for a million-segment edge set.
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

  // True while this backend is used only to composite overlays/residual
  // geometry over a ray-traced frame (set by setOverlayCompositeMode()).  Lets
  // updateGeometryCache() sweep stale entries on overlays-only frames so the
  // traced triangle geometry the RT backend owns is not kept resident here as
  // well.
  bool overlayCompositeMode = false;
  // Monotonic visit stamp for the overlay-composite sweep (see
  // VulkanCachedCommand::compositeEpoch).  Bumped once per overlays-only pass
  // while overlayCompositeMode is set.
  uint32_t overlayCompositeEpoch = 0;

  // Reusable batch-key bucket map for recordFrame()'s opaque batching pass.
  // Previously a fresh std::unordered_map per frame; reused via clear() so an
  // ordinary frame does not heap-allocate the bucket table + key vectors.
  std::unordered_map<uint64_t, std::vector<const SoRenderCommand *>> batchBucketScratch;
  // Reusable worklist produced by buildWorkItems() and consumed by the record
  // path (M1b) and, later, secondary-buffer / parallel workers (M1c/M1d).
  // Cleared per frame and refilled so an ordinary frame does not heap-allocate
  // the item vector.
  std::vector<VulkanWorkItem> workItemsScratch;

  // Reusable scratch for recordFrame()'s secondary/parallel record paths so an
  // ordinary frame does not heap-allocate them (recording is single-threaded
  // per backend; the parallel workers only read the items these hold).
  std::vector<const VulkanWorkItem *> opaqueItemsScratch;
  std::vector<std::pair<uint64_t, const VulkanWorkItem *>> heaviestScratch;
  std::vector<uint64_t> loadScratch;
  std::vector<VkCommandBuffer> executeScratch;

  // Sampler cache so textures sharing filter/wrap state reuse one VkSampler
  // instead of creating one per texture entry.  Owned by this collaborator;
  // initialized in initialize() and destroyed at shutdown() while the device
  // is still valid.  Always created lazily on first use.
  SoVulkanSamplerCache samplerCache;

  SoVulkanGeometryArena geometryArena;
  // Per-command geometry cache entries, lookup, upload paths and eviction
  // sweep; borrows the arena above and the buffer factory.  The scratch it
  // reuses across frames lives with it.
  SoVulkanGeometryCache geometryCache;
};

#endif // COIN_SOVULKANRENDERBACKEND_H
