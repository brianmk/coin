// src/rendering/SoVulkanRenderBackend.h

#ifndef COIN_SOVULKANRENDERBACKEND_H
#define COIN_SOVULKANRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include "rendering/SoVulkanShared.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanMemPool.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRecordContext.h"

#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Shared combine step for the hand-rolled hash functors below.  Keeping one
// implementation prevents the == operator and the hash from drifting apart.
static inline size_t hashCombine(size_t hash, size_t value)
{
  return hash ^ (value + 0x9e3779b9 + (hash << 6) + (hash >> 2));
}

/*!
  \brief Immutable graphics-pipeline identity.

  Pipelines are cached in \c pipelineCache keyed by this struct.  It is
  defined before VulkanCachedCommand so each cached command can remember the
  exact key it last resolved to, letting getOrCreatePipeline() skip re-hashing
  (and the map lookup) for an unchanged command on the steady-state path.
*/
struct PipelineKey {
  VkRenderPass renderPass = VK_NULL_HANDLE;
  uint8_t topology = 0;
  uint8_t fillMode = 0;
  uint8_t cullMode = 0;
  uint8_t ccwFrontFace = 1;
  bool depthTestEnable = false;
  bool depthWriteEnable = false;
  uint8_t depthFunction = 0;
  bool depthBiasEnable = false;
  float depthBiasConstantFactor = 0.0f;
  float depthBiasSlopeFactor = 0.0f;
  bool blendEnable = false;
  uint8_t blendSrcRGB = 0;
  uint8_t blendDstRGB = 0;
  uint8_t blendSrcAlpha = 0;
  uint8_t blendDstAlpha = 0;
  uint8_t blendEquationRGB = 0;
  uint8_t blendEquationAlpha = 0;
  bool stencilEnable = false;
  uint8_t stencilFunction = 0;
  uint8_t stencilReference = 0;
  uint8_t stencilCompareMask = 0xFF;
  uint8_t stencilWriteMask = 0xFF;
  uint8_t stencilFailOp = 0;
  uint8_t stencilZFailOp = 0;
  uint8_t stencilZPassOp = 0;
  uint32_t sampleCount = 1;
  bool wideLine = false;

  bool operator==(const PipelineKey & other) const
  {
    return renderPass == other.renderPass && topology == other.topology &&
      fillMode == other.fillMode && cullMode == other.cullMode &&
      ccwFrontFace == other.ccwFrontFace &&
      depthTestEnable == other.depthTestEnable &&
      depthWriteEnable == other.depthWriteEnable &&
      depthFunction == other.depthFunction &&
      depthBiasEnable == other.depthBiasEnable &&
      (!depthBiasEnable ||
       (depthBiasConstantFactor == other.depthBiasConstantFactor &&
        depthBiasSlopeFactor == other.depthBiasSlopeFactor)) &&
      blendEnable == other.blendEnable &&
      (!blendEnable ||
       (blendSrcRGB == other.blendSrcRGB &&
        blendDstRGB == other.blendDstRGB &&
        blendSrcAlpha == other.blendSrcAlpha &&
        blendDstAlpha == other.blendDstAlpha &&
        blendEquationRGB == other.blendEquationRGB &&
        blendEquationAlpha == other.blendEquationAlpha)) &&
      stencilEnable == other.stencilEnable &&
      (!stencilEnable ||
       (stencilFunction == other.stencilFunction &&
        stencilReference == other.stencilReference &&
        stencilCompareMask == other.stencilCompareMask &&
        stencilWriteMask == other.stencilWriteMask &&
        stencilFailOp == other.stencilFailOp &&
        stencilZFailOp == other.stencilZFailOp &&
        stencilZPassOp == other.stencilZPassOp)) &&
      sampleCount == other.sampleCount && wideLine == other.wideLine;
  }
};

/*!
  \brief Cached GPU geometry for one retained SoRenderCommand.

  Vulkan buffers are packed per command: one interleaved vertex buffer (fixed
  48-byte stride: position + normal + color + texcoord) and one optional
  uint32 index buffer.  Unlike the GL backend, the Vulkan backend always uses
  the same vertex layout so a single static vertex-input state can be shared
  by every pipeline.
*/
struct VulkanCachedCommand {
  VkBuffer vertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
  VkBuffer indexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory indexMemory = VK_NULL_HANDLE;
  VkDeviceSize vertexOffset = 0;
  VkDeviceSize indexOffset = 0;
  uint32_t sharedBlockId = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;

  // CPU-expanded wide-line quads (per-frame content; line width > 1 or a
  // stipple pattern).  One host-visible scratch buffer per in-flight frame
  // slot: the quad expansion is a function of the projection matrix, so the
  // content is rewritten every frame, and the rewrite for frame N +
  // maxFramesInFlight may not clobber data a still-executing frame N reads.
  // The slot for the current frame index is selected by beginFrame(), which
  // waits the slot's fence before the slot is reused.
  struct VulkanWideLineBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
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
  // Content hash of the uploaded streams: pointer identity alone cannot
  // detect in-place edits (the per-frame arena hands out the same pointers
  // for unchanged layouts), which would otherwise serve stale geometry.
  uint64_t contentHash = 0;

  // Pipeline-resolution fast path (getOrCreatePipeline()).  The exact
  // PipelineKey resolved for this command last is stored verbatim, plus the
  // handle it produced.  A match (cheap field-by-field equality, no hashing)
  // skips rebuilding the key and the per-frame pipelineCache unordered_map
  // lookup for unchanged commands.  The entry lives and dies with the
  // geometry cache, which invalidateCache() clears together with the
  // pipeline cache, so these fields never outlive the handles they name.
  PipelineKey resolvedKey;
  VkPipeline resolvedPipeline = VK_NULL_HANDLE;
  bool hasResolvedPipeline = false;
};

/*! \brief Cached GPU texture for one retained command's SoTextureData. */
struct VulkanCachedTexture {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  // Offset of `memory` into the sub-allocator block it came from (0 when the
  // memory is a standalone vkAllocateMemory -- the legacy path).  Needed to
  // return the range when FC_VULKAN_MEM_POOL is enabled.
  VkDeviceSize memoryOffset = 0;
  // Size of the `memory` range (the sub-allocated block size / allocation
  // size).  Tracked so releaseMemory() returns exactly what was allocated.
  VkDeviceSize memorySize = 0;
  VkImageView view = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  // Pool the descriptor set was allocated from (pools are append-only, so
  // the set must be returned to this pool, not the currently active one).
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

  // Command that last touched this entry (per-frame arena pointer; used
  // only as an identity key for map rebuilds after cache eviction).
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
  // Content hash of the uploaded pixels (sampled): pixel-pointer identity
  // alone cannot detect in-place edits, which would serve stale textures.
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

    The caller must already have begun \a commandBuffer and started
    \a renderPass with a compatible framebuffer; it also owns submission and
    presentation.  This path does not begin/end the command buffer, begin/end
    the render pass, create a framebuffer, or submit to the queue.  It is used
    by embedding surfaces such as QVulkanWindow whose command buffer and render
    pass lifecycle are managed by the window system.
  */
  SbBool renderExternal(const SoDrawList & drawlist,
                        const SoRenderParams & params,
                        VkCommandBuffer commandBuffer,
                        VkRenderPass renderPass);

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
    \brief Configure Vulkan-only display overlays.

    These toggle the wireframe/point edge overlays and their color.  They are
    deliberately backend state rather than SoRenderParams fields, so the
    OpenGL backend never sees them.
  */
  void setWireframeOverlay(SbBool enabled);
  void setPointsOverlay(SbBool enabled);
  void setEdgeColor(const SbColor4f & color);

private:
  // --- Initialization helpers -------------------------------------------
  bool createCommandPool();
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createLightingUniformBuffer();
  bool createLightingConstBuffer();
  bool createLightingDescriptorSet();
  bool createPipelineLayout();
  bool createRenderPass(const SoVulkanRenderTarget & target,
                        VkAttachmentLoadOp colorLoadOp,
                        VkAttachmentLoadOp depthLoadOp,
                        VkRenderPass & renderPass);
  bool createShaders(VkShaderModule & vertexModule,
                     VkShaderModule & fragmentModule);
  bool createWideLineShaders();
  bool createBackgroundResources();
  bool createPipelineCache();
  bool createBackgroundPipeline(const SoVulkanRenderTarget & target,
                                VkRenderPass renderPass,
                                VkPipeline & pipeline);
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
                           bool overlayPass = false);

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
                              bool unlit = false);
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
  VulkanCachedCommand & getOrCreateCache(const SoRenderCommand * command);
  void uploadGeometry(VulkanCachedCommand & entry,
                      const SoRenderCommand & command);
  bool uploadGeometryShared(VulkanCachedCommand & entry,
                            const SoRenderCommand & command,
                            uint32_t blockId);
  void destroyCacheEntry(VulkanCachedCommand & entry);

  struct VulkanGeometryBlock {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void * mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize used = 0;
    uint32_t refCount = 0;
  };

  uint32_t allocateGeometryBlock(VkDeviceSize capacity);
  bool allocateGeometryArena(uint32_t blockId, VkDeviceSize size,
                             VkDeviceSize & offset);
  void releaseGeometryBlock(uint32_t blockId);
  void deferReleaseGeometryBlock(uint32_t blockId);
  void destroyAllGeometryBlocks();

  // --- Texture cache ----------------------------------------------------
  bool createWhiteTexture();
  void invalidateTextureCache();
  void destroyTextureEntry(VulkanCachedTexture & entry);
  VulkanCachedTexture & getOrCreateTexture(const SoRenderCommand * command);

  // One texture waiting for its GPU-side upload (staging copy).  The host
  // side (image, memory, staging buffer) is prepared up front; the copies
  // for all pending uploads are recorded either into the current frame's
  // command buffer (own-queue path) or a single transient command buffer
  // (external path) and submitted once per frame.  The cache index (not a
  // pointer) identifies the entry, but eviction may compact the cache after
  // preparation, so the command pointer is retained to re-resolve the index
  // before the upload is consumed.
  struct PendingTextureUpload {
    size_t index = 0;
    const SoRenderCommand * command = nullptr;
    const SoTextureData * texture = nullptr;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  };
  bool prepareTextureUpload(VulkanCachedTexture & entry,
                            const SoTextureData & texture,
                            VkBuffer & staging,
                            VkDeviceMemory & stagingMemory);
  void recordTextureUpload(VkCommandBuffer commandBuffer,
                           const VulkanCachedTexture & entry,
                           const SoTextureData & texture,
                           VkBuffer staging);
  bool finalizeTexture(VulkanCachedTexture & entry,
                       const SoTextureData & texture);
  bool recordPendingTextureUploads();
  void finalizePendingTextureUploads();
  bool flushPendingTextureUploadsExternal();
  bool createSampler(SoTextureFilter minFilter, SoTextureFilter magFilter,
                     SoTextureWrap wrapS, SoTextureWrap wrapT,
                     VkSampler & sampler);
  // Format actually used for an N-component SoTextureData image.  VK_FORMAT_
  // R8_UNORM / R8G8_UNORM / R8G8B8_UNORM are not guaranteed to be sampleable
  // (they are optional formats), so when the device lacks SAMPLED_IMAGE
  // support the upload is expanded to R8G8B8A8_UNORM on the host with the
  // per-channel values matching the native sampling semantics.
  VkFormat effectiveTextureFormat(const int numComponents) const;
  bool allocateTextureDescriptorSet(VkImageView view, VkSampler sampler,
                                    VkDescriptorSet & set);
  bool ensureDescriptorPoolSpace();
  VkDescriptorSet resolveTextureSet(const SoRenderCommand & command);
  // Release a texture image or staging buffer's device memory back to the
  // sub-allocator when enabled (FC_VULKAN_MEM_POOL), else vkFreeMemory as the
  // legacy path.  The caller must have recorded the offset (from a pool alloc)
  // into the entry/staging record — when the pool is disabled the memory is a
  // standalone allocation and offset is 0.  Destroying the VkBuffer/VkImage
  // for the pool case is the caller's responsibility (the memory block outlives
  // the buffer/image); this helper only returns the memory.
  void releaseMemory(VkDeviceMemory memory, VkDeviceSize size,
                     VkDeviceSize offset);
  // True when sub-allocating transient texture memory (FC_VULKAN_MEM_POOL).
  bool usingMemPool() const { return this->memPool != nullptr; }

  // --- Render recording ---------------------------------------------------
  // Every record* helper below takes the VulkanRecordContext it records
  // into and touches no other recording state, so (future) worker threads
  // can each record their own command buffer with their own dedup cache.
  bool beginCommandBuffer();
  VkCommandBuffer currentCommandBuffer();
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
                   VulkanRecordContext & ctx);

  // --- Vulkan resource helpers -------------------------------------------
  bool createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkBuffer & buffer,
                    VkDeviceMemory & memory,
                    const void * data);
  // Device-local variant of createBuffer() for retained static geometry.
  // Uses a transient staging buffer + one-shot transfer and waits for the
  // copy to complete, so it is only meant for the rare geometry-change
  // path, never the steady-state per-frame path.
  bool createBufferDeviceLocal(VkDeviceSize size,
                               VkBufferUsageFlags usage,
                               VkBuffer & buffer,
                               VkDeviceMemory & memory,
                               const void * data);
  // Pick a memory type for `requirements` that satisfies `desired` properties
  // and a compatible memoryTypeBits.  Uses the shared cached
  // SoVulkanShared::MemoryProperties (memProps) so the memory-type search lives
  // in one place.  Returns false when no suitable type exists.
  bool selectMemoryType(const VkMemoryRequirements & requirements,
                        VkMemoryPropertyFlags desired,
                        uint32_t & memoryTypeIndex);
  // Allocate device memory for an already-created `buffer` and bind it.  On
  // failure `memory` is left null and the caller destroys `buffer`.
  bool allocateBufferMemory(VkBuffer buffer,
                            const VkMemoryRequirements & requirements,
                            VkMemoryPropertyFlags desiredProperties,
                            VkDeviceMemory & memory);
  // Create a buffer backed by memory with the desired properties.  When
  // `data` is non-null the host-visible contents are filled.  On failure
  // buffer/memory are left null.
  bool createBufferWithProperties(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags desiredProperties,
                                  VkBuffer & buffer, VkDeviceMemory & memory,
                                  const void * data = nullptr);
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
  bool swapLightingBuffer(VkBuffer newBuffer, VkDeviceMemory newMemory,
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
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  const VkAllocationCallbacks * allocator = nullptr;
  // Cached physical-device memory-properties picker (shared helper); bound to
  // physicalDevice in initialize().  Replaces the old per-backend
  // deviceMemoryProperties + deviceMemoryPropertiesValid cache.
  SoVulkanShared::MemoryProperties memProps;

  // Sub-allocator for the high-churn transient resources (texture image memory
  // and texture staging buffers), so each upload does not vkAllocateMemory /
  // vkFreeMemory against the driver (slow, and counts against
  // maxMemoryAllocationCount).  Enabled by FC_VULKAN_MEM_POOL (off by default);
  // when disabled the legacy per-resource allocate path is used.
  std::unique_ptr<SoVulkanMemPool> memPool;

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

  VkCommandPool commandPool = VK_NULL_HANDLE;
  // One command buffer and fence per in-flight frame slot.  The own-queue
  // path (render()) submits slot N's buffer and signals slot N's fence;
  // beginFrame() waits the fence before reusing the slot's UBO ring half,
  // command buffer, and deferred-destruction batch.  The external path
  // records into the caller's command buffer and never signals these
  // fences; frameFencePending stays false for those slots so beginFrame()
  // never waits on them.
  std::vector<VkCommandBuffer> frameCommandBuffers;
  std::vector<VkFence> frameFences;
  std::vector<uint8_t> frameFencePending;

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
  VkDeviceMemory lightingMemory = VK_NULL_HANDLE;
  void * lightingMapped = nullptr;
  VkDeviceSize uboSlotStride = 0;
  uint32_t uboSlotsPerFrame = 0;
  uint32_t uboFrameIndex = 0;
  uint32_t uboCmdIndex = 0;

  // Host-visible, persistently-mapped buffer holding the per-instance model
  // matrices for instanced drawing (binding 1, rate INSTANCE).  A group of
  // commands sharing geometry/material and differing only by model matrix is
  // drawn with a single vkCmdDraw(instanceCount=N); the N model matrices are
  // memcpy'd here and the attribute reads them per instance.  A one-element
  // buffer serves ordinary non-instanced draws.  Grows on demand; freed in
  // shutdown().
  VkBuffer instanceModelBuffer = VK_NULL_HANDLE;
  VkDeviceMemory instanceModelMemory = VK_NULL_HANDLE;
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

  // Lighting constant ring (set 0, binding 0, dynamic offset).  Holds a few
  // slots per in-flight frame, one per distinct SoLightingHandle the frame
  // references.  Each unique block is written once per frame (updateLightingSetup);
  // every draw that shares a handle binds the same slot through its dynamic
  // offset, so the 8-light setup is computed once, not per draw.
  VkBuffer lightingConstBuffer = VK_NULL_HANDLE;
  VkDeviceMemory lightingConstMemory = VK_NULL_HANDLE;
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
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);

  // Texture uploads gathered during updateGeometryCache().  On the own-queue
  // path they are recorded into the frame command buffer (no separate
  // submit); on the external path they are flushed through one transient
  // submission.  Indices are re-resolved from the command pointers after
  // cache eviction compacts the texture cache.
  std::vector<PendingTextureUpload> pendingUploads;

  // Texture binding (set 0, binding 1).  A 1x1 white fallback texture is
  // bound whenever a command carries no embedded SoTextureData.
  VkImage whiteImage = VK_NULL_HANDLE;
  VkDeviceMemory whiteImageMemory = VK_NULL_HANDLE;
  VkImageView whiteImageView = VK_NULL_HANDLE;
  VkSampler whiteSampler = VK_NULL_HANDLE;
  VkDescriptorSet whiteDescriptorSet = VK_NULL_HANDLE;

  VkShaderModule vertexModule = VK_NULL_HANDLE;
  VkShaderModule fragmentModule = VK_NULL_HANDLE;

  // Wide-line (line width > 1 and/or stippled line pattern) pipeline shaders.
  VkShaderModule wideLineVertexModule = VK_NULL_HANDLE;
  VkShaderModule wideLineFragmentModule = VK_NULL_HANDLE;

  // Background gradient resources (no descriptor sets; push constants only).
  VkShaderModule backgroundVertexModule = VK_NULL_HANDLE;
  VkShaderModule backgroundFragmentModule = VK_NULL_HANDLE;
  VkPipelineLayout backgroundPipelineLayout = VK_NULL_HANDLE;

  // Render passes are cached by their VkRenderPassCreateInfo identity
  // (color/depth format, sample count, image layouts).  Pipelines are keyed
  // on the render-pass handle (see PipelineKey), so reusing the same pass
  // across targets that differ only in their images/extent keeps the
  // pipeline cache warm -- in particular for swapchain targets whose images
  // cycle every frame.
  struct RenderPassIdentity {
    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkImageLayout depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // Load ops distinguish a render pass that clears its attachments at begin
    // (full-target clear fast path, FC_VULKAN_RP_CLEAR) from one that loads
    // them and clears via vkCmdClearAttachments.  Two passes that differ only
    // in loadOp must not share a cache entry.
    VkAttachmentLoadOp colorLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkAttachmentLoadOp depthLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

    bool operator==(const RenderPassIdentity & other) const
    {
      return colorFormat == other.colorFormat &&
        depthFormat == other.depthFormat &&
        sampleCount == other.sampleCount &&
        colorLayout == other.colorLayout &&
        depthLayout == other.depthLayout &&
        colorLoadOp == other.colorLoadOp &&
        depthLoadOp == other.depthLoadOp;
    }
  };
  struct RenderPassIdentityHash
  {
    size_t operator()(const RenderPassIdentity & key) const
    {
      size_t hash = std::hash<uint32_t>()(
        static_cast<uint32_t>(key.colorFormat));
      hash = hashCombine(hash,
                         std::hash<uint32_t>()(static_cast<uint32_t>(key.depthFormat)));
      hash = hashCombine(hash,
                         std::hash<uint32_t>()(static_cast<uint32_t>(key.sampleCount)));
      hash = hashCombine(hash,
                         std::hash<uint32_t>()(static_cast<uint32_t>(key.colorLayout)));
      hash = hashCombine(hash,
                         std::hash<uint32_t>()(static_cast<uint32_t>(key.depthLayout)));
      hash = hashCombine(hash,
                         std::hash<uint32_t>()(static_cast<uint32_t>(key.colorLoadOp)));
      hash = hashCombine(hash,
                         std::hash<uint32_t>()(static_cast<uint32_t>(key.depthLoadOp)));
      return hash;
    }
  };
  RenderPassIdentity renderPassIdentity(const SoVulkanRenderTarget & target) const;
  VkRenderPass getOrCreateRenderPass(const SoVulkanRenderTarget & target,
                                     VkAttachmentLoadOp colorLoadOp,
                                     VkAttachmentLoadOp depthLoadOp);
  std::unordered_map<RenderPassIdentity, VkRenderPass, RenderPassIdentityHash>
    renderPassCache;

  // Render pass used by the current frame (looked up from renderPassCache).
  VkRenderPass renderPass = VK_NULL_HANDLE;
  // Whether the current frame's render pass clears the color/depth attachment
  // via its loadOp (full-target-clear fast path).  When true, recordClear()
  // skips the redundant vkCmdClearAttachments and the begin info supplies the
  // corresponding clear value.
  bool renderPassColorCleared = false;
  bool renderPassDepthCleared = false;

  // Framebuffer cached for the current target identity (image views +
  // extent + render pass).  Swapchain targets cycle their images every
  // frame, so this is recreated on any target change while the render pass
  // itself survives in renderPassCache.
  VkFramebuffer renderPassFramebuffer = VK_NULL_HANDLE;
  VkRenderPass renderPassFramebufferPass = VK_NULL_HANDLE;
  VkImage renderPassFramebufferColorImage = VK_NULL_HANDLE;
  VkImageView renderPassFramebufferColorView = VK_NULL_HANDLE;
  VkImage renderPassFramebufferDepthImage = VK_NULL_HANDLE;
  VkImageView renderPassFramebufferDepthView = VK_NULL_HANDLE;
  VkExtent2D renderPassFramebufferExtent {0, 0};

  // Pipeline cache: keyed by the retained state that affects the created
  // pipeline.  Vulkan pipelines are immutable, so every topology/fill/depth/
  // blend/sample-count combination gets its own entry.  PipelineKey and
  // PipelineKeyHash are defined near the top of this class so VulkanCachedCommand
  // can store a resolved key.
  struct PipelineKeyHash
  {
    size_t operator()(const PipelineKey & key) const
    {
      size_t hash = std::hash<uintptr_t>()(
        reinterpret_cast<uintptr_t>(key.renderPass));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.topology));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.fillMode));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.cullMode));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.ccwFrontFace));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.depthTestEnable));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.depthWriteEnable));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.depthFunction));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.depthBiasEnable));
      hash = hashCombine(hash, std::hash<float>()(key.depthBiasConstantFactor));
      hash = hashCombine(hash, std::hash<float>()(key.depthBiasSlopeFactor));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendEnable));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendSrcRGB));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendDstRGB));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendSrcAlpha));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendDstAlpha));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendEquationRGB));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.blendEquationAlpha));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilEnable));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilFunction));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilReference));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilCompareMask));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilWriteMask));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilFailOp));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilZFailOp));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.stencilZPassOp));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.sampleCount));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.wideLine));
      return hash;
    }
  };

  std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelineCache;

  // Persistent pipeline cache.  vkCreateGraphicsPipelines() is passed this
  // handle so the driver can reuse shader/state blobs across the many variant
  // pipelines the backend creates lazily on the draw path.  Without a cache
  // (VK_NULL_HANDLE) the first appearance of each state combination on a
  // frame stutters.  Created once in initialize(), destroyed in shutdown().
  VkPipelineCache pipelineCacheHandle = VK_NULL_HANDLE;

  // Background pipeline cache: keyed on the render pass and sample count only
  // (the gradient pipeline has no retained per-command state).
  struct BackgroundPipelineKey {
    VkRenderPass renderPass = VK_NULL_HANDLE;
    uint32_t sampleCount = 1;
    bool operator==(const BackgroundPipelineKey & other) const
    {
      return renderPass == other.renderPass &&
             sampleCount == other.sampleCount;
    }
  };
  struct BackgroundPipelineKeyHash
  {
    size_t operator()(const BackgroundPipelineKey & key) const
    {
      size_t hash = std::hash<uintptr_t>()(
        reinterpret_cast<uintptr_t>(key.renderPass));
      hash = hashCombine(hash, std::hash<uint32_t>()(key.sampleCount));
      return hash;
    }
  };
  std::unordered_map<BackgroundPipelineKey, VkPipeline,
                     BackgroundPipelineKeyHash> backgroundPipelineCache;

  // Reusable CPU scratch for the wide-line quad expansion.  expandWideLines()
  // previously allocated clipCache/distances/quads as fresh std::vector per
  // line per frame -- for line-heavy scenes that is thousands of heap
  // alloc/free per frame.  These are reused via assign() (no realloc when
  // capacity is sufficient), so only the fill cost remains.
  std::vector<float> wlineClipScratch;
  std::vector<float> wlineDistScratch;
  std::vector<float> wlineQuadScratch;

  std::vector<VulkanCachedCommand> gpuCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  std::vector<VulkanCachedTexture> textureCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToTexture;

  // Reusable batch-key bucket map for recordFrame()'s opaque batching pass.
  // Previously a fresh std::unordered_map per frame; reused via clear() so an
  // ordinary frame does not heap-allocate the bucket table + key vectors.
  std::unordered_map<uint64_t, std::vector<const SoRenderCommand *>> batchBucketScratch;

  // Packed sampler-state key: minFilter | magFilter << 2 | wrapS << 4 | wrapT << 6.
  typedef uint8_t SamplerKey;
  static SamplerKey samplerKey(SoTextureFilter minFilter,
                               SoTextureFilter magFilter,
                               SoTextureWrap wrapS, SoTextureWrap wrapT);
  // Sampler cache so textures sharing filter/wrap state reuse one VkSampler
  // instead of creating one per texture entry.  Owned here; destroyed at
  // shutdown() after the texture cache.  Always created lazily on first use.
  std::unordered_map<SamplerKey, VkSampler> samplerCache;
  VkSampler cachedSampler(SoTextureFilter minFilter, SoTextureFilter magFilter,
                          SoTextureWrap wrapS, SoTextureWrap wrapT);

  std::vector<VulkanGeometryBlock> geometryBlocks;
  // Released blocks are kept as reusable ids so a geometry-change burst does
  // not grow geometryBlocks without bound; releaseGeometryBlock() returns an
  // id here once its refCount drops to 0.
  std::vector<uint32_t> freeGeometryBlockIds;
  VkDeviceSize nextGeometryBlockCapacity = 256u * 1024u;

  // Reusable scratch packing buffer for uploadGeometry().  Resized but never
  // reallocated across successive uploads, so the interleaved repack does not
  // churn the heap on every geometry change.  Stores the 32-byte packed layout
  // (see VULKAN_VERTEX_STRIDE), so it is byte-addressed.
  std::vector<uint8_t> uploadScratch;
  // Reusable per-draw "needs geometry upload" flag buffer for
  // updateGeometryCache().  Grows to the largest draw list seen, then keeps
  // its capacity so an ordinary frame does not heap-allocate a fresh vector
  // sized by the command count.
  std::vector<uint8_t> needsGeometryScratch;
  // Guards uploadScratch.  Geometry uploads run on the render path; if a
  // backend instance is ever driven from more than one thread (e.g. a shared
  // shape rendered by parallel camerase), this prevents two threads packing
  // into and reading out of the same scratch vector simultaneously.
  std::mutex uploadScratchMutex;
};

#endif // COIN_SOVULKANRENDERBACKEND_H
