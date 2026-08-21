// src/rendering/SoVulkanRenderBackend.h

#ifndef COIN_SOVULKANRENDERBACKEND_H
#define COIN_SOVULKANRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

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
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;

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
};

/*! \brief Cached GPU texture for one retained command's SoTextureData. */
struct VulkanCachedTexture {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
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

private:
  // --- Initialization helpers -------------------------------------------
  bool createCommandPool();
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createLightingUniformBuffer();
  bool createPipelineLayout();
  bool createRenderPass(const SoVulkanRenderTarget & target,
                        VkRenderPass & renderPass);
  bool createShaders(VkShaderModule & vertexModule,
                     VkShaderModule & fragmentModule);
  bool createBackgroundResources();
  bool createBackgroundPipeline(const SoVulkanRenderTarget & target,
                                VkRenderPass renderPass,
                                VkPipeline & pipeline);
  void recordBackground(const SoRenderParams & params,
                        const SoVulkanRenderTarget & target,
                        VkRenderPass renderPass);
  bool getOrCreatePipeline(const SoRenderCommand & command,
                           const SoVulkanRenderTarget & target,
                           VkRenderPass renderPass,
                           VkPipeline & pipeline,
                           bool transparent,
                           int fillModeOverride = -1,
                           bool overlayPass = false);

  // --- Per-draw lighting ------------------------------------------------
  void updateLightingUniforms(const SoDrawList & drawlist,
                              const SoRenderCommand & command,
                              const SoRenderParams & params,
                              VkDeviceSize uboOffset,
                              bool unlit = false);

  // --- Geometry cache ---------------------------------------------------
  void invalidateCache();
  void updateGeometryCache(const SoDrawList & drawlist, bool overlaysOnly = false);
  VulkanCachedCommand & getOrCreateCache(const SoRenderCommand * command);
  void uploadGeometry(VulkanCachedCommand & entry,
                      const SoRenderCommand & command);
  void destroyCacheEntry(VulkanCachedCommand & entry);

  // --- Texture cache ----------------------------------------------------
  bool createWhiteTexture();
  void invalidateTextureCache();
  void destroyTextureEntry(VulkanCachedTexture & entry);
  VulkanCachedTexture & getOrCreateTexture(const SoRenderCommand * command);

  // One texture waiting for its GPU-side upload (staging copy).  The host
  // side (image, memory, staging buffer) is prepared up front; the copies
  // for all pending uploads are recorded into a single transient command
  // buffer and submitted once per frame.  The cache index (not a pointer)
  // identifies the entry: the vector may reallocate while uploads are
  // still being prepared.
  struct PendingTextureUpload {
    size_t index = 0;
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
  bool flushTextureUploads(std::vector<PendingTextureUpload> & pending);
  bool createSampler(const SoTextureData & texture, VkSampler & sampler);
  bool allocateTextureDescriptorSet(VkImageView view, VkSampler sampler,
                                    VkDescriptorSet & set);
  bool ensureDescriptorPoolSpace();
  VkDescriptorSet resolveTextureSet(const SoRenderCommand & command);

  // --- Render recording ---------------------------------------------------
  bool beginCommandBuffer();
  void recordClear(const SoRenderParams & params,
                   const SoVulkanRenderTarget & target);
  void recordDrawCommand(const SoDrawList & drawlist,
                         const SoRenderCommand & command,
                         const SoVulkanRenderTarget & target,
                         const SoRenderParams & params,
                         VkRenderPass renderPass,
                         bool transparent,
                         int fillModeOverride = -1,
                         const float * uniformColorOverride = nullptr,
                         bool overlayPass = false);
  bool endAndSubmit();
  void applyViewport(const SoRenderParams & params,
                     const SoVulkanRenderTarget & target);
  void applyCommandViewport(const SoRenderCommand & command,
                            const SoVulkanRenderTarget & target);
  void applyScissor(const SoRenderCommand & command,
                    const SoVulkanRenderTarget & target);
  void recordOverlayDepthClear(const SoRenderCommand & command,
                               const SoVulkanRenderTarget & target);
  void recordOverlayBlock(const SoDrawList & drawlist,
                          const SoRenderParams & params,
                          const SoVulkanRenderTarget & target,
                          VkRenderPass renderPass);
  SbBool renderInternal(const SoDrawList & drawlist,
                        const SoRenderParams & params,
                        bool overlaysOnly);
  bool recordFrame(const SoDrawList & drawlist,
                   const SoRenderParams & params,
                   const SoVulkanRenderTarget & target,
                   VkRenderPass renderPass);

  // --- Vulkan resource helpers -------------------------------------------
  bool createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkBuffer & buffer,
                    VkDeviceMemory & memory,
                    const void * data);
  bool growLightingUbo(uint32_t minSlots);
  bool prepareLightingSlots(uint32_t neededDraws);
  void beginFrame();
  void flushPendingDestroys();
  void flushAllPendingDestroys();
  void deferDestroy(std::function<void()> && fn);
  void deferDestroyCacheEntry(VulkanCachedCommand & entry);
  void deferDestroyTextureEntry(VulkanCachedTexture & entry);

  // --- Owned device ------------------------------------------------------
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  const VkAllocationCallbacks * allocator = nullptr;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  // Command buffer being recorded by the current render()/renderExternal()
  // call.  This is the backend's own buffer in render(), or the caller's
  // buffer in renderExternal().
  VkCommandBuffer activeCommandBuffer = VK_NULL_HANDLE;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
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

  // Per-draw lighting/material uniform buffer (set 0, binding 0, dynamic
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

  // Resources replaced during recording are destroyed maxFramesInFlight
  // frames later, after the submissions that still reference them have
  // certainly completed.  Batch B holds the destroys deferred during the
  // frame recording of batch B's frame; it is flushed at the start of the
  // frame with the same ring index, N frames later.
  uint32_t maxFramesInFlight = 3;
  std::vector<std::vector<std::function<void()>>> pendingDestroys;

  // Texture binding (set 0, binding 1).  A 1x1 white fallback texture is
  // bound whenever a command carries no embedded SoTextureData.
  VkImage whiteImage = VK_NULL_HANDLE;
  VkDeviceMemory whiteImageMemory = VK_NULL_HANDLE;
  VkImageView whiteImageView = VK_NULL_HANDLE;
  VkSampler whiteSampler = VK_NULL_HANDLE;
  VkDescriptorSet whiteDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorPool whiteDescriptorPool = VK_NULL_HANDLE;

  VkShaderModule vertexModule = VK_NULL_HANDLE;
  VkShaderModule fragmentModule = VK_NULL_HANDLE;

  // Background gradient resources (no descriptor sets; push constants only).
  VkShaderModule backgroundVertexModule = VK_NULL_HANDLE;
  VkShaderModule backgroundFragmentModule = VK_NULL_HANDLE;
  VkPipelineLayout backgroundPipelineLayout = VK_NULL_HANDLE;

  // Render pass is owned per target identity (image + extent).
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkImage renderPassColorImage = VK_NULL_HANDLE;
  VkImageView renderPassColorView = VK_NULL_HANDLE;
  VkImage renderPassDepthImage = VK_NULL_HANDLE;
  VkImageView renderPassDepthView = VK_NULL_HANDLE;
  VkExtent2D renderPassExtent {0, 0};
  // Framebuffer cached beside the render pass (same target identity);
  // recreated only when the target changes instead of every frame.
  VkFramebuffer renderPassFramebuffer = VK_NULL_HANDLE;

  // Pipeline cache: keyed by the retained state that affects the created
  // pipeline.  Vulkan pipelines are immutable, so every topology/fill/depth/
  // blend/sample-count combination gets its own entry.
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
        sampleCount == other.sampleCount;
    }
  };

  struct PipelineKeyHash
  {
    size_t operator()(const PipelineKey & key) const
    {
      size_t hash = std::hash<uintptr_t>()(
        reinterpret_cast<uintptr_t>(key.renderPass));
      hash ^= std::hash<uint32_t>()(key.topology) + 0x9e3779b9 + (hash << 6) +
        (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.fillMode) + 0x9e3779b9 + (hash << 6) +
        (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.cullMode) + 0x9e3779b9 + (hash << 6) +
        (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.ccwFrontFace) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.depthTestEnable) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.depthWriteEnable) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.depthFunction) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.depthBiasEnable) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<float>()(key.depthBiasConstantFactor) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<float>()(key.depthBiasSlopeFactor) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendEnable) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendSrcRGB) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendDstRGB) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendSrcAlpha) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendDstAlpha) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendEquationRGB) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.blendEquationAlpha) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilEnable) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilFunction) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilReference) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilCompareMask) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilWriteMask) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilFailOp) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilZFailOp) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.stencilZPassOp) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      hash ^= std::hash<uint32_t>()(key.sampleCount) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      return hash;
    }
  };

  std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelineCache;

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
      hash ^= std::hash<uint32_t>()(key.sampleCount) + 0x9e3779b9 +
        (hash << 6) + (hash >> 2);
      return hash;
    }
  };
  std::unordered_map<BackgroundPipelineKey, VkPipeline,
                     BackgroundPipelineKeyHash> backgroundPipelineCache;

  std::vector<VulkanCachedCommand> gpuCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  std::vector<VulkanCachedTexture> textureCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToTexture;
};

#endif // COIN_SOVULKANRENDERBACKEND_H
