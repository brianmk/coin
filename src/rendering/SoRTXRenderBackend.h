// src/rendering/SoRTXRenderBackend.h

#ifndef COIN_SORTXRENDERBACKEND_H
#define COIN_SORTXRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if COIN_BUILD_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

#if COIN_BUILD_RTX_DENOISER
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
// optix_function_table_definition.h is included in exactly one .cpp (the
// denoiser TU) because it defines the optixFunctionTable symbol.
#endif

/*!
  \brief Cached GPU acceleration structure for one retained SoRenderCommand.

  The BLAS is built from a tightly packed position-only vertex buffer (stride
  12, three floats) plus the command's optional uint32 index buffer.  All
  three buffers are device-local and carry SHADER_DEVICE_ADDRESS_BIT so the
  acceleration structure can reference them.
*/
struct RTXCachedGeometry {
  VkBuffer vertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
  VkBuffer indexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory indexMemory = VK_NULL_HANDLE;
  VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
  VkBuffer blasBuffer = VK_NULL_HANDLE;
  VkDeviceMemory blasMemory = VK_NULL_HANDLE;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  // Device address of the BLAS, captured once at build time.  It never changes
  // for a given BLAS, so querying it with vkGetAccelerationStructureDeviceAddressKHR
  // every frame (per instance) is pure driver-call overhead.
  uint64_t devAddr = 0;

  // Identity keys mirroring the producer-owned storage of the last build.
  // The storage is a per-frame arena (SoIRRenderAction::geometryPool), so
  // pointers are NOT stable across frames; the content hash below is the
  // real identity signal.
  const float * posKey = nullptr;
  const uint32_t * idxKey = nullptr;
  uint32_t vertexStride = 0;
  uint32_t cacheGeneration = 0;
  uint64_t contentHash = 0;
  // Cheap per-frame change signal (hashGeometrySignal).  When it matches
  // contentHash is reused; only a mismatch triggers the full hashGeometry().
  uint64_t changeSignal = 0;
  // Command that last touched this entry (per-frame arena pointer; used
  // only as an identity key for map rebuilds after cache eviction).
  const SoRenderCommand * commandKey = nullptr;
  // Offset of this command's triangle normals in the normal pool
  // (UINT32_MAX = not uploaded yet).
  uint32_t normalPoolOffset = 0xFFFFFFFFu;
  uint32_t normalCount = 0;
  // Offset/count of this command's emissive triangles in the NEE pool
  // (UINT32_MAX = not uploaded yet).  Rebuilt per frame in buildNeePool().
  uint32_t neePoolOffset = 0xFFFFFFFFu;
  uint32_t neeCount = 0;

  // Refit state: when a content change keeps the topology intact (vertex
  // and index counts, stride, indexing unchanged) and only the vertex
  // positions moved, the BLAS is updated in place with
  // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR instead of being
  // destroyed and rebuilt.  vertexHash/indexHash split the content identity
  // so index (topology) changes can be told apart from position changes.
  bool refitPending = false;
  uint64_t vertexHash = 0;
  uint64_t indexHash = 0;
};

/*!
  \brief Vulkan ray-tracing executor for retained DrawList IR.

  Consumes the same SoDrawList/SoRenderParams contract as the raster Vulkan
  backend but replaces rasterization with VK_KHR_ray_tracing_pipeline:
  one BLAS per retained geometry command, one TLAS per frame, a five-group
  shader binding table (raygen, miss, shadow miss, closest hit, shadow
  closest hit) driving a path tracer that writes a storage image, and a
  fullscreen-triangle present pass that samples the storage image into the
  swapchain color attachment.

  The backend requires a Vulkan 1.2+ device with VK_KHR_acceleration_structure
  and VK_KHR_ray_tracing_pipeline enabled (the embedding application must
  request these via the Qt/QVulkanWindow extension hooks).  When unavailable,
  the embedding falls back to SoVulkanRenderBackend.

  v1 scope: opaque triangle geometry only (transparent commands are skipped
  in the TLAS and the material buffer is indexed by the draw-list command
  index).  No textures, no MSAA on the traced image (present pass runs at
  the swapchain sample count).
*/
/*!
  \brief One material record per draw-list command, matching the std430
  layout of the RTMaterial block in PathTrace.glsl.

  C++ packs the float arrays without padding, which matches std430: 5 vec4
  (diffuse/ambient/specular/emissive/params) + 6 arrays of 8 vec4 (lights) +
  triangleData + pbr = 848 + 32 = 880 bytes.  Defined here so the backend can
  own a reusable std::vector<RTMaterial> scratch buffer without keeping a
  per-frame heap allocation.
*/
struct RTMaterial {
  float diffuse[4];
  float ambient[4];
  float specular[4];
  float emissive[4];
  float params[4]; // x = shininess, y = twoSided, z = lightCount,
                   // w = shadingModel (0 = unlit, 1 = gouraud)
  float lightType[8 * 4];
  float lightColor[8 * 4];
  float lightDirection[8 * 4];
  float lightPosition[8 * 4];
  float lightAttenuation[8 * 4];
  float lightSpot[8 * 4];
  float triangleData[4]; // x = triangle-normal pool offset, y = normal count,
                         // z = NEE pool offset, w = NEE entry count
  float pbr[4]; // x = metalness, y = roughness, z = usePbr, w = unused
};

class SoRTXRenderBackend : public SoRenderBackend {
public:
  SoRTXRenderBackend();
  ~SoRTXRenderBackend() override;

  /*!
    \brief Ray-traced view mode (which rendering stage the tracer runs).

    This augments (does not replace) ptEnabled: ptEnabled is the legacy
    "is the ray tracer rendering" flag that also owns the accumulate/denoise
    state machine.  RtxModeAmbientOcclusion engages the ray tracer as a
    real-time single-sample preview that traces occlusion rays per pixel
    (u_state.y == 2) without accumulation or denoising; RtxModePathTrace is
    the full accumulating path tracer (u_state.y == 1).  RtxModeOff leaves
    ptEnabled untouched (callers that want raster set ptEnabled false instead).
  */
  enum class RtxViewMode { RtxModeOff = 0, RtxModeAmbientOcclusion, RtxModePathTrace, RtxModeEnvironment };

  const char * getName() const override;
  SbBool initialize(const SoRenderBackendInitParams & params) override;
  void shutdown() override;
  SbBool render(const SoDrawList & drawlist,
                const SoRenderParams & params) override;

  /*!
    \brief Enable/disable path tracing on the ray-tracing backend.

    When disabled the backend traces a single primary ray per pixel with
    direct Gouraud lighting (the v1 behavior).  When enabled it runs a
    multi-bounce path tracer with next-event estimation, shadow rays and
    Russian roulette.  The toggle takes effect on the next render() /
    renderExternal() call.  Requires the ray-tracing backend to be active.
  */
  void setPathTracingEnabled(SbBool enabled);

  //! Set the ray-traced view mode (see RtxViewMode).
  void setViewMode(RtxViewMode mode);
  //! Current ray-traced view mode (see RtxViewMode).
  RtxViewMode getViewMode(void) const;

  //! Configure the procedural environment/IBL params used by
  //! RtxModeEnvironment (sky intensity, world-space sun direction, sun color
  //! and the sky-lighting scale) plus a per-mode enable.
  void setEnvIntensity(float intensity);
  void setEnvSunDir(float x, float y, float z);
  void setEnvSunColor(float r, float g, float b);
  void setEnvSunPower(float power);
  void setEnvSkyBrightness(float brightness);

  //! True when path tracing is enabled (see setPathTracingEnabled()).
  SbBool getPathTracingEnabled(void) const;

  /*!
    \brief Start-flag for progressive path-tracing refinement.

    Setting the flag to TRUE starts (or restarts) the progressive
    accumulation: the accumulated radiance buffer is reset and one jittered
    sample per frame is added as long as the camera and scene stay
    unchanged.  Any camera move or scene change automatically drops back to
    a single-sample live preview until the flag is raised again.  Setting
    the flag to FALSE cancels accumulation.  Only meaningful while path
    tracing is enabled.
  */
  void setPathTracingStart(SbBool start);

  //! True while the accumulation is running (start flag consumed).
  SbBool getPathTracingActive(void) const;

  //! True while path tracing still needs continuous frames: accumulating, or
  //! in the post-move settle window waiting to auto-restart.  The embedding
  //! viewport uses this to keep requesting updates (a progressive renderer
  //! must not go idle between the preview and the auto-restart).
  SbBool getPathTracingRefining(void) const;

  //! Samples accumulated in the current progressive run (0 when idle).
  uint32_t getPathTracingSampleCount(void) const;

  /*!
    \brief Set the maximum number of path-tracing bounces (1..16).

    Higher bounce counts add more indirect-light transport at the cost of
    noisier early frames.  The env override FC_VULKAN_PT_BOUNCES sets the
    default before initialize().
  */
  void setPathTracingBounces(uint32_t bounces);

  /*!
    \brief Frames of a static camera before the accumulation auto-restarts
    (1..120, default 6).

    After a camera or scene change the backend drops to a 1-sample live
    preview; once the camera stays static for this many frames a fresh
    accumulation starts automatically.  The env override FC_VULKAN_PT_SETTLE
    sets the default before initialize().
  */
  void setPathTracingSettleFrames(uint32_t frames);

  /*!
    \brief Set the accumulated-sample cap before the run auto-stops (default
    256).  The env override FC_VULKAN_PT_MAXSAMPLES sets the default before
    initialize().
  */
  void setPathTracingMaxSamples(uint32_t samples);

  /*!
    \brief Enable/disable the edge-stopping denoise pass (default on).

    When disabled, the present pass shows the raw accumulated radiance, so
    the Monte-Carlo noise of the early accumulation frames is visible.
  */
  void setPathTracingDenoiseEnabled(SbBool enabled);

  /*!
    \brief Select the denoiser backend by name ("rtx", "oidn", "fsr",
    "none").  The selection is applied on the next path-tracing buffer
    (re)creation (the resolve in createDenoiseBackend is keyed on the
    store here, falling back to the FC_VULKAN_PT_DENOISER env var when the
    setter is never called).  An empty string leaves the current choice.
  */
  void setDenoiserFilter(const char * denoiser);

  /*!
    \brief Record the draw list into a caller-owned command buffer/render pass.

    Same contract as SoVulkanRenderBackend::renderExternal(): the caller has
    already begun \a commandBuffer and started \a renderPass; the backend
    records only the present pass into it and never submits the caller's
    buffer.  Acceleration-structure builds, buffer copies and the ray trace
    itself are recorded on a private one-shot command buffer which is
    submitted and waited on internally (vkCmdTraceRaysKHR and AS builds are
    not allowed inside a render pass), before the caller's pass samples the
    traced image.
  */
  SbBool renderExternal(const SoDrawList & drawlist,
                        const SoRenderParams & params,
                        VkCommandBuffer commandBuffer,
                        VkRenderPass renderPass);

private:
  // --- Initialization helpers -------------------------------------------
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createPipelines();
  bool createShaderBindingTable();
  bool createPresentPipeline(VkRenderPass renderPass,
                             VkSampleCountFlagBits sampleCount);
  bool createShaderModules();
  bool createFrameBuffer();
  bool updateDescriptors();

  // --- Buffer helpers -----------------------------------------------------
  bool createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkBuffer & buffer, VkDeviceMemory & memory);
  bool createHostVisibleBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkBuffer & buffer, VkDeviceMemory & memory);
  bool createScratchBuffer(VkDeviceSize size);
  bool createStorageImage(uint32_t width, uint32_t height);
  bool createPathTracingBuffers(uint32_t width, uint32_t height);
  VkDeviceAddress getDeviceAddress(VkBuffer buffer);

  // --- Geometry / acceleration structure cache --------------------------
  void invalidateCache();
  void updateGeometryCache(const SoDrawList & drawlist);
  RTXCachedGeometry & getOrCreateCache(const SoRenderCommand * command);
  bool buildBlas(RTXCachedGeometry & entry, const SoRenderCommand & command,
                 VkCommandBuffer cmd);
  bool refitBlas(RTXCachedGeometry & entry, const SoRenderCommand & command,
                 VkCommandBuffer cmd);
  void destroyCacheEntry(RTXCachedGeometry & entry);
  bool buildTlas(const SoDrawList & drawlist, VkCommandBuffer cmd);

  // --- Material buffer --------------------------------------------------
  void updateMaterials(const SoDrawList & drawlist);

  // --- Frame recording ---------------------------------------------------
  bool recordAccelerationStructures(const SoDrawList & drawlist,
                                    const SoRenderParams & params,
                                    const SoVulkanRenderTarget & target,
                                    VkCommandBuffer cmd);
  void updatePathTracingState(const SoDrawList & drawlist,
                              const SoRenderParams & params,
                              const SoVulkanRenderTarget & target,
                              VkCommandBuffer cmd);
  bool recordTraceAndPresent(const SoRenderParams & params,
                             const SoVulkanRenderTarget & target,
                             VkCommandBuffer cmd,
                             VkRenderPass renderPass);
  VkCommandBuffer beginTransientCommandBuffer();
  void releaseTransientCommandBuffer();

  // --- Device handles ----------------------------------------------------
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  const VkAllocationCallbacks * allocator = nullptr;

  // Cached physical-device identity used to gate the NVIDIA-only CUDA/OptiX
  // denoiser.  The ray-query compute path tracer itself is vendor-neutral
  // (it only needs VK_KHR_ray_query, which AMD RDNA2+/Intel Arc also expose),
  // but the external CUDA/OptiX denoiser only works when the Vulkan device is
  // an NVIDIA GPU and the CUDA context is bound to that same device.  Query
  // these once at initialize() so the denoiser selection can fall back to
  // OIDN instead of grabbing the wrong CUDA device on non-NVIDIA or
  // multi-GPU machines.
  uint32_t deviceVendorID = 0;
  uint32_t deviceID = 0;
  // True once the physical device has been identified as NVIDIA.  Drives the
  // RTX denoiser gate; the Vulkan ray-tracing pipeline itself is unaffected.
  bool deviceIsNvidia = false;
  // Thin-process device UUID (VK_UUID_SIZE bytes) used to bind the CUDA
  // context to the SAME GPU that owns the Vulkan device, so the
  // VK_KHR_external_memory_fd interop imports physically-identical memory.
  // Empty when VkPhysicalDeviceIDProperties is unavailable on this device.
  uint8_t deviceUUID[16] = {0};
  bool haveDeviceUUID = false;

  // Persistent transient command pool + buffer for the one-shot
  // acceleration-structure phase (BLAS/TLAS builds and buffer copies, which
  // are not allowed inside a render pass).  Allocated once instead of per
  // frame; the caller submits and waits the buffer every frame.
  VkCommandPool transientPool = VK_NULL_HANDLE;
  VkCommandBuffer transientCommandBuffer = VK_NULL_HANDLE;

  // --- RT pipeline resources ---------------------------------------------
  // The tracer has two dispatch modes:
  //
  //  - Ray tracing pipeline (FC_VULKAN_RT_SBT=1): a VK_KHR_ray_tracing_pipeline
  //    with a five-group shader binding table (raygen, miss, shadow miss,
  //    closest hit, shadow closest hit) driving a traceRayEXT-based path
  //    tracer.
  //
  //  - Ray-query compute (default): the same path tracer as a compute
  //    shader on VK_KHR_ray_query.  This is the default because NVIDIA
  //    driver 610.x hangs the GPU when a triangle hit group executes
  //    (verified against RADV, which runs the SBT path correctly).
  VkDescriptorSetLayout rtSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout presentSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  // Double-buffered descriptor sets: one frame's sets are bound while the
  // previous frame's submission may still be pending, so each frame updates
  // the inactive pair (VUID-vkUpdateDescriptorSets-None-03047).
  VkDescriptorSet rtDescriptorSets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSet presentDescriptorSets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  uint32_t descriptorSetIndex = 0;

  VkPipelineLayout rtPipelineLayout = VK_NULL_HANDLE;
  VkPipelineLayout presentPipelineLayout = VK_NULL_HANDLE;
  VkPipeline rtPipeline = VK_NULL_HANDLE; //!< ray tracing pipeline (SBT mode)
  VkPipeline computePipeline = VK_NULL_HANDLE; //!< ray-query compute pipeline
  VkPipeline presentPipeline = VK_NULL_HANDLE;
  VkRenderPass presentRenderPass = VK_NULL_HANDLE; //!< cache key for present pipeline
  VkSampleCountFlagBits presentSampleCount =
    VK_SAMPLE_COUNT_1_BIT; //!< cache key for present pipeline

  // Cached offscreen render pass + framebuffer (render()), keyed on the
  // target identity so they are not recreated every frame.
  VkRenderPass offscreenRenderPass = VK_NULL_HANDLE;
  VkFramebuffer offscreenFramebuffer = VK_NULL_HANDLE;
  VkImage offscreenColorImage = VK_NULL_HANDLE;
  VkImageView offscreenColorView = VK_NULL_HANDLE;
  VkFormat offscreenColorFormat = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits offscreenSampleCount = VK_SAMPLE_COUNT_1_BIT;
  VkExtent2D offscreenExtent = {};

  VkShaderModule pathTraceModule = VK_NULL_HANDLE; //!< ray-query compute tracer
  VkShaderModule raygenModule = VK_NULL_HANDLE;
  VkShaderModule missModule = VK_NULL_HANDLE;
  VkShaderModule shadowMissModule = VK_NULL_HANDLE;
  VkShaderModule closestHitModule = VK_NULL_HANDLE;
  VkShaderModule shadowClosestHitModule = VK_NULL_HANDLE;
  VkShaderModule presentVertexModule = VK_NULL_HANDLE;
  VkShaderModule presentFragmentModule = VK_NULL_HANDLE;

  //! Dispatch mode: TRUE = ray tracing pipeline + SBT, FALSE = ray-query
  //! compute (default; see the resource section comment for why).
  bool useSbtPipeline = FALSE;

  // Shader binding table (five records: raygen, miss, shadow miss, closest
  // hit, shadow closest hit) plus the three strided device-address regions
  // handed to vkCmdTraceRaysKHR.
  VkBuffer sbtBuffer = VK_NULL_HANDLE;
  VkDeviceMemory sbtMemory = VK_NULL_HANDLE;
  uint32_t sbtGroupHandleSize = 32; //!< raw vkGetRayTracingShaderGroupHandlesKHR size
  uint32_t sbtGroupBaseAlignment = 64; //!< required region device-address alignment
  VkDeviceSize sbtRecordSize = 32;  //!< handle size aligned up for the record stride
  VkDeviceSize sbtBaseOffset = 0;   //!< aligned-base offset inside sbtBuffer
  VkStridedDeviceAddressRegionKHR raygenSbtRegion {};
  VkStridedDeviceAddressRegionKHR missSbtRegion {};
  VkStridedDeviceAddressRegionKHR hitSbtRegion {};
  VkStridedDeviceAddressRegionKHR callableSbtRegion {};

  // --- Runtime-resolved KHR entry points --------------------------------
  // The system Vulkan loader exports only core entry points; the ray tracing
  // KHR functions are resolved per-device with vkGetDeviceProcAddr().
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
  PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
  PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = nullptr;
  PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR = nullptr;
  PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = nullptr;

  // --- Per-frame resources ------------------------------------------------
  VkImage storageImage = VK_NULL_HANDLE;
  VkDeviceMemory storageImageMemory = VK_NULL_HANDLE;
  VkImageView storageImageView = VK_NULL_HANDLE;
  VkSampler presentSampler = VK_NULL_HANDLE;
  uint32_t storageWidth = 0;
  uint32_t storageHeight = 0;
  bool storageImageNeedsLayoutInit = false;

  // --- Path tracing resources ---------------------------------------------
  // Accumulation buffer (vec4 per pixel: rgb = radiance sum, a = sample
  // count) plus first-bounce G-buffers (world normal, world position with
  // hit distance in w) for the denoising present pass.  Recreated when the
  // viewport size changes.
  VkBuffer accumBuffer = VK_NULL_HANDLE;
  VkDeviceMemory accumMemory = VK_NULL_HANDLE;
  VkBuffer normalBuffer = VK_NULL_HANDLE;
  VkDeviceMemory normalMemory = VK_NULL_HANDLE;
  VkBuffer positionBuffer = VK_NULL_HANDLE;
  VkDeviceMemory positionMemory = VK_NULL_HANDLE;
  // Adaptive sampling: per-pixel radiance sums-of-squares (variance test)
  // and a per-frame host-readable active-pixel counter.
  VkBuffer sumSqBuffer = VK_NULL_HANDLE;
  VkDeviceMemory sumSqMemory = VK_NULL_HANDLE;
  VkBuffer activeCounterBuffer = VK_NULL_HANDLE;
  VkDeviceMemory activeCounterMemory = VK_NULL_HANDLE;
  void * activeCounterMapped = nullptr;
  // Temporal reprojection history: copies of the previous traced frame's
  // accumulation, sums-of-squares and world positions.  Handles are
  // swapped with the live buffers after every traced frame, so the shader
  // can carry converged samples across camera moves.
  VkBuffer accumHistoryBuffer = VK_NULL_HANDLE;
  VkDeviceMemory accumHistoryMemory = VK_NULL_HANDLE;
  VkBuffer sumSqHistoryBuffer = VK_NULL_HANDLE;
  VkDeviceMemory sumSqHistoryMemory = VK_NULL_HANDLE;
  VkBuffer positionHistoryBuffer = VK_NULL_HANDLE;
  VkDeviceMemory positionHistoryMemory = VK_NULL_HANDLE;
  // Set once at least one traced frame has been swapped into history; the
  // reprojection path only runs with valid history (fresh buffers and
  // resizes reset it).
  SbBool ptHistoryValid = FALSE;
  // Set on the first accumulating frame after a camera-only change (or an
  // auto-restart): the shader reprojects the history buffers instead of
  // discarding the previous accumulation.
  SbBool ptReprojectFrame = FALSE;
  uint32_t ptBufferWidth = 0;
  uint32_t ptBufferHeight = 0;

  // Path tracing state (see the public setters for the semantics).
  SbBool ptEnabled = FALSE;
  SbBool ptStartLatch = FALSE;
  SbBool ptAccumulating = FALSE;
  //! True once the accumulation reached ptMaxSamples (or the adaptive stop)
  //! and the run went idle.  Distinguishes a converged image (G-buffers still
  //! valid, denoised result should be kept and presented) from a camera/move
  //! preview frame (G-buffers stale, denoised result must be dropped).
  SbBool ptConverged = FALSE;
  uint32_t ptFrameIndex = 0;
  // Ordinal of the last frame this backend rendered (copied from
  // SoRenderParams::frame in render()/renderExternal()).  Emitted in the
  // [RTDBG] adaptive ptState/blas lines so probes can correlate backend
  // traces to phase markers and frame dumps on one monotonic key.
  uint32_t ptLastFrame = 0;
  uint32_t ptMaxBounces = 4;
  // Consecutive frames with an unchanged camera/scene while not accumulating.
  // Once this reaches the settle threshold (see updatePathTracingState) a
  // fresh accumulation auto-starts, so the view refines itself after a move
  // without an explicit startPathTracing() call.
  uint32_t ptIdleFrames = 0;
  // Frames of a static camera before an auto-restart (FC_VULKAN_PT_SETTLE).
  uint32_t ptSettleFrames = 6;
  // Accumulated samples at which the run auto-stops (FC_VULKAN_PT_MAXSAMPLES):
  // the image is converged, so the viewport can go idle instead of tracing
  // forever.  A camera move resets back to the preview/restart cycle.
  uint32_t ptMaxSamples = 256;
  // Whether the edge-stopping denoise present pass is active.
  SbBool ptDenoise = TRUE;
  // Adaptive sampling state (see updateAdaptiveStats()).
  SbBool ptAdaptiveEnabled = TRUE;
  uint32_t ptAdaptiveMinSamples = 4;
  float ptAdaptiveThreshold = 0.05f;
  // Stop adaptively once the residual active-pixel fraction falls below this.
  // The observed plateau of firefly/highlight pixels on a typical scene sits
  // around 5-7% and never drops to the old 0.02, so the run only ever stopped
  // at the hard sample cap, leaving those pixels noisy.  Raising it slightly
  // above the plateau hands the last residual to the guide-based denoiser
  // instead of tracing to the cap.  FC_VULKAN_PT_STOP_FRACTION overrides.
  float ptAdaptiveStopFraction = 0.05f;
  // Firefly rejection: standard-deviation multiplier under which a sample
  // far brighter than the pixel's running mean is replaced by that mean.
  // 0 disables it.  Default 5.0 clamps extreme outliers so the stubborn
  // firefly pixels' variance genuinely drops (letting adaptive sampling
  // converge) instead of staying pinned above the threshold.  Set from
  // FC_VULKAN_PT_FIREFLY (0 = off) to override per run.
  float ptFireflySigma = 5.0f;
  uint32_t ptLastActivePixels = 0;
  float ptLastActiveFraction = 1.0f;
  //! Denoise-at-target latch.  Set when the accumulation reaches the target
  //! sample count (ptMaxSamples) so the denoiser runs exactly once on the
  //! final accumulated image instead of re-denoising a changing partial every
  //! frame (the "keeps getting grainy" churn).  The target frame keeps
  //! ptAccumulating set so the G-buffer readback is recorded; updateDenoise
  //! consumes the latch and then drops to converged-idle.
  SbBool ptDenoisePending = FALSE;
  // Temporal reprojection (FC_VULKAN_PT_TEMPORAL); the world->clip matrix
  // of the previous frame's camera for the history reprojection test.
  SbBool ptTemporalEnabled = TRUE;
  float prevViewProj[16] = {};
  void updateAdaptiveStats();
  void swapPathTracingHistory();
  float lastViewMatrix[16] = {};
  float lastProjMatrix[16] = {};
  uint32_t lastViewportWidth = 0;
  uint32_t lastViewportHeight = 0;
  SbBool haveLastView = FALSE;
  //! Generator counter of the camera that produced the last processed frame
  //! (SoRenderParams::cameraVersion).  When nonzero it is the authoritative
  //! camera-motion signal: the manager bumps it on any camera move, so the
  //! backend never has to rely on a floating-point matrix diff.
  uint32_t lastCameraVersion = 0;
  SbBool haveLastCameraVersion = FALSE;
  //! True while a camera-only move is being reprojected (viewChanged with
  //! temporal history).  On the frame the move concludes, the accumulation is
  //! restarted cleanly: the per-pixel history carried across the unsettled
  //! move is mismatch-prone (disocclusions, surfaces that only became visible
  //! mid-orbit, samples seeded at a stale jitter), and building more samples
  //! on it locks the adaptive sampler at mutually-inconsistent partial values.
  SbBool ptWasMoving = FALSE;

  // TLAS (rebuilt every frame) and scratch buffer (sized for the largest
  // build, reused by BLAS and TLAS builds).
  VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
  VkBuffer tlasBuffer = VK_NULL_HANDLE;
  VkDeviceMemory tlasMemory = VK_NULL_HANDLE;
  VkDeviceSize tlasSize = 0;
  VkBuffer instanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory instanceMemory = VK_NULL_HANDLE;
  uint32_t instanceCount = 0;
  uint32_t instanceBufferCapacity = 0;
  // Reusable per-frame instance collection (grown on demand) instead of a
  // fresh heap allocation inside buildTlas() every frame.
  std::vector<VkAccelerationStructureInstanceKHR> instanceScratch;
  VkBuffer scratchBuffer = VK_NULL_HANDLE;
  VkDeviceMemory scratchMemory = VK_NULL_HANDLE;
  VkDeviceSize scratchSize = 0;
  VkDeviceAddress scratchAddress = 0;
  VkDeviceSize asScratchAlignment = 128;

  // Host-visible material record storage (set 0, binding 3).
  VkBuffer materialBuffer = VK_NULL_HANDLE;
  VkDeviceMemory materialMemory = VK_NULL_HANDLE;
  VkDeviceSize materialBufferBytes = 0;
  void * materialMapped = nullptr;
  uint32_t materialCount = 0;

  // Host-visible frame UBO (set 0, binding 2).
  VkBuffer frameBuffer = VK_NULL_HANDLE;
  VkDeviceMemory frameMemory = VK_NULL_HANDLE;
  void * frameMapped = nullptr;

  // Host-visible present frame UBO (set 0, binding 6 of the present set):
  // the traced camera's world->view and view->clip matrices, used by the
  // present pass to write scene depth for the raster composite edge overlay.
  VkBuffer presentFrameBuffer = VK_NULL_HANDLE;
  VkDeviceMemory presentFrameMemory = VK_NULL_HANDLE;
  void * presentFrameMapped = nullptr;

  // Reusable scratch for updateMaterials(), grown on demand instead of
  // allocating a fresh std::vector<RTMaterial> every frame.
  std::vector<RTMaterial> materialScratch;
  // Cached PBR/lighting env overrides.  These are loop-invariant per frame;
  // reading them once avoids a getenv()/envFlagEnabled() per command.
  bool rtPbrEnabled = false;
  bool rtNeeEnabled = false;
  bool rtMisEnabled = false;
  bool rtMetalOverride = false;
  bool rtRoughOverride = false;
  float rtMetalValue = 0.0f;
  float rtRoughValue = 0.0f;

  // Object-space per-triangle geometric normals (set 0, binding 7).  Grow-
  // only pool appended by buildBlas(); per-command offsets are carried in
  // the RTMaterial records.
  VkBuffer normalPoolBuffer = VK_NULL_HANDLE;
  VkDeviceMemory normalPoolMemory = VK_NULL_HANDLE;
  void * normalPoolMapped = nullptr;
  VkDeviceSize normalPoolCapacity = 0;
  VkDeviceSize normalPoolUsed = 0;
  bool ensureNormalPoolCapacity(VkDeviceSize bytes);
  VkDeviceSize appendTriangleNormals(const SoRenderCommand & command,
                                     RTXCachedGeometry & entry);

  // Emissive-triangle pool for NEE (set 0, binding 13).  Rebuilt per frame
  // (buildNeePool) so baked transforms stay fresh without a BLAS rebuild;
  // per-command offsets ride in RTMaterial::triangleData (z/w).
  VkBuffer neePoolBuffer = VK_NULL_HANDLE;
  VkDeviceMemory neePoolMemory = VK_NULL_HANDLE;
  void * neePoolMapped = nullptr;
  VkDeviceSize neePoolCapacity = 0;
  VkDeviceSize neePoolUsed = 0;
  uint32_t neePoolCount = 0;
  bool ensureNeePoolCapacity(VkDeviceSize bytes);
  void buildNeePool(const SoDrawList & drawlist);

  // --- Cache bookkeeping ---------------------------------------------------
  std::vector<RTXCachedGeometry> geometryCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  //! Set by updateGeometryCache() when any cached command's geometry
  //! identity changed and a BLAS rebuild is pending this frame.
  bool cacheChanged = false;

  // Monotonic frame counter used to stamp visited cache entries; entries
  // not stamped by the end of updateGeometryCache() are evicted.
  uint32_t cacheFrame = 0;
  void deferDestroyCacheEntry(RTXCachedGeometry & entry);

  // Per-frame BLAS build statistics (reset each frame; observable via the
  // [RTDBG] breadcrumb when FC_VULKAN_RT_DEBUG is set).
  uint32_t statBlasBuilt = 0;
  uint32_t statBlasRefit = 0;
  uint32_t statBlasReused = 0;

  // Staging buffers destroyed by buildBlas() are released only after the
  // owning command buffer finished executing (destroying a bound buffer
  // while the command buffer is recording or pending invalidates it).
  std::vector<std::pair<VkBuffer, VkDeviceMemory>> pendingStagingDestroys;
  void freePendingStagingDestroys();

  // Resources replaced during a frame (resized storage image, grown
  // instance/material buffers, recreated present pipeline, ...) are
  // destroyed two frames later, once the caller's pending submissions that
  // may still reference them have certainly completed.
  std::vector<std::function<void()>> pendingDestroys[2];
  int pendingDestroyIndex = 0;
  void flushPendingDestroys();
  void deferDestroy(std::function<void()> && fn);

  // --- Denoiser backends (OIDN / RTX-OptiX / FSR) -------------------------
  // The path tracer writes G-buffers (accumulated radiance, albedo, world
  // normal, world position) that the present pass either edge-stops in
  // shader or feeds through a machine-learned denoiser.  This block owns
  // the external denoiser: the backend selector, staging buffers to move
  // the G-buffers to/from the denoiser, and the denoised output bound at
  // present binding 5.
  enum DenoiseKind { DenoiseNone = 0, DenoiseOidn, DenoiseRtx, DenoiseFsr };

  //! Current ray-traced view mode (see the public RtxViewMode enum).
  RtxViewMode rtxViewMode = RtxViewMode::RtxModeOff;

  //! Procedural IBL / environment-lit params (mirrored into the frame UBO's
  //! u_env / u_envColor).  Used by RtxModeEnvironment to shade with a
  //! procedural sky (vertical gradient + sun disk) instead of constant
  //! per-material ambient, and to write a matching environment radiance for
  //! the primary-ray miss so polished/specular surfaces pick up the sky.
  float envIntensity = 0.35f;
  float envSunDir[3] = {0.35f, 0.8f, 0.25f};
  float envSunColor[3] = {1.0f, 0.94f, 0.82f};
  float envSunPower = 12.0f;
  float envSkyBrightness = 1.0f;

  //! Backend selected (resolved) by FC_VULKAN_PT_DENOISER or, when set via
  //! setDenoiserFilter(), by the stored preference name so a runtime choice
  //! survives the next path-tracing buffer (re)creation.
  DenoiseKind denoiseKind = DenoiseNone;
  //! The denoiser the user asked for (via setDenoiserFilter() or the
  //! FC_VULKAN_PT_DENOISER env var).  Unlike denoiseKind, this survives the
  //! teardown performed by destroyDenoiser() on a viewport resize, so the
  //! recreation resolves the SAME backend instead of silently falling back to
  //! the OIDN CPU default.  DenoiseNone = "not specified", resolve from env.
  DenoiseKind denoiseKindPref = DenoiseNone;
  //! True once a denoiser backend has been successfully created.
  bool denoiserActive = false;
  //! Set by setDenoiserFilter() so createDenoiseBackend() re-resolves the
  //! kind from the stored preference (instead of the env var) on the next
  //! buffer (re)creation.
  bool denoiseKindDirty = false;

  // Host-visible staging copies of the G-buffers (device-local accum/normal/
  // albedo are copied in after the trace, denoised on the host (OIDN) or on
  // the GPU (RTX/CUDA), then written out).  Only allocated while a denoiser
  // is active; freed on resize/shutdown.
  VkBuffer denoiseColorBuf = VK_NULL_HANDLE;      //!< accum average (rgb)
  VkDeviceMemory denoiseColorMem = VK_NULL_HANDLE;
  VkBuffer denoiseAlbedoBuf = VK_NULL_HANDLE;     //!< albedo guide
  VkDeviceMemory denoiseAlbedoMem = VK_NULL_HANDLE;
  VkBuffer denoiseNormalBuf = VK_NULL_HANDLE;     //!< normal guide
  VkDeviceMemory denoiseNormalMem = VK_NULL_HANDLE;
  VkBuffer denoiseGuideBuf = VK_NULL_HANDLE;      //!< [validity mask, ...]
  VkDeviceMemory denoiseGuideMem = VK_NULL_HANDLE;
  VkBuffer denoiseOutBuf = VK_NULL_HANDLE;        //!< denoiser output (rgba)
  VkDeviceMemory denoiseOutMem = VK_NULL_HANDLE;
  void * denoiseStagingPtr = nullptr;             //!< maps the host buffers
  uint32_t denoiseWidth = 0;
  uint32_t denoiseHeight = 0;
  // Resolution at which the denoiser staging/output were last created (so a
  // viewport resize tears them down and recreates at the new dimensions).
  uint32_t denoiseStagedWidth = 0;
  uint32_t denoiseStagedHeight = 0;

  // Device-local denoised result bound at present binding 5 and sampled by
  // the present shader when the denoiser has produced a result for the
  // current frame.
  VkBuffer denoisedBuffer = VK_NULL_HANDLE;
  VkDeviceMemory denoisedMemory = VK_NULL_HANDLE;
  // Albedo G-buffer (binding 14) written by the raygen; the denoiser uses it
  // as a guide, and it is read back with the other G-buffers.
  VkBuffer albedoBuffer = VK_NULL_HANDLE;
  VkDeviceMemory albedoMemory = VK_NULL_HANDLE;
  //! When true the present pass samples denoisedBuffer instead of doing the
  //! in-shader edge-stopping filter.
  SbBool denoiseResultReady = FALSE;
  //! Scale factor baked into the present shader's denoise branch (1.0 =
  //! native resolution; >1 when the denoiser runs at reduced resolution).
  float denoiseScale = 1.0f;
  //! Minimum accumulated samples before the denoiser output is published.
  //! The denoiser is trained on partially converged images; feeding it a
  //! one- or two-sample frame produces a blurred/junk result (NVIDIA's
  //! vk_optix_denoise example only starts denoising after a start frame).
  //! When the accumulation is below this the present falls back to the raw
  //! / in-shader edge-stopped view.  0 disables the gate.
  uint32_t denoiseMinSamples = 8;

  // Per-frame gating: the readback is recorded only after an accumulating
  // frame (needs fresh G-buffers) and the denoise itself runs after the
  // submission's queue wait (host/GPU work cannot be recorded).
  SbBool oidnReadbackPending = FALSE;

  // --- OIDN backend -------------------------------------------------------
#if COIN_BUILD_OIDN
  OIDNDevice oidnDevice = nullptr;
  OIDNFilter oidnFilter = nullptr;
  void setupOidnDevice();
  bool configureOidnFilter();
#endif

  // Async OIDN execution.  OIDN's oidnExecuteFilter() on the CPU device can
  // take tens of milliseconds and, because the render (and denoise) run on the
  // GUI thread during startNextFrame(), blocks the UI for its whole duration.
  // To keep the viewport responsive the host-side OIDN work (normalize the
  // accum average, execute the filter, stamp the validity mask) runs on a
  // dedicated worker thread that owns the OIDN filter for its lifetime; the
  // Vulkan copy-back of the published result stays on the render thread once
  // the worker signals completion.  The staging block is HOST_VISIBLE|
  // HOST_COHERENT, so the worker's writes are visible to the copy-back without
  // an explicit flush, and the filter is only used by the worker while the
  // render thread is not submitting to it (guarded by the running latch).
  // The single staging well is safe to reuse because the denoiser only fires
  // once per accumulation run (denoise-at-target): after the target is reached
  // the backend is converged-idle, so no readback overwrites the staging until
  // the next camera/scene move, which cannot happen concurrently with the
  // at-target denoise in the same run.
  std::atomic<bool> oidnWorkerRunning {false};
  //! Set by the worker after it published the denoised result in the staging
  //! output region; the render thread copies it device-ward and converges.
  std::atomic<bool> oidnWorkerDone {false};
  //! The worker joinable handle (one shot per denoise-at-target).
  std::thread oidnWorker;


  // --- RTX (OptiX + CUDA) backend -----------------------------------------
#if COIN_BUILD_RTX_DENOISER
  OptixDeviceContext rtxDeviceContext = nullptr;
  OptixDenoiser rtxDenoiser = nullptr;
  OptixDenoiserModelKind rtxModelKind = OPTIX_DENOISER_MODEL_KIND_AOV;
  OptixDenoiserSizes rtxSizes {};
  CUcontext rtxCudaCtx = nullptr;
  CUstream rtxStream = nullptr;
  CUdeviceptr rtxScratch = 0;
  CUdeviceptr rtxState = 0;
  CUdeviceptr rtxIntensity = 0;
  CUdeviceptr rtxColorImage = 0;
  CUdeviceptr rtxAlbedoImage = 0;
  CUdeviceptr rtxNormalImage = 0;
  CUdeviceptr rtxOutputImage = 0;
  uint64_t rtxImagePitch = 0;
  size_t rtxScratchBytes = 0;
  size_t rtxStateBytes = 0;
  size_t rtxOutputPitchBytes = 0;
  int rtxCudaDevice = 0;
  bool rtxCudaInitFailed = false;
  // Interop: the denoiser working images are dedicated device-local Vulkan
  // buffers whose memory is exported (opaque FD) and imported into CUDA so
  // OptiX reads/writes them directly.  No host round-trip.  The Vulkan
  // handles own the allocation; the CUexternalMemory + mapped CUdeviceptr
  // alias it and must be released (cuDestroyExternalMemory) before the
  // VkDeviceMemory is freed.
  VkBuffer rtxColorVk = VK_NULL_HANDLE;
  VkBuffer rtxAlbedoVk = VK_NULL_HANDLE;
  VkBuffer rtxNormalVk = VK_NULL_HANDLE;
  VkBuffer rtxOutputVk = VK_NULL_HANDLE;
  VkDeviceMemory rtxColorMem = VK_NULL_HANDLE;
  VkDeviceMemory rtxAlbedoMem = VK_NULL_HANDLE;
  VkDeviceMemory rtxNormalMem = VK_NULL_HANDLE;
  VkDeviceMemory rtxOutputMem = VK_NULL_HANDLE;
  CUexternalMemory rtxColorExt = nullptr;
  CUexternalMemory rtxAlbedoExt = nullptr;
  CUexternalMemory rtxNormalExt = nullptr;
  CUexternalMemory rtxOutputExt = nullptr;
  // The CUDA kernel that converts the accumulated radiance sum (rgb) /
  // sample-count (w) into the average color the denoiser expects.
  CUmodule rtxNormModule = nullptr;
  CUfunction rtxNormKernel = nullptr;
  bool rtxInteropReady = false;
  // vkGetMemoryFdKHR resolved per-device (needed to export the FD).
  PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
#endif

  //! Create the per-backend denoiser resources (called by createPathTracingBuffers).
  bool createDenoiseBackend();
  //! Record the device->host readback of the G-buffers on \a cmd.
  void recordDenoiseReadback(VkCommandBuffer cmd);
  //! Run the denoiser on the staged G-buffers and publish denoiseResultReady.
  void updateDenoise();
  //! After publishing a denoised result at target: clear the denoise latch and
  //! transition to converged-idle so the viewport keeps the denoised image and
  //! stops the continuous-update loop.  Also used on the failure paths (with a
  //! published raw result) so the run does not retry a failed denoise forever.
  void convergeAfterDenoise();
  //! Destroy denoiser resources (device + staging + RTX/OIDN handles).
  void destroyDenoiser();
  void releaseDenoiseStaging();

#if COIN_BUILD_RTX_DENOISER
  //! Initialize the private CUDA context + OptiX device context.
  bool initRtxCuda();
  //! Create the OptiX denoiser, its scratch/state/intensity buffers, and the
  //! CUDA-Vulkan interop working images.
  bool initRtxDenoiser();
  //! Allocate one exportable device-local buffer and import it into CUDA;
  //! returns the mapped CUdeviceptr in \a devPtr and owns handle state in the
  //! ext/mem/buffer out-params.
  bool createRtxInteropBuffer(size_t bytes, VkBufferUsageFlags usage,
                              VkBuffer & buffer, VkDeviceMemory & memory,
                              CUexternalMemory & ext, CUdeviceptr & devPtr);
  //! Run one OptiX denoiser pass over the interop color/albedo/normal images.
  bool updateRtxDenoise(uint32_t w, uint32_t h);
  //! Free the CUDA context + OptiX denoiser state (keeps Vulkan staging).
  void teardownRtxDenoiser();
#endif
};

#endif // COIN_SORTXRENDERBACKEND_H
