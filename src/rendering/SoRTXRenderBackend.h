// src/rendering/SoRTXRenderBackend.h

#ifndef COIN_SORTXRENDERBACKEND_H
#define COIN_SORTXRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

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

  // Identity keys mirroring the producer-owned storage of the last build.
  // The storage is a per-frame arena (SoIRRenderAction::geometryPool), so
  // pointers are NOT stable across frames; the content hash below is the
  // real identity signal.
  const float * posKey = nullptr;
  const uint32_t * idxKey = nullptr;
  uint32_t vertexStride = 0;
  uint32_t cacheGeneration = 0;
  uint64_t contentHash = 0;
  // Offset of this command's triangle normals in the normal pool
  // (UINT32_MAX = not uploaded yet).
  uint32_t normalPoolOffset = 0xFFFFFFFFu;
  uint32_t normalCount = 0;
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
class SoRTXRenderBackend : public SoRenderBackend {
public:
  SoRTXRenderBackend();
  ~SoRTXRenderBackend() override;

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

  //! True while progressive accumulation is running (start flag consumed).
  SbBool getPathTracingActive(void) const;

  //! Samples accumulated in the current progressive run (0 when idle).
  uint32_t getPathTracingSampleCount(void) const;

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
  VkCommandBuffer beginTransientCommandBuffer(VkCommandPool & pool);

  // --- Device handles ----------------------------------------------------
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  const VkAllocationCallbacks * allocator = nullptr;

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
  uint32_t ptBufferWidth = 0;
  uint32_t ptBufferHeight = 0;

  // Path tracing state (see the public setters for the semantics).
  SbBool ptEnabled = FALSE;
  SbBool ptStartLatch = FALSE;
  SbBool ptAccumulating = FALSE;
  uint32_t ptFrameIndex = 0;
  uint32_t ptMaxBounces = 4;
  float lastViewMatrix[16] = {};
  float lastProjMatrix[16] = {};
  uint32_t lastViewportWidth = 0;
  uint32_t lastViewportHeight = 0;
  SbBool haveLastView = FALSE;

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

  // --- Cache bookkeeping ---------------------------------------------------
  std::vector<RTXCachedGeometry> geometryCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  //! Set by updateGeometryCache() when any cached command's geometry
  //! identity changed and a BLAS rebuild is pending this frame.
  bool cacheChanged = false;

  // Staging buffers destroyed by buildBlas() are released only after the
  // owning command buffer finished executing (destroying a bound buffer
  // while the command buffer is recording or pending invalidates it).
  std::vector<std::pair<VkBuffer, VkDeviceMemory>> pendingStagingDestroys;
  void freePendingStagingDestroys();
};

#endif // COIN_SORTXRENDERBACKEND_H
