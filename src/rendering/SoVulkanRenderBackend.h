// src/rendering/SoVulkanRenderBackend.h

#ifndef COIN_SOVULKANRENDERBACKEND_H
#define COIN_SOVULKANRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <cstddef>
#include <cstdint>
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

private:
  // --- Initialization helpers -------------------------------------------
  bool createCommandPool();
  bool createDescriptorSetLayout();
  bool createPipelineLayout();
  bool createRenderPass(const SoVulkanRenderTarget & target,
                        VkRenderPass & renderPass);
  bool createShaders(VkShaderModule & vertexModule,
                     VkShaderModule & fragmentModule);
  bool getOrCreatePipeline(const SoRenderCommand & command,
                           const SoVulkanRenderTarget & target,
                           VkRenderPass renderPass,
                           VkPipeline & pipeline,
                           bool transparent);

  // --- Geometry cache ---------------------------------------------------
  void invalidateCache();
  void updateGeometryCache(const SoDrawList & drawlist);
  VulkanCachedCommand & getOrCreateCache(const SoRenderCommand * command);
  void uploadGeometry(VulkanCachedCommand & entry,
                      const SoRenderCommand & command);
  void destroyCacheEntry(VulkanCachedCommand & entry);

  // --- Render recording ---------------------------------------------------
  bool beginCommandBuffer();
  void recordClear(const SoRenderParams & params,
                   const SoVulkanRenderTarget & target);
  void recordDrawCommand(const SoDrawList & drawlist,
                         const SoRenderCommand & command,
                         const SoVulkanRenderTarget & target,
                         VkRenderPass renderPass,
                         bool transparent);
  bool endAndSubmit();
  void applyViewport(const SoRenderParams & params,
                     const SoVulkanRenderTarget & target);

  // --- Vulkan resource helpers -------------------------------------------
  bool createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkBuffer & buffer,
                    VkDeviceMemory & memory,
                    const void * data);

  // --- Owned device ------------------------------------------------------
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;
  const VkAllocationCallbacks * allocator = nullptr;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  VkShaderModule vertexModule = VK_NULL_HANDLE;
  VkShaderModule fragmentModule = VK_NULL_HANDLE;

  // Render pass is owned per target identity (image + extent).
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkImage renderPassColorImage = VK_NULL_HANDLE;
  VkImageView renderPassColorView = VK_NULL_HANDLE;
  VkImage renderPassDepthImage = VK_NULL_HANDLE;
  VkImageView renderPassDepthView = VK_NULL_HANDLE;
  VkExtent2D renderPassExtent {0, 0};

  // Pipeline cache: keyed by (pipelineKey, transparent).  In the unlit
  // milestone all opaque commands share one pipeline and all transparent
  // commands share another, but the keying is retained for the upcoming
  // per-material/lighting pipeline specialization.
  std::unordered_map<uint64_t, VkPipeline> pipelineCache;

  std::vector<VulkanCachedCommand> gpuCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  uint32_t cacheGeneration = 0;
  size_t cachedCommandCount = 0;
  bool haveCacheGeneration = false;

  uint32_t currentFrame = 0;
};

#endif // COIN_SOVULKANRENDERBACKEND_H
