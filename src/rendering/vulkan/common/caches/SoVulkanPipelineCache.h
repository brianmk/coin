// src/rendering/vulkan/common/caches/SoVulkanPipelineCache.h
//
// Pipeline-cache state and lifetime for the Vulkan raster backend.
//
// Owns the two key -> VkPipeline maps (visual + background gradient), the
// VkPipelineCache handle and its on-disk persistence.  It is deliberately a
// pure store: the backend keeps computing the PipelineKey and the
// VkGraphicsPipelineCreateInfo (which depend on the command state, the
// render target and the wide-line predicates) and asks this class to resolve
// the key to a pipeline, creating and storing it on a miss.
//
// Extracted from SoVulkanRenderBackend as the first step of the renderer
// architecture cleanup; the class was a single ~90-method object owning the
// frame pump, every cache and the CPU wide-line expander.

#ifndef COIN_SOVULKANPIPELINECACHE_H
#define COIN_SOVULKANPIPELINECACHE_H

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

// Shared combine step for the hand-rolled hash functors below.  Keeping one
// implementation prevents the == operator and the hash from drifting apart.
static inline size_t vkPipelineHashCombine(size_t hash, size_t value)
{
  return hash ^ (value + 0x9e3779b9 + (hash << 6) + (hash >> 2));
}

/*!
  \brief Immutable graphics-pipeline identity.

  Pipelines are cached keyed by this struct.  It is defined before
  VulkanCachedCommand (which stores a resolved key) so each cached command can
  remember the exact key it last resolved to, letting the backend skip
  re-hashing and the map lookup for an unchanged command.
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
  //! GPU-instanced wide-line variant: the same wide-line output, but the
  //! vertex shader expands the segment on the GPU from an instance-rate
  //! endpoint buffer instead of drawing the CPU-expanded quads.  Shares the
  //! fragment module with `wideLine` but needs a distinct pipeline (different
  //! vertex module and vertex input layout).
  bool wideLineInstanced = false;

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
      sampleCount == other.sampleCount && wideLine == other.wideLine &&
      wideLineInstanced == other.wideLineInstanced;
  }
};

struct PipelineKeyHash
{
  size_t operator()(const PipelineKey & key) const
  {
    size_t hash = std::hash<uintptr_t>()(
      reinterpret_cast<uintptr_t>(key.renderPass));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.topology));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.fillMode));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.cullMode));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.ccwFrontFace));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.depthTestEnable));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.depthWriteEnable));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.depthFunction));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.depthBiasEnable));
    hash = vkPipelineHashCombine(hash,
                                 std::hash<float>()(key.depthBiasConstantFactor));
    hash = vkPipelineHashCombine(hash,
                                 std::hash<float>()(key.depthBiasSlopeFactor));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.blendEnable));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.blendSrcRGB));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.blendDstRGB));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.blendSrcAlpha));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.blendDstAlpha));
    hash = vkPipelineHashCombine(hash,
                                 std::hash<uint32_t>()(key.blendEquationRGB));
    hash = vkPipelineHashCombine(hash,
                                 std::hash<uint32_t>()(key.blendEquationAlpha));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilEnable));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilFunction));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilReference));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilCompareMask));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilWriteMask));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilFailOp));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilZFailOp));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.stencilZPassOp));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.sampleCount));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.wideLine));
    hash = vkPipelineHashCombine(hash,
                                 std::hash<uint32_t>()(key.wideLineInstanced));
    return hash;
  }
};

// Background gradient pipeline cache: keyed on the render pass and sample
// count only (the gradient pipeline has no retained per-command state).
struct BackgroundPipelineKey {
  VkRenderPass renderPass = VK_NULL_HANDLE;
  uint32_t sampleCount = 1;
  bool operator==(const BackgroundPipelineKey & other) const
  {
    return renderPass == other.renderPass && sampleCount == other.sampleCount;
  }
};

struct BackgroundPipelineKeyHash
{
  size_t operator()(const BackgroundPipelineKey & key) const
  {
    size_t hash = std::hash<uintptr_t>()(
      reinterpret_cast<uintptr_t>(key.renderPass));
    hash = vkPipelineHashCombine(hash, std::hash<uint32_t>()(key.sampleCount));
    return hash;
  }
};

/*!
  \brief Key -> VkPipeline store plus the persistent VkPipelineCache handle.

  The backend resolves a PipelineKey to a pipeline through find(); on a miss
  it creates the pipeline and calls store() (a creation failure is stored as
  VK_NULL_HANDLE so the failure is remembered and the warning emitted once).
  The handle returned by handle() is passed to vkCreateGraphicsPipelines /
  vkCreateComputePipelines.
*/
class SoVulkanPipelineCache {
public:
  //! Bind the owning device.  Must be called before initialize().
  void setDevice(VkDevice device, const VkAllocationCallbacks * allocator)
  {
    this->device = device;
    this->allocator = allocator;
  }

  //! On-disk persistence path (empty = no persistence).
  void setPath(const std::string & path) { this->path = path; }

  //! Content key of the compiled shaders whose pipelines this cache holds.
  //! Written into the persisted file and checked on load: a file written from
  //! different shaders is rejected, so a rebuilt shader can never be served a
  //! pipeline compiled from the previous one (the pipeline-state key alone
  //! does not capture shader code).
  void setShaderKey(uint64_t key) { this->shaderKey = key; }

  //! Route this class's informational messages (cache supplied/saved/rejected)
  //! to the backend's log callback.
  void setLogger(std::function<void(const char *)> logger)
  {
    this->logger = std::move(logger);
  }

  //! Create the VkPipelineCache, seeding it from the path when one is set.
  bool initialize();

  //! Persist (when a path is set) and destroy every cached pipeline and the
  //! handle.  Idempotent.
  void shutdown();

  VkPipelineCache handle() const { return this->cache; }

  //! Resolve \a key.  Returns true when the key was present, writing the
  //! cached pipeline (possibly VK_NULL_HANDLE for a remembered failure).
  bool find(const PipelineKey & key, VkPipeline & out) const
  {
    const auto found = this->pipelines.find(key);
    if (found == this->pipelines.end()) return false;
    out = found->second;
    return true;
  }
  void store(const PipelineKey & key, VkPipeline pipeline)
  {
    this->pipelines[key] = pipeline;
  }

  bool findBackground(const BackgroundPipelineKey & key, VkPipeline & out) const
  {
    const auto found = this->backgroundPipelines.find(key);
    if (found == this->backgroundPipelines.end()) return false;
    out = found->second;
    return true;
  }
  void storeBackground(const BackgroundPipelineKey & key, VkPipeline pipeline)
  {
    this->backgroundPipelines[key] = pipeline;
  }

private:
  void emit(const char * message) const;
  // Read the persistent cache file into \a data.  False when the path is empty
  // or the file is missing/unreadable/absurdly large.
  bool readFile(std::vector<uint8_t> & data) const;
  // Write the current cache to the path (sibling temp file + rename).  No-op
  // when the path is empty or the handle is null.
  void writeFile() const;

  VkDevice device = VK_NULL_HANDLE;
  const VkAllocationCallbacks * allocator = nullptr;
  VkPipelineCache cache = VK_NULL_HANDLE;
  std::string path;
  uint64_t shaderKey = 0;
  std::function<void(const char *)> logger;
  std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelines;
  std::unordered_map<BackgroundPipelineKey, VkPipeline,
                     BackgroundPipelineKeyHash> backgroundPipelines;
};

#endif // COIN_SOVULKANPIPELINECACHE_H
