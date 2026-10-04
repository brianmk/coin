// src/rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.h
//
// Pipeline-cache state and lifetime for the Vulkan raster backend: two
// key -> VkPipeline maps (visual + background gradient) plus the persistent
// VkPipelineCache.  A pure store -- the backend resolves keys through it.

#ifndef COIN_SOVULKANPIPELINECACHE_H
#define COIN_SOVULKANPIPELINECACHE_H

#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "rendering/SoVulkanPlatform.h"
#include <vulkan/vulkan.h>

// Shared combine step for the hand-rolled hash functors (keeps == and hash in sync).
static inline size_t vkPipelineHashCombine(size_t hash, size_t value)
{
  return hash ^ (value + 0x9e3779b9 + (hash << 6) + (hash >> 2));
}

// Fold a key's fields() tuple into a hash.  Deriving both the hash and
// operator== from the same list keeps them consistent by construction.
template <class Tuple>
static inline size_t vkPipelineHashFields(const Tuple & fields)
{
  return std::apply([](const auto &... field) {
    size_t hash = 0;
    ((hash = vkPipelineHashCombine(
        hash, std::hash<std::decay_t<decltype(field)>>{}(field))), ...);
    return hash;
  }, fields);
}

/*!
  \brief Immutable graphics-pipeline identity.

  Defined before VulkanCachedCommand so each cached command can remember the
  key it last resolved to, skipping re-hash/map lookup for an unchanged command.
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
  //! GPU-instanced wide-line variant: same output, but the vertex shader
  //! expands the segment on the GPU from an instance-rate endpoint buffer.
  bool wideLineInstanced = false;

  //! Single source of truth for pipeline identity: operator== and the hash both
  //! derive from this list, so they cannot drift apart.
  auto fields() const
  {
    return std::tie(renderPass, topology, fillMode, cullMode, ccwFrontFace,
                    depthTestEnable, depthWriteEnable, depthFunction,
                    depthBiasEnable, depthBiasConstantFactor,
                    depthBiasSlopeFactor, blendEnable, blendSrcRGB, blendDstRGB,
                    blendSrcAlpha, blendDstAlpha, blendEquationRGB,
                    blendEquationAlpha, stencilEnable, stencilFunction,
                    stencilReference, stencilCompareMask, stencilWriteMask,
                    stencilFailOp, stencilZFailOp, stencilZPassOp, sampleCount,
                    wideLine, wideLineInstanced);
  }

  bool operator==(const PipelineKey & other) const
  {
    return this->fields() == other.fields();
  }
};

struct PipelineKeyHash
{
  size_t operator()(const PipelineKey & key) const
  {
    return vkPipelineHashFields(key.fields());
  }
};

// Background gradient pipeline cache, keyed on render pass + sample count only.
struct BackgroundPipelineKey {
  VkRenderPass renderPass = VK_NULL_HANDLE;
  uint32_t sampleCount = 1;

  auto fields() const { return std::tie(renderPass, sampleCount); }

  bool operator==(const BackgroundPipelineKey & other) const
  {
    return this->fields() == other.fields();
  }
};

struct BackgroundPipelineKeyHash
{
  size_t operator()(const BackgroundPipelineKey & key) const
  {
    return vkPipelineHashFields(key.fields());
  }
};

/*!
  \brief Key -> VkPipeline store plus the persistent VkPipelineCache handle.

  find() resolves a key; on a miss the backend creates the pipeline and stores
  it (a failure is stored as VK_NULL_HANDLE, remembered and warned once).
  handle() feeds vkCreateGraphicsPipelines/vkCreateComputePipelines.
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

  //! Content key of the compiled shaders.  Persisted and checked on load: a
  //! file from different shaders is rejected (the state key lacks shader code).
  void setShaderKey(uint64_t key) { this->shaderKey = key; }

  //! Route informational messages (supplied/saved/rejected) to the log callback.
  void setLogger(std::function<void(const char *)> logger)
  {
    this->logger = std::move(logger);
  }

  //! Create the VkPipelineCache, seeding it from the path when one is set.
  bool initialize();

  //! Persist (when a path is set) and destroy all pipelines + the handle.  Idempotent.
  void shutdown();

  VkPipelineCache handle() const { return this->cache; }

  //! Resolve \a key; true when present (out may be VK_NULL_HANDLE for a failure).
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
  // Read the persistent cache file; false when empty/missing/unreadable/oversized.
  bool readFile(std::vector<uint8_t> & data) const;
  // Write the cache to path (sibling temp + rename); no-op if path empty or null.
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
