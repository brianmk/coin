// src/rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.cpp
//
// See the header for the design.

#include "rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.h"

#include "rendering/SoVulkanDebugUtils.h"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace {

// File header: [magic][shaderKey][blobSize] + driver blob.  magic distinguishes
// our wrapper from a bare blob; shaderKey rejects a blob built from older shaders.
constexpr uint64_t kPipelineCacheMagic = 0x434f494e50495045ull; // "COINPIPE"
constexpr size_t kPipelineCacheHeaderSize = sizeof(uint64_t) * 3;

} // namespace

void
SoVulkanPipelineCache::emit(const char * message) const
{
  if (this->logger) {
    this->logger(message);
  }
}

bool
SoVulkanPipelineCache::initialize()
{
  // Pipelines compile lazily on the draw path, so persist a cache to avoid
  // recompiling them on every launch.  The blob is opaque driver data keyed by
  // the device's pipelineCacheUUID; a stale or foreign blob is ignored by the
  // driver, and if creating the cache with it fails we retry empty rather than
  // failing init.
  std::vector<uint8_t> initialData;
  const bool haveInitialData = this->readFile(initialData);
  VkPipelineCacheCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  ci.initialDataSize = haveInitialData ? initialData.size() : 0;
  ci.pInitialData = haveInitialData ? initialData.data() : nullptr;
  VkResult result = vkCreatePipelineCache(this->device, &ci, this->allocator,
                                          &this->cache);
  if (result != VK_SUCCESS && haveInitialData) {
    // Unreadable as a cache (corrupt or foreign device); start empty, don't fail.
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "pipeline cache: rejected %s; starting with an empty cache",
                  this->path.c_str());
    this->emit(msg);
    ci.initialDataSize = 0;
    ci.pInitialData = nullptr;
    result = vkCreatePipelineCache(this->device, &ci, this->allocator,
                                   &this->cache);
  }
  else if (result == VK_SUCCESS && haveInitialData) {
    // "supplied" not "loaded": the impl may ignore unusable data (stale
    // pipelineCacheUUID), so acceptance does not guarantee reuse.
    char msg[192];
    std::snprintf(msg, sizeof(msg), "pipeline cache: supplied %zu bytes from %s",
                  initialData.size(), this->path.c_str());
    this->emit(msg);
  }
  if (result == VK_SUCCESS) {
    SoVulkanDebugUtils::nameObject(this->device, VK_OBJECT_TYPE_PIPELINE_CACHE,
                                   reinterpret_cast<uint64_t>(this->cache),
                                   "Coin raster pipeline cache");
  }
  return result == VK_SUCCESS;
}

void
SoVulkanPipelineCache::shutdown()
{
  for (auto & entry : this->pipelines) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->pipelines.clear();
  for (auto & entry : this->backgroundPipelines) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->backgroundPipelines.clear();
  if (this->cache != VK_NULL_HANDLE) {
    // Persist before the handle dies so lazily-built variants survive a restart.
    this->writeFile();
    vkDestroyPipelineCache(this->device, this->cache, this->allocator);
    this->cache = VK_NULL_HANDLE;
  }
}

bool
SoVulkanPipelineCache::readFile(std::vector<uint8_t> & data) const
{
  if (this->path.empty()) return false;
  std::ifstream in(this->path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  const std::streamoff size = in.tellg();
  if (size <= 0) return false;
  // Bound the read: never slurp a corrupt/foreign file wholesale at device init.
  if (size > static_cast<std::streamoff>(64u * 1024u * 1024u)) return false;
  std::vector<uint8_t> raw(static_cast<size_t>(size));
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(raw.data()),
          static_cast<std::streamsize>(raw.size()));
  if (!(in.good() || in.eof())) return false;

  // Validate the wrapper; mismatch (different shaders, bare blob, truncation)
  // means unusable, so return false and let initialize() start empty.
  if (raw.size() < kPipelineCacheHeaderSize) return false;
  uint64_t magic = 0;
  uint64_t key = 0;
  uint64_t blobSize = 0;
  std::memcpy(&magic, raw.data(), sizeof(magic));
  std::memcpy(&key, raw.data() + sizeof(uint64_t), sizeof(key));
  std::memcpy(&blobSize, raw.data() + sizeof(uint64_t) * 2, sizeof(blobSize));
  if (magic != kPipelineCacheMagic || key != this->shaderKey ||
      blobSize != raw.size() - kPipelineCacheHeaderSize) {
    return false;
  }
  data.assign(raw.begin() + kPipelineCacheHeaderSize, raw.end());
  return true;
}

void
SoVulkanPipelineCache::writeFile() const
{
  if (this->path.empty() || this->cache == VK_NULL_HANDLE) {
    return;
  }
  size_t size = 0;
  if (vkGetPipelineCacheData(this->device, this->cache, &size, nullptr) !=
        VK_SUCCESS || size == 0) {
    return;
  }
  std::vector<uint8_t> blob(size);
  if (vkGetPipelineCacheData(this->device, this->cache, &size, blob.data()) !=
      VK_SUCCESS) {
    return;
  }
  blob.resize(size);

  // Wrap the driver blob with the header readFile() validates.
  std::vector<uint8_t> data;
  data.reserve(kPipelineCacheHeaderSize + blob.size());
  const uint64_t magic = kPipelineCacheMagic;
  const uint64_t key = this->shaderKey;
  const uint64_t blobSize = blob.size();
  const auto append = [&data](const void * p, const size_t n) {
    const auto * bytes = static_cast<const uint8_t *>(p);
    data.insert(data.end(), bytes, bytes + n);
  };
  append(&magic, sizeof(magic));
  append(&key, sizeof(key));
  append(&blobSize, sizeof(blobSize));
  append(blob.data(), blob.size());

  // Write a sibling temp then swap in.  std::rename does not replace on Windows,
  // so remove the target first; losing an advisory cache to a crash is harmless.
  const std::string tmpPath = this->path + ".tmp";
  {
    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out) return;
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
    if (!out) return;
  }
  std::remove(this->path.c_str());
  std::rename(tmpPath.c_str(), this->path.c_str());
  char msg[192];
  std::snprintf(msg, sizeof(msg), "pipeline cache: saved %zu bytes to %s",
                data.size(), this->path.c_str());
  this->emit(msg);
}
