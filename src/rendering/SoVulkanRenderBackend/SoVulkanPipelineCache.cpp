// src/rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.cpp
//
// Pipeline-cache state and lifetime for the Vulkan raster backend.  See the
// header for the design.

#include "rendering/SoVulkanRenderBackend/SoVulkanPipelineCache.h"

#include "rendering/SoVulkanDebugUtils.h"

#include <cstdio>
#include <fstream>

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
  // Pipelines are created lazily on the draw path (the first time a state
  // combination is seen).  A persistent cache lets the driver keep the
  // compiled/reused shader-and-state blobs between those creations, so the
  // first frames of a scene transition do not stutter on pipeline builds.
  //
  // When a path is set, its bytes are the exact blob a previous run's
  // vkGetPipelineCacheData produced.  That blob carries the cache header and
  // the physical device's pipelineCacheUUID, so the implementation rejects a
  // file written for another device/driver; the retry below then creates an
  // empty cache instead of failing device initialization.
  std::vector<uint8_t> initialData;
  const bool haveInitialData = this->readFile(initialData);
  VkPipelineCacheCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  ci.initialDataSize = haveInitialData ? initialData.size() : 0;
  ci.pInitialData = haveInitialData ? initialData.data() : nullptr;
  VkResult result = vkCreatePipelineCache(this->device, &ci, this->allocator,
                                          &this->cache);
  if (result != VK_SUCCESS && haveInitialData) {
    // The file was unreadable as a pipeline cache (corrupt, or written for a
    // different device/driver).  Start empty rather than failing init.
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
    // "supplied", not "loaded": the implementation is free to ignore data it
    // cannot use (e.g. a stale pipelineCacheUUID) without failing, so the
    // bytes being accepted does not guarantee the driver reused them.
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
    // Persist the driver's blob before the handle dies, so the lazily-built
    // pipeline variants survive a process restart.
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
  // Bound the read: a corrupt or foreign file must not be slurped wholesale
  // into memory on the device-init path.
  if (size > static_cast<std::streamoff>(64u * 1024u * 1024u)) return false;
  data.resize(static_cast<size_t>(size));
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(data.size()));
  return in.good() || in.eof();
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
  std::vector<uint8_t> data(size);
  if (vkGetPipelineCacheData(this->device, this->cache, &size, data.data()) !=
      VK_SUCCESS) {
    return;
  }
  data.resize(size);
  // Write a sibling temp file, then swap it in.  std::rename does not replace
  // an existing file on Windows, so remove the target first; the cache is
  // advisory, so losing it to a crash mid-swap is harmless.
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
