// src/rendering/SoVulkanConfig.cpp
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanDebug.h"

#include "rendering/SoVulkanShared.h"

#include <cstdio>
#include <cstdlib>

namespace SoVulkanConfig {

namespace {

// Read a non-negative float; a missing or invalid value keeps the default.
float readNonNegativeFloat(const char * name, float fallback)
{
  if (!SoVulkanShared::envSet(name)) {
    return fallback;
  }
  const float value = SoVulkanShared::envFloat(name, fallback);
  return value >= 0.0f ? value : fallback;
}

// Read a strictly positive uint32; missing/invalid keeps the default.  strtoll
// (not envInt) saturates > INT_MAX values (e.g. GEOM_LOD_MAX_INDEX) at UINT32_MAX.
uint32_t readPositiveUint(const char * name, uint32_t fallback)
{
  const char * value = SoVulkanShared::envString(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char * end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed <= 0) {
    return fallback;
  }
  if (parsed > static_cast<long long>(UINT32_MAX)) {
    return UINT32_MAX;
  }
  return static_cast<uint32_t>(parsed);
}

// Read a non-negative uint32; missing/invalid keeps the default.  Accepts 0.
uint32_t readNonNegativeUint(const char * name, uint32_t fallback)
{
  const char * value = SoVulkanShared::envString(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char * end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed < 0) {
    return fallback;
  }
  if (parsed > static_cast<long long>(UINT32_MAX)) {
    return UINT32_MAX;
  }
  return static_cast<uint32_t>(parsed);
}

Config load()
{
  Config c;

  // GPU sub-pixel geometry LOD.
  c.geometryLod.enabled =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_GEOM_LOD", true);
  c.geometryLod.always =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_GEOM_LOD_ALWAYS", false);
  c.geometryLod.stats =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_GEOM_LOD_STATS", false);
  c.geometryLod.minAreaPixels =
    readNonNegativeFloat("COIN_VULKAN_GEOM_LOD_PIXELS", 1.0f);
  c.geometryLod.maxIndices =
    readPositiveUint("COIN_VULKAN_GEOM_LOD_MAX_INDEX", 64000000u);
  c.geometryLod.minPrims =
    readNonNegativeUint("COIN_VULKAN_GEOM_LOD_MIN_PRIMS", 256u);

  // Command recording / queue concurrency.
  c.concurrency.parallelRecord =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_PARALLEL_RECORD", false);
  if (SoVulkanShared::envSet("COIN_VULKAN_RECORD_WORKERS")) {
    const int v = SoVulkanShared::envInt("COIN_VULKAN_RECORD_WORKERS", 0);
    if (v >= 1) {
      c.concurrency.recordWorkerCap = static_cast<unsigned int>(v);
    }
  }
  c.concurrency.externalSecondary =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_EXTERNAL_SECONDARY", false);

  // Raster-path options.
  c.raster.wideLineCpu =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_WLINE_CPU", false);

  // Diagnostic tooling.
  c.diagnostics.debugUtils =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_DEBUG_UTILS", false);
  c.diagnostics.debugPrintf =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_DEBUG_PRINTF", false);
  c.diagnostics.gpuTimestamps =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_GPU_TIMING", false);
  c.diagnostics.pipelineFeedback =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_PIPELINE_FEEDBACK", false);

  return c;
}

} // namespace

const Config & get()
{
  // C++11 function-local static init is thread-safe: the environment is
  // resolved once, on first use, and is immutable for the process lifetime.
  static const Config config = load();
  return config;
}

void dump()
{
  const Config & c = get();
  SoVulkanDebug::post("[VKCONFIG] geomLod enabled=%d always=%d stats=%d "
               "pixels=%.3f maxIndex=%u minPrims=%u\n",
               c.geometryLod.enabled ? 1 : 0,
               c.geometryLod.always ? 1 : 0,
               c.geometryLod.stats ? 1 : 0,
               static_cast<double>(c.geometryLod.minAreaPixels),
               c.geometryLod.maxIndices,
               c.geometryLod.minPrims);
  char sWorkerCap[16];
  if (c.concurrency.recordWorkerCap) {
    std::snprintf(sWorkerCap, sizeof(sWorkerCap), "%u",
                  *c.concurrency.recordWorkerCap);
  }
  else {
    std::snprintf(sWorkerCap, sizeof(sWorkerCap), "-");
  }
  SoVulkanDebug::post("[VKCONFIG] parallel=%d workerCap=%s extSec=%d wlineCpu=%d\n",
               c.concurrency.parallelRecord ? 1 : 0,
               sWorkerCap,
               c.concurrency.externalSecondary ? 1 : 0,
               c.raster.wideLineCpu ? 1 : 0);
  SoVulkanDebug::post("[VKCONFIG] diagnostics debugUtils=%d debugPrintf=%d "
               "gpuTiming=%d pipelineFeedback=%d\n",
               c.diagnostics.debugUtils ? 1 : 0,
               c.diagnostics.debugPrintf ? 1 : 0,
               c.diagnostics.gpuTimestamps ? 1 : 0,
               c.diagnostics.pipelineFeedback ? 1 : 0);
}

} // namespace SoVulkanConfig
