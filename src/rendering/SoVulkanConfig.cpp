// src/rendering/SoVulkanConfig.cpp
#include "rendering/SoVulkanConfig.h"

#include "rendering/SoVulkanShared.h"

#include <cstdio>

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

// Read a strictly positive uint32; a missing/invalid value keeps the default.
uint32_t readPositiveUint(const char * name, uint32_t fallback)
{
  if (!SoVulkanShared::envSet(name)) {
    return fallback;
  }
  const int value = SoVulkanShared::envInt(name, static_cast<int>(fallback));
  return value > 0 ? static_cast<uint32_t>(value) : fallback;
}

// Optional overrides: nullopt means "leave the caller's default".
std::optional<uint32_t> optionalUintInRange(const char * name, int lo, int hi)
{
  if (!SoVulkanShared::envSet(name)) {
    return std::nullopt;
  }
  const int value = SoVulkanShared::envInt(name, 0);
  if (value >= lo && value <= hi) {
    return static_cast<uint32_t>(value);
  }
  return std::nullopt;
}

std::optional<bool> optionalBool(const char * name)
{
  if (!SoVulkanShared::envSet(name)) {
    return std::nullopt;
  }
  return SoVulkanShared::envFlagEnabled(name, false);
}

std::optional<float> optionalFloatInRange(const char * name, float lo, float hi,
                                          bool loExclusive = false)
{
  if (!SoVulkanShared::envSet(name)) {
    return std::nullopt;
  }
  const float value = SoVulkanShared::envFloat(name, 0.0f);
  const bool loOk = loExclusive ? value > lo : value >= lo;
  if (loOk && value <= hi) {
    return value;
  }
  return std::nullopt;
}

std::optional<float> optionalNonNegativeFloat(const char * name)
{
  if (!SoVulkanShared::envSet(name)) {
    return std::nullopt;
  }
  const float value = SoVulkanShared::envFloat(name, 0.0f);
  if (value >= 0.0f) {
    return value;
  }
  return std::nullopt;
}

Config load()
{
  Config c;

  c.geometryLod.enabled =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_GEOM_LOD", true);
  c.geometryLod.always =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_GEOM_LOD_ALWAYS", false);
  c.geometryLod.stats =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_GEOM_LOD_STATS", false);
  c.geometryLod.minAreaPixels =
    readNonNegativeFloat("FC_VULKAN_GEOM_LOD_PIXELS", 1.0f);
  c.geometryLod.maxIndices =
    readPositiveUint("FC_VULKAN_GEOM_LOD_MAX_INDEX", 64000000u);

  c.rtxCull.enabled =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_TLAS_CULL", false);
  c.rtxCull.pixels = readNonNegativeFloat("FC_VULKAN_TLAS_PIX", 1.0f);
  if (c.rtxCull.pixels <= 0.0f) {
    c.rtxCull.pixels = 1.0f;
  }

  c.pathTracing.bounces = optionalUintInRange("FC_VULKAN_PT_BOUNCES", 1, 16);
  c.pathTracing.settleFrames =
    optionalUintInRange("FC_VULKAN_PT_SETTLE", 1, 120);
  c.pathTracing.maxSamples =
    optionalUintInRange("FC_VULKAN_PT_MAXSAMPLES", 1, 100000);
  c.pathTracing.adaptive = optionalBool("FC_VULKAN_PT_ADAPTIVE");
  c.pathTracing.adaptiveMinSamples =
    optionalUintInRange("FC_VULKAN_PT_MIN_SAMPLES", 1, 256);
  c.pathTracing.adaptiveThreshold =
    optionalFloatInRange("FC_VULKAN_PT_THRESHOLD", 0.0f, 1.0f, true);
  c.pathTracing.adaptiveStopFraction =
    optionalFloatInRange("FC_VULKAN_PT_STOP_FRACTION", 0.0f, 1.0f);
  c.pathTracing.fireflySigma = optionalNonNegativeFloat("FC_VULKAN_PT_FIREFLY");
  c.pathTracing.temporal = optionalBool("FC_VULKAN_PT_TEMPORAL");

  c.accelerationStructures.pack = SoVulkanShared::envSet("FC_VULKAN_AS_PACK");
  c.accelerationStructures.compact =
    SoVulkanShared::envSet("FC_VULKAN_AS_COMPACT");
  c.rayTracing.sbtPipeline = SoVulkanShared::envFlagEnabled("FC_VULKAN_RT_SBT");

  c.memoryPool.enabled =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_MEM_POOL", false);
  c.concurrency.parallelRecord =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_PARALLEL_RECORD", false);
  if (SoVulkanShared::envSet("FC_VULKAN_RECORD_WORKERS")) {
    const int v = SoVulkanShared::envInt("FC_VULKAN_RECORD_WORKERS", 0);
    if (v >= 1) {
      c.concurrency.recordWorkerCap = static_cast<unsigned int>(v);
    }
  }
  c.concurrency.externalSecondary =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_EXTERNAL_SECONDARY", false);
  c.concurrency.asyncCompute = SoVulkanShared::envSet("FC_VULKAN_ASYNC_COMPUTE");
  c.raster.wideLineCpu =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_WLINE_CPU", false);

  return c;
}

} // namespace

const Config & get()
{
  // C++11 guarantees thread-safe initialization of a function-local static, so
  // the environment is resolved exactly once, on first use, and the result is
  // immutable for the process lifetime.
  static const Config config = load();
  return config;
}

void dump()
{
  const Config & c = get();
  std::fprintf(stderr,
               "[VKCONFIG] geomLod enabled=%d always=%d stats=%d "
               "pixels=%.3f maxIndex=%u\n",
               c.geometryLod.enabled ? 1 : 0,
               c.geometryLod.always ? 1 : 0,
               c.geometryLod.stats ? 1 : 0,
               static_cast<double>(c.geometryLod.minAreaPixels),
               c.geometryLod.maxIndices);
  std::fprintf(stderr,
               "[VKCONFIG] rtxCull enabled=%d pixels=%.3f\n",
               c.rtxCull.enabled ? 1 : 0,
               static_cast<double>(c.rtxCull.pixels));
  std::fprintf(stderr,
               "[VKCONFIG] as pack=%d compact=%d sbt=%d\n",
               c.accelerationStructures.pack ? 1 : 0,
               c.accelerationStructures.compact ? 1 : 0,
               c.rayTracing.sbtPipeline ? 1 : 0);
  const PathTracing & pt = c.pathTracing;
  std::fprintf(stderr,
               "[VKCONFIG] pt bounces=%s settle=%s maxSamples=%s adaptive=%s "
               "minSamples=%s threshold=%s stopFraction=%s firefly=%s "
               "temporal=%s\n",
               pt.bounces ? "set" : "-", pt.settleFrames ? "set" : "-",
               pt.maxSamples ? "set" : "-", pt.adaptive ? "set" : "-",
               pt.adaptiveMinSamples ? "set" : "-",
               pt.adaptiveThreshold ? "set" : "-",
               pt.adaptiveStopFraction ? "set" : "-",
               pt.fireflySigma ? "set" : "-", pt.temporal ? "set" : "-");
  std::fprintf(stderr,
               "[VKCONFIG] memPool=%d parallel=%d workerCap=%s extSec=%d "
               "asyncCompute=%d wlineCpu=%d\n",
               c.memoryPool.enabled ? 1 : 0,
               c.concurrency.parallelRecord ? 1 : 0,
               c.concurrency.recordWorkerCap ? "set" : "-",
               c.concurrency.externalSecondary ? 1 : 0,
               c.concurrency.asyncCompute ? 1 : 0,
               c.raster.wideLineCpu ? 1 : 0);
}

} // namespace SoVulkanConfig
