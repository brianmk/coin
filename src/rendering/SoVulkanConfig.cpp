// src/rendering/SoVulkanConfig.cpp
#include "rendering/SoVulkanConfig.h"

#include "rendering/SoVulkanShared.h"

#include <cstddef>
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

// Read a strictly positive uint32; a missing/invalid value keeps the default.
// Parsed with strtoll (not envInt) so a value above INT_MAX -- e.g. a large
// FC_VULKAN_GEOM_LOD_MAX_INDEX -- saturates at UINT32_MAX instead of
// overflowing the int round-trip.
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

  // Presence-only diagnostics (see RtxDebug): any value enables them.  The
  // fill debug is the exception: it historically honored the "0"/"false"/
  // "off" opt-out, so keep that (envFlagEnabled) rather than presence-only.
  c.rtxDebug.rtDebug = SoVulkanShared::envSet("FC_VULKAN_RT_DEBUG");
  c.rtxDebug.rtGeo = SoVulkanShared::envSet("FC_VULKAN_RT_GEO");
  c.rtxDebug.rtDebugFill =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_RT_DEBUG_FILL", false);
  c.rtxDebug.ptDebug = SoVulkanShared::envSet("FC_VULKAN_PT_DEBUG");
  c.rtxDebug.denoiserDebug = SoVulkanShared::envSet("FC_VULKAN_PT_DENOISER_DEBUG");
  c.rtxDebug.denoiseTiming = SoVulkanShared::envSet("FC_VULKAN_PT_DENOISE_TIMING");
  c.rtxDebug.asyncComputeTiming =
    SoVulkanShared::envSet("FC_VULKAN_ASYNC_COMPUTE_TIMING");

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
               "[VKCONFIG] rtxDebug rtDebug=%d rtGeo=%d rtFill=%d ptDebug=%d "
               "denoiser=%d denoiseTiming=%d asyncTiming=%d\n",
               c.rtxDebug.rtDebug ? 1 : 0,
               c.rtxDebug.rtGeo ? 1 : 0,
               c.rtxDebug.rtDebugFill ? 1 : 0,
               c.rtxDebug.ptDebug ? 1 : 0,
               c.rtxDebug.denoiserDebug ? 1 : 0,
               c.rtxDebug.denoiseTiming ? 1 : 0,
               c.rtxDebug.asyncComputeTiming ? 1 : 0);
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
  // Print the resolved value, not just presence: for a diagnostics dump the
  // value is the useful part.  "-" means the backend default is in force.
  // Format into stack buffers rather than building std::strings per optional
  // (no allocation on the debug path).
  char sBounces[16], sSettle[16], sMaxSamples[16], sAdaptive[4];
  char sMinSamples[16], sThreshold[24], sStopFraction[24], sFirefly[24];
  char sTemporal[4], sWorkerCap[16];
  auto putU = [](char * b, std::size_t n, const std::optional<uint32_t> & v) {
    if (v) { std::snprintf(b, n, "%u", *v); }
    else { std::snprintf(b, n, "-"); }
  };
  auto putF = [](char * b, std::size_t n, const std::optional<float> & v) {
    if (v) { std::snprintf(b, n, "%.6g", static_cast<double>(*v)); }
    else { std::snprintf(b, n, "-"); }
  };
  auto putB = [](char * b, std::size_t n, const std::optional<bool> & v) {
    if (v) { std::snprintf(b, n, "%d", *v ? 1 : 0); }
    else { std::snprintf(b, n, "-"); }
  };
  putU(sBounces, sizeof(sBounces), pt.bounces);
  putU(sSettle, sizeof(sSettle), pt.settleFrames);
  putU(sMaxSamples, sizeof(sMaxSamples), pt.maxSamples);
  putB(sAdaptive, sizeof(sAdaptive), pt.adaptive);
  putU(sMinSamples, sizeof(sMinSamples), pt.adaptiveMinSamples);
  putF(sThreshold, sizeof(sThreshold), pt.adaptiveThreshold);
  putF(sStopFraction, sizeof(sStopFraction), pt.adaptiveStopFraction);
  putF(sFirefly, sizeof(sFirefly), pt.fireflySigma);
  putB(sTemporal, sizeof(sTemporal), pt.temporal);
  std::fprintf(stderr,
               "[VKCONFIG] pt bounces=%s settle=%s maxSamples=%s adaptive=%s "
               "minSamples=%s threshold=%s stopFraction=%s firefly=%s "
               "temporal=%s\n",
               sBounces, sSettle, sMaxSamples, sAdaptive, sMinSamples,
               sThreshold, sStopFraction, sFirefly, sTemporal);
  if (c.concurrency.recordWorkerCap) {
    std::snprintf(sWorkerCap, sizeof(sWorkerCap), "%u",
                  *c.concurrency.recordWorkerCap);
  }
  else {
    std::snprintf(sWorkerCap, sizeof(sWorkerCap), "-");
  }
  std::fprintf(stderr,
               "[VKCONFIG] memPool=%d parallel=%d workerCap=%s extSec=%d "
               "asyncCompute=%d wlineCpu=%d\n",
               c.memoryPool.enabled ? 1 : 0,
               c.concurrency.parallelRecord ? 1 : 0,
               sWorkerCap,
               c.concurrency.externalSecondary ? 1 : 0,
               c.concurrency.asyncCompute ? 1 : 0,
               c.raster.wideLineCpu ? 1 : 0);
}

} // namespace SoVulkanConfig
