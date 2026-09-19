// src/rendering/SoVulkanConfig.h
//
// Typed, documented configuration for Coin's Vulkan renderer.  Internal to
// Coin (not installed, not public API).
//
// Every FC_VULKAN_* knob that affects renderer behaviour should be read here
// once, not via scattered getenv() calls: the defaults, the opt-out spelling
// ("0"/"false"/"off") and the valid ranges then live in one place, the whole
// set is enumerable for diagnostics, and a flag's semantics cannot silently
// diverge between call sites (the historical failure mode this replaces).
//
// This is an incremental home.  The sections below cover the geometry-LOD and
// ray-tracing-culling knobs; the remaining renderer flags migrate here as
// their subsystems are touched.  SoVulkanShared::env* remains the low-level
// environment accessor used by the loader.
//
// Opt-out spelling: boolean flags use SoVulkanShared::envFlagEnabled, which
// treats the exact values "0", "false" and "off" as disabled.  The historical
// FC_VULKAN_GEOM_LOD / FC_VULKAN_TLAS_CULL checks special-cased only the
// literal "0", so a script that set either to "false" (expecting it to stay
// on) now disables it.  Prefer "1"/"0" in scripts.

#ifndef COIN_SOVULKANCONFIG_H
#define COIN_SOVULKANCONFIG_H

#include <cstdint>
#include <optional>

namespace SoVulkanConfig {

// GPU sub-pixel geometry LOD (raster backend).
struct GeometryLod {
  //! Master switch.  FC_VULKAN_GEOM_LOD; default on, "0"/"false"/"off" off.
  bool enabled = true;
  //! Force the pre-pass while the camera is static (verification aid).
  //! FC_VULKAN_GEOM_LOD_ALWAYS; default off.
  bool always = false;
  //! Print the previous frame's survivor count per compacted command.
  //! FC_VULKAN_GEOM_LOD_STATS; default off.
  bool stats = false;
  //! Minimum projected triangle area in px^2 that survives.
  //! FC_VULKAN_GEOM_LOD_PIXELS; default 1.0, clamped to >= 0.
  float minAreaPixels = 1.0f;
  //! Largest index count that gets a compacted buffer (memory bound).
  //! FC_VULKAN_GEOM_LOD_MAX_INDEX; default 64M, clamped to > 0.
  uint32_t maxIndices = 64000000u;
};

// Ray-tracing / path-tracing diagnostics.  Presence-only (any value, including
// "0", enables) except rtDebugFill, which keeps the historical "0"/"false"/
// "off" opt-out; each field documents the flag it resolves.
struct RtxDebug {
  //! FC_VULKAN_RT_DEBUG: TLAS/BLAS build and ptState per-frame diagnostics.
  bool rtDebug = false;
  //! FC_VULKAN_RT_GEO: per-command BLAS geometry diagnostics.
  bool rtGeo = false;
  //! FC_VULKAN_RT_DEBUG_FILL: constant-fill debug output for traced triangles.
  bool rtDebugFill = false;
  //! FC_VULKAN_PT_DEBUG: path-tracer debug output.
  bool ptDebug = false;
  //! FC_VULKAN_PT_DENOISER_DEBUG: denoiser diagnostics.
  bool denoiserDebug = false;
  //! FC_VULKAN_PT_DENOISE_TIMING: denoise phase timings.
  bool denoiseTiming = false;
  //! FC_VULKAN_ASYNC_COMPUTE_TIMING: async-compute overlap timings.
  bool asyncComputeTiming = false;
};

// Ray-tracing TLAS instance culling (frustum + sub-pixel).
struct RtxCull {
  //! Instance culling master switch.  FC_VULKAN_TLAS_CULL; default OFF
  //! (opt-in).  It changes the default trace path (small/far instances can pop
  //! in), so it stays off until validated across a wider range of scenes than
  //! the single large-mesh case.  FC_VULKAN_TLAS_CULL=1 (any value other than
  //! "0"/"false"/"off") enables it.
  bool enabled = false;
  //! Sub-pixel cull threshold in pixels.  FC_VULKAN_TLAS_PIX; default 1.0,
  //! clamped to > 0.
  float pixels = 1.0f;
};

// Path-tracing tuning.  These are overrides onto the backend's own member
// defaults (which stay authoritative when the variable is unset), so each is
// optional: nullopt means "leave the backend default".
struct PathTracing {
  std::optional<uint32_t> bounces;              //!< FC_VULKAN_PT_BOUNCES, [1,16]
  std::optional<uint32_t> settleFrames;         //!< FC_VULKAN_PT_SETTLE, [1,120]
  std::optional<uint32_t> maxSamples;           //!< FC_VULKAN_PT_MAXSAMPLES, [1,100000]
  std::optional<bool> adaptive;                 //!< FC_VULKAN_PT_ADAPTIVE
  std::optional<uint32_t> adaptiveMinSamples;   //!< FC_VULKAN_PT_MIN_SAMPLES, [1,256]
  std::optional<float> adaptiveThreshold;       //!< FC_VULKAN_PT_THRESHOLD, (0,1]
  std::optional<float> adaptiveStopFraction;    //!< FC_VULKAN_PT_STOP_FRACTION, [0,1]
  std::optional<float> fireflySigma;            //!< FC_VULKAN_PT_FIREFLY, >= 0
  std::optional<bool> temporal;                 //!< FC_VULKAN_PT_TEMPORAL
};

// Ray-tracing acceleration-structure build options.
struct AccelerationStructures {
  //! Half-precision position packing.  FC_VULKAN_AS_PACK, presence-only
  //! (any value, including "0", enables -- matches the historical test gate).
  bool pack = false;
  //! Build with ALLOW_COMPACTION so a later pass can shrink residency.
  //! FC_VULKAN_AS_COMPACT, presence-only.
  bool compact = false;
};

// Ray-tracing dispatch options.
struct RayTracing {
  //! Shader-binding-table pipeline instead of the inline raygen path.
  //! FC_VULKAN_RT_SBT; default off.
  bool sbtPipeline = false;
};

// Device-memory sub-allocator (raster backend).
struct MemoryPool {
  //! FC_VULKAN_MEM_POOL; default off (legacy vkAllocateMemory path).
  bool enabled = false;
};

// Command recording / queue concurrency.
struct Concurrency {
  //! Parallel command recording.  FC_VULKAN_PARALLEL_RECORD; default off.
  bool parallelRecord = false;
  //! Upper bound on record-pool workers.  FC_VULKAN_RECORD_WORKERS; unset
  //! keeps the hardware-derived default (clamped to 8).
  std::optional<unsigned int> recordWorkerCap;
  //! Secondary command buffers for the external (caller-owned) pass.
  //! FC_VULKAN_EXTERNAL_SECONDARY; default off.
  bool externalSecondary = false;
  //! Overlap the denoiser copy on a second compute queue.
  //! FC_VULKAN_ASYNC_COMPUTE; presence-only, default off.
  bool asyncCompute = false;
};

// Raster-path options.
struct Raster {
  //! Force CPU wide-line quad expansion (A/B comparison / driver escape).
  //! FC_VULKAN_WLINE_CPU; default off.
  bool wideLineCpu = false;
};

struct Config {
  GeometryLod geometryLod;
  RtxDebug rtxDebug;
  RtxCull rtxCull;
  PathTracing pathTracing;
  AccelerationStructures accelerationStructures;
  RayTracing rayTracing;
  MemoryPool memoryPool;
  Concurrency concurrency;
  Raster raster;
};

/*!
  \brief The process-wide configuration, resolved from the environment once.

  Thread-safe: the environment is resolved once, on the first call (a C++11
  function-local static), which is the renderer's first frame/initialize, so a
  probe that sets its environment before launching FreeCAD sees the values.
  The result is immutable for the process lifetime.
*/
const Config & get();

//! Emit the resolved configuration as one diagnostic line per section.
void dump();

} // namespace SoVulkanConfig

#endif // COIN_SOVULKANCONFIG_H
