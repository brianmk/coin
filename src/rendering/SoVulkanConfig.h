// src/rendering/SoVulkanConfig.h
//
// Typed configuration for Coin's Vulkan renderer.  Internal, not public API.
// Every COIN_VULKAN_* knob is read here once (not via scattered getenv): defaults,
// ranges and opt-out spelling live in one place and are enumerable for diag.
// envFlagEnabled treats "0"/"false"/"off" as disabled (the historical GEOM_LOD
// and TLAS_CULL checks special-cased only "0"); prefer "1"/"0" in scripts.

#ifndef COIN_SOVULKANCONFIG_H
#define COIN_SOVULKANCONFIG_H

#include <cstdint>
#include <optional>

namespace SoVulkanConfig {

// GPU sub-pixel geometry LOD (raster backend).
struct GeometryLod {
  //! Master switch.  COIN_VULKAN_GEOM_LOD; default on, "0"/"false"/"off" off.
  bool enabled = true;
  //! Force the pre-pass while the camera is static (COIN_VULKAN_GEOM_LOD_ALWAYS; default off).
  bool always = false;
  //! Print the previous frame's survivor count per compacted command (COIN_VULKAN_GEOM_LOD_STATS; off).
  bool stats = false;
  //! Minimum projected triangle area in px^2 that survives (COIN_VULKAN_GEOM_LOD_PIXELS; 1.0, >= 0).
  float minAreaPixels = 1.0f;
  //! Largest index count that gets a compacted buffer, memory bound (COIN_VULKAN_GEOM_LOD_MAX_INDEX; 64M, > 0).
  uint32_t maxIndices = 64000000u;
  //! Smallest triangle count worth compacting: below this the fixed per-command
  //! cost exceeds the shading saved (COIN_VULKAN_GEOM_LOD_MIN_PRIMS; 256, 0 disables).
  uint32_t minPrims = 256u;
};

// Command recording / queue concurrency.
struct Concurrency {
  //! Parallel command recording.  COIN_VULKAN_PARALLEL_RECORD; default off.
  bool parallelRecord = false;
  //! Upper bound on record-pool workers (COIN_VULKAN_RECORD_WORKERS; unset keeps the hardware default).
  std::optional<unsigned int> recordWorkerCap;
  //! Secondary command buffers for the external caller-owned pass (COIN_VULKAN_EXTERNAL_SECONDARY; off).
  bool externalSecondary = false;
};

// Raster-path options.
struct Raster {
  //! Force CPU wide-line quad expansion for A/B comparison (COIN_VULKAN_WLINE_CPU; default off).
  bool wideLineCpu = false;
};

// Vulkan diagnostic tooling; all default off and zero-cost when disabled, never change output.
struct Diagnostics {
  //! VK_EXT_debug_utils object names and command-buffer labels (COIN_VULKAN_DEBUG_UTILS; off).
  bool debugUtils = false;
  //! VK_EXT_debug_printf shader-side diagnostics (COIN_VULKAN_DEBUG_PRINTF; off).
  bool debugPrintf = false;
  //! Per-pass GPU timestamps, VK_QUERY_TYPE_TIMESTAMP (COIN_VULKAN_GPU_TIMING; off).
  bool gpuTimestamps = false;
  //! VK_EXT_pipeline_creation_feedback logging, cache hit + cost (COIN_VULKAN_PIPELINE_FEEDBACK; off).
  bool pipelineFeedback = false;
};

struct Config {
  GeometryLod geometryLod;
  Concurrency concurrency;
  Raster raster;
  Diagnostics diagnostics;
};

/*!
  \brief The process-wide configuration, resolved from the environment once.

  Thread-safe: resolved once on the first call (a C++11 function-local static),
  i.e. the renderer's first frame/initialize, so a probe that sets its
  environment before launching FreeCAD sees the values.  Immutable afterwards.
*/
const Config & get();

//! Emit the resolved configuration as one diagnostic line per section.
void dump();

} // namespace SoVulkanConfig

#endif // COIN_SOVULKANCONFIG_H
