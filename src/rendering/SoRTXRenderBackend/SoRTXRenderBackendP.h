// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h
//
// Private helpers shared by the SoRTXRenderBackend implementation files.
// This header is internal to Coin's RTX backend; it is not installed and is
// not part of the public API.  Include it from the split SoRTXRenderBackend
// translation units via <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>.
//
// All helpers are inline (or POD structs) so a translation unit that does not
// use a given helper does not pull an out-of-line definition; the layout
// checks are function-local static_asserts so they only fire where the
// corresponding struct is actually used.

#ifndef COIN_SORTXRENDERBACKENDP_H
#define COIN_SORTXRENDERBACKENDP_H

#include "rendering/SoRTXRenderBackend.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace SoRTXBackend {

// Environment flags are enabled by presence, but honor the conventional
// "VAR=0"/"false"/"off" opt-out values.
inline bool
envFlagEnabled(const char * name)
{
  const char * value = std::getenv(name);
  if (value == nullptr) return false;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "off") != 0;
}

// std430 mirror of the RTMaterial struct in PathTrace.glsl.  One record per
// draw command, indexed by the instance custom index (the command index).
// C++ packs the float arrays without padding, which matches std430: 5 vec4
// + 6 arrays of 8 vec4 = 80 + 768 = 848 bytes.
// (struct RTMaterial is defined in SoRTXRenderBackend.h; the static assert
// below pins its size to the shader layout.)
inline void checkRtlLayout() {
  static_assert(sizeof(RTMaterial) == 880,
                "RTMaterial must match PathTrace.glsl std430 layout");
}

// std140 mirror of the FrameBlock uniform in Raygen/ClosestHit/Miss.glsl.
struct alignas(16) RTXFrameBlock {
  float view[16];
  float viewInverse[16];
  float projInverse[16];
  float cameraPos[4];
  float viewport[4]; // x = width, y = height, z = orthographic, w = unused
  float bgTop[4];
  float bgBottom[4];
  float state[4]; // x = frameIndex, y = pathTracing, z = accumulating,
                  // w = maxBounces
  float adaptive[4]; // x = minSamples, y = relErrorThreshold (0 = off)
  float prevViewProj[16]; // world -> clip of the previous frame's camera
  float temporal[4];      // x = reproject this frame
  float nee[4];           // x = emissive-triangle count, y = NEE enabled,
                          // z = MIS balance enabled
};
inline void checkRtlFrameBlockLayout() {
  static_assert(sizeof(RTXFrameBlock) == 4 * 64 + 8 * 16,
                "RTXFrameBlock must match FrameBlock std140 layout");
}

// Push constant block of the raygen shader (RaygenPush in Raygen.glsl).
// Per-frame state rides here instead of the shared UBO: the chit and miss
// stages never read it, so it stays out of the descriptor sets entirely.
struct alignas(16) RTXRaygenPush {
  uint32_t frameIndex = 0;
  uint32_t flags = 0; // bit 0 = path tracing, bit 1 = accumulating,
                      // bit 2 = debug fill
  uint32_t maxBounces = 4;
  uint32_t pad = 0;
};
inline void checkRtlRaygenPushLayout() {
  static_assert(sizeof(RTXRaygenPush) == 16,
                "RTXRaygenPush must match RaygenPush layout");
}

constexpr int SBT_GROUP_COUNT = 5; // raygen, miss, shadow miss, chit, shadow chit

constexpr int MAX_SHADER_LIGHTS = 8;
constexpr int MAX_VERTEX_COUNT = 10000000;

// Pick the first memory type matching the desired properties, or any type
// the device offers for this resource as a fallback.
inline uint32_t
findMemoryType(VkPhysicalDevice physicalDevice,
               const VkMemoryRequirements & requirements,
               VkMemoryPropertyFlags desired)
{
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags & desired) == desired) {
      return i;
    }
  }
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if (requirements.memoryTypeBits & (1u << i)) {
      return i;
    }
  }
  return 0;
}

// FNV-1a content hash of a command's geometry, sampled so full-scene
// hashing stays sub-millisecond.  The producer's geometry storage is a
// per-frame arena, so this hash -- not pointer identity -- is the scene
// change signal for the geometry cache.
inline uint64_t
hashGeometry(const SoGeometryDesc & geometry, uint32_t vertexStride,
             bool indexed)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  mix(geometry.vertexCount);
  mix(geometry.indexCount);
  mix(vertexStride);

  const size_t posStrideFloats = vertexStride / sizeof(float);
  const size_t totalFloats =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  if (totalFloats > 0) {
    const size_t samples = 512;
    const size_t step = totalFloats > samples ? totalFloats / samples : 1;
    for (size_t i = 0; i < totalFloats; i += step) {
      mix(std::bit_cast<uint32_t>(geometry.positions[i]));
    }
    mix(std::bit_cast<uint32_t>(geometry.positions[totalFloats - 1]));
  }

  if (indexed) {
    const size_t count = geometry.indexCount;
    // Hash every index for scenes up to a practical threshold (covers
    // typical CAD parts in full); beyond that fall back to uniform
    // sampling so the per-frame cost stays bounded.  Either way the first
    // and last indices are always included.
    const size_t samples = 256;
    if (count <= 65536) {
      for (size_t i = 0; i < count; ++i) {
        mix(geometry.indices[i]);
      }
    }
    else {
      const size_t step = count / samples;
      for (size_t i = 0; i < count; i += step) {
        mix(geometry.indices[i]);
      }
    }
    mix(geometry.indices[count - 1]);
  }
  return h;
}

// A cheap per-frame "change signal" for one command's geometry.  It mixes
// only the metadata plus a sampled subset of positions and indices, so it is
// far cheaper than the full hashGeometry() walk (which hashes every index for
// scenes up to 65536 indices).  When this signal is unchanged from the cache
// entry, the geometry is assumed unchanged for cache purposes and the full
// hash is reused, avoiding the per-frame full-index walk over large CAD parts.
inline uint64_t
hashGeometrySignal(const SoGeometryDesc & geometry, uint32_t vertexStride,
                   bool indexed)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  mix(geometry.vertexCount);
  mix(geometry.indexCount);
  mix(vertexStride);

  const size_t posStrideFloats = vertexStride / sizeof(float);
  const size_t totalFloats =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  if (totalFloats > 0) {
    // Sample the position data (same sampling rate as hashGeometry).
    const size_t samples = 512;
    const size_t step = totalFloats > samples ? totalFloats / samples : 1;
    for (size_t i = 0; i < totalFloats; i += step) {
      mix(std::bit_cast<uint32_t>(geometry.positions[i]));
    }
    mix(std::bit_cast<uint32_t>(geometry.positions[totalFloats - 1]));
  }

  if (indexed && geometry.indexCount > 0) {
    // Sample indices (same rate as the large-scene fallback).
    const size_t count = geometry.indexCount;
    const size_t samples = 256;
    const size_t step = count > samples ? count / samples : 1;
    for (size_t i = 0; i < count; i += step) {
      mix(geometry.indices[i]);
    }
    mix(geometry.indices[count - 1]);
  }
  return h;
}

// FNV-1a hash of a command's vertex positions only.  Separates position
// edits (refit-able: topology unchanged) from index edits (topology change,
// full rebuild required).
inline uint64_t
hashPositions(const SoGeometryDesc & geometry, uint32_t vertexStride)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  const size_t posStrideFloats = vertexStride / sizeof(float);
  const size_t totalFloats =
    static_cast<size_t>(geometry.vertexCount) * posStrideFloats;
  for (size_t i = 0; i < totalFloats; ++i) {
    mix(std::bit_cast<uint32_t>(geometry.positions[i]));
  }
  return h;
}

// FNV-1a hash of a command's index data only (full walk up to the same
// 65536 threshold as hashGeometry, uniform sampling beyond).
inline uint64_t
hashIndices(const SoGeometryDesc & geometry)
{
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
  };
  const size_t count = geometry.indexCount;
  if (count <= 65536) {
    for (size_t i = 0; i < count; ++i) {
      mix(geometry.indices[i]);
    }
  }
  else {
    const size_t samples = 256;
    const size_t step = count / samples;
    for (size_t i = 0; i < count; i += step) {
      mix(geometry.indices[i]);
    }
    mix(geometry.indices[count - 1]);
  }
  return h;
}

// Epsilon-aware float-array comparison: camera matrices may oscillate by
// tiny numerical amounts every frame without a real camera change, so a
// strict memcmp would never let the accumulation run.
inline bool
matricesNearlyEqual(const float * a, const float * b, size_t count)
{
  for (size_t i = 0; i < count; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    const float scale = std::max(std::fabs(a[i]), std::fabs(b[i]));
    if (d > 1e-4f * scale + 1e-7f) return false;
  }
  return true;
}

} // namespace SoRTXBackend

#endif // COIN_SORTXRENDERBACKENDP_H
