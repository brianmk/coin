// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendGeometry.cpp

// Split from the original monolithic SoRTXRenderBackend.cpp.  Contains the
// member functions for the "Geometry" concern of the Vulkan RTX backend.

#include "rendering/SoRTXRenderBackend.h"
#include <Inventor/errors/SoDebugError.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include "rendering/vulkan/rt/PathTrace.spv.h"
#include "rendering/vulkan/rt/Raygen.spv.h"
#include "rendering/vulkan/rt/Miss.spv.h"
#include "rendering/vulkan/rt/ShadowMiss.spv.h"
#include "rendering/vulkan/rt/ClosestHit.spv.h"
#include "rendering/vulkan/rt/ShadowClosestHit.spv.h"
#include "rendering/vulkan/rt/PresentVertex.spv.h"
#include "rendering/vulkan/rt/PresentFragment.spv.h"
#include <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>

using namespace SoRTXBackend;

bool
SoRTXRenderBackend::ensureNormalPoolCapacity(VkDeviceSize bytes)
{
  if (this->normalPoolBuffer != VK_NULL_HANDLE &&
      this->normalPoolCapacity >= bytes) {
    return true;
  }
  // Grow-only pool: double until the requested size fits.  The new buffer
  // is created (and mapped) before the old one is released, so a failed
  // allocation leaves the previous pool intact and usable.  The old buffer
  // is only referenced by acceleration-structure-phase submissions, which
  // complete before the next pool resize can run (per-frame queue drain),
  // so releasing it here is safe.
  VkDeviceSize newCapacity = std::max<VkDeviceSize>(64 * 1024, bytes);
  while (newCapacity < this->normalPoolCapacity + bytes) {
    newCapacity *= 2;
  }
  VkBuffer newBuffer = VK_NULL_HANDLE;
  VkDeviceMemory newMemory = VK_NULL_HANDLE;
  void * newMapped = nullptr;
  if (!this->createHostVisibleBuffer(
        newCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        newBuffer, newMemory)) {
    return false;
  }
  if (vkMapMemory(this->device, newMemory, 0, newCapacity, 0,
                  &newMapped) != VK_SUCCESS) {
    vkDestroyBuffer(this->device, newBuffer, this->allocator);
    vkFreeMemory(this->device, newMemory, this->allocator);
    return false;
  }
  if (this->normalPoolBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->normalPoolBuffer, this->allocator);
    this->normalPoolBuffer = VK_NULL_HANDLE;
    vkFreeMemory(this->device, this->normalPoolMemory, this->allocator);
    this->normalPoolMemory = VK_NULL_HANDLE;
    this->normalPoolMapped = nullptr;
  }
  this->normalPoolCapacity = newCapacity;
  this->normalPoolBuffer = newBuffer;
  this->normalPoolMemory = newMemory;
  this->normalPoolMapped = newMapped;
  this->normalPoolUsed = 0;
  // The pool identity changed: refresh the descriptor sets.
  return this->updateDescriptors();
}

VkDeviceSize
SoRTXRenderBackend::appendTriangleNormals(const SoRenderCommand & command,
                                          RTXCachedGeometry & entry)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);
  const bool indexed = entry.indexCount > 0 && entry.idxKey != nullptr;
  const uint32_t triangleCount =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (triangleCount == 0) return 0;

  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(triangleCount) * 4 * sizeof(float);

  // Reuse the entry's existing pool slot when the triangle count is
  // unchanged; otherwise append (the pool grows over the session).
  const uint32_t existingOffset = entry.normalPoolOffset;
  const bool reuse = existingOffset != 0xFFFFFFFFu &&
    entry.normalCount == triangleCount &&
    (static_cast<VkDeviceSize>(existingOffset) * 16 + bytes) <=
      this->normalPoolUsed;
  if (!reuse) {
    if (!this->ensureNormalPoolCapacity(this->normalPoolUsed + bytes)) {
      this->emitError("appendTriangleNormals: pool allocation failed");
      return 0;
    }
    entry.normalPoolOffset =
      static_cast<uint32_t>(this->normalPoolUsed / (4 * sizeof(float)));
    this->normalPoolUsed += bytes;
  }
  entry.normalCount = triangleCount;

  // Object-space per-triangle geometric normals (flat shading).
  float * out = static_cast<float *>(this->normalPoolMapped) +
    static_cast<size_t>(entry.normalPoolOffset) * 4;
  const auto vertex = [&geometry, posStrideFloats](uint32_t i) {
    return geometry.positions + static_cast<size_t>(i) * posStrideFloats;
  };
  for (uint32_t t = 0; t < triangleCount; ++t) {
    const uint32_t i0 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 0] : t * 3 + 0;
    const uint32_t i1 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 1] : t * 3 + 1;
    const uint32_t i2 = indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 2] : t * 3 + 2;
    const float * p0 = vertex(i0);
    const float * p1 = vertex(i1);
    const float * p2 = vertex(i2);
    const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    float nx = e1[1] * e2[2] - e1[2] * e2[1];
    float ny = e1[2] * e2[0] - e1[0] * e2[2];
    float nz = e1[0] * e2[1] - e1[1] * e2[0];
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-12f) {
      nx /= len; ny /= len; nz /= len;
    }
    else {
      nx = 0.0f; ny = 0.0f; nz = 1.0f;
    }
    out[static_cast<size_t>(t) * 4 + 0] = nx;
    out[static_cast<size_t>(t) * 4 + 1] = ny;
    out[static_cast<size_t>(t) * 4 + 2] = nz;
    out[static_cast<size_t>(t) * 4 + 3] = 0.0f;
  }
  return bytes;
}

bool
SoRTXRenderBackend::ensureNeePoolCapacity(VkDeviceSize bytes)
{
  if (this->neePoolBuffer != VK_NULL_HANDLE &&
      this->neePoolCapacity >= bytes) {
    return true;
  }
  // Grow-only pool (see ensureNormalPoolCapacity for the lifetime
  // argument; the pool is only read by per-frame drained submissions).
  VkDeviceSize newCapacity = std::max<VkDeviceSize>(64 * 1024, bytes);
  while (newCapacity < this->neePoolCapacity + bytes) {
    newCapacity *= 2;
  }
  VkBuffer newBuffer = VK_NULL_HANDLE;
  VkDeviceMemory newMemory = VK_NULL_HANDLE;
  void * newMapped = nullptr;
  if (!this->createHostVisibleBuffer(
        newCapacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        newBuffer, newMemory)) {
    return false;
  }
  if (vkMapMemory(this->device, newMemory, 0, newCapacity, 0,
                  &newMapped) != VK_SUCCESS) {
    vkDestroyBuffer(this->device, newBuffer, this->allocator);
    vkFreeMemory(this->device, newMemory, this->allocator);
    return false;
  }
  if (this->neePoolBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->neePoolBuffer, this->allocator);
    this->neePoolBuffer = VK_NULL_HANDLE;
    vkFreeMemory(this->device, this->neePoolMemory, this->allocator);
    this->neePoolMemory = VK_NULL_HANDLE;
    this->neePoolMapped = nullptr;
  }
  this->neePoolCapacity = newCapacity;
  this->neePoolBuffer = newBuffer;
  this->neePoolMemory = newMemory;
  this->neePoolMapped = newMapped;
  this->neePoolUsed = 0;
  return true;
}

void
SoRTXRenderBackend::buildNeePool(const SoDrawList & drawlist)
{
  // Full per-frame rebuild: 8 vec4 per entry (v0+v1+v2+color+mat4 xform).
  this->neePoolUsed = 0;
  this->neePoolCount = 0;
  uint32_t entryCount = 0;

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_TRANSPARENT ||
        command.pass == SO_RENDERPASS_OVERLAY) continue;
    const SoMaterialData & material = command.material;
    if (!(material.emissive[0] > 0.0f || material.emissive[1] > 0.0f ||
          material.emissive[2] > 0.0f)) {
      continue;
    }
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    RTXCachedGeometry & entry = this->geometryCache[found->second];
    if (entry.blas == VK_NULL_HANDLE) continue;

    const SoGeometryDesc & geometry = command.geometry;
    const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);
    const bool indexed = entry.indexCount > 0 && entry.idxKey != nullptr;
    const uint32_t triangleCount =
      indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
    if (triangleCount == 0) continue;

    const VkDeviceSize bytes =
      static_cast<VkDeviceSize>(triangleCount) * 8 * 4 * sizeof(float);
    if (!this->ensureNeePoolCapacity(this->neePoolUsed + bytes)) {
      this->emitError("buildNeePool: pool allocation failed");
      return;
    }
    entry.neePoolOffset = static_cast<uint32_t>(this->neePoolUsed /
                                                (8 * 4 * sizeof(float)));
    entry.neeCount = triangleCount;
    this->neePoolUsed += bytes;
    entryCount += triangleCount;

    float * out = static_cast<float *>(this->neePoolMapped) +
      static_cast<size_t>(entry.neePoolOffset) * 32;
    const auto vertex = [&geometry, posStrideFloats](uint32_t j) {
      return geometry.positions + static_cast<size_t>(j) * posStrideFloats;
    };
    const SbMatrix & m = command.modelMatrix;
    for (uint32_t t = 0; t < triangleCount; ++t) {
      const uint32_t i0 =
        indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 0] : t * 3 + 0;
      const uint32_t i1 =
        indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 1] : t * 3 + 1;
      const uint32_t i2 =
        indexed ? geometry.indices[static_cast<size_t>(t) * 3 + 2] : t * 3 + 2;
      const float * p0 = vertex(i0);
      const float * p1 = vertex(i1);
      const float * p2 = vertex(i2);
      float * e = out + static_cast<size_t>(t) * 32;
      for (int c = 0; c < 3; ++c) {
        e[c] = p0[c];
        e[4 + c] = p1[c];
        e[8 + c] = p2[c];
      }
      const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
      const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      const float cx = e1[1] * e2[2] - e1[2] * e2[1];
      const float cy = e1[2] * e2[0] - e1[0] * e2[2];
      const float cz = e1[0] * e2[1] - e1[1] * e2[0];
      e[3] = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
      e[12] = material.emissive[0];
      e[13] = material.emissive[1];
      e[14] = material.emissive[2];
      e[15] = 0.0f;
      // GLSL mat4 is column-major and applies column vectors, while
      // SbMatrix is the row-vector convention (translation in row 3), so
      // the transposed matrix goes into the pool: xform[c][r] = m[c][r].
      for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
          e[16 + c * 4 + r] = m[c][r];
        }
      }
    }
  }

  this->neePoolCount = entryCount;
  if (getenv("FC_VULKAN_RT_DEBUG") && entryCount > 0) {
    const float * e = static_cast<const float *>(this->neePoolMapped);
    fprintf(stderr, "[RTDBG] nee pool triangles=%u bytes=%llu enabled=%d "
                    "mis=%d xformT=(%.2f,%.2f,%.2f)\n",
            entryCount,
            static_cast<unsigned long long>(this->neePoolUsed),
            this->rtNeeEnabled ? 1 : 0, this->rtMisEnabled ? 1 : 0,
            e[28], e[29], e[30]);
  }
}

// --- Geometry cache -------------------------------------------------------

RTXCachedGeometry &
SoRTXRenderBackend::getOrCreateCache(const SoRenderCommand * command)
{
  const auto found = this->commandToCache.find(command);
  if (found != this->commandToCache.end()) {
    return this->geometryCache[found->second];
  }
  const size_t index = this->geometryCache.size();
  this->geometryCache.emplace_back();
  this->geometryCache.back().commandKey = command;
  this->commandToCache[command] = index;
  return this->geometryCache.back();
}

void
SoRTXRenderBackend::destroyCacheEntry(RTXCachedGeometry & entry)
{
  if (entry.blas != VK_NULL_HANDLE) {
    vkDestroyAccelerationStructureKHR(this->device, entry.blas,
                                      this->allocator);
    entry.blas = VK_NULL_HANDLE;
  }
  if (entry.blasBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, entry.blasBuffer, this->allocator);
    entry.blasBuffer = VK_NULL_HANDLE;
  }
  if (entry.blasMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.blasMemory, this->allocator);
    entry.blasMemory = VK_NULL_HANDLE;
  }
  if (entry.vertexBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, entry.vertexBuffer, this->allocator);
    entry.vertexBuffer = VK_NULL_HANDLE;
  }
  if (entry.vertexMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.vertexMemory, this->allocator);
    entry.vertexMemory = VK_NULL_HANDLE;
  }
  if (entry.indexBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, entry.indexBuffer, this->allocator);
    entry.indexBuffer = VK_NULL_HANDLE;
  }
  if (entry.indexMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.indexMemory, this->allocator);
    entry.indexMemory = VK_NULL_HANDLE;
  }
  entry = RTXCachedGeometry();
}

void
SoRTXRenderBackend::deferDestroyCacheEntry(RTXCachedGeometry & entry)
{
  if (entry.blas == VK_NULL_HANDLE && entry.vertexBuffer == VK_NULL_HANDLE &&
      entry.indexBuffer == VK_NULL_HANDLE) {
    entry = RTXCachedGeometry();
    return;
  }
  VkDevice device = this->device;
  const VkAllocationCallbacks * allocator = this->allocator;
  const PFN_vkDestroyAccelerationStructureKHR vkDestroyAS =
    this->vkDestroyAccelerationStructureKHR;
  const VkAccelerationStructureKHR blas = entry.blas;
  const VkBuffer blasBuffer = entry.blasBuffer;
  const VkDeviceMemory blasMemory = entry.blasMemory;
  const VkBuffer vertexBuffer = entry.vertexBuffer;
  const VkDeviceMemory vertexMemory = entry.vertexMemory;
  const VkBuffer indexBuffer = entry.indexBuffer;
  const VkDeviceMemory indexMemory = entry.indexMemory;
  this->deferDestroy([device, allocator, vkDestroyAS, blas, blasBuffer,
                      blasMemory, vertexBuffer, vertexMemory, indexBuffer,
                      indexMemory]() {
    if (blas != VK_NULL_HANDLE) {
      vkDestroyAS(device, blas, allocator);
    }
    if (blasBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, blasBuffer, allocator);
    }
    if (blasMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, blasMemory, allocator);
    }
    if (indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, indexBuffer, allocator);
    }
    if (indexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, indexMemory, allocator);
    }
    if (vertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, vertexBuffer, allocator);
    }
    if (vertexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, vertexMemory, allocator);
    }
  });
  entry = RTXCachedGeometry();
}

void
SoRTXRenderBackend::freePendingStagingDestroys()
{
  for (const auto & entry : this->pendingStagingDestroys) {
    if (entry.first != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, entry.first, this->allocator);
    }
    if (entry.second != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, entry.second, this->allocator);
    }
  }
  this->pendingStagingDestroys.clear();
}

void
SoRTXRenderBackend::flushPendingDestroys(bool waitForQueue)
{
  const int batch = this->pendingDestroyIndex;
  if (waitForQueue && this->queue != VK_NULL_HANDLE && !this->pendingDestroys[batch].empty()) {
    vkQueueWaitIdle(this->queue);
  }
  this->pendingDestroyIndex = batch ^ 1;
  for (const auto & fn : this->pendingDestroys[batch]) {
    if (fn) fn();
  }
  this->pendingDestroys[batch].clear();
}

void
SoRTXRenderBackend::deferDestroy(std::function<void()> && fn)
{
  this->pendingDestroys[this->pendingDestroyIndex].push_back(std::move(fn));
}

void
SoRTXRenderBackend::invalidateCache()
{
  for (RTXCachedGeometry & entry : this->geometryCache) {
    this->destroyCacheEntry(entry);
  }
  this->geometryCache.clear();
  this->commandToCache.clear();
}

void
SoRTXRenderBackend::updateGeometryCache(const SoDrawList & drawlist)
{
  // The draw-list generation is a per-frame production counter, not a
  // scene-change signal, and the producer-owned geometry storage is a
  // per-frame arena whose pointers change every frame.  Cache invalidation
  // is therefore driven by the sampled content hash of each command's
  // vertex/index data: entries whose content is unchanged keep their BLAS;
  // only genuinely new geometry triggers a rebuild.
  this->cacheChanged = false;
  const uint32_t frame = ++this->cacheFrame;

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) continue;
    if (geometry.topology != SO_TOPOLOGY_TRIANGLES) continue;

    const uint32_t vertexStride = geometry.vertexStride
      ? geometry.vertexStride : sizeof(float) * 3;
    const bool indexed = geometry.indexCount > 0 && geometry.indices != nullptr;

    const bool traced = (command.pass != SO_RENDERPASS_TRANSPARENT &&
                         command.pass != SO_RENDERPASS_OVERLAY);
    if (!traced) {
      // A traced base command that the selection highlight temporarily
      // promotes out of the OPAQUE pass (OPAQUE -> OVERLAY on select, back on
      // deselect) must not invalidate the geometry cache.  Without this the
      // command's cache entry is evicted while it is in OVERLAY and then
      // rebuilt as "new" when it returns to OPAQUE -- a false scene change
      // that restarts the accumulation and denoiser on every selection toggle.
      // Keep the matching entry alive by content signal (the cheap probe used
      // by the traced path below) so it re-keys, with no rebuild and no
      // cacheChanged, when the command returns to the traced pass.  Pure
      // non-traced commands (nav cube, axis cross, transparent shells) that
      // were never traced have no matching entry and are ignored here.
      const uint64_t signal = hashGeometrySignal(geometry, vertexStride, indexed);
      const auto found = this->commandToCache.find(&command);
      if (found != this->commandToCache.end()) {
        RTXCachedGeometry & e = this->geometryCache[found->second];
        e.cacheGeneration = frame;
        e.commandKey = &command;
        if (e.blas != VK_NULL_HANDLE && e.changeSignal != signal) {
          e.changeSignal = signal;
          e.contentHash = hashGeometry(geometry, vertexStride, indexed);
        }
      }
      else {
        for (RTXCachedGeometry & e : this->geometryCache) {
          if (e.blas != VK_NULL_HANDLE && e.changeSignal == signal &&
              e.vertexCount == geometry.vertexCount &&
              e.indexCount == geometry.indexCount &&
              e.vertexStride == vertexStride &&
              ((e.idxKey != nullptr) == indexed)) {
            e.cacheGeneration = frame;
            e.commandKey = &command;
            this->commandToCache[&command] =
              static_cast<size_t>(&e - this->geometryCache.data());
            break;
          }
        }
      }
      continue;
    }

    // A singular (degenerate) model matrix means the command collapses to a
    // point or line -- e.g. the view's hidden anchor cube, which FreeCAD
    // hides by scaling it to (0,0,0) (see View3DInventorViewer::construct*).
    // In the GL/raster path a zero-determinant transform simply renders no
    // visible pixels; in the path-traced backend it would still be built into
    // the TLAS, producing a phantom/degenerate block that shadows the NEE
    // light and shows up from the top view as a faint rectangle.  The picking
    // path already excludes these (SoPickStyle::UNPICKABLE); mirror that here
    // by skipping any command whose model matrix is non-invertible.  Leaving
    // the generation stamp stale for an existing entry evicts it below.
    {
      const SbMatrix & m = command.modelMatrix;
      // Determinant of the linear 3x3 part.  A near-zero determinant (any
      // axis flattened) makes the geometry non-renderable.
      const double det =
        static_cast<double>(m[0][0]) * (static_cast<double>(m[1][1]) * m[2][2] -
                                        static_cast<double>(m[1][2]) * m[2][1]) -
        static_cast<double>(m[0][1]) * (static_cast<double>(m[1][0]) * m[2][2] -
                                        static_cast<double>(m[1][2]) * m[2][0]) +
        static_cast<double>(m[0][2]) * (static_cast<double>(m[1][0]) * m[2][1] -
                                        static_cast<double>(m[1][1]) * m[2][0]);
      if (det < 1e-9 && det > -1e-9) {
        continue;
      }
    }

    // Cheap probe first: only run the full index hash when the change signal
    // disagrees with the cache entry.  The sampled signal costs a fraction of
    // the full hashGeometry() (which hashes every index up to 65536), so for
    // unchanged large CAD parts we reuse contentHash instead of re-walking
    // the whole index buffer every frame.
    const uint64_t signal = hashGeometrySignal(geometry, vertexStride, indexed);
    uint64_t hash = 0;
    RTXCachedGeometry * entryPtr = nullptr;

    const auto found = this->commandToCache.find(&command);
    if (found != this->commandToCache.end()) {
      RTXCachedGeometry & entry = this->geometryCache[found->second];
      hash = entry.contentHash;
      bool matches = entry.blas != VK_NULL_HANDLE &&
        entry.changeSignal == signal && entry.contentHash != 0;
      if (!matches) {
        hash = hashGeometry(geometry, vertexStride, indexed);
        matches = entry.blas != VK_NULL_HANDLE && entry.contentHash == hash;
      }
      if (!matches) {
        // Split the identity: position-only changes (same topology) refit
        // the existing BLAS in place; index/topology changes destroy and
        // rebuild.
        const uint64_t vertexHash = hashPositions(geometry, vertexStride);
        const uint64_t indexHash = indexed ? hashIndices(geometry) : 0;
        const bool topologyStable =
          entry.blas != VK_NULL_HANDLE &&
          entry.vertexCount == geometry.vertexCount &&
          entry.indexCount == geometry.indexCount &&
          entry.vertexStride == vertexStride &&
          ((entry.idxKey != nullptr) == indexed) &&
          entry.indexHash == indexHash;
        this->cacheChanged = true;
        if (topologyStable) {
          // In-place UPDATE build (see refitBlas()); keep buffers and BLAS.
          entry.refitPending = true;
          entry.posKey = geometry.positions;
          entry.idxKey = geometry.indices;
        }
        else {
          // Buffers/AS are rebuilt in recordAccelerationStructures (they
          // need a command buffer); only release old resources and record
          // the new identity here.  Destruction is deferred: a pending
          // frame may still reference the old BLAS.
          this->deferDestroyCacheEntry(entry);
          entry.posKey = geometry.positions;
          entry.idxKey = geometry.indices;
          entry.vertexCount = geometry.vertexCount;
          entry.indexCount = geometry.indexCount;
          entry.vertexStride = vertexStride;
        }
        entry.contentHash = hash;
        entry.changeSignal = signal;
        entry.vertexHash = vertexHash;
        entry.indexHash = indexHash;
      }
      entryPtr = &entry;
    }
    else {
      // The command pointer changed (draw-list storage reallocation or
      // reordering when objects are added/removed): instead of thrashing a
      // full BLAS rebuild, re-key the unclaimed entry whose content is
      // identical so the acceleration structure survives the pointer churn.
      hash = hashGeometry(geometry, vertexStride, indexed);
      RTXCachedGeometry * match = nullptr;
      for (RTXCachedGeometry & e : this->geometryCache) {
        if (e.cacheGeneration == frame) continue;
        if (e.blas != VK_NULL_HANDLE && e.contentHash == hash &&
            e.vertexCount == geometry.vertexCount &&
            e.indexCount == geometry.indexCount &&
            e.vertexStride == vertexStride &&
            ((e.idxKey != nullptr) == indexed)) {
          match = &e;
          break;
        }
      }
      if (match) {
        match->changeSignal = signal;
        this->commandToCache[&command] =
          static_cast<size_t>(match - this->geometryCache.data());
        entryPtr = match;
      }
      else {
        this->cacheChanged = true;
        RTXCachedGeometry & entry = this->getOrCreateCache(&command);
        entry.posKey = geometry.positions;
        entry.idxKey = geometry.indices;
        entry.vertexCount = geometry.vertexCount;
        entry.indexCount = geometry.indexCount;
        entry.vertexStride = vertexStride;
        entry.contentHash = hash;
        entry.changeSignal = signal;
        entry.vertexHash = hashPositions(geometry, vertexStride);
        entry.indexHash = indexed ? hashIndices(geometry) : 0;
        entryPtr = &entry;
      }
    }
    entryPtr->commandKey = &command;
    entryPtr->cacheGeneration = frame;
  }

  // Evict entries whose command disappeared from the draw list this frame
  // (their generation stamp is stale).  Command pointers live in a
  // per-frame arena, so the stamp -- not pointer identity -- decides
  // liveness; survivors rebuild the pointer map from their stored
  // commandKey.
  bool anyStale = false;
  for (RTXCachedGeometry & entry : this->geometryCache) {
    if (entry.cacheGeneration != frame) {
      this->deferDestroyCacheEntry(entry);
      anyStale = true;
    }
  }
  if (anyStale) {
    size_t write = 0;
    for (size_t idx = 0; idx < this->geometryCache.size(); ++idx) {
      if (this->geometryCache[idx].cacheGeneration == frame) {
        if (write != idx) {
          this->geometryCache[write] = std::move(this->geometryCache[idx]);
        }
        ++write;
      }
    }
    this->geometryCache.resize(write);
    this->commandToCache.clear();
    for (size_t idx = 0; idx < this->geometryCache.size(); ++idx) {
      this->commandToCache[this->geometryCache[idx].commandKey] = idx;
    }
  }
}

bool
SoRTXRenderBackend::buildBlas(RTXCachedGeometry & entry,
                              const SoRenderCommand & command,
                              VkCommandBuffer cmd)
{
  const SoGeometryDesc & geometry = command.geometry;
  const bool indexed = entry.indexCount > 0 && entry.idxKey != nullptr;
  const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);

  if (getenv("FC_VULKAN_RT_DEBUG")) {
    static uint32_t blasSeq = 0;
    fprintf(stderr,
            "[RTDBG] buildBlas #%u verts=%u idx=%u stride=%u indexed=%d "
            "pos=%p idxPtr=%p\n",
            blasSeq++, entry.vertexCount, entry.indexCount, entry.vertexStride,
            indexed ? 1 : 0, static_cast<const void *>(geometry.positions),
            static_cast<const void *>(geometry.indices));
  }

  // The path tracing compute shader shades flat faces from the object-space
  // triangle-normal pool; append this command's normals (the material
  // records pick up the offset afterwards in updateMaterials()).
  this->appendTriangleNormals(command, entry);

  // Position-only vertex buffer (tightly packed vec3) for the BLAS.
  std::vector<float> positions(static_cast<size_t>(entry.vertexCount) * 3);
  for (uint32_t i = 0; i < entry.vertexCount; ++i) {
    const float * p =
      geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    positions[static_cast<size_t>(i) * 3 + 0] = p[0];
    positions[static_cast<size_t>(i) * 3 + 1] = p[1];
    positions[static_cast<size_t>(i) * 3 + 2] = p[2];
  }
  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(entry.vertexCount) * 3 * sizeof(float);
  if (!this->createDeviceLocalBuffer(
        vertexBytes,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        entry.vertexBuffer, entry.vertexMemory)) {
    return false;
  }

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createHostVisibleBuffer(vertexBytes,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     staging, stagingMemory)) {
    return false;
  }
  void * mapped = nullptr;
  if (vkMapMemory(this->device, stagingMemory, 0, vertexBytes, 0, &mapped) !=
        VK_SUCCESS ||
      mapped == nullptr) {
    this->emitError("buildBlas: vkMapMemory (vertex staging) failed");
    vkDestroyBuffer(this->device, staging, this->allocator);
    vkFreeMemory(this->device, stagingMemory, this->allocator);
    return false;
  }
  std::memcpy(mapped, positions.data(), static_cast<size_t>(vertexBytes));
  vkUnmapMemory(this->device, stagingMemory);

  VkBuffer indexStaging = VK_NULL_HANDLE;
  VkDeviceMemory indexStagingMemory = VK_NULL_HANDLE;
  VkDeviceSize indexBytes = 0;
  if (indexed) {
    indexBytes =
      static_cast<VkDeviceSize>(entry.indexCount) * sizeof(uint32_t);
    if (!this->createDeviceLocalBuffer(
          indexBytes,
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          entry.indexBuffer, entry.indexMemory)) {
      vkDestroyBuffer(this->device, staging, this->allocator);
      vkFreeMemory(this->device, stagingMemory, this->allocator);
      return false;
    }
    if (!this->createHostVisibleBuffer(indexBytes,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       indexStaging, indexStagingMemory)) {
      vkDestroyBuffer(this->device, staging, this->allocator);
      vkFreeMemory(this->device, stagingMemory, this->allocator);
      return false;
    }
    void * imapped = nullptr;
    if (vkMapMemory(this->device, indexStagingMemory, 0, indexBytes, 0,
                    &imapped) != VK_SUCCESS ||
        imapped == nullptr) {
      this->emitError("buildBlas: vkMapMemory (index staging) failed");
      vkDestroyBuffer(this->device, staging, this->allocator);
      vkFreeMemory(this->device, stagingMemory, this->allocator);
      vkDestroyBuffer(this->device, indexStaging, this->allocator);
      vkFreeMemory(this->device, indexStagingMemory, this->allocator);
      return false;
    }
    std::memcpy(imapped, geometry.indices, static_cast<size_t>(indexBytes));
    vkUnmapMemory(this->device, indexStagingMemory);
  }

  VkBufferCopy vertexCopy {};
  vertexCopy.size = vertexBytes;
  vkCmdCopyBuffer(cmd, staging, entry.vertexBuffer, 1, &vertexCopy);
  if (indexed) {
    VkBufferCopy indexCopy {};
    indexCopy.size = indexBytes;
    vkCmdCopyBuffer(cmd, indexStaging, entry.indexBuffer, 1, &indexCopy);
  }
  VkMemoryBarrier copyBarrier {};
  copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  copyBarrier.dstAccessMask =
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &copyBarrier, 0, nullptr, 0, nullptr);

  // The staging buffers are referenced by the copy commands recorded above;
  // destroying them now would invalidate this command buffer.  Defer the
  // destruction until the submission completed (freePendingStagingDestroys).
  this->pendingStagingDestroys.emplace_back(staging, stagingMemory);
  if (indexStaging != VK_NULL_HANDLE) {
    this->pendingStagingDestroys.emplace_back(indexStaging, indexStagingMemory);
  }

  // --- Build the BLAS ----------------------------------------------------
  VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
  triangles.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress =
    this->getDeviceAddress(entry.vertexBuffer);
  triangles.vertexStride = 3 * sizeof(float);
  triangles.maxVertex = entry.vertexCount - 1;
  triangles.indexType =
    indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
  triangles.indexData.deviceAddress =
    indexed ? this->getDeviceAddress(entry.indexBuffer) : 0;

  VkAccelerationStructureGeometryKHR asGeometry {};
  asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeometry.geometry.triangles = triangles;
  asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  const uint32_t maxPrimitives =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (maxPrimitives == 0) return false;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
  buildInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  // ALLOW_UPDATE: lets position-only edits refit this BLAS in place (see
  // refitBlas()) instead of destroying and rebuilding it.
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &asGeometry;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
  sizeInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
    this->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo, &maxPrimitives, &sizeInfo);
  if (!this->createScratchBuffer(sizeInfo.buildScratchSize)) {
    return false;
  }

  if (!this->createDeviceLocalBuffer(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        entry.blasBuffer, entry.blasMemory)) {
    return false;
  }
  VkAccelerationStructureCreateInfoKHR asCI {};
  asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  asCI.buffer = entry.blasBuffer;
  asCI.size = sizeInfo.accelerationStructureSize;
  asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  if (vkCreateAccelerationStructureKHR(this->device, &asCI, this->allocator,
                                       &entry.blas) != VK_SUCCESS) {
    return false;
  }
  // Capture the BLAS device address now.  It is constant for the lifetime of
  // the BLAS, so the per-frame instance collection in buildTlas() reuses it
  // instead of calling vkGetAccelerationStructureDeviceAddressKHR every frame.
  entry.devAddr = 0;
  VkAccelerationStructureDeviceAddressInfoKHR devAddrInfo {};
  devAddrInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  devAddrInfo.accelerationStructure = entry.blas;
  entry.devAddr = vkGetAccelerationStructureDeviceAddressKHR(this->device,
                                                             &devAddrInfo);

  buildInfo.dstAccelerationStructure = entry.blas;
  buildInfo.scratchData.deviceAddress = this->scratchAddress;
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo {};
  rangeInfo.primitiveCount = maxPrimitives;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR * rangeInfos[] = {&rangeInfo};
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);

  VkMemoryBarrier blasBarrier {};
  blasBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  blasBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  blasBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &blasBarrier, 0, nullptr, 0, nullptr);
  return true;
}

bool
SoRTXRenderBackend::refitBlas(RTXCachedGeometry & entry,
                              const SoRenderCommand & command,
                              VkCommandBuffer cmd)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t posStrideFloats = entry.vertexStride / sizeof(float);

  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr,
            "[RTDBG] refitBlas verts=%u idx=%u stride=%u pos=%p\n",
            entry.vertexCount, entry.indexCount, entry.vertexStride,
            static_cast<const void *>(geometry.positions));
  }

  // Upload the new vertex positions into the EXISTING device buffers; the
  // index buffer and topology are unchanged (the refit precondition checked
  // in updateGeometryCache()).
  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(entry.vertexCount) * 3 * sizeof(float);

  // Moved vertices change the object-space flat normals: append a fresh
  // normal-pool record and let updateMaterials() pick up the new offset
  // (the pool is grow-only, matching the rebuild path).
  this->appendTriangleNormals(command, entry);
  std::vector<float> positions(static_cast<size_t>(entry.vertexCount) * 3);
  for (uint32_t i = 0; i < entry.vertexCount; ++i) {
    const float * p =
      geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    positions[static_cast<size_t>(i) * 3 + 0] = p[0];
    positions[static_cast<size_t>(i) * 3 + 1] = p[1];
    positions[static_cast<size_t>(i) * 3 + 2] = p[2];
  }

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createHostVisibleBuffer(vertexBytes,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     staging, stagingMemory)) {
    return false;
  }
  void * mapped = nullptr;
  if (vkMapMemory(this->device, stagingMemory, 0, vertexBytes, 0, &mapped) !=
        VK_SUCCESS ||
      mapped == nullptr) {
    this->emitError("refitBlas: vkMapMemory (vertex staging) failed");
    vkDestroyBuffer(this->device, staging, this->allocator);
    vkFreeMemory(this->device, stagingMemory, this->allocator);
    return false;
  }
  std::memcpy(mapped, positions.data(), static_cast<size_t>(vertexBytes));
  vkUnmapMemory(this->device, stagingMemory);

  VkBufferCopy vertexCopy {};
  vertexCopy.size = vertexBytes;
  vkCmdCopyBuffer(cmd, staging, entry.vertexBuffer, 1, &vertexCopy);
  VkMemoryBarrier copyBarrier {};
  copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  copyBarrier.dstAccessMask =
    VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &copyBarrier, 0, nullptr, 0, nullptr);
  this->pendingStagingDestroys.emplace_back(staging, stagingMemory);

  // --- In-place UPDATE build ---------------------------------------------
  VkAccelerationStructureGeometryTrianglesDataKHR triangles {};
  triangles.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress =
    this->getDeviceAddress(entry.vertexBuffer);
  triangles.vertexStride = 3 * sizeof(float);
  triangles.maxVertex = entry.vertexCount - 1;
  const bool indexed = entry.indexCount > 0;
  triangles.indexType =
    indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
  triangles.indexData.deviceAddress =
    indexed ? this->getDeviceAddress(entry.indexBuffer) : 0;

  VkAccelerationStructureGeometryKHR asGeometry {};
  asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeometry.geometry.triangles = triangles;
  asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

  const uint32_t maxPrimitives =
    indexed ? entry.indexCount / 3 : entry.vertexCount / 3;
  if (maxPrimitives == 0) return false;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
  buildInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
  buildInfo.srcAccelerationStructure = entry.blas;
  buildInfo.dstAccelerationStructure = entry.blas;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &asGeometry;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
  sizeInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
    this->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo, &maxPrimitives, &sizeInfo);
  // createScratchBuffer() is grow-only; the update scratch size is bounded
  // by the build scratch size already allocated.
  if (!this->createScratchBuffer(sizeInfo.buildScratchSize)) {
    return false;
  }
  buildInfo.scratchData.deviceAddress = this->scratchAddress;

  VkAccelerationStructureBuildRangeInfoKHR rangeInfo {};
  rangeInfo.primitiveCount = maxPrimitives;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR * rangeInfos[] = {&rangeInfo};
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);

  VkMemoryBarrier blasBarrier {};
  blasBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  blasBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  blasBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &blasBarrier, 0, nullptr, 0, nullptr);

  entry.refitPending = false;
  return true;
}

bool
SoRTXRenderBackend::buildTlas(const SoDrawList & drawlist, VkCommandBuffer cmd)
{
  // Collect instance data for every cached geometry command (opaque only).
  // instanceCustomIndex is the draw-list command index so the closest-hit
  // shader can index the material buffer with gl_InstanceCustomIndexEXT.
  // Reuse a grow-only scratch vector instead of reallocating each frame.
  std::vector<VkAccelerationStructureInstanceKHR> & instances =
    this->instanceScratch;
  instances.clear();
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_TRANSPARENT ||
        command.pass == SO_RENDERPASS_OVERLAY) continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    const RTXCachedGeometry & entry = this->geometryCache[found->second];
    if (entry.blas == VK_NULL_HANDLE) continue;
    // Skip singular model matrices (see updateGeometryCache): a degenerate
    // command must never enter the TLAS or it shadows/traces as a phantom.
    {
      const SbMatrix & m = command.modelMatrix;
      const double det =
        static_cast<double>(m[0][0]) * (static_cast<double>(m[1][1]) * m[2][2] -
                                        static_cast<double>(m[1][2]) * m[2][1]) -
        static_cast<double>(m[0][1]) * (static_cast<double>(m[1][0]) * m[2][2] -
                                        static_cast<double>(m[1][2]) * m[2][0]) +
        static_cast<double>(m[0][2]) * (static_cast<double>(m[1][0]) * m[2][1] -
                                        static_cast<double>(m[1][1]) * m[2][0]);
      if (det > -1e-9 && det < 1e-9) {
        continue;
      }
    }

    VkAccelerationStructureInstanceKHR instance {};
    // SbMatrix uses the row-vector convention (translation in row 3,
    // v*M), while VkTransformMatrixKHR applies to a column vector
    // (M*v, translation in column 3).  The transform must be
    // transposed so the instance's object-to-world matrix matches the
    // matrix the producer baked.  Copying r,c (not c,r) drops every
    // translation: m[3][0..2] (row 3) would land in matrix[0..2][3] as 0,
    // tracing every offset instance at the origin (the displaced
    // emissive-cube phantom).
    const SbMatrix & m = command.modelMatrix;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        instance.transform.matrix[r][c] = m[c][r];
      }
    }
    instance.instanceCustomIndex = static_cast<uint32_t>(i);
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = 0;
    // The BLAS device address is stable for the BLAS lifetime and was
    // captured at build time, so querying it here every frame is wasted work.
    // Fall back to a query only if the cached address is missing.
    if (entry.devAddr) {
      instance.accelerationStructureReference = entry.devAddr;
    }
    else {
      VkAccelerationStructureDeviceAddressInfoKHR addrInfo {};
      addrInfo.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      addrInfo.accelerationStructure = entry.blas;
      instance.accelerationStructureReference =
        vkGetAccelerationStructureDeviceAddressKHR(this->device, &addrInfo);
    }
    instances.push_back(instance);
  }
  this->instanceCount = static_cast<uint32_t>(instances.size());

  if (getenv("FC_VULKAN_RT_DEBUG")) {
    static uint32_t debugFrame = 0;
    if ((debugFrame++ % 120) == 0) {
      fprintf(stderr,
              "[RTDBG] buildTlas: drawlist commands=%d instances=%zu "
              "cacheEntries=%zu mapEntries=%zu\n",
              drawlist.getNumCommands(), instances.size(),
              this->geometryCache.size(), this->commandToCache.size());
    }
  }

  // Instance buffer (host-visible; rebuilt every frame).  An empty scene
  // still builds a valid empty TLAS so the descriptor references a real
  // acceleration structure (raygen would otherwise read a null one).
  VkDeviceSize instanceBytes = 0;
  if (!instances.empty()) {
    instanceBytes =
      sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
    if (this->instanceBuffer == VK_NULL_HANDLE ||
        this->instanceBufferCapacity < instances.size()) {
      if (this->instanceBuffer != VK_NULL_HANDLE) {
        // Defer: a pending frame may still read the old instance buffer.
        VkDevice device = this->device;
        const VkAllocationCallbacks * allocator = this->allocator;
        const VkBuffer buffer = this->instanceBuffer;
        const VkDeviceMemory memory = this->instanceMemory;
        this->deferDestroy([device, allocator, buffer, memory]() {
          vkDestroyBuffer(device, buffer, allocator);
          vkFreeMemory(device, memory, allocator);
        });
        this->instanceBuffer = VK_NULL_HANDLE;
        this->instanceMemory = VK_NULL_HANDLE;
      }
      if (!this->createHostVisibleBuffer(
            instanceBytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            this->instanceBuffer, this->instanceMemory)) {
        return false;
      }
      this->instanceBufferCapacity = static_cast<uint32_t>(instances.size());
    }
    void * mapped = nullptr;
    if (vkMapMemory(this->device, this->instanceMemory, 0, instanceBytes, 0,
                    &mapped) != VK_SUCCESS) {
      return false;
    }
    std::memcpy(mapped, instances.data(), static_cast<size_t>(instanceBytes));
    vkUnmapMemory(this->device, this->instanceMemory);
  }

  // TLAS build sizes.
  VkAccelerationStructureGeometryInstancesDataKHR instancesData {};
  instancesData.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  instancesData.arrayOfPointers = VK_FALSE;
  if (this->instanceBuffer != VK_NULL_HANDLE) {
    instancesData.data.deviceAddress =
      this->getDeviceAddress(this->instanceBuffer);
  }

  VkAccelerationStructureGeometryKHR geometry {};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances = instancesData;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
  buildInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &geometry;
  buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
  sizeInfo.sType =
    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(
    this->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo, &this->instanceCount, &sizeInfo);
  if (!this->createScratchBuffer(sizeInfo.buildScratchSize)) {
    return false;
  }

  // (Re)create the TLAS when it does not exist or is too small.
  if (this->tlas == VK_NULL_HANDLE ||
      this->tlasSize < sizeInfo.accelerationStructureSize) {
    if (this->tlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR(this->device, this->tlas,
                                        this->allocator);
      vkDestroyBuffer(this->device, this->tlasBuffer, this->allocator);
      vkFreeMemory(this->device, this->tlasMemory, this->allocator);
      this->tlas = VK_NULL_HANDLE;
      this->tlasBuffer = VK_NULL_HANDLE;
      this->tlasMemory = VK_NULL_HANDLE;
    }
    this->tlasSize = sizeInfo.accelerationStructureSize;
    if (!this->createDeviceLocalBuffer(
          this->tlasSize,
          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
          this->tlasBuffer, this->tlasMemory)) {
      return false;
    }
    VkAccelerationStructureCreateInfoKHR asCI {};
    asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = this->tlasBuffer;
    asCI.size = this->tlasSize;
    asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(this->device, &asCI, this->allocator,
                                         &this->tlas) != VK_SUCCESS) {
      return false;
    }
    if (!this->updateDescriptors()) {
      return false;
    }
  }

  buildInfo.dstAccelerationStructure = this->tlas;
  buildInfo.scratchData.deviceAddress = this->scratchAddress;
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo {};
  rangeInfo.primitiveCount = this->instanceCount;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.firstVertex = 0;
  rangeInfo.transformOffset = 0;
  const VkAccelerationStructureBuildRangeInfoKHR * rangeInfos[] = {&rangeInfo};
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, rangeInfos);
  return true;
}

// --- Material buffer ------------------------------------------------------

void
SoRTXRenderBackend::updateMaterials(const SoDrawList & drawlist)
{
  const int count = drawlist.getNumCommands();
  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(count) * sizeof(RTMaterial);
  if (bytes == 0) return;

  if (this->materialBuffer == VK_NULL_HANDLE ||
      bytes > this->materialBufferBytes) {
    if (this->materialBuffer != VK_NULL_HANDLE) {
      // Defer: a pending frame may still read the old material buffer.
      VkDevice device = this->device;
      const VkAllocationCallbacks * allocator = this->allocator;
      const VkBuffer buffer = this->materialBuffer;
      const VkDeviceMemory memory = this->materialMemory;
      void * mapped = this->materialMapped;
      this->deferDestroy([device, allocator, buffer, memory, mapped]() {
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, allocator);
        vkFreeMemory(device, memory, allocator);
      });
      this->materialBuffer = VK_NULL_HANDLE;
      this->materialMemory = VK_NULL_HANDLE;
      this->materialMapped = nullptr;
    }
    if (!this->createHostVisibleBuffer(
          bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          this->materialBuffer, this->materialMemory)) {
      this->emitError("updateMaterials: failed to create material buffer");
      return;
    }
    if (vkMapMemory(this->device, this->materialMemory, 0, bytes, 0,
                    &this->materialMapped) != VK_SUCCESS) {
      this->materialMapped = nullptr;
      return;
    }
    this->materialBufferBytes = bytes;
    if (!this->updateDescriptors()) {
      this->emitError("updateMaterials: descriptor update failed");
      return;
    }
  }
  this->materialCount = static_cast<uint32_t>(count);

  // Cache the PBR/lighting env overrides once for the whole frame instead of
  // calling envFlagEnabled()/getenv() inside the per-command loop below.
  // These flags never change mid-frame; they were recomputed identically for
  // every command on every frame.
  this->rtPbrEnabled = envFlagEnabled("FC_VULKAN_RT_PBR");
  {
    const char * neeEnv = getenv("FC_VULKAN_PT_NEE");
    this->rtNeeEnabled =
      (neeEnv == nullptr) ? true : envFlagEnabled("FC_VULKAN_PT_NEE");
    const char * misEnv = getenv("FC_VULKAN_PT_MIS");
    this->rtMisEnabled =
      (misEnv == nullptr) ? true : envFlagEnabled("FC_VULKAN_PT_MIS");
  }
  this->rtMetalOverride = false;
  this->rtRoughOverride = false;
  this->rtMetalValue = 0.0f;
  this->rtRoughValue = 0.0f;
  if (const char * metalEnv = getenv("FC_VULKAN_RT_METAL")) {
    this->rtMetalOverride = true;
    this->rtMetalValue = strtof(metalEnv, nullptr);
  }
  if (const char * roughEnv = getenv("FC_VULKAN_RT_ROUGH")) {
    this->rtRoughOverride = true;
    this->rtRoughValue = strtof(roughEnv, nullptr);
  }

  // Reuse a grow-only scratch buffer instead of reallocating a fresh
  // std::vector<RTMaterial> from the heap every frame.
  std::vector<RTMaterial> & materials = this->materialScratch;
  materials.resize(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const SoMaterialData & material = command.material;
    RTMaterial & out = materials[static_cast<size_t>(i)];
    std::memset(&out, 0, sizeof(out));
    out.diffuse[0] = material.diffuse[0];
    out.diffuse[1] = material.diffuse[1];
    out.diffuse[2] = material.diffuse[2];
    out.diffuse[3] = material.diffuse[3];
    out.specular[0] = material.specular[0];
    out.specular[1] = material.specular[1];
    out.specular[2] = material.specular[2];
    out.specular[3] = 1.0f;
    out.emissive[0] = material.emissive[0];
    out.emissive[1] = material.emissive[1];
    out.emissive[2] = material.emissive[2];
    out.emissive[3] = 1.0f;
    out.params[0] = material.shininess;
    out.params[1] = material.twoSidedLighting ? 1.0f : 0.0f;
    out.params[3] =
      material.shadingModel == SO_SHADING_LEGACY_GOURAUD ? 1.0f : 0.0f;

    // Offset of this command's triangle normals in the normal pool (set by
    // appendTriangleNormals() during the BLAS build).
    const auto cacheFound = this->commandToCache.find(&command);
    if (cacheFound != this->commandToCache.end()) {
      const RTXCachedGeometry & entry = this->geometryCache[cacheFound->second];
      out.triangleData[0] = static_cast<float>(entry.normalPoolOffset);
      out.triangleData[1] = static_cast<float>(entry.normalCount);
      out.triangleData[2] = static_cast<float>(entry.neePoolOffset);
      out.triangleData[3] = static_cast<float>(entry.neeCount);
    }

    // Optional PBR (metallic-roughness) parameters.  Off by default so
    // legacy Phong materials keep their exact current appearance; enable
    // with FC_VULKAN_RT_PBR=1 and optionally override the values with
    // FC_VULKAN_RT_METAL / FC_VULKAN_RT_ROUGH.  When disabled, usePbr stays
    // 0 and every shading path above falls back to the legacy model.
    out.pbr[0] = material.metalness;
    out.pbr[1] = material.roughness;
    out.pbr[2] = this->rtPbrEnabled ? 1.0f : 0.0f;
    out.pbr[3] = 0.0f;
    if (this->rtMetalOverride) {
      out.pbr[0] = this->rtMetalValue;
    }
    if (this->rtRoughOverride) {
      out.pbr[1] = this->rtRoughValue;
    }

    const SoLightingData * lighting =
      drawlist.getLighting(command.lightingHandle);
    static const SoLightingData emptyLighting;
    if (!lighting) lighting = &emptyLighting;

    // Fold the scene ambient into the material ambient (matches the raster
    // shader: litColor += ambientLight * materialAmbient).
    out.ambient[0] = lighting->ambient[0] * material.ambient[0];
    out.ambient[1] = lighting->ambient[1] * material.ambient[1];
    out.ambient[2] = lighting->ambient[2] * material.ambient[2];
    out.ambient[3] = 1.0f;

    const int lightCount = std::min<int>(
      static_cast<int>(lighting->lights.size()), MAX_SHADER_LIGHTS);
    out.params[2] = static_cast<float>(lightCount);
    for (int l = 0; l < lightCount; ++l) {
      const SoLightData & light = lighting->lights[static_cast<size_t>(l)];
      float * type = out.lightType + l * 4;
      type[0] = static_cast<float>(light.type);
      type[1] = type[2] = 0.0f;
      type[3] = 1.0f;
      float * color = out.lightColor + l * 4;
      color[0] = light.color[0];
      color[1] = light.color[1];
      color[2] = light.color[2];
      color[3] = 1.0f;
      float * direction = out.lightDirection + l * 4;
      direction[0] = light.direction[0];
      direction[1] = light.direction[1];
      direction[2] = light.direction[2];
      direction[3] = 1.0f;
      float * position = out.lightPosition + l * 4;
      position[0] = light.position[0];
      position[1] = light.position[1];
      position[2] = light.position[2];
      position[3] = 1.0f;
      float * attenuation = out.lightAttenuation + l * 4;
      attenuation[0] = light.attenuation[0];
      attenuation[1] = light.attenuation[1];
      attenuation[2] = light.attenuation[2];
      attenuation[3] = 1.0f;
      float * spot = out.lightSpot + l * 4;
      spot[0] = light.spotCutoffCos;
      spot[1] = light.spotExponent;
      spot[2] = 0.0f;
      spot[3] = 1.0f;
    }
  }
  if (this->materialMapped) {
    std::memcpy(this->materialMapped, materials.data(),
                static_cast<size_t>(bytes));
  }
}
