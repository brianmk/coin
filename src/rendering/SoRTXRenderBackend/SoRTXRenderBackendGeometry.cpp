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
#include <thread>
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
  const int batch = this->pendingDestroys.index();
  if (waitForQueue && this->queue != VK_NULL_HANDLE &&
      !this->pendingDestroys.batch(batch).empty()) {
    vkQueueWaitIdle(this->queue);
  }
  this->pendingDestroys.setIndex(batch ^ 1);
  this->pendingDestroys.flushAt(batch);
}

void
SoRTXRenderBackend::deferDestroy(std::function<void()> && fn)
{
  this->pendingDestroys.defer(std::move(fn));
}

bool
SoRTXRenderBackend::ensureCompactQueryPool()
{
  if (this->compactFence == VK_NULL_HANDLE) {
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(this->device, &fci, this->allocator,
                      &this->compactFence) != VK_SUCCESS) {
      return false;
    }
  }
  if (this->compactQueryPool == VK_NULL_HANDLE) {
    VkQueryPoolCreateInfo qpci {};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qpci.queryCount = 1;
    if (vkCreateQueryPool(this->device, &qpci, this->allocator,
                          &this->compactQueryPool) != VK_SUCCESS) {
      return false;
    }
  }
  return true;
}

// Copy a BLAS into a freshly sized buffer that holds only its compacted
// footprint, then swap the entry to the compacted AS and defer-destroy the
// original (a pending TLAS may still reference the original address this
// frame).  Returns true when compaction ran (including when it determined
// there was nothing to save).
bool
SoRTXRenderBackend::compactBlas(RTXCachedGeometry & entry)
{
  if (entry.blas == VK_NULL_HANDLE) return true;
  if (!this->ensureCompactQueryPool()) return false;

  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) return false;
  vkCmdResetQueryPool(cmd, this->compactQueryPool, 0, 1);
  const VkAccelerationStructureKHR ases[] = {entry.blas};
  this->vkCmdWriteAccelerationStructuresPropertiesKHR(
    cmd, 1, ases, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
    this->compactQueryPool, 0);
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  if (vkQueueSubmit(this->queue, 1, &si, this->compactFence) != VK_SUCCESS) {
    return false;
  }
  if (vkWaitForFences(this->device, 1, &this->compactFence, VK_TRUE,
                      UINT64_MAX) != VK_SUCCESS) {
    return false;
  }
  vkResetFences(this->device, 1, &this->compactFence);
  const VkDeviceSize origSize = entry.blasSize;
  VkDeviceSize compactSize = 0;
  if (vkGetQueryPoolResults(this->device, this->compactQueryPool, 0, 1,
                            sizeof(compactSize), &compactSize, sizeof(compactSize),
                            VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
    return false;
  }
  if (compactSize == 0 || entry.blasSize == 0 ||
      compactSize >= entry.blasSize) {
    entry.compacted = true;  // nothing to save; stop asking
    entry.wantsCompact = false;
    if (getenv("FC_VULKAN_RT_DEBUG")) {
      fprintf(stderr, "[RTDBG] compact size=%llu -> %llu saved=0\n",
              static_cast<unsigned long long>(origSize),
              static_cast<unsigned long long>(compactSize));
    }
    return true;
  }

  VkBuffer cBuf = VK_NULL_HANDLE;
  VkDeviceMemory cMem = VK_NULL_HANDLE;
  if (!this->createDeviceLocalBuffer(
        compactSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        cBuf, cMem)) {
    return false;
  }
  VkAccelerationStructureCreateInfoKHR asCI {};
  asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  asCI.buffer = cBuf;
  asCI.size = compactSize;
  asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  VkAccelerationStructureKHR cAs = VK_NULL_HANDLE;
  if (vkCreateAccelerationStructureKHR(this->device, &asCI, this->allocator,
                                       &cAs) != VK_SUCCESS ||
      cAs == VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, cBuf, this->allocator);
    vkFreeMemory(this->device, cMem, this->allocator);
    return false;
  }

  VkCommandBuffer cmd2 = this->beginTransientCommandBuffer();
  bool ok = false;
  if (cmd2 != VK_NULL_HANDLE) {
    VkCopyAccelerationStructureInfoKHR copyInfo {};
    copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
    copyInfo.src = entry.blas;
    copyInfo.dst = cAs;
    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
    this->vkCmdCopyAccelerationStructureKHR(cmd2, &copyInfo);
    vkEndCommandBuffer(cmd2);
    VkSubmitInfo si2 {};
    si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si2.commandBufferCount = 1;
    si2.pCommandBuffers = &cmd2;
    if (vkQueueSubmit(this->queue, 1, &si2, this->compactFence) == VK_SUCCESS) {
      ok = vkWaitForFences(this->device, 1, &this->compactFence, VK_TRUE,
                           UINT64_MAX) == VK_SUCCESS;
      if (ok) {
        vkResetFences(this->device, 1, &this->compactFence);
      }
    }
  }

  if (!ok) {
    vkDestroyAccelerationStructureKHR(this->device, cAs, this->allocator);
    vkDestroyBuffer(this->device, cBuf, this->allocator);
    vkFreeMemory(this->device, cMem, this->allocator);
    return false;
  }

  // Swap to the compacted AS; defer-destroy the original (a pending TLAS may
  // reference the old address until the frame that used it completes).
  VkAccelerationStructureKHR oldAs = entry.blas;
  VkBuffer oldBuf = entry.blasBuffer;
  VkDeviceMemory oldMem = entry.blasMemory;
  entry.blas = cAs;
  entry.blasBuffer = cBuf;
  entry.blasMemory = cMem;
  entry.blasSize = compactSize;
  VkAccelerationStructureDeviceAddressInfoKHR ai {};
  ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  ai.accelerationStructure = entry.blas;
  entry.devAddr = vkGetAccelerationStructureDeviceAddressKHR(this->device, &ai);
  entry.compacted = true;
  entry.wantsCompact = false;
  // The TLAS instances reference the BLAS device address, which just changed.
  // A static scene does NOT rebuild the TLAS (asDirty == false), so without
  // this the next frame reuses the previous TLAS that points at the old
  // (pre-compaction) BLAS address -- while the old BLAS is freed by the
  // deferred destroy at the start of that very frame.  That use-after-free
  // drove the driver to VK_ERROR_DEVICE_LOST on the frame after compaction.
  // Force the next frame's asDirty so buildTlas() re-points the TLAS at the
  // compacted BLASes before the old ones are released.  (asTransformChanged is
  // consumed by recordAccelerationStructures to compute asDirty and is the
  // "instance set changed" signal, which is exactly what this is.)
  this->asTransformChanged = true;
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr, "[RTDBG] compact size=%llu -> %llu saved=1\n",
            static_cast<unsigned long long>(origSize),
            static_cast<unsigned long long>(compactSize));
  }
  VkDevice device = this->device;
  const VkAllocationCallbacks * alloc = this->allocator;
  const PFN_vkDestroyAccelerationStructureKHR vkDestroyAS =
    this->vkDestroyAccelerationStructureKHR;
  this->deferDestroy([device, alloc, vkDestroyAS, oldAs, oldBuf, oldMem]() {
    if (oldAs != VK_NULL_HANDLE) {
      vkDestroyAS(device, oldAs, alloc);
    }
    if (oldBuf != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, oldBuf, alloc);
    }
    if (oldMem != VK_NULL_HANDLE) {
      vkFreeMemory(device, oldMem, alloc);
    }
  });
  return true;
}

// Compact every cached BLAS that was built with ALLOW_COMPACTION and has not
// been compacted yet.  Called after the owning frame has fully completed (the
// geometry is quiescent, so the AS contents are stable to copy-shrink).
void
SoRTXRenderBackend::compactPendingBlases()
{
  uint32_t candidates = 0;
  for (RTXCachedGeometry & entry : this->geometryCache) {
    if (entry.wantsCompact && !entry.compacted && entry.blas != VK_NULL_HANDLE) {
      ++candidates;
      this->compactBlas(entry);
    }
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr, "[RTDBG] compactSweep cache=%zu candidates=%u\n",
            this->geometryCache.size(), candidates);
  }
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

    const bool traced = (command.pass != SO_RENDERPASS_OVERLAY);
    if (!traced && getenv("FC_VULKAN_RT_GEO") &&
        geometry.vertexCount == 6 && geometry.indexCount == 0) {
      fprintf(stderr, "[GCR] FR fr=%u OVERLAY vc=6 cmd=%p pos=%p\n", frame,
              static_cast<const void *>(&command),
              static_cast<const void *>(geometry.positions));
    }
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
      // non-traced commands (nav cube, axis cross) that were never traced have
      // no matching entry and are ignored here.  Transparent commands (SO_
      // RENDERPASS_TRANSPARENT) ARE traced now so the path tracer sees the
      // translucent geometry and composites it as thin glass.
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

    // TEMP breadcrumb: per-frame pointer/thread/retained trace for the probe
    // box so we can see whether the geometry pointer is stable across frames
    // and on which thread updateGeometryCache reads it.
    if (getenv("FC_VULKAN_RT_GEO") &&
        geometry.indexCount == 0 &&
        (geometry.vertexCount == 36 || geometry.vertexCount == 6)) {
      const float * tp = geometry.positions;
      const float * tm = &command.modelMatrix[0][0];
      fprintf(stderr,
              "[GCR] FR fr=%u tid=%llx cmd=%p pos=%p ret=%d pass=%d "
              "v0=(%.3f,%.3f,%.3f) m00=%.3f m11=%.3f m22=%.3f "
              "t=(%.3f,%.3f,%.3f)\n",
              frame,
              static_cast<unsigned long long>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
              static_cast<const void *>(&command),
              static_cast<const void *>(tp),
              geometry.retained ? 1 : 0,
              static_cast<int>(command.pass),
              tp[0], tp[1], tp[2],
              tm[0], tm[5], tm[10], tm[12], tm[13], tm[14]);
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

    // A degenerate GEOmetry (all positions collapsed to a point) is the
    // geometry counterpart of the singular-transform guard above: FreeCAD's
    // selection/preselection highlight momentarily swaps a shape's base
    // geometry for a degenerate placeholder, so the base OPAQUE command may be
    // re-recorded with all vertices piled at a single point for a frame or
    // two.  Treating that as a real content change would (a) set cacheChanged,
    // restarting the path-tracing accumulation on every hover ("the lights
    // re-calc"), and (b) refit the BLAS with the collapsed point so a ray
    // transmitting through glass misses the surface and reads black.  Skip it
    // without hitting the content/transform/material hashing: re-stamp the
    // existing cache entry's liveness so it is NOT evicted when the valid
    // geometry returns next frame, and leave cacheChanged untouched so the
    // tracer keeps its settled image through the transient.
    if (geometryDegenerate(geometry, vertexStride)) {
      const auto dFound = this->commandToCache.find(&command);
      if (dFound != this->commandToCache.end()) {
        RTXCachedGeometry & e = this->geometryCache[dFound->second];
        e.cacheGeneration = frame;
        e.commandKey = &command;
      }
      continue;
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
        if (getenv("FC_VULKAN_RT_GEO")) {
          const float * p0 = static_cast<const float *>(geometry.positions);
          fprintf(stderr,
                  "[GCR] CONTENT fr=%u tid=%llx cmd=%p pass=%d vc=%u ic=%u "
                  "stride=%u ret=%d old=%016llx new=%016llx "
                  "vhash %016llx -> %016llx "
                  "ihash %016llx -> %016llx p0=(%.4f,%.4f,%.4f) "
                  "plast=(%.4f,%.4f,%.4f) posKey=%p new=%p idxKey=%p new=%p "
                  "topologyStable=%d\n",
                  frame,
                  static_cast<unsigned long long>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
                  static_cast<const void *>(&command),
                  static_cast<int>(command.pass), geometry.vertexCount,
                  geometry.indexCount, vertexStride,
                  geometry.retained ? 1 : 0,
                  static_cast<unsigned long long>(entry.contentHash),
                  static_cast<unsigned long long>(hash),
                  static_cast<unsigned long long>(entry.vertexHash),
                  static_cast<unsigned long long>(vertexHash),
                  static_cast<unsigned long long>(entry.indexHash),
                  static_cast<unsigned long long>(indexHash),
                  p0[0], p0[1], p0[2],
                  p0[3 * (geometry.vertexCount - 1)],
                  p0[3 * (geometry.vertexCount - 1) + 1],
                  p0[3 * (geometry.vertexCount - 1) + 2],
                  static_cast<const void *>(entry.posKey),
                  static_cast<const void *>(geometry.positions),
                  static_cast<const void *>(entry.idxKey),
                  static_cast<const void *>(geometry.indices),
                  topologyStable ? 1 : 0);
          for (uint32_t vi = 0; vi < geometry.vertexCount; ++vi) {
            fprintf(stderr, "[GCR] V %u (%.4f,%.4f,%.4f) bits=%08x:%08x:%08x\n",
                    vi,
                    p0[3 * vi], p0[3 * vi + 1], p0[3 * vi + 2],
                    std::bit_cast<uint32_t>(p0[3 * vi]),
                    std::bit_cast<uint32_t>(p0[3 * vi + 1]),
                    std::bit_cast<uint32_t>(p0[3 * vi + 2]));
          }
        }
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
        if (getenv("FC_VULKAN_RT_GEO")) {
          fprintf(stderr,
                  "[GCR] NEW fr=%u tid=%llx cmd=%p pass=%d vc=%u ic=%u "
                  "stride=%u ret=%d pos=%p hash=%016llx\n",
                  frame,
                  static_cast<unsigned long long>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
                  static_cast<const void *>(&command),
                  static_cast<int>(command.pass), geometry.vertexCount,
                  geometry.indexCount, vertexStride,
                  geometry.retained ? 1 : 0,
                  static_cast<const void *>(geometry.positions),
                  static_cast<unsigned long long>(hash));
        }
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
        entry.materialHash = hashMaterial(command.material);
        entryPtr = &entry;
      }
    }
    // Instance-transform change detection: a moved object (same geometry)
    // must rebuild the TLAS, but the camera never does.  Only the traced
    // opaque commands reach here, so a selection pass-flip (OPAQUE<->OVERLAY)
    // of an unchanged object does not dirty the TLAS.  The SbMatrix storage
    // is exactly float[4][4], so a 64-byte memcmp against the raw bits
    // last seen is a cheaper, collision-free stand-in for the old FNV
    // transform hash.
    {
      const float * m = &command.modelMatrix[0][0];
      if (std::memcmp(entryPtr->transformBits, m,
                      sizeof(entryPtr->transformBits)) != 0) {
        this->asTransformChanged = true;
        if (getenv("FC_VULKAN_RT_GEO")) {
          fprintf(stderr, "[GCR] TRANSFORM cmd=%p pass=%d vc=%u\n",
                  static_cast<const void *>(&command),
                  static_cast<int>(command.pass), geometry.vertexCount);
        }
        std::memcpy(entryPtr->transformBits, m,
                    sizeof(entryPtr->transformBits));
      }
    }
    // Material-content change detection: a pure material edit (recolor,
    // Transparency change) keeps the geometry content and transform hash
    // above identical, so none of {cacheChanged, asTransformChanged} would
    // fire -- and because the path tracer's scene-change signal IS
    // cacheChanged (see updatePathTracingState), a converged run would keep
    // its stale accumulation until the camera moved.  The material buffer is
    // still re-uploaded each frame with the new values, so this only needs to
    // flag the scene change so the tracer restarts; no cache entry is rebuilt.
    {
      const uint64_t mh = hashMaterial(command.material);
      if (mh != 0 && mh != entryPtr->materialHash) {
        this->cacheChanged = true;
        if (getenv("FC_VULKAN_RT_GEO")) {
          fprintf(stderr, "[GCR] MATERIAL cmd=%p pass=%d vc=%u old=%016llx new=%016llx\n",
                  static_cast<const void *>(&command),
                  static_cast<int>(command.pass), geometry.vertexCount,
                  static_cast<unsigned long long>(entryPtr->materialHash),
                  static_cast<unsigned long long>(mh));
        }
        entryPtr->materialHash = mh;
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
    if (getenv("FC_VULKAN_RT_GEO")) {
      size_t nstale = 0;
      for (size_t i = 0; i < this->geometryCache.size(); ++i) {
        if (this->geometryCache[i].cacheGeneration != frame) {
          ++nstale;
          fprintf(stderr, "[GCR] EVICT idx=%zu vc=%u ic=%u\n", i,
                  this->geometryCache[i].vertexCount,
                  this->geometryCache[i].indexCount);
        }
      }
      fprintf(stderr, "[GCR] EVICT total=%zu\n", nstale);
    }
    // Removing geometry is a scene change: the TLAS must be rebuilt (not
    // MODE_UPDATE refit) so the instance set reflects the liveness above,
    // and the accumulation/history reset.
    this->cacheChanged = true;
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

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BLACK_DEBUG")) {
    static int geoFrame = 0;
    int tri = 0, triTraced = 0, triOverlay = 0, triTrans = 0;
    uint64_t vertsTraced = 0, vertsAll = 0;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      const SoRenderCommand & c = drawlist.getCommand(i);
      if (c.geometry.topology != SO_TOPOLOGY_TRIANGLES) continue;
      tri++;
      if (c.geometry.vertexCount) vertsAll += c.geometry.vertexCount;
      const bool traced = (c.pass != SO_RENDERPASS_OVERLAY);
      if (traced) { triTraced++; vertsTraced += c.geometry.vertexCount; }
      else if (c.pass == SO_RENDERPASS_OVERLAY) triOverlay++;
    }
    int live = 0, blasLive = 0;
    for (const RTXCachedGeometry & e : this->geometryCache) {
      if (e.cacheGeneration == frame) live++;
      if (e.blas != VK_NULL_HANDLE) blasLive++;
    }
    fprintf(stderr,
            "[GEOC] frame=%d cmds=%d tri=%d traced=%d(verts=%llu) "
            "overlay=%d trans=%d allVerts=%llu cache=%d blas=%d cacheChanged=%d\n",
            geoFrame++, drawlist.getNumCommands(), tri, triTraced,
            static_cast<unsigned long long>(vertsTraced), triOverlay, triTrans,
            static_cast<unsigned long long>(vertsAll), live, blasLive,
            static_cast<int>(this->cacheChanged));
  }
}

// IEEE 754 single -> binary16, round-to-nearest-even.  Used to pack BLAS
// positions to R16G16B16_SFLOAT (FC_VULKAN_AS_PACK).  Positions are finite
// object-space coordinates; NaN/Inf are clamped to the half Inf sentinel.
static uint16_t
floatToHalf(float f)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t expField = (bits >> 23) & 0xffu;
  uint32_t mant = bits & 0x7fffffu;
  if (expField == 0xffu) {
    return static_cast<uint16_t>(sign | 0x7c00u);  // NaN/Inf
  }
  int32_t exp = static_cast<int32_t>(expField) - 127 + 15;
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);  // overflow to Inf
  }
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);  // underflow to 0
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t halfMant = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 0x400u >> (shift - 1u);
    if (rem > halfway || (rem == halfway && (halfMant & 1u))) {
      ++halfMant;
    }
    return static_cast<uint16_t>(sign | halfMant);
  }
  uint32_t halfMant = mant >> 13;
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1u))) {
    ++halfMant;
  }
  if (halfMant == 0x400u) {
    halfMant = 0;
    ++exp;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10u) |
                               halfMant);
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

  // Position-only vertex buffer for the BLAS.  Optionally packed to 16-bit
  // half floats (FC_VULKAN_AS_PACK) when the object positions fit the half
  // range: halves AS memory and traversal cost on static geometry.  The 32-bit
  // path is the default and is used whenever the gate is off or coords would
  // overflow half precision.
  const bool packEnabled = getenv("FC_VULKAN_AS_PACK") != nullptr;
  std::vector<float> positions(static_cast<size_t>(entry.vertexCount) * 3);
  float pMin[3] = {1e30f, 1e30f, 1e30f};
  float pMax[3] = {-1e30f, -1e30f, -1e30f};
  for (uint32_t i = 0; i < entry.vertexCount; ++i) {
    const float * p =
      geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    positions[static_cast<size_t>(i) * 3 + 0] = p[0];
    positions[static_cast<size_t>(i) * 3 + 1] = p[1];
    positions[static_cast<size_t>(i) * 3 + 2] = p[2];
    if (p[0] < pMin[0]) pMin[0] = p[0];
    if (p[1] < pMin[1]) pMin[1] = p[1];
    if (p[2] < pMin[2]) pMin[2] = p[2];
    if (p[0] > pMax[0]) pMax[0] = p[0];
    if (p[1] > pMax[1]) pMax[1] = p[1];
    if (p[2] > pMax[2]) pMax[2] = p[2];
  }
  for (int a = 0; a < 3; ++a) {
    entry.objectMin[a] = pMin[a];
    entry.objectMax[a] = pMax[a];
  }
  bool fitHalf = true;
  for (int a = 0; a < 3; ++a) {
    if (std::fabs(pMin[a]) > 60000.0f || std::fabs(pMax[a]) > 60000.0f) {
      fitHalf = false;
    }
  }
  const bool useHalf = packEnabled && fitHalf;
  entry.blasVertexFormat =
    useHalf ? VK_FORMAT_R16G16B16_SFLOAT : VK_FORMAT_R32G32B32_SFLOAT;
  entry.blasVertexStride = useHalf ? 6 : 12;
  std::vector<uint16_t> packedHalf;
  const void * vertexSrc = nullptr;
  if (useHalf) {
    packedHalf.resize(static_cast<size_t>(entry.vertexCount) * 3);
    for (size_t i = 0; i < positions.size(); ++i) {
      packedHalf[i] = floatToHalf(positions[i]);
    }
    vertexSrc = packedHalf.data();
  }
  else {
    vertexSrc = positions.data();
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr, "[RTDBG] blasFmt build=1 packed=%d stride=%u fmt=0x%x\n",
            useHalf ? 1 : 0, entry.blasVertexStride,
            static_cast<unsigned>(entry.blasVertexFormat));
  }
  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(entry.vertexCount) * entry.blasVertexStride;
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
  std::memcpy(mapped, vertexSrc, static_cast<size_t>(vertexBytes));
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
  triangles.vertexFormat = entry.blasVertexFormat;
  triangles.vertexData.deviceAddress =
    this->getDeviceAddress(entry.vertexBuffer);
  triangles.vertexStride = entry.blasVertexStride;
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
  // refitBlas()) instead of destroying and rebuilding it.  ALLOW_COMPACTION
  // (FC_VULKAN_AS_COMPACT) lets a later pass shrink the AS residency copy.
  // They are mutually exclusive: NVIDIA's compaction docs note that
  // ALLOW_UPDATE "must leave room for updated triangles" and that
  // PREFER_FAST_TRACE "uses its own compaction method and results can differ
  // from ALLOW_COMPACTION".  Building a BLAS with ALLOW_UPDATE + PREFER_FAST_TRACE
  // and then COMPACT-copying it into the queried compacted-size buffer produced
  // a malformed AS whose first use drove the driver to VK_ERROR_DEVICE_LOST.
  // So when compaction is requested the BLAS is built with ALLOW_COMPACTION
  // ALONE (the NVIDIA "max compaction" recipe); a compacted BLAS loses its
  // ALLOW_UPDATE refit capability, and recordAccelerationStructures already
  // rebuilds (instead of refits) any compacted entry that needs a position fix.
  const bool compactGate = getenv("FC_VULKAN_AS_COMPACT") != nullptr;
  if (compactGate) {
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    entry.wantsCompact = true;
    entry.compacted = false;
  }
  else {
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  }
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
  entry.blasSize = sizeInfo.accelerationStructureSize;
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
  // in updateGeometryCache()).  The byte size and packing must match the
  // format the BLAS was originally built with (entry.blasVertexFormat).
  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(entry.vertexCount) * entry.blasVertexStride;

  // Moved vertices change the object-space flat normals: append a fresh
  // normal-pool record and let updateMaterials() pick up the new offset
  // (the pool is grow-only, matching the rebuild path).
  this->appendTriangleNormals(command, entry);
  std::vector<float> positions(static_cast<size_t>(entry.vertexCount) * 3);
  float pMin[3] = {1e30f, 1e30f, 1e30f};
  float pMax[3] = {-1e30f, -1e30f, -1e30f};
  for (uint32_t i = 0; i < entry.vertexCount; ++i) {
    const float * p =
      geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    const float px = p[0], py = p[1], pz = p[2];
    positions[static_cast<size_t>(i) * 3 + 0] = px;
    positions[static_cast<size_t>(i) * 3 + 1] = py;
    positions[static_cast<size_t>(i) * 3 + 2] = pz;
    if (px < pMin[0]) pMin[0] = px;
    if (py < pMin[1]) pMin[1] = py;
    if (pz < pMin[2]) pMin[2] = pz;
    if (px > pMax[0]) pMax[0] = px;
    if (py > pMax[1]) pMax[1] = py;
    if (pz > pMax[2]) pMax[2] = pz;
  }
  for (int a = 0; a < 3; ++a) {
    entry.objectMin[a] = pMin[a];
    entry.objectMax[a] = pMax[a];
  }
  std::vector<uint16_t> packedHalf;
  const void * vertexSrc = nullptr;
  if (entry.blasVertexFormat == VK_FORMAT_R16G16B16_SFLOAT) {
    packedHalf.resize(static_cast<size_t>(entry.vertexCount) * 3);
    for (size_t i = 0; i < positions.size(); ++i) {
      packedHalf[i] = floatToHalf(positions[i]);
    }
    vertexSrc = packedHalf.data();
  }
  else {
    vertexSrc = positions.data();
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
  std::memcpy(mapped, vertexSrc, static_cast<size_t>(vertexBytes));
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
  triangles.vertexFormat = entry.blasVertexFormat;
  triangles.vertexData.deviceAddress =
    this->getDeviceAddress(entry.vertexBuffer);
  triangles.vertexStride = entry.blasVertexStride;
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

// A world-space AABB projects to a sub-N-pixel footprint exactly when the
// whole box is in front of the camera and its clip-space extent (in pixels)
// is below the cull threshold.  A box straddling the near plane (any corner
// behind the camera) is never culled: it is either adjacent to / crossing
// the camera, so assuming it is visible is the conservative choice and avoids
// a pop-in.  Uses the same row-vector "p * (viewMatrix * projMatrix)" clip
// the frame block and ray shaders derive from, so the NDC bounds match the
// pixels the object would actually cover.
namespace {
bool projectFootprintSubPixel(const float mn[3], const float mx[3],
                              const SbMatrix & viewProj, float vw, float vh,
                              float cullPixels)
{
  const float corners[8][3] = {
    {mn[0], mn[1], mn[2]}, {mn[0], mn[1], mx[2]},
    {mn[0], mx[1], mn[2]}, {mn[0], mx[1], mx[2]},
    {mx[0], mn[1], mn[2]}, {mx[0], mn[1], mx[2]},
    {mx[0], mx[1], mn[2]}, {mx[0], mx[1], mx[2]},
  };
  float minNx = 1e30f, maxNx = -1e30f, minNy = 1e30f, maxNy = -1e30f;
  for (int k = 0; k < 8; ++k) {
    const float x = corners[k][0], y = corners[k][1], z = corners[k][2];
    // Row-vector: out[j] = sum_r v[r] * M[r][j]  (v[3] == 1).
    const float cw =
      x * viewProj[0][3] + y * viewProj[1][3] + z * viewProj[2][3] +
      viewProj[3][3];
    if (cw <= 1e-6f) return false;  // behind / on the near plane -> keep
    const float cx =
      x * viewProj[0][0] + y * viewProj[1][0] + z * viewProj[2][0] +
      viewProj[3][0];
    const float cy =
      x * viewProj[0][1] + y * viewProj[1][1] + z * viewProj[2][1] +
      viewProj[3][1];
    const float nx = cx / cw;
    const float ny = cy / cw;
    if (nx < minNx) minNx = nx;
    if (nx > maxNx) maxNx = nx;
    if (ny < minNy) minNy = ny;
    if (ny > maxNy) maxNy = ny;
  }
  const float px = (maxNx - minNx) * vw * 0.5f;
  const float py = (maxNy - minNy) * vh * 0.5f;
  return px < cullPixels && py < cullPixels;
}
} // namespace

bool
SoRTXRenderBackend::buildTlas(const SoDrawList & drawlist,
                              const SoRenderParams & params,
                              VkCommandBuffer cmd)
{
  // Collect instance data for every cached geometry command (opaque only).
  // instanceCustomIndex is the draw-list command index so the closest-hit
  // shader can index the material buffer with gl_InstanceCustomIndexEXT.
  // Reuse a grow-only scratch vector instead of reallocating each frame.
  std::vector<VkAccelerationStructureInstanceKHR> & instances =
    this->instanceScratch;
  instances.clear();

  // Sub-pixel instance culling (FC_VULKAN_TLAS_CULL, off unless requested so
  // the default trace path is unchanged).  Culling drops instances whose
  // projected footprint is below FC_VULKAN_TLAS_PIX pixels, which for a CAD
  // viewport dominated by far/small meshes grades the TLAS traversal cost.
  // The instance set changing is tracked so the refit/UPDATE path is never
  // reused across a differing set (a MODE_UPDATE TLAS keeps stale instances).
  this->statTlasCulled = 0;
  const bool cullEnabled = getenv("FC_VULKAN_TLAS_CULL") != nullptr;
  float cullPixels = 1.0f;
  if (const char * s = getenv("FC_VULKAN_TLAS_PIX")) {
    cullPixels = static_cast<float>(std::atof(s));
  }
  if (cullPixels <= 0.0f) cullPixels = 1.0f;
  float vw = 0.0f, vh = 0.0f;
  SbMatrix viewProj;
  if (cullEnabled) {
    const SbVec2s & vpSize = params.viewport.getViewportSizePixels();
    vw = static_cast<float>(vpSize[0]);
    vh = static_cast<float>(vpSize[1]);
    // viewMatrix (world->view) then projMatrix, composing row-vector style.
    viewProj = params.viewMatrix * params.projMatrix;
  }
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
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

    // Sub-pixel cull: transform the object-space AABB into world space and
    // project it; skip the instance if it cannot cover a full pixel.  This
    // runs before the instance struct is built, and a culled instance is
    // simply absent from the TLAS (the material buffer / custom index for the
    // surviving instances is untouched, so ray hits stay correctly indexed).
    if (cullEnabled) {
      const SbMatrix & wm = command.modelMatrix;
      const float mn[3] = {entry.objectMin[0], entry.objectMin[1],
                           entry.objectMin[2]};
      const float mx[3] = {entry.objectMax[0], entry.objectMax[1],
                           entry.objectMax[2]};
      float wmin[3] = {1e30f, 1e30f, 1e30f};
      float wmax[3] = {-1e30f, -1e30f, -1e30f};
      for (int k = 0; k < 8; ++k) {
        const float x = (k & 1) ? mx[0] : mn[0];
        const float y = (k & 2) ? mx[1] : mn[1];
        const float z = (k & 4) ? mx[2] : mn[2];
        // Row-vector: world = (x,y,z,1) * modelMatrix.
        const float wx = x * wm[0][0] + y * wm[1][0] + z * wm[2][0] + wm[3][0];
        const float wy = x * wm[0][1] + y * wm[1][1] + z * wm[2][1] + wm[3][1];
        const float wz = x * wm[0][2] + y * wm[1][2] + z * wm[2][2] + wm[3][2];
        if (wx < wmin[0]) wmin[0] = wx;
        if (wy < wmin[1]) wmin[1] = wy;
        if (wz < wmin[2]) wmin[2] = wz;
        if (wx > wmax[0]) wmax[0] = wx;
        if (wy > wmax[1]) wmax[1] = wy;
        if (wz > wmax[2]) wmax[2] = wz;
      }
      if (projectFootprintSubPixel(wmin, wmax, viewProj, vw, vh, cullPixels)) {
        ++this->statTlasCulled;
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
    // Force-opaque at the instance level: the geometry is opaque triangles
    // (translucency is composited deterministically in the path loop, not via
    // any-hit), so instruct the driver to skip any-hit processing for this
    // instance even if a ray did not set the OPAQUE flag.  This lets traversal
    // use the opaque fast path and avoids any-hit shader invocation overhead
    // on primary/shadow rays.
    instance.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
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
    if (!this->lastTlasDebugLogged || this->lastTlasTotal != instances.size() ||
        this->lastTlasCulled != this->statTlasCulled) {
      fprintf(stderr,
              "[RTDBG] buildTlas: drawlist commands=%d instances=%zu "
              "culled=%u cacheEntries=%zu mapEntries=%zu\n",
              drawlist.getNumCommands(), instances.size(),
              this->statTlasCulled, this->geometryCache.size(),
              this->commandToCache.size());
      this->lastTlasTotal = instances.size();
      this->lastTlasCulled = this->statTlasCulled;
      this->lastTlasDebugLogged = true;
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
  // The TLAS is rebuilt from a host-visible instance buffer every frame, so
  // MODE_BUILD on every instance for pre-built BLASes is the common path.
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
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
    // A freshly created TLAS has no previous instance data to refit, so the
    // first build into it (this frame) must be a full MODE_BUILD.  The refit
    // path below only engages on frames after this one.
    this->tlasBuiltOnce = false;
    this->previousInstanceCount = 0;
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
  // Refit in place when the BLAS set is unchanged (only instance transforms /
  // the camera moved): the TLAS already exists and the instance count is not
  // growing.  MODE_UPDATE reads the previous instance data from
  // srcAccelerationStructure and only re-traces the instance transforms,
  // which is far cheaper than a full rebuild.  A rebuild is forced when
  // geometry content changed (cacheChanged), the instance count grew, or the
  // TLAS was just created this frame.
  const bool useUpdate = this->tlasBuiltOnce &&
    !this->cacheChanged &&
    this->statTlasCulled == 0 &&
    this->instanceCount == this->previousInstanceCount;
  if (useUpdate) {
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = this->tlas;
  }
  this->previousInstanceCount = this->instanceCount;
  this->tlasBuiltOnce = true;
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
SoRTXRenderBackend::setSceneLights(const std::vector<SoLightData> & lights,
                                   const SbVec3f & ambient)
{
  this->sceneLights = lights;
  this->sceneAmbient = ambient;
}

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
      this->deferDestroy([device, allocator, buffer, memory]() {
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
  this->rtPbrEnabled = COIN_VULKAN_ENV_FLAG("FC_VULKAN_RT_PBR");
  {
    const char * neeEnv = getenv("FC_VULKAN_PT_NEE");
    this->rtNeeEnabled =
      (neeEnv == nullptr) ? true : COIN_VULKAN_ENV_FLAG("FC_VULKAN_PT_NEE");
    const char * misEnv = getenv("FC_VULKAN_PT_MIS");
    this->rtMisEnabled =
      (misEnv == nullptr) ? true : COIN_VULKAN_ENV_FLAG("FC_VULKAN_PT_MIS");
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
    // When the GL host pushed an authoritative light set (setSceneLights),
    // use its ambient; otherwise use the per-command IR lighting ambient.
    const SbVec3f & sceneAmbient = this->sceneLights.empty()
      ? lighting->ambient : this->sceneAmbient;
    out.ambient[0] = sceneAmbient[0] * material.ambient[0];
    out.ambient[1] = sceneAmbient[1] * material.ambient[1];
    out.ambient[2] = sceneAmbient[2] * material.ambient[2];
    out.ambient[3] = 1.0f;

    // Light list: the GL-pushed authoritative set (the viewer headlight +
    // document lights) when present, else the per-command IR capture.  The
    // IR capture can drop to zero lights on the retained/replayed path
    // tracer (SoLightElement::getLights goes empty after the first frames),
    // which renders surfaces at ambient-only (near-black).  The pushed set
    // is the reliable source and keeps the headlight view-fixed in eye
    // space (the RT shader converts back through frame.u_viewInverse).
    const std::vector<SoLightData> * lightSource = &lighting->lights;
    if (!this->sceneLights.empty()) {
      lightSource = &this->sceneLights;
    }
    const int lightCount = std::min<int>(
      static_cast<int>(lightSource->size()), MAX_SHADER_LIGHTS);
    out.params[2] = static_cast<float>(lightCount);
    for (int l = 0; l < lightCount; ++l) {
      const SoLightData & light = (*lightSource)[static_cast<size_t>(l)];
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
