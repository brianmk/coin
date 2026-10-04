// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendWideLine.cpp
//
// CPU fallback expansion of wide and/or stippled lines into triangle-list quads.
// Plain wide lines normally expand on the GPU via the instanced vertex shader
// (buildInstancedLineBuffer() / WideLineInstancedVertex.glsl); this path handles
// stippled lines (order-dependent distance), line strips, a missing instance
// buffer, and the COIN_VULKAN_WLINE_CPU override.  Per segment it clip-transforms
// the endpoints, near-plane clips (interpolating the hidden end onto the plane),
// accumulates screen-space pixel distance for glLineStipple, and emits 2 triangles
// into a per-frame host-visible quad buffer.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanDebug.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"

#include <Inventor/SbString.h>
#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include <vk_mem_alloc.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace CoinVulkanDetail;

namespace {

// Below this many segments, dispatch/join cost (~tens of us) outweighs splitting.
constexpr uint32_t kWideLineSplitMinSegments = 2048;

// Row-major SbMat product, identical to the lambda in expandWideLines().
inline void wlineMultiplyMat(const SbMat & a, const SbMat & b, SbMat & out)
{
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] +
        a[r][2] * b[2][c] + a[r][3] * b[3][c];
    }
  }
}

// Clip-space transform with the same Y-flip/depth remap as the visual shader.
inline void wlineTransformPoint(const SbMat & mvp, const float * p, float out[4])
{
  const float x = p[0];
  const float y = p[1];
  const float z = p[2];
  out[0] = mvp[0][0] * x + mvp[0][1] * y + mvp[0][2] * z + mvp[0][3];
  out[1] = -(mvp[1][0] * x + mvp[1][1] * y + mvp[1][2] * z + mvp[1][3]);
  const float cz = mvp[2][0] * x + mvp[2][1] * y + mvp[2][2] * z + mvp[2][3];
  const float cw = mvp[3][0] * x + mvp[3][1] * y + mvp[3][2] * z + mvp[3][3];
  out[2] = 0.5f * cz + 0.5f * cw;
  out[3] = cw;
}

} // namespace


bool
SoVulkanRenderBackend::ensureInstanceModelRingCapacity()
{
  const VkDeviceSize bytes =
    static_cast<VkDeviceSize>(this->maxFramesInFlight) *
    static_cast<VkDeviceSize>(this->uboSlotsPerFrame) * sizeof(float) * 16;
  // uboSlotsPerFrame==0 gives zero bytes; ensureInstanceModelBuffer() clamps to
  // 64, so this is a safety bound, not a steady-state path.
  return this->ensureInstanceModelBuffer(bytes);
}

bool
SoVulkanRenderBackend::ensureInstanceModelBuffer(VkDeviceSize bytes)
{
  if (this->instanceModelBuffer != VK_NULL_HANDLE &&
      this->instanceModelCapacity >= bytes) {
    return true;
  }
  if (this->instanceModelBuffer != VK_NULL_HANDLE) {
    const VkBuffer oldBuffer = this->instanceModelBuffer;
    const VmaAllocation oldMemory = this->instanceModelMemory;
    this->instanceModelBuffer = VK_NULL_HANDLE;
    this->instanceModelMemory = nullptr;
    this->instanceModelMapped = nullptr;
    this->instanceModelCapacity = 0;
    this->deferDestroyBufferMemory(oldBuffer, oldMemory);
  }
  const VkDeviceSize cap = std::max<VkDeviceSize>(bytes, 64u);
  if (!this->createMappedBuffer(cap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                this->instanceModelBuffer,
                                this->instanceModelMemory,
                                &this->instanceModelMapped)) {
    this->emitError("ensureInstanceModelBuffer: buffer create/map failed");
    return false;
  }
  this->instanceModelCapacity = cap;
  return true;
}

bool
SoVulkanRenderBackend::expandWideLines(VulkanCachedCommand & entry,
                                       const SoRenderCommand & command,
                                       const SoRenderParams & params,
                                       const SbMat & proj,
                                       const float lineWidth)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;
  if (!vertexCount) return false;

  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t count = geometry.indexCount && geometry.indices
    ? geometry.indexCount : vertexCount;
  const bool strip = geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  const uint32_t segmentCount =
    strip ? (count > 1 ? count - 1 : 0) : count / 2;
  if (!segmentCount) return false;

  // Diagnostics only on the recording thread: worker prints just interleave for no benefit.
  const bool onOwnerThread =
    std::this_thread::get_id() == this->wlineOwnerThread;
  static thread_local int wlineDiag = 0;
  const bool isSketchCmd = vertexCount >= 900;
  const bool wdiag = onOwnerThread &&
    COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")
    && (isSketchCmd || wlineDiag < 40) && wlineDiag < 200;
  if (wdiag) {
    ++wlineDiag;
    SoVulkanDebug::post("[WLINE2] enter frame=%u cmd=%p verts=%u idx=%u strip=%d segs=%u lw=%.2f\n",
            this->uboFrameIndex, (const void*)&command, vertexCount, count, strip ? 1 : 0,
            static_cast<unsigned>(segmentCount),
            static_cast<double>(lineWidth));
  }

  // MVP: clip = P * V * M * pos.  Regular geometry uses the frame view in
  // params; an overlay command (NaviCube) has its own camera and must use its
  // own view or its wide lines drift off the cube ("edges/axes drift" bug).
  const bool wlineOverlay = (command.pass == SO_RENDERPASS_OVERLAY);
  SbMat model;
  command.modelMatrix.getValue(model);
  auto multiplyMat = [](const SbMat & a, const SbMat & b, SbMat & out) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] +
          a[r][2] * b[2][c] + a[r][3] * b[3][c];
      }
    }
  };
  SbMat view;
  if (wlineOverlay) {
    command.viewMatrix.getValue(view);
  }
  else {
    params.viewMatrix.getValue(view);
  }
  // GLSL reads the row-major SbMat raw memory as column-major, so a *_GL matrix
  // is its transpose and clip = proj_GL*view_GL*model_GL*pos equals
  // mvp = transpose(model*view*proj).  The naive proj*view*model order folds the
  // view translation into clip w, collapsing off-origin geometry onto NDC (0,0).
  SbMat vp;
  multiplyMat(view, proj, vp);
  SbMat wm;
  multiplyMat(model, vp, wm);
  SbMat mvp;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      mvp[r][c] = wm[c][r];
    }
  }
  // Normalize width against the viewport the line is drawn into: an overlay uses
  // its own sub-region, not the frame viewport (else overlay strokes displace).
  SbVec2s viewportSize;
  if (wlineOverlay && command.state.raster.viewportWidth > 0 &&
      command.state.raster.viewportHeight > 0) {
    viewportSize.setValue(static_cast<short>(command.state.raster.viewportWidth),
                          static_cast<short>(command.state.raster.viewportHeight));
  }
  else {
    viewportSize = params.viewport.getViewportSizePixels();
  }
  const float vpWidth = static_cast<float>(viewportSize[0] > 0
    ? viewportSize[0] : 1);
  const float vpHeight = static_cast<float>(viewportSize[1] > 0
    ? viewportSize[1] : 1);

  // ---- Expand-once cache ------------------------------------------------
  // Expansion is the dominant CPU cost for edge-heavy scenes; on a retained draw
  // list with an unchanged camera the quads are byte-identical.  Key the slot on
  // contentHash (in-place edits invalidate) + view/proj/width/viewport; on a match reuse.
  if (entry.wideLineBuffers.size() < this->maxFramesInFlight) {
    entry.wideLineBuffers.resize(this->maxFramesInFlight);
  }
  VulkanCachedCommand::VulkanWideLineBuffer & slot =
    entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
  uint64_t wfp = entry.contentHash;
  auto mixWide = [&wfp](uint32_t bits) {
    wfp ^= bits + 0x9E3779B97F4A7C15ULL + (wfp << 6) + (wfp >> 2);
  };
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      uint32_t bits;
      std::memcpy(&bits, &view[r][c], sizeof(bits));
      mixWide(bits);
      std::memcpy(&bits, &proj[r][c], sizeof(bits));
      mixWide(bits);
    }
  }
  uint32_t lwBits;
  std::memcpy(&lwBits, &lineWidth, sizeof(lwBits));
  mixWide(lwBits);
  mixWide(static_cast<uint32_t>(viewportSize[0]));
  mixWide(static_cast<uint32_t>(viewportSize[1]));
  if (slot.buffer != VK_NULL_HANDLE && slot.size > 0 &&
      slot.expandFingerprint == wfp) {
    entry.wideLineVertexCount = slot.expandVertexCount;
    if (onOwnerThread && COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
      static thread_local uint64_t wlineHits = 0;
      if (++wlineHits % 200 == 0) {
        SoVulkanDebug::post("[WLINE-cache] hits=%llu cmd=%p\n",
                (unsigned long long)wlineHits, (const void*)&command);
      }
    }
    return true;
  }

  // Clip transform with the visual shader's Y-flip/depth remap; the wide-line shader passes it through.
  auto transformPoint = [&mvp](const float * p, float out[4]) {
    const float x = p[0];
    const float y = p[1];
    const float z = p[2];
    const float cx = mvp[0][0] * x + mvp[0][1] * y + mvp[0][2] * z +
      mvp[0][3];
    const float cy = mvp[1][0] * x + mvp[1][1] * y + mvp[1][2] * z +
      mvp[1][3];
    const float cz = mvp[2][0] * x + mvp[2][1] * y + mvp[2][2] * z +
      mvp[2][3];
    const float cw = mvp[3][0] * x + mvp[3][1] * y + mvp[3][2] * z +
      mvp[3][3];
    out[0] = cx;
    out[1] = -cy;
    out[2] = 0.5f * cz + 0.5f * cw;
    out[3] = cw;
  };

  // Per-vertex clip-space cache plus accumulated polyline distance in WINDOW
  // PIXELS: GL stipple is screen-space, so object units would scale when zooming.
  // thread_local scratch: expansion runs on parallel workers, so a shared member would race.
  static thread_local std::vector<float> clipScratch;
  static thread_local std::vector<float> distScratch;
  static thread_local std::vector<float> quadScratch;
  clipScratch.assign(static_cast<size_t>(vertexCount) * 4, 0.0f);
  distScratch.assign(vertexCount, 0.0f);
  float * const clipCache = clipScratch.data();
  float * const distances = distScratch.data();
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t actual = geometry.indices ? geometry.indices[i] : i;
    float * clip = clipCache + static_cast<size_t>(actual) * 4;
    transformPoint(geometry.positions
                   + static_cast<size_t>(actual) * posStrideFloats, clip);
  }
  if (strip) {
    for (uint32_t i = 1; i < count; ++i) {
      const uint32_t previous = geometry.indices
        ? geometry.indices[i - 1] : i - 1;
      const uint32_t current = geometry.indices ? geometry.indices[i] : i;
      const float * c0 = clipCache + static_cast<size_t>(previous) * 4;
      const float * c1 = clipCache + static_cast<size_t>(current) * 4;
      if (c0[3] <= 0.0f || c1[3] <= 0.0f) {
        distances[current] = distances[previous];
        continue;
      }
      const float dx = (c1[0] / c1[3] - c0[0] / c0[3]) * 0.5f * vpWidth;
      const float dy = (c1[1] / c1[3] - c0[1] / c0[3]) * 0.5f * vpHeight;
      distances[current] = distances[previous] +
        std::sqrt(dx * dx + dy * dy);
    }
  }
  else {
    for (uint32_t i = 0; i + 1 < count; i += 2) {
      const uint32_t first = geometry.indices ? geometry.indices[i] : i;
      const uint32_t second = geometry.indices
        ? geometry.indices[i + 1] : i + 1;
      const float * c0 = clipCache + static_cast<size_t>(first) * 4;
      const float * c1 = clipCache + static_cast<size_t>(second) * 4;
      if (c0[3] <= 0.0f || c1[3] <= 0.0f) {
        distances[first] = 0.0f;
        distances[second] = 0.0f;
        continue;
      }
      const float dx = (c1[0] / c1[3] - c0[0] / c0[3]) * 0.5f * vpWidth;
      const float dy = (c1[1] / c1[3] - c0[1] / c0[3]) * 0.5f * vpHeight;
      distances[first] = 0.0f;
      distances[second] = std::sqrt(dx * dx + dy * dy);
    }
  }

  // 6 vertices/segment, 9 floats each: clip pos (4) + color (4) + pixel distance (1).
  const size_t quadFloats = static_cast<size_t>(segmentCount) * 6 * 9;
  quadScratch.assign(quadFloats, 0.0f);
  float * const quads = quadScratch.data();
  size_t outIndex = 0;
  size_t diagSkippedW = 0;
  size_t diagSkippedDeg = 0;
  const float nearEps = 1.0e-5f;
  for (uint32_t s = 0; s < segmentCount; ++s) {
    uint32_t i0;
    uint32_t i1;
    if (strip) {
      i0 = geometry.indices ? geometry.indices[s] : s;
      i1 = geometry.indices ? geometry.indices[s + 1] : s + 1;
    }
    else {
      i0 = geometry.indices ? geometry.indices[s * 2] : s * 2;
      i1 = geometry.indices ? geometry.indices[s * 2 + 1] : s * 2 + 1;
    }
    const float * col0 = geometry.colors
      ? geometry.colors + static_cast<size_t>(i0) * 4 : nullptr;
    const float * col1 = geometry.colors
      ? geometry.colors + static_cast<size_t>(i1) * 4 : nullptr;

    const float * c0 = clipCache + static_cast<size_t>(i0) * 4;
    const float * c1 = clipCache + static_cast<size_t>(i1) * 4;

    // Near-plane clip: visible when w > nearEps AND remapped z (cache[2]) >= 0.
    // Keep the visible half and interpolate the hidden endpoint onto the plane,
    // rather than dropping the whole straddling segment (the old bug).  This also
    // handles behind-the-eye points, which map to cache[2] < 0.
    const float fa = c0[2];
    const float fb = c1[2];
    const bool visible0 = (c0[3] > nearEps) && (fa >= 0.0f);
    const bool visible1 = (c1[3] > nearEps) && (fb >= 0.0f);
    if (!visible0 && !visible1) {
      if (wdiag) ++diagSkippedW;
      continue;
    }

    // tA/tB interpolate along c0 -> c1 for each emitted end (0/1 = original,
    // intermediate = clipped); a plane meets a segment at most once.
    float tA;
    float tB;
    if (visible0 && visible1) {
      tA = 0.0f;
      tB = 1.0f;
    }
    else {
      const float denom = fa - fb;
      const float tclip = (denom != 0.0f) ? fa / denom : 0.0f;
      tA = visible0 ? 0.0f : tclip;
      tB = visible1 ? 1.0f : tclip;
    }

    const float cA[4] = {
      c0[0] + tA * (c1[0] - c0[0]),
      c0[1] + tA * (c1[1] - c0[1]),
      c0[2] + tA * (c1[2] - c0[2]),
      c0[3] + tA * (c1[3] - c0[3]),
    };
    const float cB[4] = {
      c0[0] + tB * (c1[0] - c0[0]),
      c0[1] + tB * (c1[1] - c0[1]),
      c0[2] + tB * (c1[2] - c0[2]),
      c0[3] + tB * (c1[3] - c0[3]),
    };
    // Guard against a degenerate clip that lands on/behind the eye plane.
    if (cA[3] <= nearEps || cB[3] <= nearEps) {
      if (wdiag) ++diagSkippedW;
      continue;
    }

    const float dA = distances[i0] + tA * (distances[i1] - distances[i0]);
    const float dB = distances[i0] + tB * (distances[i1] - distances[i0]);

    const float ndc0x = cA[0] / cA[3];
    const float ndc0y = cA[1] / cA[3];
    const float ndc1x = cB[0] / cB[3];
    const float ndc1y = cB[1] / cB[3];
    const float dx = ndc1x - ndc0x;
    const float dy = ndc1y - ndc0y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.0e-8f) {
      if (wdiag) ++diagSkippedDeg;
      continue;
    }
    // Frustum reject (see expandWideLinesSplitRange): the GPU would clip an off-screen segment.
    {
      const float mx = lineWidth / vpWidth;
      const float my = lineWidth / vpHeight;
      const float minx = std::min(ndc0x, ndc1x) - mx;
      const float maxx = std::max(ndc0x, ndc1x) + mx;
      const float miny = std::min(ndc0y, ndc1y) - my;
      const float maxy = std::max(ndc0y, ndc1y) + my;
      if (maxx < -1.0f || minx > 1.0f || maxy < -1.0f || miny > 1.0f) {
        if (wdiag) ++diagSkippedDeg;
        continue;
      }
    }
    const float dirx = dx / length;
    const float diry = dy / length;
    // Anisotropic NDC offset, mirroring the GL geometry shader (perp * width / vpSize).
    const float offx = -diry * lineWidth / vpWidth;
    const float offy = dirx * lineWidth / vpHeight;

    // Corners [0]=p0+off, [1]=p0-off, [2]=p1+off, [3]=p1-off; triangles
    // (0,1,2),(2,1,3), matching the GL geometry shader's strip order.
    float corners[4][4];
    for (int corner = 0; corner < 4; ++corner) {
      const int endpoint = corner < 2 ? 0 : 1;
      const float sign = (corner % 2 == 0) ? 1.0f : -1.0f;
      const float w = endpoint == 0 ? cA[3] : cB[3];
      corners[corner][0] = (endpoint == 0 ? cA[0] : cB[0]) + sign * offx * w;
      corners[corner][1] = (endpoint == 0 ? cA[1] : cB[1]) + sign * offy * w;
      corners[corner][2] = endpoint == 0 ? cA[2] : cB[2];
      corners[corner][3] = w;
    }
    // Colors interpolated for a clipped endpoint; default opaque white if none supplied.
    const float defaultColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    const float * p0 = col0 ? col0 : defaultColor;
    const float * p1 = col1 ? col1 : defaultColor;
    float colA[4];
    float colB[4];
    for (int i = 0; i < 4; ++i) {
      colA[i] = p0[i] + tA * (p1[i] - p0[i]);
      colB[i] = p0[i] + tB * (p1[i] - p0[i]);
    }

    static const int triOrder[6] = { 0, 1, 2, 2, 1, 3 };
    for (int t = 0; t < 6; ++t) {
      const int corner = triOrder[t];
      const int endpoint = corner < 2 ? 0 : 1;
      const float * col = endpoint == 0 ? colA : colB;
      float * out = quads + outIndex;
      outIndex += 9;
      out[0] = corners[corner][0];
      out[1] = corners[corner][1];
      out[2] = corners[corner][2];
      out[3] = corners[corner][3];
      out[4] = col[0];
      out[5] = col[1];
      out[6] = col[2];
      out[7] = col[3];
      out[8] = endpoint == 0 ? dA : dB;
    }
  }
  if (outIndex < 9) {
    if (wdiag) {
      SoVulkanDebug::post("[WLINE2] FAIL cmd=%p verts=%u segs=%u outIndex=%zu "
                      "skippedW=%zu skippedDeg=%zu\n",
              (const void*)&command, vertexCount,
              static_cast<unsigned>(segmentCount), outIndex,
              diagSkippedW, diagSkippedDeg);
    }
    return false;
  }
  if (wdiag) {
    const SbMatrix vwd = params.viewMatrix;
    const SbMatrix pwd(proj);
    SoVulkanDebug::post("[WLINE2] OK cmd=%p verts=%u segs=%u quads=%zu "
                    "skippedW=%zu skippedDeg=%zu firstSegNDC=(%.3f,%.3f)->(%.3f,%.3f)\n",
            (const void*)&command, vertexCount, static_cast<unsigned>(segmentCount),
            outIndex / 9, diagSkippedW, diagSkippedDeg,
            static_cast<double>(clipCache[0] / clipCache[3]),
            static_cast<double>(clipCache[1] / clipCache[3]),
            static_cast<double>(clipCache[4] / clipCache[7]),
            static_cast<double>(clipCache[5] / clipCache[7]));
    SoVulkanDebug::post("[WLINE2]   viewT=(%.3f,%.3f,%.3f) v11=%.4f v00=%.4f "
                    "isId=%d proj00=%.4f proj11=%.4f proj33=%.4f "
                    "cmdViewT=(%.3f,%.3f,%.3f)\n",
            static_cast<double>(vwd[3][0]), static_cast<double>(vwd[3][1]),
            static_cast<double>(vwd[3][2]),
            static_cast<double>(vwd[1][1]), static_cast<double>(vwd[0][0]),
            (vwd[0][0] == 1.0f && vwd[3][0] == 0.0f && vwd[3][1] == 0.0f) ? 1 : 0,
            static_cast<double>(pwd[0][0]), static_cast<double>(pwd[1][1]),
            static_cast<double>(pwd[3][3]),
            static_cast<double>(command.viewMatrix[3][0]),
            static_cast<double>(command.viewMatrix[3][1]),
            static_cast<double>(command.viewMatrix[3][2]));
  }

  if (onOwnerThread && COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
    static thread_local int distLog = 0;
    if (distLog++ < 3) {
      SbString msg;
      msg.sprintf("[WLINE] verts=%u segs=%u quads=%zu dists:",
                  vertexCount, segmentCount, outIndex / 9);
      for (size_t q = 0; q < outIndex && q < 60; q += 9) {
        msg += SbString().sprintf(" %.1f", static_cast<double>(quads[q + 8]));
      }
      SoVulkanDebug::post("%s", msg.getString());
    }
  }

  const VkDeviceSize needed =
    static_cast<VkDeviceSize>(outIndex) * sizeof(float);
  // Ring of host-visible scratch buffers, one per in-flight slot: a slot is reused
  // only after its fence (beginFrame).  Growth defers old-buffer destruction.
  if (slot.size < needed) {
    if (slot.buffer != VK_NULL_HANDLE || slot.memory != nullptr) {
      const VkBuffer oldBuffer = slot.buffer;
      const VmaAllocation oldMemory = slot.memory;
      slot.buffer = VK_NULL_HANDLE;
      slot.memory = nullptr;
      slot.mapped = nullptr;
      slot.size = 0;
      this->deferDestroyBufferMemory(oldBuffer, oldMemory);
    }
    // Buffer is HOST_VISIBLE | HOST_COHERENT, so a memcpy needs no flush; keeping
    // the mapping alive avoids a vkMap/vkUnmap pair every frame.
    if (!this->createMappedBuffer(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                  slot.buffer, slot.memory, &slot.mapped)) {
      this->emitError("expandWideLines: quad buffer create/map failed");
      slot.size = 0;
      return false;
    }
    slot.size = needed;
    std::memcpy(slot.mapped, quads, static_cast<size_t>(needed));
  }
  else {
    std::memcpy(slot.mapped, quads, static_cast<size_t>(needed));
  }

  slot.expandFingerprint = wfp;
  slot.expandVertexCount = static_cast<uint32_t>(outIndex / 9);
  entry.wideLineVertexCount = static_cast<uint32_t>(outIndex / 9);
  return true;
}

bool
SoVulkanRenderBackend::buildInstancedLineBuffer(VulkanCachedCommand & entry,
                                                const SoRenderCommand & command)
{
  if (!isInstancedWideLine(command)) return false;
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t * const indices = geometry.indices;
  const float * const positions = geometry.positions;
  const float * const colors = geometry.colors;
  if (!positions || geometry.vertexCount == 0) return false;

  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t count = geometry.indexCount && indices
    ? geometry.indexCount : geometry.vertexCount;
  const uint32_t segmentCount = count / 2;
  if (!segmentCount) return false;

  // Endpoints are a pure function of object-space geometry, so rebuild only when
  // contentHash changes; the first call (null buffer) is a miss.
  //
  // LIMITATION: an unhashed command (contentHash==0) edited in place keeps its
  // first endpoints (0==0 short-circuits); only use it for immutable geometry.
  if (entry.instancedLineBuffer != VK_NULL_HANDLE &&
      entry.instancedLineHash == entry.contentHash) {
    return true;
  }

  const VkDeviceSize needed =
    static_cast<VkDeviceSize>(segmentCount) * 16u * sizeof(float);
  if (entry.instancedLineBuffer != VK_NULL_HANDLE ||
      entry.instancedLineMemory != nullptr) {
    const VkBuffer oldBuffer = entry.instancedLineBuffer;
    const VmaAllocation oldMemory = entry.instancedLineMemory;
    entry.instancedLineBuffer = VK_NULL_HANDLE;
    entry.instancedLineMemory = nullptr;
    entry.instancedLineSegmentCount = 0;
    this->deferDestroyBufferMemory(oldBuffer, oldMemory);
  }
  void * mapped = nullptr;
  if (!this->createMappedBuffer(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                entry.instancedLineBuffer,
                                entry.instancedLineMemory, &mapped)) {
    this->emitError(
      "buildInstancedLineBuffer: endpoint buffer create/map failed");
    entry.instancedLineBuffer = VK_NULL_HANDLE;
    entry.instancedLineMemory = nullptr;
    return false;
  }

  // Four vec4 per segment: p0, p1, c0, c1.  Colors default to opaque white; the
  // shader reads them only when the use-vertex-color flag is set.
  float * const out = static_cast<float *>(mapped);
  for (uint32_t s = 0; s < segmentCount; ++s) {
    const uint32_t i0 = indices ? indices[s * 2] : s * 2;
    const uint32_t i1 = indices ? indices[s * 2 + 1] : s * 2 + 1;
    const float * const p0 =
      positions + static_cast<size_t>(i0) * posStrideFloats;
    const float * const p1 =
      positions + static_cast<size_t>(i1) * posStrideFloats;
    const float * const c0 =
      colors ? colors + static_cast<size_t>(i0) * 4 : nullptr;
    const float * const c1 =
      colors ? colors + static_cast<size_t>(i1) * 4 : nullptr;
    float * const o = out + static_cast<size_t>(s) * 16;
    o[0] = p0[0]; o[1] = p0[1]; o[2] = p0[2]; o[3] = 1.0f;
    o[4] = p1[0]; o[5] = p1[1]; o[6] = p1[2]; o[7] = 1.0f;
    if (c0) { o[8] = c0[0]; o[9] = c0[1]; o[10] = c0[2]; o[11] = c0[3]; }
    else { o[8] = o[9] = o[10] = o[11] = 1.0f; }
    if (c1) { o[12] = c1[0]; o[13] = c1[1]; o[14] = c1[2]; o[15] = c1[3]; }
    else { o[12] = o[13] = o[14] = o[15] = 1.0f; }
  }
  entry.instancedLineSegmentCount = segmentCount;
  entry.instancedLineHash = entry.contentHash;
  return true;
}

void
SoVulkanRenderBackend::prepareWideLineBuffers(const SoDrawList & drawlist)
{
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (!isWideLine(command, -1, this->interactionLodActive)) continue;
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0) continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    VulkanCachedCommand & entry = this->gpuCache[found->second];
    if (entry.vertexBuffer == VK_NULL_HANDLE) continue;

    // GPU-instanced wide line: build the static endpoint stream once, skipping CPU quads.
    if (this->buildInstancedLineBuffer(entry, command)) {
      continue;
    }

    const uint32_t count = geometry.indexCount && geometry.indices
      ? geometry.indexCount : geometry.vertexCount;
    const bool strip = geometry.topology == SO_TOPOLOGY_LINE_STRIP;
    const uint32_t segmentCount =
      strip ? (count > 1 ? count - 1 : 0) : count / 2;
    if (!segmentCount) continue;

    if (entry.wideLineBuffers.size() < this->maxFramesInFlight) {
      entry.wideLineBuffers.resize(this->maxFramesInFlight);
    }
    VulkanCachedCommand::VulkanWideLineBuffer & slot =
      entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
    // Worst case: every segment visible (6 verts * 9 floats); expandWideLines() fits this.
    const VkDeviceSize needed =
      static_cast<VkDeviceSize>(segmentCount) * 6u * 9u * sizeof(float);
    if (slot.buffer != VK_NULL_HANDLE && slot.size >= needed) continue;

    if (slot.buffer != VK_NULL_HANDLE || slot.memory != nullptr) {
      const VkBuffer oldBuffer = slot.buffer;
      const VmaAllocation oldMemory = slot.memory;
      slot.buffer = VK_NULL_HANDLE;
      slot.memory = nullptr;
      slot.mapped = nullptr;
      slot.size = 0;
      this->deferDestroyBufferMemory(oldBuffer, oldMemory);
    }
    if (!this->createMappedBuffer(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                  slot.buffer, slot.memory, &slot.mapped)) {
      this->emitError("prepareWideLineBuffers: quad buffer create/map failed");
      slot.size = 0;
      continue;
    }
    slot.size = needed;
    // A (re)allocated buffer holds no valid expansion.
    slot.expandFingerprint = 0;
    slot.expandVertexCount = 0;
  }
}

void
SoVulkanRenderBackend::resolveCommandProj(const SoRenderCommand & command,
                                          const SoRenderParams & params,
                                          bool overlayPass,
                                          SbMat & out) const
{
  // Must match recordDrawCommand()'s projection resolution: the cache key includes
  // the projection, so a divergence forces a re-expand or reuses wrong-matrix quads.
  const bool frameCameraOverlay = isFrameCameraOverlay(command, params);
  if (overlayPass && !frameCameraOverlay) {
    command.projMatrix.getValue(out);
  }
  else {
    std::memcpy(out, this->frameProjFloats, sizeof(float) * 16);
  }
}

bool
SoVulkanRenderBackend::expandWideLinesFor(VulkanCachedCommand & entry,
                                          const SoRenderCommand & command,
                                          const SoRenderParams & params,
                                          bool overlayPass)
{
  SbMat projValue;
  this->resolveCommandProj(command, params, overlayPass, projValue);
  return this->expandWideLines(entry, command, params, projValue,
      std::max(1.0f, command.state.raster.lineWidth) * this->frameDpr);
}

void
SoVulkanRenderBackend::expandWideLinesParallel(const SoDrawList & drawlist,
                                               const SoRenderParams & params)
{
  // Always the recording thread; workers compare against it for single-threaded diagnostics.
  this->wlineOwnerThread = std::this_thread::get_id();
  // Gather drawable wide-line commands, mirroring the record path's guards (findCachedDrawable).
  std::vector<const SoRenderCommand *> & wideLines = this->wlineExpandScratch;
  wideLines.clear();
  std::vector<const SoRenderCommand *> & splitCmds = this->wlineSplitScratch;
  splitCmds.clear();
  const uint32_t W = this->maxRecordWorkers;
  const bool canSplit = W > 1 && !this->recordWorkers.empty() &&
    !COIN_VULKAN_ENV_FLAG("COIN_VULKAN_WLINE_SERIAL");
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (!isWideLine(command, -1, this->interactionLodActive)) continue;
    if (!command.geometry.positions || command.geometry.vertexCount == 0) {
      continue;
    }
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    if (this->gpuCache[found->second].vertexBuffer == VK_NULL_HANDLE) continue;
    // Handled by the GPU-instanced path; nothing to expand on the CPU.
    if (isInstancedWideLine(command) &&
        this->gpuCache[found->second].instancedLineBuffer != VK_NULL_HANDLE) {
      continue;
    }
    // A single dominant non-stippled LINE_LIST (lattice edge set) would all land on
    // one worker, so route it to the segment-range split.  Stippled lines stay serial.
    const SoGeometryDesc & geometry = command.geometry;
    const uint32_t count = geometry.indexCount && geometry.indices
      ? geometry.indexCount : geometry.vertexCount;
    const bool splittable =
      command.pass != SO_RENDERPASS_OVERLAY &&
      geometry.topology == SO_TOPOLOGY_LINES &&
      !isPatternedLine(command) &&
      count / 2 >= kWideLineSplitMinSegments;
    if (canSplit && splittable) {
      splitCmds.push_back(&command);
    }
    else {
      wideLines.push_back(&command);
    }
  }

  // Expand large (split) commands first on the owner thread; each dispatch joins before the next.
  for (const SoRenderCommand * command : splitCmds) {
    const auto found = this->commandToCache.find(command);
    if (found == this->commandToCache.end()) continue;
    VulkanCachedCommand & entry = this->gpuCache[found->second];
    SbMat projValue;
    this->resolveCommandProj(*command, params, false, projValue);
    this->expandWideLinesSplit(entry, *command, params, projValue,
        std::max(1.0f, command->state.raster.lineWidth) * this->frameDpr);
  }

  if (wideLines.empty()) return;

  if (W <= 1 || this->recordWorkers.empty()) {
    // Serial fallback (single core, or the pool failed to build).
    for (const SoRenderCommand * command : wideLines) {
      const auto found = this->commandToCache.find(command);
      VulkanCachedCommand & entry = this->gpuCache[found->second];
      this->expandWideLinesFor(entry, *command, params,
                               command->pass == SO_RENDERPASS_OVERLAY);
    }
    return;
  }

  vkBackendTrace(this->uboFrameIndex, "expandWideLines.dispatch",
                 "cmds=%zu workers=%u", wideLines.size(), W);
  // Round-robin partition balances well without the record path's sort (commands are similar size).
  for (uint32_t w = 0; w < W; ++w) {
    ParallelRecordJob & job = this->recordJobs[w];
    job.expandWideLines = true;
    // Must clear the split phase: a preceding split dispatch left it set, and the
    // worker loop keys its split branch on it (else it reruns the stale range).
    job.wlineSplitPhase = 0;
    job.params = &params;
    job.wideLineCommands.clear();
  }
  for (size_t i = 0; i < wideLines.size(); ++i) {
    this->recordJobs[i % W].wideLineCommands.push_back(wideLines[i]);
  }

  // Reset done count and bump generation under recordMutex so workers' publication
  // and the recorder's predicate check share one lock.
  {
    std::lock_guard<std::mutex> lk(this->recordMutex);
    this->recordDoneCount.store(0);
    ++this->recordJobGeneration;
  }
  this->recordCvSpawn.notify_all();
  // Worker 0 is this (recording) thread.
  {
    const ParallelRecordJob & job = this->recordJobs[0];
    for (const SoRenderCommand * command : job.wideLineCommands) {
      const auto found = this->commandToCache.find(command);
      if (found == this->commandToCache.end()) continue;
      VulkanCachedCommand & entry = this->gpuCache[found->second];
      this->expandWideLinesFor(entry, *command, params,
                               command->pass == SO_RENDERPASS_OVERLAY);
    }
    this->recordJobs[0].ok = true;
  }
  {
    std::unique_lock<std::mutex> lk(this->recordMutex);
    this->recordCvDone.wait(lk, [this] {
      return this->recordDoneCount.load() >= this->maxRecordWorkers - 1;
    });
  }
}

// --- Intra-command split --------------------------------------------------
//
// expandWideLinesParallel() round-robins by command, so one scene-wide edge set
// lands on a single worker; this path partitions that command's segments instead.
// Limited to non-stippled LINE_LIST (no order-dependent state), as large BRep edge
// sets produce.  Output is byte-identical: phase 1 marks, prefix-sum compacts, phase 2 emits.

bool
SoVulkanRenderBackend::expandWideLinesSplit(VulkanCachedCommand & entry,
                                            const SoRenderCommand & command,
                                            const SoRenderParams & params,
                                            const SbMat & proj,
                                            const float lineWidth)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;
  if (!vertexCount) return false;

  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t count = geometry.indexCount && geometry.indices
    ? geometry.indexCount : vertexCount;
  const uint32_t segmentCount = count / 2;
  if (!segmentCount) return false;

  // Main-pass only (overlays go serial), so the frame view/projection apply, as in expandWideLines().
  SbMat model;
  command.modelMatrix.getValue(model);
  SbMat view;
  params.viewMatrix.getValue(view);
  SbMat vp;
  wlineMultiplyMat(view, proj, vp);
  SbMat wm;
  wlineMultiplyMat(model, vp, wm);
  SbMat mvp;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      mvp[r][c] = wm[c][r];
    }
  }

  const SbVec2s viewportSize = params.viewport.getViewportSizePixels();
  const float vpWidth = static_cast<float>(viewportSize[0] > 0
    ? viewportSize[0] : 1);
  const float vpHeight = static_cast<float>(viewportSize[1] > 0
    ? viewportSize[1] : 1);

  if (entry.wideLineBuffers.size() < this->maxFramesInFlight) {
    entry.wideLineBuffers.resize(this->maxFramesInFlight);
  }
  VulkanCachedCommand::VulkanWideLineBuffer & slot =
    entry.wideLineBuffers[this->uboFrameIndex % this->maxFramesInFlight];
  // Same fingerprint as expandWideLines(), so recordDrawCommand()'s inline call is a cache hit.
  uint64_t wfp = entry.contentHash;
  auto mixWide = [&wfp](uint32_t bits) {
    wfp ^= bits + 0x9E3779B97F4A7C15ULL + (wfp << 6) + (wfp >> 2);
  };
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      uint32_t bits;
      std::memcpy(&bits, &view[r][c], sizeof(bits));
      mixWide(bits);
      std::memcpy(&bits, &proj[r][c], sizeof(bits));
      mixWide(bits);
    }
  }
  uint32_t lwBits;
  std::memcpy(&lwBits, &lineWidth, sizeof(lwBits));
  mixWide(lwBits);
  mixWide(static_cast<uint32_t>(viewportSize[0]));
  mixWide(static_cast<uint32_t>(viewportSize[1]));
  if (slot.buffer != VK_NULL_HANDLE && slot.size > 0 &&
      slot.expandFingerprint == wfp) {
    entry.wideLineVertexCount = slot.expandVertexCount;
    return true;
  }

  // Grow-only scratch, sized on the owner thread before any worker reads it.
  const size_t clipFloats = static_cast<size_t>(vertexCount) * 4;
  if (this->wlineSplitClip.size() < clipFloats) {
    this->wlineSplitClip.resize(clipFloats);
  }
  if (this->wlineSplitValid.size() < segmentCount) {
    this->wlineSplitValid.resize(segmentCount);
  }
  if (this->wlineSplitOffsets.size() < segmentCount) {
    this->wlineSplitOffsets.resize(segmentCount);
  }

  WideLineSplitCtx & c = this->wlineSplitCtx;
  c.command = &command;
  c.geometry = &geometry;
  std::memcpy(&c.mvp, &mvp, sizeof(SbMat));
  c.vpWidth = vpWidth;
  c.vpHeight = vpHeight;
  c.lineWidth = lineWidth;
  c.nearEps = 1.0e-5f;
  c.outBase = nullptr;
  c.posStrideFloats = posStrideFloats;
  c.segmentCount = segmentCount;

  this->dispatchWideLineSplit(1, segmentCount, params);

  // Exclusive prefix sum of emitted-quad offsets in floats (one add per segment).
  size_t total = 0;
  for (uint32_t s = 0; s < segmentCount; ++s) {
    this->wlineSplitOffsets[s] = total;
    if (this->wlineSplitValid[s]) total += 54;
  }
  if (total < 9) {
    // Nothing visible (mirrors expandWideLines()'s outIndex < 9 early-out).
    return false;
  }

  const VkDeviceSize needed = static_cast<VkDeviceSize>(total) * sizeof(float);
  if (slot.size < needed) {
    if (slot.buffer != VK_NULL_HANDLE || slot.memory != nullptr) {
      const VkBuffer oldBuffer = slot.buffer;
      const VmaAllocation oldMemory = slot.memory;
      slot.buffer = VK_NULL_HANDLE;
      slot.memory = nullptr;
      slot.mapped = nullptr;
      slot.size = 0;
      this->deferDestroyBufferMemory(oldBuffer, oldMemory);
    }
    if (!this->createMappedBuffer(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                  slot.buffer, slot.memory, &slot.mapped)) {
      this->emitError("expandWideLinesSplit: quad buffer create/map failed");
      slot.size = 0;
      return false;
    }
    slot.size = needed;
  }

  // Phase 2 emits compacted quads into the slot's mapping (pre-sized by prepareWideLineBuffers()).
  c.outBase = static_cast<float *>(slot.mapped);
  this->dispatchWideLineSplit(2, segmentCount, params);

  slot.expandFingerprint = wfp;
  slot.expandVertexCount = static_cast<uint32_t>(total / 9);
  entry.wideLineVertexCount = static_cast<uint32_t>(total / 9);
  return true;
}

void
SoVulkanRenderBackend::dispatchWideLineSplit(int phase, uint32_t count,
                                             const SoRenderParams & params)
{
  const uint32_t W = this->maxRecordWorkers;
  for (uint32_t w = 0; w < W; ++w) {
    ParallelRecordJob & job = this->recordJobs[w];
    job.expandWideLines = true;
    job.params = &params;
    job.wideLineCommands.clear();
    // The owner (w == 0) is not a pool thread; it runs its range inline.
    job.wlineSplitPhase = (w == 0) ? 0 : phase;
    job.wlineSplitBegin = static_cast<uint32_t>(
      (static_cast<uint64_t>(count) * w) / W);
    job.wlineSplitEnd = static_cast<uint32_t>(
      (static_cast<uint64_t>(count) * (w + 1)) / W);
  }
  // Reset/bump under recordMutex; see expandWideLinesParallel().
  {
    std::lock_guard<std::mutex> lk(this->recordMutex);
    this->recordDoneCount.store(0);
    ++this->recordJobGeneration;
  }
  this->recordCvSpawn.notify_all();
  this->expandWideLinesSplitRange(phase, this->recordJobs[0].wlineSplitBegin,
                                  this->recordJobs[0].wlineSplitEnd);
  this->recordJobs[0].ok = true;
  {
    std::unique_lock<std::mutex> lk(this->recordMutex);
    this->recordCvDone.wait(lk, [this] {
      return this->recordDoneCount.load() >= this->maxRecordWorkers - 1;
    });
  }
}

void
SoVulkanRenderBackend::expandWideLinesSplitRange(int phase, uint32_t begin,
                                                 uint32_t end)
{
  const WideLineSplitCtx & c = this->wlineSplitCtx;
  const SoGeometryDesc & geometry = *c.geometry;
  const uint32_t * const indices = geometry.indices;
  const float * const positions = geometry.positions;
  const float * const colors = geometry.colors;
  float * const clipCache = this->wlineSplitClip.data();
  uint8_t * const valid = this->wlineSplitValid.data();
  const float nearEps = c.nearEps;

  if (phase == 1) {
    // Clip transform + per-segment visibility; writes only this range, so ranges don't overlap.
    for (uint32_t s = begin; s < end; ++s) {
      const uint32_t i0 = indices ? indices[s * 2] : s * 2;
      const uint32_t i1 = indices ? indices[s * 2 + 1] : s * 2 + 1;
      float * const c0 = clipCache + static_cast<size_t>(i0) * 4;
      float * const c1 = clipCache + static_cast<size_t>(i1) * 4;
      wlineTransformPoint(c.mvp,
        positions + static_cast<size_t>(i0) * c.posStrideFloats, c0);
      wlineTransformPoint(c.mvp,
        positions + static_cast<size_t>(i1) * c.posStrideFloats, c1);

      const float fa = c0[2];
      const float fb = c1[2];
      const bool visible0 = (c0[3] > nearEps) && (fa >= 0.0f);
      const bool visible1 = (c1[3] > nearEps) && (fb >= 0.0f);
      bool ok = false;
      if (visible0 || visible1) {
        float tA;
        float tB;
        if (visible0 && visible1) {
          tA = 0.0f;
          tB = 1.0f;
        }
        else {
          const float denom = fa - fb;
          const float tclip = (denom != 0.0f) ? fa / denom : 0.0f;
          tA = visible0 ? 0.0f : tclip;
          tB = visible1 ? 1.0f : tclip;
        }
        const float cA3 = c0[3] + tA * (c1[3] - c0[3]);
        const float cB3 = c0[3] + tB * (c1[3] - c0[3]);
        if (cA3 > nearEps && cB3 > nearEps) {
          const float cA0 = c0[0] + tA * (c1[0] - c0[0]);
          const float cA1 = c0[1] + tA * (c1[1] - c0[1]);
          const float cB0 = c0[0] + tB * (c1[0] - c0[0]);
          const float cB1 = c0[1] + tB * (c1[1] - c0[1]);
          const float ndc0x = cA0 / cA3;
          const float ndc0y = cA1 / cA3;
          const float ndc1x = cB0 / cB3;
          const float ndc1y = cB1 / cB3;
          const float dx = ndc1x - ndc0x;
          const float dy = ndc1y - ndc0y;
          ok = std::sqrt(dx * dx + dy * dy) >= 1.0e-8f;
          if (ok) {
            // Frustum reject: otherwise off-screen segments are still expanded and
            // uploaded for the GPU to clip.  Pad the NDC box by lineWidth/viewport.
            const float mx = c.lineWidth / c.vpWidth;
            const float my = c.lineWidth / c.vpHeight;
            const float minx = std::min(ndc0x, ndc1x) - mx;
            const float maxx = std::max(ndc0x, ndc1x) + mx;
            const float miny = std::min(ndc0y, ndc1y) - my;
            const float maxy = std::max(ndc0y, ndc1y) + my;
            if (maxx < -1.0f || minx > 1.0f || maxy < -1.0f || miny > 1.0f) {
              ok = false;
            }
          }
        }
      }
      valid[s] = ok ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }
    return;
  }

  // Phase 2: emit the visible segments at their prefix-summed offsets.
  const size_t * const offsets = this->wlineSplitOffsets.data();
  float * const quads = c.outBase;
  const float defaultColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
  static const int triOrder[6] = { 0, 1, 2, 2, 1, 3 };
  for (uint32_t s = begin; s < end; ++s) {
    if (!valid[s]) continue;
    const uint32_t i0 = indices ? indices[s * 2] : s * 2;
    const uint32_t i1 = indices ? indices[s * 2 + 1] : s * 2 + 1;
    const float * const col0 = colors
      ? colors + static_cast<size_t>(i0) * 4 : nullptr;
    const float * const col1 = colors
      ? colors + static_cast<size_t>(i1) * 4 : nullptr;
    const float * const c0 = clipCache + static_cast<size_t>(i0) * 4;
    const float * const c1 = clipCache + static_cast<size_t>(i1) * 4;

    const float fa = c0[2];
    const float fb = c1[2];
    const bool visible0 = (c0[3] > nearEps) && (fa >= 0.0f);
    const bool visible1 = (c1[3] > nearEps) && (fb >= 0.0f);
    float tA;
    float tB;
    if (visible0 && visible1) {
      tA = 0.0f;
      tB = 1.0f;
    }
    else {
      const float denom = fa - fb;
      const float tclip = (denom != 0.0f) ? fa / denom : 0.0f;
      tA = visible0 ? 0.0f : tclip;
      tB = visible1 ? 1.0f : tclip;
    }

    const float cA[4] = {
      c0[0] + tA * (c1[0] - c0[0]),
      c0[1] + tA * (c1[1] - c0[1]),
      c0[2] + tA * (c1[2] - c0[2]),
      c0[3] + tA * (c1[3] - c0[3]),
    };
    const float cB[4] = {
      c0[0] + tB * (c1[0] - c0[0]),
      c0[1] + tB * (c1[1] - c0[1]),
      c0[2] + tB * (c1[2] - c0[2]),
      c0[3] + tB * (c1[3] - c0[3]),
    };
    // Phase 1 already rejected these; recomputed for the emit math below.
    if (cA[3] <= nearEps || cB[3] <= nearEps) continue;

    const float ndc0x = cA[0] / cA[3];
    const float ndc0y = cA[1] / cA[3];
    const float ndc1x = cB[0] / cB[3];
    const float ndc1y = cB[1] / cB[3];
    const float dx = ndc1x - ndc0x;
    const float dy = ndc1y - ndc0y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.0e-8f) continue;
    const float dirx = dx / length;
    const float diry = dy / length;
    const float offx = -diry * c.lineWidth / c.vpWidth;
    const float offy = dirx * c.lineWidth / c.vpHeight;

    // Quad corners: [0]=p0+off, [1]=p0-off, [2]=p1+off, [3]=p1-off.
    float corners[4][4];
    for (int corner = 0; corner < 4; ++corner) {
      const int endpoint = corner < 2 ? 0 : 1;
      const float sign = (corner % 2 == 0) ? 1.0f : -1.0f;
      const float w = endpoint == 0 ? cA[3] : cB[3];
      corners[corner][0] = (endpoint == 0 ? cA[0] : cB[0]) + sign * offx * w;
      corners[corner][1] = (endpoint == 0 ? cA[1] : cB[1]) + sign * offy * w;
      corners[corner][2] = endpoint == 0 ? cA[2] : cB[2];
      corners[corner][3] = w;
    }
    const float * const p0 = col0 ? col0 : defaultColor;
    const float * const p1 = col1 ? col1 : defaultColor;
    float colA[4];
    float colB[4];
    for (int i = 0; i < 4; ++i) {
      colA[i] = p0[i] + tA * (p1[i] - p0[i]);
      colB[i] = p0[i] + tB * (p1[i] - p0[i]);
    }

    float * out = quads + offsets[s];
    for (int t = 0; t < 6; ++t) {
      const int corner = triOrder[t];
      const int endpoint = corner < 2 ? 0 : 1;
      const float * col = endpoint == 0 ? colA : colB;
      out[0] = corners[corner][0];
      out[1] = corners[corner][1];
      out[2] = corners[corner][2];
      out[3] = corners[corner][3];
      out[4] = col[0];
      out[5] = col[1];
      out[6] = col[2];
      out[7] = col[3];
      out[8] = 0.0f;  // non-stippled split path: distance is unused
      out += 9;
    }
  }
}

