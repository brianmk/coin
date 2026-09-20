// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendWideLine.cpp
//
// CPU expansion of wide and/or stippled lines into triangle-list quads.  This
// is the fallback path: plain wide lines are normally expanded on the GPU by
// the instanced vertex shader (see buildInstancedLineBuffer() and
// WideLineInstancedVertex.glsl).  The CPU path still handles stippled lines
// (order-dependent distance), line strips, a missing instance buffer, and the
// FC_VULKAN_WLINE_CPU override.  expandWideLines() walks each segment and:
//
//   - transforms the endpoints to clip space
//   - near-plane clips, interpolating the hidden endpoint onto the plane
//   - accumulates the screen-space polyline distance (in pixels, for the
//     glLineStipple pattern)
//   - emits 2 triangles / 6 vertices per segment into one host-visible quad
//     buffer per in-flight frame slot (drawn by the wide-line pipeline)

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include "vk_mem_alloc.h"

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

// Segments below this stay on the whole-command path: the dispatch/join cost
// (~tens of microseconds) is not worth splitting a small command.
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

// Clip-space transform with the same Y-flip / depth remap as the visual
// vertex shader (mirrors the lambda in expandWideLines()).
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
  // Guard against an uninitialised ring (uboSlotsPerFrame==0 -> zero bytes):
  // ensureInstanceModelBuffer() already clamps to a minimum of 64 bytes, so a
  // zero request still allocates a valid single-element buffer.  The per-draw
  // slot-index math is only ever exercised after prepareLightingSlots() sizes
  // uboSlotsPerFrame, so this is a safety bound, not a steady-state path.
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
  if (!this->buffers.createMapped(cap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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

  // Diagnostics only on the recording thread: worker-thread prints interleave
  // with it for no benefit and are the only shared I/O on this path.
  const bool onOwnerThread =
    std::this_thread::get_id() == this->wlineOwnerThread;
  static thread_local int wlineDiag = 0;
  const bool isSketchCmd = vertexCount >= 900;
  const bool wdiag = onOwnerThread &&
    SoVulkanConfig::get().debug.backendDebug
    && (isSketchCmd || wlineDiag < 40) && wlineDiag < 200;
  if (wdiag) {
    ++wlineDiag;
    fprintf(stderr, "[WLINE2] enter frame=%u cmd=%p verts=%u idx=%u strip=%d segs=%u lw=%.2f\n",
            this->uboFrameIndex, (const void*)&command, vertexCount, count, strip ? 1 : 0,
            static_cast<unsigned>(segmentCount),
            static_cast<double>(lineWidth));
  }

  // MVP: clip = P * V * M * pos.  Regular geometry shares the frame view
  // matrix in params.  An overlay command (NaviCube corner sub-region) is
  // its own camera and must use its own view matrix so the wide-line quad
  // expansion matches the filled geometry, which the visual shader
  // transforms with command.viewMatrix for the overlay.  Using the frame
  // view here transformed the overlay wide lines with the wrong camera,
  // displacing the NaviCube edges / origin axes off the cube (the
  // 'edges/axes drift away from the cube' bug).
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
  // Match updateLightingUniforms()/resolveCommandProj(): a self-camera overlay
  // (the NaviCube sub-scene) uses the command's own view, but a frame-camera
  // overlay (the selection/preselection highlight spanning the frame viewport)
  // and every main-pass command use the frame view.  Selecting the view on
  // `pass == OVERLAY` alone transformed a frame-camera overlay with the scene
  // camera's recorded matrix, which lags one frame behind navigation, so its
  // wide lines stayed at the previous camera pose after a move.
  if (wlineOverlay && !isFrameCameraOverlay(command, params)) {
    command.viewMatrix.getValue(view);
  }
  else {
    params.viewMatrix.getValue(view);
  }
  // The GPU visual path computes clip = proj_GL * view_GL * model_GL * pos,
  // where a *_GL matrix is the transpose of the stored (row-major) SbMat,
  // because Vulkan/GLSL interprets the raw memory as column-major.  The CPU
  // quad expansion below must produce the exact same clip positions (the
  // wide-line vertex shader passes them through).  With the stored accessor
  // matrices this is equivalent to mvp = transpose(model * view * proj):
  // (W^T * v)_i = sum_{l,k,j} model[l][k] view[k][j] proj[j][i] v_l
  //   = sum_j proj_GL[i][j] * (view_GL * model_GL * v)_j.
  // Multiplying in the naive order (proj*view*model) would place the view
  // translation into the clip w component (-P.x*... *v + 1), producing huge
  // negative w for off-origin geometry, a failing near-plane test, and
  // geometry collapsed onto NDC (0,0).
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
  // The width offset is normalized against the viewport the line is drawn
  // into.  An overlay line draws into its own sub-region (the NaviCube rect),
  // so normalize against that, not the full-screen frame viewport -- using
  // the frame size scaled every overlay width by frame/overlay and, on a
  // fractional-DPI display, rendered the edge/axis strokes displaced.
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
  // The quad expansion (clip transform + per-segment geometry + distance
  // accumulation below) is the dominant per-frame CPU cost for line and edge
  // heavy scenes, and on a retained (replayed) draw list with an unchanged
  // camera the positions, model, view, projection, width and viewport are
  // byte-identical every frame -- so the already-expanded quads in this slot
  // are still exact.  Key the slot on the authoritative geometry content hash
  // (the same one updateGeometryCache uses, so in-place edits invalidate here
  // too) plus model/view/proj/width/viewport; on a match reuse the buffer
  // instead of re-expanding and re-uploading it.
  //
  // The model matrix MUST be in the key: the quads are expanded into clip
  // space (mvp bakes in the object transform), so an object that is moved /
  // rotated while its geometry content hash is unchanged would otherwise reuse
  // quads transformed by the old model -- its wide lines stayed at the
  // previous location while the filled geometry (which applies the live model
  // in the shader) moved.
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
      std::memcpy(&bits, &model[r][c], sizeof(bits));
      mixWide(bits);
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
    if (onOwnerThread && SoVulkanConfig::get().debug.backendDebug) {
      static thread_local uint64_t wlineHits = 0;
      if (++wlineHits % 200 == 0) {
        fprintf(stderr, "[WLINE-cache] hits=%llu cmd=%p\n",
                (unsigned long long)wlineHits, (const void*)&command);
      }
    }
    return true;
  }

  // Transform a position to clip space with the same Y-flip and depth remap
  // as the visual vertex shader, so the wide-line vertex shader can pass the
  // quad corners through unchanged.
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

  // Per-vertex clip-space cache plus accumulated polyline distance in
  // WINDOW PIXELS.  Classic GL applies the stipple pattern in screen space
  // (glLineStipple: each bit covers linePatternScaleFactor pixels), so the
  // fragment discard below must operate on pixel distances, not object
  // units -- an object-unit period changes size when zooming.
  // Per-thread scratch: the expansion runs on the parallel record workers, so
  // a shared member would race.  thread_local vectors keep the reuse (no
  // realloc once grown) while giving each worker its own storage.
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

  // 6 vertices per segment (two triangles), 9 floats each:
  // clip position (4) + color (4) + distance in pixels (1).
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

    // Near-plane clip in the remapped clip space.  A point is visible when
    // it is in front of the eye (w > nearEps) AND in front of the near plane
    // (the remapped z, cache[2] >= 0, i.e. z_ndc >= 0).  The old code
    // dropped the WHOLE segment when either endpoint had w <= 0, so a
    // segment merely crossing the near plane vanished at close-in views.
    // Instead, clip against the boundary: when only one endpoint is visible
    // the segment straddles the near plane, so keep the visible half and
    // interpolate the hidden endpoint onto the plane.  Behind-the-eye points
    // also land behind the near plane (cache[2] < 0), so this single clip
    // surface handles both the near-plane and eye-plane crossings.
    const float fa = c0[2];
    const float fb = c1[2];
    const bool visible0 = (c0[3] > nearEps) && (fa >= 0.0f);
    const bool visible1 = (c1[3] > nearEps) && (fb >= 0.0f);
    if (!visible0 && !visible1) {
      if (wdiag) ++diagSkippedW;
      continue;
    }

    // tA/tB give the interpolation fraction along c0 -> c1 for each emitted
    // end (0 = original c0, 1 = original c1, intermediate = clipped onto the
    // near plane).  Exactly one end is ever clipped because a plane meets a
    // line segment at most once.
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
    // Frustum reject (see expandWideLinesSplitRange): a segment wholly off
    // the sides of the view is not worth expanding -- the GPU would clip it.
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
    // Anisotropic NDC offset mirroring the GL geometry shader
    // (perp * lineWidth / u_vpSize with per-axis division).
    const float offx = -diry * lineWidth / vpWidth;
    const float offy = dirx * lineWidth / vpHeight;

    // Quad corners: [0]=p0+off, [1]=p0-off, [2]=p1+off, [3]=p1-off.
    // Triangles: (0,1,2), (2,1,3) -- same emission order as the GL
    // geometry shader's triangle strip.
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
    // Colors, interpolated for a clipped (near-plane) endpoint.  The
    // default is opaque white when the command has no per-vertex colors.
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
      fprintf(stderr, "[WLINE2] FAIL cmd=%p verts=%u segs=%u outIndex=%zu "
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
    fprintf(stderr, "[WLINE2] OK cmd=%p verts=%u segs=%u quads=%zu "
                    "skippedW=%zu skippedDeg=%zu firstSegNDC=(%.3f,%.3f)->(%.3f,%.3f)\n",
            (const void*)&command, vertexCount, static_cast<unsigned>(segmentCount),
            outIndex / 9, diagSkippedW, diagSkippedDeg,
            static_cast<double>(clipCache[0] / clipCache[3]),
            static_cast<double>(clipCache[1] / clipCache[3]),
            static_cast<double>(clipCache[4] / clipCache[7]),
            static_cast<double>(clipCache[5] / clipCache[7]));
    fprintf(stderr, "[WLINE2]   viewT=(%.3f,%.3f,%.3f) v11=%.4f v00=%.4f "
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

  if (onOwnerThread && SoVulkanConfig::get().debug.backendDebug) {
    static thread_local int distLog = 0;
    if (distLog++ < 3) {
      fprintf(stderr, "[WLINE] verts=%u segs=%u quads=%zu dists:",
              vertexCount, segmentCount, outIndex / 9);
      for (size_t q = 0; q < outIndex && q < 60; q += 9) {
        fprintf(stderr, " %.1f", static_cast<double>(quads[q + 8]));
      }
      fprintf(stderr, "\n");
    }
  }

  const VkDeviceSize needed =
    static_cast<VkDeviceSize>(outIndex) * sizeof(float);
  // Ring of host-visible scratch buffers, one per in-flight frame slot: the
  // slot selected for the current frame index is reused only after the same
  // slot's previous submission has completed (beginFrame waits the slot
  // fence).  Growth defers the old buffer's destruction instead of destroying
  // it synchronously, since a still-executing frame may reference it.
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
    // Persistent host mapping.  The buffer is HOST_VISIBLE | HOST_COHERENT, so
    // the GPU observes a memcpy without any explicit flush, and keeping the
    // mapping alive avoids a vkMapMemory/vkUnmapMemory pair every frame.
    if (!this->buffers.createMapped(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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

  // The endpoint stream is a pure function of the (object-space) geometry, so
  // rebuild only when the content hash changes.  contentHash==0 (unhashed) is
  // treated as a miss on the first call, when the buffer is still null.
  //
  // LIMITATION: an unhashed command (contentHash==0) that is edited in place
  // keeps its first-built endpoints, because 0 == 0 short-circuits here once
  // the buffer exists.  The hash is the only change signal, so a producer that
  // mutates geometry without updating contentHash (or invalidating
  // instancedLineHash) would render stale wide lines.  Leave contentHash==0
  // for genuinely immutable geometry.
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
  if (!this->buffers.createMapped(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                entry.instancedLineBuffer,
                                entry.instancedLineMemory, &mapped)) {
    this->emitError(
      "buildInstancedLineBuffer: endpoint buffer create/map failed");
    entry.instancedLineBuffer = VK_NULL_HANDLE;
    entry.instancedLineMemory = nullptr;
    return false;
  }

  // Four vec4 per segment: p0, p1, c0, c1.  Colors default to opaque white
  // when the geometry carries none; the shader only reads them when the
  // use-vertex-color flag is set, in which case the producer supplied them.
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
    VulkanCachedCommand * entryPtr = this->geometryCache.find(&command);
    if (entryPtr == nullptr) continue;
    VulkanCachedCommand & entry = *entryPtr;
    if (entry.vertexBuffer == VK_NULL_HANDLE) continue;

    // GPU-instanced wide line: build the static endpoint stream (once) and
    // skip the CPU quad buffer -- the vertex shader expands on the GPU.  A
    // failed build falls through to the CPU expansion below.
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
    // Worst case: every segment visible, 6 vertices, 9 floats per vertex.
    // expandWideLines() then only ever fills a buffer this size or smaller.
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
    if (!this->buffers.createMapped(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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
  // Must stay identical to recordDrawCommand()'s projection resolution: the
  // expansion cache key includes the projection, so a divergence would either
  // force a pointless re-expansion or reuse quads projected with the wrong
  // matrix.
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
  // This is always the recording thread; the workers it dispatches compare
  // against it to keep the diagnostics single-threaded.
  this->wlineOwnerThread = std::this_thread::get_id();
  // Gather the drawable wide-line commands.  This mirrors the guards the
  // record path applies (findCachedDrawable) so the pre-pass only touches
  // entries that will actually be drawn.
  std::vector<const SoRenderCommand *> & wideLines = this->wlineExpandScratch;
  wideLines.clear();
  std::vector<const SoRenderCommand *> & splitCmds = this->wlineSplitScratch;
  splitCmds.clear();
  const uint32_t W = this->maxRecordWorkers;
  const bool canSplit = W > 1 && !this->recordWorkers.empty() &&
    !SoVulkanConfig::get().raster.wideLineSerial;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (!isWideLine(command, -1, this->interactionLodActive)) continue;
    if (!command.geometry.positions || command.geometry.vertexCount == 0) {
      continue;
    }
    const VulkanCachedCommand * entryPtr = this->geometryCache.find(&command);
    if (entryPtr == nullptr || entryPtr->vertexBuffer == VK_NULL_HANDLE) continue;
    // Drawn by the GPU-instanced path: the vertex shader expands the segment,
    // so there is nothing to expand on the CPU.
    if (isInstancedWideLine(command) &&
        entryPtr->instancedLineBuffer != VK_NULL_HANDLE) {
      continue;
    }
    // A single dominant non-stippled LINE_LIST command (a lattice edge set)
    // cannot be balanced by the whole-command round-robin below -- one worker
    // would expand all of it.  Route it to the segment-range split instead.
    // Stippled lines stay serial: their per-vertex distance is order-dependent.
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

  // Expand the large commands first, on this (owner) thread; each split
  // dispatch joins before the next, so the pool is idle when it runs.
  for (const SoRenderCommand * command : splitCmds) {
    VulkanCachedCommand * entryPtr = this->geometryCache.find(command);
    if (entryPtr == nullptr) continue;
    VulkanCachedCommand & entry = *entryPtr;
    SbMat projValue;
    this->resolveCommandProj(*command, params, false, projValue);
    this->expandWideLinesSplit(entry, *command, params, projValue,
        std::max(1.0f, command->state.raster.lineWidth) * this->frameDpr);
  }

  if (wideLines.empty()) return;

  if (W <= 1 || this->recordWorkers.empty()) {
    // Serial fallback (single core, or the pool failed to build).
    for (const SoRenderCommand * command : wideLines) {
      VulkanCachedCommand & entry = *this->geometryCache.find(command);
      this->expandWideLinesFor(entry, *command, params,
                               command->pass == SO_RENDERPASS_OVERLAY);
    }
    return;
  }

  vkBackendTrace(this->uboFrameIndex, "expandWideLines.dispatch",
                 "cmds=%zu workers=%u", wideLines.size(), W);
  // Round-robin partition: the commands are of similar size, so this balances
  // well without the sort the record path needs for its batched items.
  for (uint32_t w = 0; w < W; ++w) {
    ParallelRecordJob & job = this->recordJobs[w];
    job.expandWideLines = true;
    // Must clear the split phase: a preceding split dispatch left it set on
    // the workers, and the worker loop keys the split branch on it.  Without
    // this reset they would re-run the stale split range instead of their
    // assigned wideLineCommands.
    job.wlineSplitPhase = 0;
    job.params = &params;
    job.wideLineCommands.clear();
  }
  for (size_t i = 0; i < wideLines.size(); ++i) {
    this->recordJobs[i % W].wideLineCommands.push_back(wideLines[i]);
  }

  // Reset the done counter and bump the generation under recordMutex so the
  // workers' count publication and the recording thread's predicate check are
  // ordered by the same lock (see recordJobWorker's increment).
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
      VulkanCachedCommand * entryPtr = this->geometryCache.find(command);
      if (entryPtr == nullptr) continue;
      VulkanCachedCommand & entry = *entryPtr;
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
// The round-robin in expandWideLinesParallel() balances by COMMAND, so one
// command holding an entire scene's edges is expanded by a single worker.
// This path partitions that one command's segments instead.  It is limited to
// non-stippled LINE_LIST geometry (each segment then references exactly its
// own two vertices and carries no order-dependent state), which is what a
// large BRep edge set produces.  The output is byte-identical to the serial
// expansion: phase 1 marks each segment, an exclusive prefix sum compacts the
// visible ones, and phase 2 emits them at the prefix offsets.

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

  // Main-pass only (the caller routes overlay commands to the serial path),
  // so the frame view/projection apply, exactly as in expandWideLines().
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
  // Same fingerprint as expandWideLines() (including the model matrix, which
  // the clip-space quads bake in), so the inline call made by
  // recordDrawCommand() during the record pass is a cache hit and does not
  // re-expand.
  uint64_t wfp = entry.contentHash;
  auto mixWide = [&wfp](uint32_t bits) {
    wfp ^= bits + 0x9E3779B97F4A7C15ULL + (wfp << 6) + (wfp >> 2);
  };
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      uint32_t bits;
      std::memcpy(&bits, &model[r][c], sizeof(bits));
      mixWide(bits);
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

  // Exclusive prefix sum of the emitted-quad offsets, in floats.  Cheap
  // (one add per segment) next to the expansion it compacts.
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
    if (!this->buffers.createMapped(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                  slot.buffer, slot.memory, &slot.mapped)) {
      this->emitError("expandWideLinesSplit: quad buffer create/map failed");
      slot.size = 0;
      return false;
    }
    slot.size = needed;
  }

  // Phase 2 emits the compacted quads directly into the slot's persistent
  // mapping (prepareWideLineBuffers() already sized it for the worst case).
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
    // Clip transform + per-segment visibility.  Writes only this range's
    // vertices and slots, so the ranges never overlap.
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
            // Frustum reject.  Without it a segment wholly off the sides of
            // the view is still expanded into quads and uploaded, only for
            // the GPU to clip it; that keeps the per-frame cost independent
            // of how much of a large edge set is actually on screen.  The
            // quad extends sideways by half a line width, so pad the segment's
            // NDC box by that margin (|off| <= lineWidth/viewport).
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

