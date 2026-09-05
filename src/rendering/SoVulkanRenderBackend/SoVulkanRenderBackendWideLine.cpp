// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendWideLine.cpp
//
// CPU expansion of wide and/or stippled lines into triangle-list quads.
// expandWideLines() walks each segment and:
//
//   - transforms the endpoints to clip space
//   - near-plane clips, interpolating the hidden endpoint onto the plane
//   - accumulates the screen-space polyline distance (in pixels, for the
//     glLineStipple pattern)
//   - emits 2 triangles / 6 vertices per segment into one host-visible quad
//     buffer per in-flight frame slot (drawn by the wide-line pipeline)

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace CoinVulkanDetail;

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

  static int wlineDiag = 0;
  const bool isSketchCmd = vertexCount >= 900;
  const bool wdiag = COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")
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
  if (wlineOverlay) {
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
  // camera the positions, view, projection, width and viewport are
  // byte-identical every frame -- so the already-expanded quads in this slot
  // are still exact.  Key the slot on the authoritative geometry content hash
  // (the same one updateGeometryCache uses, so in-place edits invalidate here
  // too) plus view/proj/width/viewport; on a match reuse the buffer instead of
  // re-expanding and re-uploading it.
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
    if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
      static uint64_t wlineHits = 0;
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
  this->wlineClipScratch.assign(static_cast<size_t>(vertexCount) * 4, 0.0f);
  this->wlineDistScratch.assign(vertexCount, 0.0f);
  float * const clipCache = this->wlineClipScratch.data();
  float * const distances = this->wlineDistScratch.data();
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
  this->wlineQuadScratch.assign(quadFloats, 0.0f);
  float * const quads = this->wlineQuadScratch.data();
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

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
    static int distLog = 0;
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
    if (slot.buffer != VK_NULL_HANDLE || slot.memory != VK_NULL_HANDLE) {
      const VkDevice device = this->device;
      const VkAllocationCallbacks * allocator = this->allocator;
      const VkBuffer oldBuffer = slot.buffer;
      const VkDeviceMemory oldMemory = slot.memory;
      slot.buffer = VK_NULL_HANDLE;
      slot.memory = VK_NULL_HANDLE;
      slot.mapped = nullptr;
      slot.size = 0;
      this->deferDestroy([device, allocator, oldBuffer, oldMemory]() {
        if (oldBuffer != VK_NULL_HANDLE) {
          vkDestroyBuffer(device, oldBuffer, allocator);
        }
        if (oldMemory != VK_NULL_HANDLE) {
          vkFreeMemory(device, oldMemory, allocator);
        }
      });
    }
    if (!this->createBuffer(needed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            slot.buffer, slot.memory, nullptr)) {
      this->emitError("expandWideLines: failed to create quad buffer");
      slot.size = 0;
      return false;
    }
    slot.size = needed;
    // Establish the persistent host mapping.  The buffer is HOST_VISIBLE |
    // HOST_COHERENT, so the GPU observes a memcpy without any explicit flush,
    // and keeping the mapping alive avoids a vkMapMemory/vkUnmapMemory pair on
    // every subsequent frame for this slot.
    if (vkMapMemory(this->device, slot.memory, 0, needed, 0, &slot.mapped)
        != VK_SUCCESS) {
      this->emitError("expandWideLines: vkMapMemory failed");
      return false;
    }
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
