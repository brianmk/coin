// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendPathTracing.cpp

// Split from the original monolithic SoRTXRenderBackend.cpp.  Contains the
// member functions for the "PathTracing" concern of the Vulkan RTX backend.

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

void
SoRTXRenderBackend::updatePathTracingState(const SoDrawList & drawlist,
                                           const SoRenderParams & params,
                                           const SoVulkanRenderTarget & target,
                                           VkCommandBuffer cmd)
{
  if (!this->createPathTracingBuffers(target.extent.width,
                                      target.extent.height)) {
    this->emitError("updatePathTracingState: failed to create path tracing buffers");
    return;
  }

  // Detect camera motion (view + projection) and scene changes; both reset
  // the progressive accumulation back to a live preview.  The scene-change
  // signal is the geometry-cache dirtiness computed by
  // updateGeometryCache() (the draw-list generation is a per-frame counter
  // and cannot be used for this).
  float viewMatrix[16];
  float projMatrix[16];
  SbMat viewValue;
  params.viewMatrix.getValue(viewValue);
  std::memcpy(viewMatrix, &viewValue[0][0], sizeof(viewMatrix));
  SbMat projValue;
  params.projMatrix.getValue(projValue);
  std::memcpy(projMatrix, &projValue[0][0], sizeof(projMatrix));
  const SbVec2s & vpSize = params.viewport.getViewportSizePixels();
  // Prefer the manager's camera generation counter when supplied: it is the
  // authoritative camera-motion signal and cannot be fooled by floating-point
  // equality tolerances (a real pan/rotation may produce a matrix variation
  // swallowed by the epsilon below, so a stale-converged viewport never
  // resumes).  Fall back to the matrix diff only when no counter is present
  // (0 == "not supplied", e.g. a direct backend test harness).
  bool viewChanged = false;
  if (params.cameraVersion != 0) {
    viewChanged = !this->haveLastCameraVersion ||
                  params.cameraVersion != this->lastCameraVersion ||
                  this->lastViewportWidth != static_cast<uint32_t>(vpSize[0]) ||
                  this->lastViewportHeight != static_cast<uint32_t>(vpSize[1]);
  }
  else {
    viewChanged =
      !this->haveLastView ||
      !matricesNearlyEqual(viewMatrix, this->lastViewMatrix, 16) ||
      !matricesNearlyEqual(projMatrix, this->lastProjMatrix, 16) ||
      this->lastViewportWidth != static_cast<uint32_t>(vpSize[0]) ||
      this->lastViewportHeight != static_cast<uint32_t>(vpSize[1]);
  }
  const bool sceneChanged = this->cacheChanged;

  // Background / environment change detection.  The viewport gradient, sky and
  // sun drive the environment radiance but NEVER the acceleration structures,
  // so a change must restart the tracer while leaving the (skippable) AS phase
  // unchanged.  Matches the bg values the frame block writes below.
  float curBgTop[4];
  float curBgBottom[4];
  if (this->envMapId >= 0) {
    curBgTop[0] = this->envSkyTop[0];
    curBgTop[1] = this->envSkyTop[1];
    curBgTop[2] = this->envSkyTop[2];
    curBgBottom[0] = this->envSkyBottom[0];
    curBgBottom[1] = this->envSkyBottom[1];
    curBgBottom[2] = this->envSkyBottom[2];
  }
  else {
    curBgTop[0] = params.backgroundTopColor[0];
    curBgTop[1] = params.backgroundTopColor[1];
    curBgTop[2] = params.backgroundTopColor[2];
    curBgBottom[0] = params.backgroundBottomColor[0];
    curBgBottom[1] = params.backgroundBottomColor[1];
    curBgBottom[2] = params.backgroundBottomColor[2];
  }
  curBgTop[3] = 1.0f;
  curBgBottom[3] = 1.0f;
  const float curEnvIntensity = this->envMapId >= 0
    ? this->envIntensity * this->envSkyBrightness : 0.0f;
  // Epsilon compare: the viewport gradient colours are stable, but a tiny
  // re-derivation (e.g. a gradient recomputed per redraw) must not read as a
  // background change -- that would clear a pending denoise at the sample
  // ceiling and the run would never publish a denoised frame.
  const auto fne = [](float a, float b) {
    const float d = a - b;
    return (d > 1.0e-4f) || (d < -1.0e-4f);
  };
  bool backgroundChanged = !this->haveLastBackground;
  if (!backgroundChanged) {
    backgroundChanged =
      this->envMapId != this->lastEnvMapId ||
      fne(curEnvIntensity, this->lastEnvIntensity) ||
      fne(curBgTop[0], this->lastBgTopColors[0]) ||
      fne(curBgTop[1], this->lastBgTopColors[1]) ||
      fne(curBgTop[2], this->lastBgTopColors[2]) ||
      fne(curBgBottom[0], this->lastBgBottomColors[0]) ||
      fne(curBgBottom[1], this->lastBgBottomColors[1]) ||
      fne(curBgBottom[2], this->lastBgBottomColors[2]) ||
      fne(this->envSunDir[0], this->lastSunDir[0]) ||
      fne(this->envSunDir[1], this->lastSunDir[1]) ||
      fne(this->envSunDir[2], this->lastSunDir[2]) ||
      fne(this->envSunColor[0], this->lastSunColor[0]) ||
      fne(this->envSunColor[1], this->lastSunColor[1]) ||
      fne(this->envSunColor[2], this->lastSunColor[2]);
  }
  std::memcpy(this->lastBgTopColors, curBgTop, sizeof(curBgTop));
  std::memcpy(this->lastBgBottomColors, curBgBottom, sizeof(curBgBottom));
  std::memcpy(this->lastSunDir, this->envSunDir, sizeof(this->envSunDir));
  std::memcpy(this->lastSunColor, this->envSunColor, sizeof(this->envSunColor));
  this->lastEnvIntensity = curEnvIntensity;
  this->lastEnvMapId = this->envMapId;
  this->haveLastBackground = true;

  // The temporal-reprojection path starts every frame disabled; only the
  // camera-move and auto-restart branches below re-enable it for exactly
  // one frame.  The previous frame's camera is still in lastViewMatrix /
  // lastProjMatrix here (they are overwritten further down), so compose
  // the world->clip matrix of the previous frame for the shader's
  // history-reprojection test.  Row-vector convention (Coin's SbMatrix):
  // world->clip = view * proj, matching the column-vector layout the
  // shader receives from the raw matrix copies.
  this->ptReprojectFrame = FALSE;
  {
    SbMatrix prevView(lastViewMatrix[0], lastViewMatrix[1],
                      lastViewMatrix[2], lastViewMatrix[3],
                      lastViewMatrix[4], lastViewMatrix[5],
                      lastViewMatrix[6], lastViewMatrix[7],
                      lastViewMatrix[8], lastViewMatrix[9],
                      lastViewMatrix[10], lastViewMatrix[11],
                      lastViewMatrix[12], lastViewMatrix[13],
                      lastViewMatrix[14], lastViewMatrix[15]);
    SbMatrix prevProj(lastProjMatrix[0], lastProjMatrix[1],
                      lastProjMatrix[2], lastProjMatrix[3],
                      lastProjMatrix[4], lastProjMatrix[5],
                      lastProjMatrix[6], lastProjMatrix[7],
                      lastProjMatrix[8], lastProjMatrix[9],
                      lastProjMatrix[10], lastProjMatrix[11],
                      lastProjMatrix[12], lastProjMatrix[13],
                      lastProjMatrix[14], lastProjMatrix[15]);
    prevView.multRight(prevProj);
    SbMat prevVpValue;
    prevView.getValue(prevVpValue);
    std::memcpy(this->prevViewProj, &prevVpValue[0][0], sizeof(float) * 16);
  }

  if (!this->ptEnabled ||
      this->rtxViewMode == RtxViewMode::RtxModeAmbientOcclusion ||
      this->rtxViewMode == RtxViewMode::RtxModeEnvironment) {
    // Disabled, or the single-sample AO / Environment previews: never
    // accumulate, so the live image is recomputed fresh every redraw.
    this->ptAccumulating = FALSE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
    this->ptConverged = FALSE;
    this->ptWasMoving = FALSE;
    this->ptDenoisePending = FALSE;
  }
  else if (this->ptStartLatch) {
    // The start flag: reset the accumulation and begin a fresh progressive
    // run, regardless of camera/scene changes.
    this->ptStartLatch = FALSE;
    this->ptAccumulating = TRUE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
    this->ptConverged = FALSE;
    this->ptWasMoving = FALSE;
    this->ptDenoisePending = FALSE;
    this->denoiseResultReady = FALSE;
    this->ptForceFullResolve = TRUE;
  }
  else if (backgroundChanged || sceneChanged || (viewChanged && !this->haveLastView)) {
    // A scene edit invalidates the history (surface colors may be stale
    // even where positions match), a background/environment change changes
    // the environment radiance, and the very first frame has nothing to
    // reproject: drop to the 1-spp live preview; the settle counter then
    // restarts accumulation.  Camera-only changes fall through to the
    // reprojection branch below.
    this->ptAccumulating = FALSE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
    this->ptConverged = FALSE;
    this->ptWasMoving = FALSE;
    this->ptDenoisePending = FALSE;
    this->denoiseResultReady = FALSE;
    this->ptForceFullResolve = TRUE;
  }
  else if (viewChanged) {
    // --- Reset-on-move ---------------------------------------------------
    // A camera move invalidates the accumulated history: per-pixel positions
    // and normals are stale, and the adaptive sampler's per-pixel variance
    // estimate no longer corresponds to the pixels a static view would see.
    // Carrying that history forward (temporal reprojection) lets the adaptive
    // early-out freeze pixels at mutually-inconsistent partial values, which
    // reads as "the image gets grainy after I move the camera and never
    // cleans up."  So a move resets the run to a fresh 1-spp preview; the
    // settle counter below auto-restarts a clean accumulation against the new
    // camera once it has been static for a short window.
    this->ptAccumulating = FALSE;
    this->ptFrameIndex = 0;
    this->ptIdleFrames = 0;
    this->ptConverged = FALSE;
    this->ptWasMoving = FALSE;
    this->ptDenoisePending = FALSE;
    this->denoiseResultReady = FALSE;
    this->ptForceFullResolve = TRUE;
  }
  else if (this->ptAccumulating) {
    // --- Accumulate-while-static -----------------------------------------
    // Static camera/scene: keep adding one jittered sample per frame.  The
    // present pass shows the in-shader edge-stopped running mean, which
    // monotonically cleans up as samples accumulate (no denoiser churn).
    ++this->ptFrameIndex;
    // Convergence: stop accumulating once the sample cap is reached or the
    // adaptive active-pixel fraction fell below the stop threshold.  On the
    // frame the target is reached, KEEP accumulating (don't drop to preview)
    // so the G-buffer readback for the final image is still recorded and the
    // denoiser runs exactly once (denoise-at-target).  The idle/settle
    // transition is driven from updateDenoise after it publishes the result.
    const bool adaptivelyConverged =
      !this->useSbtPipeline && this->ptAdaptiveEnabled &&
      this->ptFrameIndex >= this->ptAdaptiveMinSamples &&
      this->ptLastActiveFraction < this->ptAdaptiveStopFraction &&
      !this->ptForceFullResolve;
    if (this->ptFrameIndex >= this->ptMaxSamples || adaptivelyConverged) {
      // A full-resolve run (after a move or fresh start) only idles once the
      // hard sample cap is reached, so the tiny/off-centre boxes that the
      // adaptive sampler would otherwise freeze at a thin background-leaning
      // mean get solid coverage before the viewport holds a faint frame.
      this->ptForceFullResolve = FALSE;
      if (this->denoiserActive && this->denoisedBuffer != VK_NULL_HANDLE) {
        // Denoise-at-target: keep accumulating the final frame (so the
        // G-buffer readback is recorded) and let updateDenoise run once,
        // then transition to converged-idle from there.
        this->ptDenoisePending = TRUE;
      }
      else {
        // No denoiser: converge directly to the raw accumulated image.  The
        // settle window keeps refining TRUE for a few frames so the final
        // edge-stopped mean is presented, then saturates to idle.
        this->ptConverged = TRUE;
        this->ptAccumulating = FALSE;
        this->ptIdleFrames = 0;
      }
    }
    else {
      this->ptConverged = FALSE;
    }
  }
  else if (this->ptIdleFrames < this->ptSettleFrames) {
    // Static but not accumulating.  Two distinct cases:
    //  - Converged (denoise-at-target published the final image): the
    //    denoised result is already ready and was presented; do NOT
    //    auto-restart, just finish the settle window so the refresh loop can
    //    go idle.  ptConverged keeps it from re-accumulating.
    //  - Post-reset (a camera/scene move dropped the run to preview): count
    //    idle frames so the auto-restart below starts a fresh accumulation
    //    against the new camera.
    ++this->ptIdleFrames;
    if (this->ptConverged) {
      if (this->ptIdleFrames >= this->ptSettleFrames) {
        this->ptIdleFrames = this->ptSettleFrames;
      }
    }
    else if (this->ptIdleFrames >= this->ptSettleFrames) {
      this->ptIdleFrames = 0;
      this->ptAccumulating = TRUE;
      this->ptFrameIndex = 0;
      this->ptWasMoving = FALSE;
      // The reset-on-move architecture restarts a clean accumulation: do not
      // reproject the (now stale) history across the reset, so the first
      // frame of the new run clears the buffers and samples fresh.
      this->ptReprojectFrame = FALSE;
    }
  }
  // else: converged idle -- nothing to do until the camera or scene moves.

  if (getenv("FC_VULKAN_RT_DEBUG") && this->ptEnabled) {
    fprintf(stderr,
            "[RTDBG] ptState frame=%u viewChanged=%d sceneChanged=%d "
            "accum=%d frameIndex=%u idle=%u reproject=%d\n",
            params.frame,
            viewChanged ? 1 : 0, sceneChanged ? 1 : 0,
            this->ptAccumulating ? 1 : 0, this->ptFrameIndex,
            this->ptIdleFrames, this->ptReprojectFrame ? 1 : 0);
  }

  if (getenv("FC_VULKAN_PT_DEBUG")) {
    static uint32_t debugFrame = 0;
    if ((debugFrame++ % 30) == 0 || viewChanged || sceneChanged) {
      float maxViewDelta = 0.0f;
      int maxIdx = -1;
      for (int i = 0; i < 16; ++i) {
        const float d = std::fabs(viewMatrix[i] - this->lastViewMatrix[i]);
        if (d > maxViewDelta) { maxViewDelta = d; maxIdx = i; }
      }
      if (viewChanged && debugFrame < 40) {
        if (maxIdx >= 0) {
          fprintf(stderr,
                  "[PTDBG] f=%u view[%d] %.6f -> %.6f (t: %.3f,%.3f,%.3f)\n",
                  debugFrame - 1, maxIdx, this->lastViewMatrix[maxIdx],
                  viewMatrix[maxIdx], viewMatrix[12], viewMatrix[13],
                  viewMatrix[14]);
        }
        else {
          fprintf(stderr, "[PTDBG] f=%u view unchanged (viewport resize)\n",
                  debugFrame - 1);
        }
      }
      fprintf(stderr,
              "[PTDBG] f=%u latch=%d accum=%d frameIndex=%u "
              "viewChanged=%d sceneChanged=%d maxViewDelta=%g\n",
              debugFrame - 1, this->ptStartLatch ? 1 : 0,
              this->ptAccumulating ? 1 : 0, this->ptFrameIndex,
              viewChanged ? 1 : 0, sceneChanged ? 1 : 0, maxViewDelta);
    }
  }

  // Store the identity of this frame's view/scene for the next comparison.
  std::memcpy(this->lastViewMatrix, viewMatrix, sizeof(viewMatrix));
  std::memcpy(this->lastProjMatrix, projMatrix, sizeof(projMatrix));
  this->lastViewportWidth = static_cast<uint32_t>(vpSize[0]);
  this->lastViewportHeight = static_cast<uint32_t>(vpSize[1]);
  this->lastCameraVersion = params.cameraVersion;
  this->haveLastCameraVersion = params.cameraVersion != 0;
  this->haveLastView = TRUE;
  // NOTE: cacheChanged is NOT consumed here.  recordAccelerationStructures()
  // reads it again after this call to decide the AS rebuild (asDirty);
  // clearing it here used to leave that check seeing a stale false, skipping
  // the TLAS build and the descriptor update on the first frame of a freshly
  // initialized backend -- the trace then dispatched against a never-updated
  // descriptor set (VUID-vkCmdDispatch-None-08114) and hung the GPU until the
  // driver reset the device.  It is consumed at the asDirty computation.

  // A zero frame index means a fresh accumulation: clear the accumulation
  // and sums-of-squares buffers with a fill recorded here (still outside
  // any render pass).  This runs before the caller's render pass on the
  // same submission ordering, so the tracer observes the cleared buffers.
  // Reprojection frames skip the clear: their history replaces the buffer
  // contents per-pixel in the shader.  Both members of each ping-pong pair
  // are cleared: the post-frame swap makes the history buffer the next
  // frame's live buffer, and a stale accumulation must never leak into a
  // fresh run.
  if (this->ptEnabled && this->ptFrameIndex == 0 &&
      !this->ptReprojectFrame) {
    vkCmdFillBuffer(cmd, this->accumBuffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, this->sumSqBuffer, 0, VK_WHOLE_SIZE, 0);
    if (this->accumHistoryBuffer != VK_NULL_HANDLE) {
      vkCmdFillBuffer(cmd, this->accumHistoryBuffer, 0, VK_WHOLE_SIZE, 0);
    }
    if (this->sumSqHistoryBuffer != VK_NULL_HANDLE) {
      vkCmdFillBuffer(cmd, this->sumSqHistoryBuffer, 0, VK_WHOLE_SIZE, 0);
    }
    VkMemoryBarrier fillBarrier {};
    fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
  }
  // The active-pixel counter is per-frame: zero it before every traced
  // frame (the host reads it back after the submission's queue wait).
  if (this->ptEnabled && this->activeCounterBuffer != VK_NULL_HANDLE) {
    vkCmdFillBuffer(cmd, this->activeCounterBuffer, 0, VK_WHOLE_SIZE, 0);
    // Make the fill visible to the compute tracer's atomics.
    VkMemoryBarrier counterBarrier {};
    counterBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    counterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    counterBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &counterBarrier, 0, nullptr, 0, nullptr);
  }
}

void
SoRTXRenderBackend::swapPathTracingHistory()
{
  // Called after the submission's queue wait on traced frames: hand the
  // just-written live buffers to the shader as next frame's history and
  // retire the old history into the live slots.  Only the compute tracer
  // maintains the accumulation buffers the history mirrors.
  if (!this->ptEnabled || !this->ptAccumulating || this->useSbtPipeline) {
    return;
  }
  std::swap(this->accumBuffer, this->accumHistoryBuffer);
  std::swap(this->accumMemory, this->accumHistoryMemory);
  std::swap(this->sumSqBuffer, this->sumSqHistoryBuffer);
  std::swap(this->sumSqMemory, this->sumSqHistoryMemory);
  std::swap(this->positionBuffer, this->positionHistoryBuffer);
  std::swap(this->positionMemory, this->positionHistoryMemory);
  this->ptHistoryValid = TRUE;
  this->ptReprojectFrame = FALSE;
  // The descriptor sets still reference the previous buffer handles;
  // updateDescriptors() rewrites them at the start of the next frame.
}

void
SoRTXRenderBackend::updateAdaptiveStats()
{
  // Called after the submission's queue wait: the host-visible counter
  // holds this frame's active-pixel count (counts[0]) and the number of
  // pixels that accepted reprojected history (counts[1]).
  uint32_t active = 0;
  uint32_t reprojected = 0;
  if (this->activeCounterMapped && this->ptEnabled && this->ptAccumulating) {
    const uint32_t * counts =
      static_cast<const uint32_t *>(this->activeCounterMapped);
    active = counts[0];
    reprojected = counts[1];
  }
  this->ptLastActivePixels = active;
  const uint64_t total =
    static_cast<uint64_t>(this->ptBufferWidth) * this->ptBufferHeight;
  // Non-accumulating frames carry no meaningful count; report fully active
  // so a stale value can never prematurely stop the next progressive run.
  this->ptLastActiveFraction =
    (this->ptEnabled && this->ptAccumulating && total > 0)
      ? static_cast<float>(active) / static_cast<float>(total) : 1.0f;
  if (getenv("FC_VULKAN_RT_DEBUG") && this->ptEnabled) {
    fprintf(stderr,
            "[RTDBG] adaptive frame=%u active=%u/%llu fraction=%.4f "
            "frameIndex=%u accum=%d self=%p buf=%ux%u reprojected=%u "
            "maxSamp=%u minSamp=%u fill=%d\n",
            this->ptLastFrame, active,
            static_cast<unsigned long long>(total),
            this->ptLastActiveFraction, this->ptFrameIndex,
            this->ptAccumulating ? 1 : 0, static_cast<const void *>(this),
            this->ptBufferWidth, this->ptBufferHeight, reprojected,
            this->ptMaxSamples, this->ptAdaptiveMinSamples,
            this->ptForceFullResolve ? 1 : 0);
  }
}

bool
SoRTXRenderBackend::recordAccelerationStructures(
  const SoDrawList & drawlist, const SoRenderParams & params,
  const SoVulkanRenderTarget & target, VkCommandBuffer cmd)
{
  // The geometry cache update must run first: it computes cacheChanged,
  // which the path tracing state machine consumes as the scene-change
  // signal (see updatePathTracingState()).
  this->updateGeometryCache(drawlist);
  this->updatePathTracingState(drawlist, params, target, cmd);
  // An accumulated result is only valid while accumulating; a camera move or
  // scene change (which updatePathTracingState may turn into a non-
  // accumulating preview frame) must fall back to the raw/in-shader result
  // until the next fresh readback completes.  A converged (idle) run is NOT
  // such a case: the G-buffers still hold the final accumulated image, so the
  // last denoised result must be preserved and presented (dropping it just
  // before the viewport goes idle would flash the raw/edge-stopped image on
  // the converged output).
  if (!this->ptAccumulating && !this->ptConverged) {
    this->denoiseResultReady = FALSE;
  }
  if (!this->createStorageImage(target.extent.width, target.extent.height)) {
    this->emitError("recordAccelerationStructures: failed to create storage image");
    return false;
  }

  // The storage image is written by the raygen shader and sampled by the
  // present shader, both in GENERAL layout.  Transition it once per
  // (re)creation here, outside the caller's render pass (the pass has no
  // subpass self-dependency, so layout transitions cannot be recorded
  // inside it).
  if (this->storageImageNeedsLayoutInit && this->storageImage != VK_NULL_HANDLE) {
    VkImageMemoryBarrier imageBarrier {};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = this->storageImage;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.layerCount = 1;
    imageBarrier.srcAccessMask = 0;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    this->storageImageNeedsLayoutInit = false;
  }

  // BLAS builds (lazy per command; only for entries without an AS) and
  // refits (topology-stable position edits).  The builds also (re)upload
  // the triangle-normal pool entries consumed by updateMaterials(), which
  // therefore runs after this loop.
  this->statBlasBuilt = 0;
  this->statBlasRefit = 0;
  this->statBlasReused = 0;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_TRANSPARENT ||
        command.pass == SO_RENDERPASS_OVERLAY) continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    RTXCachedGeometry & entry = this->geometryCache[found->second];
    if (entry.refitPending) {
      // A compacted BLAS cannot be MODE_UPDATE'd safely (the compact copy is a
      // fresh AS); rebuild it instead of refitting in place.
      if (entry.compacted) {
        ++this->statBlasBuilt;
        if (entry.blas != VK_NULL_HANDLE) {
          this->deferDestroyCacheEntry(entry);
        }
        else {
          entry = RTXCachedGeometry();
        }
        if (!this->buildBlas(entry, command, cmd)) {
          this->emitError("recordAccelerationStructures: failed to rebuild "
                          "compacted BLAS");
          return false;
        }
      }
      else {
        ++this->statBlasRefit;
        if (!this->refitBlas(entry, command, cmd)) {
          this->emitError("recordAccelerationStructures: failed to refit BLAS");
          return false;
        }
      }
    }
    else if (entry.blas == VK_NULL_HANDLE) {
      ++this->statBlasBuilt;
      if (!this->buildBlas(entry, command, cmd)) {
        this->emitError("recordAccelerationStructures: failed to build BLAS");
        return false;
      }
    }
    else {
      ++this->statBlasReused;
    }
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
      fprintf(stderr,
              "[RTDBG] blas frame=%u built=%u refit=%u reused=%u cache=%zu\n",
              params.frame, this->statBlasBuilt, this->statBlasRefit,
              this->statBlasReused, this->geometryCache.size());
  }

  // Decide whether the acceleration structures need (re)building this frame.
  // Geometry content changes (cacheChanged) or an instance-transform change
  // (asTransformChanged) both require an AS rebuild; a camera-only orbit does
  // not (the AS is world-space), so the whole buildTlas / NEE-pool / material /
  // descriptor phase is skipped and the previous frame's AS + descriptors are
  // reused for the re-trace.  After a camera move of a static scene this drops
  // the per-frame TLAS build (asGpu) to zero.
  //
  // Both flags are consumed HERE: cacheChanged was read (but kept) by
  // updatePathTracingState() above for its scene-change restart, and the AS
  // rebuild decision below is its other -- and final -- consumer.
  this->asDirty = this->cacheChanged || this->asTransformChanged;
  this->cacheChanged = false;
  this->asTransformChanged = false;

  if (this->asDirty) {
    // Alternate the descriptor pair so the set we (re)populate below is not
    // the one the previous, still-in-flight submission bound.  On non-dirty
    // frames the index is left untouched so the trace keeps binding the set
    // that was last populated -- the root cause of the alternate-frame black
    // flash was tracing through a set that had never been updated.
    this->descriptorSetIndex = (this->descriptorSetIndex + 1) & 1u;
    // Emissive-triangle pool for NEE.  Rebuilt only when the AS is dirty so
    // the baked object-to-world transforms stay fresh on transform-only
    // edits (which refit BLASes instead of rebuilding geometry).  Runs before
    // updateMaterials(), which carries the pool offsets into the RTMaterial
    // records.
    this->buildNeePool(drawlist);
    this->updateMaterials(drawlist);

    // TLAS build (instances reference the BLASes built above).  The TLAS
    // handle may change here, so refresh the binding-0 descriptor before the
    // trace phase runs.
    if (!this->buildTlas(drawlist, params, cmd)) {
      this->emitError("recordAccelerationStructures: failed to build TLAS");
      return false;
    }
    if (!this->updateDescriptors()) {
      this->emitError("recordAccelerationStructures: descriptor update failed");
      return false;
    }

    // Barrier: BLAS/TLAS builds -> ray tracing shaders.  Recorded here, still
    // outside the render pass (acceleration-structure builds and buffer copies
    // are not allowed inside one).
    VkMemoryBarrier asBarrier {};
    asBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    asBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    asBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &asBarrier, 0, nullptr, 0, nullptr);
  }

  // Guard the descriptor-validity invariant.  The trace phase below binds
  // rtDescriptorSets[descriptorSetIndex]; that set is written only inside the
  // asDirty block above, and a camera-only (non-dirty) frame reuses the
  // last-written set.  A resource teardown (device lost / re-init) resets the
  // sets to NULL while descriptorSetIndex carries over, so the first frame of
  // the new generation can be non-dirty and bind a freshly (re)allocated but
  // never-written set -> VUID-vkCmdDispatch-None-08114.  The torn set cannot be
  // referenced by an in-flight submission, so repopulating it here (still
  // outside the render pass) is legal and closes that window.
  if (this->tlas != VK_NULL_HANDLE && !this->rtSetValid[this->descriptorSetIndex]) {
    if (!this->updateDescriptors()) {
      this->emitError("recordAccelerationStructures: descriptor update failed");
      return false;
    }
    VkMemoryBarrier asBarrier {};
    asBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    asBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    asBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &asBarrier, 0, nullptr, 0, nullptr);
  }

  // --- Frame uniform data (host-visible; no barrier needed) --------------
  if (this->frameMapped) {
    RTXFrameBlock frame {};    SbMat viewValue;
    params.viewMatrix.getValue(viewValue);
    std::memcpy(frame.view, &viewValue[0][0], sizeof(float) * 16);

    SbMatrix viewInverse = params.viewMatrix.inverse();
    SbMat viValue;
    viewInverse.getValue(viValue);
    std::memcpy(frame.viewInverse, &viValue[0][0], sizeof(float) * 16);

    SbMatrix projInverse = params.projMatrix.inverse();
    SbMat piValue;
    projInverse.getValue(piValue);
    std::memcpy(frame.projInverse, &piValue[0][0], sizeof(float) * 16);

    SbMat pValue;
    params.projMatrix.getValue(pValue);

    frame.cameraPos[0] = frame.viewInverse[12];
    frame.cameraPos[1] = frame.viewInverse[13];
    frame.cameraPos[2] = frame.viewInverse[14];
    frame.cameraPos[3] = 1.0f;

    const SbVec2s & vpSize = params.viewport.getViewportSizePixels();
    frame.viewport[0] = static_cast<float>(vpSize[0]);
    frame.viewport[1] = static_cast<float>(vpSize[1]);
    // Orthographic flag: Coin's ortho projection is affine (proj[3][3] == 1)
    // while the perspective projection has proj[3][3] == 0.  The compute
    // tracer picks parallel vs fanning primary rays from this bit.
    frame.viewport[2] = params.projMatrix[3][3] != 0.0f ? 1.0f : 0.0f;
    frame.viewport[3] = 0.0f;

    // Sky: the viewport background gradient by default; when an environment
    // preset (cubemap) is selected it overrides these so the analytic
    // envSkyColor samples the cubemap sky rather than the viewport's colors.
    if (this->envMapId >= 0) {
      frame.bgTop[0] = this->envSkyTop[0];
      frame.bgTop[1] = this->envSkyTop[1];
      frame.bgTop[2] = this->envSkyTop[2];
      frame.bgBottom[0] = this->envSkyBottom[0];
      frame.bgBottom[1] = this->envSkyBottom[1];
      frame.bgBottom[2] = this->envSkyBottom[2];
    }
    else {
      frame.bgTop[0] = params.backgroundTopColor[0];
      frame.bgTop[1] = params.backgroundTopColor[1];
      frame.bgTop[2] = params.backgroundTopColor[2];
      frame.bgBottom[0] = params.backgroundBottomColor[0];
      frame.bgBottom[1] = params.backgroundBottomColor[1];
      frame.bgBottom[2] = params.backgroundBottomColor[2];
    }
    frame.bgTop[3] = 1.0f;
    frame.bgBottom[3] = 1.0f;

    frame.state[0] = static_cast<float>(this->ptFrameIndex);
    // u_state.y selects the ray-tracer path in the compute shader:
    //   0 = single-primary-ray direct-lighting preview (raster/preview)
    //   1 = multi-bounce path tracing (accumulating)
    //   2 = real-time ambient occlusion (single sample, occlusion rays)
    //   3 = environment / IBL preview (single sample, sky-lit)
    //   4 = debug constant fill (FC_VULKAN_RT_DEBUG_FILL)
    // AO (mode 2) and the Environment preview (mode 3) are real-time
    // previews: they never accumulate, so they must also force the
    // accumulate flag off to keep the state machine honest.
    frame.state[1] = COIN_VULKAN_ENV_FLAG("FC_VULKAN_RT_DEBUG_FILL")
      ? 4.0f
      : (this->rtxViewMode == RtxViewMode::RtxModeAmbientOcclusion ? 2.0f
         : (this->rtxViewMode == RtxViewMode::RtxModeEnvironment ? 3.0f
            : (this->ptEnabled ? 1.0f : 0.0f)));
    frame.state[2] = this->ptAccumulating ? 1.0f : 0.0f;
    frame.state[3] = static_cast<float>(this->ptMaxBounces);

    // Adaptive sampling parameters; a zero threshold disables the early-out.
    frame.adaptive[0] = static_cast<float>(this->ptAdaptiveMinSamples);
    frame.adaptive[1] = this->ptAdaptiveEnabled
      ? this->ptAdaptiveThreshold : 0.0f;
    frame.adaptive[2] = this->ptFireflySigma;
    // u_adaptive.w gates the per-pixel freeze.  Zero disables it: while the
    // camera is reprojecting (ptReprojectFrame), or on the very frame a move
    // just concluded (ptWasMoving still latched into the run), the history a
    // pixel's variance is measured against is per-pixel mismatched, so a pixel
    // could pass the convergence test against a stale mean and freeze at a
    // value that does not match its neighbors.  Only allow the freeze once the
    // camera has been static for a frame (neither condition), so converged
    // pixels from a clean run keep early-outing but no pixel locks during or
    // immediately after a move.
    frame.adaptive[3] = (this->ptReprojectFrame || this->ptWasMoving ||
                         this->ptForceFullResolve)
      ? 0.0f : 1.0f;

    // Temporal reprojection: the previous frame's camera (world -> clip)
    // and the reproject-this-frame flag.
    std::memcpy(frame.prevViewProj, this->prevViewProj, sizeof(float) * 16);
    frame.temporal[0] = this->ptReprojectFrame ? 1.0f : 0.0f;
    frame.temporal[1] = 0.0f;
    frame.temporal[2] = 0.0f;
    frame.temporal[3] = 0.0f;

    // NEE: emissive-triangle pool size plus the per-run switches.  NEE and
    // MIS default ON; FC_VULKAN_PT_NEE=0 restores the BSDF-only emissive
    // path, FC_VULKAN_PT_MIS=0 drops the balance weight (BSDF hits of
    // emissive surfaces then contribute nothing, avoiding double count).
    frame.nee[0] = static_cast<float>(this->neePoolCount);
    frame.nee[1] = this->rtNeeEnabled ? 1.0f : 0.0f;
    frame.nee[2] = this->rtMisEnabled ? 1.0f : 0.0f;
    frame.nee[3] = 0.0f;

    // Procedural IBL environment params (RtxModeEnvironment): normalized
    // world-space sun direction plus the sky/sun shading scales.  The sky
    // brightness folds into u_env.x so no extra block field is needed.
    // When no environment preset is active (envMapId < 0) the intensity is
    // driven to zero so the PT/preview miss falls back to the plain viewport
    // gradient and the default view is unchanged.
    frame.env[0] = this->envMapId >= 0
      ? this->envIntensity * this->envSkyBrightness : 0.0f;
    float sdx = this->envSunDir[0];
    float sdy = this->envSunDir[1];
    float sdz = this->envSunDir[2];
    float sunLen = std::sqrt(sdx * sdx + sdy * sdy + sdz * sdz);
    if (sunLen > 1e-6f) { sdx /= sunLen; sdy /= sunLen; sdz /= sunLen; }
    frame.env[1] = sdx;
    frame.env[2] = sdy;
    frame.env[3] = sdz;
    frame.envColor[0] = this->envSunColor[0];
    frame.envColor[1] = this->envSunColor[1];
    frame.envColor[2] = this->envSunColor[2];
    frame.envColor[3] = this->envSunPower;

    // Room-cove environment: when envMapMode is set the shader traces a
    // camera-centered box (floor + four walls + ceiling) instead of the sky
    // gradient, so the cubemap reads as a real scene (desk / table / white
    // lab) whose reflections carry the room.  Straightforward packing of the
    // per-preset room parameters; all unused components are zeroed.
    frame.envRoom[0] = this->envWallColor[0];
    frame.envRoom[1] = this->envWallColor[1];
    frame.envRoom[2] = this->envWallColor[2];
    frame.envRoom[3] = this->envMapMode == 1 ? 1.0f : 0.0f;
    frame.envRoomFloor[0] = this->envFloorColor[0];
    frame.envRoomFloor[1] = this->envFloorColor[1];
    frame.envRoomFloor[2] = this->envFloorColor[2];
    frame.envRoomFloor[3] = this->envRoomFloorY;
    frame.envRoomCeil[0] = this->envCeilColor[0];
    frame.envRoomCeil[1] = this->envCeilColor[1];
    frame.envRoomCeil[2] = this->envCeilColor[2];
    frame.envRoomCeil[3] = this->envRoomCeilY;
    frame.envRoomScale[0] = this->envRoomHalfExtent;

    std::memcpy(this->frameMapped, &frame, sizeof(frame));

    // Present frame UBO: the traced camera's world->view and view->clip
    // matrices (row-major SbMat copied directly, matching the shader's
    // column-major interpretation exactly as frame.view/u_view does).  The
    // present pass projects the first-bounce hit position through these to
    // write scene depth, letting the composite edge overlay occlude hidden
    // edges.
    if (this->presentFrameMapped) {
      float * pf = static_cast<float *>(this->presentFrameMapped);
      std::memcpy(pf, &viewValue[0][0], sizeof(float) * 16);
      std::memcpy(pf + 16, &pValue[0][0], sizeof(float) * 16);
    }

    if (getenv("FC_VULKAN_RT_DEBUG")) {
      static uint32_t debugFrame = 0;
      if ((debugFrame++ % 120) == 0) {
        fprintf(stderr,
                "[RTDBG] frame: cam=(%.2f,%.2f,%.2f) vp=(%.0fx%.0f) "
                "bgTop=(%.2f,%.2f,%.2f) bgBottom=(%.2f,%.2f,%.2f) "
                "state=(%.0f,%.0f,%.0f,%.0f) "
                "viewR0=(%.2f,%.2f,%.2f) viewR1=(%.2f,%.2f,%.2f) "
                "viewR2=(%.2f,%.2f,%.2f) ortho=%.0f "
                "projDiag=(%.3f,%.3f,%.3f,%.3f)\n",
                frame.cameraPos[0], frame.cameraPos[1], frame.cameraPos[2],
                frame.viewport[0], frame.viewport[1],
                frame.bgTop[0], frame.bgTop[1], frame.bgTop[2],
                frame.bgBottom[0], frame.bgBottom[1], frame.bgBottom[2],
                frame.state[0], frame.state[1], frame.state[2],
                frame.state[3],
                frame.view[0], frame.view[1], frame.view[2],
                frame.view[4], frame.view[5], frame.view[6],
                frame.view[8], frame.view[9], frame.view[10],
                frame.viewport[2],
                frame.projInverse[0], frame.projInverse[5],
                frame.projInverse[10], frame.projInverse[15]);
      }
    }
  }

  // --- Trace --------------------------------------------------------------
  // The path tracer runs as a ray tracing pipeline dispatched with
  // vkCmdTraceRaysKHR outside the caller's render pass (ray tracing is not
  // allowed inside one).  The raygen receives its frame state through the
  // 16-byte push constant block; the descriptor set stays as updated after
  // the TLAS (re)build above.
  if (this->tlas == VK_NULL_HANDLE) {
    // No traceable geometry (empty scene; the view's zero-scaled anchor cube
    // is filtered out upstream): the TLAS was never built and the descriptor
    // set was never populated, so recording the trace would dispatch against
    // an invalid acceleration-structure binding.  Clear the storage image to
    // the background color instead so the present pass shows the viewport
    // backdrop rather than undefined memory.
    VkClearColorValue clearColor {};
    clearColor.float32[0] = this->lastBgTopColors[0];
    clearColor.float32[1] = this->lastBgTopColors[1];
    clearColor.float32[2] = this->lastBgTopColors[2];
    clearColor.float32[3] = 1.0f;
    VkImageSubresourceRange fullRange {};
    fullRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    fullRange.levelCount = 1;
    fullRange.layerCount = 1;
    vkCmdClearColorImage(cmd, this->storageImage, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &fullRange);
  }
  else if (this->useSbtPipeline) {
    RTXRaygenPush raygenPush;
    raygenPush.frameIndex = this->ptFrameIndex;
    raygenPush.flags = (this->ptEnabled ? 1u : 0u) |
      (this->ptAccumulating ? 2u : 0u) |
      (COIN_VULKAN_ENV_FLAG("FC_VULKAN_RT_DEBUG_FILL") ? 4u : 0u);
    raygenPush.maxBounces = this->ptMaxBounces;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      this->rtPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            this->rtPipelineLayout, 0, 1,
                            &this->rtDescriptorSets[this->descriptorSetIndex],
                            0, nullptr);
    vkCmdPushConstants(cmd, this->rtPipelineLayout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0,
                       sizeof(raygenPush), &raygenPush);
    vkCmdTraceRaysKHR(cmd, &this->raygenSbtRegion, &this->missSbtRegion,
                      &this->hitSbtRegion, &this->callableSbtRegion,
                      this->storageWidth, this->storageHeight, 1);
  }
  else {
    // Ray-query compute dispatch (default): the state rides in the frame
    // UBO (u_state) and no push constants are needed.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      this->computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->rtPipelineLayout, 0, 1,
                            &this->rtDescriptorSets[this->descriptorSetIndex],
                            0, nullptr);
    vkCmdDispatch(cmd, (this->storageWidth + 7) / 8,
                  (this->storageHeight + 7) / 8, 1);
  }

  VkMemoryBarrier traceBarrier {};
  traceBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  traceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  traceBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1,
                       &traceBarrier, 0, nullptr, 0, nullptr);

  // Denoiser readback: only on the target frame that reached the sample count
  // (ptDenoisePending).  Every other accumulating frame presents the in-shader
  // edge-stopped running mean, so it needs no G-buffer readback; copying every
  // frame only to denoise a changing partial is the churn the denoise-at-target
  // design removes.
  if (this->denoiserActive && this->ptEnabled && this->ptAccumulating &&
      this->ptDenoisePending && !this->oidnWorkerRunning) {
    this->recordDenoiseReadback(cmd);
  }
  return true;
}

bool
SoRTXRenderBackend::recordTraceAndPresent(const SoRenderParams & params,
                                           const SoVulkanRenderTarget & target,
                                           VkCommandBuffer cmd,
                                           VkRenderPass renderPass)
{
   if (!this->createStorageImage(target.extent.width, target.extent.height)) {
    this->emitError("recordTraceAndPresent: failed to create storage image");
    return false;
  }
  if (!this->createPresentPipeline(renderPass, target.sampleCount)) {
    this->emitError("recordTraceAndPresent: failed to create present pipeline");
    return false;
  }

  // NOTE: descriptor sets are intentionally NOT updated here.  The AS phase
  // already refreshed them (storage image creation, PT buffers, TLAS build),
  // and updating a set that the trace bound above would invalidate this
  // still-recording command buffer (VUID-vkUpdateDescriptorSets-None-03047).

  // The trace itself was recorded in the acceleration-structure phase:
  // vkCmdTraceRaysKHR cannot be issued inside a render pass instance, and
  // the caller's pass also has no subpass self-dependency for layout
  // transitions.  Only the present pass (a plain fullscreen draw) is
  // recorded here.

  // Present fullscreen triangle into the swapchain color attachment.
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, this->presentPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          this->presentPipelineLayout, 0, 1,
                          &this->presentDescriptorSets[this->descriptorSetIndex],
                          0, nullptr);

  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  VkViewport viewport {};
  viewport.x = static_cast<float>(origin[0]);
  viewport.y = static_cast<float>(static_cast<int32_t>(target.extent.height) -
                                 static_cast<int32_t>(origin[1]) -
                                 static_cast<int32_t>(size[1]));
  viewport.width = static_cast<float>(size[0]);
  viewport.height = static_cast<float>(size[1]);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor {};
  scissor.offset = {0, 0};
  scissor.extent = target.extent;
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  // Present push constant: u_present = (width, height, denoiseOn,
  // frameIndex), u_origin = (ox, oy, 0, 0), u_denoise = (denoisedReady,
  // scale, 0, 0).  The present shader uses u_denoise.x to decide whether to
  // sample the denoised output (binding 5) or the in-shader edge-stopping
  // filter, and u_denoise.y as the upscale scale.  u_present.z must remain
  // the path-tracing + denoise toggle (it selects the denoising branch vs
  // the raw-image preview).
  //
  // u_present.z selects the present shader branch: >=0.5 reads the
  // path-traced accumulation buffer (binding 2) with its denoise/edge-stopped
  // filters; <0.5 samples the storage image (binding 1) directly.  Only the
  // multi-bounce path-tracing mode writes the accumulation buffer.  The
  // single-sample AO and Environment previews (and the raster off state)
  // render straight into the storage image, so presenting them through the
  // accum branch reads a never-written buffer and shows black.  Gate the
  // accum branch on the path-traced mode, not on ptEnabled (which is also
  // true for AO/Environment).
  //
  // The accumulation buffer is written ONLY while the run is actively
  // accumulating (ptAccumulating) or has converged (ptConverged).  During the
  // pre-settle preview and the post-move 1-spp preview (and when geometry is
  // missing so the run never starts) the raygen writes the storage image, not
  // the accumulation buffer.  Selecting the accum branch in that state reads a
  // never-written buffer and presents pure black, so require the accum state
  // here too.  This is the "screen goes black" guard: a preview or an invalid
  // (empty-cache) run falls through to the storage-image branch.
  const bool pathTraceMode =
    this->rtxViewMode == RtxViewMode::RtxModePathTrace;
  const bool accumBufferValid = this->ptAccumulating || this->ptConverged;
  const float presentPush[12] = {
    static_cast<float>(size[0]),
    static_cast<float>(size[1]),
    pathTraceMode && this->ptEnabled && this->ptDenoise && accumBufferValid
      ? 1.0f : 0.0f,
    static_cast<float>(this->ptFrameIndex),
    static_cast<float>(origin[0]),
    static_cast<float>(origin[1]),
    0.0f,
    0.0f,
    (this->denoiseResultReady && (this->ptAccumulating || this->ptConverged))
      ? 1.0f : 0.0f,
    this->denoiseEffectiveScale,
    0.0f,
    0.0f};
  if (getenv("FC_VULKAN_PT_DENOISE_TIMING")) {
    fprintf(stderr,
            "[DENOISE-STATE] ord=%u frame=%u accum=%d pend=%d ready=%d "
            "denoise=%d kind=%d\n",
            params.frame, this->ptFrameIndex, this->ptAccumulating ? 1 : 0,
            this->ptDenoisePending ? 1 : 0, this->denoiseResultReady ? 1 : 0,
            this->ptDenoise ? 1 : 0, static_cast<int>(this->denoiseKind));
  }
  vkCmdPushConstants(cmd, this->presentPipelineLayout,
                     VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(presentPush), presentPush);

  vkCmdDraw(cmd, 3, 1, 0, 0);
  return true;
}
