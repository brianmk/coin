// src/rendering/vulkan/manager/SoVulkanRenderManager.cpp

#include <Inventor/rendering/vulkan/SoVulkanRenderManager.h>

#include <Inventor/SbViewportRegion.h>
#include <Inventor/SbXfBox3f.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoTransformSeparator.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/vulkan/SoVulkanRenderTarget.h>

#include "rendering/backend/SoRenderBackend.h"
#include "rendering/backend/SoClippingPlanes.h"
#include "rendering/backend/SoRenderIRP.h"
#include "rendering/vulkan/raster/SoVulkanRenderBackend.h"
#include "rendering/vulkan/raytracing/rtx/SoRTXRenderBackend.h"
#include "rendering/vulkan/common/core/SoVulkanShared.h"
#include "rendering/vulkan/common/core/SoVulkanConfig.h"

class SoVulkanRenderManagerP;

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include "rendering/vulkan/manager/SoVulkanRenderManagerP.h"

using namespace SoVulkanManagerDetail;

// Mark the graph fingerprint dirty when any part of the main scene is notified
// (a field write or child-list edit anywhere in the subtree) -- the exact
// condition the O(N) fingerprint walk detects, so it can be skipped until the
// scene actually changes.
void
vulkanSceneGraphChangedCallback(void * data, SoSensor * /*sensor*/)
{
  auto * pimpl = static_cast<SoVulkanRenderManagerP *>(data);
  pimpl->sceneGraphDirty = TRUE;
}


SoVulkanRenderManager::SoVulkanRenderManager()
  : pimpl(new SoVulkanRenderManagerP)
{
}

SoVulkanRenderManager::~SoVulkanRenderManager()
{
  // Shut down through the manager entry point so both the raster and RTX
  // backends release their resources in the documented order and the shared
  // init context is invalidated.  Letting only the raster backend shut down
  // here left the RTX backend's deferred destruction state to its implicit
  // member destructor, which can run after the manager has already torn down
  // surrounding state.
  this->shutdown();
  delete this->pimpl;
}

void
SoVulkanRenderManager::setSceneGraph(SoNode * root)
{
  if (this->pimpl->scene == root) {
    return;
  }
  setRetainedNode(this->pimpl->scene, root);
  // The bbox is cached in world space; a different scene graph invalidates it.
  this->pimpl->sceneBBoxCached = false;
  this->pimpl->sceneBBoxScene = nullptr;
  // Re-arm the graph-fingerprint dirty sensor on the new scene: detach from the
  // previous scene and attach to the new one, and mark the fingerprint dirty so
  // the next frame re-walks rather than trusting a stale cached fingerprint.
  if (this->pimpl->sceneGraphSensor) {
    this->pimpl->sceneGraphSensor->detach();
    if (root) {
      this->pimpl->sceneGraphSensor->attach(root);
    }
    this->pimpl->sceneGraphDirty = TRUE;
  }
}

SoNode *
SoVulkanRenderManager::getSceneGraph(void) const
{
  return this->pimpl->scene;
}

void
SoVulkanRenderManager::setOverlaySceneGraph(SoNode * root)
{
  setRetainedNode(this->pimpl->overlayScene, root);
}

SoNode *
SoVulkanRenderManager::getOverlaySceneGraph(void) const
{
  return this->pimpl->overlayScene;
}

void
SoVulkanRenderManager::setDecorationSceneGraph(SoNode * root)
{
  setRetainedNode(this->pimpl->decorationScene, root);
}

SoNode *
SoVulkanRenderManager::getDecorationSceneGraph(void) const
{
  return this->pimpl->decorationScene;
}

void
SoVulkanRenderManager::setCamera(SoCamera * camera)
{
  // The scene graph is the single camera authority and stays that way: this
  // retained pointer is only a fallback/hint for scenes that carry no camera
  // (overlay-only or off-screen render setups).  When the scene DOES contain a
  // camera, refreshActiveCamera() re-resolves it from the scene graph each
  // frame, overrides this pointer, and owns that node's lifetime -- so the
  // GL viewer and every Vulkan backend (raster and path tracing) all render
  // from the same camera node with no sync between them.
  // FreeCAD replaces the camera node when the user toggles between the
  // perspective and orthographic views.  Without a reference the old node is
  // destroyed and this raw pointer dangles, crashing the next render
  // (segfault in setClippingPlanes / SoBase::isOfType).  Keep the camera
  // alive for as long as the manager references it.
  setRetainedNode(this->pimpl->camera, camera);
}

SoCamera *
SoVulkanRenderManager::getCamera(void) const
{
  return this->pimpl->camera;
}

void
SoVulkanRenderManager::setAutoClipping(AutoClippingStrategy strategy)
{
  this->pimpl->autoClipping = strategy;
}

SoVulkanRenderManager::AutoClippingStrategy
SoVulkanRenderManager::getAutoClipping(void) const
{
  return this->pimpl->autoClipping;
}

void
SoVulkanRenderManager::setNearPlaneValue(float value)
{
  this->pimpl->nearplanevalue = value;
}

float
SoVulkanRenderManager::getNearPlaneValue(void) const
{
  return this->pimpl->nearplanevalue;
}

void
SoVulkanRenderManager::setViewportRegion(const SbViewportRegion & region)
{
  this->pimpl->viewportRegion = region;
}

const SbViewportRegion &
SoVulkanRenderManager::getViewportRegion(void) const
{
  return this->pimpl->viewportRegion;
}

void
SoVulkanRenderManager::setBackgroundColor(const SbColor4f & color)
{
  this->pimpl->backgroundColor = color;
}

const SbColor4f &
SoVulkanRenderManager::getBackgroundColor(void) const
{
  return this->pimpl->backgroundColor;
}

void
SoVulkanRenderManager::setDevicePixelRatio(float ratio)
{
  this->pimpl->devicePixelRatio = ratio;
}

float
SoVulkanRenderManager::getDevicePixelRatio(void) const
{
  return this->pimpl->devicePixelRatio;
}

void
SoVulkanRenderManager::setExternalRevision(uint64_t revision)
{
  this->pimpl->externalRevision = revision;
}

void
SoVulkanRenderManager::setBackgroundGradient(SbBool enabled,
                                              const SbColor4f & topColor,
                                              const SbColor4f & bottomColor)
{
  this->pimpl->backgroundGradient = enabled;
  this->pimpl->backgroundTopColor = topColor;
  this->pimpl->backgroundBottomColor = bottomColor;
}

void
SoVulkanRenderManager::setWireframeOverlay(SbBool enabled)
{
  this->pimpl->wireframeOverlay = enabled;
  this->pimpl->backend.setWireframeOverlay(enabled);
}

void
SoVulkanRenderManager::setPointsOverlay(SbBool enabled)
{
  this->pimpl->pointsOverlay = enabled;
  this->pimpl->backend.setPointsOverlay(enabled);
  // Whether Part's BRep point set emits its base vertex markers is decided
  // during traversal, so the retained main draw list encodes it.  Invalidate
  // the replay so toggling "show vertices" takes effect on the next frame
  // instead of waiting for the scene graph to change.
  this->pimpl->graphFingerprintValid = FALSE;
  this->pimpl->sceneGraphDirty = TRUE;
}

void
SoVulkanRenderManager::setTessellationOverlay(SbBool enabled)
{
  this->pimpl->tessellationOverlay = enabled;
  this->pimpl->backend.setTessellationOverlay(enabled);
}

void
SoVulkanRenderManager::setEdgeOverlay(SbBool enabled)
{
  this->pimpl->edgeOverlay = enabled;
  this->pimpl->backend.setEdgeOverlayVisible(enabled);
}

void
SoVulkanRenderManager::setEdgeColor(const SbColor4f & color)
{
  this->pimpl->edgeColor = color;
  this->pimpl->backend.setEdgeColor(color);
}

void
SoVulkanRenderManager::setViewSettings(const SoVulkanViewSettings & settings)
{
  // Single diff for the whole blob: the individual setters are unconditional,
  // so re-applying an unchanged blob every frame would be pure waste.
  if (this->pimpl->viewSettingsApplied
      && settings == this->pimpl->viewSettings) {
    return;
  }
  this->pimpl->viewSettings = settings;
  this->pimpl->viewSettingsApplied = TRUE;

  this->setBackgroundColor(settings.backgroundColor);
  this->setBackgroundGradient(settings.backgroundGradient,
                              settings.backgroundTop,
                              settings.backgroundBottom);
  this->setWireframeOverlay(settings.wireframeOverlay ? TRUE : FALSE);
  this->setPointsOverlay(settings.pointsOverlay ? TRUE : FALSE);
  this->setTessellationOverlay(settings.tessellationOverlay ? TRUE : FALSE);
  this->setEdgeOverlay(settings.edgeOverlay ? TRUE : FALSE);
  this->setEdgeColor(settings.edgeColor);
  // Raster HDR output transform (no-op when disabled).
  this->pimpl->backend.setHdrOutput(settings.hdrOutput ? TRUE : FALSE,
                                    settings.hdrExposure,
                                    settings.hdrToneMap);

  // The RTX-forwarded fields are only meaningful once the RT backend exists;
  // applying them earlier emits the "not initialized" warnings.  A raster-only
  // view never builds it, and a later build is covered by
  // invalidateViewSettings() (the renderer invalidates when the RTX engine is
  // created), which re-applies this blob.
  if (this->pimpl->rtxBackendInitialized) {
    this->setViewMode(settings.viewMode);
    this->setEnvMap(settings.envMap);
    this->setPathTracingBounces(
      static_cast<uint32_t>(settings.pathTracingBounces));
    this->setPathTracingSettleFrames(
      static_cast<uint32_t>(settings.pathTracingSettleFrames));
    this->setPathTracingMaxSamples(
      static_cast<uint32_t>(settings.pathTracingMaxSamples));
    this->setPathTracingDenoiseEnabled(settings.pathTracingDenoise ? TRUE
                                                                    : FALSE);
    this->setPathTracingDenoiser(settings.pathTracingDenoiser.empty()
                                   ? nullptr
                                   : settings.pathTracingDenoiser.c_str());
    this->setPathTracingDenoiserScale(settings.pathTracingDenoiserScale);
    this->setPathTracingGlass(settings.pathTracingGlassIor,
                              settings.pathTracingGlassAbsorption);
    this->setHdrOutput(settings.hdrOutput ? TRUE : FALSE,
                       settings.hdrExposure,
                       settings.hdrToneMap);
    // Re-apply the interaction-LOD state: the RT backend starts with it off,
    // so a bring-up after this state was set (device re-init / lazy RT build)
    // must pick it up.  Idempotent (the setter early-returns when unchanged).
    this->setInteractionLod(this->pimpl->interactionLod);
  }
}

void
SoVulkanRenderManager::invalidateViewSettings(void)
{
  this->pimpl->viewSettingsApplied = FALSE;
}

SbBool
SoVulkanRenderManager::getWireframeOverlay(void) const
{
  return this->pimpl->wireframeOverlay;
}

SbBool
SoVulkanRenderManager::getPointsOverlay(void) const
{
  return this->pimpl->pointsOverlay;
}

SbBool
SoVulkanRenderManager::getTessellationOverlay(void) const
{
  return this->pimpl->tessellationOverlay;
}

SbBool
SoVulkanRenderManager::getEdgeOverlay(void) const
{
  return this->pimpl->edgeOverlay;
}

const SbColor4f &
SoVulkanRenderManager::getEdgeColor(void) const
{
  return this->pimpl->edgeColor;
}

void
SoVulkanRenderManager::setClearEnabled(SbBool clearwindow, SbBool clearzbuffer)
{
  this->pimpl->clearWindow = clearwindow;
  this->pimpl->clearDepth = clearzbuffer;
}

void
SoVulkanRenderManager::getClearEnabled(SbBool & clearwindow,
                                       SbBool & clearzbuffer) const
{
  clearwindow = this->pimpl->clearWindow;
  clearzbuffer = this->pimpl->clearDepth;
}

SbBool
SoVulkanRenderManager::initialize(SoVulkanDeviceContext * context)
{
  // QVulkanWindow invokes renderer::initResources() on every Expose/Hide/
  // Resize/Move event, which re-enters this method.  Those events only
  // recreate the swapchain (a separate initSwapChainResources() call for the
  // frame-level resources); the Vulkan instance/device/queue survive.  When
  // the device context is unchanged this is NOT a device reset, so tearing
  // down and rebuilding both backends -- and every cached BLAS/acceleration
  // structure and RT pipeline -- again would be pure waste and the dominant
  // cost while navigating in the raster path.  Keep the live backends and
  // their caches alive; just refresh the retained context (a new stack-allocated
  // context points at the same window-owned device handles).
  //
  // NOTE (known Qt-side validation artifact, not a FreeCAD defect): with the
  // Vulkan validation layer enabled, the first few frames of a freshly shown
  // window may log
  //   "vkQueueSubmit(): pSubmits[0].pSignalSemaphores[0] ... may still be in
  //    use by VkSwapchainKHR ..." (VUID-vkQueueSubmit-pSignalSemaphores-00067).
  // That submit is QVulkanWindow's OWN internal present, not one from this
  // manager: we never call vkQueuePresentKHR/vkAcquireNextImage and never
  // submit a signal semaphore (our submissions are fence-based; the only
  // signal-semaphore use is the CUDA-interop external semaphores in
  // SoRTXRenderBackendDenoise.cpp).  QVulkanWindow reuses its per-swapchain
  // render-finished semaphore during initial swapchain/surface setup, which
  // the validation layer flags.  It is transient (fires at viewport open
  // before path tracing starts), is emitted only with validation enabled, and
  // has never been observed to cause VK_ERROR_DEVICE_LOST or any functional
  // degradation in this project.  It is unrelated to the RT descriptor-set
  // device-lost fixed in SoRTXRenderBackend* (VUID-vkCmdDispatch-None-08114).
  // Do not chase it: the remedy (per-swapchain-image semaphores) is a Qt/
  // QVulkanWindow change, not a FreeCAD one.
  if (this->pimpl->backendInitialized && this->pimpl->initContext
      && this->pimpl->initContext->device == context->device) {
    this->pimpl->initContext = context;
    return TRUE;
  }

  SoRenderBackendInitParams params;
  params.userData = context;
  // Forward the persistent pipeline-cache path before the backend creates its
  // VkPipelineCache (createPipelineCache() reads it once, during initialize()).
  this->pimpl->backend.setPipelineCachePath(this->pimpl->pipelineCachePath);
  if (!this->pimpl->backend.initialize(params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::initialize",
                              "backend initialization failed");
    return FALSE;
  }
  this->pimpl->backendInitialized = TRUE;
  // One-shot, after a successful device init (the early return above skips
  // re-entry), so FC_VULKAN_BACKEND_DEBUG runs get a resolved-config dump.
  if (SoVulkanConfig::get().debug.backendDebug) {
    SoVulkanConfig::dump();
  }
  // Retain the borrowed context so ensureRayTracing() can bring the RT
  // backend up later if it was skipped at startup (path tracing off).
  this->pimpl->initContext = context;

  // Ray tracing is best-effort and only attempted when it was requested
  // (setRayTracing(TRUE)).  A device created without the KHR extensions
  // (ray tracing disabled) can never resolve the RT entry points, so
  // probing it anyway just emits a misleading error every time the window
  // re-initializes.  When it IS requested but unavailable, fall back to
  // the raster backend with a warning.
  if (this->pimpl->rayTracing) {
    if (this->pimpl->rtxBackend.initialize(params)) {
      this->pimpl->rtxBackendInitialized = TRUE;
    }
    else {
      SoDebugError::postWarning(
        "SoVulkanRenderManager::initialize",
        "ray-tracing backend unavailable; raster Vulkan backend will be used");
    }
  }
  return TRUE;
}

void
SoVulkanRenderManager::shutdown(void)
{
  if (this->pimpl->backendInitialized) {
    this->pimpl->backend.shutdown();
    this->pimpl->backendInitialized = FALSE;
  }
  if (this->pimpl->rtxBackendInitialized) {
    this->pimpl->rtxBackend.shutdown();
    this->pimpl->rtxBackendInitialized = FALSE;
  }
  // The retained context is only valid for the window's device/queue
  // lifetime, which ends around releaseResources(); do not reuse it after.
  this->pimpl->initContext = nullptr;
}

void
SoVulkanRenderManager::setRayTracing(SbBool enabled)
{
  // Pure request, honored by initialize() when it runs afterwards, or followed
  // by ensureRayTracing()/requestRayTracing() when the backend was skipped at
  // startup.  This method deliberately does NOT observe the current backend
  // state: converting "the RTX backend is not up yet" into "ray tracing can
  // never work" here is what broke a runtime raster -> path-tracing toggle
  // (setRayTracing(TRUE) after a raster-first initialize() used to warn and
  // clear the very flag the caller just raised, so the lazy build never ran).
  // Whether ray tracing actually came up is reported by getRayTracingActive();
  // the callers use that (or requestRayTracing()) to decide.
  this->pimpl->rayTracing = enabled;
}

SbBool
SoVulkanRenderManager::requestRayTracing(SbBool enabled)
{
  this->pimpl->rayTracing = enabled;
  if (!enabled) {
    if (this->pimpl->rtxBackendInitialized) {
      this->pimpl->rtxBackend.setPathTracingEnabled(FALSE);
    }
    return this->getRayTracingActive();
  }
  // enabled == TRUE: bring the RTX backend up if it is not already.
  if (this->pimpl->initContext) {
    // The lazy build (initialize the RT backend, warn on failure) lives in
    // ensureRayTracing(); reuse it rather than duplicating its "already
    // initialized / no context / build failed" branches.  The TRUE request
    // set above satisfies ensureRayTracing()'s requirement that the backend
    // was actually requested.
    if (!this->ensureRayTracing()) {
      // Build failed (device lacks ray tracing): clear the request so we keep
      // the raster backend -- the hard fall-back.
      this->pimpl->rayTracing = FALSE;
      return FALSE;
    }
  }
  else {
    // Before initialize(): no context yet, so the request is honored later by
    // initialize(); nothing is active.
    return FALSE;
  }
  // If a prior requestRayTracing(FALSE) disabled path tracing, re-enable it.
  this->pimpl->rtxBackend.setPathTracingEnabled(TRUE);
  return this->getRayTracingActive();
}

SbBool
SoVulkanRenderManager::ensureRayTracing(void)
{
  if (this->pimpl->rtxBackendInitialized) {
    return TRUE;
  }
  // Only bring the RT backend up when it was actually requested (setRayTracing
  // honored it) and a device context is available to initialize against.
  if (!this->pimpl->rayTracing) {
    return FALSE;
  }
  if (!this->pimpl->initContext) {
    SoDebugError::postWarning("SoVulkanRenderManager::ensureRayTracing",
                              "no device context available");
    return FALSE;
  }
  SoRenderBackendInitParams params;
  params.userData = this->pimpl->initContext;
  if (this->pimpl->rtxBackend.initialize(params)) {
    // Apply the frames-in-flight count stored before the RT backend existed.
    this->pimpl->rtxBackend.setMaxFramesInFlight(
      this->pimpl->maxFramesInFlight);
    this->pimpl->rtxBackendInitialized = TRUE;
    return TRUE;
  }
  return FALSE;
}

void
SoVulkanRenderManager::setMaxFramesInFlight(uint32_t count)
{
  this->pimpl->maxFramesInFlight = count;
  this->pimpl->backend.setMaxFramesInFlight(count);
  // The RT backend may not be built yet (path tracing is enabled lazily); the
  // stored count is applied in ensureRayTracing().  Forward immediately when it
  // is already up so a swapchain resize takes effect at once.
  if (this->pimpl->rtxBackendInitialized) {
    this->pimpl->rtxBackend.setMaxFramesInFlight(count);
  }
}

void
SoVulkanRenderManager::setPipelineCachePath(const std::string & path)
{
  // Stored and forwarded in initialize(), because createPipelineCache() runs
  // during the backend's initialize() and reads the path once.  Forwarding
  // here too covers a caller that sets it on an already-initialized manager
  // (the next backend initialize(), e.g. after a window reset, picks it up).
  this->pimpl->pipelineCachePath = path;
  this->pimpl->backend.setPipelineCachePath(path);
}

SbBool
SoVulkanRenderManager::getRayTracingActive(void) const
{
  return this->pimpl->rayTracing && this->pimpl->rtxBackendInitialized;
}

void
SoVulkanRenderManager::setPathTracingEnabled(SbBool enabled)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingEnabled",
                       "path tracing is unavailable",
                       [enabled](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingEnabled(enabled);
                       });
}

void
SoVulkanRenderManager::setViewMode(SoVulkanViewMode mode)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setViewMode",
                       "view mode is unavailable",
                       [mode](SoRTXRenderBackend & rtx) {
                         rtx.setViewMode(mode);
                       });
}

SoVulkanViewMode
SoVulkanRenderManager::getViewMode(void) const
{
  if (!this->pimpl->rtxBackendInitialized) return SoVulkanViewMode::RtxModeOff;
  return this->pimpl->rtxBackend.getViewMode();
}

void
SoVulkanRenderManager::setEnvMap(const int index)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setEnvMap",
                       "environment is unavailable",
                       [index](SoRTXRenderBackend & rtx) {
                         rtx.setEnvMap(index);
                       });
}

void
SoVulkanRenderManager::setSceneLights(const SoLightingData & lighting)
{
  // Both Vulkan backends share the authoritative viewer-light set.  The raster
  // executor would otherwise light from the world-fixed IR capture, so its
  // highlights would not follow the camera the way Coin GL (and the RT path)
  // do.  The raster backend is always available; the RT backend only once its
  // device resources are up (a later push reaches it).
  this->pimpl->backend.setSceneLights(lighting);
  if (this->pimpl->rtxBackendInitialized) {
    this->pimpl->rtxBackend.setSceneLights(lighting);
  }
}

int
SoVulkanRenderManager::getEnvMap(void) const
{
  if (!this->pimpl->rtxBackendInitialized) return -1;
  return this->pimpl->rtxBackend.getEnvMap();
}

int
SoVulkanRenderManager::getEnvMapCount(void)
{
  return SoRTXRenderBackend::getEnvMapCount();
}

const char *
SoVulkanRenderManager::getEnvMapName(const int index)
{
  return SoRTXRenderBackend::getEnvMapName(index);
}

SbBool
SoVulkanRenderManager::getPathTracingEnabled(void) const
{
  return this->pimpl->rtxBackendInitialized &&
    this->pimpl->rtxBackend.getPathTracingEnabled();
}

void
SoVulkanRenderManager::setPathTracingStart(SbBool start)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingStart",
                       "path tracing is unavailable",
                       [start](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingStart(start);
                       });
}

SbBool
SoVulkanRenderManager::getPathTracingActive(void) const
{
  return this->pimpl->rtxBackendInitialized &&
    this->pimpl->rtxBackend.getPathTracingActive();
}

SbBool
SoVulkanRenderManager::getPathTracingRefining(void) const
{
  return this->pimpl->rtxBackendInitialized &&
    this->pimpl->rtxBackend.getPathTracingRefining();
}

uint32_t
SoVulkanRenderManager::getPathTracingSampleCount(void) const
{
  if (!this->pimpl->rtxBackendInitialized) return 0;
  return this->pimpl->rtxBackend.getPathTracingSampleCount();
}

void
SoVulkanRenderManager::setPathTracingBounces(const uint32_t bounces)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingBounces",
                       "setting ignored",
                       [bounces](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingBounces(bounces);
                       });
}

void
SoVulkanRenderManager::setPathTracingSettleFrames(const uint32_t frames)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingSettleFrames",
                       "setting ignored",
                       [frames](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingSettleFrames(frames);
                       });
}

void
SoVulkanRenderManager::setInteractionLod(SbBool active)
{
  this->pimpl->interactionLod = active;
  // Routine, per-navigation state push, not a user-facing tuning setter: forward
  // it silently when the RT backend exists and remember it for the eventual
  // bring-up (setViewSettings re-applies it).  Routing this through withRtx()
  // emitted a "ray-tracing backend is not initialized" warning on every camera
  // move in a raster view, spamming the console on a huge scene.
  if (this->pimpl->rtxBackendInitialized) {
    this->pimpl->rtxBackend.setInteractionLod(active);
  }
}

void
SoVulkanRenderManager::setPathTracingMaxSamples(const uint32_t samples)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingMaxSamples",
                       "setting ignored",
                       [samples](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingMaxSamples(samples);
                       });
}

void
SoVulkanRenderManager::setPathTracingDenoiseEnabled(SbBool enabled)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingDenoiseEnabled",
                       "setting ignored",
                       [enabled](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingDenoiseEnabled(enabled);
                       });
}

void
SoVulkanRenderManager::setPathTracingDenoiser(const char * denoiser)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingDenoiser",
                       "setting ignored",
                       [denoiser](SoRTXRenderBackend & rtx) {
                         rtx.setDenoiserFilter(denoiser);
                       });
}

void
SoVulkanRenderManager::setPathTracingDenoiserScale(const float scale)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingDenoiserScale",
                       "setting ignored",
                       [scale](SoRTXRenderBackend & rtx) {
                         rtx.setDenoiserScale(scale);
                       });
}

void
SoVulkanRenderManager::setPathTracingGlass(const float ior,
                                           const float absorption)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setPathTracingGlass",
                       "setting ignored",
                       [ior, absorption](SoRTXRenderBackend & rtx) {
                         rtx.setPathTracingGlass(ior, absorption);
                       });
}

void
SoVulkanRenderManager::setHdrOutput(SbBool enabled, float exposure, int toneMap)
{
  this->pimpl->withRtx("SoVulkanRenderManager::setHdrOutput",
                       "setting ignored",
                       [enabled, exposure, toneMap](SoRTXRenderBackend & rtx) {
                         rtx.setHdrOutput(enabled, exposure, toneMap);
                       });
}

void
SoVulkanRenderManager::setRenderTarget(void * target)
{
  this->pimpl->renderTarget = target;
}

void *
SoVulkanRenderManager::getRenderTarget(void) const
{
  return this->pimpl->renderTarget;
}

SbBool
SoVulkanRenderManager::render(SbBool clearwindow, SbBool clearzbuffer)
{
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer, drawlist,
                                        params)) {
    return FALSE;
  }
  params.frame = ++this->pimpl->frameOrdinal;
  const bool rtActive = this->getRayTracingActive();
  // While ray tracing owns the scene triangles, the raster backend only draws
  // overlays/residue; tell it so its geometry sweep releases the traced meshes
  // instead of keeping a second resident copy.
  this->pimpl->backend.setOverlayCompositeMode(rtActive);
  if (rtActive) {
    if (!this->pimpl->rtxBackend.render(*drawlist, params)) {
      SoDebugError::postWarning("SoVulkanRenderManager::render",
                                "RT backend render failed (%d draw commands)",
                                drawlist->getNumCommands());
      return FALSE;
    }
    // The traced scene has no screen-space overlays (navigation cube);
    // composite them on top with the raster backend.
    if (!this->pimpl->backend.renderOverlaysOnly(*drawlist, params)) {
      SoDebugError::postWarning("SoVulkanRenderManager::render",
                                "overlay render failed (%d draw commands)",
                                drawlist->getNumCommands());
      return FALSE;
    }
    return TRUE;
  }
  if (!this->pimpl->backend.render(*drawlist, params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::render",
                              "backend render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderManager::renderExternal(SbBool clearwindow,
                                      SbBool clearzbuffer,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass,
                                      VkFramebuffer framebuffer)
{
  const long renderBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer,
                                        drawlist, params)) {
    return FALSE;
  }
  params.frame = ++this->pimpl->frameOrdinal;
  if (renderBcStart) {
    vkRenderBreadcrumbSince(renderBcStart, 5000, "renderExternal prepareRenderParams end");
  }
  const long backendBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  const bool rtActive = this->getRayTracingActive();
  // See render(): in RT mode the raster backend only composites, so its sweep
  // may release the traced triangle geometry the RT backend owns.
  this->pimpl->backend.setOverlayCompositeMode(rtActive);
  if (rtActive) {
    if (!this->pimpl->rtxBackend.renderExternal(*drawlist, params,
                                                commandBuffer, renderPass)) {
      SoDebugError::postWarning("SoVulkanRenderManager::renderExternal",
                                "RT backend render failed (%d draw commands)",
                                drawlist->getNumCommands());
      return FALSE;
    }
    // Screen-space overlays (navigation cube) are not part of the traced
    // scene; rasterize them on top with the raster backend.
    if (!this->pimpl->backend.renderExternalOverlay(*drawlist, params,
                                                    commandBuffer,
                                                    renderPass)) {
      SoDebugError::postWarning("SoVulkanRenderManager::renderExternal",
                                "overlay render failed (%d draw commands)",
                                drawlist->getNumCommands());
      return FALSE;
    }
    if (backendBcStart) {
      vkRenderBreadcrumbSince(backendBcStart, 5000, "renderExternal rtxBackend end");
    }
    return TRUE;
  }
  if (!this->pimpl->backend.renderExternal(*drawlist, params, commandBuffer,
                                           renderPass, framebuffer)) {
    SoDebugError::postWarning("SoVulkanRenderManager::renderExternal",
                              "backend render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  if (backendBcStart) {
    vkRenderBreadcrumbSince(backendBcStart, 5000, "renderExternal rasterBackend end");
  }
  return TRUE;
}

SbBool
SoVulkanRenderManager::renderExternalHdr(SbBool clearwindow,
                                         SbBool clearzbuffer,
                                         VkCommandBuffer commandBuffer,
                                         VkRenderPass outputPass,
                                         VkFramebuffer outputFramebuffer)
{
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer, drawlist,
                                        params)) {
    return FALSE;
  }
  params.frame = ++this->pimpl->frameOrdinal;
  // The HDR output path is raster-only; the RT modes present through their own
  // pass (see PresentFragment.glsl).
  this->pimpl->backend.setOverlayCompositeMode(FALSE);
  if (!this->pimpl->backend.renderExternalHdr(*drawlist, params, commandBuffer,
                                              outputPass, outputFramebuffer)) {
    SoDebugError::postWarning("SoVulkanRenderManager::renderExternalHdr",
                              "backend HDR render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderManager::isHdrRasterActive() const
{
  return (this->pimpl->viewSettings.hdrOutput
          && !this->getRayTracingActive())
    ? TRUE
    : FALSE;
}

SoVulkanRenderBackend *
SoVulkanRenderManager::getBackend(void) const
{
  return &this->pimpl->backend;
}

SoRTXRenderBackend *
SoVulkanRenderManager::getRayTracingBackend(void) const
{
  return this->pimpl->rtxBackendInitialized ? &this->pimpl->rtxBackend
                                            : nullptr;
}

bool
SoVulkanRenderManager::pickRay(const float origin[3], const float direction[3],
                               float tMax, VulkanPickHit & out) const
{
  out = VulkanPickHit {};
  SoRTXRenderBackend * rtx = this->getRayTracingBackend();
  if (!rtx) {
    return false;
  }
  SoRTXRenderBackend::RTPickHit hit;
  if (!rtx->pickRay(origin, direction, tMax, hit)) {
    return false;
  }
  out.hit = hit.hit;
  out.t = hit.t;
  out.worldPos[0] = hit.worldPos[0];
  out.worldPos[1] = hit.worldPos[1];
  out.worldPos[2] = hit.worldPos[2];
  out.commandIndex = hit.commandIndex;
  out.primitiveId = hit.primitiveId;
  out.userData = hit.userData;
  out.primitiveOffset = hit.primitiveOffset;
  return true;
}

uint32_t
SoVulkanRenderManager::getRenderFrameCount(void) const
{
  return this->pimpl->frameOrdinal;
}
