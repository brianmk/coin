// src/rendering/SoVulkanRenderManager.cpp

#include <Inventor/rendering/SoVulkanRenderManager.h>

#include <Inventor/SbViewportRegion.h>
#include <Inventor/SbXfBox3f.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include "rendering/SoRenderBackend.h"
#include "rendering/SoRenderIRP.h"
#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoRTXRenderBackend.h"

#include <cmath>
#include <limits>
#include <memory>

class SoVulkanRenderManagerP {
public:
  SoVulkanRenderManagerP()
    : irAction(SbViewportRegion())
  {
    this->viewportRegion.setWindowSize(1, 1);
  }

  ~SoVulkanRenderManagerP()
  {
    if (this->camera) {
      this->camera->unref();
    }
    if (this->scene) {
      this->scene->unref();
    }
    if (this->overlayScene) {
      this->overlayScene->unref();
    }
  }

  SoNode * scene = nullptr;
  SoNode * overlayScene = nullptr;
  SoCamera * camera = nullptr;
  SbViewportRegion viewportRegion;
  SbColor4f backgroundColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbBool backgroundGradient = FALSE;
  SbColor4f backgroundTopColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbColor4f backgroundBottomColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbBool wireframeOverlay = FALSE;
  SbBool pointsOverlay = FALSE;
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);
  SbBool clearWindow = TRUE;
  SbBool clearDepth = TRUE;
  void * renderTarget = nullptr;

  SoVulkanRenderManager::AutoClippingStrategy autoClipping =
    SoVulkanRenderManager::NO_AUTO_CLIPPING;
  float nearplanevalue = 0.6f;

  // Near/far planes computed by setClippingPlanes(), consumed by
  // prepareRenderParams().  Deliberately NOT written back into
  // SoCamera::nearDistance/farDistance: the camera node is shared with the
  // hidden GL viewer (FreeCAD), whose SoRenderManager concurrently writes
  // the same fields with its own GL-side values.  Reading those fields back
  // to build the projection matrix races with the GL manager and
  // intermittently renders with the wrong near plane -- the front face of
  // the object clips away and the interior shows through while rotating.
  // Keeping the Vulkan planes private to this manager makes the two
  // renderers independent and gives CAD-grade zoom behavior on both axes
  // (near plane hugs the closest geometry, far plane grows with distance).
  float computedNear = 1.0f;
  float computedFar = 10.0f;
  // Camera back-off along the view direction applied by the zoom wall (see
  // setClippingPlanes()); 0.0f when the camera is clear of the surface.
  float cameraShiftZ = 0.0f;

  SoIRRenderAction irAction;
  SoVulkanRenderBackend backend;
  SoRTXRenderBackend rtxBackend;
  SbBool backendInitialized = FALSE;
  SbBool rtxBackendInitialized = FALSE;
  SbBool rayTracing = FALSE;

  // Re-compute the camera near/far clipping planes from the scene bounding
  // box in camera coordinates.  Mirrors SoRenderManagerP::setClippingPlanes()
  // (the GL auto-clipping); the camera is a separate member here, so the
  // camera-to-world matrix is built directly from the camera node instead of
  // being looked up in the scene graph.
  void setClippingPlanes(void);

  // Traverse the scene and harvest view/projection into params.  Returns
  // FALSE when the backend or target is unavailable.
  SbBool prepareRenderParams(SbBool clearwindow,
                             SbBool clearzbuffer,
                             SoDrawList *& drawlist,
                             SoRenderParams & params);
};

SoVulkanRenderManager::SoVulkanRenderManager()
  : pimpl(new SoVulkanRenderManagerP)
{
}

SoVulkanRenderManager::~SoVulkanRenderManager()
{
  if (this->pimpl->backendInitialized) {
    this->pimpl->backend.shutdown();
  }
  delete this->pimpl;
}

void
SoVulkanRenderManager::setSceneGraph(SoNode * root)
{
  SoNode *& stored = this->pimpl->scene;
  if (stored == root) {
    return;
  }
  if (stored) {
    stored->unref();
  }
  stored = root;
  if (stored) {
    stored->ref();
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
  SoNode *& stored = this->pimpl->overlayScene;
  if (stored == root) {
    return;
  }
  if (stored) {
    stored->unref();
  }
  stored = root;
  if (stored) {
    stored->ref();
  }
}

SoNode *
SoVulkanRenderManager::getOverlaySceneGraph(void) const
{
  return this->pimpl->overlayScene;
}

void
SoVulkanRenderManager::setCamera(SoCamera * camera)
{
  // FreeCAD replaces the camera node when the user toggles between the
  // perspective and orthographic views.  Without a reference the old node is
  // destroyed and this raw pointer dangles, crashing the next render
  // (segfault in setClippingPlanes / SoBase::isOfType).  Keep the camera
  // alive for as long as the manager references it.
  SoCamera *& stored = this->pimpl->camera;
  if (stored == camera) {
    return;
  }
  if (stored) {
    stored->unref();
  }
  stored = camera;
  if (stored) {
    stored->ref();
  }
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
}

void
SoVulkanRenderManager::setPointsOverlay(SbBool enabled)
{
  this->pimpl->pointsOverlay = enabled;
}

void
SoVulkanRenderManager::setEdgeColor(const SbColor4f & color)
{
  this->pimpl->edgeColor = color;
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
  SoRenderBackendInitParams params;
  params.userData = context;
  if (!this->pimpl->backend.initialize(params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::initialize",
                              "backend initialization failed");
    return FALSE;
  }
  this->pimpl->backendInitialized = TRUE;

  // Ray tracing is best-effort: the raster backend is always available and
  // remains the fallback when the device lacks the RT extensions/features.
  if (this->pimpl->rtxBackend.initialize(params)) {
    this->pimpl->rtxBackendInitialized = TRUE;
  }
  else {
    SoDebugError::postWarning(
      "SoVulkanRenderManager::initialize",
      "ray-tracing backend unavailable; raster Vulkan backend will be used");
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
}

void
SoVulkanRenderManager::setRayTracing(SbBool enabled)
{
  this->pimpl->rayTracing = enabled;
  if (enabled && !this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning("SoVulkanRenderManager::setRayTracing",
                              "ray-tracing backend is not initialized; "
                              "keeping the raster backend");
    this->pimpl->rayTracing = FALSE;
  }
}

void
SoVulkanRenderManager::setMaxFramesInFlight(uint32_t count)
{
  this->pimpl->backend.setMaxFramesInFlight(count);
}

SbBool
SoVulkanRenderManager::getRayTracingActive(void) const
{
  return this->pimpl->rayTracing && this->pimpl->rtxBackendInitialized;
}

void
SoVulkanRenderManager::setPathTracingEnabled(SbBool enabled)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning("SoVulkanRenderManager::setPathTracingEnabled",
                              "ray-tracing backend is not initialized; "
                              "path tracing is unavailable");
    return;
  }
  this->pimpl->rtxBackend.setPathTracingEnabled(enabled);
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
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning("SoVulkanRenderManager::setPathTracingStart",
                              "ray-tracing backend is not initialized; "
                              "path tracing is unavailable");
    return;
  }
  this->pimpl->rtxBackend.setPathTracingStart(start);
}

SbBool
SoVulkanRenderManager::getPathTracingActive(void) const
{
  return this->pimpl->rtxBackendInitialized &&
    this->pimpl->rtxBackend.getPathTracingActive();
}

uint32_t
SoVulkanRenderManager::getPathTracingSampleCount(void) const
{
  if (!this->pimpl->rtxBackendInitialized) return 0;
  return this->pimpl->rtxBackend.getPathTracingSampleCount();
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
  if (this->getRayTracingActive()) {
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
                                      VkRenderPass renderPass)
{
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer, drawlist,
                                        params)) {
    return FALSE;
  }
  if (this->getRayTracingActive()) {
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
    return TRUE;
  }
  if (!this->pimpl->backend.renderExternal(*drawlist, params, commandBuffer,
                                           renderPass)) {
    SoDebugError::postWarning("SoVulkanRenderManager::renderExternal",
                              "backend render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  return TRUE;
}

void
SoVulkanRenderManagerP::setClippingPlanes(void)
{
  if (!this->camera || !this->scene) return;

  SoGetBoundingBoxAction bboxaction(this->viewportRegion);
  bboxaction.apply(this->scene);
  SbXfBox3f xbox = bboxaction.getXfBoundingBox();

  // Transform the world-space bounding box into camera coordinates.  The
  // managed scene graph is geometry-only (the camera is a separate member),
  // so the transform is built directly from the camera node: translate to
  // the camera origin, then rotate by the inverse orientation.  This is the
  // same math SoRenderManagerP::setClippingPlanes() applies after looking up
  // the camera-to-world matrix.
  SbMatrix mat;
  mat.setTranslate(-this->camera->position.getValue());
  xbox.transform(mat);
  mat = this->camera->orientation.getValue().inverse();
  xbox.transform(mat);
  SbBox3f box = xbox.project();

  float sizeX, sizeY, sizeZ;
  box.getSize(sizeX, sizeY, sizeZ);
  float boxDiagonal = std::sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ);

  // Clipping offset is 1% of the bounding box diagonal or at most 1.0 and at
  // least std::numeric_limits<float>::epsilon() (same as SoRenderManagerP).
  float clippingOffset = SbMin(1.0f, SbMax(std::numeric_limits<float>::epsilon(),
                                           0.01f * boxDiagonal));
  float zmin = box.getMin()[2];
  float zmax = box.getMax()[2];

  // Vector-graphics zoom wall: a CAD camera must never clip into a solid.
  // Once the nearest scene boundary comes within delta of the camera (or
  // crosses behind it), back the *effective* camera out along the view
  // direction so the nearest surface stays delta in front.  Zooming in then
  // scales features continuously -- the near plane keeps hugging the
  // surface -- until the wall is reached, where the view pins instead of
  // showing the interior of the solid.  The shift is applied to the box
  // here and to the view matrix in prepareRenderParams(); the shared camera
  // node itself is never touched (the hidden GL viewer owns it).
  //
  // delta scales with the scene: 0.001 * clippingOffset yields 0.1% of the
  // 1% diagonal offset, i.e. ~100000x magnification before the wall on a
  // typical part -- deep enough for any practical CAD inspection while the
  // near plane (delta, times the 0.1% slack below) stays strictly in front
  // of the surface.
  float shiftZ = 0.0f;
  // Only engage the wall when geometry actually spans the view direction:
  // zmin < 0 means something is in front of the camera.  With the whole
  // scene behind the camera (looking away), backing out would flip the
  // view around -- leave the planes alone instead.
  if (!box.isEmpty() && zmin < 0.0f) {
    const float delta = clippingOffset * 0.001f;
    shiftZ = zmax + delta;
    if (shiftZ < 0.0f) {
      shiftZ = 0.0f;
    }
    zmin -= shiftZ;
    zmax -= shiftZ;
  }
  this->cameraShiftZ = shiftZ;

  float nearval = -zmax - clippingOffset;
  float farval = -zmin + clippingOffset;

  if (!this->camera->isOfType(SoOrthographicCamera::getClassTypeId())
      && farval <= 0.0f) {
    return;
  }

  if (box.isEmpty()) {
    nearval = 1;
    farval = 10;
  }

  // If the whole scene is behind the camera, keep the current near/far planes
  // (they were computed on the previous frame when the scene was in front).
  // Collapsing them to a tiny range here is what makes the view appear
  // "locked": the scene only becomes visible again once it rotates within the
  // collapsed volume.  This mirrors SoRenderManagerP::setClippingPlanes(),
  // which returns without touching the planes in this case (its early return
  // only fires for perspective cameras, so an orthographic camera must handle
  // it here instead).
  if (farval <= 0.0f) {
    return;
  }

  if (getenv("FC_VULKAN_CLIP_DEBUG")) {
    static float lastNear = -1.0f, lastFar = -1.0f;
    static int emptyCount = 0;
    bool empty = box.isEmpty();
    if (empty) { emptyCount++; }
    else { emptyCount = 0; }
    if (empty || SbAbs(nearval - lastNear) > 0.05f * SbMax(SbAbs(nearval), 1.0f)
        || SbAbs(farval - lastFar) > 0.05f * SbMax(SbAbs(farval), 1.0f)) {
      lastNear = nearval;
      lastFar = farval;
      SbVec3f p = this->camera->position.getValue();
      SbRotation o = this->camera->orientation.getValue();
      float q0, q1, q2, q3;
      o.getValue(q0, q1, q2, q3);
      float x0, y0, z0, x1, y1, z1;
      box.getBounds(x0, y0, z0, x1, y1, z1);
      fprintf(stderr, "[CLIP] pos=(%.3f,%.3f,%.3f) quat=(%.3f,%.3f,%.3f,%.3f) "
                      "boxz=[%.3f,%.3f] empty=%d nearval=%.6f farval=%.6f closest=%.6f shiftZ=%.6f\n",
              p[0], p[1], p[2], q0, q1, q2, q3,
              z0, z1, empty ? 1 : 0, nearval, farval, -zmax, this->cameraShiftZ);
    }
  }

  if (this->camera->isOfType(SoPerspectiveCamera::getClassTypeId())) {
    float nearlimit;
    if (this->autoClipping == SoVulkanRenderManager::FIXED_NEAR_PLANE) {
      nearlimit = this->nearplanevalue;
    }
    else {
      int depthbits = 32;
      int use_bits = static_cast<int>(float(depthbits) * (1.0f - this->nearplanevalue));
      float r = static_cast<float>(std::pow(2.0, static_cast<double>(use_bits)));
      nearlimit = farval / r;
    }

    if (nearlimit >= farval) {
      nearlimit = farval / 5000.0f;
    }

    if (nearval < nearlimit) {
      nearval = nearlimit;
    }
  }

  // Never let the near plane fall beyond the closest geometry in front of the
  // camera.  When zoomed in close, the 1% diagonal offset and the
  // VARIABLE_NEAR_PLANE precision floor can push the near plane past nearby
  // surfaces, clipping them during close-up rotation.  Only clamp while the
  // camera is outside the bounding box (closest > 0) so the near plane always
  // stays in front of the camera.
  //
  // closest is the distance to the nearest boundary of the *shifted* box
  // (-zmax): with the zoom wall active it is delta, keeping the near plane
  // in front of the pinned surface.  Reading the unshifted box here made the
  // near plane fall behind the surface (camera inside -> closest < 0 ->
  // near plane = clippingOffset >> delta) and cut into the solid.
  const float closest = -zmax;
  if (closest > 0.0f) {
    if (nearval > closest) {
      nearval = closest;
    }
    // The camera sits just outside the scene bounds: the 1% clipping offset
    // can exceed the distance to the nearest geometry, pushing the near
    // plane behind the camera (nearval <= 0).  A negative or zero near
    // plane inverts the view volume and clips geometry that is actually in
    // front of the camera; SoCamera::viewAll() can also leave the near
    // plane at exactly 0.  Keep the near plane strictly in front of the
    // camera and behind the closest geometry.
    if (nearval <= 0.0f) {
      nearval = SbMin(closest, clippingOffset);
      if (nearval <= 0.0f) {
        nearval = std::numeric_limits<float>::epsilon();
      }
    }
  }
  else {
    // The camera is inside or behind the scene bounds, so the bbox-derived
    // nearval is negative.  A negative near plane inverts the projection and
    // clips everything (nothing renders / object "cut away"), which is what
    // FreeCAD's GL renderer avoids by keeping a small positive near plane.
    // Fall back to a small positive plane anchored on the clipping offset.
    if (nearval < clippingOffset) {
      nearval = clippingOffset;
    }
  }

  // The far plane can also land behind the camera (whole scene behind it) or
  // invert relative to near; keep the view volume well-formed.
  if (farval <= nearval) {
    farval = nearval + clippingOffset;
  }

  const float SLACK = 0.001f;
  const float newnear = nearval >= 0 ? nearval * (1.0f - SLACK)
                                     : nearval * (1.0f + SLACK);
  const float newfar = farval >= 0 ? farval * (1.0f + SLACK)
                                   : farval * (1.0f - SLACK);

  // Store the planes privately; see the computedNear/computedFar comment in
  // the pimpl declaration for why the camera fields must stay untouched.
  this->computedNear = newnear;
  this->computedFar = newfar;
}

SbBool
SoVulkanRenderManagerP::prepareRenderParams(SbBool clearwindow,
                                            SbBool clearzbuffer,
                                            SoDrawList *& drawlist,
                                            SoRenderParams & params)
{
  if (!this->backendInitialized || !this->renderTarget) {
    SoDebugError::postWarning("SoVulkanRenderManager::prepareRenderParams",
                              "backend %s, render target %s",
                              this->backendInitialized ? "initialized" : "NOT initialized",
                              this->renderTarget ? "set" : "NOT set");
    return FALSE;
  }

  // Keep the near/far planes tight around the scene so zooming and orbiting
  // never push geometry outside the view volume.  The GL SoRenderManager does
  // this automatically (VARIABLE_NEAR_PLANE); without the equivalent here,
  // the Vulkan viewport clips near faces when the camera is close and far
  // faces when the camera is far (FreeCAD's hidden GL viewer never renders,
  // so its auto-clipping never runs).
  if (this->autoClipping != SoVulkanRenderManager::NO_AUTO_CLIPPING) {
    this->setClippingPlanes();
  }

  SoIRRenderAction & action = this->irAction;
  action.setViewportRegion(this->viewportRegion);

  params.viewport = this->viewportRegion;
  params.viewMatrix.makeIdentity();
  params.projMatrix.makeIdentity();
  params.clearColor = this->backgroundColor;
  if (getenv("FC_VULKAN_BREADCRUMBS")) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      fprintf(stderr, "[VK-TRACE] prepareRenderParams backgroundGradient=%d\n", this->backgroundGradient ? 1 : 0);
    }
  }
  params.backgroundGradient = this->backgroundGradient;
  params.backgroundTopColor = this->backgroundTopColor;
  params.backgroundBottomColor = this->backgroundBottomColor;
  params.wireframeOverlay = this->wireframeOverlay;
  params.pointsOverlay = this->pointsOverlay;
  params.edgeColor = this->edgeColor;
  params.clearDepth = 1.0f;
  params.flags = 0;
  if (clearwindow || this->clearWindow) {
    params.flags |= SO_PARAM_CLEAR_WINDOW;
  }
  if (clearzbuffer || this->clearDepth) {
    params.flags |= SO_PARAM_CLEAR_DEPTH;
  }
  params.renderTarget = this->renderTarget;

  // SoIRRenderAction::apply() resets the frame, so the camera and the scene
  // must be traversed in a single apply() call.  The managed scene graph is
  // geometry-only: the camera is stored as a separate member (matching
  // SoRenderManager/SoSceneManager, which apply the camera independently of
  // the scene root).  Build a path [camera, scene, overlayScene] so
  // SoCamera::doAction() installs the projection/viewing matrix elements
  // before any geometry is recorded; otherwise every command carries identity
  // matrices and the view renders at the origin with an identity projection
  // (blank/wrong view, invisible geometry, and a camera that appears not to
  // follow navigation).  The overlay scene is traversed last so its commands
  // record after the main scene and render in the overlay pass.
  if (this->scene || this->camera || this->overlayScene) {
    SoSeparator * root = new SoSeparator;
    root->ref();
    if (this->camera) {
      root->addChild(this->camera);
    }
    if (this->scene) {
      root->addChild(this->scene);
    }
    if (this->overlayScene) {
      root->addChild(this->overlayScene);
    }
    action.apply(root);
    root->unref();
  }
  else {
    action.beginFrame();
  }

  SoDrawList & list = action.getMutableDrawList();

  // The frame view/projection matrices drive every non-overlay command, so
  // they must come from the camera this manager was told to use
  // (setCamera()), not from whatever camera node happens to sit inside the
  // traversed scene graph.  FreeCAD swaps the camera node in its scene
  // graph when the projection type changes; if that swap is not mirrored
  // into the manager, harvesting the matrices from the first recorded
  // command renders with the scene's (new) camera while auto-clipping and
  // the viewport use the manager's (stale) camera -- or vice versa.  The
  // result is a viewport whose near/far planes do not belong to the camera
  // actually rendering: the swapped-in camera keeps its default planes
  // (near=1, far=10) and culls anything beyond 10 units as soon as the
  // camera moves away from the object.
  //
  // Build the matrices directly from the camera node, mirroring
  // SoCamera::doAction().  The managed camera is always traversed at the
  // top of a fresh separator (no model transforms above it), so the
  // view-volume/projection computed here matches what the traversal would
  // install when the scene graph contains the same camera node.
  if (this->camera) {
    // Build the view volume with the near/far planes this manager computed
    // (setClippingPlanes()), NOT with SoCamera::nearDistance/farDistance.
    // The camera fields are shared with FreeCAD's hidden GL viewer, whose
    // own render manager rewrites them concurrently; harvesting the volume
    // from the fields races with that writer and intermittently projects
    // with a near plane behind the front surface (visible clipping / seeing
    // into the object while navigating).  When auto-clipping is off the
    // fields are authoritative and are used as-is.
    float nearplane = this->camera->nearDistance.getValue();
    float farplane = this->camera->farDistance.getValue();
    if (this->autoClipping != SoVulkanRenderManager::NO_AUTO_CLIPPING) {
      nearplane = this->computedNear;
      farplane = this->computedFar;
    }

    SbViewVolume vv;
    if (this->camera->isOfType(SoPerspectiveCamera::getClassTypeId())) {
      const auto * pc =
        static_cast<const SoPerspectiveCamera *>(this->camera);
      vv.perspective(pc->heightAngle.getValue(),
                     this->viewportRegion.getViewportAspectRatio(),
                     nearplane, farplane);
    }
    else if (this->camera->isOfType(SoOrthographicCamera::getClassTypeId())) {
      const auto * oc = static_cast<const SoOrthographicCamera *>(this->camera);
      const float halfheight = oc->height.getValue() * 0.5f;
      const float halfwidth =
        halfheight * this->viewportRegion.getViewportAspectRatio();
      vv.ortho(-halfwidth, halfwidth, -halfheight, halfheight,
               nearplane, farplane);
    }
    else {
      vv = this->camera->getViewVolume(
        this->viewportRegion.getViewportAspectRatio());
    }

    if (vv.getDepth() == 0.0f || vv.getWidth() == 0.0f
        || vv.getHeight() == 0.0f) {
      // Empty scenes: SoCamera::doAction() installs identity matrices.
      params.viewMatrix.makeIdentity();
      params.projMatrix.makeIdentity();
    }
    else {
      vv.rotateCamera(this->camera->orientation.getValue());
      if (this->cameraShiftZ > 0.0f
          && this->autoClipping != SoVulkanRenderManager::NO_AUTO_CLIPPING) {
        // Zoom wall: render from the backed-off camera position (see
        // setClippingPlanes()); the camera node itself is left alone.
        SbVec3f forward;
        this->camera->orientation.getValue().multVec(
          SbVec3f(0.0f, 0.0f, -1.0f), forward);
        vv.translateCamera(this->camera->position.getValue()
                           - forward * this->cameraShiftZ);
      }
      else {
        vv.translateCamera(this->camera->position.getValue());
      }
      vv.getMatrices(params.viewMatrix, params.projMatrix);
    }
  }
  else if (list.getNumCommands() > 0) {
    // No managed camera: fall back to the first recorded command.
    const SoRenderCommand & first = list.getCommand(0);
    params.viewMatrix = first.viewMatrix;
    params.projMatrix = first.projMatrix;
  }

  // Diagnose a manager-camera vs scene-camera mismatch: if the scene graph
  // itself contains a camera node, its doAction() state overrides the
  // manager camera for any geometry recorded after it, so the rendered
  // view/projection (and face culling) come from a different camera than the
  // one the viewport is using.
  if (getenv("FC_VULKAN_CLIP_DEBUG")) {
    static bool sceneCamLogged = false;
    static int sceneDumpCount = 0;
    if ((!sceneCamLogged || sceneDumpCount < 3) && this->scene) {
      sceneCamLogged = true;
      sceneDumpCount += 1;
      if (this->scene->getTypeId().isDerivedFrom(SoSeparator::getClassTypeId())) {
        SoSeparator * sep = static_cast<SoSeparator*>(this->scene);
        for (int i = 0; i < sep->getNumChildren(); ++i) {
          SoNode * child = sep->getChild(i);
          const char * extra = "";
          if (child->isOfType(SoCamera::getClassTypeId())) {
            static char buf[160];
            std::snprintf(buf, sizeof(buf),
                          " ptr=%p (manager-camera=%p %s)",
                          (void*)child, (void*)this->camera,
                          this->camera
                            ? this->camera->getTypeId().getName().getString()
                            : "null");
            extra = buf;
          }
          fprintf(stderr, "[CLIP] SCENECHILD[%d] %s%s\n", i,
                  child->getTypeId().getName().getString(), extra);
        }
      }
      {
        // Recursive type-only dump of the scene (up to 5 levels deep) to spot
        // any stray camera or matrix nodes.
        SoSeparator * sep = static_cast<SoSeparator*>(this->scene);
        std::function<void(SoNode*, int, int*)> dumpLevel =
            [&dumpLevel](SoNode * n, int depth, int * counter) {
          if (!n) return;
          bool isGroup = n->getTypeId().isDerivedFrom(SoGroup::getClassTypeId());
          SoGroup * g = isGroup ? static_cast<SoGroup*>(n) : nullptr;
          if (g) {
            fprintf(stderr, "[CLIP] TREE%*s%s (%d children)\n", depth * 2, "",
                    n->getTypeId().getName().getString(), g->getNumChildren());
            if (depth < 7) {
              for (int i = 0; i < g->getNumChildren(); ++i) {
                *counter += 1;
                if (*counter > 120) break;
                SoNode * c = g->getChild(i);
                if (c->getTypeId().isDerivedFrom(SoGroup::getClassTypeId())) {
                  dumpLevel(c, depth + 1, counter);
                }
                else {
                  const char * extra = "";
                  if (c->getTypeId().isDerivedFrom(SoScale::getClassTypeId())) {
                    static char sbuf[128];
                    const SoScale * s = static_cast<const SoScale*>(c);
                    SbVec3f sf = s->scaleFactor.getValue();
                    std::snprintf(sbuf, sizeof(sbuf), " scaleFactor=(%.3f,%.3f,%.3f)",
                                  sf[0], sf[1], sf[2]);
                    extra = sbuf;
                  }
                  fprintf(stderr, "[CLIP] TREE%*s%s%s\n", (depth + 1) * 2, "",
                          c->getTypeId().getName().getString(), extra);
                }
              }
            }
          }
          else {
            fprintf(stderr, "[CLIP] TREE%*s%s\n", depth * 2, "",
                    n->getTypeId().getName().getString());
          }
        };
        int counter = 0;
        dumpLevel(this->scene, 0, &counter);
      }
      SoSearchAction search;
      search.setType(SoCamera::getClassTypeId());
      search.setSearchingAll(TRUE);
      search.apply(this->scene);
      const SoPathList & paths = search.getPaths();
      for (int i = 0; i < paths.getLength(); ++i) {
        SoCamera * c = static_cast<SoCamera*>(paths[i]->getTail());
        fprintf(stderr, "[CLIP] SCENE-CAMERA #%d ptr=%p %s pos=(%.2f,%.2f,%.2f) "
                        "near=%.4f far=%.4f\n",
                i, (void*)c, c->getTypeId().getName().getString(),
                c->position.getValue()[0], c->position.getValue()[1],
                c->position.getValue()[2],
                c->nearDistance.getValue(), c->farDistance.getValue());
      }
      if (paths.getLength() > 0) {
        fprintf(stderr, "[CLIP] SCENE-CAMERAS=%d manager-camera=%p %s\n",
                paths.getLength(), (void*)this->camera,
                this->camera ? this->camera->getTypeId().getName().getString() : "null");
      }
      SoType brepType = SoType::fromName("SoBrepFaceSet");
      if (brepType != SoType::badType()) {
        SoSearchAction bs;
        bs.setType(brepType);
        bs.setSearchingAll(TRUE);
        bs.apply(this->scene);
        fprintf(stderr, "[CLIP] SCENE-BREPFACESETS=%d\n", bs.getPaths().getLength());
      }
      SoType cubeType = SoType::fromName("Cube");
      if (cubeType != SoType::badType()) {
        SoSearchAction cs;
        cs.setType(cubeType);
        cs.setSearchingAll(TRUE);
        cs.apply(this->scene);
        fprintf(stderr, "[CLIP] SCENE-CUBES=%d\n", cs.getPaths().getLength());
      }
    }
  }

  list.buildSortedOrder(params.viewMatrix);
  drawlist = &list;

  // Dump the draw list when COIN_DEBUG_RENDER_IR is set so the overlay
  // commands recorded by the highlight/selection paths can be inspected
  // (pass, depth state, diffuse color, vertex count).
  static int dumpCount = 0;
  if (coin_render_ir_trace_enabled() && dumpCount++ < 8) {
    SoIRDumpSummary(list);
    SoIRDumpFirstN(list, list.getNumCommands());
  }

  // Diagnostic trace for the Vulkan viewport pipeline.  The view/projection
  // matrices are harvested from the first recorded command (see above).
  // Identity values mean the scene graph did not contribute a camera node (or
  // no geometry was recorded at all), which renders as a blank view.  Log the
  // transition to non-identity matrices (the first real camera frame) rather
  // than the initial empty frame so the camera fix can be verified at runtime.
  static bool loggedReady = false;
  if (!loggedReady && list.getNumCommands() > 0
      && (params.viewMatrix[3][3] != 1.0f || params.projMatrix[3][3] != 1.0f
          || params.projMatrix[2][3] != 0.0f)) {
    loggedReady = true;
    SbVec2s vpsize = this->viewportRegion.getViewportSizePixels();
    SoDebugError::postInfo(
      "SoVulkanRenderManager::prepareRenderParams",
      "scene=%p camera=%p viewport=%dx%d commands=%d clearColor=(%.3f,%.3f,%.3f,%.3f) "
      "clearWindow=%d clearDepth=%d",
      this->scene, this->camera, vpsize[0], vpsize[1],
      list.getNumCommands(), this->backgroundColor[0], this->backgroundColor[1],
      this->backgroundColor[2], this->backgroundColor[3],
      this->clearWindow ? 1 : 0, this->clearDepth ? 1 : 0);
    SoDebugError::postInfo(
      "SoVulkanRenderManager::prepareRenderParams",
      "first-command view[3][3]=%.6f proj[3][3]=%.6f proj[2][3]=%.6f",
      params.viewMatrix[3][3], params.projMatrix[3][3], params.projMatrix[2][3]);
  }

  // Reconstruct near/far from the recorded projection matrix and compare with
  // the auto-clipped values so mismatches (per-object clipping) are obvious.
  if (getenv("FC_VULKAN_CLIP_DEBUG")) {
    static int frames = 0;
    if (++frames == 10 || frames == 50 || frames % 25 == 0) {
      SbMatrix m = params.projMatrix;
      SbMatrix v = params.viewMatrix;
      // OpenGL-style perspective: col2=(0,0,a,-1), col3=(0,0,b,0) with
      // a=-(f+n)/(f-n), b=-2fn/(f-n)  ->  n=b/(a-1), f=b/(a+1).
      // Depth-range form (ortho): m22=-2/(f-n), m32=-(f+n)/(f-n)
      // ->  n=(m32+1)/m22, f=(m32-1)/m22.
      float nearf = -1.0f, farf = -1.0f;
      if (m[2][3] == -1.0f && m[3][3] == 0.0f) {
        const float a = m[2][2];
        const float b = m[3][2];
        nearf = b / (a - 1.0f);
        farf = b / (a + 1.0f);
      }
      else {
        const float m22 = m[2][2];
        const float m32 = m[3][2];
        if (m22 != 0.0f) {
          nearf = (m32 + 1.0f) / m22;
          farf = (m32 - 1.0f) / m22;
        }
      }
      fprintf(stderr,
              "[CLIP] cmd0 cam-near=%.4f cam-far=%.4f use-near=%.4f use-far=%.4f "
              "focal=%.4f pos=(%.2f,%.2f,%.2f) "
              "ncd=%.4f fcd=%.4f cmds=%d m00=%.3f m11=%.3f m22=%.4f m32=%.4f m23=%.4f\n",
              this->camera ? this->camera->nearDistance.getValue() : -1.0f,
              this->camera ? this->camera->farDistance.getValue() : -1.0f,
              this->computedNear, this->computedFar,
              this->camera ? this->camera->focalDistance.getValue() : -1.0f,
              this->camera ? this->camera->position.getValue()[0] : 0.0f,
              this->camera ? this->camera->position.getValue()[1] : 0.0f,
              this->camera ? this->camera->position.getValue()[2] : 0.0f,
              nearf, farf,
              list.getNumCommands(), m[0][0], m[1][1], m[2][2], m[3][2], m[2][3]);
      if (list.getNumCommands() > 0) {
        const int show = std::min(4, static_cast<int>(list.getNumCommands()));
        for (int ci = 0; ci < show; ++ci) {
          const SoRenderCommand & c0 = list.getCommand(ci);
          SbMatrix cm;
          c0.modelMatrix.getValue(cm);
          fprintf(stderr,
                  "[CLIP] cmd%d pass=%d verts=%u model00=%.3f trans=(%.3f,%.3f,%.3f) "
                  "m11=%.3f m22=%.3f\n",
                  ci, static_cast<int>(c0.pass),
                  c0.geometry.vertexCount,
                  cm[0][0], cm[3][0], cm[3][1], cm[3][2],
                  cm[1][1], cm[2][2]);
          if (ci == 0 && c0.geometry.positions && c0.geometry.vertexCount >= 3) {
            const float * p = c0.geometry.positions;
            float mnx = 1e30f, mny = 1e30f, mnz = 1e30f, mxx = -1e30f, myy = -1e30f, mzz = -1e30f;
            const unsigned nv = c0.geometry.vertexCount;
            for (unsigned v = 0; v < nv; ++v) {
              mnx = std::min(mnx, p[v*3+0]); mny = std::min(mny, p[v*3+1]); mnz = std::min(mnz, p[v*3+2]);
              mxx = std::max(mxx, p[v*3+0]); myy = std::max(myy, p[v*3+1]); mzz = std::max(mzz, p[v*3+2]);
            }
            fprintf(stderr, "[CLIP] cmd0 verts0=(%.2f,%.2f,%.2f) bbox=[%.2f,%.2f]x[%.2f,%.2f]x[%.2f,%.2f]\n",
                    p[0], p[1], p[2], mnx, mxx, mny, myy, mnz, mzz);
          }
          if (ci == 2 && c0.geometry.positions && c0.geometry.vertexCount >= 3) {
            const float * p = c0.geometry.positions;
            float mnx = 1e30f, mny = 1e30f, mnz = 1e30f, mxx = -1e30f, myy = -1e30f, mzz = -1e30f;
            const unsigned nv = c0.geometry.vertexCount;
            for (unsigned v = 0; v < nv; ++v) {
              mnx = std::min(mnx, p[v*3+0]); mny = std::min(mny, p[v*3+1]); mnz = std::min(mnz, p[v*3+2]);
              mxx = std::max(mxx, p[v*3+0]); myy = std::max(myy, p[v*3+1]); mzz = std::max(mzz, p[v*3+2]);
            }
            fprintf(stderr, "[CLIP] cmd2 verts0=(%.2f,%.2f,%.2f) bbox=[%.2f,%.2f]x[%.2f,%.2f]x[%.2f,%.2f]\n",
                    p[0], p[1], p[2], mnx, mxx, mny, myy, mnz, mzz);
          }
        }
      }
      // Compare the box-center position in camera space derived from the
      // camera NODE's own fields vs the harvested params.viewMatrix.  If they
      // disagree, the matrix the GPU uses is not built from this camera node.
      if (this->camera && this->scene) {
        SoGetBoundingBoxAction bba(this->viewportRegion);
        bba.apply(this->scene);
        SbBox3f wbox = bba.getBoundingBox();
        if (!wbox.isEmpty()) {
          SbVec3f center = wbox.getCenter();
          SbVec3f camBased, mtxBased;
          SbMatrix camMat, rotMat;
          camMat.setTranslate(-this->camera->position.getValue());
          rotMat = this->camera->orientation.getValue().inverse();
          camMat.multRight(rotMat);
          camMat.multVecMatrix(center, camBased);
          params.viewMatrix.multVecMatrix(center, mtxBased);
          float q0, q1, q2, q3;
          this->camera->orientation.getValue().getValue(q0, q1, q2, q3);
          fprintf(stderr,
                  "[CLIP] centerCam cam=(%.2f,%.2f,%.2f) mtx=(%.2f,%.2f,%.2f) "
                  "quat=(%.3f,%.3f,%.3f,%.3f) dist=%.2f\n",
                  camBased[0], camBased[1], camBased[2],
                  mtxBased[0], mtxBased[1], mtxBased[2],
                  q0, q1, q2, q3,
                  (this->camera->position.getValue() - center).length());
        }
      }
    }
    static int typeLogged = 0;
    if (typeLogged++ < 3 && this->camera && list.getNumCommands() > 0) {
      fprintf(stderr, "[CLIP] camera-type=%s pos=(%.3f,%.3f,%.3f) ortho=%d persp=%d camptr=%p\n",
              this->camera->getTypeId().getName().getString(),
              this->camera->position.getValue()[0],
              this->camera->position.getValue()[1],
              this->camera->position.getValue()[2],
              this->camera->isOfType(SoOrthographicCamera::getClassTypeId()) ? 1 : 0,
              this->camera->isOfType(SoPerspectiveCamera::getClassTypeId()) ? 1 : 0,
              (void*)this->camera);
    }

    // Cross-check the near/far source: transform the scene bounding box by
    // the ACTUAL view matrix (what the GPU uses) and print the z-range, so a
    // mismatch with the [CLIP] boxz (from setClippingPlanes' own transform)
    // is obvious.  This isolates whether the near plane is cutting geometry
    // because setClippingPlanes computes a wrong camera-space box.
    if (frames % 250 == 0 && this->scene) {
      SoGetBoundingBoxAction bboxAction(this->viewportRegion);
      bboxAction.apply(this->scene);
      SbBox3f wbox = bboxAction.getBoundingBox();
      if (!wbox.isEmpty()) {
        float zmin = 1e30f, zmax = -1e30f;
        const SbVec3f & mn = wbox.getMin();
        const SbVec3f & mx = wbox.getMax();
        for (int ix = 0; ix < 2; ++ix) {
          for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
              SbVec3f c(ix ? mx[0] : mn[0],
                        iy ? mx[1] : mn[1],
                        iz ? mx[2] : mn[2]);
              SbVec3f v;
              params.viewMatrix.multVecMatrix(c, v);
              zmin = std::min(zmin, v[2]);
              zmax = std::max(zmax, v[2]);
            }
          }
        }
        SbVec3f center = wbox.getCenter();
        fprintf(stderr,
                "[CLIP] viewbox worldCenter=(%.2f,%.2f,%.2f) "
                "worldSize=(%.2f,%.2f,%.2f) viewZ=[%.3f,%.3f]\n",
                center[0], center[1], center[2],
                wbox.getSize()[0], wbox.getSize()[1], wbox.getSize()[2],
                zmin, zmax);
      }
    }

    // Project the first few commands' vertices into NDC the same way the
    // backend vertex shader does (gl_Position = proj * view * model * pos,
    // column-vector math on column-major matrices) to see whether the model
    // geometry actually lands inside the clip volume at this view.
    if (frames % 250 == 0 && list.getNumCommands() > 0) {
      auto mv = [](const SbMatrix & M, float x, float y, float z,
                   float * ox, float * oy, float * oz, float * ow) {
        *ox = M[0][0] * x + M[1][0] * y + M[2][0] * z + M[3][0];
        *oy = M[0][1] * x + M[1][1] * y + M[2][1] * z + M[3][1];
        *oz = M[0][2] * x + M[1][2] * y + M[2][2] * z + M[3][2];
        *ow = M[0][3] * x + M[1][3] * y + M[2][3] * z + M[3][3];
      };
      const int show = std::min(4, static_cast<int>(list.getNumCommands()));
      for (int i = 0; i < show; ++i) {
        const SoRenderCommand & cmd = list.getCommand(i);
        const SoGeometryDesc & geo = cmd.geometry;
        if (!geo.positions || geo.vertexCount == 0) continue;
        float x = geo.positions[0], y = geo.positions[1], z = geo.positions[2];
        float wx, wy, wz, ww;
        mv(cmd.modelMatrix, x, y, z, &wx, &wy, &wz, &ww);
        float vx, vy, vz, vw;
        mv(params.viewMatrix, wx, wy, wz, &vx, &vy, &vz, &vw);
        float nx, ny, nz, nw;
        mv(params.projMatrix, vx, vy, vz, &nx, &ny, &nz, &nw);
        fprintf(stderr,
                "[CLIP] cmd%d pass=%d verts=%d cull=%d "
                "world=(%.3f,%.3f,%.3f) viewz=%.3f ndc=(%.3f,%.3f,%.3f,%.3f)\n",
                i, static_cast<int>(cmd.pass), static_cast<int>(geo.vertexCount),
                static_cast<int>(cmd.state.raster.cullMode),
                wx, wy, wz, vz, nx, ny, nz, nw);
      }
    }
  }

  return TRUE;
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
