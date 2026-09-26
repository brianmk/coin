// src/rendering/vulkan/manager/SoVulkanRenderManagerRender.cpp

// Frame-preparation / clipping / record half of SoVulkanRenderManager: the
// SoVulkanRenderManagerP:: methods split out of the monolithic manager TU.
// Shares the P-impl struct and helper namespace via SoVulkanRenderManagerP.h.

#include "rendering/vulkan/manager/SoVulkanRenderManagerP.h"

using namespace SoVulkanManagerDetail;

uint64_t
SoVulkanRenderManagerP::computeGraphFingerprint() const
{
  uint64_t h = 0xcbf29ce484222325ULL;
  mixHash(h, reinterpret_cast<uintptr_t>(this->camera));
  mixHash(h, reinterpret_cast<uintptr_t>(this->scene));
  mixHash(h, reinterpret_cast<uintptr_t>(this->overlayScene));
  mixHash(h, reinterpret_cast<uintptr_t>(this->decorationScene));
  if (SoVulkanConfig::get().debug.lightReplayDebug) {
    uint64_t hScene = 0xcbf29ce484222325ULL;
    uint64_t hOverlay = 0xcbf29ce484222325ULL;
    uint64_t hDecor = 0xcbf29ce484222325ULL;
    graphFingerprintWalk(this->scene, this->camera, hScene);
    graphFingerprintWalk(this->overlayScene, this->camera, hOverlay);
    graphFingerprintWalk(this->decorationScene, this->camera, hDecor);
    if (vkLightFpDbgBudget-- > 0) {
      fprintf(stderr,
              "[FP] scene=%016lx overlay=%016lx decor=%016lx extRev=%llu"
              " size=%dx%d\n",
              (unsigned long)hScene, (unsigned long)hOverlay,
              (unsigned long)hDecor,
              (unsigned long long)this->externalRevision,
              (int)this->viewportRegion.getViewportSizePixels()[0],
              (int)this->viewportRegion.getViewportSizePixels()[1]);
    }
  }
  // The replay gate is keyed on the MAIN scene only (plus the viewport and the
  // caller-published revision).  The overlay/decoration scene node-ids churn
  // every frame -- the navigation cube and the axis cross mirror the camera,
  // so their nodes are re-touched each frame even when the geometry they draw
  // is unchanged.  Folding those ids into the same hash made the fingerprint
  // change every frame and forced a full re-traversal of an otherwise-stable
  // main scene (the held hotspot).  The overlay/decoration are now re-recorded
  // separately every frame (cheap) and the main scene is replayed when THIS
  // fingerprint is stable; their pointer mixes below stay constant so an
  // overlay-scene swap still invalidates.
  graphFingerprintWalk(this->scene, this->camera, h);
  const SbVec2s size = this->viewportRegion.getViewportSizePixels();
  mixHash(h, static_cast<uint32_t>(size[0]));
  mixHash(h, static_cast<uint32_t>(size[1]));
  uint32_t dprBits = 0;
  std::memcpy(&dprBits, &this->devicePixelRatio, sizeof(dprBits));
  mixHash(h, dprBits);
  mixHash(h, static_cast<uint32_t>(this->autoClipping));
  mixHash(h, this->externalRevision);
  return h;
}

SoCamera *
SoVulkanRenderManagerP::resolveActiveCamera()
{
  // The scene graph passed to setSceneGraph() is the GL viewer's superscene,
  // which CONTAINS the camera node that navigation actually mutates (FreeCAD
  // keeps the camera inside the scene root separator).  Prefer that node: it
  // is the single authority and cannot go stale, whereas the retained pointer
  // set by setCamera() is a snapshot that diverges as soon as the camera is
  // rotated/panned without an intervening sync.
  if (this->scene) {
    if (this->scene->getTypeId().isDerivedFrom(SoSeparator::getClassTypeId())) {
      SoSeparator * sep = static_cast<SoSeparator *>(this->scene);
      for (int i = 0; i < sep->getNumChildren(); ++i) {
        SoNode * child = sep->getChild(i);
        if (child && child->isOfType(SoCamera::getClassTypeId())) {
          return static_cast<SoCamera *>(child);
        }
      }
    }
    // The camera may be nested deeper (a subset/child separator).  Search the
    // subtree for the first camera, mirroring SoCamera::doAction() semantics of
    // using the camera encountered first in traversal order.
    SoSearchAction search;
    search.setType(SoCamera::getClassTypeId());
    search.setSearchingAll(TRUE);
    search.apply(this->scene);
    const SoPathList & paths = search.getPaths();
    if (paths.getLength() > 0) {
      return static_cast<SoCamera *>(paths[0]->getTail());
    }
  }
  // No camera in the scene graph: fall back to the retained pointer (used by
  // overlay-only or programmatic render setups that manage a camera outside
  // the scene).
  return this->camera;
}

// Refresh the retained camera pointer from the authoritative scene-graph
// camera and bump the generation counter when the active camera (or its
// pose) changes.  Called at the top of every prepareRenderParams() so all
// downstream reads of `this->camera` see the node that is actually in the
// scene -- the node navigation mutates -- instead of a possibly-stale
// snapshot.  The refcount is managed so the node stays alive for the frame.
void
SoVulkanRenderManagerP::refreshActiveCamera()
{
  SoCamera * resolved = this->resolveActiveCamera();
  if (resolved && resolved != this->camera) {
    setRetainedNode(this->camera, resolved);
    this->cameraVersion++;
  }
  else if (resolved == this->camera) {
    // Same node: detect pose changes (a rotation/pan/zoom mutates the node in
    // place), which a pointer comparison alone cannot see.  Compare the pose
    // fingerprint so the backend's viewChanged reliably fires on a camera move.
    SbVec3f pos = this->camera->position.getValue();
    SbRotation ori = this->camera->orientation.getValue();
    SbVec3f fp;
    ori.multVec(SbVec3f(0.0f, 0.0f, 1.0f), fp);
    uint32_t fp1 = (uint32_t)((int)(pos[0] * 256.0f)) ^
                   (uint32_t)((int)(pos[1] * 256.0f)) ^
                   (uint32_t)((int)(pos[2] * 256.0f)) ^
                   (uint32_t)((int)(fp[0] * 256.0f));
    if (fp1 != this->cameraPoseFingerprint) {
      this->cameraPoseFingerprint = fp1;
      this->cameraVersion++;
    }
  }
}

void
SoVulkanRenderManagerP::setClippingPlanes(void)
{
  SoCamera * camera = this->resolveActiveCamera();
  if (!camera || !this->scene) return;

  // Recompute the world-space bounding box only when the scene pointer changed
  // or the previous frame's main-command fingerprint differs.  The fingerprint
  // covers the world transform AND geometry identity of every main command, so
  // a moved/rotated object (same command count) still invalidates the cache.
  // Only the main commands are hashed: overlay/decoration commands (appended
  // after mainCommandCount) are re-recorded every frame with camera-dependent
  // model matrices, so including them would invalidate the cache on pure
  // camera moves.  The camera pose is applied below every frame; the
  // whole-scene bbox traversal is the expensive part and is now skipped on
  // unchanged scenes.
  const int mainCount = static_cast<int>(this->mainCommandCount);
  // Reuse the cached fingerprint when the retained main list provably has
  // not changed: the scene pointer and the retained main command count are
  // the same, and the scene sensor has not fired since the last full walk
  // (any main-scene change notifies the scene root, which raises
  // sceneGraphDirty).  On those camera-only frames the O(main-commands)
  // model-matrix/geometry hash would just reproduce last frame's value.
  uint64_t sceneFp;
  if (this->sceneFpValid && !this->sceneGraphDirty &&
      this->scene == this->sceneFpScene &&
      this->mainCommandCount == this->sceneFpMainCount) {
    sceneFp = this->sceneFpCached;
  }
  else {
    sceneFp = computeSceneFingerprint(this->irAction, mainCount);
    this->sceneFpCached = sceneFp;
    this->sceneFpScene = this->scene;
    this->sceneFpMainCount = this->mainCommandCount;
    this->sceneFpValid = TRUE;
  }
  if (!this->sceneBBoxCached) {
    SoGetBoundingBoxAction bboxaction(this->viewportRegion);
    bboxaction.apply(this->scene);
    this->sceneWorldBBox = bboxaction.getXfBoundingBox();
    this->sceneBBoxScene = this->scene;
    this->sceneBBoxFingerprint = sceneFp;
    this->sceneBBoxCached = true;
  } else if (this->sceneBBoxScene != this->scene ||
             this->sceneBBoxFingerprint != sceneFp) {
    SoGetBoundingBoxAction bboxaction(this->viewportRegion);
    bboxaction.apply(this->scene);
    this->sceneWorldBBox = bboxaction.getXfBoundingBox();
    this->sceneBBoxScene = this->scene;
    this->sceneBBoxFingerprint = sceneFp;
  }
  SbXfBox3f xbox = this->sceneWorldBBox;

  // The camera-coupled ground grid lives in the per-frame decoration scene, not
  // the main scene, so extend the clip box with the decoration bounds.  Without
  // this the far plane stays scene-sized and the rasterizer clips the ground to
  // a thin band.  SoGroundPlane sizes its footprint independently of the far
  // plane, so including it here cannot feed back into the next frame.
  if (this->decorationScene) {
    SoGetBoundingBoxAction decorationbboxaction(this->viewportRegion);
    decorationbboxaction.apply(this->decorationScene);
    xbox.extendBy(decorationbboxaction.getXfBoundingBox());
  }

  // Transform the world-space bounding box into camera coordinates.  The
  // managed scene graph is geometry-only (the camera is a separate member),
  // so the transform is built directly from the camera node: translate to
  // the camera origin, then rotate by the inverse orientation.  This is the
  // same math SoRenderManagerP::setClippingPlanes() applies after looking up
  // the camera-to-world matrix.
  SbMatrix mat;
  mat.setTranslate(-camera->position.getValue());
  xbox.transform(mat);
  mat = camera->orientation.getValue().inverse();
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

  // The former vector-graphics "zoom wall" (and the near-plane clamp that
  // supported it) is intentionally gone.  It backed the effective camera out
  // and pinned the nearest bounding-box boundary at delta = 0.001 *
  // clippingOffset whenever the camera was inside the scene bounding box --
  // i.e. for any close view of a large model.  That forced computedNear down
  // to ~delta while far stayed scene-sized, destroying a 24-bit depth buffer's
  // precision across the model: opaque faces z-fight and resolve arbitrarily,
  // so the interior shows through (Coin, whose auto-clip keeps near ~ far/2^12,
  // renders the same view correctly).  Keep GL parity instead:
  // coinComputeClippingPlanes() below applies the VARIABLE_NEAR_PLANE precision
  // floor (farval / 2^12 for the default nearplanevalue), the same near plane
  // the legacy GL SoRenderManagerP uses.
  float shiftZ = 0.0f;
  this->cameraShiftZ = shiftZ;

  // Rebuild the box from the shifted z extent so the shared clipping core
  // below reads the zoom-wall-adjusted depth.
  SbBox3f clippedBox = box;
  if (shiftZ != 0.0f) {
    float x0, y0, z0, x1, y1, z1;
    box.getBounds(x0, y0, z0, x1, y1, z1);
    clippedBox.setBounds(x0, y0, zmin, x1, y1, zmax);
  }

  // Shared near/far computation (diagonal offset, empty-box defaults,
  // perspective near limit) with the legacy GL SoRenderManager.
  const bool isOrtho = camera->isOfType(
    SoOrthographicCamera::getClassTypeId());
  const bool isPersp = camera->isOfType(
    SoPerspectiveCamera::getClassTypeId());
  float nearval, farval;
  if (!coinComputeClippingPlanes(clippedBox, isOrtho, isPersp,
                                 static_cast<int>(this->autoClipping),
                                 this->nearplanevalue, nearval, farval)) {
    return;
  }

  // Do NOT bail out when farval <= 0 here.  For an orthographic camera a
  // negative near/far pair is meaningful: the ortho view volume is symmetric
  // and may extend behind the projection point, so a scene that is wholly
  // behind the camera still renders (the legacy GL manager writes exactly
  // these signed values in SoRenderManagerP::setClippingPlanes).  Returning
  // early instead keeps whatever planes were last stored -- on a freshly
  // opened document that is the pimpl default near=1/far=10 -- so the Vulkan
  // viewport culls the whole scene and goes blank while the Coin renderer,
  // which has no such guard, keeps drawing it.  Perspective cameras are
  // already rejected inside coinComputeClippingPlanes().

  if (clipDebugEnabled()) {
    static float lastNear = -1.0f, lastFar = -1.0f;
    const bool empty = box.isEmpty();
    if (empty || SbAbs(nearval - lastNear) > 0.05f * SbMax(SbAbs(nearval), 1.0f)
        || SbAbs(farval - lastFar) > 0.05f * SbMax(SbAbs(farval), 1.0f)) {
      lastNear = nearval;
      lastFar = farval;
      SbVec3f p = camera->position.getValue();
      SbRotation o = camera->orientation.getValue();
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

  // Do NOT lower the near plane to the nearest bounding-box boundary here.
  // The previous clamp did that (to support the zoom wall above) and, for a
  // large scene whose bounding box the camera sits inside, drove computedNear
  // down to ~0.001 while far stayed ~45000.  That ~4e7:1 depth ratio leaves a
  // 24-bit depth buffer with no usable precision across the model, so opaque
  // faces z-fight and render see-through.  coinComputeClippingPlanes() has
  // already raised the perspective near plane to the VARIABLE_NEAR_PLANE
  // precision floor (farval / 2^12 for the default nearplanevalue), which is
  // the same near plane the legacy GL SoRenderManagerP uses and which renders
  // these close-up views correctly.  For an orthographic camera the shared
  // core keeps the meaningful signed planes untouched.

  // The far plane can also land behind the camera (whole scene behind it) or
  // invert relative to near; keep the view volume well-formed.
  if (farval <= nearval) {
    farval = nearval + clippingOffset;
  }

  const float SLACK = kSoClippingSlack;
  const float newnear = nearval >= 0 ? nearval * (1.0f - SLACK)
                                     : nearval * (1.0f + SLACK);
  const float newfar = farval >= 0 ? farval * (1.0f + SLACK)
                                   : farval * (1.0f - SLACK);

  // Store the planes privately (prepareRenderParams reads these back) and
  // publish them onto the camera field, exactly as the legacy GL
  // SoRenderManagerP::setClippingPlanes does (SoRenderManagerP.cpp:170,173).
  // The camera node is the single authoritative view volume for every non-GL
  // consumer; in particular SoRayPickAction (the CPU hover/select path in the
  // raster Vulkan viewport) reads camera->nearDistance/farDistance, so without
  // this write-back the pick uses the stale manager default far (10) and can
  // never reach a scene-sized model such as BIMExample.
  this->computedNear = newnear;
  this->computedFar = newfar;
  // Publish only on an actual change.  SoSFFloat::setValue() notifies the
  // field's auditors unconditionally, and the active camera carries a node
  // sensor on the FreeCAD side: an unconditional write here fires the sensor
  // from inside the render, which requests another frame, which calls
  // setClippingPlanes() again -- a redraw feedback loop that keeps the
  // viewport rendering at full rate even while completely idle.
  if (camera->nearDistance.getValue() != newnear) {
    camera->nearDistance = newnear;
  }
  if (camera->farDistance.getValue() != newfar) {
    camera->farDistance = newfar;
  }
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

  const long prepBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  const bool wantCpuTiming = frameTimingEnabled();
  double cpuClipMs = 0.0, cpuApplyMs = 0.0, cpuReplayMs = 0.0, cpuSortMs = 0.0;
  // The scene graph is the single camera authority.  Refresh the retained
  // camera pointer (and the generation counter) from the camera node inside
  // the scene every frame so auto-clipping and the matrix build always track
  // the node navigation actually mutates, never a stale snapshot.
  this->refreshActiveCamera();
  if (prepBcStart) {
    vkRenderBreadcrumbSince(prepBcStart, 2000, "prepare refreshActiveCamera end");
  }

  // Keep the near/far planes tight around the scene so zooming and orbiting
  // never push geometry outside the view volume.  The GL SoRenderManager does
  // this automatically (VARIABLE_NEAR_PLANE); without the equivalent here,
  // the Vulkan viewport clips near faces when the camera is close and far
  // faces when the camera is far (FreeCAD's hidden GL viewer never renders,
  // so its auto-clipping never runs).
  const long clipBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  const long clipT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
  if (this->autoClipping != SoVulkanRenderManager::NO_AUTO_CLIPPING) {
    this->setClippingPlanes();
  }
  if (wantCpuTiming) {
    cpuClipMs = (vkRenderBreadcrumbNowUs() - clipT0) * 0.001;
  }
  if (clipBcStart) {
    vkRenderBreadcrumbSince(clipBcStart, 2000, "prepare setClippingPlanes end");
  }

  const long applyBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  SoIRRenderAction & action = this->irAction;
  action.setViewportRegion(this->viewportRegion);
  {
    static bool loggedAction = false;
    if (!loggedAction && SoVulkanConfig::get().debug.backendDebug) {
      loggedAction = true;
      fprintf(stderr,
              "[DRAWLIST] manager irAction=%p overlayAction=%p scene=%p\n",
              (void *)&this->irAction, (void *)&this->overlayIrAction,
              (void *)this->scene);
    }
  }

  params.viewport = this->viewportRegion;
  // The viewport region is in device pixels, so carry the device-pixel ratio
  // into the render params.  The GL and Vulkan backends scale logical line
  // widths / point sizes by this (SoDrawStyle values are logical points);
  // without it the ratio stayed 1.0 and, on a fractional-scaling display,
  // lines/dots rendered 1/dpr too thin.
  params.devicePixelRatio = this->devicePixelRatio;
  params.viewMatrix.makeIdentity();
  params.projMatrix.makeIdentity();
  {
    static int camDiag = 0;
    if (breadcrumbsEnabled() && camDiag++ < 8) {
      const char * cname = this->camera
        ? this->camera->getTypeId().getName().getString() : "NULL";
      SbVec3f cpos(0.0f, 0.0f, 0.0f);
      float cheight = 0.0f;
      if (this->camera) {
        cpos = this->camera->position.getValue();
        if (this->camera->isOfType(SoOrthographicCamera::getClassTypeId())) {
          cheight = static_cast<const SoOrthographicCamera*>(this->camera)->height.getValue();
        }
      }
      fprintf(stderr, "[VK-TRACE] params cam=%s pos=(%.3f,%.3f,%.3f) height=%.3f "
                      "vpAspect=%.3f autoClip=%d near=%.4f far=%.4f\n",
              cname,
              static_cast<double>(cpos[0]), static_cast<double>(cpos[1]),
              static_cast<double>(cpos[2]), static_cast<double>(cheight),
              static_cast<double>(this->viewportRegion.getViewportAspectRatio()),
              static_cast<int>(this->autoClipping),
              static_cast<double>(this->computedNear),
              static_cast<double>(this->computedFar));
    }
  }
  params.clearColor = this->backgroundColor;
  if (breadcrumbsEnabled()) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      fprintf(stderr, "[VK-TRACE] prepareRenderParams backgroundGradient=%d\n", this->backgroundGradient ? 1 : 0);
    }
  }
  params.backgroundGradient = this->backgroundGradient;
  params.backgroundTopColor = this->backgroundTopColor;
  params.backgroundBottomColor = this->backgroundBottomColor;
  params.clearDepth = 1.0f;
  params.flags = 0;
  if (clearwindow || this->clearWindow) {
    params.flags |= SO_PARAM_CLEAR_WINDOW;
  }
  if (clearzbuffer || this->clearDepth) {
    params.flags |= SO_PARAM_CLEAR_DEPTH;
  }
  params.renderTarget = this->renderTarget;
  // Hand the camera generation counter to the backends so a camera move is
  // detected unambiguously (see SoRenderParams::cameraVersion).
  params.cameraVersion = this->cameraVersion;
  // Interaction LOD is driven by the viewport adapter's camera-move timer and
  // applies to every backend: the RT backend lowers its bounce count, the
  // raster Vulkan backend draws wide lines as plain 1px lines (no CPU quad
  // expansion) for the duration of the motion.
  params.interactionLod = this->interactionLod ? TRUE : FALSE;

  // SoIRRenderAction::apply() resets the frame, so the camera and the scene
  // must be traversed in a single apply() call.  The managed scene graph is
  // geometry-only: the camera is stored as a separate member (matching
  // SoRenderManager/SoSceneManager, which apply the camera independently of
  // the scene root).  Build a path [camera, scene] so SoCamera::doAction()
  // installs the projection/viewing matrix elements before any geometry is
  // recorded; otherwise every command carries identity matrices and the view
  // renders at the origin with an identity projection (blank/wrong view,
  // invisible geometry, and a camera that appears not to follow navigation).
  // The overlay/decoration scenes are NOT traversed here: their node-ids churn
  // every frame with the camera, and folding them into this traversal would
  // both re-record the (stable) main geometry and defeat the retained-IR replay
  // below.  They are re-recorded separately afterwards (cheap) and merged onto
  // this main list.
   SbBool irReplayed = FALSE;
   // The graph fingerprint walk is O(N) over the scene nodes, so on a retained
   // (replayed) frame with no scene change at all it is pure waste.  The walk
   // folds scene node-ids but deliberately skips camera, light, environment and
   // tag/infra nodes, so the fingerprint is invariant under camera motion; the
   // only thing that changes it is a change to a render-affecting node.  An
   // SoNodeSensor attached to the scene root fires whenever any descendant is
   // notified (a field write or a child-list edit, including the camera pose --
   // FreeCAD keeps the camera inside the scene graph).  When the sensor has NOT
   // fired since the last walk the cached fingerprint is still exact and the
   // walk/re-traversal can be skipped via the branch below.  When it HAS fired
   // we must recompute the fingerprint (see the else) to distinguish camera-only
   // churn (identical fingerprint -> replay) from a real content change
   // (different fingerprint -> re-traverse).
   uint64_t graphFp;
   const SbVec2s fpVpSize = this->viewportRegion.getViewportSizePixels();
    if (this->graphFingerprintValid && this->lastFpValid &&
        !this->sceneGraphDirty && this->scene == this->lastFpScene &&
        fpVpSize == this->lastFpViewport && this->devicePixelRatio == this->lastFpDpr) {
      graphFp = this->graphFingerprint;
    }
    else {
      // Recompute the graph fingerprint whenever the scene sensor has fired.
      // graphFingerprintWalk folds the node-id of every non-camera-coupled,
      // render-affecting node, so a real content change that alters what the
      // walk sees -- an add/remove, a geometry rebuild, a transform/model-matrix
      // write, a material/selection field write, or a SoSwitch::whichChild
      // visibility toggle -- bumps at least one folded id and produces a
      // DIFFERENT fingerprint, correctly forcing a re-traverse below.  The
      // root sensor also fires on camera pose/headlight motion, but those nodes
      // are EXCLUDED from the walk, so camera-only frames yield an identical
      // (camera-invariant) fingerprint and the retained main list replays.
      //
      // Do NOT short-circuit this walk with a cheap hash of the retained draw
      // list: that is unsound.  The retained list is the PREVIOUS frame's graph
      // output, so a visibility toggle (SoSwitch::whichChild) changes the scene
      // without yet changing the retained commands -- their fingerprint is
      // therefore unchanged, and skipping the walk would replay the stale list
      // forever, leaving an object's show/hide state never reflected in the
      // viewport.  The walk is the authoritative signal and is O(nodes) (a few
      // mixHash per node), far cheaper than an actual re-traversal.
      graphFp = this->computeGraphFingerprint();
      this->sceneGraphDirty = FALSE;
    }
   this->lastFpScene = this->scene;
   this->lastFpViewport = fpVpSize;
   this->lastFpDpr = this->devicePixelRatio;
   this->lastFpValid = TRUE;
    if (this->scene || this->camera || this->overlayScene
        || this->decorationScene) {
     if (irReplayEnabled() && this->graphFingerprintValid &&
         graphFp == this->graphFingerprint) {
      // Camera-only frame: the main graph, the viewport, and the
      // caller-published revision are unchanged, so the retained main IR draw
      // list is exactly what a full traversal would produce -- keep it (and
      // the geometry/texture caches keyed on it) and restamp the frame view
      // after the matrices are built below.
      // The graph-fingerprint walk folds node-id of every non-camera-coupled
      // reachable node, so a real edit -- a transform/matrix move, a geometry
      // rebuild, an add/remove, a material/selection field write (SoShape::
      // notify() bumps its own id and drops the retained tessellation) --
      // re-bumps at least one folded id and therefore produces a DIFFERENT
      // fingerprint, correctly forcing a re-traverse.  Camera pose/headlight
      // motion is excluded from the walk, so it leaves the fingerprint
      // unchanged and replays.  Relying on fingerprint equality (not the
      // sensor dirty flag) is what makes this robust: FreeCAD keeps the
      // camera inside the scene graph, so the "camera never dirties the
      // scene sensor" assumption the dirty flag encodes is false here, and
      // without this the retained main list would be re-traversed (O(scene))
      // every navigation frame even though the geometry is unchanged.
      irReplayed = TRUE;
    }
    else {
      // Reuse the persistent traversal root: clear its children only when
      // the child set actually changes so the refcount stays balanced and
      // navigation frames stop re-triggering child-list notifications.
      SoSeparator * root = this->frameRoot;
      SoNode * const want[4] = { this->camera, this->scene,
                                 nullptr, nullptr };
      const SbBool sameChildren = this->rootChildrenValid &&
        this->rootChildren[0] == want[0] &&
        this->rootChildren[1] == want[1] &&
        this->rootChildren[2] == want[2] &&
        this->rootChildren[3] == want[3];
      if (!sameChildren) {
        root->removeAllChildren();
        for (int i = 0; i < 4; ++i) {
          if (want[i]) {
            root->addChild(want[i]);
          }
        }
        for (int i = 0; i < 4; ++i) {
          this->rootChildren[i] = want[i];
        }
        this->rootChildrenValid = TRUE;
      }
        // Tell Part's BRep point set whether to emit its base vertex markers.
        // The backend cannot distinguish them from any other point primitive,
        // so the decision is made at traversal time (Sketcher/Points/Draft point
        // primitives are different nodes and are unaffected).
        action.setModelPointsVisible(this->pointsOverlay
                                     || SoVulkanConfig::get().raster.points);
        const long applyT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
        action.apply(root);
        if (wantCpuTiming) {
          cpuApplyMs = (vkRenderBreadcrumbNowUs() - applyT0) * 0.001;
        }
       this->mainCommandCount =
         action.getDrawList().getNumCommands();
       this->graphFingerprint = graphFp;
       this->graphFingerprintValid = TRUE;
       this->lastFrameViewValid = FALSE;
    }
  }
  else {
    action.beginFrame();
    this->graphFingerprintValid = FALSE;
    this->rootChildrenValid = FALSE;
  }
   if (applyBcStart) {
    vkRenderBreadcrumbSince(applyBcStart, 2000, "prepare action.apply end");
  }

  // A replayed frame retains the main list verbatim and no traversal ran, so
  // the retained main geometry content is bit-identical to the previous
  // frame: backends may skip re-hashing geometry whose pointer identity
  // already matches (the content hash exists to catch in-place edits, which
  // only a traversal produces).  Freshly recorded overlay/decoration commands
  // are out of scope for this guarantee.
  params.geometryContentUnchanged = irReplayed;

  const long matricesBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  SoDrawList & list = action.getMutableDrawList();

  // ---- Always re-record the overlay/decoration (cheap) and merge ----------
  // The nav cube and the axis cross mirror the camera, so their scene node-ids
  // are re-touched every frame.  They cannot be retained with the stable main
  // list, so re-traverse them here into a separate IR action and append their
  // fresh commands onto the (retained) main list.  Truncate the previous
  // frame's overlay region first -- it references geometry storage owned by the
  // overlay action, which apply() just reset -- so the draw list the backend
  // sees is [main..., overlay...] with no stale/dangling overlay commands.
  SbBool overlayApplied = FALSE;
  {
    SoSeparator * oroot = this->overlayRoot;
    SoNode * const owant[3] = { this->camera, this->overlayScene,
                                this->decorationScene };
    const SbBool oSame = this->overlayRootChildrenValid &&
      this->overlayRootChildren[0] == owant[0] &&
      this->overlayRootChildren[1] == owant[1] &&
      this->overlayRootChildren[2] == owant[2];
    if (!oSame) {
      oroot->removeAllChildren();
      // Decorations (axis cross) are added after the overlay scene so their
      // overlay-pass commands draw on top of the navigation cube, matching GL's
      // foreground/decoration render order.
      for (int i = 0; i < 3; ++i) {
        if (owant[i]) {
          oroot->addChild(owant[i]);
        }
      }
      for (int i = 0; i < 3; ++i) {
        this->overlayRootChildren[i] = owant[i];
      }
      this->overlayRootChildrenValid = TRUE;
    }
    this->overlayIrAction.setViewportRegion(this->viewportRegion);
    this->overlayIrAction.setModelPointsVisible(this->pointsOverlay
                                                || SoVulkanConfig::get().raster.points);
    if (oroot->getNumChildren() > 0) {
      // apply() resets the frame first (beginFrame), so the overlay action's
      // previous draw list/geometry pool is released before re-recording.
      this->overlayIrAction.apply(oroot);
      overlayApplied = TRUE;
    }
    else {
      // No overlay/decoration this frame: clear the previous frame's overlay
      // list so the merge below does not append stale commands.
      this->overlayIrAction.beginFrame();
    }
  }
  {
    // The main region is whatever the MAIN IR action recorded, and on a fresh
    // document that is often just the hidden anchor cube (a tiny non-indexed
    // SoCube, ~36 vertices) - the real document shapes may not be in it at
    // all.  Do not assume "the scene rendered" just because main > 0: check
    // mainMaxVc against the shape's real vertex count.  Feature work that is
    // only ever exercised against the nav cube can appear to work while never
    // touching real, indexed document geometry (tools/fcprobe/vk_geomlod_probe.py
    // is the check that guards against exactly this).
    const int numMain = static_cast<int>(this->mainCommandCount);
    if (numMain < list.getNumCommands()) {
      list.truncate(numMain);
    }
    if (overlayApplied) {
      const SoDrawList & ovl = this->overlayIrAction.getDrawList();
      const int ovlCount = ovl.getNumCommands();
      for (int i = 0; i < ovlCount; ++i) {
        list.addCommand(ovl.getCommand(i));
      }
    }
    if (SoVulkanConfig::get().debug.backendDebug) {
      int mainMax = 0;
      int totalMax = 0;
      for (int i = 0; i < list.getNumCommands(); ++i) {
        const int vc = static_cast<int>(list.getCommand(i).geometry.vertexCount);
        if (i < numMain && vc > mainMax) mainMax = vc;
        if (vc > totalMax) totalMax = vc;
      }
      fprintf(stderr,
              "[DRAWLIST] main=%d total=%d replayed=%d mainMaxVc=%d totalMaxVc=%d\n",
              numMain, list.getNumCommands(), irReplayed ? 1 : 0, mainMax,
              totalMax);
    }
  }

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
  if (clipDebugEnabled()) {
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

  if (matricesBcStart) {
    vkRenderBreadcrumbSince(matricesBcStart, 2000, "prepare matrix build end");
  }

  int dbgRestamped = -1;
  if (irReplayed) {
    // Camera-only frame: restamp the frame viewing matrix into every
    // non-overlay command that carried the previous traversal's viewing
    // element (commands stamped by a sub-camera keep their own matrix).
    // Lighting setups are world-space and need no re-derivation here.
    if (this->lastFrameViewValid) {
      const long replayT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
      // SbMatrix stores exactly float[4][4] (16 contiguous floats), so a
      // full-storage bit-compare says whether the viewing matrix changed at
      // all.  On a static camera (idle scene, no navigation) the replay
      // frame's view is bit-identical to the previous one: nothing to
      // restamp, so the O(N) per-command getValue + memcmp + matrix-copy
      // loop is skipped entirely.
      if (std::memcmp(&this->lastFrameView[0][0], &params.viewMatrix[0][0],
                      sizeof(float) * 16) != 0) {
        SbMat lastView;
        this->lastFrameView.getValue(lastView);
        const int numCommands = list.getNumCommands();
        int restamped = 0;
        for (int i = 0; i < numCommands; ++i) {
          SoRenderCommand & command = list.getCommand(i);
          if (command.pass == SO_RENDERPASS_OVERLAY) {
            continue;
          }
          SbMat cmdView;
          command.viewMatrix.getValue(cmdView);
          if (std::memcmp(&cmdView[0][0], &lastView[0][0], sizeof(cmdView)) == 0) {
            command.viewMatrix = params.viewMatrix;
            ++restamped;
          }
        }
        dbgRestamped = restamped;
        this->lastFrameView = params.viewMatrix;
      }
      if (wantCpuTiming) {
        cpuReplayMs = (vkRenderBreadcrumbNowUs() - replayT0) * 0.001;
      }
    }
  }
  else if (!this->lastFrameViewValid) {
    // Remember the exact viewing-element bits the current traversal stamped
    // so the next replay (if any) can identify restrikable entries.
    const int numCommands = list.getNumCommands();
    for (int i = 0; i < numCommands; ++i) {
      const SoRenderCommand & command = list.getCommand(i);
      if (command.pass != SO_RENDERPASS_OVERLAY) {
        this->lastFrameView = command.viewMatrix;
        this->lastFrameViewValid = TRUE;
        break;
      }
    }
  }
  if (SoVulkanConfig::get().debug.lightReplayDebug && vkLightFrameDbgBudget-- > 0) {
    const SbMatrix & v = params.viewMatrix;
    float qx = 0, qy = 0, qz = 0, qw = 1;
    SbVec3f camPos(0.0f, 0.0f, 0.0f);
    if (this->camera) {
      const SbRotation camRot = this->camera->orientation.getValue();
      camRot.getValue(qx, qy, qz, qw);
      camPos = this->camera->position.getValue();
    }
    fprintf(stderr,
            "[VKS] fp=%016lx replayed=%d lastViewValid=%d restamped=%d"
            " viewT=(%.2f,%.2f,%.2f) viewM00=%.3f camPos=(%.1f,%.1f,%.1f)"
            " camQ=(%.3f,%.3f,%.3f,%.3f) scene=%p\n",
            (unsigned long)graphFp,
            (int)irReplayed, (int)this->lastFrameViewValid, dbgRestamped,
            v[0][3], v[1][3], v[2][3], v[0][0],
            camPos[0], camPos[1], camPos[2],
            qx, qy, qz, qw,
            reinterpret_cast<const void *>(this->scene));
  }
  const long sortBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  const long sortT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
  const bool rtActive = this->rayTracing && this->rtxBackendInitialized;
  if (rtActive) {
    // The path-traced frame consumes no painter's-algorithm order: the trace
    // is draw-order independent, and the raster overlay composite
    // (recordTracedComposite/recordOverlayBlock) iterates the list in
    // insertion order.  Re-sorting every RT frame is pure CPU waste, and the
    // staleness flag makes the first non-RT frame re-sort unconditionally.
    this->lastSortValid = FALSE;
  }
  else if (irReplayed && this->lastSortValid &&
           list.getNumCommands() == this->lastSortCommandCount &&
           std::memcmp(&this->lastSortView[0][0], &params.viewMatrix[0][0],
                       sizeof(float) * 16) == 0) {
    // A retained (replayed) list with a bit-identical view and an unchanged
    // command count sorts exactly as the previous frame: reuse the previous
    // frame's sorted order instead of re-deriving every command's sort key and
    // re-running the stable sort.  The count check is required because the
    // overlay region above was truncated/re-appended this frame; if its size
    // changed, the previous order holds stale (out-of-range) indices.
  }
  else {
    list.buildSortedOrder(params.viewMatrix);
    this->lastSortView = params.viewMatrix;
    this->lastSortValid = TRUE;
    this->lastSortCommandCount = list.getNumCommands();
  }
  if (wantCpuTiming) {
    cpuSortMs = (vkRenderBreadcrumbNowUs() - sortT0) * 0.001;
    std::fprintf(stderr,
                 "[RTDBG] cpuTiming clip=%.2f apply=%.2f restamp=%.2f "
                 "sort=%.2f cmds=%d rt=%d\n",
                 cpuClipMs, cpuApplyMs, cpuReplayMs, cpuSortMs,
                 list.getNumCommands(), rtActive ? 1 : 0);
    std::fflush(stderr);
  }
  vkRenderBreadcrumbSince(sortBcStart, 2000, "prepare buildSortedOrder end");
  drawlist = &list;

  // Dump the draw list when COIN_DEBUG_RENDER_IR is set so the overlay
  // commands recorded by the highlight/selection paths can be inspected
  // (pass, depth state, diffuse color, vertex count).
  static int dumpCount = 0;
  if (coin_render_ir_trace_enabled() && dumpCount++ < 300) {
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
  // This is a diagnostic: its [CLIP] traces walk the draw-list vertices (a
  // multi-million-vertex mesh costs seconds per dump), so it MUST stay behind
  // the FC_VULKAN_CLIP_DEBUG gate.  Without the gate it dominates the frame on
  // a large scene.
  if (clipDebugEnabled()) {
    this->dumpClipDebug(list, params);
  }

  return TRUE;
}


// [CLIP] diagnostic trace, env-gated by FC_VULKAN_CLIP_DEBUG (verbose adds
// FC_VULKAN_CLIP_VERBOSE).  Kept out of prepareRenderParams() so the frame
// hot path is not dominated by this print-only branch.
void
SoVulkanRenderManagerP::dumpClipDebug(SoDrawList & list,
                                      const SoRenderParams & params)
{
  static int frames = 0;
  ++frames;
  if (clipVerboseEnabled() || frames == 10 || frames == 50 ||
      frames % 25 == 0) {
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

