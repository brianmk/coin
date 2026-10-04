// src/rendering/SoVulkanRenderManager.cpp

#include <Inventor/rendering/SoVulkanRenderManager.h>
#include "rendering/SoVulkanDebug.h"

#include <Inventor/SoPath.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/SbXfBox3f.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/misc/SoNotRec.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/sensors/SoDataSensor.h>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoTransformSeparator.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include "rendering/SoRenderBackend.h"
#include "rendering/SoClippingPlanes.h"
#include "rendering/SoRenderIRP.h"
#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanShared.h"
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanReplayKey.h"

class SoVulkanRenderManagerP;
static void vulkanSceneGraphChangedCallback(void * data, SoSensor * sensor);

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

// Graph-fingerprint helpers live in SoVulkanReplayKey.h (testable).
using namespace CoinVulkanReplay;

namespace {

// Cached env-var checks: on the per-frame path, and getenv() is not
// thread-safe, so resolve once. All use SoVulkanShared::envFlagEnabled
// (honors "0"/"false"/"off" opt-outs).
// Per-phase CPU timing for the fcprobe harness, gated by COIN_VULKAN_FRAME_TIMING.
// Emits its own [RTDBG] cpuTiming (clip/apply/restamp/sort).
bool frameTimingEnabled()
{
  static const bool enabled = SoVulkanShared::envFlagEnabled("COIN_VULKAN_FRAME_TIMING");
  return enabled;
}

long vkRenderBreadcrumbNowUs()
{
  return SoVulkanShared::steadyNowUs();
}

int vkLightFrameDbgBudget = 192;
int vkLightFpDbgBudget = 192;

// Content fingerprint of the first \a mainCount commands: mixes model matrix,
// geometry pointers and counts. The scene bbox/auto-clip planes depend on main
// geometry + world transform; command count alone is unsound (moving a body
// keeps the count but changes extent). Overlay/decoration are excluded; this is
// cheaper than a per-frame bbox walk and is a sound change signal.
uint64_t computeSceneFingerprint(const SoIRRenderAction & action, int mainCount)
{
  const SoDrawList & drawList = action.getDrawList();
  uint64_t h = 0x7f4a7c159e3779b9ULL;
  const int n = std::min(mainCount, drawList.getNumCommands());
  for (int i = 0; i < n; ++i) {
    const SoRenderCommand & c = drawList.getCommand(i);
    const SoGeometryDesc & g = c.geometry;
    // Mix the world transform so object/ancestor motion still invalidates.
    for (int k = 0; k < 16; ++k) {
      uint32_t bits;
      std::memcpy(&bits, &(c.modelMatrix[k >> 2][k & 3]), sizeof(bits));
      const uint64_t v = bits;
      h ^= v + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
    }
    // Mix geometry identity (pointers track reallocated buffers) and counts.
    const uint64_t ids[5] = {
      reinterpret_cast<uintptr_t>(g.positions),
      reinterpret_cast<uintptr_t>(g.normals),
      reinterpret_cast<uintptr_t>(g.texcoords),
      reinterpret_cast<uintptr_t>(g.colors),
      reinterpret_cast<uintptr_t>(g.indices),
    };
    for (uint64_t v : ids) {
      h ^= v + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
    }
    h ^= g.vertexCount + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
    h ^= g.indexCount + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
    h ^= g.normalCount + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
    h ^= static_cast<uint64_t>(g.topology) + 0x517cc1b727220a95ULL +
         (h << 6) + (h >> 2);
  }
  return h;
}

// COIN_VULKAN_IR_REPLAY=0 forces a full scene re-traversal every frame.
bool irReplayEnabled()
{
  // Default on; shared helper honors the full 0/false/off opt-out set.
  static const bool enabled =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_IR_REPLAY", true);
  return enabled;
}

} // namespace

class SoVulkanRenderManagerP {
public:
  SoVulkanRenderManagerP()
    : irAction(SbViewportRegion()),
      overlayIrAction(SbViewportRegion())
  {
    this->viewportRegion.setWindowSize(1, 1);
    // Persist one traversal root so prepareRenderParams() need not allocate
    // and ref/unref a separator each frame; only the root is retained.
    this->frameRoot = new SoSeparator;
    this->frameRoot->ref();
    // Separate root for the always-re-recorded overlay/decoration scenes.
    this->overlayRoot = new SoSeparator;
    this->overlayRoot->ref();
    // Dirty-tracking sensor on the main scene: fires on any descendant
    // notification, letting computeGraphFingerprint() skip the O(N) walk on
    // unchanged (static/camera-only) frames.
    this->sceneGraphSensor =
      new SoNodeSensor(vulkanSceneGraphChangedCallback, this);
    // Priority 0 makes it an "immediate" sensor AND makes SoDataSensor
    // populate the trigger node/op; at default priority the trigger is null
    // and the callback cannot tell a camera write from a real scene change.
    this->sceneGraphSensor->setPriority(0);
  }

  ~SoVulkanRenderManagerP()
  {
    // Detach/destroy the scene sensor first: its scene node is unref'd below.
    if (this->sceneGraphSensor) {
      this->sceneGraphSensor->detach();
      delete this->sceneGraphSensor;
      this->sceneGraphSensor = nullptr;
    }
    if (this->camera) {
      this->camera->unref();
    }
    if (this->scene) {
      this->scene->unref();
    }
    if (this->overlayScene) {
      this->overlayScene->unref();
    }
    if (this->decorationScene) {
      this->decorationScene->unref();
    }
    if (this->frameRoot) {
      this->frameRoot->unref();
    }
    if (this->overlayRoot) {
      this->overlayRoot->unref();
    }
  }

  SoNode * scene = nullptr;
  SoNode * overlayScene = nullptr;
  SoNode * decorationScene = nullptr;
  SoCamera * camera = nullptr;
  //! Cached resolveActiveCamera() result: the first camera in the scene and the
  //! child-index path to it. Revalidated in O(depth) rather than an O(scene)
  //! search; a camera-pose write keeps the path valid, a child-list edit on the
  //! path fails it and re-runs the search.
  SoCamera * resolvedCamera = nullptr;
  SoNode * resolvedCameraScene = nullptr;
  std::vector<int> resolvedCameraPath;
  //! Retained camera when the search last ran; caches the "no camera in scene"
  //! result so the fruitless O(scene) search (~44 ms on a 1600-shape scene) is
  //! not repeated every frame.
  SoCamera * resolvedCameraFallback = nullptr;
  bool resolvedCameraCached = false;
  // Persistent traversal root (see constructor).
  SoSeparator * frameRoot = nullptr;
  //! Persistent root for the always-re-recorded overlay/decoration scenes.
  SoSeparator * overlayRoot = nullptr;
  SoNode * overlayRootChildren[3] = {nullptr, nullptr, nullptr};
  SbBool overlayRootChildrenValid = FALSE;
  SbViewportRegion viewportRegion;
  SbColor4f backgroundColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbBool backgroundGradient = FALSE;
  SbColor4f backgroundTopColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbColor4f backgroundBottomColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbBool wireframeOverlay = FALSE;
  SbBool pointsOverlay = FALSE;
  SbBool tessellationOverlay = FALSE;
  //! Interaction LOD, persisted so a later RT-backend bring-up can re-apply it.
  SbBool interactionLod = FALSE;
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);
  //! Last setViewSettings() blob; the applied flag makes the first call always apply.
  SoVulkanViewSettings viewSettings;
  SbBool viewSettingsApplied = FALSE;
  SbBool clearWindow = TRUE;
  SbBool clearDepth = TRUE;
  void * renderTarget = nullptr;
  //! Device-pixel ratio of the Vulkan surface. The viewport region is in
  //! device pixels, so logical SoDrawStyle widths/sizes are scaled by this in
  //! the backends; also exposed to the SoDevicePixelRatio element.
  float devicePixelRatio = 1.0f;

  SoVulkanRenderManager::AutoClippingStrategy autoClipping =
    SoVulkanRenderManager::NO_AUTO_CLIPPING;
  float nearplanevalue = 0.6f;

  //! Bumped whenever the active camera's identity or pose changes. The RT
  //! backend reads this instead of diffing float view matrices, which are
  //! fragile (epsilon-swallowed moves, single-precision aliasing under a hash).
  uint32_t cameraVersion = 0;

  //! Pose fingerprint (position + forward) to catch in-place pose changes of
  //! the same camera node that a pointer comparison cannot see.
  uint32_t cameraPoseFingerprint = 0;

  //! 1-based ordinal of the last presented frame; bumped once per
  //! render()/renderExternal() and copied into SoRenderParams::frame so
  //! backends/dumps/probes share one monotonic key.
  uint32_t frameOrdinal = 0;

  //! --- Retained-IR replay state (camera-only frame fast path) ----------
  //! Graph fold from the last full traversal (graphFingerprintWalk); a match
  //! means the retained draw list (and its geometry/texture caches) reproduces.
  uint64_t graphFingerprint = 0;
  SbBool graphFingerprintValid = FALSE;
  //! Inputs of the last graph-fingerprint walk; with sceneGraphDirty (the root
  //! sensor) this skips the O(N) walk on unchanged frames. See prepareRenderParams.
  SbBool lastFpValid = FALSE;
  SoNode * lastFpScene = nullptr;
  SbVec2s lastFpViewport = SbVec2s(0, 0);
  float lastFpDpr = 1.0f;
  SbBool sceneGraphDirty = TRUE;
  SoNodeSensor * sceneGraphSensor = nullptr;
  //! Caller-published revision the walk cannot see (selection behind highlight roots).
  uint64_t externalRevision = 0;
  //! Set when the last traversal hit a camera-dependent node (e.g.
  //! SoShapeScale/SoAutoZoomTranslation -> setCameraDependent); such a scene
  //! must be re-recorded whenever the camera moves, not replayed.
  SbBool sceneCameraDependent = FALSE;
  //! cameraVersion of the retained list: replay refused for a camera-dependent scene while it differs.
  uint32_t sceneCameraDependentVersion = 0;
  //! Viewing matrix (SoViewingMatrixElement bits) of the last traversal; replay restamp key.
  SbMatrix lastFrameView;
  SbBool lastFrameViewValid = FALSE;
  //! View matrix the retained list's painter order was built for
  //! (buildSortedOrder); a bit-match means the prior order is still exact.
  SbMatrix lastSortView;
  SbBool lastSortValid = FALSE;
  //! Command count the sorted order was built for. Overlay is truncated and
  //! re-appended each frame, so a replay may reuse the prior order only while
  //! the total count is unchanged; else it holds out-of-range indices.
  int lastSortCommandCount = -1;
  //! Child pointers last installed in frameRoot, avoiding child-list churn on navigation.
  SoNode * rootChildren[4] = {nullptr, nullptr, nullptr, nullptr};
  SbBool rootChildrenValid = FALSE;

  //! Current render-affecting graph fingerprint (see graphFingerprintWalk).
  uint64_t computeGraphFingerprint() const;

  //! Resolve this frame's camera. The scene graph is the authority (FreeCAD
  //! navigation mutates the in-scene camera); the retained setCamera() pointer
  //! is only a fallback for cameras outside the scene root. nullptr if none.
  SoCamera * resolveActiveCamera();

  //! Refresh the retained camera from the scene authority; bump cameraVersion on change.
  void refreshActiveCamera();


  // Near/far planes from setClippingPlanes(), consumed by prepareRenderParams().
  // The projection is built from these, never SoCamera::nearDistance/farDistance:
  // the camera node is shared with FreeCAD's hidden GL viewer, whose render
  // manager writes those fields concurrently (racing reads gave an intermittent
  // wrong near plane). The planes are also published onto the node (on change)
  // so SoRayPickAction's ray depth range sees scene-fitted planes.
  float computedNear = 1.0f;
  float computedFar = 10.0f;
  // Camera back-off along the view direction from the zoom wall; 0 when clear of the surface.
  float cameraShiftZ = 0.0f;

  // Cached world-space scene bbox for setClippingPlanes(). Only the (static)
  // world box is cached; each frame re-applies the cheap camera transform rather
  // than a full traversal. Invalidated on scene change or IR command-count change.
  SbXfBox3f sceneWorldBBox;
  SoNode * sceneBBoxScene = nullptr;
  uint64_t sceneBBoxFingerprint = 0;
  bool sceneBBoxCached = false;
  //! Cached scene fingerprint: on camera-only frames (list retained, sensor
  //! quiet) the O(main-commands) hash is invariant, so skip it. Revalidated
  //! against scene pointer and retained main command count.
  uint64_t sceneFpCached = 0;
  SoNode * sceneFpScene = nullptr;
  uint32_t sceneFpMainCount = 0;
  SbBool sceneFpValid = FALSE;

  SoIRRenderAction irAction;
  //! Re-records the overlay/decoration scenes every frame (their node-ids churn
  //! with the camera); commands are appended onto the replayed main list.
  SoIRRenderAction overlayIrAction;
  //! Main (non-overlay) command count in irAction, separating replay from fresh overlay.
  uint32_t mainCommandCount = 0;
  SoVulkanRenderBackend backend;
  SbBool backendInitialized = FALSE;
  // Pipeline-cache path set before initialize(); forwarded to the backend there.
  std::string pipelineCachePath;
  // Device context borrowed at initialize() (not owned); retained for lazy RT
  // bring-up. Cleared in shutdown().
  SoVulkanDeviceContext * initContext = nullptr;

  // Recompute near/far clipping planes from the camera-space scene bbox,
  // mirroring SoRenderManagerP::setClippingPlanes() (GL auto-clipping). The
  // camera is a separate member, so its matrix is built directly from the node.
  void setClippingPlanes(void);

  // Traverse the scene into params. FALSE when backend/target unavailable.
  SbBool prepareRenderParams(SbBool clearwindow,
                             SbBool clearzbuffer,
                             SoDrawList *& drawlist,
                             SoRenderParams & params);

};

// Retained-pointer assignment: ref the new node, unref the old, no-op when
// unchanged (the scene setters, setCamera() and refreshActiveCamera() did this by hand).
template <typename T>
static void
setRetainedNode(T *& slot, T * node)
{
  if (slot == node) return;
  if (slot) slot->unref();
  slot = node;
  if (slot) slot->ref();
}

// Mark the graph fingerprint dirty on a render-affecting main-scene change. The
// root sensor fires on ANY subtree notification, including the camera/headlight
// writes FreeCAD does every navigation frame; those don't change the retained
// list (their ids are skipped via fingerprintSkipsNodeId), so treating them as
// dirty forced the O(scene) walk (~55 ms/1600 shapes) every frame. Sensor
// priority 0 (immediate) is required so SoDataSensor populates the trigger.
static void
vulkanSceneGraphChangedCallback(void * data, SoSensor * sensor)
{
  auto * pimpl = static_cast<SoVulkanRenderManagerP *>(data);
  auto * dataSensor = static_cast<SoDataSensor *>(sensor);
  const SoNotRec::OperationType op =
    sensor ? dataSensor->getTriggerOperationType() : SoNotRec::UNSPECIFIED;
  const bool fieldChange =
    (op == SoNotRec::FIELD_UPDATE || op == SoNotRec::UNSPECIFIED);
  SoNode * trigger = sensor ? dataSensor->getTriggerNode() : nullptr;
  if (fieldChange && trigger && fingerprintSkipsNodeId(trigger)) {
    return;
  }
  pimpl->sceneGraphDirty = TRUE;
}

SoVulkanRenderManager::SoVulkanRenderManager()
  : pimpl(new SoVulkanRenderManagerP)
{
}

SoVulkanRenderManager::~SoVulkanRenderManager()
{
  // Shut down through the manager entry point so the backend releases its
  // resources while the borrowed device/queue are still valid and the init
  // context is invalidated; relying on the member destructor alone could run
  // after the embedder has torn the device down.
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
  // Re-arm the dirty sensor on the new scene and mark the fingerprint dirty so
  // the next frame re-walks instead of trusting a stale cache.
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
  // The scene graph is the single camera authority: this retained pointer is only
  // a fallback for scenes with no camera. When the scene does contain a camera,
  // refreshActiveCamera() re-resolves it each frame and owns its lifetime.
  // FreeCAD replaces the node on perspective/ortho toggle; keep a reference.
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
}

void
SoVulkanRenderManager::setTessellationOverlay(SbBool enabled)
{
  this->pimpl->tessellationOverlay = enabled;
  this->pimpl->backend.setTessellationOverlay(enabled);
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
  // Diff the whole blob: individual setters are unconditional, so re-applying it is waste.
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
  this->setEdgeColor(settings.edgeColor);
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
  // QVulkanWindow re-enters initResources() on every Expose/Hide/Resize/Move;
  // those events only recreate the swapchain, the instance/device/queue
  // survive. An unchanged device context is NOT a device reset, so rebuilding
  // the backend and every cached pipeline would be pure waste.
  //
  // NOTE (known Qt artifact, not a FreeCAD defect): with validation enabled the
  // first frames may log VUID-vkQueueSubmit-pSignalSemaphores-00067. That submit
  // is QVulkanWindow's own present (ours are fence-based), reusing its
  // per-swapchain render-finished semaphore. Transient; remedy is a Qt change.
  if (this->pimpl->backendInitialized && this->pimpl->initContext
      && this->pimpl->initContext->device == context->device) {
    this->pimpl->initContext = context;
    return TRUE;
  }

  SoRenderBackendInitParams params;
  params.userData = context;
  // Forward the pipeline-cache path before the backend creates its VkPipelineCache.
  this->pimpl->backend.setPipelineCachePath(this->pimpl->pipelineCachePath);
  if (!this->pimpl->backend.initialize(params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::initialize",
                              "backend initialization failed");
    return FALSE;
  }
  this->pimpl->backendInitialized = TRUE;
  // One-shot after successful init; COIN_VULKAN_BACKEND_DEBUG gets a config dump.
  if (SoVulkanShared::envFlagEnabled("COIN_VULKAN_BACKEND_DEBUG")) {
    SoVulkanConfig::dump();
  }
  // Retain the borrowed context (used by the same-device early return); cleared in shutdown().
  this->pimpl->initContext = context;
  return TRUE;
}

void
SoVulkanRenderManager::shutdown(void)
{
  if (this->pimpl->backendInitialized) {
    this->pimpl->backend.shutdown();
    this->pimpl->backendInitialized = FALSE;
  }
  // Retained context is only valid for the window device/queue lifetime; do not reuse after.
  this->pimpl->initContext = nullptr;
}

void
SoVulkanRenderManager::setMaxFramesInFlight(uint32_t count)
{
  this->pimpl->backend.setMaxFramesInFlight(count);
}

void
SoVulkanRenderManager::setPipelineCachePath(const std::string & path)
{
  // Stored and forwarded in initialize() (createPipelineCache() reads it once);
  // forwarding here too covers a caller setting it on an initialized manager.
  this->pimpl->pipelineCachePath = path;
  this->pimpl->backend.setPipelineCachePath(path);
}

void
SoVulkanRenderManager::setSceneLights(const SoLightingData & lighting)
{
  // Raster executor lighting; without it, lighting comes from the world-fixed
  // IR capture and highlights don't follow the camera like Coin GL.
  this->pimpl->backend.setSceneLights(lighting);
}

void
SoVulkanRenderManager::setInteractionLod(SbBool active)
{
  this->pimpl->interactionLod = active;
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
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer,
                                        drawlist, params)) {
    return FALSE;
  }
  params.frame = ++this->pimpl->frameOrdinal;
  if (!this->pimpl->backend.renderExternal(*drawlist, params, commandBuffer,
                                           renderPass, framebuffer)) {
    SoDebugError::postWarning("SoVulkanRenderManager::renderExternal",
                              "backend render failed (%d draw commands)",
                              drawlist->getNumCommands());
    return FALSE;
  }
  return TRUE;
}

uint64_t
SoVulkanRenderManagerP::computeGraphFingerprint() const
{
  uint64_t h = 0xcbf29ce484222325ULL;
  mixHash(h, reinterpret_cast<uintptr_t>(this->camera));
  mixHash(h, reinterpret_cast<uintptr_t>(this->scene));
  mixHash(h, reinterpret_cast<uintptr_t>(this->overlayScene));
  mixHash(h, reinterpret_cast<uintptr_t>(this->decorationScene));
  // The replay gate keys on the MAIN scene only (plus viewport and external
  // revision). Overlay/decoration node-ids churn every frame (the nav cube/axis
  // cross mirror the camera), so folding them in forced a re-traversal every
  // frame; they are re-recorded separately. Their pointer mixes stay constant.
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
  // Cache the resolved node. Runs twice per frame (refreshActiveCamera() then
  // setClippingPlanes()); the fallback SoSearchAction is an O(scene) search
  // measured at ~41 ms/1600 shapes, and the root sensor fires on the camera-pose
  // write every navigation frame, so gating on the dirty flag alone kept it on
  // the moving frames. Re-validate with the stored child-index path: O(depth),
  // no search (a camera-pose write keeps it; a child-list edit re-runs it).
  if (this->resolvedCameraCached && this->resolvedCameraScene == this->scene) {
    if (this->resolvedCamera) {
      SoNode * node = this->scene;
      bool valid = node != nullptr;
      for (int index : this->resolvedCameraPath) {
        if (!node || !node->isOfType(SoGroup::getClassTypeId())) {
          valid = false;
          break;
        }
        SoGroup * group = static_cast<SoGroup *>(node);
        if (index < 0 || index >= group->getNumChildren()) {
          valid = false;
          break;
        }
        node = group->getChild(index);
      }
      if (valid && node == this->resolvedCamera) {
        return this->resolvedCamera;
      }
    }
    else if (this->resolvedCameraFallback == this->camera) {
      // Cached "no camera in scene"; the retained camera is the authority.
      return this->camera;
    }
  }
  // About to run the search: reset and record its inputs.
  this->resolvedCameraCached = true;
  this->resolvedCamera = nullptr;
  this->resolvedCameraScene = this->scene;
  this->resolvedCameraFallback = this->camera;
  this->resolvedCameraPath.clear();

  // The setSceneGraph() root is the GL viewer's superscene, which CAN contain the
  // navigation camera, so prefer an in-scene camera (the single authority).
  // FreeCAD usually sets the camera explicitly, so the empty result is cached.
  if (this->scene) {
    if (this->scene->getTypeId().isDerivedFrom(SoSeparator::getClassTypeId())) {
      SoSeparator * sep = static_cast<SoSeparator *>(this->scene);
      for (int i = 0; i < sep->getNumChildren(); ++i) {
        SoNode * child = sep->getChild(i);
        if (child && child->isOfType(SoCamera::getClassTypeId())) {
          this->resolvedCamera = static_cast<SoCamera *>(child);
          this->resolvedCameraPath.push_back(i);
          return this->resolvedCamera;
        }
      }
    }
    // Deeper nesting: search the subtree for the first camera, mirroring
    // SoCamera::doAction() traversal order.
    SoSearchAction search;
    search.setType(SoCamera::getClassTypeId());
    search.setSearchingAll(TRUE);
    search.apply(this->scene);
    const SoPathList & paths = search.getPaths();
    if (paths.getLength() > 0) {
      SoPath * path = paths[0];
      this->resolvedCamera = static_cast<SoCamera *>(path->getTail());
      // Record descent from the root: start at 1 (0 is the root itself).
      for (int i = 1; i < path->getLength(); ++i) {
        this->resolvedCameraPath.push_back(path->getIndex(i));
      }
      return this->resolvedCamera;
    }
  }
  // No in-scene camera: fall back to the retained pointer.
  return this->camera;
}

// Refresh the retained camera from the scene authority and bump cameraVersion
// on change. Called at the top of prepareRenderParams() so downstream reads of
// `this->camera` see the node navigation mutates, not a stale snapshot.
void
SoVulkanRenderManagerP::refreshActiveCamera()
{
  SoCamera * resolved = this->resolveActiveCamera();
  if (resolved && resolved != this->camera) {
    setRetainedNode(this->camera, resolved);
    this->cameraVersion++;
  }
  else if (resolved == this->camera) {
    // Same node: detect in-place pose changes (rotation/pan/zoom) via the pose
    // fingerprint so the backend's viewChanged reliably fires.
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

  // Recompute the world bbox only when the scene pointer changed or the main-
  // command fingerprint differs (it covers world transform + geometry identity,
  // so a moved object with the same count still invalidates). Overlay commands
  // are excluded (camera-dependent). The camera pose is applied below each frame.
  const int mainCount = static_cast<int>(this->mainCommandCount);
  // Reuse it when the main list provably has not changed: same scene pointer and
  // main count and the sensor quiet (any change raises sceneGraphDirty); on those
  // camera-only frames the O(main-commands) hash would just repeat last value.
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
  // A scene change with an empty cached box forces a fresh bbox. The fingerprint
  // hashes the *retained* list (one frame behind), so a SoSwitch::whichChild
  // toggle leaves it unchanged and would keep a stale empty default (near=1/
  // far=10) that clips GUI trackers at the origin (notably the Draft grid). Only
  // the empty case recomputes; a non-empty box still brackets the origin.
  if (!this->sceneBBoxCached ||
      (this->sceneGraphDirty && this->sceneWorldBBox.isEmpty())) {
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

  // Transform the world bbox into camera coordinates: translate to the camera
  // origin, then rotate by the inverse orientation (same math as
  // SoRenderManagerP::setClippingPlanes()). The camera is a separate member.
  SbMatrix mat;
  mat.setTranslate(-camera->position.getValue());
  xbox.transform(mat);
  mat = camera->orientation.getValue().inverse();
  xbox.transform(mat);
  SbBox3f box = xbox.project();

  float sizeX, sizeY, sizeZ;
  box.getSize(sizeX, sizeY, sizeZ);
  float boxDiagonal = std::sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ);

  // Clipping offset: 1% of the bbox diagonal, clamped to [float epsilon, 1.0].
  float clippingOffset = SbMin(1.0f, SbMax(std::numeric_limits<float>::epsilon(),
                                           0.01f * boxDiagonal));
  float zmin = box.getMin()[2];
  float zmax = box.getMax()[2];

  // Vector-graphics zoom wall: a CAD camera must never clip into a solid. Once
  // the nearest boundary is within delta (or behind), back the *effective* camera
  // out along the view so the nearest surface stays delta in front; zooming then
  // scales continuously until the wall pins the view. The shift applies to the
  // box here and the view matrix in prepareRenderParams(); the camera node is
  // untouched. delta = 0.001 * clippingOffset (~100000x magnification).
  float shiftZ = 0.0f;
  // Only engage when geometry spans the view direction (zmin < 0); with the
  // whole scene behind the camera, backing out would flip the view.
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

  // Rebuild the box with the shifted z extent for the clipping core below.
  SbBox3f clippedBox = box;
  if (shiftZ != 0.0f) {
    float x0, y0, z0, x1, y1, z1;
    box.getBounds(x0, y0, z0, x1, y1, z1);
    clippedBox.setBounds(x0, y0, zmin, x1, y1, zmax);
  }

  // Shared near/far computation with the legacy GL SoRenderManager.
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

  // Do NOT bail when farval <= 0: an orthographic negative near/far pair is
  // meaningful (ortho extends behind the projection point), matching
  // SoRenderManagerP::setClippingPlanes; bailing would blank a fresh document.

  // Never let the near plane fall beyond the closest geometry in front: zoomed
  // in the 1% offset and VARIABLE_NEAR_PLANE floor can push it past surfaces.
  // Clamp only while the camera is outside the box (closest > 0); closest is from
  // the *shifted* box (-zmax), so the zoom wall keeps it at delta.
  const float closest = -zmax;
  if (closest > 0.0f) {
    if (nearval > closest) {
      nearval = closest;
    }
    // Just outside, the 1% offset can exceed the nearest distance and push the
    // near plane behind the camera (nearval <= 0), inverting the volume and
    // clipping front geometry (viewAll() can also leave it at 0).
    if (nearval <= 0.0f) {
      nearval = SbMin(closest, clippingOffset);
      if (nearval <= 0.0f) {
        nearval = std::numeric_limits<float>::epsilon();
      }
    }
  }
  else if (zmin < 0.0f) {
    // Camera inside the bounds (zmin < 0 < zmax): the bbox near is negative
    // and would invert the projection and clip everything; fall back to a
    // small positive plane anchored on the clipping offset.
    if (nearval < clippingOffset) {
      nearval = clippingOffset;
    }
  }
  // else: whole scene behind the camera (zmin >= 0) -- keep the signed
  // near/far, the orthographic case the legacy GL manager renders as-is.

  // Keep the view volume well-formed if far lands behind the camera or inverts.
  if (farval <= nearval) {
    farval = nearval + clippingOffset;
  }

  const float SLACK = kSoClippingSlack;
  const float newnear = nearval >= 0 ? nearval * (1.0f - SLACK)
                                     : nearval * (1.0f + SLACK);
  const float newfar = farval >= 0 ? farval * (1.0f + SLACK)
                                   : farval * (1.0f - SLACK);

  // Store privately: prepareRenderParams() builds the projection from these,
  // independent of concurrent GL-manager writes to the camera node.
  this->computedNear = newnear;
  this->computedFar = newfar;

  // Publish the planes onto the shared camera node too: it is the clipping
  // authority other Coin consumers read (SoRayPickAction's ray depth range), and
  // FreeCAD keeps the GL viewer hidden (fields would stay at near=0 for ortho).
  // Write only on change: setValue() notifies unconditionally and the FreeCAD
  // camera has a sensor, so an unconditional write would render every frame idle.
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

  const bool wantCpuTiming = frameTimingEnabled();
  double cpuClipMs = 0.0, cpuApplyMs = 0.0, cpuReplayMs = 0.0, cpuSortMs = 0.0;
  // Scene graph is the camera authority: refresh the retained camera (and
  // generation counter) from it every frame so clipping and matrices track
  // the node navigation mutates, not a stale snapshot.
  this->refreshActiveCamera();

  // Keep near/far tight so zooming/orbiting never clips geometry. The GL
  // SoRenderManager does this (VARIABLE_NEAR_PLANE), but FreeCAD's hidden GL
  // viewer never renders, so its auto-clipping never runs: do it here.
  const long clipT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
  if (this->autoClipping != SoVulkanRenderManager::NO_AUTO_CLIPPING) {
    this->setClippingPlanes();
  }
  if (wantCpuTiming) {
    cpuClipMs = (vkRenderBreadcrumbNowUs() - clipT0) * 0.001;
  }

  SoIRRenderAction & action = this->irAction;
  action.setViewportRegion(this->viewportRegion);
  {
    static bool loggedAction = false;
    if (!loggedAction && SoVulkanShared::envFlagEnabled("COIN_VULKAN_BACKEND_DEBUG")) {
      loggedAction = true;
      SoVulkanDebug::post("[DRAWLIST] manager irAction=%p overlayAction=%p scene=%p\n",
              (void *)&this->irAction, (void *)&this->overlayIrAction,
              (void *)this->scene);
    }
  }

  params.viewport = this->viewportRegion;
  // Viewport is in device pixels, so carry the ratio into the params; backends
  // scale logical line widths/point sizes by it (SoDrawStyle is logical points),
  // else lines/dots render 1/dpr too thin on fractional-scaling displays.
  params.devicePixelRatio = this->devicePixelRatio;
  params.viewMatrix.makeIdentity();
  params.projMatrix.makeIdentity();
  params.clearColor = this->backgroundColor;
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
  // Pass the camera generation counter so backends detect a camera move unambiguously.
  params.cameraVersion = this->cameraVersion;
  // Interaction LOD (viewport camera-move timer) applies to all backends: RT
  // lowers bounces, raster draws wide lines as 1px (no CPU quad expansion).
  params.interactionLod = this->interactionLod ? TRUE : FALSE;

  // SoIRRenderAction::apply() resets the frame, so camera and scene must be
  // traversed in one apply(). The graph is geometry-only (camera is a separate
  // member, matching SoRenderManager), so build [camera, scene] to let
  // SoCamera::doAction() install the projection/view before geometry is recorded
  // (else every command carries identity matrices and the view is blank/wrong).
  // Overlay/decoration are traversed separately below (node-ids churn each frame).
   SbBool irReplayed = FALSE;
   // The graph-fingerprint walk is O(N), so it is pure waste on an unchanged
   // replayed frame. It folds render-affecting node-ids but skips camera/light/
   // environment/tag nodes, so it is invariant under camera motion. The scene-
   // root sensor fires on any descendant notification, including the camera pose
   // (FreeCAD keeps the camera in the scene): if it has NOT fired the cached fp is
   // exact and the branch below skips the walk, else recompute (see else).
   uint64_t graphFp;
   const SbVec2s fpVpSize = this->viewportRegion.getViewportSizePixels();
    if (this->graphFingerprintValid && this->lastFpValid &&
        !this->sceneGraphDirty && this->scene == this->lastFpScene &&
        fpVpSize == this->lastFpViewport && this->devicePixelRatio == this->lastFpDpr) {
      graphFp = this->graphFingerprint;
    }
    else {
      // Recompute when the sensor fires. The walk folds every non-camera-coupled
      // render-affecting node-id, so a real change -- add/remove, geometry
      // rebuild, transform/material/selection write, SoSwitch::whichChild -- bumps
      // an id and forces a re-traverse; camera/headlight motion is excluded.
      // Do NOT short-circuit with a hash of the retained draw list: it is the
      // PREVIOUS frame's output, so a visibility toggle has not changed it yet and
      // skipping the walk would replay stale output forever (show/hide lost).
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
         graphFp == this->graphFingerprint &&
         (!this->sceneCameraDependent ||
          this->cameraVersion == this->sceneCameraDependentVersion)) {
      // Camera-only frame: graph, viewport and caller revision unchanged, so the
      // retained main list is exactly what a full traversal would produce -- keep
      // it (and its caches) and restamp the view below. A real edit (transform,
      // geometry rebuild, add/remove, material/selection -- SoShape::notify()
      // drops retained tessellation) changes the fingerprint and re-traverses;
      // camera motion is excluded. Relying on fingerprint equality, not the
      // sensor flag, is robust because FreeCAD keeps the camera in the scene.
      irReplayed = TRUE;
    }
    else {
      // Reuse the traversal root; only touch children when the set changes, so
      // the refcount stays balanced and navigation stops re-notifying.
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
        const long applyT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
        action.apply(root);
        if (wantCpuTiming) {
          cpuApplyMs = (vkRenderBreadcrumbNowUs() - applyT0) * 0.001;
        }
       this->mainCommandCount =
         action.getDrawList().getNumCommands();
       this->sceneCameraDependent = action.isCameraDependent();
       this->sceneCameraDependentVersion = this->cameraVersion;
       this->graphFingerprint = graphFp;
       this->graphFingerprintValid = TRUE;
       this->lastFrameViewValid = FALSE;
    }
  }
  else {
    action.beginFrame();
    this->graphFingerprintValid = FALSE;
    this->sceneCameraDependent = FALSE;
    this->rootChildrenValid = FALSE;
  }
  // On a replay no traversal ran, so the retained main geometry is bit-
  // identical to last frame: backends may skip re-hashing geometry whose
  // pointer identity matches. Fresh overlay/decoration commands are excluded.
  params.geometryContentUnchanged = irReplayed;

  SoDrawList & list = action.getMutableDrawList();

  // ---- Always re-record the overlay/decoration (cheap) and merge ----------
  // The nav cube/axis cross mirror the camera, so their node-ids churn and cannot
  // be retained: re-traverse into overlayIrAction and append onto the main list.
  // Truncate the previous overlay region first (its geometry was just reset).
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
      // Decorations (axis cross) after the overlay scene, matching GL's
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
    if (oroot->getNumChildren() > 0) {
      // apply() resets the frame first (beginFrame), so the overlay action's
      // previous draw list/geometry pool is released before re-recording.
      this->overlayIrAction.apply(oroot);
      overlayApplied = TRUE;
    }
    else {
      // No overlay this frame: clear the previous overlay list before merging.
      this->overlayIrAction.beginFrame();
    }
  }
  {
    // Main region is whatever the MAIN IR action recorded; on a fresh document
    // that is often just the hidden anchor cube (~36 vertices). Do not assume
    // "the scene rendered" because main > 0: check mainMaxVc against the shape's
    // real vertex count (tools/fcprobe/vk_geomlod_probe.py guards this).
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
    if (SoVulkanShared::envFlagEnabled("COIN_VULKAN_BACKEND_DEBUG")) {
      int mainMax = 0;
      int totalMax = 0;
      for (int i = 0; i < list.getNumCommands(); ++i) {
        const int vc = static_cast<int>(list.getCommand(i).geometry.vertexCount);
        if (i < numMain && vc > mainMax) mainMax = vc;
        if (vc > totalMax) totalMax = vc;
      }
      SoVulkanDebug::post("[DRAWLIST] main=%d total=%d replayed=%d mainMaxVc=%d totalMaxVc=%d\n",
              numMain, list.getNumCommands(), irReplayed ? 1 : 0, mainMax,
              totalMax);
    }
  }

  // The frame view/projection must come from the camera this manager uses
  // (setCamera()), not whatever camera sits in the scene graph: FreeCAD swaps
  // the node when the projection changes, and a mismatch renders with one while
  // clipping/viewport use the other. Build the matrices directly from the camera
  // node (mirroring SoCamera::doAction()); the managed camera sits at the top of
  // a fresh separator, so it matches an in-scene traversal.
  if (this->camera) {
    // Build the view volume with the manager's computed near/far, NOT the camera
    // fields: those are shared with FreeCAD's hidden GL viewer, whose render
    // manager rewrites them concurrently (racing reads give an intermittent near
    // plane behind the surface). When auto-clipping is off the fields win.
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
        // Zoom wall: render from the backed-off position; the camera node is untouched.
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

  int dbgRestamped = -1;
  if (irReplayed) {
    // Camera-only frame: restamp the frame view into every non-overlay command
    // carrying the previous traversal's view element (sub-camera commands keep
    // their own). Lighting is world-space and needs no re-derivation.
    if (this->lastFrameViewValid) {
      const long replayT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
      // SbMatrix is exactly float[4][4], so a full-storage bit-compare detects
      // any view change. On a static camera the replay view is bit-identical,
      // so the O(N) restamp loop is skipped entirely.
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
    // Remember the view bits this traversal stamped so the next replay can restamp.
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
  const long sortT0 = wantCpuTiming ? vkRenderBreadcrumbNowUs() : 0;
  if (irReplayed && this->lastSortValid &&
           list.getNumCommands() == this->lastSortCommandCount &&
           std::memcmp(&this->lastSortView[0][0], &params.viewMatrix[0][0],
                       sizeof(float) * 16) == 0) {
    // Replayed list, bit-identical view, unchanged command count: reuse the
    // previous sorted order instead of re-deriving every sort key. The count
    // check is required because the overlay region was truncated/re-appended
    // this frame; if its size changed the order holds stale indices.
  }
  else {
    list.buildSortedOrder(params.viewMatrix);
    this->lastSortView = params.viewMatrix;
    this->lastSortValid = TRUE;
    this->lastSortCommandCount = list.getNumCommands();
  }
  if (wantCpuTiming) {
    cpuSortMs = (vkRenderBreadcrumbNowUs() - sortT0) * 0.001;
    SoVulkanDebug::post("[RTDBG] cpuTiming clip=%.2f apply=%.2f restamp=%.2f "
                 "sort=%.2f cmds=%d\n",
                 cpuClipMs, cpuApplyMs, cpuReplayMs, cpuSortMs,
                 list.getNumCommands());
  }
  drawlist = &list;

  // Diagnostic: identity view/proj means no camera node (or no geometry) and a
  // blank view. Log the transition to non-identity (the first real camera
  // frame) so the camera fix can be verified at runtime.
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

  return TRUE;
}


void
SoVulkanRenderManager::resetExternalGpuQueries(VkCommandBuffer commandBuffer)
{
  this->pimpl->backend.resetExternalGpuQueries(commandBuffer);
}

uint32_t
SoVulkanRenderManager::getRenderFrameCount(void) const
{
  return this->pimpl->frameOrdinal;
}
