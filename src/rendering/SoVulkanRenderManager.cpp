// src/rendering/SoVulkanRenderManager.cpp

#include <Inventor/rendering/SoVulkanRenderManager.h>

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
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include "rendering/SoRenderBackend.h"
#include "rendering/SoClippingPlanes.h"
#include "rendering/SoRenderIRP.h"
#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoRTXRenderBackend.h"
#include "rendering/SoVulkanShared.h"

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

namespace {

// Cached environment checks for the diagnostic flags.  These sit on the
// per-frame path and the environment does not change during a process
// lifetime, so the getenv() lookup (not thread-safe) is performed at most
// once instead of every frame.  All flags use the shared
// SoVulkanShared::envFlagEnabled policy (honors "0"/"false"/"off" opt-outs).
bool clipDebugEnabled()
{
  static const bool enabled = SoVulkanShared::envFlagEnabled("FC_VULKAN_CLIP_DEBUG");
  return enabled;
}

bool breadcrumbsEnabled()
{
  static const bool enabled = SoVulkanShared::envFlagEnabled("FC_VULKAN_BREADCRUMBS");
  return enabled;
}

// Per-phase CPU timing for the fcprobe profile harness.  Gated by the same
// FC_VULKAN_FRAME_TIMING flag as the RTX [RTDBG] frameTiming line; the manager
// emits its own [RTDBG] cpuTiming line (clip/apply/restamp/sort) so the
// existing frameTiming regex in vk_profile_probe.check.py is untouched.
bool frameTimingEnabled()
{
  static const bool enabled = SoVulkanShared::envFlagEnabled("FC_VULKAN_FRAME_TIMING");
  return enabled;
}

long vkRenderBreadcrumbNowUs()
{
  return (long)std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool vkRenderBreadcrumbEnabled()
{
  static const bool enabled = SoVulkanShared::envFlagEnabled("FC_GUI_OPEN_BREADCRUMB");
  return enabled;
}

int vkLightFrameDbgBudget = 192;
int vkLightFpDbgBudget = 192;

[[maybe_unused]] void vkRenderBreadcrumb(const char* phase)
{
  if (!vkRenderBreadcrumbEnabled()) {
    return;
  }
  std::fprintf(stderr, "[VKRENDER] %ld %s\n", vkRenderBreadcrumbNowUs(), phase);
  std::fflush(stderr);
}

void vkRenderBreadcrumbSince(long startUs, long thresholdUs, const char* phase)
{
  if (!vkRenderBreadcrumbEnabled()) {
    return;
  }
  static int logged = 0;
  const long now = vkRenderBreadcrumbNowUs();
  if (logged < 30 && now - startUs >= thresholdUs) {
    ++logged;
    std::fprintf(stderr, "[VKRENDER] %ld %s dur_us=%ld\n", startUs, phase, now - startUs);
    std::fflush(stderr);
  }
}

// When FC_VULKAN_CLIP_VERBOSE is set, the near/far probe below logs every
// frame instead of the sparse every-25-frame sampler, so a probe can assert
// that the auto-clipping near/far planes recompute after a scene transform
// change (the cached-bbox correctness case).
bool clipVerboseEnabled()
{
  static const bool enabled = SoVulkanShared::envFlagEnabled("FC_VULKAN_CLIP_VERBOSE");
  return enabled;
}

// Cheap content fingerprint over the MAIN part of the IR draw list: the
// world model transform and geometry identity of the first \a mainCount
// commands.  The scene bbox (and thus the auto-clipping near/far planes)
// depends on the main scene's geometry + world transform, so a change in any
// main command's model matrix (an object or ancestor moved/rotated), its
// geometry streams, or its counts means the world extent can differ and the
// cache must refresh.  Command count alone is not a sound proxy: moving a
// body keeps the same number of draw commands but changes its world extent.
// Only the main commands are hashed: the overlay/decoration commands appended
// after index mainCommandCount are re-recorded every frame with
// camera-dependent model matrices, so hashing them would change the
// fingerprint on pure camera moves and defeat the cache.  This is far cheaper
// than re-running SoGetBoundingBoxAction over the whole scene every frame,
// and it is a sound signal -- any main-scene extent change necessarily
// implies a geometry or model-transform change in these commands.
uint64_t computeSceneFingerprint(const SoIRRenderAction & action, int mainCount)
{
  const SoDrawList & drawList = action.getDrawList();
  uint64_t h = 0x7f4a7c159e3779b9ULL;
  const int n = std::min(mainCount, drawList.getNumCommands());
  for (int i = 0; i < n; ++i) {
    const SoRenderCommand & c = drawList.getCommand(i);
    const SoGeometryDesc & g = c.geometry;
    // Mix the world transform (model matrix) so object/ancestor motion and
    // rotation (which do not change command count) still invalidate the cache.
    for (int k = 0; k < 16; ++k) {
      uint32_t bits;
      std::memcpy(&bits, &(c.modelMatrix[k >> 2][k & 3]), sizeof(bits));
      const uint64_t v = bits;
      h ^= v + 0x517cc1b727220a95ULL + (h << 6) + (h >> 2);
    }
    // Mix geometry identity (buffers are reallocated on rebuild, so pointer
    // identity tracks content) and counts.
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

// IR replay kill switch: retained-drawlist replay is on by default; set
// FC_VULKAN_IR_REPLAY=0 to force a full scene re-traversal every frame.
bool irReplayEnabled()
{
  static const bool enabled = []() {
    const char * value = std::getenv("FC_VULKAN_IR_REPLAY");
    return !(value && value[0] == '0');
  }();
  return enabled;
}

void mixHash(uint64_t & h, uint64_t v)
{
  h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
}

// Recursively fold (node pointer, SoNode::getNodeId()) of every reachable
// node into \a h.  Any change that can alter the IR draw list -- a field
// write, a child-list edit, a geometry rebuild -- notifies through the node,
// and SoNode::notify() bumps its unique id, so matching ids mean every
// retained command was produced from exactly the current graph.  Group
// children are folded via SoGroup; non-group child containers would have to
// route through SoChildList notifications, which bump the owning node's id
// and are caught by its own entry.
//
// NODE-ID EXCLUSIONS (camera-coupled infra): Coin propagates a notification
// up the parent chain, so any changed node re-bumps every ancestor's node-id.
// Two classes of node must be excluded or the fingerprint changes on every
// camera-only frame and defeats the retained-IR replay:
//   * SoCamera                                     -- its pose is the very
//      change replay exists for (restamped/re-lit after a frame-view change).
//   * The headlight envelope (SoRotation / SoTransformSeparator / SoLight /
//      SoEnvironment) plus bare SoGroup/SoSeparator aggregation containers.
//      FreeCAD re-aims the headlight ROTATION to follow the camera every
//      navigation frame, and the container's node-ids are re-bumped purely by
//      propagation.  None of these nodes produce the rasterized fill-geometry
//      in the draw list -- lighting is re-derived every frame by the backend's
//      updateLightingSetup() -- so excluding their ids only suppresses the
//      camera-coupled chatter.  Real geometry edits use SoTransform/SoMatrix
//      /shape/selection nodes, which still fold their ids, so an in-place
//      edit, a move, an add/remove or a material/texture swap still
//      invalidates the draw list and forces a re-record.
void graphFingerprintWalk(SoNode * node, const SoNode * skip, uint64_t & h)
{
  if (!node || node == skip) return;
  mixHash(h, reinterpret_cast<uintptr_t>(node));
  const bool skipId =
    node->isOfType(SoCamera::getClassTypeId()) ||
    node->isOfType(SoLight::getClassTypeId()) ||
    node->isOfType(SoEnvironment::getClassTypeId()) ||
    node->isOfType(SoRotation::getClassTypeId()) ||
    node->isOfType(SoTransformSeparator::getClassTypeId()) ||
    node->getTypeId() == SoGroup::getClassTypeId() ||
    node->getTypeId() == SoSeparator::getClassTypeId();
  if (!skipId) {
    mixHash(h, static_cast<uint64_t>(node->getNodeId()));
  }
  if (node->isOfType(SoGroup::getClassTypeId())) {
    const SoGroup * group = static_cast<const SoGroup *>(node);
    const int num = group->getNumChildren();
    mixHash(h, static_cast<uint64_t>(num));
    for (int i = 0; i < num; ++i) {
      graphFingerprintWalk(group->getChild(i), skip, h);
    }
  }
}

} // namespace

class SoVulkanRenderManagerP {
public:
  SoVulkanRenderManagerP()
    : irAction(SbViewportRegion()),
      overlayIrAction(SbViewportRegion())
  {
    this->viewportRegion.setWindowSize(1, 1);
    // Persist one traversal root so prepareRenderParams() does not heap-allocate
    // + ref/unref a new separator on every frame.  Children are cleared and
    // re-added each frame; only the root node itself is retained.
    this->frameRoot = new SoSeparator;
    this->frameRoot->ref();
    // A separate root for the always-re-recorded overlay/decoration scenes
    // (nav cube, axis cross), kept apart from the replayed main scene.
    this->overlayRoot = new SoSeparator;
    this->overlayRoot->ref();
    // Dirty-tracking sensor for the graph-fingerprint fast-path: the sensor is
    // attached to the main scene and fires whenever any descendant is notified,
    // so computeGraphFingerprint() can skip the O(N) scene walk on frames
    // where the scene has not changed (static / camera-only frames).
    this->sceneGraphSensor =
      new SoNodeSensor(vulkanSceneGraphChangedCallback, this);
  }

  ~SoVulkanRenderManagerP()
  {
    // Detach/destroy the scene-dirty sensor FIRST: it is attached to the main
    // scene node, which is unref'd below and may be destroyed here.
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
  // Persistent traversal root (see the constructor comment).
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
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);
  SbBool clearWindow = TRUE;
  SbBool clearDepth = TRUE;
  void * renderTarget = nullptr;
  //! Device-pixel ratio of the Vulkan surface.  The swapchain/viewport region
  //! is in device pixels, so renderer widths/sizes (logical SoDrawStyle
  //! points) must be scaled by this; kept in the render params for the
  //! backends and also exposed to the SoDevicePixelRatio element.
  float devicePixelRatio = 1.0f;

  SoVulkanRenderManager::AutoClippingStrategy autoClipping =
    SoVulkanRenderManager::NO_AUTO_CLIPPING;
  float nearplanevalue = 0.6f;

  //! Generation counter bumped every time the active camera's identity
  //! changes (a different node, position, orientation or projection).  The
  //! ray-tracing backend reads this instead of diffing floating-point view
  //! matrices, which are fragile (a real camera move can produce variations
  //! swallowed by the equality epsilon, and single-precision translation can
  //! alias under a hash).  A monotonically increasing integer is unambiguous.
  uint32_t cameraVersion = 0;

  //! Fingerprint of the camera pose (position + forward direction) used to
  //! detect in-place pose changes of the same camera node, because a pointer
  //! comparison cannot see a rotation/pan/zoom that mutates the node.
  uint32_t cameraPoseFingerprint = 0;

  //! 1-based ordinal of the last presented frame.  Bumped exactly once per
  //! render()/renderExternal() call and copied into SoRenderParams::frame,
  //! so backends, frame dumps and probe phase markers can correlate on one
  //! monotonic key independent of stream ordering.
  uint32_t frameOrdinal = 0;

  //! --- Retained-IR replay state (camera-only frame fast path) ----------
  //! Fold of the render-affecting graph (see graphFingerprintWalk) from the
  //! last full traversal; a match means the retained IR draw list (and the
  //! geometry/texture caches keyed on it) is still exactly reproducible.
  uint64_t graphFingerprint = 0;
  SbBool graphFingerprintValid = FALSE;
  //! Inputs the last graph-fingerprint walk was computed from, plus an
  //! SoNodeSensor dirty flag: the sensor fires whenever the main scene graph
  //! is notified, so the O(N) scene walk can be skipped on frames where the
  //! scene has not changed (camera-only / static frames).  See
  //! prepareRenderParams.
  SbBool lastFpValid = FALSE;
  SoNode * lastFpScene = nullptr;
  SbVec2s lastFpViewport = SbVec2s(0, 0);
  float lastFpDpr = 1.0f;
  SbBool sceneGraphDirty = TRUE;
  SoNodeSensor * sceneGraphSensor = nullptr;
  //! Caller-published revision of state the graph walk cannot see (the
  //! selection model behind FreeCAD's highlight roots); mixed in verbatim.
  uint64_t externalRevision = 0;
  //! Viewing matrix (SoViewingMatrixElement bits) stamped into the commands
  //! of the last full traversal; the replay restamp key.
  SbMatrix lastFrameView;
  SbBool lastFrameViewValid = FALSE;
  //! Viewing matrix the retained list's painter's-algorithm order was last
  //! built for (SoDrawList::buildSortedOrder); a bit-match with a replayed
  //! list means the previous frame's sorted order is still exact.
  SbMatrix lastSortView;
  SbBool lastSortValid = FALSE;
  //! Child pointers last installed in frameRoot, so navigation frames stop
  //! churning the separator's child list (and its notifications).
  SoNode * rootChildren[4] = {nullptr, nullptr, nullptr, nullptr};
  SbBool rootChildrenValid = FALSE;

  //! Current render-affecting graph fingerprint (see graphFingerprintWalk).
  uint64_t computeGraphFingerprint() const;

  //! Resolve the camera that will render this frame.  The scene graph is the
  //! single camera authority (FreeCAD's navigation mutates the camera node
  //! inside the scene it passes to setSceneGraph), so the camera is found
  //! there first.  The retained pointer set by setCamera() is only a
  //! fallback/hint for the case where the camera lives outside the scene
  //! root (overlay-only setups).  Returns nullptr if no camera is available.
  SoCamera * resolveActiveCamera();

  //! Refresh the retained camera pointer from the scene-graph authority and
  //! bump cameraVersion when the active camera (or its pose) changes.
  void refreshActiveCamera();


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

  // Cached world-space scene bounding box for setClippingPlanes().  The box in
  // camera coordinates still depends on the camera pose, which changes every
  // frame, so only the (static) world-space box is cached: each frame re-applies
  // the cheap camera transform instead of running a full scene bbox traversal.
  // The cache is invalidated when the scene pointer changes (setSceneGraph) or
  // when the previous frame's IR command count differs (a cheap structural-
  // change proxy for geometry edits).
  SbXfBox3f sceneWorldBBox;
  SoNode * sceneBBoxScene = nullptr;
  uint64_t sceneBBoxFingerprint = 0;
  bool sceneBBoxCached = false;
  //! Cached scene fingerprint (see cachedSceneFingerprint): on camera-only
  //! frames the main draw list is retained verbatim and the scene sensor has
  //! not fired, so the O(main-commands) hash is invariant and the walk is
  //! skipped.  Revalidated against the scene pointer and the retained main
  //! command count, both of which any content change must disturb.
  uint64_t sceneFpCached = 0;
  SoNode * sceneFpScene = nullptr;
  uint32_t sceneFpMainCount = 0;
  SbBool sceneFpValid = FALSE;

  // Cheap draw-list fingerprint gate for the expensive graph-fingerprint walk.
  // computeSceneFingerprint() hashes only the retained main commands (world
  // matrices + geometry buffer pointers + counts), so it changes on a real
  // geometry/transform edit but is invariant under camera motion.  When it
  // matches the previous frame's value, computeGraphFingerprint() (a full scene
  // tree walk) is skipped and the cached graph fingerprint is reused.
  uint64_t drawFpCached = 0;
  SoNode * drawFpScene = nullptr;
  uint32_t drawFpMainCount = 0;
  SbBool drawFpValid = FALSE;

  SoIRRenderAction irAction;
  //! Second IR action used to re-record the overlay/decoration scenes every
  //! frame (their node-ids churn with the camera, so they cannot be retained);
  //! its commands are appended onto the replayed main list in prepareRenderParams.
  SoIRRenderAction overlayIrAction;
  //! Number of main (non-overlay) commands retained in irAction's draw list,
  //! used to separate the replayed main region from the fresh overlay region.
  uint32_t mainCommandCount = 0;
  SoVulkanRenderBackend backend;
  SoRTXRenderBackend rtxBackend;
  SbBool backendInitialized = FALSE;
  SbBool rtxBackendInitialized = FALSE;
  SbBool rayTracing = FALSE;
  // Device context borrowed at initialize(); retained (not owned) so the RT
  // backend can be brought up lazily by ensureRayTracing() after a startup
  // that skipped it.  Cleared in shutdown().
  SoVulkanDeviceContext * initContext = nullptr;

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

// Mark the graph fingerprint dirty when any part of the main scene is notified
// (a field write or child-list edit anywhere in the subtree) -- the exact
// condition the O(N) fingerprint walk detects, so it can be skipped until the
// scene actually changes.
static void
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
SoVulkanRenderManager::setDecorationSceneGraph(SoNode * root)
{
  SoNode *& stored = this->pimpl->decorationScene;
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
SoVulkanRenderManager::setEdgeColor(const SbColor4f & color)
{
  this->pimpl->edgeColor = color;
  this->pimpl->backend.setEdgeColor(color);
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
  if (!this->pimpl->backend.initialize(params)) {
    SoDebugError::postWarning("SoVulkanRenderManager::initialize",
                              "backend initialization failed");
    return FALSE;
  }
  this->pimpl->backendInitialized = TRUE;
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
    this->pimpl->rtxBackendInitialized = TRUE;
    return TRUE;
  }
  return FALSE;
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

void
SoVulkanRenderManager::setViewMode(int mode)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning("SoVulkanRenderManager::setViewMode",
                              "ray-tracing backend is not initialized; "
                              "view mode is unavailable");
    return;
  }
  this->pimpl->rtxBackend.setViewMode(
    static_cast<SoRTXRenderBackend::RtxViewMode>(mode));
}

int
SoVulkanRenderManager::getViewMode(void) const
{
  if (!this->pimpl->rtxBackendInitialized) return 0;
  return static_cast<int>(this->pimpl->rtxBackend.getViewMode());
}

void
SoVulkanRenderManager::setEnvMap(const int index)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning("SoVulkanRenderManager::setEnvMap",
                              "ray-tracing backend is not initialized; "
                              "environment is unavailable");
    return;
  }
  this->pimpl->rtxBackend.setEnvMap(index);
}

void
SoVulkanRenderManager::setSceneLights(const std::vector<SoLightData> & lights,
                                      const SbVec3f & ambient)
{
  if (!this->pimpl->rtxBackendInitialized) {
    return;
  }
  this->pimpl->rtxBackend.setSceneLights(lights, ambient);
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
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning("SoVulkanRenderManager::setPathTracingBounces",
                              "ray-tracing backend is not initialized; "
                              "setting ignored");
    return;
  }
  this->pimpl->rtxBackend.setPathTracingBounces(bounces);
}

void
SoVulkanRenderManager::setPathTracingSettleFrames(const uint32_t frames)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning(
      "SoVulkanRenderManager::setPathTracingSettleFrames",
      "ray-tracing backend is not initialized; setting ignored");
    return;
  }
  this->pimpl->rtxBackend.setPathTracingSettleFrames(frames);
}

void
SoVulkanRenderManager::setPathTracingMaxSamples(const uint32_t samples)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning(
      "SoVulkanRenderManager::setPathTracingMaxSamples",
      "ray-tracing backend is not initialized; setting ignored");
    return;
  }
  this->pimpl->rtxBackend.setPathTracingMaxSamples(samples);
}

void
SoVulkanRenderManager::setPathTracingDenoiseEnabled(SbBool enabled)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning(
      "SoVulkanRenderManager::setPathTracingDenoiseEnabled",
      "ray-tracing backend is not initialized; setting ignored");
    return;
  }
  this->pimpl->rtxBackend.setPathTracingDenoiseEnabled(enabled);
}

void
SoVulkanRenderManager::setPathTracingDenoiser(const char * denoiser)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning(
      "SoVulkanRenderManager::setPathTracingDenoiser",
      "ray-tracing backend is not initialized; setting ignored");
    return;
  }
  this->pimpl->rtxBackend.setDenoiserFilter(denoiser);
}

void
SoVulkanRenderManager::setPathTracingDenoiserScale(const float scale)
{
  if (!this->pimpl->rtxBackendInitialized) {
    SoDebugError::postWarning(
      "SoVulkanRenderManager::setPathTracingDenoiserScale",
      "ray-tracing backend is not initialized; setting ignored");
    return;
  }
  this->pimpl->rtxBackend.setDenoiserScale(scale);
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
                                      VkRenderPass renderPass,
                                      VkFramebuffer framebuffer)
{
  const long renderBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
  SoRenderParams params;
  SoDrawList * drawlist = nullptr;
  if (!this->pimpl->prepareRenderParams(clearwindow, clearzbuffer, drawlist,
                                        params)) {
    return FALSE;
  }
  if (renderBcStart) {
    vkRenderBreadcrumbSince(renderBcStart, 5000, "renderExternal prepareRenderParams end");
  }
  params.frame = ++this->pimpl->frameOrdinal;
  const long backendBcStart = vkRenderBreadcrumbEnabled() ? vkRenderBreadcrumbNowUs() : 0;
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

uint64_t
SoVulkanRenderManagerP::computeGraphFingerprint() const
{
  uint64_t h = 0xcbf29ce484222325ULL;
  mixHash(h, reinterpret_cast<uintptr_t>(this->camera));
  mixHash(h, reinterpret_cast<uintptr_t>(this->scene));
  mixHash(h, reinterpret_cast<uintptr_t>(this->overlayScene));
  mixHash(h, reinterpret_cast<uintptr_t>(this->decorationScene));
  if (std::getenv("FC_VULKAN_LIGHTREPLAY_DBG")) {
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
    SoCamera *& stored = this->camera;
    if (stored) {
      stored->unref();
    }
    stored = resolved;
    stored->ref();
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

  // If the whole scene is behind the camera, keep the current near/far planes
  // (they were computed on the previous frame when the scene was in front).
  // Collapsing them to a tiny range here is what makes the view appear
  // "locked": the scene only becomes visible again once it rotates within the
  // collapsed volume.  The shared core already returns early for perspective
  // cameras; an orthographic camera must handle it here instead.
  if (farval <= 0.0f) {
    return;
  }

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

  const float SLACK = kSoClippingSlack;
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
   // Cheap fast-path: the graph fingerprint walk is O(N) over the whole scene,
   // and on a retained (replayed) frame with no scene change it is pure waste.
   // The walk folds scene node-ids but deliberately SKIPS the camera, so the
   // fingerprint is invariant under camera motion; the only thing that changes
   // it is a change to the main scene graph.  An SoNodeSensor attached to the
   // scene root fires whenever any descendant is notified (a field write or a
   // child-list edit), which is precisely a main-scene change -- so when the
   // sensor has NOT fired since the last walk, the cached fingerprint is still
   // exact and the O(N) walk is skipped.  This catches every change the walk
   // would (Coin propagates notify() up to the root), independent of any
   // external revision wiring, and camera-only frames still produce the same
   // (camera-invariant) fingerprint.
   //
   // The sensor firing is itself the replay gate, not the fingerprint: the
   // fingerprint folds node identity only (pointer + node-id + child count),
   // so a FIELD-ONLY write (material, transform, preselection state) leaves it
   // unchanged.  Replaying after such a write is unsafe anyway: the write
   // notifies the shape (SoShape::notify() drops its retained tessellation),
   // so the retained list's raw geometry pointers may reference freed storage.
   // Re-traverse whenever the sensor has fired since the last walk; camera
   // motion alone never fires the scene-root sensor (the camera lives outside
   // the scene graph), so camera-only frames still replay.
   const SbBool graphChanged = this->sceneGraphDirty;
   uint64_t graphFp;
   const SbVec2s fpVpSize = this->viewportRegion.getViewportSizePixels();
    if (this->graphFingerprintValid && this->lastFpValid &&
        !this->sceneGraphDirty && this->scene == this->lastFpScene &&
        fpVpSize == this->lastFpViewport && this->devicePixelRatio == this->lastFpDpr) {
      graphFp = this->graphFingerprint;
    }
    else {
      // Recomputing computeGraphFingerprint() walks the ENTIRE scene-graph tree
      // (graphFingerprintWalk, O(nodes) + geometry identity) and is measurable
      // on a many-object scene (1000 boxes -> ~28 ms).  It only needs to run
      // when the retained main draw list actually changed.  graphSceneDirty is
      // set by a scene-root sensor that fires on ANY descendant notification,
      // including the camera pose (FreeCAD keeps the camera inside the scene
      // graph), so camera-orbit frames set it too -- and recomputing the full
      // graph fingerprint there is pure waste: camera motion never changes the
      // retained main-list content.  Gate the expensive walk on the cheap
      // draw-list fingerprint (computeSceneFingerprint hashes only the retained
      // commands: world matrix + geometry pointers + counts).  Camera-only
      // frames produce the same draw-list fingerprint, so we keep the previous
      // graph fingerprint (which is likewise camera-invariant) and the retained
      // list replays instead of re-traversing.
      const uint64_t drawFp = computeSceneFingerprint(
        this->irAction, static_cast<int>(this->mainCommandCount));
      if (this->drawFpValid && this->drawFpCached == drawFp &&
          this->scene == this->drawFpScene &&
          this->mainCommandCount == this->drawFpMainCount) {
        graphFp = this->graphFingerprint;  // draw list unchanged -> reuse
      }
      else {
        graphFp = this->computeGraphFingerprint();
        this->drawFpCached = drawFp;
        this->drawFpScene = this->scene;
        this->drawFpMainCount = this->mainCommandCount;
        this->drawFpValid = TRUE;
      }
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
  if (std::getenv("FC_VULKAN_LIGHTREPLAY_DBG") && vkLightFrameDbgBudget-- > 0) {
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
           std::memcmp(&this->lastSortView[0][0], &params.viewMatrix[0][0],
                       sizeof(float) * 16) == 0) {
    // A retained (replayed) list with a bit-identical view sorts exactly as
    // the previous frame: reuse the previous frame's sorted order instead of
    // re-deriving every command's sort key and re-running the stable sort.
  }
  else {
    list.buildSortedOrder(params.viewMatrix);
    this->lastSortView = params.viewMatrix;
    this->lastSortValid = TRUE;
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
  if (clipDebugEnabled()) {
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

uint32_t
SoVulkanRenderManager::getRenderFrameCount(void) const
{
  return this->pimpl->frameOrdinal;
}
