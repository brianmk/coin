// src/rendering/vulkan/manager/SoVulkanRenderManagerP.h

// Private implementation header for SoVulkanRenderManager: the P-impl struct
// and the file-local render helpers, shared by SoVulkanRenderManager.cpp
// (public API + device lifecycle) and SoVulkanRenderManagerRender.cpp
// (frame preparation, clipping and recording).

#pragma once

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
static void vulkanSceneGraphChangedCallback(void * data, SoSensor * sensor);

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

namespace SoVulkanManagerDetail {


// Cached environment checks for the diagnostic flags.  These sit on the
// per-frame path and the environment does not change during a process
// lifetime, so the getenv() lookup (not thread-safe) is performed at most
// once instead of every frame.  All flags use the shared
// SoVulkanShared::envFlagEnabled policy (honors "0"/"false"/"off" opt-outs).
inline bool clipDebugEnabled()
{
  static const bool enabled = SoVulkanConfig::get().debug.clipDebug;
  return enabled;
}

inline bool breadcrumbsEnabled()
{
  static const bool enabled = SoVulkanConfig::get().debug.breadcrumbs;
  return enabled;
}

// Per-phase CPU timing for the fcprobe profile harness.  Gated by the same
// FC_VULKAN_FRAME_TIMING flag as the RTX [RTDBG] frameTiming line; the manager
// emits its own [RTDBG] cpuTiming line (clip/apply/restamp/sort) so the
// existing frameTiming regex in vk_profile_probe.check.py is untouched.
inline bool frameTimingEnabled()
{
  static const bool enabled = SoVulkanConfig::get().debug.frameTiming;
  return enabled;
}

inline long vkRenderBreadcrumbNowUs()
{
  return SoVulkanShared::steadyNowUs();
}

inline bool vkRenderBreadcrumbEnabled()
{
  return SoVulkanShared::breadcrumbsEnabled();
}

inline int vkLightFrameDbgBudget = 192;
inline int vkLightFpDbgBudget = 192;

[[maybe_unused]] inline void vkRenderBreadcrumb(const char* phase)
{
  if (!vkRenderBreadcrumbEnabled()) {
    return;
  }
  std::fprintf(stderr, "[VKRENDER] %ld %s\n", vkRenderBreadcrumbNowUs(), phase);
  std::fflush(stderr);
}

inline int vkRenderBreadcrumbLogBudget = 0;

inline void vkRenderBreadcrumbSince(long startUs, long thresholdUs, const char* phase)
{
  SoVulkanShared::breadcrumbSince(vkRenderBreadcrumbLogBudget, "[VKRENDER]",
                                  startUs, thresholdUs, phase);
}

// When FC_VULKAN_CLIP_VERBOSE is set, the near/far probe below logs every
// frame instead of the sparse every-25-frame sampler, so a probe can assert
// that the auto-clipping near/far planes recompute after a scene transform
// change (the cached-bbox correctness case).
inline bool clipVerboseEnabled()
{
  static const bool enabled = SoVulkanConfig::get().debug.clipVerbose;
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
inline uint64_t computeSceneFingerprint(const SoIRRenderAction & action, int mainCount)
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
inline bool irReplayEnabled()
{
  // On by default; the shared helper honors the full 0/false/off opt-out set
  // (this site used to accept only a leading '0', unlike every other flag).
  static const bool enabled = SoVulkanConfig::get().debug.irReplay;
  return enabled;
}

inline void mixHash(uint64_t & h, uint64_t v)
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
inline void graphFingerprintWalk(SoNode * node, const SoNode * skip, uint64_t & h)
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


} // namespace SoVulkanManagerDetail

// Defined once in SoVulkanRenderManager.cpp; the P-impl attaches it as the
// scene-dirty sensor callback.
void vulkanSceneGraphChangedCallback(void * data, SoSensor * sensor);

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
  SbBool tessellationOverlay = FALSE;
  SbBool edgeOverlay = TRUE;
  //! Interaction LOD state, forwarded to the RT backend.  Persisted here so a
  //! later RT-backend bring-up (setViewSettings/invalidateViewSettings) can
  //! re-apply it.
  SbBool interactionLod = FALSE;
  SbColor4f edgeColor = SbColor4f(0.05f, 0.05f, 0.05f, 1.0f);
  //! Last settings blob applied through setViewSettings(), and whether one has
  //! been applied yet (so the first call always applies).
  SoVulkanViewSettings viewSettings;
  SbBool viewSettingsApplied = FALSE;
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
  //! Command count the retained list's sorted order was built for.  The
  //! overlay/decoration region is truncated and re-appended every frame, so a
  //! replayed frame may reuse the previous order only while the total command
  //! count is unchanged; otherwise the order still holds indices of overlay
  //! commands that no longer exist (an out-of-range getCommand()).
  int lastSortCommandCount = -1;
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
  // prepareRenderParams().  These are ALSO published onto the camera node's
  // nearDistance/farDistance fields (see setClippingPlanes), exactly like the
  // legacy GL SoRenderManagerP, because the CPU pick (SoRayPickAction) builds
  // its view volume from those camera fields: a pick whose far plane stops
  // short of the model can never hover/select it.  The private copies are kept
  // so prepareRenderParams() reads back the exact values used for the pick even
  // if a later navigation step overwrites the camera fields before the frame is
  // recorded.
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
  // Frames in flight requested by the embedding (setMaxFramesInFlight).  Stored
  // so the lazily-created RT backend gets it at initialize() time.
  uint32_t maxFramesInFlight = 2;
  // Persistent pipeline-cache path set by the embedding application before
  // initialize(); forwarded to the backend there (see setPipelineCachePath()).
  std::string pipelineCachePath;
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

  // Dump the [CLIP] diagnostic trace (env-gated by FC_VULKAN_CLIP_DEBUG;
  // FC_VULKAN_CLIP_VERBOSE adds the per-25-frame verbose lines).  Extracted
  // from prepareRenderParams() so the per-frame hot path stays readable; the
  // body is inert unless the flag is set.
  void dumpClipDebug(SoDrawList & list, const SoRenderParams & params);

  // Run `fn` against the RT backend only when it is initialized; otherwise
  // emit the standard "not initialized" warning naming `caller`.  `unavailable`
  // is the per-setter tail the caller used ("setting ignored", "path tracing
  // is unavailable", ...), preserved so existing log greps keep matching.
  // Collapses the identical guard+warning boilerplate the RT setters repeated.
  template <typename F>
  void withRtx(const char * caller, const char * unavailable, F && fn)
  {
    if (!this->rtxBackendInitialized) {
      SoDebugError::postWarning(
        caller, "ray-tracing backend is not initialized; %s", unavailable);
      return;
    }
    fn(this->rtxBackend);
  }
};

// Retained-pointer assignment: keep the new node referenced and drop the old
// one, no-op when unchanged.  The three scene setters, setCamera() and
// refreshActiveCamera() all performed this exact refcount dance by hand.
template <typename T>
static void
setRetainedNode(T *& slot, T * node)
{
  if (slot == node) return;
  if (slot) slot->unref();
  slot = node;
  if (slot) slot->ref();
}
