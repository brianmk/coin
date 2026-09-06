// include/Inventor/rendering/SoVulkanRenderManager.h

#ifndef COIN_SOVULKANRENDERMANAGER_H
#define COIN_SOVULKANRENDERMANAGER_H

/*!
  The whole class is compiled only when COIN_BUILD_VULKAN_RENDERER is set (by
  Coin's own build and by applications that opt in).  Without it the header
  expands to nothing, so an installed Coin built without the Vulkan renderer
  does not force a Vulkan SDK dependency on its consumers.
*/

#ifndef COIN_BUILD_VULKAN_RENDERER
#define COIN_BUILD_VULKAN_RENDERER 0
#endif

#if COIN_BUILD_VULKAN_RENDERER

#include <Inventor/SbColor4f.h>
#include <Inventor/SbVec2s.h>
#include <Inventor/SbVec3f.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <vector>

// Pull in Vulkan handle types for renderExternal().  This header is only
// fully compiled when COIN_BUILD_VULKAN_RENDERER is enabled.
#include <vulkan/vulkan.h>

class SbViewportRegion;
class SoCamera;
class SoNode;
class SoIRRenderAction;
class SoVulkanRenderBackend;
class SoRTXRenderBackend;

struct SoVulkanDeviceContext;

/*!
  \class SoVulkanRenderManager SoVulkanRenderManager.h
  \brief Qt-free scene-to-Vulkan orchestrator for the render-backend path.

  This class is the Vulkan counterpart of the legacy SoRenderManager.  It
  traverses a scene graph with SoIRRenderAction to produce a backend-neutral
  SoDrawList, then submits it to a SoVulkanRenderBackend bound to a
  SoVulkanRenderTarget supplied by the caller.

  Unlike SoRenderManager, it does not own a window system surface, a camera
  sensor, stereo handling, or superimpositions.  The caller owns the Vulkan
  device and the render target and drives render() once per frame.
*/
class COIN_DLL_API SoVulkanRenderManager {
public:
  SoVulkanRenderManager();
  ~SoVulkanRenderManager();

  void setSceneGraph(SoNode * root);
  SoNode * getSceneGraph(void) const;

  /*!
    \brief Set an optional screen-space overlay scene graph.

    The overlay scene is traversed after the main scene every frame (its
    commands are recorded into the same draw list) and drawn last in the
    overlay render pass, each command using its own view/projection matrices
    and viewport/scissor region.  Used for the navigation cube: the overlay
    node renders itself into a viewport corner without affecting the main
    scene's bounding box or camera.
  */
  void setOverlaySceneGraph(SoNode * root);
  SoNode * getOverlaySceneGraph(void) const;

  /*!
    \brief Set an optional decoration scene graph (axis cross overlay).

    Traversed after the overlay scene graph every frame (commands recorded
    into the same draw list, drawn in the overlay pass after the overlay
    scene's commands).  Like the overlay scene, its nodes carry their own
    view/projection matrices and viewport/scissor regions (screen-space
    decorations such as the axis cross).
  */
  void setDecorationSceneGraph(SoNode * root);
  SoNode * getDecorationSceneGraph(void) const;

  void setCamera(SoCamera * camera);
  SoCamera * getCamera(void) const;

  void setViewportRegion(const SbViewportRegion & region);
  const SbViewportRegion & getViewportRegion(void) const;

  /*!
    \brief Strategy for automatically adjusting the camera clipping planes.

    Mirrors SoRenderManager::AutoClippingStrategy.  With anything other than
    NO_AUTO_CLIPPING, render()/renderExternal() re-compute the camera's
    nearDistance/farDistance every frame from the scene bounding box so that
    navigation (zoom, orbit) never pushes geometry outside the view volume.
    Defaults to NO_AUTO_CLIPPING to match SoRenderManager; embedding
    applications that used the legacy GL auto-clipping should enable
    VARIABLE_NEAR_PLANE.
  */
  enum AutoClippingStrategy {
    NO_AUTO_CLIPPING,
    FIXED_NEAR_PLANE,
    VARIABLE_NEAR_PLANE
  };
  void setAutoClipping(AutoClippingStrategy strategy);
  AutoClippingStrategy getAutoClipping(void) const;

  //! Fraction of the depth range kept for the near plane (see SoRenderManager).
  void setNearPlaneValue(float value);
  float getNearPlaneValue(void) const;

  void setBackgroundColor(const SbColor4f & color);
  const SbColor4f & getBackgroundColor(void) const;

  //! Device-pixel ratio of the Vulkan surface.  The swapchain is in device
  //! pixels, so the renderer scales logical line widths / point sizes by
  //! this (see SoRenderParams::devicePixelRatio).
  void setDevicePixelRatio(float ratio);
  float getDevicePixelRatio(void) const;

  /*!
    \brief Publish a revision counter for state the scene graph cannot show.

    The camera-only-frame replay compares a fingerprint of the traversal-
    relevant scene graph against the last full traversal.  Changes made
    purely through application side models (FreeCAD's selection/preselection
    state rendered by SoFCSelectionRoot without touching node fields) do not
    move that fingerprint, so the embedding widget must bump \a revision
    whenever such state changes; any change forces a full re-traversal.
  */
  void setExternalRevision(uint64_t revision);

  /*!
    \brief Configure a vertical screen-space background gradient.

    When \a enabled is TRUE, render()/renderExternal() fills the viewport
    with a top-to-bottom gradient between \a topColor and \a bottomColor
    before drawing geometry (instead of the flat clear color).
  */
  void setBackgroundGradient(SbBool enabled,
                             const SbColor4f & topColor,
                             const SbColor4f & bottomColor);

  /*!
    \brief Configure Vulkan-only display overlays.

    These toggle the wireframe/point edge overlays and their color.  They are
    deliberately not part of the shared retained render state, so the OpenGL
    backend never consults them.
  */
  void setWireframeOverlay(SbBool enabled);
  void setPointsOverlay(SbBool enabled);
  void setEdgeColor(const SbColor4f & color);
  SbBool getWireframeOverlay(void) const;
  SbBool getPointsOverlay(void) const;
  const SbColor4f & getEdgeColor(void) const;

  void setClearEnabled(SbBool clearwindow, SbBool clearzbuffer);
  void getClearEnabled(SbBool & clearwindow, SbBool & clearzbuffer) const;

  /*!
    \brief Initialize the owned Vulkan backend from a device context.

    The context is borrowed; the caller must keep it alive until shutdown()
    or destruction.  Returns FALSE if the backend cannot initialize.
  */
  SbBool initialize(SoVulkanDeviceContext * context);

  /*!
    \brief Declare how many recorded frames the caller may keep in flight.

    Forwards to the raster backend (see SoVulkanRenderBackend::
    setMaxFramesInFlight()); drives deferred-resource destruction and the
    lighting UBO ring size.  Call once after initialize() and before the
    first render when the caller submits frames concurrently.
  */
  void setMaxFramesInFlight(uint32_t count);

  /*!
    \brief Shut down the owned backend while the Vulkan device/queue are
    still valid.

    Idempotent.  Call this before the window system tears down the Vulkan
    device; the manager destructor also attempts a shutdown, but that may be
    too late by then.
  */
  void shutdown(void);

  //! Render target used by the next render(); borrowed, not owned.
  void setRenderTarget(void * target);
  void * getRenderTarget(void) const;

  /*!
    \brief Traverse the scene and submit it to the Vulkan backend.

    The camera view/projection matrices are taken from the traversed scene
    state (normally produced by the camera node).  When the scene provides no
    geometry, the draw list is empty and only the clear state is applied.
  */
  SbBool render(SbBool clearwindow = TRUE, SbBool clearzbuffer = TRUE);

  /*!
    \brief Record the scene into a caller-owned command buffer/render pass.

    Mirrors render() but does not begin/end a command buffer, begin/end a
    render pass, create a framebuffer, or submit to the queue.  The caller
    must already be inside a render pass on \a commandBuffer with \a renderPass
    and a compatible framebuffer, and owns submission/presentation.
  */
  SbBool renderExternal(SbBool clearwindow,
                        SbBool clearzbuffer,
                        VkCommandBuffer commandBuffer,
                        VkRenderPass renderPass,
                        VkFramebuffer framebuffer);

  /*!
    \brief Select the ray-tracing backend for the next render() calls.

    Ray tracing requires a Vulkan 1.2+ device with VK_KHR_acceleration_structure
    and VK_KHR_ray_tracing_pipeline enabled (the embedding application is
    responsible for enabling them on the QVulkanWindow device).  When the RT
    backend cannot initialize, this method logs a warning and keeps the raster
    backend; getRayTracingActive() reports the effective state.  Must be called
    BEFORE initialize(): the setting is only honored by initialize() (which
    brings the RT backend up or falls back to raster), and subsequent calls
    merely toggle path tracing on an already-active RT backend.
  */
  void setRayTracing(SbBool enabled);

  //! True when the ray-tracing backend is active (initialized and enabled).
  SbBool getRayTracingActive(void) const;

  /*!
    \brief Lazily bring up the ray-tracing backend after initialization.

    initialize() only builds the RT backend when setRayTracing(TRUE) preceded
    it.  When it was NOT built (path tracing was off at startup, so nothing
    paid for the RT stack on the raster-only path), this method can still bring
    it up later from the device context saved by initialize(), so enabling path
    tracing at runtime needs no window re-initialization.  Returns TRUE when
    the RT backend is ready after the call (already initialized, or brought up
    and the device supports it); FALSE when the device lacks ray tracing or no
    context is available.
  */
  SbBool ensureRayTracing(void);

  /*!
    \brief Request (or cancel) path tracing, bringing the RT backend up lazily.

    This is the single cohesive entry for a runtime raster <-> path-tracing
    toggle.  It sets the ray-tracing request flag, lazily initializes the RTX
    backend when it was skipped at startup (so a toggle needs no window
    re-initialization), and enables/disables path tracing -- all in one call,
    instead of requiring the caller to orchestrate setRayTracing() up front and
    then ensureRayTracing() and setPathTracingEnabled().  When \a enabled is
    TRUE and the device cannot bring up ray tracing, the request is cleared and
    the raster backend is used; returns FALSE.  Has no effect before
    initialize(): the request is honored there (mirrors setRayTracing()).
    Returns TRUE when ray tracing is active after the call.
  */
  SbBool requestRayTracing(SbBool enabled);

  /*!
    \brief Enable/disable path tracing on the ray-tracing backend.

    Path tracing runs a multi-bounce path tracer with shadow rays,
    progressive per-pixel accumulation and an edge-stopping denoise pass.
    A no-op when the ray-tracing backend is not active.
  */
  void setPathTracingEnabled(SbBool enabled);

  //! True when path tracing is enabled.
  SbBool getPathTracingEnabled(void) const;

  /*!
    \brief Select the ray-traced view mode.

    \a mode is one of the RtxViewMode values in SoRTXRenderBackend:
      0 = off/raster (interactive), 1 = single-sample ambient-occlusion
    preview, 2 = accumulating path tracer.  Only meaningful while the
    ray-tracing backend is active (see setPathTracingEnabled()).
  */
  void setViewMode(int mode);
  //! Current ray-traced view mode (RtxViewMode value, 0=off.
  int getViewMode(void) const;
  /*!
    \brief Select the "cubemap" environment preset for the environment-lit
    view (and the path-tracer background).

    \a index indexes the backend's procedural environment presets; -1 restores
    the viewport background gradient.  Only meaningful while the ray-tracing
    backend is active.  See also getEnvMapCount().
  */
  void setEnvMap(int index);
  //! Current environment/cubemap preset index (-1 = viewport background).
  int getEnvMap(void) const;
  //! Number of available environment/cubemap presets.
  static int getEnvMapCount(void);
  /*!
    \brief Provide the authoritative scene lighting (GL host -> RT backend).

    \a lights is the effective eye-space light set (the viewer headlight plus
    any document SoLight nodes) and \a ambient the intensity-scaled scene
    ambient.  Fired through to the RT backend so the path tracer uses the
    host's lights instead of the IR draw-list lighting capture, which can
    drop to zero lights on the retained/replayed frame and render surfaces
    at ambient-only (near-black).  Passing an empty \a lights restores the
    per-command IR lighting.
  */
  void setSceneLights(const std::vector<SoLightData> & lights,
                      const SbVec3f & ambient);
  //! Human-readable name of an environment preset index.
  static const char * getEnvMapName(int index);
  /*!
    \brief Start flag for progressive path-tracing refinement.

    TRUE starts (or restarts) the progressive accumulation; any camera move
    or scene change drops back to a single-sample preview until the flag is
    raised again.  FALSE cancels accumulation.
  */
  void setPathTracingStart(SbBool start);

  //! True while the ray-tracing backend is accumulating a progressive run.
  SbBool getPathTracingActive(void) const;

  //! True while path tracing still needs continuous frames (accumulating or
  //! in the post-move settle window).  The embedding viewport drives its
  //! continuous update from this.
  SbBool getPathTracingRefining(void) const;

  //! Samples accumulated in the current progressive run (0 when idle).
  uint32_t getPathTracingSampleCount(void) const;

  //! Maximum path-tracing bounces (1..16); forwarded to the RT backend.
  void setPathTracingBounces(uint32_t bounces);

  //! Frames of a static camera before the accumulation auto-restarts
  //! (1..120); forwarded to the RT backend.
  void setPathTracingSettleFrames(uint32_t frames);

  //! Accumulated-sample cap before the run auto-stops (1..4096); forwarded
  //! to the RT backend.
  void setPathTracingMaxSamples(uint32_t samples);

  //! Enable/disable the edge-stopping denoise pass; forwarded to the RT
  //! backend.
  void setPathTracingDenoiseEnabled(SbBool enabled);

  //! Select the denoiser backend by name ("rtx", "oidn", "fsr", "none");
  //! forwarded to the RT backend and applied on the next buffer create.
  void setPathTracingDenoiser(const char * denoiser);

  //! Denoiser upscale factor (>= 1); forwarded to the RT backend and applied
  //! on the next buffer create.
  void setPathTracingDenoiserScale(float scale);

  SoVulkanRenderBackend * getBackend(void) const;

  //! The RT backend, or NULL when ray tracing is unavailable.
  SoRTXRenderBackend * getRayTracingBackend(void) const;

  //! Ordinal of the last presented frame (1-based; 0 before the first render).
  //! Bumped exactly once per render()/renderExternal() and copied into that
  //! frame's SoRenderParams::frame.  A stable correlation key shared by
  //! backend debug traces, captured frame dumps and probe phase markers.
  uint32_t getRenderFrameCount(void) const;

private:
  class SoVulkanRenderManagerP * pimpl;
};

#endif // COIN_BUILD_VULKAN_RENDERER

#endif // COIN_SOVULKANRENDERMANAGER_H
