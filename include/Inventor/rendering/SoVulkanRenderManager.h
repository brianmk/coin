// include/Inventor/rendering/SoVulkanRenderManager.h

#ifndef COIN_SOVULKANRENDERMANAGER_H
#define COIN_SOVULKANRENDERMANAGER_H

#include <Inventor/C/basic.h>

// Compiled only with the capability the installed Coin exports
// (COIN_HAVE_VULKAN_RENDERER); otherwise expands to nothing.

/* Honour the installed capability; fall back to off for a pre-existing
   basic.h that predates it. */
#ifndef COIN_HAVE_VULKAN_RENDERER
#define COIN_HAVE_VULKAN_RENDERER 0
#endif

// Public settings blob; always available (no Vulkan dependency).
#include <Inventor/rendering/SoVulkanViewSettings.h>

#if COIN_HAVE_VULKAN_RENDERER

#include <Inventor/SbColor4f.h>
#include <Inventor/SbVec2s.h>
#include <Inventor/SbVec3f.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <string>
#include <vector>

// Vulkan handle types for renderExternal(); only compiled with the renderer.
// On Windows <vulkan/vulkan.h> pulls in <windows.h>, whose min/max macros break
// std::min/std::max; suppress them for this translation unit only.
#if defined(_WIN32) && !defined(NOMINMAX)
#  define NOMINMAX
#endif
#include <vulkan/vulkan.h>

class SbViewportRegion;
class SoCamera;
class SoNode;
class SoIRRenderAction;

struct SoVulkanDeviceContext;

/*!
  \class SoVulkanRenderManager SoVulkanRenderManager.h
  \brief Qt-free scene-to-Vulkan orchestrator for the render-backend path.

  Traverses a scene graph with SoIRRenderAction to a backend-neutral SoDrawList,
  then submits it to a SoVulkanRenderBackend bound to a caller-supplied
  SoVulkanRenderTarget.  Unlike SoRenderManager it owns no window surface, camera
  sensor, stereo handling or superimpositions; the caller owns the device and
  target and drives render() once per frame.
*/
class COIN_DLL_API SoVulkanRenderManager {
public:
  SoVulkanRenderManager();
  ~SoVulkanRenderManager();

  void setSceneGraph(SoNode * root);
  SoNode * getSceneGraph(void) const;

  //! Optional screen-space overlay scene (e.g. navigation cube); drawn last in the overlay pass, own matrices/viewport.
  void setOverlaySceneGraph(SoNode * root);
  SoNode * getOverlaySceneGraph(void) const;

  //! Optional decoration scene (axis cross), after the overlay scene, own matrices/viewport.
  void setDecorationSceneGraph(SoNode * root);
  SoNode * getDecorationSceneGraph(void) const;

  void setCamera(SoCamera * camera);
  SoCamera * getCamera(void) const;

  void setViewportRegion(const SbViewportRegion & region);
  const SbViewportRegion & getViewportRegion(void) const;

  //! Camera auto-clipping strategy; non-NO recomputes near/far from the bbox each frame (mirrors SoRenderManager).
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

  //! Device-pixel ratio; the renderer scales logical line widths/point sizes by it (see SoRenderParams).
  void setDevicePixelRatio(float ratio);
  float getDevicePixelRatio(void) const;

  //! Bump when app-side state invisible to the graph fingerprint changes (e.g. FreeCAD selection); forces re-traversal.
  void setExternalRevision(uint64_t revision);

  //! Vertical top-to-bottom background gradient (else flat clear), applied before geometry.
  void setBackgroundGradient(SbBool enabled,
                             const SbColor4f & topColor,
                             const SbColor4f & bottomColor);

  //! Vulkan-only wireframe/point edge overlays + color (never consulted by the OpenGL backend).
  void setWireframeOverlay(SbBool enabled);
  void setPointsOverlay(SbBool enabled);
  void setTessellationOverlay(SbBool enabled);
  void setEdgeColor(const SbColor4f & color);
  SbBool getWireframeOverlay(void) const;
  SbBool getPointsOverlay(void) const;
  SbBool getTessellationOverlay(void) const;
  const SbColor4f & getEdgeColor(void) const;

  //! Apply the display/tuning settings blob; re-applied only on change, so callers may push it every frame.
  void setViewSettings(const SoVulkanViewSettings & settings);
  //! Force the next setViewSettings() to re-apply even if unchanged.
  void invalidateViewSettings(void);

  void setClearEnabled(SbBool clearwindow, SbBool clearzbuffer);
  void getClearEnabled(SbBool & clearwindow, SbBool & clearzbuffer) const;

  //! Initialize from a borrowed context (kept alive until shutdown); FALSE on failure.
  SbBool initialize(SoVulkanDeviceContext * context);

  //! Recorded frames the caller may keep in flight; drives deferred destruction and the UBO ring.
  void setMaxFramesInFlight(uint32_t count);

  //! Persistent on-disk pipeline cache: loaded at initialize(), written back at shutdown(); set before initialize().
  void setPipelineCachePath(const std::string & path);

  //! Shut down while the device/queue are valid (idempotent; call before the window tears them down).
  void shutdown(void);

  //! Render target used by the next render(); borrowed, not owned.
  void setRenderTarget(void * target);
  void * getRenderTarget(void) const;

  //! Traverse the scene and submit it; view/projection from the traversed state.  Empty geometry -> clear only.
  SbBool render(SbBool clearwindow = TRUE, SbBool clearzbuffer = TRUE);

  //! Record into the caller's already-begun pass/framebuffer; caller owns submission.  Backend handles setup + LOD pre-pass.
  SbBool renderExternal(SbBool clearwindow,
                        SbBool clearzbuffer,
                        VkCommandBuffer commandBuffer,
                        VkRenderPass renderPass,
                        VkFramebuffer framebuffer);

  //! Record the frame's GPU-timestamp query reset on the caller's buffer,
  //! immediately before vkCmdBeginRenderPass (vkCmdResetQueryPool is illegal
  //! inside a pass).  No-op unless COIN_VULKAN_GPU_TIMING is active.
  void resetExternalGpuQueries(VkCommandBuffer commandBuffer);

  //! Host's camera-anchored lights + ambient for both backends (empty list restores per-command IR lighting).
  void setSceneLights(const SoLightingData & lighting);

  //! Interaction LOD while moving: drop motion-invisible work, restore quality when stopped.
  void setInteractionLod(SbBool active);

  //! Ordinal of the last presented frame (1-based; 0 before first render), bumped once per render.
  uint32_t getRenderFrameCount(void) const;

private:
  class SoVulkanRenderManagerP * pimpl;
};

#endif // COIN_HAVE_VULKAN_RENDERER

#endif // COIN_SOVULKANRENDERMANAGER_H
