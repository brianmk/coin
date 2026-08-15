// include/Inventor/rendering/SoVulkanRenderManager.h

#ifndef COIN_SOVULKANRENDERMANAGER_H
#define COIN_SOVULKANRENDERMANAGER_H

#include <Inventor/SbColor4f.h>
#include <Inventor/SbVec2s.h>

// Pull in Vulkan handle types for renderExternal().  This header is only
// built when COIN_BUILD_VULKAN_RENDERER is enabled.
#include <vulkan/vulkan.h>

class SbViewportRegion;
class SoCamera;
class SoNode;
class SoIRRenderAction;
class SoVulkanRenderBackend;

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

  void setCamera(SoCamera * camera);
  SoCamera * getCamera(void) const;

  void setViewportRegion(const SbViewportRegion & region);
  const SbViewportRegion & getViewportRegion(void) const;

  void setBackgroundColor(const SbColor4f & color);
  const SbColor4f & getBackgroundColor(void) const;

  void setClearEnabled(SbBool clearwindow, SbBool clearzbuffer);
  void getClearEnabled(SbBool & clearwindow, SbBool & clearzbuffer) const;

  /*!
    \brief Initialize the owned Vulkan backend from a device context.

    The context is borrowed; the caller must keep it alive until shutdown()
    or destruction.  Returns FALSE if the backend cannot initialize.
  */
  SbBool initialize(SoVulkanDeviceContext * context);

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
                        VkRenderPass renderPass);

  SoVulkanRenderBackend * getBackend(void) const;

private:
  class SoVulkanRenderManagerP * pimpl;
};

#endif // COIN_SOVULKANRENDERMANAGER_H
