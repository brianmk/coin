#ifndef COIN_SORENDERBACKEND_H
#define COIN_SORENDERBACKEND_H

#include <Inventor/SbBasic.h>
#include <Inventor/SbColor4f.h>
#include <Inventor/SbMatrix.h>
#include <Inventor/SbViewportRegion.h>

#include <cstdint>

#include <Inventor/rendering/SoRenderIR.h>

class SoDrawList;

typedef void (*SoRenderBackendLogFn)(const char * message, void * userdata);

/*! \struct SoRenderParams \brief Per-render values consumed by a retained-rendering backend.
  Describes the bound framebuffer and the view rendered into it; target ownership
  and orchestration stay outside this interface. */
struct SoRenderParams {
  SbViewportRegion viewport;
  SbMatrix         viewMatrix;
  SbMatrix         projMatrix;
  float            devicePixelRatio = 1.0f;
  SbColor4f        clearColor;
  float            clearDepth = 1.0f;
  uint32_t         clearStencil = 0;
  uint32_t         flags = 0;

  // Vertical screen-space gradient: when set, fill the viewport top-to-bottom
  // between backgroundTopColor and backgroundBottomColor instead of a flat clearColor.
  SbBool           backgroundGradient = FALSE;
  SbColor4f        backgroundTopColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbColor4f        backgroundBottomColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);

  /*! \brief Backend-defined render destination for this frame, borrowed for render() only.
    Uninterpreted by the base interface; Vulkan expects a SoVulkanRenderTarget
    (Inventor/rendering/SoVulkanRenderTarget.h). NULL = the backend's current binding. */
  void * renderTarget = nullptr;

  //! Camera generation counter, bumped on camera node/pose change (0 = not
  //! supplied, fall back to matrices). Detects moves without diffing floats.
  uint32_t cameraVersion = 0;

  //! 1-based presented-frame ordinal bumped once per frame by
  //! SoVulkanRenderManager; correlation key for RTDBG traces/frame dumps/phase markers (0 = unset).
  uint32_t frame = 0;

  //! Set by SoVulkanRenderManager on a retained-IR replay (scene graph unchanged,
  //! no traversal): retained geometry is bit-identical, so a backend may skip its
  //! content-hash re-verification when pointer identity matches its cache. Overlay
  //! commands are re-produced every frame and still verified. FALSE by default.
  SbBool geometryContentUnchanged = FALSE;

  //! Set by the embedding while the camera moves (interaction LOD): a backend
  //! may drop motion-invisible work and restore quality when stopped. Vulkan draws
  //! wide lines as 1px GPU lines vs CPU-expanding segments into quads. FALSE by default.
  SbBool interactionLod = FALSE;
};

/*! \struct SoRenderBackendInitParams \brief Minimal backend initialization hooks. */
struct SoRenderBackendInitParams {
  void *               userData = nullptr;
  SoRenderBackendLogFn logCallback = nullptr;
  SoRenderBackendLogFn errorCallback = nullptr;
};

/*!
  \class SoRenderBackend
  \brief Retained-render lifecycle and DrawList execution interface.

  The retained IR depends on neither this interface nor a graphics API; concrete
  backends own all device resources. Effectively the Vulkan backend interface:
  SoVulkanRenderBackend is production; SoGLRenderBackend is a testsuite-only
  reference (not in libCoin; the GL viewport still uses SoGLRenderAction).
*/
class SoRenderBackend {
public:
  SoRenderBackend();
  virtual ~SoRenderBackend();

  virtual const char * getName() const = 0;

  virtual SbBool initialize(const SoRenderBackendInitParams & params) = 0;
  virtual void shutdown() = 0;
  virtual SbBool render(const SoDrawList & drawlist,
                        const SoRenderParams & params) = 0;

  SbBool isInitialized() const;

protected:
  void setInitialized(SbBool state);
  void setInitParams(const SoRenderBackendInitParams & params);
  const SoRenderBackendInitParams & getInitParams() const;

  void emitLog(const char * message) const;
  void emitError(const char * message) const;

  void debugValidateDrawList(const SoDrawList & drawlist) const;

private:
  SbBool                    initialized;
  SoRenderBackendInitParams initparams;
};

#endif // COIN_SORENDERBACKEND_H
