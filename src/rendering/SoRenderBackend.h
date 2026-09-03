// src/rendering/SoRenderBackend.h

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

/*!
  struct SoRenderParams
  rief Per-render values consumed by a retained-rendering backend.

  The values describe the currently bound framebuffer and the view being
  rendered into it.  Target ownership and application orchestration remain
  outside this interface.
*/
struct SoRenderParams {
  SbViewportRegion viewport;
  SbMatrix         viewMatrix;
  SbMatrix         projMatrix;
  float            devicePixelRatio = 1.0f;
  SbColor4f        clearColor;
  float            clearDepth = 1.0f;
  uint32_t         clearStencil = 0;
  uint32_t         flags = 0;

  // Background gradient (vertical, screen-space).  When backgroundGradient
  // is set the backend fills the viewport with a top-to-bottom gradient
  // between backgroundTopColor and backgroundBottomColor before drawing
  // geometry, instead of a flat clearColor.
  SbBool           backgroundGradient = FALSE;
  SbColor4f        backgroundTopColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);
  SbColor4f        backgroundBottomColor = SbColor4f(0.0f, 0.0f, 0.0f, 1.0f);

  /*!
    \brief Backend-defined render destination for this frame.

    The base interface does not interpret this pointer.  Concrete backends
    document the structure they expect.  The Vulkan backend expects it to
    point to a SoVulkanRenderTarget (see Inventor/rendering/
    SoVulkanRenderTarget.h).  A null pointer means "render into whatever
    destination the backend is currently bound to", which is backend
    specific.  The pointer is borrowed for the duration of render() only.
  */
  void * renderTarget = nullptr;

  //! Monotonically increasing generation counter for the camera used this
  //! frame.  Bumped whenever the active camera node changes or its pose
  //! (position/orientation/projection) changes, so a backend can reliably
  //! detect a camera move without diffing floating-point matrices.  A value
  //! of 0 means "not supplied" (the backend should fall back to the matrices).
  uint32_t cameraVersion = 0;

  //! 1-based ordinal of the presented frame this render belongs to, bumped
  //! exactly once per frame by the embedding (SoVulkanRenderManager).  Used
  //! as a stable, ordering-independent correlation key between backend
  //! debug traces (RTDBG lines), captured frame dumps and probe phase
  //! markers.  0 means "not supplied".
  uint32_t frame = 0;
};

/*!
  struct SoRenderBackendInitParams
  rief Minimal backend initialization hooks.
*/
struct SoRenderBackendInitParams {
  void *               userData = nullptr;
  SoRenderBackendLogFn logCallback = nullptr;
  SoRenderBackendLogFn errorCallback = nullptr;
};

/*!
  class SoRenderBackend
  rief Backend-neutral lifecycle and DrawList execution interface.

  The retained IR does not depend on this interface or on a graphics API.
  Concrete backends own all device resources.
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
