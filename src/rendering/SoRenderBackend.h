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
  uint32_t         flags = 0;

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
