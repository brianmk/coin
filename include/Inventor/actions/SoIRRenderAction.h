#ifndef COIN_SOIRRENDERACTION_H
#define COIN_SOIRRENDERACTION_H

#include <Inventor/actions/SoAction.h>
#include <Inventor/actions/SoSubAction.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/lists/SbList.h>

#include <Inventor/rendering/SoRenderIR.h>

#include <cstddef>
class SoPrimitiveVertex;
class SoPath;
class SoPathList;
class SoIRRenderActionP;

/*!
  \class SoIRRenderAction SoIRRenderAction.h
  \brief Traversal front-end recording geometry/material/state into a
  backend-neutral SoDrawList (issued later by a backend), not OpenGL.
  \ingroup coin_actions
*/
class COIN_DLL_API SoIRRenderAction : public SoAction {
  typedef SoAction inherited;
  SO_ACTION_HEADER(SoIRRenderAction);

public:
  //! Streams primitives from generatePrimitives() fallback; active collector is a stack.
  class PrimitiveCollector {
  public:
    virtual ~PrimitiveCollector() {}
    virtual void onTriangle(const SoPrimitiveVertex * v1,
                            const SoPrimitiveVertex * v2,
                            const SoPrimitiveVertex * v3) = 0;
    virtual void onLine(const SoPrimitiveVertex * v1,
                        const SoPrimitiveVertex * v2) = 0;
    virtual void onPoint(const SoPrimitiveVertex * v) = 0;
  };

  static void initClass(void);

  /*!\brief Traverse a node via its doAction() entry point. */
  static void callDoAction(SoAction * action, SoNode * node);

  SoIRRenderAction(const SbViewportRegion & vp);
  virtual ~SoIRRenderAction();

  //! Clear the current draw list and begin a new retained frame.
  void beginFrame();

  void setViewportRegion(const SbViewportRegion & vp);
  const SbViewportRegion & getViewportRegion(void) const { return this->vpRegion; }

  //! Mark frame camera-dependent: re-record rather than replay a camera-baked cache (SoShapeScale).
  void setCameraDependent(SbBool on);
  SbBool isCameraDependent(void) const;

  // Standard entry points, mirroring SoGLRenderAction
  virtual void apply(SoNode * root) override;
  virtual void apply(SoPath * path) override;
  virtual void apply(const SoPathList & pathlist, SbBool obeysrules = FALSE) override;

  //! Return the generated draw list for the current frame.
  const SoDrawList & getDrawList(void) const { return this->drawlist; }
  //! Mutable access to the generated draw list for the current frame.
  SoDrawList & getMutableDrawList() { return this->drawlist; }

  //! Per-frame geometry owned by the action; valid until frame clear/pool rewind.
  void * allocateGeometryStorage(size_t bytes, size_t alignment = alignof(float));

  //! Clear all transient geometry owned by the current frame.
  void clearGeometryPool();

  //! Push a primitive collector for subsequent fallback primitive generation.
  void pushPrimitiveCollector(PrimitiveCollector * collector);
  //! Pop the current primitive collector. The caller must pop in stack order.
  void popPrimitiveCollector(PrimitiveCollector * collector);
  //! Return the currently active primitive collector, or NULL.
  PrimitiveCollector * getActivePrimitiveCollector(void) const;

protected:
  virtual void beginTraversal(SoNode * node) override;
  virtual void endTraversal(SoNode * node) override;

private:
  void resetFrameResources();

  SbViewportRegion vpRegion;
  SoDrawList       drawlist;
  SoIRRenderActionP * pimpl;
};

#endif // COIN_SOIRRENDERACTION_H
