#include <Inventor/SoDB.h>
#include <Inventor/SoRenderManager.h>
#include <Inventor/SoSceneManager.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/elements/SoDevicePixelRatioElement.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoSeparator.h>

#if COIN_HAVE_LEGACY_GL_RENDERER
#error "SceneManagerCoreTest requires LegacyGL to be disabled"
#endif

static void
renderManagerCallback(void *, SoRenderManager *)
{
}

static void
sceneManagerCallback(void *, SoSceneManager *)
{
}

static int
checkRenderManager(SoRenderManager & manager)
{
  manager.setWindowSize(SbVec2s(800, 600));
  if (manager.getWindowSize() != SbVec2s(800, 600)) return 1;

  if (manager.getDevicePixelRatio() != 1.0f) return 1;
  manager.setDevicePixelRatio(2.0f);
  if (manager.getDevicePixelRatio() != 2.0f) return 1;

  manager.setSize(SbVec2s(640, 480));
  if (manager.getSize() != SbVec2s(640, 480)) return 1;

  manager.setOrigin(SbVec2s(17, 19));
  if (manager.getOrigin() != SbVec2s(17, 19)) return 1;

  const SbViewportRegion region(SbVec2s(1024, 768));
  manager.setViewportRegion(region);
  if (manager.getViewportRegion() != region) return 1;

  SoSeparator * root = new SoSeparator;
  manager.setSceneGraph(root);
  if (manager.getSceneGraph() != root) return 1;
  manager.setSceneGraph(NULL);

  SoOrthographicCamera * camera = new SoOrthographicCamera;
  manager.setCamera(camera);
  if (manager.getCamera() != camera) return 1;
  manager.setCamera(NULL);

  manager.setRenderCallback(renderManagerCallback);
  if (!manager.isAutoRedraw()) return 1;
  manager.scheduleRedraw();
  manager.setRenderCallback(NULL);
  if (manager.isAutoRedraw()) return 1;

  return 0;
}

static int
checkSceneManager(SoSceneManager & manager)
{
  manager.setWindowSize(SbVec2s(800, 600));
  if (manager.getWindowSize() != SbVec2s(800, 600)) return 1;

  if (manager.getDevicePixelRatio() != 1.0f) return 1;
  manager.setDevicePixelRatio(2.0f);
  if (manager.getDevicePixelRatio() != 2.0f) return 1;

  manager.setSize(SbVec2s(640, 480));
  if (manager.getSize() != SbVec2s(640, 480)) return 1;

  manager.setOrigin(SbVec2s(17, 19));
  if (manager.getOrigin() != SbVec2s(17, 19)) return 1;

  const SbViewportRegion region(SbVec2s(1024, 768));
  manager.setViewportRegion(region);
  if (manager.getViewportRegion() != region) return 1;

  SoSeparator * root = new SoSeparator;
  manager.setSceneGraph(root);
  if (manager.getSceneGraph() != root) return 1;
  manager.setSceneGraph(NULL);

  SoOrthographicCamera * camera = new SoOrthographicCamera;
  manager.setCamera(camera);
  if (manager.getCamera() != camera) return 1;
  manager.setCamera(NULL);

  manager.setRenderCallback(sceneManagerCallback);
  if (!manager.isAutoRedraw()) return 1;
  manager.scheduleRedraw();
  manager.setRenderCallback(NULL);
  if (manager.isAutoRedraw()) return 1;

  return 0;
}

int
main(void)
{
  SoDB::init();

  SoRenderManager renderManager;
  if (checkRenderManager(renderManager) != 0) return 1;

  SoSceneManager sceneManager;
  if (checkSceneManager(sceneManager) != 0) return 1;

  return 0;
}
