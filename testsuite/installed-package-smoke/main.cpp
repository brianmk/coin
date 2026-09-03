#include <Inventor/SoDB.h>
#include <Inventor/SoRenderManager.h>
#include <Inventor/SoSceneManager.h>
#include <Inventor/system/gl.h>

int
main()
{
  SoDB::init();
  {
    SoRenderManager renderManager;
    SoSceneManager sceneManager;
    (void) renderManager;
    (void) sceneManager;
  }
  SoDB::finish();
  return 0;
}
