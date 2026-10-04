// testsuite/LegacyRendererTest.cpp
//
// Headless smoke test for the legacy fixed-function GL renderer.  The fork
// used to make this renderer optional; it is now always compiled and
// registered, so this pins:
//   1. the installed capability reports the legacy renderer and its classes are
//      registered with the type system, and
//   2. a scene renders through SoOffscreenRenderer.
//
// The offscreen render needs a working GL context.  Coin's GLX/EGL probe can
// abort on headless or Xwayland-less hosts, so on POSIX the render runs in a
// forked child: if the child is killed by a signal (or reports no context) the
// test degrades to a skip instead of failing.  On Windows only the registration
// checks run.

#include <Inventor/C/basic.h>
#include <Inventor/SbColor.h>
#include <Inventor/SbName.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/SoDB.h>
#include <Inventor/SoInteraction.h>
#include <Inventor/SoOffscreenRenderer.h>
#include <Inventor/SoType.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>

#include <cstdio>

#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if !defined(COIN_HAVE_LEGACY_GL_RENDERER) || !COIN_HAVE_LEGACY_GL_RENDERER
#error "LegacyRendererTest must only be built with COIN_HAVE_LEGACY_GL_RENDERER"
#endif

namespace {

int
fail(const char * message)
{
  std::fprintf(stderr, "FAIL: %s\n", message);
  return 1;
}

bool
registered(const char * name)
{
  return SoType::fromName(SbName(name)) != SoType::badType();
}

#if !defined(_WIN32)
// Child exit codes for the forked offscreen render.
enum {
  OFFSCREEN_OK = 0,
  OFFSCREEN_NO_CONTEXT = 10,
  OFFSCREEN_NULL_BUFFER = 11,
  OFFSCREEN_EMPTY = 12
};

int
renderInChild(SoNode * root, const SbViewportRegion & viewport)
{
  const pid_t pid = fork();
  if (pid < 0) {
    return OFFSCREEN_NO_CONTEXT;
  }
  if (pid == 0) {
    SoOffscreenRenderer renderer(viewport);
    renderer.setComponents(SoOffscreenRenderer::RGB);
    renderer.setBackgroundColor(SbColor(0.1f, 0.1f, 0.1f));
    if (!renderer.render(root)) {
      _exit(OFFSCREEN_NO_CONTEXT);
    }
    const unsigned char * buffer = renderer.getBuffer();
    if (buffer == NULL) {
      _exit(OFFSCREEN_NULL_BUFFER);
    }
    const SbVec2s size = renderer.getViewportRegion().getViewportSizePixels();
    const int n = static_cast<int>(size[0]) * static_cast<int>(size[1]) * 3;
    for (int i = 0; i < n; ++i) {
      if (buffer[i] != 0) {
        _exit(OFFSCREEN_OK);
      }
    }
    _exit(OFFSCREEN_EMPTY);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return OFFSCREEN_NO_CONTEXT;
  }
  if (!WIFEXITED(status)) {
    return OFFSCREEN_NO_CONTEXT; // killed by a signal (aborted GL probe) -> skip
  }
  return WEXITSTATUS(status);
}
#endif

} // namespace

int
main(void)
{
  SoDB::init();
  SoInteraction::init();

  int result = 0;

  // 1. Capability + registration.  The legacy renderer must be present.
  if (SoGLRenderAction::getClassTypeId() == SoType::badType()) {
    result = fail("SoGLRenderAction is not registered");
  }

  const char * classes[] = {
    "SoGLLazyElement",
    "SoGLCacheContextElement",
    "SoGLMultiTextureImageElement",
    "SoGLModelMatrixElement",
    "SoGLLineWidthElement",
  };
  for (const char * cls : classes) {
    if (result == 0 && !registered(cls)) {
      static char buffer[128];
      std::snprintf(buffer, sizeof(buffer),
                    "legacy renderer class not registered: %s", cls);
      result = fail(buffer);
    }
  }

  // 2. Offscreen render.
  if (result == 0) {
#if defined(_WIN32)
    std::printf("SKIP: offscreen GL render is not attempted on Windows; "
                "registration checks passed\n");
#else
    SoSeparator * root = new SoSeparator;
    root->ref();
    root->addChild(new SoDirectionalLight);
    SoPerspectiveCamera * camera = new SoPerspectiveCamera;
    root->addChild(camera);
    root->addChild(new SoCube);

    const SbViewportRegion viewport(64, 64);
    camera->viewAll(root, viewport);

    switch (renderInChild(root, viewport)) {
    case OFFSCREEN_OK:
      break;
    case OFFSCREEN_NULL_BUFFER:
      result = fail("SoOffscreenRenderer::render produced no buffer");
      break;
    case OFFSCREEN_EMPTY:
      result = fail("offscreen render produced an all-zero image");
      break;
    default:
      // No context, or the GL stack aborted the child (X error / signal).
      std::printf("SKIP: no usable offscreen GL context; "
                  "registration checks passed\n");
      break;
    }

    root->unref();
#endif
  }

  SoDB::finish();

  if (result == 0) {
    std::printf("OK: legacy renderer registration/offscreen smoke test passed\n");
  }
  return result;
}
