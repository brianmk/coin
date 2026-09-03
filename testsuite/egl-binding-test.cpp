#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "glue/gl_egl.h"

#include <Inventor/C/glue/gl.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

namespace {

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

EGLDisplay acquire_display()
{
  const char * platform = std::getenv("EGL_PLATFORM");
  if (platform && std::strcmp(platform, "surfaceless") == 0) {
    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!getPlatformDisplay) return EGL_NO_DISPLAY;
    return getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                              EGL_DEFAULT_DISPLAY, NULL);
  }
  return eglGetDisplay(EGL_DEFAULT_DISPLAY);
}

bool check(bool condition, const char * message)
{
  if (!condition) std::cerr << "FAIL: " << message << std::endl;
  return condition;
}

} // namespace

int main()
{
#ifdef _WIN32
  _putenv_s("COIN_EGL", "1");
#else
  setenv("COIN_EGL", "1", 1);
#endif

  EGLDisplay display = acquire_display();
  if (display == EGL_NO_DISPLAY) return skip("no EGL display");

  EGLint major = 0;
  EGLint minor = 0;
  if (eglInitialize(display, &major, &minor) == EGL_FALSE) {
    return skip("EGL display could not be initialized");
  }

  EGLContext foreignContext = EGL_NO_CONTEXT;
  EGLSurface drawSurface = EGL_NO_SURFACE;
  EGLSurface readSurface = EGL_NO_SURFACE;
  void * coinContext = NULL;
  bool coinMadeCurrent = false;
  int result = 1;

  do {
    if (eglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE) {
      result = skip("OpenGL ES client API is unavailable");
      break;
    }

    EGLint configAttributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_NONE
    };
    EGLConfig config = (EGLConfig) 0;
    EGLint configCount = 0;
    if (eglChooseConfig(display, configAttributes, &config, 1, &configCount) == EGL_FALSE ||
        configCount == 0) {
      result = skip("no OpenGL ES pbuffer configuration");
      break;
    }

    EGLint surfaceAttributes[] = {
      EGL_WIDTH, 16,
      EGL_HEIGHT, 16,
      EGL_NONE
    };
    drawSurface = eglCreatePbufferSurface(display, config, surfaceAttributes);
    readSurface = eglCreatePbufferSurface(display, config, surfaceAttributes);
    if (drawSurface == EGL_NO_SURFACE || readSurface == EGL_NO_SURFACE) {
      result = skip("OpenGL ES pbuffers could not be created");
      break;
    }

    EGLint contextAttributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE
    };
    foreignContext = eglCreateContext(display, config, EGL_NO_CONTEXT,
                                      contextAttributes);
    if (foreignContext == EGL_NO_CONTEXT ||
        eglMakeCurrent(display, drawSurface, readSurface, foreignContext) == EGL_FALSE) {
      result = skip("OpenGL ES context could not be made current");
      break;
    }

    const EGLenum previousApi = eglQueryAPI();
    const EGLDisplay previousDisplay = eglGetCurrentDisplay();
    const EGLContext previousContext = eglGetCurrentContext();
    const EGLSurface previousDrawSurface = eglGetCurrentSurface(EGL_DRAW);
    const EGLSurface previousReadSurface = eglGetCurrentSurface(EGL_READ);

    // Let Coin exercise reuse of the active display even when the test had to
    // request a surfaceless display to create the foreign binding.
#ifdef _WIN32
    _putenv_s("EGL_PLATFORM", "");
#else
    unsetenv("EGL_PLATFORM");
#endif

    coinContext = cc_glglue_context_create_offscreen(16, 16);
    if (!check(coinContext != NULL, "Coin offscreen context creation failed")) {
      break;
    }
    if (!check(eglQueryAPI() == previousApi,
               "Coin did not restore the caller API after context creation")) {
      break;
    }
    if (!check(cc_glglue_context_make_current(coinContext),
               "Coin offscreen context could not be made current")) {
      break;
    }
    coinMadeCurrent = true;
    if (!check(eglQueryAPI() == EGL_OPENGL_API,
               "Coin context did not bind the OpenGL client API")) {
      break;
    }
    if (!check(eglGetCurrentDisplay() == previousDisplay,
               "Coin context did not use the caller's EGL display")) {
      break;
    }

    cc_glglue_context_reinstate_previous(coinContext);
    coinMadeCurrent = false;
    if (!check(eglQueryAPI() == previousApi,
               "Coin did not restore the caller API")) {
      break;
    }
    if (!check(eglGetCurrentDisplay() == previousDisplay &&
               eglGetCurrentContext() == previousContext &&
               eglGetCurrentSurface(EGL_DRAW) == previousDrawSurface &&
               eglGetCurrentSurface(EGL_READ) == previousReadSurface,
               "Coin did not restore the complete caller EGL binding")) {
      break;
    }

    cc_glglue_context_destruct(coinContext);
    coinContext = NULL;
    eglglue_cleanup();
    if (!check(eglQueryAPI() == previousApi &&
               eglGetCurrentDisplay() == previousDisplay &&
               eglGetCurrentContext() == previousContext &&
               eglGetCurrentSurface(EGL_DRAW) == previousDrawSurface &&
               eglGetCurrentSurface(EGL_READ) == previousReadSurface,
               "Coin cleanup did not preserve the borrowed EGL binding")) {
      break;
    }

    result = 0;

  } while (false);

  if (coinMadeCurrent) cc_glglue_context_reinstate_previous(coinContext);
  if (coinContext) cc_glglue_context_destruct(coinContext);
  eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  if (foreignContext != EGL_NO_CONTEXT) eglDestroyContext(display, foreignContext);
  if (drawSurface != EGL_NO_SURFACE) eglDestroySurface(display, drawSurface);
  if (readSurface != EGL_NO_SURFACE) eglDestroySurface(display, readSurface);
  eglTerminate(display);
  return result;
}
