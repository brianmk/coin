#include "rendering/CoinOffscreenGLCanvas.h"

#include <Inventor/C/glue/gl.h>
#include <Inventor/SoDB.h>

#include <cstdlib>
#include <iostream>

namespace {

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

bool check(bool condition, const char * message)
{
  if (!condition) std::cerr << "FAIL: " << message << std::endl;
  return condition;
}

void set_environment(const char * name, const char * value)
{
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

bool can_create_offscreen_context()
{
  void * context = cc_glglue_context_create_offscreen(2, 2);
  if (context == NULL) return false;

  if (!cc_glglue_context_make_current(context)) {
    cc_glglue_context_destruct(context);
    return false;
  }

  cc_glglue_context_reinstate_previous(context);
  cc_glglue_context_destruct(context);
  return true;
}

}

int main()
{
  set_environment("COIN_EGL", "1");
  set_environment("EGL_PLATFORM", "surfaceless");
  set_environment("COIN_EGL_CORE_PROFILE", "1");

  SoDB::init();

  if (!can_create_offscreen_context()) {
    SoDB::finish();
    return skip("core EGL offscreen context could not be established");
  }

  int result = 0;
  {
    CoinOffscreenGLCanvas canvas;
    canvas.setWantedSize(SbVec2s(2, 2));
    if (canvas.activateGLContext() == 0) {
      result = skip("core EGL offscreen context is unavailable");
    }
    else {
      glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      glPixelStorei(GL_PACK_ALIGNMENT, 8);
      glPixelStorei(GL_PACK_ROW_LENGTH, 7);
      glPixelStorei(GL_PACK_SKIP_ROWS, 3);
      glPixelStorei(GL_PACK_SKIP_PIXELS, 2);

      uint8_t pixels[2 * 2 * 4] = { 0 };
      canvas.readPixels(pixels, SbVec2s(2, 2), 2, 4);

      GLint packAlignment;
      GLint packRowLength;
      GLint packSkipRows;
      GLint packSkipPixels;
      glGetIntegerv(GL_PACK_ALIGNMENT, &packAlignment);
      glGetIntegerv(GL_PACK_ROW_LENGTH, &packRowLength);
      glGetIntegerv(GL_PACK_SKIP_ROWS, &packSkipRows);
      glGetIntegerv(GL_PACK_SKIP_PIXELS, &packSkipPixels);
      if (!check(packAlignment == 8, "PACK_ALIGNMENT was not restored")) result = 1;
      if (!check(packRowLength == 7, "PACK_ROW_LENGTH was not restored")) result = 1;
      if (!check(packSkipRows == 3, "PACK_SKIP_ROWS was not restored")) result = 1;
      if (!check(packSkipPixels == 2, "PACK_SKIP_PIXELS was not restored")) result = 1;

      canvas.deactivateGLContext();

      for (unsigned int i = 0; i < sizeof(pixels); i += 4) {
        if (!check(pixels[i + 0] >= 60 && pixels[i + 0] <= 70,
                   "red readback component is incorrect")) result = 1;
        if (!check(pixels[i + 1] >= 120 && pixels[i + 1] <= 135,
                   "green readback component is incorrect")) result = 1;
        if (!check(pixels[i + 2] >= 185 && pixels[i + 2] <= 200,
                   "blue readback component is incorrect")) result = 1;
        if (!check(pixels[i + 3] >= 245,
                   "alpha readback component is incorrect")) result = 1;
      }
    }
  }

  SoDB::finish();
  return result;
}
