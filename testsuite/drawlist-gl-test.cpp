#include "rendering/CoinOffscreenGLCanvas.h"
#include "rendering/SoGLRenderBackend.h"

#include <Inventor/SoDB.h>
#include <Inventor/system/gl.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

void set_environment(const char * name, const char * value)
{
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

SoRenderParams renderParams()
{
  SoRenderParams params;
  params.viewport = SbViewportRegion(32, 32);
  params.viewport.setViewportPixels(SbVec2s(0, 0), SbVec2s(32, 32));
  params.viewMatrix.makeIdentity();
  params.projMatrix.makeIdentity();
  params.clearColor.setValue(0.0f, 0.0f, 0.0f, 1.0f);
  params.clearDepth = 1.0f;
  params.flags = SO_PARAM_CLEAR_WINDOW | SO_PARAM_CLEAR_DEPTH;
  return params;
}

std::vector<uint8_t> readPixels(const CoinOffscreenGLCanvas & canvas)
{
  std::vector<uint8_t> pixels(32 * 32 * 4, 0);
  canvas.readPixels(pixels.data(), SbVec2s(32, 32), 32, 4);
  return pixels;
}

const uint8_t * pixelAt(const std::vector<uint8_t> & pixels, int x, int y)
{
  return &pixels[static_cast<size_t>(y * 32 + x) * 4];
}

bool nearColor(const uint8_t * pixel, int red, int green, int blue)
{
  return std::abs(static_cast<int>(pixel[0]) - red) < 35 &&
    std::abs(static_cast<int>(pixel[1]) - green) < 35 &&
    std::abs(static_cast<int>(pixel[2]) - blue) < 35;
}

SoRenderCommand texturedQuad(const float * positions,
                             const uint32_t * indices,
                             const float * texcoords,
                             const unsigned char * pixels,
                             int components,
                             const SbVec4f & color)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = positions;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 4;
  command.material.diffuse = color;
  command.material.texture.pixels = pixels;
  command.material.texture.width = 1;
  command.material.texture.height = 1;
  command.material.texture.numComponents = components;
  return command;
}

} // namespace

static int
runTest()
{
  set_environment("COIN_EGL", "1");
  set_environment("EGL_PLATFORM", "surfaceless");
  set_environment("COIN_EGL_CORE_PROFILE", "1");

  SoDB::init();

  CoinOffscreenGLCanvas canvas;
  canvas.setWantedSize(SbVec2s(32, 32));
  if (canvas.activateGLContext() == 0) {
    return skip("core EGL offscreen context is unavailable");
  }

  SoGLRenderBackend backend;
  SoRenderBackendInitParams initparams;
  if (!backend.initialize(initparams)) {
    canvas.deactivateGLContext();
    return skip("core OpenGL draw-list backend could not initialize");
  }

  const uint32_t indices[] = { 0, 1, 2, 0, 2, 3 };
  const float texcoords[] = {
    0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f
  };
  const float quads[][12] = {
    { -1.0f, -1.0f, 0.0f, -0.5f, -1.0f, 0.0f,
      -0.5f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f },
    { -0.5f, -1.0f, 0.0f,  0.0f, -1.0f, 0.0f,
       0.0f, 1.0f, 0.0f, -0.5f, 1.0f, 0.0f },
    {  0.0f, -1.0f, 0.0f,  0.5f, -1.0f, 0.0f,
       0.5f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f },
    {  0.5f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,
       1.0f, 1.0f, 0.0f,  0.5f, 1.0f, 0.0f }
  };
  const unsigned char luminance[] = { 220 };
  const unsigned char luminanceAlpha[] = { 100, 255 };
  const unsigned char rgb[] = { 0, 220, 0 };
  const unsigned char rgba[] = { 0, 0, 220, 255 };

  SoDrawList drawlist;
  drawlist.addCommand(texturedQuad(quads[0], indices, texcoords,
                                   luminance, 1,
                                   SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));
  drawlist.addCommand(texturedQuad(quads[1], indices, texcoords,
                                   luminanceAlpha, 2,
                                   SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));
  drawlist.addCommand(texturedQuad(quads[2], indices, texcoords,
                                   rgb, 3,
                                   SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));
  drawlist.addCommand(texturedQuad(quads[3], indices, texcoords,
                                   rgba, 4,
                                   SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));

  const SoRenderParams params = renderParams();
  int result = 0;
  SoDrawList empty;
  if (!backend.render(empty, params)) {
    std::cerr << "FAIL: empty draw list was not accepted" << std::endl;
    result = 1;
  }

  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: indexed draw-list execution failed" << std::endl;
    result = 1;
  }
  else {
    glFinish();
    const std::vector<uint8_t> pixels = readPixels(canvas);
    if (!nearColor(pixelAt(pixels, 4, 16), 220, 220, 220)) {
      std::cerr << "FAIL: L texture upload produced unexpected pixels" << std::endl;
      result = 1;
    }
    if (!nearColor(pixelAt(pixels, 12, 16), 100, 100, 100)) {
      std::cerr << "FAIL: LA texture upload produced unexpected pixels" << std::endl;
      result = 1;
    }
    if (!nearColor(pixelAt(pixels, 20, 16), 0, 220, 0)) {
      std::cerr << "FAIL: RGB texture upload produced unexpected pixels" << std::endl;
      result = 1;
    }
    if (!nearColor(pixelAt(pixels, 28, 16), 0, 0, 220)) {
      std::cerr << "FAIL: RGBA texture upload produced unexpected pixels" << std::endl;
      result = 1;
    }
  }

  // An unchanged DrawList must be safe to execute repeatedly without another
  // upload or a change in output.
  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: repeated DrawList execution failed" << std::endl;
    result = 1;
  }

  // Exercise unindexed geometry and the basic vertex-color path.
  const float triangle[] = {
    -0.8f, -0.8f, 0.0f,
     0.8f, -0.8f, 0.0f,
     0.0f,  0.8f, 0.0f
  };
  const float vertexColors[] = {
    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f
  };
  drawlist.clear();
  SoRenderCommand unindexed;
  unindexed.modelMatrix.makeIdentity();
  unindexed.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  unindexed.geometry.vertexCount = 3;
  unindexed.geometry.positions = triangle;
  unindexed.geometry.vertexStride = sizeof(float) * 3;
  unindexed.geometry.colors = vertexColors;
  unindexed.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  drawlist.addCommand(unindexed);
  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: unindexed vertex-color execution failed" << std::endl;
    result = 1;
  }
  else {
    glFinish();
    const std::vector<uint8_t> pixels = readPixels(canvas);
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 255, 0)) {
      std::cerr << "FAIL: vertex color was not used for unindexed geometry"
                << std::endl;
      result = 1;
    }
  }

  // clear() changes the generation. Replacing the command must not reuse the
  // previous frame's GPU data.
  const float replacement[] = {
    -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f
  };
  drawlist.clear();
  SoRenderCommand replaced;
  replaced.modelMatrix.makeIdentity();
  replaced.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  replaced.geometry.vertexCount = 4;
  replaced.geometry.indexCount = 6;
  replaced.geometry.positions = replacement;
  replaced.geometry.indices = indices;
  replaced.geometry.vertexStride = sizeof(float) * 3;
  replaced.material.diffuse = SbVec4f(0.0f, 0.0f, 1.0f, 1.0f);
  drawlist.addCommand(replaced);
  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: generation-invalidated DrawList execution failed" << std::endl;
    result = 1;
  }
  else {
    glFinish();
    const std::vector<uint8_t> pixels = readPixels(canvas);
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 0, 255)) {
      std::cerr << "FAIL: clear()/generation change reused stale GPU data" << std::endl;
      result = 1;
    }
  }

  // The active context remains valid while normal GL destruction happens.
  backend.shutdown();
  if (backend.isInitialized()) {
    std::cerr << "FAIL: backend remained initialized after shutdown" << std::endl;
    result = 1;
  }
  canvas.deactivateGLContext();
  return result;
}

int
main()
{
  const int result = runTest();
  SoDB::finish();
  return result;
}
