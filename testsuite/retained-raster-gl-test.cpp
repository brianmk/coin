#include "rendering/CoinOffscreenGLCanvas.h"
#include "rendering/SoGLRenderBackend.h"

#include <Inventor/SoDB.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

void setEnvironment(const char * name, const char * value)
{
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

struct Fixture {
  CoinOffscreenGLCanvas canvas;
  SoGLRenderBackend backend;

  bool initialize()
  {
    canvas.setWantedSize(SbVec2s(64, 64));
    if (canvas.activateGLContext() == 0) return false;
    SoRenderBackendInitParams init = {};
    if (backend.initialize(init)) return true;
    canvas.deactivateGLContext();
    return false;
  }

  std::vector<uint8_t> render(SoDrawList & drawlist,
                              const SbVec4f & clearColor,
                              float dpr = 1.0f)
  {
    SoRenderParams params = {};
    params.viewport = SbViewportRegion(64, 64);
    params.viewport.setViewportPixels(SbVec2s(0, 0), SbVec2s(64, 64));
    params.viewMatrix.makeIdentity();
    params.projMatrix.makeIdentity();
    params.devicePixelRatio = dpr;
    params.clearColor = clearColor;
    params.clearDepth = 1.0f;
    params.flags = SO_PARAM_CLEAR_WINDOW | SO_PARAM_CLEAR_DEPTH;
    backend.render(drawlist, params);
    glFinish();
    std::vector<uint8_t> pixels(64 * 64 * 4, 0);
    canvas.readPixels(pixels.data(), SbVec2s(64, 64), 64, 4);
    return pixels;
  }

  void shutdown()
  {
    backend.shutdown();
    canvas.deactivateGLContext();
  }
};

const uint8_t * pixelAt(const std::vector<uint8_t> & pixels, int x, int y)
{
  return &pixels[static_cast<size_t>(y * 64 + x) * 4];
}

bool check(bool condition, const char * message)
{
  if (!condition) std::cerr << "FAIL: " << message << std::endl;
  return condition;
}

SoRenderCommand coloredCommand(SoPrimitiveTopology topology,
                               const float * positions,
                               uint32_t vertexCount,
                               const SbVec4f & color)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = topology;
  command.geometry.positions = positions;
  command.geometry.vertexCount = vertexCount;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = color;
  command.material.shadingModel = SO_SHADING_UNLIT;
  return command;
}

bool testWideLine(Fixture & fixture)
{
  const float positions[] = { -0.8f, 0.0f, 0.0f, 0.8f, 0.0f, 0.0f };
  SoDrawList drawlist;
  SoRenderCommand command = coloredCommand(SO_TOPOLOGY_LINES, positions, 2,
                                           SbVec4f(1, 0, 0, 1));
  command.state.raster.lineWidth = 4.0f;
  drawlist.addCommand(command);
  const std::vector<uint8_t> pixels = fixture.render(drawlist,
                                                      SbVec4f(0, 0, 0, 1), 2.0f);
  const uint8_t * center = pixelAt(pixels, 32, 32);
  const uint8_t * edge = pixelAt(pixels, 32, 35);
  return check(center[0] > 200 && center[1] < 50 &&
               edge[0] > 150 && edge[1] < 80,
               "wide-line geometry shader did not apply DPR-scaled width");
}

bool testRoundPoint(Fixture & fixture)
{
  const float position[] = { 0.0f, 0.0f, 0.0f };
  SoDrawList drawlist;
  SoRenderCommand command = coloredCommand(SO_TOPOLOGY_POINTS, position, 1,
                                           SbVec4f(0, 1, 0, 1));
  command.state.raster.pointSize = 12.0f;
  command.state.raster.pointShape = SO_POINT_SHAPE_ROUND;
  drawlist.addCommand(command);
  const std::vector<uint8_t> pixels = fixture.render(drawlist,
                                                      SbVec4f(0, 0, 0, 1));
  const uint8_t * center = pixelAt(pixels, 32, 32);
  return check(center[1] > 200 && center[0] < 50,
               "round point pipeline did not render the point");
}

bool testPixelDraw(Fixture & fixture)
{
  const float positions[] = {
    -0.1f, -0.1f, 0.0f,  0.1f, -0.1f, 0.0f,
     0.1f,  0.1f, 0.0f, -0.1f,  0.1f, 0.0f
  };
  const float texcoords[] = {
    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f
  };
  const uint32_t indices[] = { 0, 1, 2, 0, 2, 3 };
  const unsigned char image[] = {
    255, 0, 0, 255, 255, 0, 0, 255,
    255, 0, 0, 255, 255, 0, 0, 255
  };
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
  command.material.flags = SO_MAT_HAS_TEXTURE | SO_MAT_IS_PIXEL_IMAGE;
  command.material.texture.pixels = image;
  command.material.texture.width = 2;
  command.material.texture.height = 2;
  command.material.texture.numComponents = 4;
  command.pixelText.originX = 20;
  command.pixelText.originY = 20;
  command.material.shadingModel = SO_SHADING_UNLIT;
  SoDrawList drawlist;
  drawlist.addCommand(command);
  const std::vector<uint8_t> pixels = fixture.render(drawlist,
                                                      SbVec4f(0, 0, 1, 1));
  const uint8_t * pixel = pixelAt(pixels, 20, 20);
  return check(pixel[0] > 200 && pixel[1] < 50 && pixel[2] < 50,
               "pixel pipeline did not sample the retained image at its origin");
}

} // namespace

int main()
{
  setEnvironment("COIN_EGL", "1");
  setEnvironment("EGL_PLATFORM", "surfaceless");
  setEnvironment("COIN_EGL_CORE_PROFILE", "1");
  SoDB::init();
  Fixture fixture;
  if (!fixture.initialize()) {
    SoDB::finish();
    return skip("core EGL raster context is unavailable");
  }

  int result = 0;
  if (!testWideLine(fixture)) result = 1;
  if (!testRoundPoint(fixture)) result = 1;
  if (!testPixelDraw(fixture)) result = 1;
  fixture.shutdown();
  SoDB::finish();
  return result;
}
