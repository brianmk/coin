#include "rendering/CoinOffscreenGLCanvas.h"
#include "rendering/SoGLRenderBackend.h"

#include <Inventor/SoDB.h>

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

void setEnvironment(const char * name, const char * value)
{
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

struct RenderFixture {
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
                              const SbVec4f & clearColor)
  {
    SoRenderParams params = {};
    params.viewport = SbViewportRegion(64, 64);
    params.viewport.setViewportPixels(SbVec2s(0, 0), SbVec2s(64, 64));
    params.viewMatrix.makeIdentity();
    params.projMatrix.makeIdentity();
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

const uint32_t quadIndices[] = { 0, 1, 2, 0, 2, 3 };
const float quadPositions[] = {
  -0.8f, -0.8f, 0.0f,  0.8f, -0.8f, 0.0f,
   0.8f,  0.8f, 0.0f, -0.8f,  0.8f, 0.0f
};
const float halfLeftPositions[] = {
  -1.0f, -0.8f, 0.0f,  0.0f, -0.8f, 0.0f,
   0.0f,  0.8f, 0.0f, -1.0f,  0.8f, 0.0f
};
const float halfRightPositions[] = {
   0.0f, -0.8f, 0.0f,  1.0f, -0.8f, 0.0f,
   1.0f,  0.8f, 0.0f,  0.0f,  0.8f, 0.0f
};
const float sideFacingPositions[] = {
  -0.25f, -0.25f, 0.0f,  0.25f, -0.25f, 0.0f,
   0.25f,  0.25f, 0.0f, -0.25f,  0.25f, 0.0f
};

SoRenderCommand baseCommand(const float * positions)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = positions;
  command.geometry.indices = quadIndices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.featureFlags = SO_FEAT_BASE_COLOR;
  command.state.depth.enabled = TRUE;
  command.state.depth.writeEnabled = TRUE;
  command.state.depth.func = SO_DEPTH_LEQUAL;
  return command;
}

void enableAlphaBlend(SoRenderCommand & command)
{
  command.pass = SO_RENDERPASS_TRANSPARENT;
  command.state.blend.enabled = TRUE;
  command.state.blend.srcRGBFactor = SO_BLEND_FACTOR_SRC_ALPHA;
  command.state.blend.dstRGBFactor = SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  command.state.blend.srcAlphaFactor = SO_BLEND_FACTOR_ONE;
  command.state.blend.dstAlphaFactor = SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
}

const uint8_t * pixelAt(const std::vector<uint8_t> & pixels, int x, int y)
{
  return &pixels[static_cast<size_t>(y * 64 + x) * 4];
}

bool check(bool condition, const char * message)
{
  if (!condition) std::cerr << "FAIL: " << message << std::endl;
  return condition;
}

bool testTextureWrap(RenderFixture & fixture)
{
  const unsigned char pixels[] = {
    255, 0, 0, 255,
    0, 255, 0, 255
  };
  const float texcoords[] = {
    1.25f, 0.5f, 0.0f, 0.0f, 1.25f, 0.5f, 0.0f, 0.0f,
    1.25f, 0.5f, 0.0f, 0.0f, 1.25f, 0.5f, 0.0f, 0.0f
  };
  SoDrawList drawlist;
  SoRenderCommand command = baseCommand(quadPositions);
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 4;
  command.material.texture.pixels = pixels;
  command.material.texture.width = 2;
  command.material.texture.height = 1;
  command.material.texture.numComponents = 4;
  command.material.texture.minFilter = SO_TEXTURE_FILTER_NEAREST;
  command.material.texture.magFilter = SO_TEXTURE_FILTER_NEAREST;
  command.material.texture.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;

  command.material.texture.wrapS = SO_TEXTURE_WRAP_REPEAT;
  drawlist.addCommand(command);
  std::vector<uint8_t> repeated = fixture.render(drawlist, SbVec4f(0, 0, 0, 1));
  drawlist.clear();
  command.material.texture.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  drawlist.addCommand(command);
  std::vector<uint8_t> clamped = fixture.render(drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * repeatPixel = pixelAt(repeated, 32, 32);
  const uint8_t * clampPixel = pixelAt(clamped, 32, 32);
  return check(repeatPixel[0] > 200 && repeatPixel[1] < 50 &&
               clampPixel[1] > 200 && clampPixel[0] < 50,
               "retained texture wrap state did not affect sampling");
}

bool testTextureAlphaOnce(RenderFixture & fixture)
{
  const unsigned char pixels[] = { 255, 0, 0, 128 };
  const float texcoords[] = {
    0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f
  };
  SoRenderCommand command = baseCommand(quadPositions);
  command.material.diffuse = SbVec4f(1, 1, 1, 0.5f);
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 4;
  command.material.texture.pixels = pixels;
  command.material.texture.width = 1;
  command.material.texture.height = 1;
  command.material.texture.numComponents = 4;
  command.material.textureAlphaIncludesOpacity = false;
  enableAlphaBlend(command);
  SoDrawList drawlist;
  drawlist.addCommand(command);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 1, 1));
  const uint8_t * pixel = pixelAt(rendered, 32, 32);
  return check(pixel[0] >= 50 && pixel[0] <= 80 &&
               pixel[2] >= 175 && pixel[2] <= 205,
               "texture and material opacity were not composed exactly once");
}

bool testTextureModels(RenderFixture & fixture)
{
  const unsigned char pixels[] = { 128, 128, 128, 128 };
  const float texcoords[] = {
    0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f
  };
  const SoTextureModel models[] = {
    SO_TEXTURE_MODEL_MODULATE, SO_TEXTURE_MODEL_DECAL,
    SO_TEXTURE_MODEL_BLEND, SO_TEXTURE_MODEL_REPLACE
  };
  const uint8_t expected[][3] = {
    { 100, 25, 12 }, { 164, 89, 76 }, { 100, 152, 12 }, { 128, 128, 128 }
  };

  for (int i = 0; i < 4; ++i) {
    SoRenderCommand command = baseCommand(quadPositions);
    command.material.diffuse = SbVec4f(200.0f / 255.0f,
                                        50.0f / 255.0f,
                                        25.0f / 255.0f, 1.0f);
    command.geometry.texcoords = texcoords;
    command.geometry.texcoordStride = sizeof(float) * 4;
    command.material.texture.pixels = pixels;
    command.material.texture.width = 1;
    command.material.texture.height = 1;
    command.material.texture.numComponents = 4;
    command.material.texture.model = models[i];
    command.material.texture.blendColor = SbVec4f(0, 1, 0, 1);

    SoDrawList drawlist;
    drawlist.addCommand(command);
    const std::vector<uint8_t> rendered = fixture.render(
      drawlist, SbVec4f(0, 0, 0, 1));
    const uint8_t * pixel = pixelAt(rendered, 32, 32);
    if (!check(std::abs(static_cast<int>(pixel[0]) - expected[i][0]) < 25 &&
               std::abs(static_cast<int>(pixel[1]) - expected[i][1]) < 25 &&
               std::abs(static_cast<int>(pixel[2]) - expected[i][2]) < 25,
               "retained texture model semantics were not executed")) {
      return false;
    }
  }
  return true;
}

bool testTexturedLightingComposition(RenderFixture & fixture)
{
  const unsigned char pixels[] = { 255, 255, 255, 255 };
  const float texcoords[] = {
    0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f
  };
  SoRenderCommand command = baseCommand(quadPositions);
  command.material.featureFlags = 0;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
  command.material.diffuse = SbVec4f(0.8f, 0.8f, 0.8f, 1.0f);
  command.material.ambient = SbVec4f(0, 0, 0, 1);
  command.material.specular = SbVec4f(0, 0, 0, 1);
  command.material.emissive = SbVec4f(0, 0, 0, 1);
  command.material.texture.pixels = pixels;
  command.material.texture.width = 1;
  command.material.texture.height = 1;
  command.material.texture.numComponents = 4;
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 4;

  SoLightingData lighting;
  lighting.ambient.setValue(0, 0, 0);
  SoLightData directional;
  directional.color.setValue(0.25f, 0.25f, 0.25f);
  directional.direction.setValue(0, 0, 1);
  lighting.lights.push_back(directional);
  SoDrawList drawlist;
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  drawlist.addCommand(command);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * pixel = pixelAt(rendered, 32, 32);
  return check(pixel[0] >= 40 && pixel[0] <= 65 &&
               pixel[1] >= 40 && pixel[1] <= 65 &&
               pixel[2] >= 40 && pixel[2] <= 65,
               "textured geometry bypassed retained lighting");
}

bool testEmissiveIsIndependent(RenderFixture & fixture)
{
  SoRenderCommand command = baseCommand(quadPositions);
  command.material.featureFlags = 0;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
  command.material.diffuse = SbVec4f(0.8f, 0.8f, 0.8f, 1.0f);
  command.material.ambient = SbVec4f(0, 0, 0, 1);
  command.material.specular = SbVec4f(0, 0, 0, 1);
  command.material.emissive = SbVec4f(0.2f, 0.2f, 0.2f, 1.0f);
  SoLightingData lighting;
  lighting.ambient.setValue(0, 0, 0);
  SoDrawList drawlist;
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  drawlist.addCommand(command);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * pixel = pixelAt(rendered, 32, 32);
  return check(pixel[0] >= 45 && pixel[0] <= 60 &&
               pixel[1] >= 45 && pixel[1] <= 60 &&
               pixel[2] >= 45 && pixel[2] <= 60,
               "emissive material was folded into diffuse twice");
}

bool testTwoSidedLightingUsesFacing(RenderFixture & fixture)
{
  const float normals[] = {
    1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0
  };
  SoRenderCommand command = baseCommand(sideFacingPositions);
  command.material.featureFlags = 0;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
  command.material.diffuse = SbVec4f(1, 1, 1, 1);
  command.material.ambient = SbVec4f(0, 0, 0, 1);
  command.material.specular = SbVec4f(0, 0, 0, 1);
  command.material.emissive = SbVec4f(0, 0, 0, 1);
  command.material.twoSidedLighting = true;
  command.geometry.normals = normals;
  command.geometry.normalCount = 4;
  command.modelMatrix.setTranslate(SbVec3f(0.5f, 0, 0));
  SoLightingData lighting;
  lighting.ambient.setValue(0, 0, 0);
  SoLightData directional;
  directional.direction.setValue(-1, 0, 0);
  lighting.lights.push_back(directional);
  SoDrawList drawlist;
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  drawlist.addCommand(command);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * pixel = pixelAt(rendered, 48, 32);
  return check(pixel[0] > 220 && pixel[1] > 220 && pixel[2] > 220,
               "two-sided lighting did not use the actual viewer-facing side");
}

bool testExecutorLightLimit(RenderFixture & fixture)
{
  SoRenderCommand command = baseCommand(quadPositions);
  command.material.featureFlags = 0;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
  command.material.diffuse = SbVec4f(1, 1, 1, 1);
  command.material.ambient = SbVec4f(0, 0, 0, 1);
  command.material.specular = SbVec4f(0, 0, 0, 1);
  command.material.emissive = SbVec4f(0, 0, 0, 1);
  SoLightingData lighting;
  lighting.ambient.setValue(0, 0, 0);
  lighting.lights.resize(9);
  for (int i = 0; i < 8; ++i) {
    lighting.lights[static_cast<size_t>(i)].color.setValue(0, 0, 0);
  }
  lighting.lights[8].direction.setValue(0, 0, 1);
  SoDrawList drawlist;
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  drawlist.addCommand(command);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * pixel = pixelAt(rendered, 32, 32);
  return check(pixel[0] < 10 && pixel[1] < 10 && pixel[2] < 10,
               "GL executor uploaded more lights than its declared shader cap");
}

bool testTransparentDepthWriteContract(RenderFixture & fixture)
{
  SoRenderCommand command = baseCommand(quadPositions);
  command.pass = SO_RENDERPASS_TRANSPARENT;
  command.material.featureFlags = SO_FEAT_BASE_COLOR;
  command.material.diffuse = SbVec4f(1, 0, 0, 0.5f);
  command.state.depth.writeEnabled = TRUE;
  enableAlphaBlend(command);
  SoDrawList drawlist;
  drawlist.addCommand(command);
  fixture.render(drawlist, SbVec4f(0, 0, 0, 1));
  GLboolean depthWrite = GL_FALSE;
  glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWrite);
  return check(depthWrite == GL_FALSE,
               "transparent retained commands unexpectedly wrote depth");
}

bool testMaterialTransparency(RenderFixture & fixture)
{
  SoRenderCommand left = baseCommand(halfLeftPositions);
  left.material.diffuse = SbVec4f(1, 0, 0, 0.25f);
  enableAlphaBlend(left);
  SoRenderCommand right = baseCommand(halfRightPositions);
  right.material.diffuse = SbVec4f(0, 1, 0, 0.75f);
  enableAlphaBlend(right);
  SoDrawList drawlist;
  drawlist.addCommand(left);
  drawlist.addCommand(right);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * leftPixel = pixelAt(rendered, 16, 32);
  const uint8_t * rightPixel = pixelAt(rendered, 48, 32);
  return check(leftPixel[0] >= 50 && leftPixel[0] <= 80 &&
               rightPixel[1] >= 175 && rightPixel[1] <= 205,
               "per-material transparency was not preserved per command");
}

bool testNonUniformScaleLighting(RenderFixture & fixture)
{
  const float normals[] = {
    0.7071067f, 0.0f, 0.7071067f,
    0.7071067f, 0.0f, 0.7071067f,
    0.7071067f, 0.0f, 0.7071067f,
    0.7071067f, 0.0f, 0.7071067f
  };
  SoRenderCommand command = baseCommand(quadPositions);
  command.material.featureFlags = 0;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
  command.material.diffuse = SbVec4f(1, 1, 1, 1);
  command.material.ambient = SbVec4f(0, 0, 0, 1);
  command.material.specular = SbVec4f(0, 0, 0, 1);
  command.material.emissive = SbVec4f(0, 0, 0, 1);
  command.material.shininess = 0.0f;
  command.geometry.normals = normals;
  command.geometry.normalCount = 4;
  SoLightingData lighting;
  lighting.ambient.setValue(0, 0, 0);
  SoLightData directional;
  directional.type = SO_LIGHT_DIRECTIONAL;
  directional.direction.setValue(0, 0, 1);
  lighting.lights.push_back(directional);
  SoDrawList drawlist;
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  command.modelMatrix.setScale(SbVec3f(4, 1, 1));
  drawlist.addCommand(command);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * pixel = pixelAt(rendered, 32, 32);
  return check(pixel[0] > 220 && pixel[1] > 220 && pixel[2] > 220,
               "non-uniform model scale corrupted retained normal lighting");
}

bool testDepth(RenderFixture & fixture)
{
  SoRenderCommand back = baseCommand(quadPositions);
  back.material.diffuse = SbVec4f(0, 0, 1, 1);
  back.modelMatrix.setTranslate(SbVec3f(0, 0, 0.5f));
  SoRenderCommand front = baseCommand(quadPositions);
  front.material.diffuse = SbVec4f(1, 0, 0, 1);
  front.modelMatrix.setTranslate(SbVec3f(0, 0, 0.0f));
  SoDrawList drawlist;
  drawlist.addCommand(back);
  drawlist.addCommand(front);
  const std::vector<uint8_t> rendered = fixture.render(
    drawlist, SbVec4f(0, 0, 0, 1));
  const uint8_t * pixel = pixelAt(rendered, 32, 32);
  return check(pixel[0] > 200 && pixel[1] < 50 && pixel[2] < 50,
               "retained depth testing did not select the nearer command");
}

bool testAlphaTest(RenderFixture & fixture)
{
  SoRenderCommand command = baseCommand(quadPositions);
  command.material.diffuse = SbVec4f(1, 0, 0, 0.5f);
  command.state.alphaTest.policy = SO_ALPHA_TEST_POLICY_EXPLICIT;
  command.state.alphaTest.function = SO_ALPHA_TEST_GREATER;
  command.state.alphaTest.reference = 0.75f;
  SoDrawList drawlist;
  drawlist.addCommand(command);
  const std::vector<uint8_t> rejected = fixture.render(
    drawlist, SbVec4f(0, 0, 1, 1));
  const uint8_t * rejectedPixel = pixelAt(rejected, 32, 32);
  if (!check(rejectedPixel[2] > 200 && rejectedPixel[0] < 50,
             "alpha-test rejection did not discard the fragment")) {
    return false;
  }

  drawlist.clear();
  command.state.alphaTest.function = SO_ALPHA_TEST_LESS;
  drawlist.addCommand(command);
  const std::vector<uint8_t> accepted = fixture.render(
    drawlist, SbVec4f(0, 0, 1, 1));
  const uint8_t * acceptedPixel = pixelAt(accepted, 32, 32);
  return check(acceptedPixel[0] > 200 && acceptedPixel[2] < 50,
               "alpha-test acceptance did not preserve the fragment");
}

} // namespace

int main()
{
  setEnvironment("COIN_EGL", "1");
  setEnvironment("EGL_PLATFORM", "surfaceless");
  setEnvironment("COIN_EGL_CORE_PROFILE", "1");
  SoDB::init();
  RenderFixture fixture;
  if (!fixture.initialize()) {
    SoDB::finish();
    return skip("core EGL retained-material context is unavailable");
  }

  int result = 0;
  if (!testTextureWrap(fixture)) result = 1;
  if (!testTextureAlphaOnce(fixture)) result = 1;
  if (!testTextureModels(fixture)) result = 1;
  if (!testTexturedLightingComposition(fixture)) result = 1;
  if (!testEmissiveIsIndependent(fixture)) result = 1;
  if (!testTwoSidedLightingUsesFacing(fixture)) result = 1;
  if (!testExecutorLightLimit(fixture)) result = 1;
  if (!testTransparentDepthWriteContract(fixture)) result = 1;
  if (!testMaterialTransparency(fixture)) result = 1;
  if (!testNonUniformScaleLighting(fixture)) result = 1;
  if (!testDepth(fixture)) result = 1;
  if (!testAlphaTest(fixture)) result = 1;
  fixture.shutdown();
  SoDB::finish();
  return result;
}
