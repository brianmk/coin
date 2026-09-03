// testsuite/vulkan-backend-texture-models-test.cpp
//
// Exercises the DECAL and BLEND texture models alongside REPLACE to verify the
// texture-model switch in the fragment shader.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float quad[] = {
  -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const float texcoords[] = {
  0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

// 1x1 red RGBA texel, alpha 0.
const unsigned char transparentRed[4] = {255, 0, 0, 0};
// 1x1 red RGBA texel, alpha 255.
const unsigned char opaqueRed[4] = {255, 0, 0, 255};

SoRenderCommand texturedQuad(const unsigned char * texel,
                             SoTextureModel model,
                             SbVec4f diffuse,
                             SbVec4f blendColor)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 2;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = diffuse;
  command.material.texture.pixels = texel;
  command.material.texture.width = 1;
  command.material.texture.height = 1;
  command.material.texture.numComponents = 4;
  command.material.texture.model = model;
  command.material.texture.blendColor = blendColor;
  return command;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;
  const SoRenderParams params = harness.renderParams();

  // DECAL with a fully transparent texel leaves the base color unchanged.
  {
    SoDrawList drawlist;
    drawlist.addCommand(texturedQuad(transparentRed, SO_TEXTURE_MODEL_DECAL,
                                     SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                     SbVec4f(0.0f, 0.0f, 0.0f, 1.0f)));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: DECAL texture render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 255, 255)) {
      std::cerr << "FAIL: DECAL did not preserve the base color for "
                   "transparent texels" << std::endl;
      ++failures;
    }
  }

  // REPLACE with an opaque red texel replaces the base color.
  {
    SoDrawList drawlist;
    drawlist.addCommand(texturedQuad(opaqueRed, SO_TEXTURE_MODEL_REPLACE,
                                     SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                     SbVec4f(0.0f, 0.0f, 0.0f, 1.0f)));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: REPLACE texture render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: REPLACE did not replace the base color" << std::endl;
      ++failures;
    }
  }

  // BLEND with a red texel and green blend color mixes to cyan.
  {
    SoDrawList drawlist;
    drawlist.addCommand(texturedQuad(opaqueRed, SO_TEXTURE_MODEL_BLEND,
                                     SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                     SbVec4f(0.0f, 1.0f, 0.0f, 1.0f)));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: BLEND texture render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 255)) {
      std::cerr << "FAIL: BLEND did not mix the blend color by texel red"
                << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
