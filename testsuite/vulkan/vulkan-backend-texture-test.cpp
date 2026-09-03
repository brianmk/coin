// testsuite/vulkan-backend-texture-test.cpp
//
// Exercises the embedded-texture path: a REPLACE-textured quad must render the
// texel color directly, and a MODULATE-textured quad must multiply the base
// color by the texel color.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float leftQuad[] = {
  -1.0f, -1.0f, 0.0f,
   0.0f, -1.0f, 0.0f,
   0.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const float rightQuad[] = {
   0.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
   0.0f,  1.0f, 0.0f
};
const float texcoords[] = {
  0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

// 2x2 solid-red RGBA texture.
const unsigned char redTexel[16] = {
  255, 0, 0, 255,  255, 0, 0, 255,
  255, 0, 0, 255,  255, 0, 0, 255
};
// 2x2 solid-green RGBA texture.
const unsigned char greenTexel[16] = {
  0, 255, 0, 255,  0, 255, 0, 255,
  0, 255, 0, 255,  0, 255, 0, 255
};

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  auto makeTexturedQuad = [](const float * positions, const unsigned char * texel,
                             SoTextureModel model) {
    SoRenderCommand command;
    command.modelMatrix.makeIdentity();
    command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
    command.geometry.vertexCount = 4;
    command.geometry.indexCount = 6;
    command.geometry.positions = positions;
    command.geometry.texcoords = texcoords;
    command.geometry.texcoordStride = sizeof(float) * 2;
    command.geometry.indices = indices;
    command.geometry.vertexStride = sizeof(float) * 3;
    command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
    command.material.texture.pixels = texel;
    command.material.texture.width = 2;
    command.material.texture.height = 2;
    command.material.texture.numComponents = 4;
    command.material.texture.model = model;
    return command;
  };

  SoDrawList drawlist;
  drawlist.addCommand(makeTexturedQuad(leftQuad, redTexel,
                                       SO_TEXTURE_MODEL_REPLACE));
  drawlist.addCommand(makeTexturedQuad(rightQuad, greenTexel,
                                       SO_TEXTURE_MODEL_MODULATE));

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: texture render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();

  // Left half: REPLACE red texture.
  if (!nearColor(pixelAt(pixels, 8, 16), 255, 0, 0)) {
    const uint8_t * p = pixelAt(pixels, 8, 16);
    std::cerr << "FAIL: REPLACE texture produced (B,G,R)=("
              << static_cast<int>(p[0]) << "," << static_cast<int>(p[1]) << ","
              << static_cast<int>(p[2]) << ")" << std::endl;
    ++failures;
  }

  // Right half: MODULATE white * green = green.
  if (!nearColor(pixelAt(pixels, 24, 16), 0, 255, 0)) {
    const uint8_t * p = pixelAt(pixels, 24, 16);
    std::cerr << "FAIL: MODULATE texture produced (B,G,R)=("
              << static_cast<int>(p[0]) << "," << static_cast<int>(p[1]) << ","
              << static_cast<int>(p[2]) << ")" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
