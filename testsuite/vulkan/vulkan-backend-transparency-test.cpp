// testsuite/vulkan-backend-transparency-test.cpp
//
// Verifies transparent-pass rendering: a translucent quad drawn over an opaque
// background must blend (alpha < 1) rather than replace the background.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float quad[] = {
  -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  SoDrawList drawlist;

  // Opaque red quad covering the whole viewport.
  SoRenderCommand background;
  background.modelMatrix.makeIdentity();
  background.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  background.geometry.vertexCount = 4;
  background.geometry.indexCount = 6;
  background.geometry.positions = quad;
  background.geometry.indices = indices;
  background.geometry.vertexStride = sizeof(float) * 3;
  background.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  drawlist.addCommand(background);

  // Translucent green quad covering the whole viewport.  With source-alpha
  // blending this must leave a green-tinted red rather than pure green.
  SoRenderCommand overlay;
  overlay.modelMatrix.makeIdentity();
  overlay.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  overlay.geometry.vertexCount = 4;
  overlay.geometry.indexCount = 6;
  overlay.geometry.positions = quad;
  overlay.geometry.indices = indices;
  overlay.geometry.vertexStride = sizeof(float) * 3;
  overlay.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 0.5f);
  overlay.pass = SO_RENDERPASS_TRANSPARENT;
  drawlist.addCommand(overlay);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: transparency render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const uint8_t * center = pixelAt(pixels, 16, 16);

  // Expected blend: 0.5 * green + (1 - 0.5) * red = (0.5, 0.5, 0) in linear
  // float terms → ~(127, 127, 0) in 8-bit.  Allow generous tolerance.
  if (!nearColor(center, 127, 127, 0)) {
    std::cerr << "FAIL: translucent overlay produced B,G,R = "
              << static_cast<int>(center[0]) << ","
              << static_cast<int>(center[1]) << ","
              << static_cast<int>(center[2]) << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
