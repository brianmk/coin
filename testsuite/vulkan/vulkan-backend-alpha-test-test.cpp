// testsuite/vulkan-backend-alpha-test-test.cpp
//
// Verifies the fragment alpha-test path: NEVER always discards, GREATER
// discards low alpha, LESS keeps low alpha, and NONE keeps everything.

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

SoRenderCommand alphaQuad(float alpha, SoAlphaTestFunction function,
                          float reference)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, alpha);
  command.state.alphaTest.function = function;
  command.state.alphaTest.reference = reference;
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
  const uint8_t * center = nullptr;

  // NEVER: discard everything -> cleared black.
  {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(1.0f, SO_ALPHA_TEST_NEVER, 0.5f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: NEVER alpha test render failed" << std::endl;
      ++failures;
    }
    center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 0, 0)) {
      std::cerr << "FAIL: NEVER alpha test did not discard" << std::endl;
      ++failures;
    }
  }

  // GREATER with alpha 0.25 < 0.5: discard.
  {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(0.25f, SO_ALPHA_TEST_GREATER, 0.5f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: GREATER alpha test render failed" << std::endl;
      ++failures;
    }
    center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 0, 0)) {
      std::cerr << "FAIL: GREATER alpha test did not discard low alpha" << std::endl;
      ++failures;
    }
  }

  // LESS with alpha 0.25 < 0.5: keep (blended to ~64 red on black).
  {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(0.25f, SO_ALPHA_TEST_LESS, 0.5f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: LESS alpha test render failed" << std::endl;
      ++failures;
    }
    center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 64, 0, 0)) {
      std::cerr << "FAIL: LESS alpha test discarded matching alpha" << std::endl;
      ++failures;
    }
  }

  // NONE with opaque alpha: keep fully.
  {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(1.0f, SO_ALPHA_TEST_NONE, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: NONE alpha test render failed" << std::endl;
      ++failures;
    }
    center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: NONE alpha test did not keep the fragment" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
