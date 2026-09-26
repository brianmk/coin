// testsuite/vulkan-backend-clear-test.cpp
//
// Verifies the SO_PARAM_CLEAR_WINDOW / SO_PARAM_CLEAR_DEPTH flags are
// honored independently across frames.

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

SoRenderCommand redQuad()
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  command.state.depth.enabled = FALSE;
  command.state.depth.writeEnabled = FALSE;
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
  SoRenderParams params = harness.renderParams();

  // Empty list with CLEAR_WINDOW|CLEAR_DEPTH -> black.
  {
    SoDrawList drawlist;
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: empty clear render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 0, 0)) {
      std::cerr << "FAIL: window clear did not clear to black" << std::endl;
      ++failures;
    }
  }

  // Draw red with no clear flags.
  {
    SoDrawList drawlist;
    drawlist.addCommand(redQuad());
    SoRenderParams noClear = params;
    noClear.flags = 0;
    if (!harness.backend.render(drawlist, noClear)) {
      std::cerr << "FAIL: no-clear red render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: red quad did not render" << std::endl;
      ++failures;
    }
  }

  // Empty list with only CLEAR_DEPTH -> color must persist (still red).
  {
    SoDrawList drawlist;
    SoRenderParams depthOnly = params;
    depthOnly.flags = SO_PARAM_CLEAR_DEPTH;
    if (!harness.backend.render(drawlist, depthOnly)) {
      std::cerr << "FAIL: depth-only clear render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: depth-only clear erased the color buffer"
                << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
