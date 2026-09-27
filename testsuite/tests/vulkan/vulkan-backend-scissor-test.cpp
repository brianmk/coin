// testsuite/vulkan-backend-scissor-test.cpp
//
// Verifies per-command scissor rectangles are honored: a fullscreen quad with
// scissor enabled only rasterizes inside the scissor rect, leaving the clear
// color everywhere else.

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
  const SoRenderParams params = harness.renderParams();

  // Center 16x16 rect, matching the extent produced by NDC [-1,1] viewport.
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
  command.state.raster.scissorEnabled = TRUE;
  command.state.raster.scissorX = 8;
  command.state.raster.scissorY = 8;
  command.state.raster.scissorWidth = 16;
  command.state.raster.scissorHeight = 16;

  SoDrawList drawlist;
  drawlist.addCommand(command);

  if (!harness.backend.render(drawlist, params)) {
    std::cerr << "FAIL: scissor render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();

  if (!nearColor(pixelAt(pixels, 16, 16), 255, 0, 0)) {
    std::cerr << "FAIL: scissor did not draw inside the rect" << std::endl;
    ++failures;
  }
  if (!nearColor(pixelAt(pixels, 0, 0), 0, 0, 0)) {
    std::cerr << "FAIL: scissor drew outside the rect (top-left)" << std::endl;
    ++failures;
  }
  if (!nearColor(pixelAt(pixels, 31, 31), 0, 0, 0)) {
    std::cerr << "FAIL: scissor drew outside the rect (bottom-right)"
              << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
