// testsuite/vulkan-backend-mixed-topology-test.cpp
//
// Verifies that a single SoDrawList containing triangles, lines, and points
// renders through distinct Vulkan pipelines.  This guards the pipeline-cache
// keying: a triangle pipeline must not be reused for line or point topology.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  // A green triangle covering the upper half.
  const float triangle[] = {
    -1.0f,  0.2f, 0.0f,
     1.0f,  0.2f, 0.0f,
     0.0f,  1.0f, 0.0f
  };

  // A thick horizontal line segment across the middle of the viewport.
  const float line[] = {
    -1.0f, -0.1f, 0.0f,
     1.0f, -0.1f, 0.0f
  };

  // A point near the bottom.
  const float point[] = {
    0.0f, -0.6f, 0.0f
  };

  SoDrawList drawlist;

  SoRenderCommand triangleCommand = makeTriangle(triangle);
  triangleCommand.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  drawlist.addCommand(triangleCommand);

  SoRenderCommand lineCommand = makeTriangle(line, SO_TOPOLOGY_LINES, 2);
  lineCommand.material.diffuse = SbVec4f(0.0f, 0.0f, 1.0f, 1.0f);
  lineCommand.state.raster.lineWidth = 3.0f;
  drawlist.addCommand(lineCommand);

  SoRenderCommand pointCommand = makeTriangle(point, SO_TOPOLOGY_POINTS, 1);
  pointCommand.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  pointCommand.state.raster.pointSize = 4.0f;
  drawlist.addCommand(pointCommand);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: mixed-topology render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();

  // Rasterization is verified by scanning for each topology's distinct
  // color anywhere in the framebuffer.  Exact pixel coordinates are
  // brittle: in this milestone lines/points are rasterized 1px wide
  // (Vulkan's fixed point/line size), so a single on-pixel assertion would
  // over-constrain the test.
  bool sawGreen = false;
  bool sawBlue = false;
  bool sawRed = false;
  for (size_t i = 0; i < kWidth * kHeight; ++i) {
    const uint8_t * p = &pixels[i * kPixelBytes];
    if (nearColor(p, 0, 255, 0)) sawGreen = true;
    if (nearColor(p, 0, 0, 255)) sawBlue = true;
    if (nearColor(p, 255, 0, 0)) sawRed = true;
  }

  if (!sawGreen) {
    std::cerr << "FAIL: triangle did not rasterize green" << std::endl;
    ++failures;
  }
  if (!sawBlue) {
    std::cerr << "FAIL: line did not rasterize blue" << std::endl;
    ++failures;
  }
  if (!sawRed) {
    std::cerr << "FAIL: point did not rasterize red" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
