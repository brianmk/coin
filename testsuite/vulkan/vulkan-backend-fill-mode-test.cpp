// testsuite/vulkan-backend-fill-mode-test.cpp
//
// Verifies the rasterizer fill-mode pipeline variants: wireframe (lines) and
// points render only edges/vertices, leaving the triangle interior untouched.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  // Large triangle spanning most of the viewport.
  static const float triangle[] = {
    -0.95f, -0.95f, 0.0f,
     0.95f, -0.95f, 0.0f,
     0.0f,  0.95f, 0.0f
  };
  const SoRenderParams params = harness.renderParams();

  // --- Wireframe ----------------------------------------------------------
  {
    SoDrawList drawlist;
    SoRenderCommand command = makeTriangle(triangle);
    command.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
    command.state.raster.fillMode = 1; // lines
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: wireframe render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    const int green = countNear(pixels, 0, 255, 0);
    if (green == 0) {
      std::cerr << "FAIL: wireframe produced no edge pixels" << std::endl;
      ++failures;
    }
    // Interior centroid must stay clear in wireframe.
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 0, 0)) {
      std::cerr << "FAIL: wireframe rasterized the triangle interior" << std::endl;
      ++failures;
    }
  }

  // --- Points -------------------------------------------------------------
  {
    SoDrawList drawlist;
    SoRenderCommand command = makeTriangle(triangle);
    command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
    command.state.raster.fillMode = 2; // points
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: point-mode render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    const int red = countNear(pixels, 255, 0, 0);
    if (red == 0) {
      std::cerr << "FAIL: point-mode produced no vertex pixels" << std::endl;
      ++failures;
    }
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 0, 0)) {
      std::cerr << "FAIL: point-mode rasterized the triangle interior" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
