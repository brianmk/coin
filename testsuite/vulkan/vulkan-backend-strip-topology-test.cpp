// testsuite/vulkan-backend-strip-topology-test.cpp
//
// Verifies triangle-strip and line-strip primitive topologies are rasterized
// by distinct pipelines in the same frame.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  // Triangle strip: two triangles covering the whole viewport.
  static const float strip[] = {
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f, 0.0f
  };
  // Line strip: an L-shaped path across the lower half.
  static const float lineStrip[] = {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
     0.5f,  0.5f, 0.0f
  };

  SoDrawList drawlist;

  SoRenderCommand tris;
  tris.modelMatrix.makeIdentity();
  tris.geometry.topology = SO_TOPOLOGY_TRIANGLE_STRIP;
  tris.geometry.vertexCount = 4;
  tris.geometry.positions = strip;
  tris.geometry.vertexStride = sizeof(float) * 3;
  tris.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  drawlist.addCommand(tris);

  SoRenderCommand lines;
  lines.modelMatrix.makeIdentity();
  lines.geometry.topology = SO_TOPOLOGY_LINE_STRIP;
  lines.geometry.vertexCount = 3;
  lines.geometry.positions = lineStrip;
  lines.geometry.vertexStride = sizeof(float) * 3;
  lines.material.diffuse = SbVec4f(0.0f, 0.0f, 1.0f, 1.0f);
  drawlist.addCommand(lines);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: strip-topology render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const int green = countNear(pixels, 0, 255, 0);
  const int blue = countNear(pixels, 0, 0, 255);

  if (green < 100) {
    std::cerr << "FAIL: triangle strip rasterized too few green pixels (" << green
              << ")" << std::endl;
    ++failures;
  }
  if (blue == 0) {
    std::cerr << "FAIL: line strip rasterized no blue pixels" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
