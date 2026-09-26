// testsuite/vulkan-backend-mixed-topology-test.cpp
//
// Verifies that distinct primitive topologies and rasterizer fill modes
// render through distinct Vulkan pipelines: triangles/lines/points mixed in
// one draw list, triangle/line strips, and wireframe/point fill modes
// leaving the triangle interior untouched.  This guards the pipeline-cache
// keying: a triangle pipeline must not be reused for line or point
// topology, and a fill variant must not leak into the solid fill.  (Merged
// from the former mixed-topology, strip-topology, and fill-mode tests.)

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

// Large triangle spanning most of the viewport (fill-mode cases).
static const float bigTriangle[] = {
  -0.95f, -0.95f, 0.0f,
   0.95f, -0.95f, 0.0f,
   0.0f,  0.95f, 0.0f
};

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  CaseRunner cases;

  cases.add("triangles, lines, and points use distinct pipelines",
            [&harness] {
              // A green triangle covering the upper half.
              static const float triangle[] = {
                -1.0f,  0.2f, 0.0f,
                 1.0f,  0.2f, 0.0f,
                 0.0f,  1.0f, 0.0f
              };
              // A thick horizontal line segment across the middle.
              static const float line[] = {
                -1.0f, -0.1f, 0.0f,
                 1.0f, -0.1f, 0.0f
              };
              // A point near the bottom.
              static const float point[] = {0.0f, -0.6f, 0.0f};

              SoDrawList drawlist;
              SoRenderCommand triangleCommand = makeTriangle(triangle);
              triangleCommand.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f,
                                                         1.0f);
              drawlist.addCommand(triangleCommand);

              SoRenderCommand lineCommand =
                makeTriangle(line, SO_TOPOLOGY_LINES, 2);
              lineCommand.material.diffuse = SbVec4f(0.0f, 0.0f, 1.0f, 1.0f);
              lineCommand.state.raster.lineWidth = 3.0f;
              drawlist.addCommand(lineCommand);

              SoRenderCommand pointCommand =
                makeTriangle(point, SO_TOPOLOGY_POINTS, 1);
              pointCommand.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
              pointCommand.state.raster.pointSize = 4.0f;
              drawlist.addCommand(pointCommand);

              VK_CHECK(harness.backend.render(drawlist,
                                              harness.renderParams()),
                       "render failed");
              const std::vector<uint8_t> pixels = harness.readback();

              // Rasterization is verified by scanning for each topology's
              // distinct color anywhere in the framebuffer.  Exact pixel
              // coordinates are brittle: line/point width is
              // driver-dependent (wide lines render at the requested width,
              // points at the requested point size), so a single on-pixel
              // assertion would over-constrain the test.
              bool sawGreen = false;
              bool sawBlue = false;
              bool sawRed = false;
              for (size_t i = 0; i < kWidth * kHeight; ++i) {
                const uint8_t * p = &pixels[i * kPixelBytes];
                if (nearColor(p, 0, 255, 0)) sawGreen = true;
                if (nearColor(p, 0, 0, 255)) sawBlue = true;
                if (nearColor(p, 255, 0, 0)) sawRed = true;
              }
              VK_CHECK(sawGreen, "triangle did not rasterize green");
              VK_CHECK(sawBlue, "line did not rasterize blue");
              VK_CHECK(sawRed, "point did not rasterize red");
            });

  cases.add("triangle strip and line strip use distinct pipelines",
            [&harness] {
              // Triangle strip: two triangles covering the whole viewport,
              // placed behind the line strip (opaque draw order is not
              // insertion order, so the nearer line strip must win on depth).
              static const float strip[] = {
                -1.0f, -1.0f, 0.8f,
                 1.0f, -1.0f, 0.8f,
                -1.0f,  1.0f, 0.8f,
                 1.0f,  1.0f, 0.8f
              };
              // Line strip: an L-shaped path across the lower half.
              static const float lineStrip[] = {
                -0.5f, -0.5f, 0.2f,
                 0.5f, -0.5f, 0.2f,
                 0.5f,  0.5f, 0.2f
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

              VK_CHECK(harness.backend.render(drawlist,
                                              harness.renderParams()),
                       "render failed");
              const std::vector<uint8_t> pixels = harness.readback();
              const int green = countNear(pixels, 0, 255, 0);
              const int blue = countNear(pixels, 0, 0, 255);
              VK_CHECK(green >= 100, "triangle strip rasterized too few "
                     "green pixels (" << green << ")");
              VK_CHECK(blue > 0, "line strip rasterized no blue pixels");
            });

  cases.add("wireframe fill leaves the triangle interior empty",
            [&harness] {
              SoDrawList drawlist;
              SoRenderCommand command = makeTriangle(bigTriangle);
              command.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
              command.state.raster.fillMode = 1; // lines
              drawlist.addCommand(command);

              VK_CHECK(harness.backend.render(drawlist,
                                              harness.renderParams()),
                       "render failed");
              const std::vector<uint8_t> pixels = harness.readback();
              const int green = countNear(pixels, 0, 255, 0);
              VK_CHECK(green > 0, "wireframe produced no edge pixels");
              // Interior centroid must stay clear in wireframe.
              VK_CHECK(nearColor(pixelAt(pixels, 16, 16), 0, 0, 0),
                       "wireframe rasterized the triangle interior: "
                       << describePixel(pixelAt(pixels, 16, 16)));
            });

  cases.add("point fill leaves the triangle interior empty", [&harness] {
    SoDrawList drawlist;
    SoRenderCommand command = makeTriangle(bigTriangle);
    command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
    command.state.raster.fillMode = 2; // points
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const std::vector<uint8_t> pixels = harness.readback();
    const int red = countNear(pixels, 255, 0, 0);
    VK_CHECK(red > 0, "point-mode produced no vertex pixels");
    VK_CHECK(nearColor(pixelAt(pixels, 16, 16), 0, 0, 0),
             "point-mode rasterized the triangle interior: "
             << describePixel(pixelAt(pixels, 16, 16)));
  });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
