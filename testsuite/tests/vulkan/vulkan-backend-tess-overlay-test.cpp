// testsuite/vulkan/vulkan-backend-tess-overlay-test.cpp
//
// Verifies the raster overlay re-draw semantics in SoVulkanRenderBackend.
// The tessellation-edges overlay re-draws triangle commands in LINES mode
// (their outline only, in the overlay color) and deliberately skips line
// commands so feature edges are not drawn twice.  The edge overlay
// (wireframe) re-draws only line commands in LINES mode, so a plain filled
// triangle is left untouched.  The re-draws use the polygon-mode LINES
// pipeline, which requires the fillModeNonSolid device feature; the test
// skips on devices without it.
//
// Geometry: a triangle whose vertical edge and a feature LINE (width 3)
// land on known pixel rows/columns (the rasterizer applies a half-pixel
// shift, verified empirically: a line at clip y = -0.5 covers pixel row 23,
// and the triangle's right edge at clip x = 0.3125 covers pixel column 20).
// All vertices stay strictly inside the clip volume; vertices on the clip
// boundary rasterize unreliably.

#include "VulkanTestHarness.h"

#include <cstdint>

using namespace vulkan_test;

namespace {

// Right triangle, fill = green, z = 0.5.  The vertical edge sits at clip
// x = 0.3125 (nominal pixel 21.0, rendered at 20.5), so the 1px LINES
// re-draw of that edge covers pixel column 20.
static const float triVerts[] = {
  -0.9f, -0.9f, 0.5f,
   0.3125f, -0.9f, 0.5f,
   0.3125f, 0.9f, 0.5f
};

SoRenderCommand tri()
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 3;
  command.geometry.positions = triVerts;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  return command;
}

// Feature edge (LINE, width 3), own color = blue, z = 0.1 (nearer than the
// triangle).  Runs horizontally at clip y = -0.5 (nominal pixel row 24,
// rendered centered on row 23) from x = -0.8 (column ~3) to x = 0.2125
// (column ~20), clear of the triangle's edge column.  Pixel (6, 23) sits on
// the line's center row but outside the triangle (black background).
static const float lineVerts[] = {
  -0.8f, -0.5f, 0.1f,
   0.2125f, -0.5f, 0.1f
};

SoRenderCommand featureLine()
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_LINES;
  command.geometry.vertexCount = 2;
  command.geometry.positions = lineVerts;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(0.0f, 0.0f, 1.0f, 1.0f);
  command.state.raster.lineWidth = 3.0f;
  return command;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;
  if (!harness.haveFillModeNonSolid) {
    harness.shutdown();
    SoDB::finish();
    return skip("device lacks fillModeNonSolid; LINES re-draws cannot "
                "render");
  }

  harness.backend.setEdgeColor(SbColor4f(1.0f, 0.0f, 1.0f, 1.0f));

  CaseRunner cases;

  cases.add("baseline: fill + feature line, no overlay", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(tri());
    drawlist.addCommand(featureLine());
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const std::vector<uint8_t> px = harness.readback();
    VK_CHECK(nearColor(pixelAt(px, 20, 8), 0, 255, 0),
             "triangle edge should be plain fill: "
               << describePixel(pixelAt(px, 20, 8)));
    VK_CHECK(nearColor(pixelAt(px, 6, 23), 0, 0, 255),
             "feature line should keep its own color: "
               << describePixel(pixelAt(px, 6, 23)));
    VK_CHECK(nearColor(pixelAt(px, 16, 16), 0, 255, 0),
             "triangle interior should be filled: "
               << describePixel(pixelAt(px, 16, 16)));
  });

  cases.add("tess overlay: triangle outline re-drawn, feature line "
            "untouched", [&harness] {
    harness.backend.setTessellationOverlay(TRUE);
    SoDrawList drawlist;
    drawlist.addCommand(tri());
    drawlist.addCommand(featureLine());
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const std::vector<uint8_t> px = harness.readback();
    VK_CHECK(nearColor(pixelAt(px, 20, 8), 255, 0, 255),
             "triangle edge should be re-drawn in the overlay color: "
               << describePixel(pixelAt(px, 20, 8)));
    VK_CHECK(nearColor(pixelAt(px, 20, 16), 255, 0, 255),
             "triangle edge (lower sample) should be re-drawn: "
               << describePixel(pixelAt(px, 20, 16)));
    VK_CHECK(nearColor(pixelAt(px, 6, 23), 0, 0, 255),
             "feature line must not be re-drawn by the tess overlay: "
               << describePixel(pixelAt(px, 6, 23)));
    VK_CHECK(nearColor(pixelAt(px, 16, 16), 0, 255, 0),
             "triangle interior must stay filled: "
               << describePixel(pixelAt(px, 16, 16)));
    harness.backend.setTessellationOverlay(FALSE);
  });

  cases.add("edge overlay: only line commands re-drawn", [&harness] {
    harness.backend.setWireframeOverlay(TRUE);
    SoDrawList drawlist;
    drawlist.addCommand(tri());
    drawlist.addCommand(featureLine());
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const std::vector<uint8_t> px = harness.readback();
    VK_CHECK(nearColor(pixelAt(px, 6, 23), 255, 0, 255),
             "feature line should be re-drawn in the overlay color: "
               << describePixel(pixelAt(px, 6, 23)));
    VK_CHECK(nearColor(pixelAt(px, 20, 8), 0, 255, 0),
             "filled triangle must not be re-drawn by the edge overlay: "
               << describePixel(pixelAt(px, 20, 8)));
    VK_CHECK(nearColor(pixelAt(px, 16, 16), 0, 255, 0),
             "triangle interior must stay filled: "
               << describePixel(pixelAt(px, 16, 16)));
    harness.backend.setWireframeOverlay(FALSE);
  });

  cases.add("both overlays: union of re-draws", [&harness] {
    harness.backend.setTessellationOverlay(TRUE);
    harness.backend.setWireframeOverlay(TRUE);
    SoDrawList drawlist;
    drawlist.addCommand(tri());
    drawlist.addCommand(featureLine());
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const std::vector<uint8_t> px = harness.readback();
    VK_CHECK(nearColor(pixelAt(px, 20, 8), 255, 0, 255),
             "triangle edge should be re-drawn by the tess overlay: "
               << describePixel(pixelAt(px, 20, 8)));
    VK_CHECK(nearColor(pixelAt(px, 6, 23), 255, 0, 255),
             "feature line should be re-drawn by the edge overlay: "
               << describePixel(pixelAt(px, 6, 23)));
    VK_CHECK(nearColor(pixelAt(px, 16, 16), 0, 255, 0),
             "triangle interior must stay filled: "
               << describePixel(pixelAt(px, 16, 16)));
    harness.backend.setTessellationOverlay(FALSE);
    harness.backend.setWireframeOverlay(FALSE);
  });

  const int failed = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failed ? 1 : 0;
}
