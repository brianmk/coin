// testsuite/vulkan-backend-depth-test.cpp
//
// Verifies the depth state machine: LEQUAL occlusion is independent of draw
// order, depth-write-off lets a farther fragment overwrite, depth-test-off
// lets the later fragment overwrite, and the per-command compare functions
// ALWAYS/NEVER/GREATER behave.  Also verifies buildSortedOrder(): transparent
// commands are sorted back-to-front and the transparent pass honors that
// order.  (Merged from the former depth, depth-function, and sorted-order
// tests.)  The opaque pass is reordered (depth-bucketed), so cases that need
// a specific draw order use two frames or the transparent pass instead.

#include "VulkanTestHarness.h"

#include <cstring>

using namespace vulkan_test;

namespace {

// Vulkan clip-space depth is [0,1]; identity matrices pass object Z straight
// through, so near = small Z and far = large Z.
static const float farQuad[] = {
  -1.0f, -1.0f,  0.8f,
   1.0f, -1.0f,  0.8f,
   1.0f,  1.0f,  0.8f,
  -1.0f,  1.0f,  0.8f
};
static const float nearQuad[] = {
  -1.0f, -1.0f,  0.2f,
   1.0f, -1.0f,  0.2f,
   1.0f,  1.0f,  0.2f,
  -1.0f,  1.0f,  0.2f
};
static const float fartherQuad[] = {
  -1.0f, -1.0f,  0.99f,
   1.0f, -1.0f,  0.99f,
   1.0f,  1.0f,  0.99f,
  -1.0f,  1.0f,  0.99f
};

SoRenderCommand zQuad(const float * positions, float r, float g, float b)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = positions;
  command.geometry.indices = quadIndices();
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(r, g, b, 1.0f);
  return command;
}

// Sorted-order geometry: distinct static buffers per command so the two
// commands never alias one another across uploads.  The commands are marked
// TRANSPARENT because that is the pass buildSortedOrder() orders (the opaque
// pass is depth-bucketed, and depth-disabled opaque commands go to the on-top
// annotation pass in insertion order, so neither honors the sort).
static const float baseQuad[] = {
  -0.5f, -0.5f, 0.0f,
   0.5f, -0.5f, 0.0f,
   0.5f,  0.5f, 0.0f,
  -0.5f,  0.5f, 0.0f
};
float farPositions[12];
float nearPositions[12];

SoRenderCommand translatedQuad(float * positions, float z, float r, float g,
                               float b)
{
  std::memcpy(positions, baseQuad, sizeof(float) * 12);
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.modelMatrix.setTranslate(SbVec3f(0.0f, 0.0f, z));
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = positions;
  command.geometry.indices = quadIndices();
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(r, g, b, 1.0f);
  command.pass = SO_RENDERPASS_TRANSPARENT;
  command.state.depth.writeEnabled = FALSE;
  return command;
}

// Frame-2 params: keep the color and depth attachments from the previous
// frame so a second, single-command frame can probe the depth buffer state
// the first frame left behind.
SoRenderParams noClear(Harness & harness)
{
  SoRenderParams params = harness.renderParams();
  params.flags = 0;
  return params;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;
  if (!harness.haveDepth) {
    harness.shutdown();
    SoDB::finish();
    return skip("no depth attachment");
  }

  CaseRunner cases;

  cases.add("near occludes far (far drawn first)", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(zQuad(farQuad, 1.0f, 0.0f, 0.0f));
    drawlist.addCommand(zQuad(nearQuad, 0.0f, 1.0f, 0.0f));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "near fragment lost to far: "
                                           << describePixel(center));
  });

  cases.add("near occludes far (near drawn first)", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(zQuad(nearQuad, 0.0f, 1.0f, 0.0f));
    drawlist.addCommand(zQuad(farQuad, 1.0f, 0.0f, 0.0f));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "depth test is order-dependent: "
                                           << describePixel(center));
  });

  cases.add("depth write off: farther fragment overwrites", [&harness] {
    // The opaque pass is reordered, so a single frame with two coplanar-ish
    // quads is order-dependent.  Use two frames instead: frame 1 draws the
    // near quad with depth writes off, frame 2 (no clear) draws the far quad
    // and must pass because the near quad left the depth buffer untouched.
    SoRenderCommand nearNoWrite = zQuad(nearQuad, 0.0f, 1.0f, 0.0f);
    nearNoWrite.state.depth.writeEnabled = FALSE;
    SoDrawList first;
    first.addCommand(nearNoWrite);
    VK_CHECK(harness.backend.render(first, harness.renderParams()),
             "render failed");
    VK_CHECK(nearColor(pixelAt(harness.readback(), 16, 16), 0, 255, 0),
             "near fragment did not draw");

    SoDrawList second;
    second.addCommand(zQuad(farQuad, 1.0f, 0.0f, 0.0f));
    VK_CHECK(harness.backend.render(second, noClear(harness)),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "farther fragment did not "
             "overwrite with depth write off: "
             << describePixel(center));
  });

  cases.add("depth test off: later fragment overwrites", [&harness] {
    SoRenderCommand farNoTest = zQuad(farQuad, 1.0f, 0.0f, 0.0f);
    farNoTest.state.depth.enabled = FALSE;
    SoDrawList drawlist;
    drawlist.addCommand(zQuad(nearQuad, 0.0f, 1.0f, 0.0f));
    drawlist.addCommand(farNoTest);
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "later fragment did not overwrite "
             "with depth test off: "
             << describePixel(center));
  });

  cases.add("ALWAYS overwrites regardless of depth", [&harness] {
    // Frame 1 leaves depth 0.8 from a green far quad; frame 2 draws a red
    // quad *farther still* (0.99) with ALWAYS, which must pass where LEQUAL
    // would not.  (Single-frame is order-dependent: whichever quad is
    // reordered last wins.)
    SoDrawList first;
    first.addCommand(zQuad(farQuad, 0.0f, 1.0f, 0.0f));
    VK_CHECK(harness.backend.render(first, harness.renderParams()),
             "render failed");

    SoRenderCommand always = zQuad(fartherQuad, 1.0f, 0.0f, 0.0f);
    always.state.depth.func = SO_DEPTH_ALWAYS;
    SoDrawList second;
    second.addCommand(always);
    VK_CHECK(harness.backend.render(second, noClear(harness)),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "ALWAYS did not overwrite: "
                                           << describePixel(center));
  });

  cases.add("NEVER discards", [&harness] {
    SoDrawList drawlist;
    SoRenderCommand never = zQuad(farQuad, 1.0f, 0.0f, 0.0f);
    never.state.depth.func = SO_DEPTH_NEVER;
    drawlist.addCommand(never);
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 0, 0), "NEVER did not discard: "
                                         << describePixel(center));
  });

  cases.add("GREATER passes a farther fragment", [&harness] {
    // Frame 1 leaves depth 0.2 from a green near quad; frame 2 draws a red
    // far quad (0.8) with GREATER, which must pass 0.8 > 0.2.
    SoDrawList first;
    first.addCommand(zQuad(nearQuad, 0.0f, 1.0f, 0.0f));
    VK_CHECK(harness.backend.render(first, harness.renderParams()),
             "render failed");

    SoRenderCommand greater = zQuad(farQuad, 1.0f, 0.0f, 0.0f);
    greater.state.depth.func = SO_DEPTH_GREATER;
    SoDrawList second;
    second.addCommand(greater);
    VK_CHECK(harness.backend.render(second, noClear(harness)),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "GREATER did not pass the farther "
             "fragment: "
             << describePixel(center));
  });

  cases.add("sorted order draws transparent far-to-near", [&harness] {
    // Insertion order: near (red) first, far (green) second.  Both are
    // TRANSPARENT, so buildSortedOrder() orders them back-to-front (far
    // first, near last) and the transparent pass honors that order: near red
    // is drawn last and wins.  The view matrix is the sort camera (looking
    // down -Z), independent of the identity render camera.
    SoDrawList drawlist;
    drawlist.addCommand(translatedQuad(nearPositions, 0.3f, 1.0f, 0.0f,
                                       0.0f));
    drawlist.addCommand(translatedQuad(farPositions, 0.7f, 0.0f, 1.0f,
                                       0.0f));
    SbMatrix view;
    view.makeIdentity();
    view.setScale(SbVec3f(1.0f, 1.0f, -1.0f));
    drawlist.buildSortedOrder(view);

    const std::vector<int> & order = drawlist.getSortedOrder();
    VK_CHECK(order.size() == 2, "sorted order has wrong size");
    VK_CHECK(order[0] == 1 && order[1] == 0,
             "transparent commands not sorted back-to-front: order="
             << order[0] << "," << order[1]);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const std::vector<uint8_t> pixels = harness.readback();
    const int red = countNear(pixels, 255, 0, 0);
    const int green = countNear(pixels, 0, 255, 0);
    VK_CHECK(red > 0, "near command not visible (red=" << red << ")");
    VK_CHECK(green == 0, "far command drawn last, sort order ignored "
           "(green=" << green << ")");
  });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
