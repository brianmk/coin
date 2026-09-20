// testsuite/vulkan-backend-sorted-order-test.cpp
//
// Verifies that buildSortedOrder() is honored: with depth testing disabled,
// the sorted order (far-to-near for opaque commands) determines which
// fragment wins rather than insertion order.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const uint32_t indices[] = {0, 1, 2, 0, 2, 3};
const float baseQuad[] = {
  -0.5f, -0.5f, 0.0f,
   0.5f, -0.5f, 0.0f,
   0.5f,  0.5f, 0.0f,
  -0.5f,  0.5f, 0.0f
};

// Distinct static buffers per command so the two commands never alias.
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
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(r, g, b, 1.0f);
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

  // Insertion order: near (red) first, far (green) second.  With depth test
  // off and no sorting, green (drawn last) would win.  buildSortedOrder()
  // sorts opaque commands far-to-near, so far green is drawn first and near
  // red is drawn last.
  SoDrawList drawlist;
  drawlist.addCommand(translatedQuad(nearPositions, 0.3f, 1.0f, 0.0f, 0.0f));
  drawlist.addCommand(translatedQuad(farPositions, 0.7f, 0.0f, 1.0f, 0.0f));

  SbMatrix view;
  view.makeIdentity();
  drawlist.buildSortedOrder(view);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: sorted-order render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const int red = countNear(pixels, 255, 0, 0);
  const int green = countNear(pixels, 0, 255, 0);

  if (red == 0) {
    std::cerr << "FAIL: sorted-order did not draw the near command last"
              << std::endl;
    ++failures;
  }
  if (green != 0) {
    std::cerr << "FAIL: sorted-order drew the far command last (insertion "
                 "order leaked)" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
