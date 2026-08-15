// testsuite/vulkan-backend-culling-test.cpp
//
// Verifies back-face culling.  With the Y-flip compensation, a CCW triangle is
// front-facing and visible while a CW triangle is culled.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  // After the shader's Y-flip compensation and VK_FRONT_FACE_CLOCKWISE, a CW
  // source triangle is front-facing (visible) and a CCW triangle is culled.
  // CW in the source coordinate system.
  static const float front[] = {
    -0.8f, -0.8f, 0.0f,
     0.0f,  0.8f, 0.0f,
     0.8f, -0.8f, 0.0f
  };
  // CCW in the source coordinate system.
  static const float back[] = {
    -0.8f, -0.8f, 0.0f,
     0.8f, -0.8f, 0.0f,
     0.0f,  0.8f, 0.0f
  };

  SoDrawList drawlist;

  SoRenderCommand frontCommand = makeTriangle(front);
  frontCommand.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  frontCommand.state.raster.cullMode = 1; // cull back faces
  drawlist.addCommand(frontCommand);

  SoRenderCommand backCommand = makeTriangle(back);
  backCommand.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  backCommand.state.raster.cullMode = 1; // cull back faces
  drawlist.addCommand(backCommand);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: culling render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const int red = countNear(pixels, 255, 0, 0);
  const int green = countNear(pixels, 0, 255, 0);

  if (red == 0) {
    std::cerr << "FAIL: front-facing triangle was incorrectly culled" << std::endl;
    ++failures;
  }
  if (green != 0) {
    std::cerr << "FAIL: back-facing triangle was not culled" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
