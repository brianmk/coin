// testsuite/vulkan-backend-depth-test.cpp
//
// Verifies depth testing: a nearer fragment occludes a farther one regardless
// of draw order, and disabling depth write lets a later, farther fragment
// overwrite the nearer one.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const uint32_t quadIndices[] = {0, 1, 2, 0, 2, 3};

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

  int failures = 0;

  // Vulkan clip-space depth is [0,1]; identity matrices pass object Z
  // straight through, so near = small Z and far = large Z.
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

  auto quad = [](const float * positions, float r, float g, float b) {
    SoRenderCommand command;
    command.modelMatrix.makeIdentity();
    command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
    command.geometry.vertexCount = 4;
    command.geometry.indexCount = 6;
    command.geometry.positions = positions;
    command.geometry.indices = quadIndices;
    command.geometry.vertexStride = sizeof(float) * 3;
    command.material.diffuse = SbVec4f(r, g, b, 1.0f);
    return command;
  };

  const SoRenderParams params = harness.renderParams();

  // --- Draw far first, then near: near must win ---------------------------
  {
    SoDrawList drawlist;
    drawlist.addCommand(quad(farQuad, 1.0f, 0.0f, 0.0f));
    drawlist.addCommand(quad(nearQuad, 0.0f, 1.0f, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: far-then-near depth render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 255, 0)) {
      std::cerr << "FAIL: near fragment did not occlude far fragment" << std::endl;
      ++failures;
    }
  }

  // --- Draw near first, then far: near must still win --------------------
  {
    SoDrawList drawlist;
    drawlist.addCommand(quad(nearQuad, 0.0f, 1.0f, 0.0f));
    drawlist.addCommand(quad(farQuad, 1.0f, 0.0f, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: near-then-far depth render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 255, 0)) {
      std::cerr << "FAIL: near fragment lost to far fragment (order-dependent)"
                << std::endl;
      ++failures;
    }
  }

  // --- Depth write disabled: farther fragment overwrites -----------------
  {
    SoRenderCommand nearNoWrite = quad(nearQuad, 0.0f, 1.0f, 0.0f);
    nearNoWrite.state.depth.writeEnabled = FALSE;
    SoDrawList drawlist;
    drawlist.addCommand(nearNoWrite);
    drawlist.addCommand(quad(farQuad, 1.0f, 0.0f, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: no-depth-write render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    if (!nearColor(pixelAt(pixels, 16, 16), 255, 0, 0)) {
      std::cerr << "FAIL: farther fragment should overwrite when depth write is "
                   "disabled" << std::endl;
      ++failures;
    }
  }

  // --- Depth test disabled: later fragment always overwrites -------------
  {
    SoRenderCommand farNoTest = quad(farQuad, 1.0f, 0.0f, 0.0f);
    farNoTest.state.depth.enabled = FALSE;
    SoDrawList drawlist;
    drawlist.addCommand(quad(nearQuad, 0.0f, 1.0f, 0.0f));
    drawlist.addCommand(farNoTest);
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: no-depth-test render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    if (!nearColor(pixelAt(pixels, 16, 16), 255, 0, 0)) {
      std::cerr << "FAIL: later fragment should overwrite when depth test is "
                   "disabled" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
