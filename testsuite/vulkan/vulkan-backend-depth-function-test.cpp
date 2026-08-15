// testsuite/vulkan-backend-depth-function-test.cpp
//
// Verifies per-command depth-compare functions: ALWAYS overwrites, NEVER
// discards, and GREATER passes a farther fragment.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

// Distinct static buffers per depth value so command geometry pointers stay
// valid and never alias one another across uploads.
const float z02[] = {
  -1.0f, -1.0f, 0.2f,  1.0f, -1.0f, 0.2f,
   1.0f,  1.0f, 0.2f, -1.0f,  1.0f, 0.2f
};
const float z08[] = {
  -1.0f, -1.0f, 0.8f,  1.0f, -1.0f, 0.8f,
   1.0f,  1.0f, 0.8f, -1.0f,  1.0f, 0.8f
};

SoRenderCommand depthQuad(const float * positions, SoDepthFunction function,
                          float r, float g, float b)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = positions;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(r, g, b, 1.0f);
  command.state.depth.enabled = TRUE;
  command.state.depth.writeEnabled = TRUE;
  command.state.depth.func = function;
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
  const SoRenderParams params = harness.renderParams();

  // ALWAYS: later red (z=0.8) overwrites green (z=0.2).
  {
    SoDrawList drawlist;
    drawlist.addCommand(depthQuad(z02, SO_DEPTH_LEQUAL, 0.0f, 1.0f, 0.0f));
    drawlist.addCommand(depthQuad(z08, SO_DEPTH_ALWAYS, 1.0f, 0.0f, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: ALWAYS depth render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: ALWAYS depth function did not overwrite" << std::endl;
      ++failures;
    }
  }

  // NEVER: red (z=0.8) never passes -> cleared black.
  {
    SoDrawList drawlist;
    drawlist.addCommand(depthQuad(z08, SO_DEPTH_NEVER, 1.0f, 0.0f, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: NEVER depth render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 0, 0)) {
      std::cerr << "FAIL: NEVER depth function did not discard" << std::endl;
      ++failures;
    }
  }

  // GREATER: green (z=0.2) writes depth 0.2; red (z=0.8) passes GREATER.
  {
    SoDrawList drawlist;
    drawlist.addCommand(depthQuad(z02, SO_DEPTH_LEQUAL, 0.0f, 1.0f, 0.0f));
    drawlist.addCommand(depthQuad(z08, SO_DEPTH_GREATER, 1.0f, 0.0f, 0.0f));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: GREATER depth render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: GREATER depth function did not pass farther fragment"
                << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
