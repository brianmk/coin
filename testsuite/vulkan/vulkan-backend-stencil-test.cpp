// testsuite/vulkan-backend-stencil-test.cpp
//
// Verifies stencil write and compare: a fullscreen quad writes stencil
// reference 1; a second quad only renders where the stencil equals 1.  The
// stencil buffer persists between frames because the depth/stencil attachment
// uses LOAD/STORE.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float quad[] = {
  -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

SoRenderCommand coloredQuad(float r, float g, float b)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
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
  if (!harness.haveDepth) {
    harness.shutdown();
    SoDB::finish();
    return skip("no depth/stencil attachment");
  }

  int failures = 0;
  SoRenderParams params = harness.renderParams();

  // Clear stencil to 0 (and color to black).
  {
    SoDrawList drawlist;
    params.clearStencil = 0;
    params.flags = SO_PARAM_CLEAR_WINDOW | SO_PARAM_CLEAR_DEPTH |
                   SO_PARAM_CLEAR_STENCIL;
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: stencil clear render failed" << std::endl;
      ++failures;
    }
  }

  // Write reference 1 everywhere (red), with color on.
  {
    SoDrawList drawlist;
    SoRenderCommand command = coloredQuad(1.0f, 0.0f, 0.0f);
    command.state.stencil.enabled = TRUE;
    command.state.stencil.function = SO_STENCIL_FUNC_ALWAYS;
    command.state.stencil.reference = 1;
    command.state.stencil.compareMask = 0xFF;
    command.state.stencil.writeMask = 0xFF;
    command.state.stencil.failOp = SO_STENCIL_OP_KEEP;
    command.state.stencil.zfailOp = SO_STENCIL_OP_KEEP;
    command.state.stencil.zpassOp = SO_STENCIL_OP_REPLACE;
    drawlist.addCommand(command);

    params.flags = 0; // keep the stencil writes visible
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: stencil write render failed" << std::endl;
      ++failures;
    }
  }

  // Clear color only; stencil must persist.
  {
    SoDrawList drawlist;
    params.flags = SO_PARAM_CLEAR_WINDOW;
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: color-only clear render failed" << std::endl;
      ++failures;
    }
  }

  // Render green only where stencil == 1.
  {
    SoDrawList drawlist;
    SoRenderCommand command = coloredQuad(0.0f, 1.0f, 0.0f);
    command.state.stencil.enabled = TRUE;
    command.state.stencil.function = SO_STENCIL_FUNC_EQUAL;
    command.state.stencil.reference = 1;
    command.state.stencil.compareMask = 0xFF;
    command.state.stencil.writeMask = 0xFF;
    command.state.stencil.failOp = SO_STENCIL_OP_KEEP;
    command.state.stencil.zfailOp = SO_STENCIL_OP_KEEP;
    command.state.stencil.zpassOp = SO_STENCIL_OP_KEEP;
    drawlist.addCommand(command);

    params.flags = 0;
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: stencil compare render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: stencil == 1 did not allow the green quad"
                << std::endl;
      ++failures;
    }
  }

  // Render blue only where stencil == 0: must be rejected (still green).
  {
    SoDrawList drawlist;
    SoRenderCommand command = coloredQuad(0.0f, 0.0f, 1.0f);
    command.state.stencil.enabled = TRUE;
    command.state.stencil.function = SO_STENCIL_FUNC_EQUAL;
    command.state.stencil.reference = 0;
    command.state.stencil.compareMask = 0xFF;
    command.state.stencil.writeMask = 0xFF;
    command.state.stencil.failOp = SO_STENCIL_OP_KEEP;
    command.state.stencil.zfailOp = SO_STENCIL_OP_KEEP;
    command.state.stencil.zpassOp = SO_STENCIL_OP_KEEP;
    drawlist.addCommand(command);

    params.flags = 0;
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: stencil reject render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: stencil == 0 incorrectly allowed the blue quad"
                << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
