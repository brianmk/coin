// testsuite/vulkan-backend-blending-test.cpp
//
// Verifies explicit blend factors and equations are honored.  All commands are
// opaque (depth/blend pass split never interferes) and use alpha = 1 so the
// only blending is the explicit state.

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

SoRenderCommand blendedQuad(float r, float g, float b, float a,
                            SoBlendFactor src, SoBlendFactor dst)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(r, g, b, a);
  command.state.depth.enabled = FALSE;
  command.state.depth.writeEnabled = FALSE;
  command.state.blend.enabled = TRUE;
  command.state.blend.srcRGBFactor = src;
  command.state.blend.dstRGBFactor = dst;
  command.state.blend.srcAlphaFactor = SO_BLEND_FACTOR_ONE;
  command.state.blend.dstAlphaFactor = SO_BLEND_FACTOR_ZERO;
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

  // Additive blending: red + green -> yellow (independent of draw order).
  {
    SoDrawList drawlist;
    drawlist.addCommand(blendedQuad(1.0f, 0.0f, 0.0f, 1.0f,
                                    SO_BLEND_FACTOR_ONE, SO_BLEND_FACTOR_ONE));
    drawlist.addCommand(blendedQuad(0.0f, 1.0f, 0.0f, 1.0f,
                                    SO_BLEND_FACTOR_ONE, SO_BLEND_FACTOR_ONE));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: additive blend render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 255, 0)) {
      std::cerr << "FAIL: ONE/ONE additive blend did not produce yellow"
                << std::endl;
      ++failures;
    }
  }

  // SRC_ALPHA / ONE_MINUS_SRC_ALPHA with alpha=0.5 over red: expected
  // ~50% green + ~50% red = (127, 127, 0).
  {
    SoDrawList drawlist;
    SoRenderCommand base = blendedQuad(1.0f, 0.0f, 0.0f, 1.0f,
                                       SO_BLEND_FACTOR_ONE, SO_BLEND_FACTOR_ZERO);
    base.state.blend.enabled = FALSE;
    drawlist.addCommand(base);
    drawlist.addCommand(blendedQuad(0.0f, 1.0f, 0.0f, 0.5f,
                                    SO_BLEND_FACTOR_SRC_ALPHA,
                                    SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: alpha blend render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // 50% red + 50% green.  Rasterization of the first quad uses the same
    // blend state, so expect exactly the alpha-composited value.
    if (!nearColor(center, 127, 127, 0)) {
      std::cerr << "FAIL: SRC_ALPHA blend did not composite 50/50"
                << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
