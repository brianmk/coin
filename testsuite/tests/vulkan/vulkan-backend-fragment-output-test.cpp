// testsuite/vulkan-backend-fragment-output-test.cpp
//
// Verifies how fragment output is selected and composited: the fragment
// alpha-test functions (NEVER/GREATER/LESS/NONE), explicit blend
// factors/equations (additive ONE/ONE and SRC_ALPHA/ONE_MINUS_SRC_ALPHA), and
// the transparent pass blending a translucent overlay over an opaque
// background.  (Merged from the former alpha-test, blending, and
// transparency tests.)

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

SoRenderCommand alphaQuad(float alpha, SoAlphaTestFunction function,
                          float reference)
{
  SoRenderCommand command = makeQuad(SbVec4f(1.0f, 0.0f, 0.0f, alpha));
  command.state.alphaTest.function = function;
  command.state.alphaTest.reference = reference;
  return command;
}

SoRenderCommand blendedQuad(float r, float g, float b, float a,
                            SoBlendFactor src, SoBlendFactor dst)
{
  SoRenderCommand command = makeQuad(SbVec4f(r, g, b, a));
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

  CaseRunner cases;

  cases.add("alpha test NEVER discards everything", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(1.0f, SO_ALPHA_TEST_NEVER, 0.5f));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 0, 0), "NEVER did not discard: "
                                         << describePixel(center));
  });

  cases.add("alpha test GREATER discards low alpha", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(0.25f, SO_ALPHA_TEST_GREATER, 0.5f));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 0, 0), "GREATER kept low alpha: "
                                         << describePixel(center));
  });

  cases.add("alpha test LESS keeps low alpha", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(0.25f, SO_ALPHA_TEST_LESS, 0.5f));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 64, 0, 0), "LESS discarded matching alpha: "
                                          << describePixel(center));
  });

  cases.add("alpha test NONE keeps everything", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(alphaQuad(1.0f, SO_ALPHA_TEST_NONE, 0.0f));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "NONE did not keep the fragment: "
                                           << describePixel(center));
  });

  cases.add("additive ONE/ONE blend: red + green = yellow", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(blendedQuad(1.0f, 0.0f, 0.0f, 1.0f,
                                    SO_BLEND_FACTOR_ONE,
                                    SO_BLEND_FACTOR_ONE));
    drawlist.addCommand(blendedQuad(0.0f, 1.0f, 0.0f, 1.0f,
                                    SO_BLEND_FACTOR_ONE,
                                    SO_BLEND_FACTOR_ONE));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 255, 0), "additive blend did not produce "
             "yellow: "
             << describePixel(center));
  });

  cases.add("SRC_ALPHA/ONE_MINUS_SRC_ALPHA composites 50/50", [&harness] {
    SoDrawList drawlist;
    SoRenderCommand base = blendedQuad(1.0f, 0.0f, 0.0f, 1.0f,
                                       SO_BLEND_FACTOR_ONE,
                                       SO_BLEND_FACTOR_ZERO);
    base.state.blend.enabled = FALSE;
    drawlist.addCommand(base);
    drawlist.addCommand(blendedQuad(0.0f, 1.0f, 0.0f, 0.5f,
                                    SO_BLEND_FACTOR_SRC_ALPHA,
                                    SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // 50% red + 50% green.
    VK_CHECK(nearColor(center, 127, 127, 0), "SRC_ALPHA blend did not "
             "composite 50/50: "
             << describePixel(center));
  });

  cases.add("transparent pass blends over the opaque background",
            [&harness] {
              SoDrawList drawlist;
              drawlist.addCommand(makeQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f)));
              SoRenderCommand overlay =
                makeQuad(SbVec4f(0.0f, 1.0f, 0.0f, 0.5f));
              overlay.pass = SO_RENDERPASS_TRANSPARENT;
              drawlist.addCommand(overlay);
              VK_CHECK(harness.backend.render(drawlist,
                                              harness.renderParams()),
                       "render failed");
              const uint8_t * center = pixelAt(harness.readback(), 16, 16);
              // 0.5 * green + 0.5 * red = ~(127, 127, 0) in 8-bit.
              VK_CHECK(nearColor(center, 127, 127, 0), "translucent overlay "
                       "did not blend: "
                       << describePixel(center));
            });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
