// testsuite/vulkan-backend-texture-models-test.cpp
//
// Exercises every SoTextureModel fragment combination: REPLACE shows the
// texel color directly, MODULATE multiplies the base color by the texel
// color, DECAL leaves transparent texels showing the base color, and BLEND
// mixes the blend color by the texel's red channel.  (Merged from the former
// separate texture and texture-models tests.)

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

// 2x2 solid-red RGBA texture.
const unsigned char redTexel[16] = {
  255, 0, 0, 255,  255, 0, 0, 255,
  255, 0, 0, 255,  255, 0, 0, 255
};
// 2x2 solid-green RGBA texture.
const unsigned char greenTexel[16] = {
  0, 255, 0, 255,  0, 255, 0, 255,
  0, 255, 0, 255,  0, 255, 0, 255
};
// 1x1 red RGBA texels.
const unsigned char transparentRed[4] = {255, 0, 0, 0};
const unsigned char opaqueRed[4] = {255, 0, 0, 255};

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  CaseRunner cases;

  cases.add("REPLACE: texel replaces the base color", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(makeTexturedQuad(redTexel, 2, 2,
                                         SO_TEXTURE_MODEL_REPLACE,
                                         SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "REPLACE produced "
                                           << describePixel(center));
  });

  cases.add("MODULATE: white base x green texel = green", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(makeTexturedQuad(greenTexel, 2, 2,
                                         SO_TEXTURE_MODEL_MODULATE,
                                         SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "MODULATE produced "
                                           << describePixel(center));
  });

  cases.add("MODULATE: red base x green texel = black", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(makeTexturedQuad(greenTexel, 2, 2,
                                         SO_TEXTURE_MODEL_MODULATE,
                                         SbVec4f(1.0f, 0.0f, 0.0f, 1.0f)));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 0, 0), "MODULATE produced "
                                         << describePixel(center));
  });

  cases.add("DECAL: transparent texel preserves the base color", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(makeTexturedQuad(transparentRed, 1, 1,
                                         SO_TEXTURE_MODEL_DECAL,
                                         SbVec4f(1.0f, 1.0f, 1.0f, 1.0f)));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 255, 255), "DECAL produced "
                                               << describePixel(center));
  });

  cases.add("BLEND: red texel + green blend color = cyan", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(makeTexturedQuad(opaqueRed, 1, 1, SO_TEXTURE_MODEL_BLEND,
                                         SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                         SbVec4f(0.0f, 1.0f, 0.0f, 1.0f)));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 255), "BLEND produced "
                                             << describePixel(center));
  });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
