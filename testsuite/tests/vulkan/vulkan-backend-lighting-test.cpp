// testsuite/vulkan-backend-lighting-test.cpp
//
// Exercises the Gouraud lighting path end-to-end: the set 0 / binding 0
// lighting uniform buffer and the per-vertex lighting evaluation in the
// visual vertex shader.  Covers a directly-lit quad (diffuse dominates),
// point-light attenuation, a spot light outside its cone, and the far end of
// the std140 VisualBlock layout (eighth light slot and emissive slot).
// (Merged from the former lighting, light-variants, and uniform-packing
// tests.)

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

SoLightData directional(SbVec3f direction, SbVec3f color)
{
  SoLightData light;
  light.type = SO_LIGHT_DIRECTIONAL;
  light.direction = direction;
  light.color = color;
  return light;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  CaseRunner cases;

  cases.add("directional headlight lights the quad", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.2f, 0.2f, 0.2f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));

    SoRenderCommand command =
      makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f), SO_SHADING_LEGACY_GOURAUD);
    command.material.ambient = SbVec4f(0.2f, 0.2f, 0.2f, 1.0f);
    command.material.shininess = 0.2f;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // Directly lit, directly facing: diffuse red dominates.  Allow tolerance
    // for the ambient term (green/blue ~0.04).
    VK_CHECK(nearColor(center, 255, 0, 0), "Gouraud quad produced "
                                           << describePixel(center));
  });

  cases.add("point light overhead saturates the diffuse term", [&harness] {
    // A point light very far along +Z approximates a directional headlight
    // for a full-viewport quad (NdotL ~ 1 everywhere), so the diffuse term
    // saturates.
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoLightData light;
    light.type = SO_LIGHT_POINT;
    light.color = SbVec3f(1.0f, 1.0f, 1.0f);
    light.position = SbVec3f(0.0f, 0.0f, 100.0f);
    light.attenuation = SbVec3f(0.0f, 0.0f, 1.0f);
    lighting.lights.push_back(light);
    SoRenderCommand command =
      makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f), SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "overhead point light did not "
             "fully light the quad: "
             << describePixel(center));
  });

  cases.add("distant point light attenuates", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoLightData light;
    light.type = SO_LIGHT_POINT;
    light.color = SbVec3f(1.0f, 1.0f, 1.0f);
    light.position = SbVec3f(0.0f, 0.0f, 10.0f);
    light.attenuation = SbVec3f(0.0f, 1.0f, 0.0f); // att = 1/distance = 0.1
    lighting.lights.push_back(light);
    SoRenderCommand command =
      makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f), SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // 0.1 * 255 = ~25; accept a broad dim band, but must not be fully lit.
    VK_CHECK(center[2] > 5 && center[2] <= 120,
             "distant point light produced unexpected red = "
             << static_cast<int>(center[2]));
    VK_CHECK(center[1] <= 40 && center[0] <= 40,
             "distant point light leaked green/blue: "
             << describePixel(center));
  });

  cases.add("spot light outside its cone stays dark", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.1f, 0.0f, 0.0f);
    SoLightData light;
    light.type = SO_LIGHT_SPOT;
    light.color = SbVec3f(1.0f, 1.0f, 1.0f);
    light.direction = SbVec3f(0.0f, 0.0f, 1.0f);
    light.position = SbVec3f(0.0f, 0.0f, 5.0f);
    light.attenuation = SbVec3f(0.0f, 0.0f, 1.0f);
    light.spotCutoffCos = 0.5f;
    light.spotExponent = 1.0f;
    lighting.lights.push_back(light);
    SoRenderCommand command =
      makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f), SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // The fragment faces away from the cone: only the 0.1 ambient red
    // remains.
    VK_CHECK(center[2] <= 60, "spot light leaked through outside its cone "
           "(red=" << static_cast<int>(center[2]) << ")");
    VK_CHECK(center[1] <= 40 && center[0] <= 40,
             "spot light leaked green/blue: " << describePixel(center));
  });

  cases.add("eighth light slot (std140 tail) reaches the shader",
            [&harness] {
              // Eight lights; only the last (slot 7) is red, along +Z so
              // NdotL = 1.  If any per-light array is mis-packed (wrong
              // stride or offset), the eighth light lands in the wrong slot
              // and this fails.
              SoDrawList drawlist;
              SoLightingData lighting;
              lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
              for (int i = 0; i < 8; ++i) {
                lighting.lights.push_back(
                  directional(SbVec3f(0.0f, 0.0f, 1.0f),
                              (i == 7) ? SbVec3f(1.0f, 0.0f, 0.0f)
                                       : SbVec3f(0.0f, 0.0f, 0.0f)));
              }
              SoRenderCommand command =
                makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                            SO_SHADING_LEGACY_GOURAUD);
              command.lightingHandle = drawlist.addLightingSetup(lighting);
              drawlist.addCommand(command);

              VK_CHECK(harness.backend.render(drawlist,
                                              harness.renderParams()),
                       "render failed");
              const uint8_t * center = pixelAt(harness.readback(), 16, 16);
              VK_CHECK(nearColor(center, 255, 0, 0), "eighth light (slot 7) "
                       "was not lit red: "
                       << describePixel(center));
            });

  cases.add("emissive slot reaches the shader", [&harness] {
    SoDrawList drawlist;
    SoRenderCommand command =
      makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f), SO_SHADING_LEGACY_GOURAUD);
    command.material.emissive = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
    command.lightingHandle = drawlist.addLightingSetup(SoLightingData {});
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "emissive color was not emitted: "
                                           << describePixel(center));
  });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
