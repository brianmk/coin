// testsuite/vulkan-backend-material-test.cpp
//
// Exercises the retained material model beyond the base headlight case:
// emissive color with no lights, two-sided lighting flip, multiple light
// accumulation, and per-vertex colors overriding the uniform diffuse for
// both the unlit and Gouraud paths.  (Merged from the former material and
// vertex-color tests.)

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

static const float greenVertexColors[] = {
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f
};

// Normals pointing away from the orthographic viewer direction (+Z) used by
// the shader when the projection has no perspective term.
static const float normalsDown[] = {
  0.0f, 0.0f, -1.0f,
  0.0f, 0.0f, -1.0f,
  0.0f, 0.0f, -1.0f,
  0.0f, 0.0f, -1.0f
};

// Full-viewport Gouraud quad with green per-vertex colors over a red
// diffuse: the vertex colors must win.
SoRenderCommand vertexColorQuad(SoShadingModel shading)
{
  SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                       shading);
  command.geometry.colors = greenVertexColors;
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

  CaseRunner cases;

  cases.add("emissive with no lights", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.emissive = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "emissive material did not "
             "render red: "
             << describePixel(center));
  });

  cases.add("two-sided lighting flips the back-facing normal", [&harness] {
    // The shader's orthographic branch fixes the viewer direction at
    // V=(0,0,1), so give the quad -Z normals: they face away from the viewer
    // (dot(N,V) < 0) and are flipped to +Z.  The light travels along +Z, so
    // the flipped normal is lit while the unflipped one is not.
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));

    SoDrawList oneSided;
    SoRenderCommand plain = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                        SO_SHADING_LEGACY_GOURAUD);
    plain.geometry.normals = normalsDown;
    plain.lightingHandle = oneSided.addLightingSetup(lighting);
    oneSided.addCommand(plain);
    VK_CHECK(harness.backend.render(oneSided, harness.renderParams()),
             "render failed");
    const uint8_t * oneSidedPixel = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(oneSidedPixel, 0, 0, 0),
             "back-facing normal was lit without two-sided lighting: "
             << describePixel(oneSidedPixel));

    SoDrawList twoSided;
    SoRenderCommand flipped = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    flipped.geometry.normals = normalsDown;
    flipped.material.twoSidedLighting = true;
    flipped.lightingHandle = twoSided.addLightingSetup(lighting);
    twoSided.addCommand(flipped);
    VK_CHECK(harness.backend.render(twoSided, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "two-sided lighting did not flip "
             "the normal: "
             << describePixel(center));
  });

  cases.add("multiple lights accumulate", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(0.5f, 0.5f, 0.5f)));
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(0.5f, 0.5f, 0.5f)));
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // Two 0.5-intensity lights sum to full intensity.
    VK_CHECK(nearColor(center, 255, 0, 0), "two lights did not accumulate "
             "to full brightness: "
             << describePixel(center));
  });

  cases.add("Gouraud: vertex color overrides the diffuse", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command =
      vertexColorQuad(SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "vertex color was not used for "
             "Gouraud shading: "
             << describePixel(center));
  });

  cases.add("unlit: vertex color overrides the diffuse", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(vertexColorQuad(SO_SHADING_UNLIT));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "unlit vertex color did not "
             "override diffuse: "
             << describePixel(center));
  });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
