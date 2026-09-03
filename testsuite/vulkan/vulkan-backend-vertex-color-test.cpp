// testsuite/vulkan-backend-vertex-color-test.cpp
//
// Verifies that per-vertex colors override the uniform diffuse color when a
// geometry carries a color stream, for both the unlit and Gouraud paths.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float quad[] = {
  -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const float normals[] = {
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f
};
const float colors[] = {
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

SoRenderCommand vertexColorQuad(SoShadingModel shading)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.normals = normals;
  command.geometry.colors = colors;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  command.material.ambient = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.shadingModel = shading;
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
  const SoRenderParams params = harness.renderParams();

  // Unlit: vertex green must override the red diffuse.
  {
    SoDrawList drawlist;
    drawlist.addCommand(vertexColorQuad(SO_SHADING_UNLIT));
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: unlit vertex-color render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: unlit vertex color did not override diffuse"
                << std::endl;
      ++failures;
    }
  }

  // Gouraud with a white headlight along +Z: vertex green stays green.
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoLightData light;
    light.type = SO_LIGHT_DIRECTIONAL;
    light.color = SbVec3f(1.0f, 1.0f, 1.0f);
    light.direction = SbVec3f(0.0f, 0.0f, 1.0f);
    lighting.lights.push_back(light);

    SoRenderCommand command = vertexColorQuad(SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: Gouraud vertex-color render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: Gouraud vertex color did not override diffuse"
                << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
