// testsuite/vulkan-backend-uniform-packing-test.cpp
//
// Exercises the far end of the std140 VisualBlock layout: eight lights with
// the eighth light carrying the color, and the emissive slot.  If any of the
// per-light arrays are mis-packed (wrong stride or offset), the eighth light
// and emissive color land in the wrong slot and this test fails.

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
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

SoRenderCommand litQuad(SoShadingModel shading)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.normals = normals;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
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

  // Eight lights; only the last (slot 7) is red, along +Z so NdotL = 1.
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 8; ++i) {
      SoLightData light;
      light.type = SO_LIGHT_DIRECTIONAL;
      light.direction = SbVec3f(0.0f, 0.0f, 1.0f);
      light.color = (i == 7) ? SbVec3f(1.0f, 0.0f, 0.0f)
                             : SbVec3f(0.0f, 0.0f, 0.0f);
      lighting.lights.push_back(light);
    }

    SoRenderCommand command = litQuad(SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: eighth-light render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: eighth light (slot 7) was not lit red" << std::endl;
      ++failures;
    }
  }

  // Emissive with no lights: the emissive slot must reach the shader.
  {
    SoDrawList drawlist;
    SoRenderCommand command = litQuad(SO_SHADING_LEGACY_GOURAUD);
    command.material.emissive = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
    command.lightingHandle = drawlist.addLightingSetup(SoLightingData{});
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: emissive render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: emissive color was not emitted" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
