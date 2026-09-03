// testsuite/vulkan-backend-lighting-test.cpp
//
// Exercises the Gouraud lighting path: a red, directly-lit quad must render
// bright red (diffuse term dominates) rather than the unlit diffuse color.
// This validates the set 0 / binding 0 lighting uniform buffer and the
// per-vertex lighting evaluation in the visual vertex shader.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  // Full-viewport quad with all normals pointing +Z (toward the camera).
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

  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.normals = normals;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  command.material.ambient = SbVec4f(0.2f, 0.2f, 0.2f, 1.0f);
  command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.shininess = 0.2f;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;

  SoDrawList drawlist;
  SoLightingData lighting;
  lighting.ambient = SbVec3f(0.2f, 0.2f, 0.2f);
  SoLightData headlight;
  headlight.type = SO_LIGHT_DIRECTIONAL;
  headlight.color = SbVec3f(1.0f, 1.0f, 1.0f);
  headlight.direction = SbVec3f(0.0f, 0.0f, 1.0f);
  lighting.lights.push_back(headlight);
  command.lightingHandle = drawlist.addLightingSetup(lighting);

  drawlist.addCommand(command);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: lighting render failed" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const uint8_t * center = pixelAt(pixels, 16, 16);

  // Directly lit, directly facing: diffuse red dominates.  Allow tolerance
  // for the ambient term (green/blue ~0.04).
  if (!nearColor(center, 255, 0, 0)) {
    std::cerr << "FAIL: Gouraud quad produced B,G,R = "
              << static_cast<int>(center[0]) << ","
              << static_cast<int>(center[1]) << ","
              << static_cast<int>(center[2]) << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
