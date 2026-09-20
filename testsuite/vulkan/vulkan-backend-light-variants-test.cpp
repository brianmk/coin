// testsuite/vulkan-backend-light-variants-test.cpp
//
// Exercises the retained Gouraud model's non-directional lights: point-light
// attenuation, a distant point light's dim contribution, and spot-light
// falloff outside the cone.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

// Full-viewport quad facing the camera (normals +Z).
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

SoRenderCommand litQuad()
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
  command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  command.material.ambient = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.shininess = 0.0f;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
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

  // --- Point light with constant attenuation, effectively overhead --------
  // A point light very far along +Z approximates a directional headlight for
  // a full-viewport quad (NdotL ~ 1 everywhere), so the diffuse term saturates.
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoLightData light;
    light.type = SO_LIGHT_POINT;
    light.color = SbVec3f(1.0f, 1.0f, 1.0f);
    light.position = SbVec3f(0.0f, 0.0f, 100.0f);
    light.attenuation = SbVec3f(0.0f, 0.0f, 1.0f);
    lighting.lights.push_back(light);

    SoRenderCommand command = litQuad();
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: near point light render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    if (!nearColor(pixelAt(pixels, 16, 16), 255, 0, 0)) {
      std::cerr << "FAIL: overhead point light did not fully light the quad" << std::endl;
      ++failures;
    }
  }

  // --- Distant point light with linear attenuation: dim red ---------------
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoLightData light;
    light.type = SO_LIGHT_POINT;
    light.color = SbVec3f(1.0f, 1.0f, 1.0f);
    light.position = SbVec3f(0.0f, 0.0f, 10.0f);
    light.attenuation = SbVec3f(0.0f, 1.0f, 0.0f); // att = 1/distance = 0.1
    lighting.lights.push_back(light);

    SoRenderCommand command = litQuad();
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: distant point light render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    const uint8_t * center = pixelAt(pixels, 16, 16);
    // 0.1 * 255 = ~25; accept a broad dim band, but must not be fully lit.
    if (center[2] > 120 || center[2] < 5) {
      std::cerr << "FAIL: distant point light produced unexpected red = "
                << static_cast<int>(center[2]) << std::endl;
      ++failures;
    }
    if (center[1] > 40 || center[0] > 40) {
      std::cerr << "FAIL: distant point light leaked green/blue" << std::endl;
      ++failures;
    }
  }

  // --- Spot light outside its cone: only ambient --------------------------
  {
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

    SoRenderCommand command = litQuad();
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: spot light render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    const uint8_t * center = pixelAt(pixels, 16, 16);
    // The fragment faces away from the cone: only the 0.1 ambient red remains.
    if (center[2] > 60) {
      std::cerr << "FAIL: spot light leaked through outside its cone (red="
                << static_cast<int>(center[2]) << ")" << std::endl;
      ++failures;
    }
    if (center[1] > 40 || center[0] > 40) {
      std::cerr << "FAIL: spot light leaked green/blue" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
