// testsuite/vulkan/vulkan-backend-material-change-test.cpp
//
// Guards the retained material cache / dirty-hash against a stale-material
// regression.  The retained-geometry change test covers a changed *position
// pointer*; this covers the complementary case the batching cache must not
// miss: the geometry pointers are identical (retained) and only a material
// field (roughness) is edited between frames.  A cache keyed on geometry alone
// would reuse frame 1's pipeline/push state and the smooth-metal highlight
// would survive the roughening.
//
//   frame 1: retained quad, physical metal, roughness 0.05 -> mirror highlight
//   frame 2: same retained geometry, roughness 0.9      -> spread/dim centre
//
// The centre luminance must change.

#include "VulkanTestHarness.h"

#include <cstdlib>
#include <iostream>

using namespace vulkan_test;

namespace {

const float quad[] = {
  -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const float normalsUp[] = {
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};

// A retained command (stable pointers) with white physical metal.
SoRenderCommand retainedMetal(float roughness)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.normals = normalsUp;
  command.geometry.normalCount = 4;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.geometry.retained = true;
  command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
  command.material.ambient = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.shininess = 0.0f;
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
  command.material.physicalMaterial = true;
  command.material.metalness = 1.0f;
  command.material.roughness = roughness;
  return command;
}

int
centerLuminance(Harness & harness, float roughness, int & failures,
                const char * frameLabel)
{
  SoDrawList drawlist;
  SoLightingData lighting;
  lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
  SoLightData light;
  light.type = SO_LIGHT_DIRECTIONAL;
  light.direction = SbVec3f(0.0f, 0.0f, 1.0f);
  light.color = SbVec3f(1.0f, 1.0f, 1.0f);
  lighting.lights.push_back(light);

  SoRenderCommand command = retainedMetal(roughness);
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  drawlist.addCommand(command);

  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: " << frameLabel << " render failed" << std::endl;
    ++failures;
  }
  const uint8_t * center = pixelAt(harness.readback(), 16, 16);
  return static_cast<int>(center[0]) + center[1] + center[2];
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  const int smooth = centerLuminance(harness, 0.05f, failures, "smooth");
  const int rough = centerLuminance(harness, 0.9f, failures, "rough");

  std::cout << "[INFO] smooth lum=" << smooth << " rough lum=" << rough
            << std::endl;

  // Frame 1 must show the near-mirror physical highlight.
  if (smooth < 600) {
    std::cerr << "FAIL: smooth metal did not saturate the highlight (lum="
              << smooth << ")" << std::endl;
    ++failures;
  }
  // Frame 2 must visibly change: a stale material cache would keep the smooth
  // highlight and leave the luminance unchanged.
  if (!(rough < smooth - 100)) {
    std::cerr << "FAIL: roughness edit did not change the rendered material "
              << "(smooth=" << smooth << ", rough=" << rough << ")" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
