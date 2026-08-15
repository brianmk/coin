// testsuite/vulkan-backend-material-test.cpp
//
// Exercises the retained material model beyond the base headlight case:
// emissive color with no lights, two-sided lighting flip, multiple light
// accumulation, and vertex-color Gouraud shading.

#include "VulkanTestHarness.h"

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

SoRenderCommand gouraudQuad()
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = quad;
  command.geometry.normals = normalsUp;
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

  int failures = 0;
  const SoRenderParams params = harness.renderParams();

  // --- Emissive with no lights: emissive color wins -----------------------
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoRenderCommand command = gouraudQuad();
    command.material.emissive = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: emissive render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: emissive material did not render red" << std::endl;
      ++failures;
    }
  }

  // --- Two-sided lighting flips the back-facing normal --------------------
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(directional(SbVec3f(0.0f, 0.0f, -1.0f),
                                          SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command = gouraudQuad();
    // Position the quad at +Z so the view vector V=(0,0,-1) points away from
    // the +Z normal, exercising the two-sided flip.
    command.modelMatrix.setTranslate(SbVec3f(0.0f, 0.0f, 1.0f));
    command.material.twoSidedLighting = true;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: two-sided render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: two-sided lighting did not flip the normal" << std::endl;
      ++failures;
    }
  }

  // --- Multiple lights accumulate -----------------------------------------
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(directional(SbVec3f(0.0f, 0.0f, 1.0f),
                                          SbVec3f(0.5f, 0.5f, 0.5f)));
    lighting.lights.push_back(directional(SbVec3f(0.0f, 0.0f, 1.0f),
                                          SbVec3f(0.5f, 0.5f, 0.5f)));
    SoRenderCommand command = gouraudQuad();
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: multi-light render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // Two 0.5-intensity lights sum to full intensity.
    if (!nearColor(center, 255, 0, 0)) {
      std::cerr << "FAIL: two lights did not accumulate to full brightness"
                << std::endl;
      ++failures;
    }
  }

  // --- Vertex color with Gouraud shading ----------------------------------
  {
    const float triangle[] = {
      -0.9f, -0.9f, 0.0f,
       0.9f, -0.9f, 0.0f,
       0.0f,  0.9f, 0.0f
    };
    const float normals[] = {
      0.0f, 0.0f, 1.0f,
      0.0f, 0.0f, 1.0f,
      0.0f, 0.0f, 1.0f
    };
    const float colors[] = {
      0.0f, 1.0f, 0.0f, 1.0f,
      0.0f, 1.0f, 0.0f, 1.0f,
      0.0f, 1.0f, 0.0f, 1.0f
    };

    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(directional(SbVec3f(0.0f, 0.0f, 1.0f),
                                          SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command;
    command.modelMatrix.makeIdentity();
    command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
    command.geometry.vertexCount = 3;
    command.geometry.positions = triangle;
    command.geometry.normals = normals;
    command.geometry.colors = colors;
    command.geometry.vertexStride = sizeof(float) * 3;
    command.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
    command.material.ambient = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
    command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
    command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
    command.material.shininess = 0.0f;
    command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: vertex-color Gouraud render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: vertex color was not used for Gouraud shading" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
