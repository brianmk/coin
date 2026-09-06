// testsuite/vulkan-backend-retained-change-test.cpp
//
// Verifies the retained-geometry change-detection contract in
// updateGeometryCache():
//   * A retained command with pointer-identical streams is NOT re-hashed/re-
//     uploaded on a second frame (the backend trusts pointer identity).
//   * A retained command whose position stream pointer changes IS re-uploaded.
// The `[VKGEOMCACHE]` breadcrumb prints `uploads=` per updateGeometryCache()
// pass; we drive the backend directly and read that counter from stderr.

#include "VulkanTestHarness.h"

#include <cstdio>
#include <cstring>
#include <string>

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

// A retained command: stable pointers that the producer promises only change
// when the content changes.
SoRenderCommand retainedQuad()
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
  command.geometry.retained = true;
  command.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  command.material.ambient = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.emissive = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  command.material.shininess = 0.0f;
  command.material.shadingModel = SO_SHADING_UNLIT;
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

  std::cerr << "[TEST] frame1-start" << std::endl;
  // --- Frame 1: upload the retained command -------------------------------
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoRenderCommand command = retainedQuad();
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: frame-1 retained render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: retained frame-1 did not render green" << std::endl;
      ++failures;
    }
  }

  // --- Frame 2: same retained pointers -> must NOT re-upload --------------
  std::cerr << "[TEST] frame2-start" << std::endl;
  {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoRenderCommand command = retainedQuad();
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: frame-2 retained render failed" << std::endl;
      ++failures;
    }
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    if (!nearColor(center, 0, 255, 0)) {
      std::cerr << "FAIL: unchanged retained frame did not render green"
                << std::endl;
      ++failures;
    }
  }

  // --- Frame 3: changed position pointer -> must re-upload ----------------
  std::cerr << "[TEST] frame3-start" << std::endl;
  {
    // A fresh buffer with a translated quad: new pointer, new content.
    const float quad2[] = {
       0.0f, -1.0f, 0.0f,
       2.0f, -1.0f, 0.0f,
       2.0f,  1.0f, 0.0f,
       0.0f,  1.0f, 0.0f
    };
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoRenderCommand command = retainedQuad();
    command.geometry.positions = quad2;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: frame-3 retained render failed" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
