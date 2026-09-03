// testsuite/vulkan-rendermanager-test.cpp
//
// End-to-end smoke test for SoVulkanRenderManager: traverses a real scene
// graph (camera + light model + material + cube) through SoIRRenderAction
// and submits the resulting draw list to the Vulkan backend.

#include "VulkanTestHarness.h"

#include <Inventor/SoDB.h>
#include <Inventor/SbColor.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/rendering/SoVulkanRenderManager.h>

using namespace vulkan_test;

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  SoSeparator * root = new SoSeparator;
  root->ref();

  SoPerspectiveCamera * camera = new SoPerspectiveCamera;
  camera->position.setValue(0.0f, 0.0f, 5.0f);
  camera->nearDistance = 0.1f;
  camera->farDistance = 100.0f;
  camera->heightAngle = 0.785398f; // 45 degrees
  camera->orientation.setValue(SbVec3f(0.0f, 0.0f, 1.0f), 0.0f);
  root->addChild(camera);

  SoLightModel * lightModel = new SoLightModel;
  lightModel->model = SoLightModel::BASE_COLOR;
  root->addChild(lightModel);

  SoMaterial * material = new SoMaterial;
  material->diffuseColor.setValue(1.0f, 0.0f, 0.0f);
  root->addChild(material);

  SoCube * cube = new SoCube;
  cube->width = 2.0f;
  cube->height = 2.0f;
  cube->depth = 2.0f;
  root->addChild(cube);

  {
    SoVulkanRenderManager manager;
    manager.setSceneGraph(root);
    manager.setCamera(camera);

    SbViewportRegion viewport(kWidth, kHeight);
    viewport.setViewportPixels(SbVec2s(0, 0), SbVec2s(kWidth, kHeight));
    manager.setViewportRegion(viewport);
    manager.setBackgroundColor(SbColor4f(0.0f, 0.0f, 0.0f, 1.0f));
    manager.setRenderTarget(&harness.target);

    if (!manager.initialize(&harness.deviceContext)) {
      std::cerr << "FAIL: manager could not initialize the Vulkan backend"
                << std::endl;
      ++failures;
    }
    else if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: manager render failed" << std::endl;
      ++failures;
    }
    else {
      const std::vector<uint8_t> pixels = harness.readback();
      const uint8_t * center = pixelAt(pixels, 16, 16);
      if (!nearColor(center, 255, 0, 0)) {
        std::cerr << "FAIL: cube was not rendered red at center (R="
                  << static_cast<int>(center[2])
                  << " G=" << static_cast<int>(center[1])
                  << " B=" << static_cast<int>(center[0]) << ")" << std::endl;
        ++failures;
      }
    }
  }

  root->unref();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
