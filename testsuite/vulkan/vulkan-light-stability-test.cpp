// testsuite/vulkan/vulkan-light-stability-test.cpp
//
// Regression test: scene lights must stay fixed in world space while the
// camera moves.  Renders a PHONG-lit cube with a fixed directional light
// twice -- once viewed from +Z (the light is orthogonal to the visible
// face) and once from +X (the light faces the visible face) -- and asserts
// the +X view is dramatically brighter.  A light that followed the camera
// (the reported bug: "the lighting seems to move with the camera") would
// light the visible face equally in both views.

#include "VulkanTestHarness.h"

#include <Inventor/SoDB.h>
#include <Inventor/SbColor.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/rendering/SoVulkanRenderManager.h>

using namespace vulkan_test;

static double
averageLuminance(const std::vector<uint8_t> & pixels)
{
  double sum = 0.0;
  for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
    const uint8_t * p = &pixels[static_cast<size_t>(i) * kPixelBytes];
    sum += static_cast<double>(p[2]) + static_cast<double>(p[1]) +
      static_cast<double>(p[0]);
  }
  return sum / (3.0 * static_cast<double>(kWidth * kHeight));
}

static SoSeparator *
makeScene(SoPerspectiveCamera * camera)
{
  SoSeparator * root = new SoSeparator;
  root->ref();

  camera->nearDistance = 0.1f;
  camera->farDistance = 100.0f;
  camera->heightAngle = 0.785398f; // 45 degrees
  root->addChild(camera);

  SoLightModel * lightModel = new SoLightModel;
  lightModel->model = SoLightModel::PHONG;
  root->addChild(lightModel);

  SoEnvironment * environment = new SoEnvironment;
  environment->ambientColor.setValue(0.0f, 0.0f, 0.0f);
  environment->ambientIntensity = 0.0f;
  root->addChild(environment);

  // Coin convention: the direction field points FROM the light toward the
  // scene, so (-1, 0, 0) makes light travel toward +X.  The light node sits
  // at the scene root (identity model matrix), so the light is fixed in
  // world space.
  SoDirectionalLight * light = new SoDirectionalLight;
  light->direction.setValue(-1.0f, 0.0f, 0.0f);
  light->intensity = 1.0f;
  light->color.setValue(1.0f, 1.0f, 1.0f);
  root->addChild(light);

  SoMaterial * material = new SoMaterial;
  material->diffuseColor.setValue(1.0f, 1.0f, 1.0f);
  material->ambientColor.setValue(0.0f, 0.0f, 0.0f);
  material->specularColor.setValue(1.0f, 1.0f, 1.0f);
  material->shininess = 0.0f;
  root->addChild(material);

  SoCube * cube = new SoCube;
  cube->width = 2.0f;
  cube->height = 2.0f;
  cube->depth = 2.0f;
  root->addChild(cube);

  return root;
}

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  SoPerspectiveCamera * camera = new SoPerspectiveCamera;
  SoSeparator * root = makeScene(camera);

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
      root->unref();
      harness.shutdown();
      SoDB::finish();
      return failures == 0 ? 0 : 1;
    }

    // View A: camera on +Z looking at the origin; the visible +Z face is
    // perpendicular to the light direction, so it must be dark (no diffuse
    // contribution; black ambient).
    camera->position.setValue(0.0f, 0.0f, 5.0f);
    camera->orientation.setValue(SbVec3f(0.0f, 0.0f, 1.0f), 0.0f);
    if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: view A render failed" << std::endl;
      ++failures;
    }
    else {
      const double luminanceA = averageLuminance(harness.readback());
      // View B: camera on +X looking at the origin; the visible +X face
      // faces the light and must be bright.
      camera->position.setValue(5.0f, 0.0f, 0.0f);
      camera->pointAt(SbVec3f(0.0f, 0.0f, 0.0f));
      if (!manager.render(TRUE, TRUE)) {
        std::cerr << "FAIL: view B render failed" << std::endl;
        ++failures;
      }
      else {
        const double luminanceB = averageLuminance(harness.readback());
        std::cout << "luminance +Z-view=" << luminanceA
                  << " +X-view=" << luminanceB << std::endl;
        // The lit face is ~5x brighter than the perpendicular face (which
        // only receives background/specular-free diffuse at NdotL ~ 0).
        if (luminanceB < luminanceA * 2.0 || luminanceB < 40.0) {
          std::cerr << "FAIL: light appears to move with the camera: the "
                       "world-fixed +X light did not keep the +X face lit ("
                    << luminanceA << " vs " << luminanceB << ")" << std::endl;
          ++failures;
        }
      }
    }
  }

  root->unref();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
