// testsuite/vulkan/vulkan-rendermanager-visibility-test.cpp
//
// End-to-end test of the retained-IR replay gate in SoVulkanRenderManager:
// FreeCAD hides/shows an object by setting SoSwitch::whichChild on the view's
// root switch.  A garbage frame (camera-only) first settles the retained-list
// fingerprint cache; a subsequent visibility toggle changes the scene WITHOUT
// changing the (yet-to-be-rebuilt) retained draw list.  The replay gate must
// NOT treat that as "unchanged" and replay the stale list -- the cube must
// actually disappear on the next render.
//
// Before the fix the graph-fingerprint walk was short-circuited by a cheap hash
// of the retained draw list; the toggle left that hash unchanged, the walk was
// skipped, and the stale list replayed forever -- the cube stayed visible.

#include "VulkanTestHarness.h"

#include <Inventor/SoDB.h>
#include <Inventor/SbColor.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
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

  SoSwitch * sw = new SoSwitch;
  sw->whichChild = 0;
  SoCube * cube = new SoCube;
  cube->width = 2.0f;
  cube->height = 2.0f;
  cube->depth = 2.0f;
  sw->addChild(cube);
  root->addChild(sw);

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

    auto countRed = [&](const char * tag) {
      const std::vector<uint8_t> pixels = harness.readback();
      const int red = countNear(pixels, 255, 0, 0);
      const uint8_t * center = pixelAt(pixels, 16, 16);
      std::cerr << "[TEST] " << tag << " redPixels=" << red
                << " center=(" << (int)center[2] << "," << (int)center[1]
                << "," << (int)center[0] << ")" << std::endl;
      return red;
    };

    // Frame 1: initial build -> cube visible.
    if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: frame-1 render failed" << std::endl;
      ++failures;
    } else if (countRed("frame1 build") <= 0) {
      std::cerr << "FAIL: cube not rendered red on frame 1" << std::endl;
      ++failures;
    }

    // Frame 2: camera-only "settling" frame.  Moving the camera inside the
    // scene fires the scene-dirty sensor but leaves the retained content
    // unchanged, so the retained-list fingerprint cache settles (drawFpCached
    // now equals the cube list) and the gate is armed for the steady state.
    camera->position.setValue(0.0f, 0.0f, 6.0f);
    SoDB::getSensorManager()->processDelayQueue(TRUE);
    if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: frame-2 framing render failed" << std::endl;
      ++failures;
    } else if (countRed("frame2 settle") <= 0) {
      std::cerr << "FAIL: cube not rendered red on settle frame" << std::endl;
      ++failures;
    }

    // Frame 3: hide the cube via SoSwitch::whichChild.  The retained list is
    // NOT rebuilt yet, so its fingerprint is unchanged; only the graph walk can
    // see the toggle.  Old code replays the stale list; the fix re-traverses.
    sw->whichChild = -1;
    SoDB::getSensorManager()->processDelayQueue(TRUE);
    if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: frame-3 (hide) render failed" << std::endl;
      ++failures;
    } else if (countRed("frame3 hide") != 0) {
      std::cerr << "FAIL: cube STILL rendered after hide (stale replay)"
                << std::endl;
      ++failures;
    }

    // Frame 4: show the cube again.
    sw->whichChild = 0;
    SoDB::getSensorManager()->processDelayQueue(TRUE);
    if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: frame-4 (show) render failed" << std::endl;
      ++failures;
    } else if (countRed("frame4 show") <= 0) {
      std::cerr << "FAIL: cube not rendered on frame-4 after show"
                << std::endl;
      ++failures;
    }
  } // manager destroyed before harness.shutdown()

  root->unref();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
