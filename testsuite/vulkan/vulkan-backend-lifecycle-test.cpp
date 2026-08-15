// testsuite/vulkan-backend-lifecycle-test.cpp
//
// Exercises SoVulkanRenderBackend lifecycle contracts: render-before-init
// rejection, render-without-target rejection, initialize idempotency, and
// shutdown idempotency.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

// File-scope storage so the command's geometry pointers stay valid for the
// lifetime of the render.  A stack-local array would dangle once the helper
// returns.
const float kTrianglePositions[] = {
  -0.8f, -0.8f, 0.0f,
   0.8f, -0.8f, 0.0f,
   0.0f,  0.8f, 0.0f
};

SoRenderCommand solidTriangle()
{
  SoRenderCommand command = makeTriangle(kTrianglePositions);
  command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
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

  // A freshly constructed, un-initialized backend must refuse to render.
  {
    SoVulkanRenderBackend uninitialized;
    SoDrawList empty;
    SoRenderParams params = harness.renderParams();
    if (uninitialized.render(empty, params)) {
      std::cerr << "FAIL: uninitialized backend accepted render" << std::endl;
      ++failures;
    }
    if (uninitialized.isInitialized()) {
      std::cerr << "FAIL: uninitialized backend reports initialized" << std::endl;
      ++failures;
    }
  }

  // An initialized backend must refuse a render without a target.
  {
    SoDrawList empty;
    SoRenderParams params = harness.renderParams();
    params.renderTarget = nullptr;
    if (harness.backend.render(empty, params)) {
      std::cerr << "FAIL: render without target was accepted" << std::endl;
      ++failures;
    }
  }

  // initialize() must be idempotent while already initialized.
  {
    SoRenderBackendInitParams initParams;
    initParams.userData = &harness.deviceContext;
    if (!harness.backend.initialize(initParams)) {
      std::cerr << "FAIL: re-initialize returned false" << std::endl;
      ++failures;
    }
    if (!harness.backend.isInitialized()) {
      std::cerr << "FAIL: backend not initialized after re-initialize" << std::endl;
      ++failures;
    }
  }

  // A valid render must succeed and produce non-black pixels.
  {
    SoDrawList drawlist;
    drawlist.addCommand(solidTriangle());
    if (!harness.backend.render(drawlist, harness.renderParams())) {
      std::cerr << "FAIL: valid render failed" << std::endl;
      ++failures;
    }
    const std::vector<uint8_t> pixels = harness.readback();
    int nonBlack = 0;
    for (size_t i = 0; i < kWidth * kHeight; ++i) {
      const uint8_t * p = &pixels[i * kPixelBytes];
      if (p[0] || p[1] || p[2]) ++nonBlack;
    }
    if (nonBlack == 0) {
      std::cerr << "FAIL: valid render produced no fragments" << std::endl;
      ++failures;
    }
  }

  // shutdown() must be idempotent and clear the initialized flag.
  harness.backend.shutdown();
  harness.backendInitialized = false;
  if (harness.backend.isInitialized()) {
    std::cerr << "FAIL: backend still initialized after shutdown" << std::endl;
    ++failures;
  }
  harness.backend.shutdown(); // must be safe
  if (harness.backend.isInitialized()) {
    std::cerr << "FAIL: backend re-initialized by double shutdown" << std::endl;
    ++failures;
  }

  // Rendering after shutdown must be rejected.
  {
    SoDrawList drawlist;
    drawlist.addCommand(solidTriangle());
    if (harness.backend.render(drawlist, harness.renderParams())) {
      std::cerr << "FAIL: render after shutdown was accepted" << std::endl;
      ++failures;
    }
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
