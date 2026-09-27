// testsuite/vulkan-backend-parallel-lifecycle-test.cpp
//
// Regression test for the parallel recorder's resource lifecycle and its
// failure reporting:
//
//  * Scenario A (normal): with VK_EXT_nested_command_buffer enabled, toggle
//    between the M1c serial-secondary and M1d parallel recorder while resizing
//    the frames-in-flight count.  Each resize frees and reallocates the
//    per-worker secondary command buffers, so a wrong-pool free or a leak in
//    releaseFrameResources() surfaces here.  With the validation layer present
//    the test asserts no vkFreeCommandBuffers validation error is raised.
//
//  * Scenario B (forced spawn failure): FC_VULKAN_TEST_FAIL_RECORD_POOL forces
//    buildRecordPool() down its "worker spawn failed" branch.  Initialization
//    must still succeed via the serial fallback (not fail), report the
//    fallback loudly, and render.
//
// The scene is deliberately small; FC_VULKAN_PARALLEL_MIN_ITEMS=1 makes the
// parallel branch engage without a large draw list.

#include "VulkanTestHarness.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace vulkan_test;

namespace {

const float kTriangle[] = {
  -0.8f, -0.8f, 0.0f,
   0.8f, -0.8f, 0.0f,
   0.0f,  0.8f, 0.0f
};

SoDrawList smallScene()
{
  SoDrawList drawlist;
  for (int i = 0; i < 8; ++i) {
    SoRenderCommand c = makeTriangle(kTriangle);
    c.material.diffuse =
      SbVec4f(static_cast<float>(i) / 8.0f, 0.7f, 0.2f, 1.0f);
    c.state.depth.enabled = true;
    c.state.depth.writeEnabled = true;
    SbMatrix m;
    m.makeIdentity();
    m[3][0] = 0.3f * (i - 4);
    c.modelMatrix = m;
    drawlist.addCommand(c);
  }
  return drawlist;
}

// Renders once and asserts the frame is not black.
void renderAndCheck(Harness & harness, const SoDrawList & drawlist,
                    const char * what, int & failures)
{
  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: render failed (" << what << ")" << std::endl;
    ++failures;
    return;
  }
  const std::vector<uint8_t> pixels = harness.readback();
  int nonBlack = 0;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    if (pixels[i] || pixels[i + 1] || pixels[i + 2]) ++nonBlack;
  }
  if (nonBlack == 0) {
    std::cerr << "FAIL: frame is entirely black (" << what << ")" << std::endl;
    ++failures;
  }
}

} // namespace

int
main()
{
  // Make the (small) scene cross the parallel threshold so the M1d branch
  // engages.  Resolved once, before any backend exists.
  setenv("FC_VULKAN_PARALLEL_MIN_ITEMS", "1", 1);

  int failures = 0;
  const SoDrawList drawlist = smallScene();

  // --- Scenario A: normal nested + serial/parallel toggling + resize --------
  {
    Harness harness;
    harness.wantNestedCommandBuffer = true;
    harness.wantValidation = true;
    const int initResult = harness.init();
    if (initResult != 0) return initResult;
    if (!harness.haveNestedCommandBuffer) {
      return skip("device lacks VK_EXT_nested_command_buffer; secondary / "
                  "parallel lifecycle cannot be exercised");
    }
    std::printf("lifecycle scenario A: validation=%s\n",
                harness.haveValidation ? "on" : "unavailable");

    clearValidationMessages();
    clearBackendErrors();
    const uint32_t frameCounts[] = {1, 2, 3, 2, 1, 3};
    for (uint32_t frames : frameCounts) {
      harness.backend.setMaxFramesInFlight(frames);
      harness.backend.setParallelRecordEnabled(FALSE);
      renderAndCheck(harness, drawlist, "serial resize", failures);
      harness.backend.setParallelRecordEnabled(TRUE);
      renderAndCheck(harness, drawlist, "parallel resize", failures);
    }

    if (harness.backend.parallelRecordFrameCount() == 0) {
      std::cerr << "FAIL: parallel recorder never engaged during the resize "
                   "sweep" << std::endl;
      ++failures;
    }

    // A wrong-pool free during releaseFrameResources() shows up as this VUID
    // under validation; a secondary that failed to set the dynamic viewport
    // shows up as the vkCmdDraw viewport VUID.  Scope the check to these
    // path-specific VUIDs so unrelated diagnostics do not fail the test.
    if (harness.haveValidation &&
        (sinkContains(validationSink(), "vkFreeCommandBuffers") ||
         sinkContains(validationSink(), "vkCmdDraw-None-07831"))) {
      std::cerr << "FAIL: validation reported a secondary-record resource or "
                   "dynamic-state error during the lifecycle sweep" << std::endl;
      ++failures;
    }

    harness.shutdown();
  }

  // --- Scenario B: forced record-pool spawn failure -------------------------
  {
    setenv("FC_VULKAN_TEST_FAIL_RECORD_POOL", "1", 1);
    clearBackendErrors();
    clearValidationMessages();

    Harness harness;
    harness.wantNestedCommandBuffer = true;
    harness.wantValidation = true;
    const int initResult = harness.init();
    if (initResult != 0) {
      // The spawn failure must be a serial fallback, not a fatal init error.
      std::cerr << "FAIL: forced record-pool spawn failure aborted backend "
                   "initialization" << std::endl;
      ++failures;
    }
    else {
      if (!sinkContains(backendErrorSink(), "forced worker-spawn failure")) {
        std::cerr << "FAIL: forced spawn failure was not reported loudly"
                  << std::endl;
        ++failures;
      }
      if (harness.backend.parallelRecordFrameCount() != 0) {
        std::cerr << "FAIL: parallel path ran after a forced spawn failure"
                  << std::endl;
        ++failures;
      }
      renderAndCheck(harness, drawlist, "forced-failure serial fallback",
                     failures);
      if (harness.haveValidation &&
          sinkContains(validationSink(), "vkFreeCommandBuffers")) {
        std::cerr << "FAIL: validation reported an invalid "
                     "vkFreeCommandBuffers after the spawn fallback"
                  << std::endl;
        ++failures;
      }
      harness.shutdown();
    }
    unsetenv("FC_VULKAN_TEST_FAIL_RECORD_POOL");
  }

  SoDB::finish();
  unsetenv("FC_VULKAN_PARALLEL_MIN_ITEMS");
  return failures == 0 ? 0 : 1;
}
