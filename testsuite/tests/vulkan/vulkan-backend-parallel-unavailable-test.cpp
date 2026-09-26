// testsuite/vulkan-backend-parallel-unavailable-test.cpp
//
// Regression test for the "parallel recording requested but unavailable"
// silent fallback.  With FC_VULKAN_PARALLEL_RECORD set but the device NOT
// created with VK_EXT_nested_command_buffer, the backend used to quietly
// record fully inline.  It must now report the unmet precondition through its
// error sink, and still render correctly (the fallback itself is fine -- the
// silence is not).
//
// The environment is set before any backend is created because SoVulkanConfig
// resolves it once per process.

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

} // namespace

int
main()
{
  // Resolve the config with parallel recording explicitly requested.  This
  // test deliberately does NOT ask the harness to enable nested command
  // buffers, so the request cannot be satisfied.
  setenv("FC_VULKAN_PARALLEL_RECORD", "1", 1);
  clearBackendErrors();

  int failures = 0;

  Harness harness;
  harness.wantNestedCommandBuffer = false;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  // The backend must have said why parallel recording is unavailable.
  if (!sinkContains(backendErrorSink(), "FC_VULKAN_PARALLEL_RECORD is set")) {
    std::cerr << "FAIL: parallel recording was requested but unavailable and "
                 "the backend did not report it" << std::endl;
    ++failures;
  }

  // The fallback must still render (no lost frame).
  SoDrawList drawlist;
  SoRenderCommand c = makeTriangle(kTriangle);
  c.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
  drawlist.addCommand(c);
  if (!harness.backend.render(drawlist, harness.renderParams())) {
    std::cerr << "FAIL: serial fallback render failed" << std::endl;
    ++failures;
  }
  const std::vector<uint8_t> pixels = harness.readback();
  int nonBlack = 0;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    if (pixels[i] || pixels[i + 1] || pixels[i + 2]) ++nonBlack;
  }
  if (nonBlack == 0) {
    std::cerr << "FAIL: serial fallback produced no fragments" << std::endl;
    ++failures;
  }

  // The fallback must not have silently claimed to run the parallel path.
  if (harness.backend.parallelRecordFrameCount() != 0) {
    std::cerr << "FAIL: parallel path ran without nested command buffer support"
              << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  unsetenv("FC_VULKAN_PARALLEL_RECORD");
  return failures == 0 ? 0 : 1;
}
