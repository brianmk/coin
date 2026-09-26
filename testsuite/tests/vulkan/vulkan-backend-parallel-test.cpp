// testsuite/vulkan-backend-parallel-test.cpp
//
// Exercises the M1c/M1d secondary/parallel recording path with a heavy, mixed
// scene: many render-order-independent opaque depth-tested commands (shared
// geometry -> instanced batches, plus unique-material singles), a few
// painter-order transparent quads, and a depth-off on-top annotation.
//
// The device must be created with VK_EXT_nested_command_buffer, or the backend
// cannot execute secondaries inside the subpass and silently records fully
// inline -- which is exactly what this test must not accept.  The harness
// enables the extension when the device supports it; if it does not, the test
// skips loudly (77) instead of passing vacuously.
//
// It renders the scene twice on one device:
//   pass 1: parallel recording disabled  -> M1c serial secondary path
//   pass 2: parallel recording enabled    -> M1d parallel path
// and requires:
//   * each pass to have actually used its intended path (frame counters), and
//   * the pixel hashes to match (the parallel path must be pixel-identical to
//     the serial secondary path).
// PAR_HASH=<hex> is printed per pass for manual cross-run comparison.

#include "VulkanTestHarness.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace vulkan_test;

// FNV-1a 64-bit over the readback, printed for cross-run comparison.
static uint64_t
hashPixels(const std::vector<uint8_t> & pixels)
{
  uint64_t h = 1469598103934665603ULL;
  for (const uint8_t b : pixels) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

namespace {

// Builds the heavy mixed scene.  Geometry is static so the same draw list
// shape can be rebuilt for each recording mode.
SoDrawList buildScene()
{
  // Shared unit triangle geometry; every opaque command below instantiates
  // it with its own model matrix, so the bucket groups them into instanced
  // batches (content-identical, material-identical, model-only
  // differences).
  static const float tri[] = {
    -0.02f, -0.02f, 0.0f,
     0.02f, -0.02f, 0.0f,
     0.00f,  0.02f, 0.0f
  };
  static const float quadPos[] = {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
    -0.5f,  0.5f, 0.0f,
     0.5f,  0.5f, 0.0f
  };
  static const uint32_t quadIdx[] = {0, 1, 2, 2, 1, 3};

  SoDrawList drawlist;

  // 240 batchable opaque triangles (shared geometry + material, per-command
  // model translation only) spread over the viewport at varied depths.
  for (int i = 0; i < 240; ++i) {
    SoRenderCommand c = makeTriangle(tri);
    SbMatrix m;
    m.makeIdentity();
    const float x = -0.9f + (static_cast<float>(i % 20) / 20.0f) * 1.8f;
    const float y = -0.9f + (static_cast<float>((i / 20) % 12) / 12.0f) * 1.8f;
    const float z = -static_cast<float>(i) * 0.001f;
    m[3][0] = x;
    m[3][1] = y;
    m[3][2] = z;
    c.modelMatrix = m;
    c.material.diffuse = SbVec4f(0.8f, 0.3f, 0.1f, 1.0f);
    c.state.depth.enabled = true;
    c.state.depth.writeEnabled = true;
    drawlist.addCommand(c);
  }

  // 100 unique-material opaque triangles (singles: material participates in
  // the batch key, so each color breaks the batch into its own item -- this
  // pushes the recordToSecondary item count past the 64-item parallel
  // threshold so the parallel recorder actually dispatches workers).
  for (int i = 0; i < 100; ++i) {
    SoRenderCommand c = makeTriangle(tri);
    SbMatrix m;
    m.makeIdentity();
    const float x = -0.9f + (static_cast<float>(i % 10) / 10.0f) * 1.8f;
    const float y = -0.9f + (static_cast<float>((i / 10) % 10) / 10.0f) * 1.8f;
    const float z = 0.5f - static_cast<float>(i) * 0.004f;
    m[3][0] = x;
    m[3][1] = y;
    m[3][2] = z;
    c.modelMatrix = m;
    c.material.diffuse =
      SbVec4f(static_cast<float>(i % 7) / 7.0f,
              static_cast<float>((i / 7) % 7) / 7.0f,
              0.9f, 1.0f);
    c.state.depth.enabled = true;
    c.state.depth.writeEnabled = true;
    drawlist.addCommand(c);
  }

  // Two transparent quads (painter order, recorded inline after the
  // secondaries).
  for (int i = 0; i < 2; ++i) {
    SoRenderCommand q;
    q.modelMatrix.makeIdentity();
    q.geometry.topology = SO_TOPOLOGY_TRIANGLES;
    q.geometry.vertexCount = 4;
    q.geometry.indexCount = 6;
    q.geometry.positions = quadPos;
    q.geometry.indices = quadIdx;
    q.geometry.vertexStride = sizeof(float) * 3;
    q.material.diffuse =
      (i == 0) ? SbVec4f(0.2f, 0.9f, 0.3f, 0.4f)
               : SbVec4f(0.1f, 0.2f, 0.9f, 0.4f);
    q.pass = SO_RENDERPASS_TRANSPARENT;
    drawlist.addCommand(q);
  }

  // Depth-off on-top annotation (recorded inline after everything else).
  {
    SoRenderCommand a = makeTriangle(tri);
    SbMatrix m;
    m.makeIdentity();
    m[3][0] = 0.3f;
    m[3][1] = -0.3f;
    a.modelMatrix = m;
    a.material.diffuse = SbVec4f(1.0f, 1.0f, 0.0f, 1.0f);
    a.state.depth.enabled = false;
    drawlist.addCommand(a);
  }

  return drawlist;
}

// Renders \a drawlist through \a harness for a few frames, readbacks, and
// returns the pixel hash.  Accumulates failures in \a failures (render
// failures, black frame).
uint64_t renderHashed(Harness & harness, const SoDrawList & drawlist,
                      int & failures)
{
  const SoRenderParams params = harness.renderParams();
  for (int frame = 0; frame < 4; ++frame) {
    if (!harness.backend.render(drawlist, params)) {
      std::cerr << "FAIL: render failed on frame " << frame << std::endl;
      ++failures;
    }
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const uint64_t hash = hashPixels(pixels);
  std::printf("PAR_HASH=%016llx\n",
              static_cast<unsigned long long>(hash));

  // Sanity: the frame must not be black (a lost-draw / failed-record
  // signal).
  int nonBlack = 0;
  for (size_t i = 0; i < pixels.size(); i += 4) {
    if (pixels[i] || pixels[i + 1] || pixels[i + 2]) ++nonBlack;
  }
  if (nonBlack == 0) {
    std::cerr << "FAIL: frame is entirely black" << std::endl;
    ++failures;
  }
  return hash;
}

} // namespace

int
main()
{
  int failures = 0;

  Harness harness;
  harness.wantNestedCommandBuffer = true;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  // Without the extension the backend cannot execute secondaries inside the
  // subpass: both passes would be fully inline and the hash comparison would
  // prove nothing.  Skip loudly rather than report a vacuous pass.
  if (!harness.haveNestedCommandBuffer) {
    return skip("device lacks VK_EXT_nested_command_buffer; the secondary / "
                "parallel recorder cannot be exercised");
  }

  const SoDrawList drawlist = buildScene();

  // Pass 1: M1c serial secondary.
  const uint64_t secondaryBefore = harness.backend.secondaryRecordFrameCount();
  harness.backend.setParallelRecordEnabled(FALSE);
  const uint64_t serialHash = renderHashed(harness, drawlist, failures);
  const uint64_t secondaryAfter = harness.backend.secondaryRecordFrameCount();
  if (secondaryAfter == secondaryBefore) {
    std::cerr << "FAIL: the serial pass did not use the secondary (M1c) path"
              << std::endl;
    ++failures;
  }
  if (harness.backend.parallelRecordFrameCount() != 0) {
    std::cerr << "FAIL: the serial pass used the parallel (M1d) path"
              << std::endl;
    ++failures;
  }

  // Pass 2: M1d parallel.
  const uint64_t parallelBefore = harness.backend.parallelRecordFrameCount();
  harness.backend.setParallelRecordEnabled(TRUE);
  const uint64_t parallelHash = renderHashed(harness, drawlist, failures);
  const uint64_t parallelAfter = harness.backend.parallelRecordFrameCount();
  if (parallelAfter == parallelBefore) {
    std::cerr << "FAIL: the parallel pass did not engage the parallel (M1d) "
                 "recorder; it silently fell back (see errors above)"
              << std::endl;
    ++failures;
  }

  if (serialHash != parallelHash) {
    std::cerr << "FAIL: parallel path differs from the serial path "
              "(serial=" << std::hex << serialHash << " parallel="
              << parallelHash << std::dec << ")" << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
