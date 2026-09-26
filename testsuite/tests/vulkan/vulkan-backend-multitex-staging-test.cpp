// testsuite/vulkan-backend-multitex-staging-test.cpp
//
// Verifies the shared staging pool coalesces multiple concurrent texture
// uploads in one frame without aliasing: several quads, each sampling a
// different-colored 1x1 texel, must each render their own color.  This
// exercises the per-frame stagingPoolCursor offset math in
// prepareTextureUpload() with distinct pixel payloads.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float texcoords[] = {
  0.0f, 0.0f,
  1.0f, 0.0f,
  1.0f, 1.0f,
  0.0f, 1.0f
};
const uint32_t indices[] = {0, 1, 2, 0, 2, 3};
// Two disjoint half-window quads (persistent storage: the backend borrows the
// position pointers for the frame).  Left spans x=[-1,0], right spans x=[0,1].
const float leftQuad[] = {
  -1.0f, -1.0f, 0.0f,
   0.0f, -1.0f, 0.0f,
   0.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const float rightQuad[] = {
   0.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
   0.0f,  1.0f, 0.0f
};

// A quad covering the left or right half of the window, sampling a solid-
// colored texel.  The two halves use disjoint half-window coordinates so the
// quads cannot overlap regardless of model matrix.
SoRenderCommand coloredHalfQuad(bool leftHalf, const unsigned char * texel)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = leftHalf ? leftQuad : rightQuad;
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 2;
  command.geometry.indices = indices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
  command.material.texture.pixels = texel;
  command.material.texture.width = 1;
  command.material.texture.height = 1;
  command.material.texture.numComponents = 4;
  command.material.texture.model = SO_TEXTURE_MODEL_REPLACE;
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

  static const unsigned char red[] = {255, 0, 0, 255};
  static const unsigned char green[] = {0, 255, 0, 255};

  SoDrawList drawlist;
  drawlist.addCommand(coloredHalfQuad(true, red));
  drawlist.addCommand(coloredHalfQuad(false, green));
  if (!harness.backend.render(drawlist, params)) {
    std::cerr << "FAIL: multi-texture staging render failed" << std::endl;
    ++failures;
  }

  const auto & pixels = harness.readback();
  // Left half center should be red; right half center should be green.  If the
  // two uploads aliased to the same staging offset, both halves would read the
  // same texel and one color check would fail.
  const uint8_t * left = pixelAt(pixels, 8, 16);
  const uint8_t * right = pixelAt(pixels, 24, 16);
  if (!nearColor(left, 255, 0, 0)) {
    std::cerr << "FAIL: left half did not sample its own red texel" << std::endl;
    ++failures;
  }
  if (!nearColor(right, 0, 255, 0)) {
    std::cerr << "FAIL: right half did not sample its own green texel"
              << std::endl;
    ++failures;
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
