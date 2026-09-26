// testsuite/vulkan/vulkan-backend-yflip-test.cpp
//
// Two focused checks on the visual pass:
//
//  1. Y convention.  The visual vertex shader flips clip.y ("Coin/OpenGL uses
//     a bottom-left origin; Vulkan uses top-left").  A quad placed in the
//     upper NDC half (eye y > 0) must land in the top half of the framebuffer
//     (GL-compatible: scene-up -> image-up) and NOT be mirrored.  Renders two
//     disjoint unlit quads with identity view/projection and reports which
//     half of the image each lands in.
//
//  2. Ambient term.  Isolates the retained model's ambient contribution
//     (litColor = u_ambientLight * u_materialAmbient, before any light loop).
//     A camera-facing quad with a GOURAUD model, zero lights, white material
//     ambient and a red scene ambient must come out red -- black would mean
//     the ambient term is dropped.

#include <vulkan/vulkan.h>

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

// Borrowed geometry pointers must stay valid until the backend consumes the
// frame, so the vertex data lives at namespace scope (like the other
// backend tests).
float redQuadPos[12];
float greenQuadPos[12];
float ambientQuadPos[12];
const uint32_t quadIndices[] = {0, 1, 2, 0, 2, 3};
const float quadNormals[] = {
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 0.0f, 1.0f
};

void fillQuad(float * pos, float y0, float y1)
{
  pos[0] = -0.9f;  pos[1] = y0;      pos[2] = 0.0f;
  pos[3] =  0.9f;  pos[4] = y0;      pos[5] = 0.0f;
  pos[6] =  0.9f;  pos[7] = y1;      pos[8] = 0.0f;
  pos[9] = -0.9f;  pos[10] = y1;     pos[11] = 0.0f;
}

// Unlit quad spanning x in [-0.9, 0.9] and y in [y0, y1] (NDC, since the
// harness uses identity view/projection matrices).
SoRenderCommand unlitQuad(const float * pos)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.positions = pos;
  command.geometry.indices = quadIndices;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.state.depth.enabled = FALSE;
  command.state.depth.writeEnabled = FALSE;
  return command;
}

int runYFlipCheck(Harness & harness)
{
  SoDrawList drawlist;
  fillQuad(redQuadPos, 0.2f, 0.9f);
  fillQuad(greenQuadPos, -0.9f, -0.2f);
  SoRenderCommand red = unlitQuad(redQuadPos);
  red.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  SoRenderCommand green = unlitQuad(greenQuadPos);
  green.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);
  drawlist.addCommand(red);
  drawlist.addCommand(green);

  const SoRenderParams params = harness.renderParams();
  if (!harness.backend.render(drawlist, params)) {
    std::cerr << "FAIL: render failed" << std::endl;
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  if (getenv("YFLIP_VERBOSE")) {
    for (int y = 0; y < 32; ++y) {
      std::string row;
      for (int x = 0; x < 32; ++x) {
        const uint8_t * p = pixelAt(pixels, x, y);
        row += nearColor(p, 255, 0, 0) ? 'R'
               : nearColor(p, 0, 255, 0) ? 'G'
               : nearColor(p, 255, 255, 0) ? 'Y'
               : (p[0] | p[1] | p[2]) ? '.' : ' ';
      }
      std::cout << "row " << y << ": [" << row << "]" << std::endl;
    }
  }
  const uint8_t * top = pixelAt(pixels, 16, 6);
  const uint8_t * bottom = pixelAt(pixels, 16, 25);

  const bool topRed = nearColor(top, 255, 0, 0);
  const bool topGreen = nearColor(top, 0, 255, 0);
  const bool bottomRed = nearColor(bottom, 255, 0, 0);
  const bool bottomGreen = nearColor(bottom, 0, 255, 0);

  if (topRed && bottomGreen) {
    std::cout << "YFLIP: scene-up (+Y) renders at the TOP of the image "
                 "(GL-compatible, not mirrored)" << std::endl;
    return 0;
  }
  if (topGreen && bottomRed) {
    std::cout << "YFLIP: scene-up (+Y) renders at the BOTTOM of the image "
                 "(vertically mirrored)" << std::endl;
    return 2;
  }
  std::cerr << "FAIL(yflip): unexpected pattern (top red=" << topRed
            << " green=" << topGreen << "; bottom red=" << bottomRed
            << " green=" << bottomGreen << ")" << std::endl;
  return 1;
}

int runAmbientCheck(Harness & harness)
{
  fillQuad(ambientQuadPos, -0.9f, 0.9f);
  SoRenderCommand command = unlitQuad(ambientQuadPos);
  command.geometry.normals = quadNormals;
  command.material.diffuse = SbVec4f(0.0f, 1.0f, 0.0f, 1.0f);   // green
  command.material.ambient = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);   // white
  command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.specular = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
  command.material.shadingModel = SO_SHADING_LEGACY_GOURAUD;

  SoDrawList drawlist;
  SoLightingData lighting;
  lighting.ambient = SbVec3f(1.0f, 0.0f, 0.0f);  // red, no lights
  command.lightingHandle = drawlist.addLightingSetup(lighting);
  drawlist.addCommand(command);

  const SoRenderParams params = harness.renderParams();
  if (!harness.backend.render(drawlist, params)) {
    std::cerr << "FAIL: ambient render failed" << std::endl;
    return 1;
  }

  const std::vector<uint8_t> pixels = harness.readback();
  const uint8_t * center = pixelAt(pixels, 16, 16);
  std::cout << "AMBIENT center: R=" << (int)center[2]
            << " G=" << (int)center[1] << " B=" << (int)center[0]
            << std::endl;

  // Ambient only: red ambient * white material ambient = red.  A black
  // result means the ambient term is being dropped.
  if (nearColor(center, 255, 0, 0)) {
    std::cout << "AMBIANT: ambient term applied (red as expected)"
              << std::endl;
    return 0;
  }
  std::cerr << "FAIL(ambient): expected red (255,0,0) from the ambient "
               "term, got R=" << (int)center[2] << " G=" << (int)center[1]
            << " B=" << (int)center[0] << std::endl;
  return 1;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = runYFlipCheck(harness);
  failures += runAmbientCheck(harness);

  harness.shutdown();
  return failures == 0 ? 0 : 1;
}
