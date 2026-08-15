// testsuite/vulkan-backend-text-overlay-test.cpp
//
// Exercises the pixel-text (SoText2-style overlay) rendering path.  Pixel text
// is CPU-rasterized by the producer and must be emitted verbatim by the
// backend, matching the legacy glDrawPixels behavior instead of being
// modulated by material diffuse (which would double-tint and darken text).

#include "VulkanTestHarness.h"

#include <Inventor/SoDB.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/rendering/SoVulkanRenderManager.h>

using namespace vulkan_test;

namespace {

// File-scope storage so the returned command's geometry pointers stay valid
// for the lifetime of the render (a stack-local array would dangle).
const float kQuadPositions[] = {
  -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   1.0f,  1.0f, 0.0f,
  -1.0f,  1.0f, 0.0f
};
const float kQuadTexcoords[] = {
  0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f
};
const uint32_t kQuadIndices[] = {0, 1, 2, 0, 2, 3};

// File-scope RGBA pixels (red, fully opaque) so the producer-owned texture
// buffer stays valid for the lifetime of the render.
unsigned char kPixelTextPixels[4 * 4 * 4] = {0};

// A full-viewport quad (two triangles) carrying embedded RGBA pixels.
SoRenderCommand pixelTextQuad(const unsigned char * pixels, int w, int h)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  command.geometry.vertexCount = 4;
  command.geometry.indexCount = 6;
  command.geometry.vertexStride = sizeof(float) * 3;
  command.geometry.texcoordStride = sizeof(float) * 2;
  command.geometry.positions = kQuadPositions;
  command.geometry.texcoords = kQuadTexcoords;
  command.geometry.indices = kQuadIndices;
  command.material.flags |= SO_MAT_HAS_TEXTURE | SO_MAT_IS_PIXEL_TEXT;
  command.material.texture.pixels = pixels;
  command.material.texture.width = w;
  command.material.texture.height = h;
  command.material.texture.numComponents = 4;
  command.material.texture.model = SO_TEXTURE_MODEL_REPLACE;
  command.material.textureAlphaIncludesOpacity = true;
  command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
  command.pixelText.originX = 0;
  command.pixelText.originY = 0;
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

  // 1) Synthetic pixel-text command: the texture is pure red (255,0,0,255)
  //    but the material diffuse is white, so any modulation is detectable.
  //    The backend must emit the texture color verbatim.
  {
    for (int i = 0; i < 4 * 4; ++i) {
      kPixelTextPixels[i * 4 + 0] = 255;
      kPixelTextPixels[i * 4 + 1] = 0;
      kPixelTextPixels[i * 4 + 2] = 0;
      kPixelTextPixels[i * 4 + 3] = 255;
    }

    SoDrawList drawlist;
    drawlist.addCommand(pixelTextQuad(kPixelTextPixels, 4, 4));
    if (!harness.backend.render(drawlist, harness.renderParams())) {
      std::cerr << "FAIL: synthetic pixel-text render failed" << std::endl;
      ++failures;
    }
    else {
      const std::vector<uint8_t> framebuffer = harness.readback();
      const uint8_t * center = pixelAt(framebuffer, 16, 16);
      if (!nearColor(center, 255, 0, 0)) {
        std::cerr << "FAIL: pixel text was not emitted verbatim (R="
                  << static_cast<int>(center[2])
                  << " G=" << static_cast<int>(center[1])
                  << " B=" << static_cast<int>(center[0]) << ")" << std::endl;
        ++failures;
      }
    }
  }

  // 2) Real SoText2 scene graph: red glyphs must be visible in the center of
  //    the framebuffer after the full scene traversal.
  {
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

    SoFont * font = new SoFont;
    font->name.setValue("defaultFont");
    font->size = 0.7f;
    root->addChild(font);

    SoMaterial * material = new SoMaterial;
    material->diffuseColor.setValue(1.0f, 0.0f, 0.0f);
    root->addChild(material);

    SoText2 * text = new SoText2;
    text->string.set1Value(0, "ABC");
    text->justification = SoText2::CENTER;
    root->addChild(text);

    SoVulkanRenderManager manager;
    manager.setSceneGraph(root);
    manager.setCamera(camera);
    SbViewportRegion viewport(kWidth, kHeight);
    viewport.setViewportPixels(SbVec2s(0, 0), SbVec2s(kWidth, kHeight));
    manager.setViewportRegion(viewport);
    manager.setBackgroundColor(SbColor4f(0.0f, 0.0f, 0.0f, 1.0f));
    manager.setRenderTarget(&harness.target);

    if (!manager.initialize(&harness.deviceContext)) {
      std::cerr << "FAIL: manager could not initialize" << std::endl;
      ++failures;
    }
    else if (!manager.render(TRUE, TRUE)) {
      std::cerr << "FAIL: SoText2 manager render failed" << std::endl;
      ++failures;
    }
    else {
      const std::vector<uint8_t> framebuffer = harness.readback();
      const int redPixels = countNear(framebuffer, 255, 0, 0);
      if (redPixels == 0) {
        std::cerr << "FAIL: SoText2 glyphs produced no red pixels"
                  << std::endl;
        ++failures;
      }
    }

    root->unref();
  }

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
