#include <Inventor/SoDB.h>
#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoAlphaTest.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoDepthBuffer.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoMaterialBinding.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTexture2.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool nearlyEqual(float lhs, float rhs)
{
  return std::fabs(lhs - rhs) < 0.0001f;
}

}

int
runTest()
{
  SoDB::init();

  SoSeparator * root = new SoSeparator;
  root->ref();

  SoDepthBuffer * depth = new SoDepthBuffer;
  depth->test = TRUE;
  depth->write = FALSE;
  depth->function = SoDepthBuffer::GREATER;
  depth->range = SbVec2f(0.2f, 0.8f);
  root->addChild(depth);

  SoAlphaTest * alphaTest = new SoAlphaTest;
  alphaTest->function = SoAlphaTest::GREATER;
  alphaTest->value = 0.25f;
  root->addChild(alphaTest);

  const unsigned char texturePixel[] = { 255, 128, 64, 255 };
  SoTexture2 * texture = new SoTexture2;
  texture->image.setValue(SbVec2s(1, 1), 4, texturePixel);
  texture->wrapS = SoTexture2::REPEAT;
  texture->wrapT = SoTexture2::CLAMP;
  texture->model = SoTexture2::BLEND;
  texture->blendColor.setValue(0.1f, 0.2f, 0.3f);
  root->addChild(texture);

  SoDirectionalLight * light = new SoDirectionalLight;
  light->color.setValue(0.25f, 0.5f, 0.75f);
  root->addChild(light);
  for (int i = 0; i < 8; ++i) {
    root->addChild(new SoDirectionalLight);
  }

  SoMaterial * material = new SoMaterial;
  material->ambientColor.setValue(0.1f, 0.2f, 0.3f);
  material->diffuseColor.setValue(0.4f, 0.5f, 0.6f);
  material->specularColor.setValue(0.7f, 0.8f, 0.9f);
  material->emissiveColor.setValue(0.15f, 0.1f, 0.05f);
  material->shininess = 0.5f;
  material->transparency = 0.25f;
  root->addChild(material);
  root->addChild(new SoCube);

  SoIRRenderAction action(SbViewportRegion(64, 64));
  action.apply(root);

  int result = 0;
  if (action.getDrawList().getNumCommands() != 1) {
    std::cerr << "FAIL: material scene did not emit one retained command" << std::endl;
    result = 1;
  }
  else {
    const SoRenderCommand & command = action.getDrawList().getCommand(0);
    if (!command.state.depth.enabled || command.state.depth.writeEnabled ||
        command.state.depth.func != SO_DEPTH_GREATER ||
        !nearlyEqual(command.state.depth.range[0], 0.2f) ||
        !nearlyEqual(command.state.depth.range[1], 0.8f)) {
      std::cerr << "FAIL: SoDepthBuffer state was not retained" << std::endl;
      result = 1;
    }
    if (command.state.alphaTest.function != SO_ALPHA_TEST_GREATER ||
        !nearlyEqual(command.state.alphaTest.reference, 0.25f)) {
      std::cerr << "FAIL: SoAlphaTest state was not retained" << std::endl;
      result = 1;
    }
    if (command.material.texture.wrapS != SO_TEXTURE_WRAP_REPEAT ||
        command.material.texture.wrapT != SO_TEXTURE_WRAP_CLAMP_TO_EDGE ||
        command.material.texture.model != SO_TEXTURE_MODEL_BLEND ||
        !nearlyEqual(command.material.texture.blendColor[1], 0.2f) ||
        command.material.texture.minFilter != SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR ||
        command.material.texture.magFilter != SO_TEXTURE_FILTER_LINEAR ||
        command.material.texture.pixels == nullptr) {
      std::cerr << "FAIL: texture model and sampler state were not retained" << std::endl;
      result = 1;
    }
    if (!nearlyEqual(command.material.ambient[0], 0.1f) ||
        !nearlyEqual(command.material.diffuse[1], 0.5f) ||
        !nearlyEqual(command.material.specular[2], 0.9f) ||
        !nearlyEqual(command.material.diffuse[3], 0.75f) ||
        !nearlyEqual(command.material.emissive[0], 0.15f)) {
      std::cerr << "FAIL: complete material state was not retained" << std::endl;
      result = 1;
    }
    const SoLightingData * lighting = action.getDrawList().getLighting(command.lightingHandle);
    if (!lighting || lighting->lights.size() != 9 ||
        !nearlyEqual(lighting->lights[0].color[1], 0.5f)) {
      std::cerr << "FAIL: SoLight state was not retained" << std::endl;
      result = 1;
    }
  }

  root->unref();

  struct BindingCase {
    SoMaterialBinding::Binding binding;
    bool indexed;
    const char * name;
  };
  const BindingCase cases[] = {
    { SoMaterialBinding::PER_PART, false, "PER_PART" },
    { SoMaterialBinding::PER_PART_INDEXED, true, "PER_PART_INDEXED" },
    { SoMaterialBinding::PER_FACE, false, "PER_FACE" },
    { SoMaterialBinding::PER_FACE_INDEXED, true, "PER_FACE_INDEXED" },
    { SoMaterialBinding::PER_VERTEX, false, "PER_VERTEX" },
    { SoMaterialBinding::PER_VERTEX_INDEXED, true, "PER_VERTEX_INDEXED" }
  };

  const SbVec3f points[] = {
    SbVec3f(-1.0f, -1.0f, 0.0f), SbVec3f(0.0f, -1.0f, 0.0f),
    SbVec3f(-0.5f,  0.0f, 0.0f), SbVec3f(0.0f,  0.0f, 0.0f),
    SbVec3f(1.0f,  0.0f, 0.0f), SbVec3f(0.5f,  1.0f, 0.0f)
  };
  const int32_t coordIndices[] = { 0, 1, 2, -1, 3, 4, 5, -1 };
  const int32_t faceMaterialIndices[] = { 0, 1 };
  const int32_t vertexMaterialIndices[] = { 0, 1, 2, 3, 4, 5 };
  const SbColor materialColors[] = {
    SbColor(1.0f, 0.0f, 0.0f), SbColor(0.0f, 1.0f, 0.0f),
    SbColor(0.0f, 0.0f, 1.0f), SbColor(1.0f, 1.0f, 0.0f),
    SbColor(1.0f, 0.0f, 1.0f), SbColor(0.0f, 1.0f, 1.0f)
  };
  const float materialTransparency[] = { 0.0f, 0.5f, 0.0f,
                                          0.5f, 0.0f, 0.5f };

  for (const BindingCase & bindingCase : cases) {
    SoSeparator * batchRoot = new SoSeparator;
    batchRoot->ref();

    SoCoordinate3 * coordinates = new SoCoordinate3;
    coordinates->point.setValues(0, 6, points);
    batchRoot->addChild(coordinates);

    SoMaterial * batchMaterial = new SoMaterial;
    batchMaterial->diffuseColor.setValues(0, 6, materialColors);
    batchMaterial->transparency.setValues(0, 6, materialTransparency);
    batchRoot->addChild(batchMaterial);

    SoMaterialBinding * materialBinding = new SoMaterialBinding;
    materialBinding->value = bindingCase.binding;
    batchRoot->addChild(materialBinding);

    SoIndexedFaceSet * faces = new SoIndexedFaceSet;
    faces->coordIndex.setValues(0, 8, coordIndices);
    if (bindingCase.binding == SoMaterialBinding::PER_FACE_INDEXED ||
        bindingCase.binding == SoMaterialBinding::PER_PART_INDEXED) {
      faces->materialIndex.setValues(0, 2, faceMaterialIndices);
    }
    else if (bindingCase.binding == SoMaterialBinding::PER_VERTEX_INDEXED) {
      faces->materialIndex.setValues(0, 6, vertexMaterialIndices);
    }
    const std::vector<int32_t> originalCoordIndices(
      faces->coordIndex.getValues(0), faces->coordIndex.getValues(0) + 8);
    const int originalMaterialIndexCount = faces->materialIndex.getNum();
    const std::vector<int32_t> originalMaterialIndices = originalMaterialIndexCount
      ? std::vector<int32_t>(faces->materialIndex.getValues(0),
                             faces->materialIndex.getValues(0) + originalMaterialIndexCount)
      : std::vector<int32_t>();
    batchRoot->addChild(faces);

    SoIRRenderAction batchAction(SbViewportRegion(64, 64));
    batchAction.apply(batchRoot);

    bool hasGeneratedColors = false;
    bool hasTransparentBatch = false;
    const int expectedCommands =
      (bindingCase.binding == SoMaterialBinding::PER_PART ||
       bindingCase.binding == SoMaterialBinding::PER_PART_INDEXED ||
      (bindingCase.binding == SoMaterialBinding::PER_FACE ||
       bindingCase.binding == SoMaterialBinding::PER_FACE_INDEXED)) ? 2 : 1;
    if (batchAction.getDrawList().getNumCommands() != expectedCommands) {
      std::cerr << "FAIL: " << bindingCase.name
                << " did not produce two material batches" << std::endl;
      result = 1;
    }
    for (int i = 0; i < batchAction.getDrawList().getNumCommands(); ++i) {
      const SoRenderCommand & command = batchAction.getDrawList().getCommand(i);
      if (command.geometry.colors) {
        hasGeneratedColors = true;
        if (!command.material.vertexColorAlphaIncludesOpacity) {
          std::cerr << "FAIL: " << bindingCase.name
                    << " lost generated vertex alpha policy" << std::endl;
          result = 1;
        }
      }
      if (command.pass == SO_RENDERPASS_TRANSPARENT) {
        hasTransparentBatch = true;
      }
    }
    if ((bindingCase.binding == SoMaterialBinding::PER_VERTEX ||
         bindingCase.binding == SoMaterialBinding::PER_VERTEX_INDEXED) &&
        !hasGeneratedColors) {
      std::cerr << "FAIL: " << bindingCase.name
                << " did not retain per-vertex colors" << std::endl;
      result = 1;
    }
    if (!hasTransparentBatch) {
      std::cerr << "FAIL: " << bindingCase.name
                << " did not classify transparent material data" << std::endl;
      result = 1;
    }

    if (std::vector<int32_t>(faces->coordIndex.getValues(0),
                             faces->coordIndex.getValues(0) + 8) != originalCoordIndices ||
        faces->materialIndex.getNum() != originalMaterialIndexCount ||
        (originalMaterialIndexCount &&
         std::vector<int32_t>(faces->materialIndex.getValues(0),
                              faces->materialIndex.getValues(0) + originalMaterialIndexCount)
           != originalMaterialIndices)) {
      std::cerr << "FAIL: " << bindingCase.name
                << " modified the source scene graph" << std::endl;
      result = 1;
    }

    batchRoot->unref();
  }

  return result;
}

int
main()
{
  const int result = runTest();
  SoDB::finish();
  return result;
}
