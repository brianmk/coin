// testsuite/material-block-pack-test.cpp
//
// Host contract tests for the consolidated material mapping every retained
// backend shares: SoRenderIR::packMaterialBlock, the SoMaterialBlock layout,
// and SoRenderIR::resolveOptical.
//
// The raster Vulkan backend stages these fields into its per-draw DrawBlock
// UBO and the ray tracer copies them into its RTMaterial record, so a silent
// field remap here shades wrongly in one or both backends.  These tests pin:
//   * the packing defaults and the optical/transmission mapping,
//   * the optional-map presence bitmask (including zero-extent maps, which
//     must count as absent so the shaders never sample a null/empty texture),
//   * per-field offsets (a size-only guard passes even if two fields swap),
//   * a byte-exact compare against a hand-built expected block,
//   * the global-vs-authored optics precedence (resolveOptical).
//
// No GPU, backend or window system is involved.

#include <Inventor/rendering/SoRenderIR.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void
check(bool condition, const std::string & message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    ++failures;
  }
}

void
checkFloat(float actual, float expected, const std::string & message)
{
  if (actual != expected) {
    std::ostringstream os;
    os << message << " (actual=" << actual << ", expected=" << expected << ")";
    check(false, os.str());
  }
}

SoTextureData
makeMap(int width, int height, int components, unsigned char * pixels)
{
  SoTextureData texture;
  texture.pixels = pixels;
  texture.width = width;
  texture.height = height;
  texture.numComponents = components;
  return texture;
}

}  // namespace

int
main()
{
  // --- Compile-time layout locks ------------------------------------------
  // SoRenderIR.h already static_asserts these; repeating them here keeps the
  // failure visible in the test log and documents the contract.
  static_assert(sizeof(SoRenderIR::SoMaterialBlock) == 128,
                "SoMaterialBlock must be 8 tightly packed vec4");
  static_assert(offsetof(SoRenderIR::SoMaterialBlock, pbr) == 80,
                "SoMaterialBlock.pbr offset");
  static_assert(offsetof(SoRenderIR::SoMaterialBlock, mapParams) == 96,
                "SoMaterialBlock.mapParams offset");
  static_assert(offsetof(SoRenderIR::SoMaterialBlock, optical) == 112,
                "SoMaterialBlock.optical offset");

  check(offsetof(SoRenderIR::SoMaterialBlock, diffuse) == 0, "diffuse offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, ambient) == 16, "ambient offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, specular) == 32,
        "specular offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, emissive) == 48,
        "emissive offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, params) == 64, "params offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, pbr) == 80, "pbr offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, mapParams) == 96,
        "mapParams offset");
  check(offsetof(SoRenderIR::SoMaterialBlock, optical) == 112,
        "optical offset");

  // --- Packing defaults ---------------------------------------------------
  {
    SoMaterialData material;
    SoRenderIR::SoMaterialBlock block;
    SoRenderIR::packMaterialBlock(block, material);

    checkFloat(block.diffuse[0], 0.8f, "default diffuse");
    checkFloat(block.diffuse[3], 1.0f, "default diffuse alpha");
    checkFloat(block.ambient[3], 1.0f, "ambient alpha forced opaque");
    checkFloat(block.specular[3], 1.0f, "specular alpha forced opaque");
    checkFloat(block.emissive[3], 1.0f, "emissive alpha forced opaque");
    checkFloat(block.params[0], 0.2f, "default shininess");
    checkFloat(block.params[2], 0.0f, "light count is caller-filled");

    checkFloat(block.pbr[0], 0.0f, "default metalness");
    checkFloat(block.pbr[1], 0.5f, "default roughness");
    checkFloat(block.pbr[2], 0.0f, "physical material off by default");
    checkFloat(block.pbr[3], 0.0f, "pbr reserved slot");

    checkFloat(block.mapParams[0], 1.0f, "default roughnessStrength");
    checkFloat(block.mapParams[1], 1.0f, "default normalStrength");
    checkFloat(block.mapParams[2], 1.0f, "default emissiveIntensity");
    checkFloat(block.mapParams[3], 0.0f, "no maps present by default");

    checkFloat(block.optical[0], 1.5f, "default IOR");
    checkFloat(block.optical[1], 0.0f, "default absorption");
    checkFloat(block.optical[2], 1.0f, "transmittance == opacity default");
    checkFloat(block.optical[3], 0.0f, "optics not authored by default");
  }

  // --- Map-presence bitmask ----------------------------------------------
  {
    unsigned char pixels[4] = {1, 2, 3, 255};
    SoMaterialData material;
    material.roughnessTexture = makeMap(2, 2, 1, pixels);

    SoRenderIR::SoMaterialBlock block;
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.mapParams[3], 1.0f, "roughness-only -> bit 0");

    material.normalTexture = makeMap(2, 2, 3, pixels);
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.mapParams[3], 3.0f, "roughness+normal -> bits 0,1");

    material.emissiveTexture = makeMap(2, 2, 3, pixels);
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.mapParams[3], 7.0f, "all three maps -> bits 0,1,2");

    // A map with a pixel pointer but a zero extent must count as absent: the
    // shaders would otherwise sample a zero-width/zero-component texture.
    SoTextureData emptyButPointing = makeMap(0, 0, 1, pixels);
    material.roughnessTexture = emptyButPointing;
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.mapParams[3], 6.0f,
               "zero width/height + pointer counts absent");

    material.roughnessTexture = makeMap(2, 2, 0, pixels);
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.mapParams[3], 6.0f,
               "zero numComponents + pointer counts absent");

    material.normalTexture = SoTextureData {};
    material.emissiveTexture = SoTextureData {};
    material.roughnessTexture = makeMap(2, 2, 1, pixels);
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.mapParams[3], 1.0f, "back to roughness-only");
  }

  // --- Optical mapping ----------------------------------------------------
  {
    SoMaterialData material;
    material.opacity = 0.6f;
    material.transmissionIor = 1.7f;
    material.transmissionAbsorption = 0.4f;

    SoRenderIR::SoMaterialBlock block;
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.optical[2], material.opacity, "optical[2] == opacity");
    checkFloat(block.optical[3], 0.0f,
               "optical[3] false when not authored");

    material.transmissionAuthored = true;
    SoRenderIR::packMaterialBlock(block, material);
    checkFloat(block.optical[3], 1.0f,
               "optical[3] true when transmissionAuthored");
  }

  // --- Byte-exact full packing -------------------------------------------
  {
    SoMaterialData material;
    material.diffuse.setValue(0.1f, 0.2f, 0.3f, 0.4f);
    material.ambient.setValue(0.5f, 0.6f, 0.7f, 0.8f);
    material.specular.setValue(0.9f, 0.11f, 0.12f, 0.13f);
    material.emissive.setValue(0.14f, 0.15f, 0.16f, 0.17f);
    material.shininess = 0.75f;
    material.twoSidedLighting = true;
    material.shadingModel = SO_SHADING_LEGACY_GOURAUD;
    material.metalness = 0.85f;
    material.roughness = 0.21f;
    material.physicalMaterial = true;
    material.roughnessStrength = 0.6f;
    material.normalStrength = 0.3f;
    material.emissiveIntensity = 2.5f;
    unsigned char pixels[4] = {1, 2, 3, 255};
    material.roughnessTexture = makeMap(2, 2, 1, pixels);
    material.normalTexture = makeMap(2, 2, 3, pixels);
    material.transmissionIor = 1.7f;
    material.transmissionAbsorption = 0.4f;
    material.opacity = 0.6f;
    material.transmissionAuthored = true;

    SoRenderIR::SoMaterialBlock packed;
    SoRenderIR::packMaterialBlock(packed, material);

    SoRenderIR::SoMaterialBlock expected;
    std::memset(&expected, 0, sizeof(expected));
    const float diffuse[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float ambient[4] = {0.5f, 0.6f, 0.7f, 1.0f};
    const float specular[4] = {0.9f, 0.11f, 0.12f, 1.0f};
    const float emissive[4] = {0.14f, 0.15f, 0.16f, 1.0f};
    const float params[4] = {0.75f, 1.0f, 0.0f, 1.0f};
    const float pbr[4] = {0.85f, 0.21f, 1.0f, 0.0f};
    const float mapParams[4] = {0.6f, 0.3f, 2.5f, 3.0f};
    const float optical[4] = {1.7f, 0.4f, 0.6f, 1.0f};
    std::memcpy(expected.diffuse, diffuse, sizeof(diffuse));
    std::memcpy(expected.ambient, ambient, sizeof(ambient));
    std::memcpy(expected.specular, specular, sizeof(specular));
    std::memcpy(expected.emissive, emissive, sizeof(emissive));
    std::memcpy(expected.params, params, sizeof(params));
    std::memcpy(expected.pbr, pbr, sizeof(pbr));
    std::memcpy(expected.mapParams, mapParams, sizeof(mapParams));
    std::memcpy(expected.optical, optical, sizeof(optical));

    check(std::memcmp(&packed, &expected, sizeof(packed)) == 0,
          "packed block must be byte-identical to the hand-built expected block");
  }

  // --- resolveOptical: global default vs authored -------------------------
  {
    SoRenderIR::SoMaterialBlock unauthored;
    std::memset(&unauthored, 0, sizeof(unauthored));
    unauthored.optical[0] = 1.5f;
    unauthored.optical[1] = 0.0f;
    unauthored.optical[3] = 0.0f;
    SoRenderIR::resolveOptical(unauthored, 2.2f, 0.3f);
    checkFloat(unauthored.optical[0], 2.2f,
               "unauthored optics take the global IOR");
    checkFloat(unauthored.optical[1], 0.3f,
               "unauthored optics take the global absorption");

    SoRenderIR::SoMaterialBlock authored;
    std::memset(&authored, 0, sizeof(authored));
    authored.optical[0] = 1.7f;
    authored.optical[1] = 0.4f;
    authored.optical[3] = 1.0f;
    SoRenderIR::resolveOptical(authored, 2.2f, 0.3f);
    checkFloat(authored.optical[0], 1.7f,
               "authored optics survive the global override");
    checkFloat(authored.optical[1], 0.4f,
               "authored absorption survives the global override");

    // Exactly 0.5 counts as authored (the condition is < 0.5).
    SoRenderIR::SoMaterialBlock boundary;
    std::memset(&boundary, 0, sizeof(boundary));
    boundary.optical[0] = 1.9f;
    boundary.optical[1] = 0.5f;
    boundary.optical[3] = 0.5f;
    SoRenderIR::resolveOptical(boundary, 2.2f, 0.3f);
    checkFloat(boundary.optical[0], 1.9f,
               "optical[3] == 0.5 is treated as authored");
  }

  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
  }
  return failures == 0 ? 0 : 1;
}
