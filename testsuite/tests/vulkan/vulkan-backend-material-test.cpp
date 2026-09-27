// testsuite/vulkan-backend-material-test.cpp
//
// Exercises the retained material model beyond the base headlight case:
// emissive color with no lights, two-sided lighting flip, multiple light
// accumulation, and per-vertex colors overriding the uniform diffuse for
// both the unlit and Gouraud paths.  (Merged from the former material and
// vertex-color tests.)

#include "VulkanTestHarness.h"

#include <cstdlib>

#include <Inventor/SbViewportRegion.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoPhysicalMaterial.h>
#include <Inventor/nodes/SoSeparator.h>

using namespace vulkan_test;

namespace {

SoLightData directional(SbVec3f direction, SbVec3f color)
{
  SoLightData light;
  light.type = SO_LIGHT_DIRECTIONAL;
  light.direction = direction;
  light.color = color;
  return light;
}

static const float greenVertexColors[] = {
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f
};

// Normals pointing away from the orthographic viewer direction (+Z) used by
// the shader when the projection has no perspective term.
static const float normalsDown[] = {
  0.0f, 0.0f, -1.0f,
  0.0f, 0.0f, -1.0f,
  0.0f, 0.0f, -1.0f,
  0.0f, 0.0f, -1.0f
};

// Base texture coordinates for the roughness-map case (a 1x1 map samples its
// texel for any coordinate, but the stream keeps v_texcoord well-defined).
static const float texcoords[] = {
  0.0f, 0.0f,
  1.0f, 0.0f,
  1.0f, 1.0f,
  0.0f, 1.0f
};

// White metal quad, textured, rough scalar (0.9) -- the common shape for the
// roughness-map cases below.
SoRenderCommand roughMetalQuad()
{
  SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                        SO_SHADING_LEGACY_GOURAUD);
  command.geometry.texcoords = texcoords;
  command.geometry.texcoordStride = sizeof(float) * 2;
  command.material.physicalMaterial = true;
  command.material.metalness = 1.0f;
  command.material.roughness = 0.9f;
  return command;
}

// Full-viewport Gouraud quad with green per-vertex colors over a red
// diffuse: the vertex colors must win.
SoRenderCommand vertexColorQuad(SoShadingModel shading)
{
  SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                       shading);
  command.geometry.colors = greenVertexColors;
  command.state.depth.enabled = FALSE;
  command.state.depth.writeEnabled = FALSE;
  return command;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  CaseRunner cases;

  cases.add("emissive with no lights", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.emissive = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "emissive material did not "
             "render red: "
             << describePixel(center));
  });

  cases.add("two-sided lighting flips the back-facing normal", [&harness] {
    // The shader's orthographic branch fixes the viewer direction at
    // V=(0,0,1), so give the quad -Z normals: they face away from the viewer
    // (dot(N,V) < 0) and are flipped to +Z.  The light travels along +Z, so
    // the flipped normal is lit while the unflipped one is not.
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));

    SoDrawList oneSided;
    SoRenderCommand plain = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                        SO_SHADING_LEGACY_GOURAUD);
    plain.geometry.normals = normalsDown;
    plain.lightingHandle = oneSided.addLightingSetup(lighting);
    oneSided.addCommand(plain);
    VK_CHECK(harness.backend.render(oneSided, harness.renderParams()),
             "render failed");
    const uint8_t * oneSidedPixel = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(oneSidedPixel, 0, 0, 0),
             "back-facing normal was lit without two-sided lighting: "
             << describePixel(oneSidedPixel));

    SoDrawList twoSided;
    SoRenderCommand flipped = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    flipped.geometry.normals = normalsDown;
    flipped.material.twoSidedLighting = true;
    flipped.lightingHandle = twoSided.addLightingSetup(lighting);
    twoSided.addCommand(flipped);
    VK_CHECK(harness.backend.render(twoSided, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 0, 0), "two-sided lighting did not flip "
             "the normal: "
             << describePixel(center));
  });

  cases.add("multiple lights accumulate", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(0.5f, 0.5f, 0.5f)));
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(0.5f, 0.5f, 0.5f)));
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    // Two 0.5-intensity lights sum to full intensity.
    VK_CHECK(nearColor(center, 255, 0, 0), "two lights did not accumulate "
             "to full brightness: "
             << describePixel(center));
  });

  cases.add("Gouraud: vertex color overrides the diffuse", [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command =
      vertexColorQuad(SO_SHADING_LEGACY_GOURAUD);
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "vertex color was not used for "
             "Gouraud shading: "
             << describePixel(center));
  });

  cases.add("unlit: vertex color overrides the diffuse", [&harness] {
    SoDrawList drawlist;
    drawlist.addCommand(vertexColorQuad(SO_SHADING_UNLIT));
    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 0, 255, 0), "unlit vertex color did not "
             "override diffuse: "
             << describePixel(center));
  });

  cases.add("legacy Blinn-Phong is used when no physical material is set",
            [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    // Black diffuse, white specular, high shininess: the legacy Blinn-Phong
    // highlight is the only contributor, so a white centre proves the
    // fragment shader took the legacy path.  metalness/roughness are set but
    // physicalMaterial stays false, so they must be ignored.
    SoRenderCommand command = makeLitQuad(SbVec4f(0.0f, 0.0f, 0.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.specular = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
    command.material.shininess = 1.0f;
    command.material.metalness = 1.0f;
    command.material.roughness = 0.05f;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 255, 255),
             "legacy Blinn-Phong highlight missing (physical path leaked?): "
             << describePixel(center));
  });

  cases.add("physical metal: smooth surface produces a bright highlight",
            [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    // White metal, near-mirror roughness, lit head-on: F0 == white and the
    // GGX lobe is tight, so the centre saturates.
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.physicalMaterial = true;
    command.material.metalness = 1.0f;
    command.material.roughness = 0.05f;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 255, 255),
             "smooth metal did not saturate the highlight: "
             << describePixel(center));
  });

  cases.add("physical metal: roughness spreads and dims the highlight",
            [&harness] {
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.physicalMaterial = true;
    command.material.metalness = 1.0f;
    command.material.roughness = 0.9f;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(!nearColor(center, 255, 255, 255),
             "rough metal produced a mirror-like highlight: "
             << describePixel(center));
    VK_CHECK((static_cast<int>(center[0]) + center[1] + center[2]) > 15,
             "rough metal went black instead of dim: "
             << describePixel(center));
  });

  cases.add("roughness map multiplies the scalar roughness", [&harness] {
    // White metal, rough scalar (0.9), lit head-on.  A dark roughness texel
    // (sampled .r == 0) drives the effective roughness toward a mirror and the
    // centre saturates; a white texel is a no-op; strength 0 ignores the map.
    const auto renderCenter = [&harness](const SoRenderCommand & command) {
      SoDrawList drawlist;
      SoLightingData lighting;
      lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
      lighting.lights.push_back(
        directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
      SoRenderCommand copy = command;
      copy.lightingHandle = drawlist.addLightingSetup(lighting);
      drawlist.addCommand(copy);
      VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
               "render failed");
      const uint8_t * center = pixelAt(harness.readback(), 16, 16);
      return static_cast<int>(center[0]) + center[1] + center[2];
    };
    static const unsigned char darkTexel[] = {0};
    static const unsigned char whiteTexel[] = {255};
    const auto withMap = [](const unsigned char * texel) {
      SoRenderCommand command = roughMetalQuad();
      command.material.roughnessTexture.pixels = texel;
      command.material.roughnessTexture.width = 1;
      command.material.roughnessTexture.height = 1;
      command.material.roughnessTexture.numComponents = 1;
      command.material.roughnessTexture.minFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.roughnessTexture.magFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.roughnessTexture.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.roughnessTexture.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      return command;
    };

    const int noMap = renderCenter(roughMetalQuad());
    const int whiteMap = renderCenter(withMap(whiteTexel));
    const int darkMap = renderCenter(withMap(darkTexel));
    SoRenderCommand strengthZero = withMap(darkTexel);
    strengthZero.material.roughnessStrength = 0.0f;
    const int zeroStrength = renderCenter(strengthZero);

    VK_CHECK(darkMap > noMap + 60,
             "dark roughness map did not sharpen the highlight: noMap="
             << noMap << " darkMap=" << darkMap);
    VK_CHECK(std::abs(whiteMap - noMap) <= 30,
             "white roughness map altered the result: noMap=" << noMap
             << " whiteMap=" << whiteMap);
    VK_CHECK(std::abs(zeroStrength - noMap) <= 30,
             "roughness strength 0 did not ignore the map: noMap=" << noMap
             << " zeroStrength=" << zeroStrength);
  });

  cases.add("emissive map adds light with no lights", [&harness] {
    // No lights, black diffuse and black emissive scalar: the only light is the
    // emissive map (set 2, binding 2).  A green 1x1 texel at intensity 1 must
    // produce green; intensity 0 must ignore the map and stay black.
    static const unsigned char greenTexel[] = {0, 255, 0, 255};
    const auto emissiveQuad = [](float intensity) {
      SoRenderCommand command = makeLitQuad(SbVec4f(0.0f, 0.0f, 0.0f, 1.0f),
                                            SO_SHADING_LEGACY_GOURAUD);
      command.geometry.texcoords = texcoords;
      command.geometry.texcoordStride = sizeof(float) * 2;
      command.material.emissive = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
      command.material.emissiveTexture.pixels = greenTexel;
      command.material.emissiveTexture.width = 1;
      command.material.emissiveTexture.height = 1;
      command.material.emissiveTexture.numComponents = 4;
      command.material.emissiveTexture.minFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.emissiveTexture.magFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.emissiveTexture.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.emissiveTexture.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.emissiveIntensity = intensity;
      return command;
    };
    const auto render = [&harness](const SoRenderCommand & command) {
      SoDrawList drawlist;
      SoLightingData lighting;
      lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
      SoRenderCommand copy = command;
      copy.lightingHandle = drawlist.addLightingSetup(lighting);
      drawlist.addCommand(copy);
      VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
               "render failed");
      return pixelAt(harness.readback(), 16, 16);
    };

    const uint8_t * lit = render(emissiveQuad(1.0f));
    VK_CHECK(nearColor(lit, 0, 255, 0),
             "emissive map did not emit with no lights: "
             << describePixel(lit));
    const uint8_t * off = render(emissiveQuad(0.0f));
    VK_CHECK(nearColor(off, 0, 0, 0),
             "emissive intensity 0 did not ignore the map: "
             << describePixel(off));
  });

  cases.add("normal map perturbs the shading normal", [&harness] {
    // Head-on light, +Z geometric normal.  A flat normal-map texel keeps N at
    // +Z (bright); a texel tilted to (+X,0,0) turns N perpendicular to the
    // light (dark).  The tangent basis is derived in-shader (derivative TBN).
    static const unsigned char flatTexel[] = {128, 128, 255, 255};
    static const unsigned char tiltTexel[] = {255, 128, 128, 255};
    const auto normalQuad = [](const unsigned char * texel) {
      SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                            SO_SHADING_LEGACY_GOURAUD);
      command.geometry.texcoords = texcoords;
      command.geometry.texcoordStride = sizeof(float) * 2;
      command.material.normalTexture.pixels = texel;
      command.material.normalTexture.width = 1;
      command.material.normalTexture.height = 1;
      command.material.normalTexture.numComponents = 4;
      command.material.normalTexture.minFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.normalTexture.magFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.normalTexture.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.normalTexture.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.normalStrength = 1.0f;
      return command;
    };
    const auto render = [&harness](const SoRenderCommand & command) {
      SoDrawList drawlist;
      SoLightingData lighting;
      lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
      lighting.lights.push_back(
        directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
      SoRenderCommand copy = command;
      copy.lightingHandle = drawlist.addLightingSetup(lighting);
      drawlist.addCommand(copy);
      VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
               "render failed");
      const uint8_t * center = pixelAt(harness.readback(), 16, 16);
      return static_cast<int>(center[0]) + center[1] + center[2];
    };

    const int flatLum = render(normalQuad(flatTexel));
    VK_CHECK(flatLum > 400,
             "flat normal map altered the lit surface: lum=" << flatLum);
    const int tiltLum = render(normalQuad(tiltTexel));
    VK_CHECK(flatLum > tiltLum + 200,
             "normal map did not tilt the normal: flat=" << flatLum
             << " tilt=" << tiltLum);
  });

  cases.add("SoPhysicalMaterial node drives the rendered PBR path",
            [&harness] {
    // Traverse a scene carrying a physical-material node, confirm its
    // parameters reach the IR, then render that material and confirm the
    // smooth-metal highlight is produced on the GPU.
    SoSeparator * root = new SoSeparator;
    root->ref();
    SoPhysicalMaterial * physical = new SoPhysicalMaterial;
    physical->metalness.setValue(1.0f);
    physical->roughness.setValue(0.05f);
    root->addChild(physical);
    root->addChild(new SoCube);

    SoIRRenderAction ir(SbViewportRegion(64, 64));
    ir.apply(root);
    bool enabled = false;
    float metalness = 0.0f;
    float roughness = 1.0f;
    if (ir.getDrawList().getNumCommands() > 0) {
      const SoMaterialData & md = ir.getDrawList().getCommand(0).material;
      enabled = md.physicalMaterial;
      metalness = md.metalness;
      roughness = md.roughness;
    }
    root->unref();
    VK_CHECK(enabled && metalness == 1.0f && roughness == 0.05f,
             "SoPhysicalMaterial did not reach the IR");

    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.physicalMaterial = enabled;
    command.material.metalness = metalness;
    command.material.roughness = roughness;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(nearColor(center, 255, 255, 255),
             "traversed physical material did not render smooth metal: "
             << describePixel(center));
  });

  cases.add("normal strength 0 ignores the normal map", [&harness] {
    // Mirror of the roughness-strength-0 case for the normal map: strength 0
    // must leave the geometric normal untouched even when the texel would
    // tilt it strongly.
    static const unsigned char flatTexel[] = {128, 128, 255, 255};
    static const unsigned char tiltTexel[] = {255, 128, 128, 255};
    const auto renderLum = [&harness](const unsigned char * texel,
                                      float strength) {
      SoDrawList drawlist;
      SoLightingData lighting;
      lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
      lighting.lights.push_back(
        directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
      SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                            SO_SHADING_LEGACY_GOURAUD);
      command.geometry.texcoords = texcoords;
      command.geometry.texcoordStride = sizeof(float) * 2;
      command.material.normalTexture.pixels = texel;
      command.material.normalTexture.width = 1;
      command.material.normalTexture.height = 1;
      command.material.normalTexture.numComponents = 4;
      command.material.normalTexture.minFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.normalTexture.magFilter = SO_TEXTURE_FILTER_NEAREST;
      command.material.normalTexture.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.normalTexture.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      command.material.normalStrength = strength;
      command.lightingHandle = drawlist.addLightingSetup(lighting);
      drawlist.addCommand(command);
      VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
               "render failed");
      const uint8_t * center = pixelAt(harness.readback(), 16, 16);
      return static_cast<int>(center[0]) + center[1] + center[2];
    };

    const int flat = renderLum(flatTexel, 1.0f);
    const int tiltOn = renderLum(tiltTexel, 1.0f);
    const int tiltOff = renderLum(tiltTexel, 0.0f);
    VK_CHECK(flat > tiltOn + 200,
             "normal map did not tilt the normal: flat=" << flat
             << " tilt=" << tiltOn);
    VK_CHECK(std::abs(tiltOff - flat) <= 30,
             "normal strength 0 did not ignore the map: flat=" << flat
             << " tiltOff=" << tiltOff);
  });

  cases.add("zero-size secondary map is ignored", [&harness] {
    // A map with a pixel pointer but a zero extent must count as absent, so
    // the shader never samples a zero-width/zero-component texture (which
    // would raise a validation-layer error or read out of bounds).
    static const unsigned char whiteTexel[] = {255};
    const auto renderCenter = [&harness](const SoRenderCommand & command) {
      SoDrawList drawlist;
      SoLightingData lighting;
      lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
      lighting.lights.push_back(
        directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
      SoRenderCommand copy = command;
      copy.lightingHandle = drawlist.addLightingSetup(lighting);
      drawlist.addCommand(copy);
      VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
               "render failed");
      const uint8_t * center = pixelAt(harness.readback(), 16, 16);
      return static_cast<int>(center[0]) + center[1] + center[2];
    };

    const int noMap = renderCenter(roughMetalQuad());
    SoRenderCommand zeroSize = roughMetalQuad();
    zeroSize.material.roughnessTexture.pixels = whiteTexel;
    zeroSize.material.roughnessTexture.width = 0;
    zeroSize.material.roughnessTexture.height = 0;
    zeroSize.material.roughnessTexture.numComponents = 1;
    const int zero = renderCenter(zeroSize);
    VK_CHECK(std::abs(zero - noMap) <= 30,
             "zero-size map changed the result: noMap=" << noMap
             << " zeroSize=" << zero);
  });

  cases.add("normal map without UVs degrades to flat shading", [&harness] {
    // The derivative TBN degenerates when the geometry has no texture
    // coordinates; the shader must fall back to the geometric normal instead
    // of producing NaN or crashing.
    static const unsigned char tiltTexel[] = {255, 128, 128, 255};
    SoDrawList drawlist;
    SoLightingData lighting;
    lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
    lighting.lights.push_back(
      directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
    SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, 1.0f),
                                          SO_SHADING_LEGACY_GOURAUD);
    command.material.normalTexture.pixels = tiltTexel;
    command.material.normalTexture.width = 1;
    command.material.normalTexture.height = 1;
    command.material.normalTexture.numComponents = 4;
    command.material.normalTexture.minFilter = SO_TEXTURE_FILTER_NEAREST;
    command.material.normalTexture.magFilter = SO_TEXTURE_FILTER_NEAREST;
    command.material.normalTexture.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
    command.material.normalTexture.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
    command.material.normalStrength = 1.0f;
    command.lightingHandle = drawlist.addLightingSetup(lighting);
    drawlist.addCommand(command);

    VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
             "render with a normal map but no UVs failed");
    const uint8_t * center = pixelAt(harness.readback(), 16, 16);
    VK_CHECK(center[0] + center[1] + center[2] > 300,
             "no-UV normal map did not degrade to flat shading: "
             << describePixel(center));
  });

  cases.add("physical material with opacity < 1 blends", [&harness] {
    const auto renderLum = [&harness](float opacity) {
      SoDrawList drawlist;
      SoLightingData lighting;
      lighting.ambient = SbVec3f(0.0f, 0.0f, 0.0f);
      lighting.lights.push_back(
        directional(SbVec3f(0.0f, 0.0f, 1.0f), SbVec3f(1.0f, 1.0f, 1.0f)));
      // Transparency rides in both the diffuse alpha (the raster push
      // constant's alpha) and the packed optical transmittance.
      SoRenderCommand command = makeLitQuad(SbVec4f(1.0f, 1.0f, 1.0f, opacity),
                                            SO_SHADING_LEGACY_GOURAUD);
      command.material.physicalMaterial = true;
      command.material.metalness = 1.0f;
      command.material.roughness = 0.05f;
      command.material.opacity = opacity;
      command.state.blend.enabled = TRUE;
      command.state.blend.srcRGBFactor = SO_BLEND_FACTOR_SRC_ALPHA;
      command.state.blend.dstRGBFactor = SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      command.state.blend.srcAlphaFactor = SO_BLEND_FACTOR_ONE;
      command.state.blend.dstAlphaFactor = SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      command.lightingHandle = drawlist.addLightingSetup(lighting);
      drawlist.addCommand(command);
      VK_CHECK(harness.backend.render(drawlist, harness.renderParams()),
               "render failed");
      const uint8_t * center = pixelAt(harness.readback(), 16, 16);
      return static_cast<int>(center[0]) + center[1] + center[2];
    };

    const int opaque = renderLum(1.0f);
    const int translucent = renderLum(0.5f);
    VK_CHECK(translucent < opaque - 60,
             "opacity 0.5 did not dim the physical highlight: opaque="
             << opaque << " translucent=" << translucent);
    VK_CHECK(translucent > 30,
             "opacity blend produced black: translucent=" << translucent);
  });

  const int failures = cases.run();
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
