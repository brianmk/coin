#include "rendering/SoRenderIRP.h"

#include <Inventor/C/tidbits.h>
#include <Inventor/elements/SoDepthBufferElement.h>
#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/elements/SoEnvironmentElement.h>
#include <Inventor/elements/SoLazyElement.h>
#include <Inventor/elements/SoLightAttenuationElement.h>
#include <Inventor/elements/SoLightElement.h>
#include <Inventor/elements/SoLightModelElement.h>
#include <Inventor/elements/SoLinePatternElement.h>
#include <Inventor/elements/SoLineWidthElement.h>
#include <Inventor/elements/SoMultiTextureEnabledElement.h>
#include <Inventor/elements/SoMultiTextureImageElement.h>
#include <Inventor/elements/SoPointSizeElement.h>
#include <Inventor/elements/SoModelMatrixElement.h>
#include <Inventor/elements/SoProjectionMatrixElement.h>
#include <Inventor/elements/SoTextureQualityElement.h>
#include <Inventor/elements/SoShapeHintsElement.h>
#include <Inventor/elements/SoViewportRegionElement.h>
#include <Inventor/elements/SoViewingMatrixElement.h>
#include <Inventor/elements/SoPolygonOffsetElement.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoPointLight.h>
#include <Inventor/nodes/SoShape.h>
#include <Inventor/nodes/SoSpotLight.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <inttypes.h>

namespace {

bool
lightingEqual(const SoLightData & lhs, const SoLightData & rhs)
{
  return lhs.type == rhs.type &&
         lhs.color == rhs.color &&
         lhs.direction == rhs.direction &&
         lhs.position == rhs.position &&
         lhs.attenuation == rhs.attenuation &&
         lhs.spotCutoffCos == rhs.spotCutoffCos &&
         lhs.spotExponent == rhs.spotExponent;
}

bool
lightingEqual(const SoLightingData & lhs, const SoLightingData & rhs)
{
  if (lhs.ambient != rhs.ambient || lhs.lights.size() != rhs.lights.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.lights.size(); ++i) {
    if (!lightingEqual(lhs.lights[i], rhs.lights[i])) {
      return false;
    }
  }
  return true;
}

SoBlendFactor
blendFactorFromLegacyGL(const int value)
{
  // Keep GL values local to this conversion boundary; no GL enum is stored in the public IR.
  switch (value) {
  case 0x0000: return SO_BLEND_FACTOR_ZERO;                    // GL_ZERO
  case 0x0001: return SO_BLEND_FACTOR_ONE;                     // GL_ONE
  case 0x0300: return SO_BLEND_FACTOR_SRC_COLOR;              // GL_SRC_COLOR
  case 0x0301: return SO_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
  case 0x0302: return SO_BLEND_FACTOR_SRC_ALPHA;
  case 0x0303: return SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case 0x0304: return SO_BLEND_FACTOR_DST_ALPHA;
  case 0x0305: return SO_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  case 0x0306: return SO_BLEND_FACTOR_DST_COLOR;
  case 0x0307: return SO_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case 0x0308: return SO_BLEND_FACTOR_SRC_ALPHA_SATURATE;
  case 0x8001: return SO_BLEND_FACTOR_CONSTANT_COLOR;
  case 0x8002: return SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
  case 0x8003: return SO_BLEND_FACTOR_CONSTANT_ALPHA;
  case 0x8004: return SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
  case 0x8589:
    return SO_BLEND_FACTOR_SRC1_ALPHA;                    // GL_SRC1_ALPHA
  case 0x88F9:
    return SO_BLEND_FACTOR_SRC1_COLOR;                    // GL_SRC1_COLOR
  case 0x88FA:
    return SO_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;          // GL_ONE_MINUS_SRC1_COLOR
  case 0x88FB:
    return SO_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;           // GL_ONE_MINUS_SRC1_ALPHA
  default:     return SO_BLEND_FACTOR_ONE;
  }
}

SoTextureModel
textureModelFromLegacy(SoMultiTextureImageElement::Model model)
{
  switch (model) {
  case SoMultiTextureImageElement::DECAL:
    return SO_TEXTURE_MODEL_DECAL;
  case SoMultiTextureImageElement::BLEND:
    return SO_TEXTURE_MODEL_BLEND;
  case SoMultiTextureImageElement::REPLACE:
    return SO_TEXTURE_MODEL_REPLACE;
  case SoMultiTextureImageElement::MODULATE:
  default:
    return SO_TEXTURE_MODEL_MODULATE;
  }
}

float
textureQualityLimit(const char * name, const float fallback)
{
  const char * value = coin_getenv(name);
  if (!value) return fallback;
  const float parsed = static_cast<float>(std::atof(value));
  return parsed >= 0.0f && parsed <= 1.0f ? parsed : fallback;
}

void
textureFiltersFromQuality(const float quality, SoTextureData & texture)
{
  // Keep in lockstep with SoGLImageP::applyFilter()'s LegacyGL thresholds; the IR stores effective sampler state so backends need no Coin quality policy.
  static const float linearLimit =
    textureQualityLimit("COIN_TEX2_LINEAR_LIMIT", 0.2f);
  static const float mipmapLimit =
    textureQualityLimit("COIN_TEX2_MIPMAP_LIMIT", 0.5f);
  static const float linearMipmapLimit =
    textureQualityLimit("COIN_TEX2_LINEAR_MIPMAP_LIMIT", 0.8f);

  if (quality < linearLimit) {
    texture.minFilter = SO_TEXTURE_FILTER_NEAREST;
    texture.magFilter = SO_TEXTURE_FILTER_NEAREST;
  }
  else if (quality < mipmapLimit) {
    texture.minFilter = SO_TEXTURE_FILTER_LINEAR;
    texture.magFilter = SO_TEXTURE_FILTER_LINEAR;
  }
  else if (quality < linearMipmapLimit) {
    texture.minFilter = SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR;
    texture.magFilter = SO_TEXTURE_FILTER_LINEAR;
  }
  else {
    texture.minFilter = SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR;
    texture.magFilter = SO_TEXTURE_FILTER_LINEAR;
  }
}

} // namespace

SoIRBuffer::SoIRBuffer()
{
}

constexpr size_t SoIRBuffer::MIN_CHUNK_SIZE;

void
SoIRBuffer::clear()
{
  // Track high-water mark so we can pre-size on next frame
  if (this->totalAllocated > this->highWaterMark) {
    this->highWaterMark = this->totalAllocated;
  }
  // Reset cursors but keep chunks allocated
  for (auto & chunk : this->chunks) {
    chunk->cursor = 0;
  }
  this->totalAllocated = 0;
}

void
SoIRBuffer::reserve(size_t bytes)
{
  // Ensure the first chunk is at least this large
  if (this->chunks.empty()) {
    std::unique_ptr<Chunk> c(new Chunk);
    c->data.resize(std::max(bytes, MIN_CHUNK_SIZE));
    this->chunks.push_back(std::move(c));
  } else if (bytes > this->chunks[0]->data.size()) {
    // Only resize the first chunk if it hasn't been used yet
    if (this->chunks[0]->cursor == 0) {
      this->chunks[0]->data.resize(bytes);
    }
  }
}

void *
SoIRBuffer::allocate(size_t bytes, size_t alignment)
{
  if (alignment == 0) alignment = 1;

  // Try to allocate from an existing chunk
  for (auto & chunk : this->chunks) {
    size_t aligned = (chunk->cursor + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes <= chunk->data.size()) {
      void * ptr = chunk->data.data() + aligned;
      chunk->cursor = aligned + bytes;
      this->totalAllocated += bytes;
      return ptr;
    }
  }

  // New chunk sized to fit this allocation while avoiding many small chunks.
  size_t chunkSize = std::max({bytes, MIN_CHUNK_SIZE, this->highWaterMark / 2});
  std::unique_ptr<Chunk> c(new Chunk);
  c->data.resize(chunkSize);
  c->cursor = bytes;
  void * ptr = c->data.data();
  this->chunks.push_back(std::move(c));
  this->totalAllocated += bytes;
  return ptr;
}

SoDrawList::SoDrawList()
{
}

void
SoDrawList::clear()
{
  this->commands.clear();
  this->lightingSetups.clear();
  this->lightingRaws.clear();
  this->sortedOrder.clear();
  this->generation++;
}

void
SoDrawList::truncate(int count)
{
  if (count < static_cast<int>(this->commands.size())) {
    this->commands.resize(static_cast<size_t>(count));
    // Commands stay insertion-ordered; sortedOrder is rebuilt when the backend prepares the frame.
  }
}

void
SoDrawList::reserve(int count)
{
  this->commands.reserve(static_cast<size_t>(count));
}

void
SoDrawList::addCommand(const SoRenderCommand & cmd)
{
  this->commands.push_back(cmd);
}

SoRenderCommand &
SoDrawList::emplaceCommand()
{
  this->commands.emplace_back();
  return this->commands.back();
}

int
SoDrawList::getNumCommands() const
{
  return static_cast<int>(this->commands.size());
}

SoRenderCommand &
SoDrawList::getCommand(int i)
{
  return this->commands[static_cast<size_t>(i)];
}

const SoRenderCommand &
SoDrawList::getCommand(int i) const
{
  return this->commands[static_cast<size_t>(i)];
}

namespace {

// Bitwise matrix equality: both sides copy one element value, so identical bits are the right identity test (no epsilon).
bool matrixBitsEqual(const SbMatrix & a, const SbMatrix & b)
{
  SbMat av, bv;
  a.getValue(av);
  b.getValue(bv);
  return std::memcmp(&av[0][0], &bv[0][0], sizeof(av)) == 0;
}

bool rawLightEqual(const SoLightingRaw::RawLight & a,
                   const SoLightingRaw::RawLight & b)
{
  return a.type == b.type &&
    std::memcmp(&a.sceneDirection[0], &b.sceneDirection[0],
                sizeof(float) * 3) == 0 &&
    std::memcmp(&a.scenePosition[0], &b.scenePosition[0],
                sizeof(float) * 3) == 0 &&
    matrixBitsEqual(a.sceneMatrix, b.sceneMatrix);
}

bool rawEqual(const SoLightingRaw & a, const SoLightingRaw & b)
{
  if (a.hasRaw != b.hasRaw) return false;
  if (!a.hasRaw) return true;
  if (a.lights.size() != b.lights.size()) return false;
  // NOTE: viewUsed is deliberately NOT compared: setups are world-space/view-independent, so
  // identical raw geometry is the same light; comparing it would fragment dedup and inflate the lighting ring.
  for (size_t i = 0; i < a.lights.size(); ++i) {
    if (!rawLightEqual(a.lights[i], b.lights[i])) return false;
  }
  return true;
}

} // namespace

SoLightingHandle
SoDrawList::addLightingSetup(const SoLightingData & lighting)
{
  for (size_t i = 0; i < this->lightingSetups.size(); ++i) {
    if (lightingEqual(this->lightingSetups[i], lighting) &&
        rawEqual(this->lightingRaws[i], SoLightingRaw())) {
      return static_cast<SoLightingHandle>(i + 1);
    }
  }
  this->lightingSetups.push_back(lighting);
  this->lightingRaws.emplace_back();
  return static_cast<SoLightingHandle>(this->lightingSetups.size());
}

SoLightingHandle
SoDrawList::addLightingSetup(const SoLightingData & lighting,
                             const SoLightingRaw & raw)
{
  for (size_t i = 0; i < this->lightingSetups.size(); ++i) {
    if (lightingEqual(this->lightingSetups[i], lighting) &&
        rawEqual(this->lightingRaws[i], raw)) {
      return static_cast<SoLightingHandle>(i + 1);
    }
  }
  this->lightingSetups.push_back(lighting);
  this->lightingRaws.push_back(raw);
  return static_cast<SoLightingHandle>(this->lightingSetups.size());
}

namespace {

void packLightIntoBlock(SoLightingBlock & block, int slot,
                        const SoLightData & light)
{
  float * type = block.lightType + slot * 4;
  type[0] = static_cast<float>(light.type);
  type[1] = type[2] = 0.0f;
  type[3] = 1.0f;

  float * color = block.lightColor + slot * 4;
  color[0] = light.color[0];
  color[1] = light.color[1];
  color[2] = light.color[2];
  color[3] = 1.0f;

  float * direction = block.lightDirection + slot * 4;
  direction[0] = light.direction[0];
  direction[1] = light.direction[1];
  direction[2] = light.direction[2];
  direction[3] = 1.0f;

  float * position = block.lightPosition + slot * 4;
  position[0] = light.position[0];
  position[1] = light.position[1];
  position[2] = light.position[2];
  position[3] = 1.0f;

  float * attenuation = block.lightAttenuation + slot * 4;
  attenuation[0] = light.attenuation[0];
  attenuation[1] = light.attenuation[1];
  attenuation[2] = light.attenuation[2];
  attenuation[3] = 1.0f;

  float * spot = block.lightSpotParams + slot * 4;
  spot[0] = light.spotCutoffCos;
  spot[1] = light.spotExponent;
  spot[2] = 0.0f;
  spot[3] = 1.0f;
}

} // namespace

SoLightData
SoRenderIR::lightToEye(const SoLightData & world, const SbMatrix & view)
{
  SoLightData eye = world;
  view.multDirMatrix(world.direction, eye.direction);
  if (eye.direction.normalize() == 0.0f) {
    eye.direction = world.direction;
  }
  if (world.type != SO_LIGHT_DIRECTIONAL) {
    view.multVecMatrix(world.position, eye.position);
  }
  return eye;
}

SoLightData
SoRenderIR::lightToWorld(const SoLightData & eye, const SbMatrix & inverseView)
{
  SoLightData world = eye;
  inverseView.multDirMatrix(eye.direction, world.direction);
  if (world.direction.normalize() == 0.0f) {
    world.direction = eye.direction;
  }
  if (eye.type != SO_LIGHT_DIRECTIONAL) {
    inverseView.multVecMatrix(eye.position, world.position);
  }
  return world;
}

int
SoRenderIR::fillLightingBlock(SoLightingBlock & block,
                              const SoLightingData & world,
                              const SbMatrix * toEye)
{
  std::memset(&block, 0, sizeof(block));
  block.ambientLight[0] = world.ambient[0];
  block.ambientLight[1] = world.ambient[1];
  block.ambientLight[2] = world.ambient[2];
  block.ambientLight[3] = 1.0f;

  const int count = world.lightCount();
  for (int i = 0; i < count; ++i) {
    const SoLightData & light = world.lights[static_cast<size_t>(i)];
    if (toEye != nullptr) {
      packLightIntoBlock(block, i, lightToEye(light, *toEye));
    }
    else {
      packLightIntoBlock(block, i, light);
    }
  }

  return count;
}

const SoLightingData *
SoDrawList::getLighting(SoLightingHandle handle) const
{
  if (handle == 0) {
    return nullptr;
  }
  const size_t index = static_cast<size_t>(handle - 1);
  if (index >= this->lightingSetups.size()) {
    return nullptr;
  }
  return &this->lightingSetups[index];
}

SoRenderCommand *
SoDrawList::begin()
{
  return this->commands.empty() ? nullptr : this->commands.data();
}

SoRenderCommand *
SoDrawList::end()
{
  return this->commands.empty() ? nullptr : this->commands.data() + this->commands.size();
}

const SoRenderCommand *
SoDrawList::begin() const
{
  return this->commands.empty() ? nullptr : this->commands.data();
}

const SoRenderCommand *
SoDrawList::end() const
{
  return this->commands.empty() ? nullptr : this->commands.data() + this->commands.size();
}

void
SoDrawList::buildSortedOrder(const SbMatrix & viewMatrix)
{
  int n = static_cast<int>(this->commands.size());
  sortedOrder.resize(n);
  for (int i = 0; i < n; i++) sortedOrder[i] = i;
  if (n <= 1) return;

  SoRenderCommand * arr = this->commands.data();

  // Compute camera-space depth for each command using the model matrix origin.
  SbMat v;
  viewMatrix.getValue(v);
  for (int i = 0; i < n; i++) {
    SoRenderCommand & cmd = arr[i];
    SbMat m;
    cmd.modelMatrix.getValue(m);
    float wx = m[3][0], wy = m[3][1], wz = m[3][2];
    float eyeZ = v[0][2] * wx + v[1][2] * wy + v[2][2] * wz + v[3][2];
    float depth = -eyeZ;

    // Float-to-uint reinterpretation for monotonic ordering
    uint32_t bits;
    std::memcpy(&bits, &depth, sizeof(bits));
    if (bits & 0x80000000u) {
      bits = ~bits;
    } else {
      bits |= 0x80000000u;
    }
    uint32_t depthBucket = (bits >> 8) & 0x00FFFFFFu;

    // Transparent: back-to-front (invert depth)
    uint32_t passOrder = static_cast<uint32_t>(cmd.pass);
    if (cmd.pass == SO_RENDERPASS_TRANSPARENT) {
      depthBucket = 0x00FFFFFFu - depthBucket;
    }
    cmd.sortKey = SoIRComputeSortKey(passOrder, depthBucket);
  }

  // Sort the INDEX array by sort key, leaving commands in place
  std::stable_sort(sortedOrder.begin(), sortedOrder.end(),
    [arr](int a, int b) {
      return arr[a].sortKey < arr[b].sortKey;
    });
}

uint64_t
SoIRComputeSortKey(uint32_t passOrderBits,
                   uint32_t depthBucket)
{
  const uint64_t passbits = (static_cast<uint64_t>(passOrderBits) & 0xffULL) << 56;
  const uint64_t depthbits = (static_cast<uint64_t>(depthBucket) & 0x00ffffffULL) << 32;
  return passbits | depthbits;
}

namespace SoRenderIR {

static SoTextureWrap
textureWrapFromLegacy(SoMultiTextureImageElement::Wrap wrap)
{
  switch (wrap) {
  case SoMultiTextureImageElement::REPEAT:
    return SO_TEXTURE_WRAP_REPEAT;
  case SoMultiTextureImageElement::CLAMP_TO_BORDER:
    return SO_TEXTURE_WRAP_CLAMP_TO_BORDER;
  case SoMultiTextureImageElement::CLAMP:
  default:
    // GL_CLAMP is the historical Coin spelling for edge clamping here.
    return SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  }
}

void
fillMaterialFromState(SoState * state, SoMaterialData & material,
                      int materialIndex)
{
  SoState * mutableState = state;
  const SbColor & diffuse = SoLazyElement::getDiffuse(mutableState, materialIndex);
  const SbColor & ambient = SoLazyElement::getAmbient(mutableState);
  const SbColor & specular = SoLazyElement::getSpecular(mutableState);
  const SbColor & emissive = SoLazyElement::getEmissive(mutableState);
  const float transparency = SoLazyElement::getTransparency(mutableState, materialIndex);

  // Keep diffuse and emissive independent: the shader owns emissive, so inferring diffuse from a default-looking material would double-count emissive-only ones.
  material.diffuse.setValue(diffuse[0], diffuse[1], diffuse[2],
                            1.0f - transparency);

  // Capture the shading contract explicitly: Coin's PHONG model maps to the legacy-compatible
  // Gouraud path now, but a true per-fragment PHONG path can come without changing the payload.
  const int lightModel = SoLightModelElement::get(mutableState);
  const bool baseColor = lightModel == SoLightModelElement::BASE_COLOR;
  material.shadingModel = baseColor
    ? SO_SHADING_UNLIT
    : SO_SHADING_LEGACY_GOURAUD;
  material.twoSidedLighting = SoLazyElement::getTwoSidedLighting(mutableState) != FALSE;
  material.featureFlags = baseColor ? SO_FEAT_BASE_COLOR : 0;
  material.ambient.setValue(ambient[0], ambient[1], ambient[2], 1.0f);
  material.specular.setValue(specular[0], specular[1], specular[2], 1.0f);
  material.emissive.setValue(emissive[0], emissive[1], emissive[2], 1.0f);
  material.shininess = SoLazyElement::getShininess(mutableState);
  material.opacity = 1.0f - transparency;

  material.metalness = 0.0f;   // dielectric (Blinn-Phong equivalent)
  material.roughness = 0.5f;   // moderate roughness

  material.diffuseTexture = NULL;
  material.normalTexture = NULL;
  material.emissiveTexture = NULL;
  material.flags = 0;
  material.texture = SoTextureData();
  material.textureAlphaIncludesOpacity = false;
  material.vertexColorAlphaIncludesOpacity = false;
}

void
fillTextureFromState(SoState * state, SoIRRenderAction * action,
                     SoMaterialData & material)
{
  if (!state || !action || !SoMultiTextureEnabledElement::get(state, 0)) {
    return;
  }

  SbVec2s size;
  int numComponents = 0;
  SoMultiTextureImageElement::Wrap wrapS;
  SoMultiTextureImageElement::Wrap wrapT;
  SoMultiTextureImageElement::Model model;
  SbColor blendColor;
  const unsigned char * bytes = SoMultiTextureImageElement::get(
    state, 0, size, numComponents, wrapS, wrapT, model, blendColor);
  if (!bytes || size[0] <= 0 || size[1] <= 0 ||
      numComponents < 1 || numComponents > 4) {
    return;
  }

  const size_t pixelCount = static_cast<size_t>(size[0]) *
                            static_cast<size_t>(size[1]);
  const size_t byteCount = pixelCount * static_cast<size_t>(numComponents);
  unsigned char * copy = static_cast<unsigned char *>(
    action->allocateGeometryStorage(byteCount, alignof(unsigned char)));
  std::memcpy(copy, bytes, byteCount);

  material.texture.pixels = copy;
  material.texture.width = size[0];
  material.texture.height = size[1];
  material.texture.numComponents = numComponents;
  material.texture.wrapS = textureWrapFromLegacy(wrapS);
  material.texture.wrapT = textureWrapFromLegacy(wrapT);
  material.texture.model = textureModelFromLegacy(model);
  material.texture.blendColor.setValue(blendColor[0], blendColor[1],
                                       blendColor[2], 1.0f);
  textureFiltersFromQuality(SoTextureQualityElement::get(state),
                            material.texture);
  material.flags |= SO_MAT_HAS_TEXTURE;
}

void
fillRenderStateFromState(SoState * state, SoRenderState & rs)
{
  SoState * mutableState = state;
  SbBool depthtest = TRUE;
  SbBool depthwrite = TRUE;
  SoDepthBufferElement::DepthWriteFunction depthfunc =
    SoDepthBufferElement::LEQUAL;
  SbVec2f range;
  SoDepthBufferElement::get(mutableState, depthtest, depthwrite, depthfunc, range);

  rs.depth.enabled = depthtest;
  rs.depth.writeEnabled = depthwrite;
  rs.depth.func = static_cast<SoDepthFunction>(depthfunc);
  rs.depth.range = range;

  int srcfactor = 0;
  int dstfactor = 0;
  rs.blend.enabled = SoLazyElement::getBlending(mutableState, srcfactor, dstfactor);
  rs.blend.srcRGBFactor = blendFactorFromLegacyGL(srcfactor);
  rs.blend.dstRGBFactor = blendFactorFromLegacyGL(dstfactor);

  // A regular glBlendFunc applies RGB factors to alpha too; when separate-alpha state is present retain its factors verbatim, including ZERO (previously indistinguishable from "unset").
  int srcAlphaFactor = 0;
  int dstAlphaFactor = 0;
  if (SoLazyElement::getAlphaBlending(mutableState,
                                      srcAlphaFactor, dstAlphaFactor)) {
    rs.blend.srcAlphaFactor = blendFactorFromLegacyGL(srcAlphaFactor);
    rs.blend.dstAlphaFactor = blendFactorFromLegacyGL(dstAlphaFactor);
  } else {
    rs.blend.srcAlphaFactor = rs.blend.srcRGBFactor;
    rs.blend.dstAlphaFactor = rs.blend.dstRGBFactor;
  }


  // LegacyGL exposes no blend-equation element; ADD is its effective (and only deterministically capturable) equation.
  rs.blend.rgbEquation = SO_BLEND_EQUATION_ADD;
  rs.blend.alphaEquation = SO_BLEND_EQUATION_ADD;

  float alphaTestValue = 0.5f;
  const int alphaTestFunction = SoLazyElement::getAlphaTestSemantic(
    mutableState, alphaTestValue);
  rs.alphaTest.function = static_cast<SoAlphaTestFunction>(alphaTestFunction);
  rs.alphaTest.reference = alphaTestValue;
  rs.alphaTest.policy = rs.alphaTest.function == SO_ALPHA_TEST_NONE
    ? SO_ALPHA_TEST_POLICY_NONE
    : SO_ALPHA_TEST_POLICY_EXPLICIT;

  SoDrawStyleElement::Style style = SoDrawStyleElement::get(mutableState);
  uint8_t fillmode = 0;
  switch (style) {
  case SoDrawStyleElement::LINES:
    fillmode = 1;
    break;
  case SoDrawStyleElement::POINTS:
    fillmode = 2;
    break;
  default:
    fillmode = 0;
    break;
  }
  rs.raster.fillMode = fillmode;
  // Native GL_POINTS are square unless point smoothing is on; keep the primitive shape explicit so backends do not choose independently.

  // Backface culling from SoShapeHintsElement: GL culls declared solids with an explicit winding, both CLOCKWISE and COUNTERCLOCKWISE (SoGLLazyElement).
  {
    SoShapeHintsElement::VertexOrdering vo;
    SoShapeHintsElement::ShapeType st;
    SoShapeHintsElement::FaceType ft;
    SoShapeHintsElement::get(mutableState, vo, st, ft);
    rs.raster.cullMode = (vo != SoShapeHintsElement::UNKNOWN_ORDERING
                       && st == SoShapeHintsElement::SOLID) ? 1 : 0;
    rs.raster.ccwFrontFace =
      (vo == SoShapeHintsElement::CLOCKWISE) ? 0 : 1;
  }
  rs.raster.scissorEnabled = FALSE;
  rs.raster.lineWidth = SoLineWidthElement::get(mutableState);
  rs.raster.pointSize = SoPointSizeElement::get(mutableState);
  rs.raster.linePattern = static_cast<uint16_t>(
    SoLinePatternElement::get(mutableState));
  rs.raster.linePatternScale = static_cast<int16_t>(std::max(
    1, SoLinePatternElement::getScaleFactor(mutableState)));

  const SbViewportRegion & viewport = SoViewportRegionElement::get(mutableState);
  const SbVec2s & viewportOrigin = viewport.getViewportOriginPixels();
  const SbVec2s & viewportSize = viewport.getViewportSizePixels();
  rs.raster.viewportEnabled = viewportSize[0] > 0 && viewportSize[1] > 0;
  rs.raster.viewportX = viewportOrigin[0];
  rs.raster.viewportY = viewportOrigin[1];
  rs.raster.viewportWidth = viewportSize[0];
  rs.raster.viewportHeight = viewportSize[1];

  float offsetfactor = 0.0f;
  float offsetunits = 0.0f;
  SoPolygonOffsetElement::Style offsetstyle = SoPolygonOffsetElement::FILLED;
  SbBool offseton = FALSE;
  SoPolygonOffsetElement::get(mutableState, offsetfactor, offsetunits,
                              offsetstyle, offseton);
  if (!offseton) {
    offsetfactor = 0.0f;
    offsetunits = 0.0f;
  }
  rs.raster.polygonOffsetFactor = offsetfactor;
  rs.raster.polygonOffsetUnits = offsetunits;

  rs.opaqueKey = 0;
  rs.translucentKey = 0;
}

SoLightingHandle
fillLightingFromState(SoState * state, SoDrawList & drawlist)
{
  SoLightingData lighting;

  // Scene-space inputs recorded alongside the (world-space) setup for provenance/dedup; the setup needs no camera-move re-derivation.
  SoLightingRaw raw;
  raw.hasRaw = true;
  // Strips the view out of SoLightElement::getMatrix (== model * view), leaving the light's pure world transform.
  const SbMatrix viewInverse = SoViewingMatrixElement::get(state).inverse();

  const SbColor & ambientColor = SoEnvironmentElement::getAmbientColor(state);
  const float ambientIntensity = SoEnvironmentElement::getAmbientIntensity(state);
  lighting.ambient.setValue(ambientColor[0] * ambientIntensity,
                            ambientColor[1] * ambientIntensity,
                            ambientColor[2] * ambientIntensity);

  const SbVec3f & attenuation = SoLightAttenuationElement::get(state);
  const SoNodeList & lights = SoLightElement::getLights(state);
  const int numLights = lights.getLength();
  lighting.lights.reserve(numLights);

  for (int i = 0; i < numLights; ++i) {
    SoLight * light = static_cast<SoLight *>(lights[i]);
    if (!light || !light->on.getValue()) {
      continue;
    }

    const SbColor lightColor = light->color.getValue();
    SoLightData lightData;
    lightData.color.setValue(lightColor[0] * light->intensity.getValue(),
                             lightColor[1] * light->intensity.getValue(),
                             lightColor[2] * light->intensity.getValue());

    const SbMatrix & lightMatrix = SoLightElement::getMatrix(state, i);

    SoLightingRaw::RawLight rawLight;
    rawLight.color = lightData.color;
    rawLight.attenuation = attenuation;
    // lightMatrix == model * view, so the scene-space model matrix follows.
    rawLight.sceneMatrix = lightMatrix * viewInverse;

    // Raw fields are light-LOCAL; the light's scene matrix maps them to world space. A scene-root
    // light (identity) keeps its raw geometry; a camera-parented headlight inherits camera rotation
    // and stays head-fixed (legacy GL modelview semantics). Stored data is view-independent:
    // eye-space consumers use SoRenderIR::lightToEye(); the stored fields stay world-space.
    if (light->isOfType(SoDirectionalLight::getClassTypeId())) {
      SoDirectionalLight * directional = static_cast<SoDirectionalLight *>(light);
      lightData.type = SO_LIGHT_DIRECTIONAL;
      rawLight.type = SO_LIGHT_DIRECTIONAL;
      rawLight.sceneDirection = -(directional->direction.getValue());
      rawLight.sceneMatrix.multDirMatrix(rawLight.sceneDirection,
                                         lightData.direction);
      if (lightData.direction.normalize() == 0.0f) {
        lightData.direction = rawLight.sceneDirection;
        lightData.direction.normalize();
      }
    }
    else if (light->isOfType(SoPointLight::getClassTypeId())) {
      SoPointLight * point = static_cast<SoPointLight *>(light);
      lightData.type = SO_LIGHT_POINT;
      rawLight.type = SO_LIGHT_POINT;
      rawLight.scenePosition = point->location.getValue();
      lightData.attenuation = attenuation;
      rawLight.sceneMatrix.multVecMatrix(rawLight.scenePosition,
                                         lightData.position);
    }
    else if (light->isOfType(SoSpotLight::getClassTypeId())) {
      SoSpotLight * spot = static_cast<SoSpotLight *>(light);
      lightData.type = SO_LIGHT_SPOT;
      rawLight.type = SO_LIGHT_SPOT;
      rawLight.scenePosition = spot->location.getValue();
      rawLight.sceneDirection = spot->direction.getValue();
      lightData.attenuation = attenuation;
      rawLight.sceneMatrix.multVecMatrix(rawLight.scenePosition,
                                         lightData.position);
      rawLight.sceneMatrix.multDirMatrix(rawLight.sceneDirection,
                                         lightData.direction);
      if (lightData.direction.normalize() == 0.0f) {
        lightData.direction = rawLight.sceneDirection;
        lightData.direction.normalize();
      }
      float cutoff = spot->cutOffAngle.getValue();
      if (cutoff < 0.0f) cutoff = 0.0f;
      if (cutoff > float(M_PI) * 0.5f) cutoff = float(M_PI) * 0.5f;
      lightData.spotCutoffCos = std::cos(cutoff);
      float dropoff = spot->dropOffRate.getValue();
      if (dropoff < 0.0f) dropoff = 0.0f;
      if (dropoff > 1.0f) dropoff = 1.0f;
      lightData.spotExponent = dropoff * 128.0f;
      rawLight.spotCutoffCos = lightData.spotCutoffCos;
      rawLight.spotExponent = lightData.spotExponent;
    }
    else {
      continue;
    }

    lighting.lights.push_back(lightData);
    raw.lights.push_back(rawLight);
  }

  raw.ambient = lighting.ambient;
  return drawlist.addLightingSetup(lighting, raw);
}

bool
isMaterialTransparent(const SoMaterialData & material)
{
  return material.opacity < 0.999f;
}

void
ensureMaterialBlendState(SoRenderState & renderState,
                         const SoMaterialData & material)
{
  // SoIRRenderAction captures logical material state, while legacy GL enables the conventional
  // blend function as part of transparency setup; make that implicit IR contract explicit without
  // replacing an actual non-standard blend state.
  if (renderState.blend.enabled ||
      (!isMaterialTransparent(material) &&
       (material.flags & SO_MAT_HAS_TEXTURE) == 0)) {
    return;
  }

  renderState.blend.enabled = TRUE;
  renderState.blend.srcRGBFactor = SO_BLEND_FACTOR_SRC_ALPHA;
  renderState.blend.dstRGBFactor = SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  renderState.blend.srcAlphaFactor = SO_BLEND_FACTOR_SRC_ALPHA;
  renderState.blend.dstAlphaFactor = SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  renderState.blend.rgbEquation = SO_BLEND_EQUATION_ADD;
  renderState.blend.alphaEquation = SO_BLEND_EQUATION_ADD;
  renderState.raster.pointShape = SO_POINT_SHAPE_SQUARE;
}

} // namespace SoRenderIR
