// src/rendering/SoRenderIR.cpp

#include "rendering/SoRenderIRP.h"

#include <Inventor/C/tidbits.h>
#include <Inventor/elements/SoDepthBufferElement.h>
#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/elements/SoEnvironmentElement.h>
#include <Inventor/elements/SoLazyElement.h>
#include <Inventor/elements/SoLightAttenuationElement.h>
#include <Inventor/elements/SoLightElement.h>
#include <Inventor/elements/SoLightModelElement.h>
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
  // Keep the GL values local to this conversion boundary. No GL enum is
  // stored in the public IR.
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
  // Keep this mapping in lockstep with SoGLImageP::applyFilter() and its
  // documented LegacyGL quality thresholds. The IR stores the effective
  // sampler state so a backend does not need to know Coin's quality policy.
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

SbBool
coin_render_ir_trace_enabled()
{
  static int initialized = 0;
  static SbBool enabled = FALSE;
  if (!initialized) {
    enabled = coin_getenv("COIN_DEBUG_RENDER_IR") ? TRUE : FALSE;
    initialized = 1;
  }
  return enabled;
}

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

  // Need a new chunk — size it to at least fit this allocation
  // and to avoid many small chunks
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
  this->sortedOrder.clear();
  this->generation++;
}

void
SoDrawList::truncate(int count)
{
  if (count < static_cast<int>(this->commands.size())) {
    this->commands.resize(static_cast<size_t>(count));
    // The command vector remains insertion-ordered; sortedOrder is rebuilt
    // when the backend prepares the frame.
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

SoLightingHandle
SoDrawList::addLightingSetup(const SoLightingData & lighting)
{
  for (size_t i = 0; i < this->lightingSetups.size(); ++i) {
    if (lightingEqual(this->lightingSetups[i], lighting)) {
      return static_cast<SoLightingHandle>(i + 1);
    }
  }
  this->lightingSetups.push_back(lighting);
  return static_cast<SoLightingHandle>(this->lightingSetups.size());
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
    cmd.sortKey = SoIRComputeSortKey(cmd, passOrder, depthBucket);
  }

  // Sort the INDEX array by sort key, leaving commands in place
  std::stable_sort(sortedOrder.begin(), sortedOrder.end(),
    [arr](int a, int b) {
      return arr[a].sortKey < arr[b].sortKey;
    });
}

uint64_t
SoIRComputeSortKey(const SoRenderCommand & cmd,
                   uint32_t passOrderBits,
                   uint32_t depthBucket)
{
  const uint64_t passbits = (static_cast<uint64_t>(passOrderBits) & 0xffULL) << 56;
  const uint64_t depthbits = (static_cast<uint64_t>(depthBucket) & 0x00ffffffULL) << 32;
  const uint64_t pipelinebits = cmd.pipelineKey & 0x00000000ffffffffULL;
  return passbits | depthbits | pipelinebits;
}

static const char *
renderpass_name(SoRenderPassType pass)
{
  switch (pass) {
  case SO_RENDERPASS_OPAQUE: return "opaque";
  case SO_RENDERPASS_TRANSPARENT: return "transparent";
  default: return "unknown";
  }
}

void
SoIRDumpSummary(const SoDrawList & drawlist)
{
  if (!coin_render_ir_trace_enabled()) {
    return;
  }

  int counts[SO_RENDERPASS_COUNT] = { 0 };
  uint32_t minVerts = UINT32_MAX;
  uint32_t maxVerts = 0;
  const int num = drawlist.getNumCommands();
  for (int i = 0; i < num; ++i) {
    const SoRenderCommand & cmd = drawlist.getCommand(i);
    const uint32_t vc = cmd.geometry.vertexCount;
    minVerts = std::min(minVerts, vc);
    maxVerts = std::max(maxVerts, vc);
    if (cmd.pass < SO_RENDERPASS_COUNT) {
      counts[cmd.pass]++;
    }
  }

  SoDebugError::postInfo("SoDrawList",
                         "commands=%d opaque=%d transparent=%d minVerts=%u maxVerts=%u",
                         num,
                         counts[SO_RENDERPASS_OPAQUE],
                         counts[SO_RENDERPASS_TRANSPARENT],
                         minVerts == UINT32_MAX ? 0 : minVerts,
                         maxVerts);
}

void
SoIRDumpFirstN(const SoDrawList & drawlist, int count)
{
  if (!coin_render_ir_trace_enabled()) {
    return;
  }

  const int num = drawlist.getNumCommands();
  const int limit = std::min(num, count);
  for (int i = 0; i < limit; ++i) {
    const SoRenderCommand & cmd = drawlist.getCommand(i);
    const SbVec4f & diffuse = cmd.material.diffuse;
    const SoLightingData * lighting = drawlist.getLighting(cmd.lightingHandle);
    int numlights = lighting ? static_cast<int>(lighting->lights.size()) : -1;
    SbVec3f ambient(0.0f, 0.0f, 0.0f);
    if (lighting) {
      ambient = lighting->ambient;
    }
    SoDebugError::postInfo("SoDrawList",
                           "[%d] pass=%s depth=%d topo=%d verts=%u idx=%u colors=%p diffuse=(%.3f, %.3f, %.3f, %.3f) lights=%d ambient=(%.3f, %.3f, %.3f) pipeline=0x%016" PRIx64,
                           i,
                           renderpass_name(cmd.pass),
                           cmd.state.depth.enabled,
                           static_cast<int>(cmd.geometry.topology),
                           cmd.geometry.vertexCount,
                           cmd.geometry.indexCount,
                           cmd.geometry.colors,
                           diffuse[0],
                           diffuse[1],
                           diffuse[2],
                           diffuse[3],
                           numlights,
                           ambient[0],
                           ambient[1],
                           ambient[2],
                           static_cast<uint64_t>(cmd.pipelineKey));
  }
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

  // Keep diffuse and emissive independent. The explicit lighting shader owns
  // emissive contribution, so inferring diffuse from a default-looking
  // material would double-count emissive-only materials.
  material.diffuse.setValue(diffuse[0], diffuse[1], diffuse[2],
                            1.0f - transparency);

  // Capture the effective shading contract explicitly. Coin's traditional
  // PHONG light model currently maps to the legacy-compatible Gouraud path;
  // a true per-fragment PHONG path can be introduced without changing the
  // material/light payload carried by the IR.
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

  // A regular glBlendFunc applies the RGB factors to alpha as well. When
  // Coin's separate-alpha state is present, retain its factors verbatim,
  // including ZERO, which was previously indistinguishable from "not set".
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


  // LegacyGL does not expose a Coin state element for blend equations. ADD
  // is its effective equation and is the only value that can be captured
  // deterministically from traversal.
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
  // Native GL_POINTS are square unless point smoothing is enabled. Keep the
  // primitive shape explicit in the IR so backends do not choose independently.

  // Backface culling from SoShapeHintsElement:
  // vertexOrdering == COUNTERCLOCKWISE + shapeType == SOLID → cull back faces
  {
    SoShapeHintsElement::VertexOrdering vo;
    SoShapeHintsElement::ShapeType st;
    SoShapeHintsElement::FaceType ft;
    SoShapeHintsElement::get(mutableState, vo, st, ft);
    rs.raster.cullMode = (vo == SoShapeHintsElement::COUNTERCLOCKWISE
                       && st == SoShapeHintsElement::SOLID) ? 1 : 0;
  }
  rs.raster.scissorEnabled = FALSE;
  rs.raster.lineWidth = SoLineWidthElement::get(mutableState);
  rs.raster.pointSize = SoPointSizeElement::get(mutableState);

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

    if (light->isOfType(SoDirectionalLight::getClassTypeId())) {
      SoDirectionalLight * directional = static_cast<SoDirectionalLight *>(light);
      lightData.type = SO_LIGHT_DIRECTIONAL;
      lightMatrix.multDirMatrix(-(directional->direction.getValue()), lightData.direction);
      if (lightData.direction.normalize() == 0.0f) {
        lightData.direction.setValue(0.0f, 0.0f, 1.0f);
      }
    }
    else if (light->isOfType(SoPointLight::getClassTypeId())) {
      SoPointLight * point = static_cast<SoPointLight *>(light);
      lightData.type = SO_LIGHT_POINT;
      lightData.attenuation = attenuation;
      lightMatrix.multVecMatrix(point->location.getValue(), lightData.position);
    }
    else if (light->isOfType(SoSpotLight::getClassTypeId())) {
      SoSpotLight * spot = static_cast<SoSpotLight *>(light);
      lightData.type = SO_LIGHT_SPOT;
      lightData.attenuation = attenuation;
      lightMatrix.multVecMatrix(spot->location.getValue(), lightData.position);
      lightMatrix.multDirMatrix(spot->direction.getValue(), lightData.direction);
      if (lightData.direction.normalize() == 0.0f) {
        lightData.direction.setValue(0.0f, 0.0f, -1.0f);
      }
      float cutoff = spot->cutOffAngle.getValue();
      if (cutoff < 0.0f) cutoff = 0.0f;
      if (cutoff > float(M_PI) * 0.5f) cutoff = float(M_PI) * 0.5f;
      lightData.spotCutoffCos = std::cos(cutoff);
      float dropoff = spot->dropOffRate.getValue();
      if (dropoff < 0.0f) dropoff = 0.0f;
      if (dropoff > 1.0f) dropoff = 1.0f;
      lightData.spotExponent = dropoff * 128.0f;
    }
    else {
      continue;
    }

    lighting.lights.push_back(lightData);
  }

  return drawlist.addLightingSetup(lighting);
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
  // SoIRRenderAction captures Coin's logical material state, while the
  // legacy GL action enables the conventional blend function as part of its
  // transparency setup. Make that implicit IR contract explicit without
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
