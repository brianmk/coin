#ifndef COIN_SORENDERIR_H
#define COIN_SORENDERIR_H

#include <Inventor/SbBasic.h>
#include <Inventor/SbMatrix.h>
#include <Inventor/SbVec2f.h>
#include <Inventor/SbVec3f.h>
#include <Inventor/SbVec4f.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/*!
  \file SoRenderIR.h
  \brief Retained, API-neutral intermediate representation (IR) for the Vulkan renderer.

  Data flow:

    SoIRRenderAction  --traverses scene graph-->  SoDrawList  --consumed by-->  SoVulkanRenderBackend

  SoIRRenderAction walks a scene graph and appends a SoRenderCommand per draw
  call to a SoDrawList. SoVulkanRenderBackend consumes that list to produce
  pixels. Every type here uses semantic values instead of OpenGL enums, so the IR
  does not depend on any particular graphics API.

  \note This is the Vulkan/retained path, not a general Coin abstraction:

    - The production OpenGL viewport still renders through SoGLRenderAction and
      never traverses the IR.
    - SoGLRenderBackend consumes a SoDrawList only as a testsuite reference
      (testsuite/drawlist-gl-test.cpp); it is not built into libCoin.

  \note Lifetime: geometry and texture pointers are borrowed from the producer and
  refer to the current SoIRRenderAction frame's storage. Do not retain them after
  that frame is cleared, rewound, or replaced. Device objects and caches belong to
  the consumer. */

/*! \enum SoPrimitiveTopology \brief How primitives referenced by a geometry buffer are interpreted. */
enum SoPrimitiveTopology : uint8_t {
  SO_TOPOLOGY_TRIANGLES = 0,
  SO_TOPOLOGY_LINES,
  SO_TOPOLOGY_POINTS,
  SO_TOPOLOGY_TRIANGLE_STRIP,
  SO_TOPOLOGY_LINE_STRIP,
  SO_TOPOLOGY_COUNT
};

// Semantic point coverage; backends may emulate when native point rasterization cannot provide it.
enum SoPointShape : uint8_t {
  SO_POINT_SHAPE_SQUARE = 0,
  SO_POINT_SHAPE_ROUND
};

/*! \struct SoGeometryDesc \brief Vertex/index data for one draw call.
  Pointers are producer-owned, valid while the backend consumes the frame, and may
  point into the action's frame pool (never retain past clear/rewind); backends may
  copy. Strides are byte distances: 0 position/normal = 3 packed floats, 0 texcoord
  = 4 packed floats. normalCount may be < vertexCount. */
struct SoGeometryDesc {
  SoPrimitiveTopology topology = SO_TOPOLOGY_TRIANGLES;
  uint32_t            vertexCount = 0;
  uint32_t            normalCount = 0;
  uint32_t            indexCount = 0;

  const float *       positions = nullptr;
  const float *       normals = nullptr;
  const float *       texcoords = nullptr;
  const float *       colors = nullptr;
  const uint32_t *    indices = nullptr;

  uint32_t            vertexStride = 0;   //!< Position/normal stride in bytes.
  uint32_t            texcoordStride = 0; //!< Texture-coordinate stride in bytes.

  //!< Index of this command's first primitive in the source shape's stream (material-split
  //!< shapes emit contiguous commands, see soshape_emit_ir_commands) so a backend can map a
  //!< per-command primitive id to a global index (e.g. SoBrepFaceSet face via partIndex); 0 = start.
  uint32_t            primitiveOffset = 0;

  //!< Lifetime owners for the borrowed streams above: the command co-owns the storage its
  //!< raw pointers offset into, so it outlives the producer's reference (e.g. a tessellation
  //!< cache invalidated via SoShape::notify()); without them it can read a freed chunk. Null for frame arenas.
  std::shared_ptr<const std::vector<float>>    positionOwner;
  std::shared_ptr<const std::vector<float>>    normalOwner;
  std::shared_ptr<const std::vector<float>>    texcoordOwner;
  std::shared_ptr<const std::vector<uint32_t>> indexOwner;

  //!< Producer guarantees stable shape-retained streams whose pointers change exactly with
  //!< content; backends may then skip the content hash. False for in-place-rewritten arenas.
  bool                retained = false;
};

// --- Material flags (SoMaterialData::flags) ---
static constexpr uint32_t SO_MAT_HAS_TEXTURE = 0x1;  //!< Command carries embedded texture data
static constexpr uint32_t SO_MAT_IS_PIXEL_TEXT = 0x2;
static constexpr uint32_t SO_MAT_IS_PIXEL_IMAGE = 0x4;

// --- Feature flags (SoMaterialData::featureFlags) ---
static constexpr uint32_t SO_FEAT_BASE_COLOR = 0x1;   //!< Flat/unlit rendering (BASE_COLOR light model)

/*! \enum SoShadingModel \brief Effective shading contract carried by a render command.
  LEGACY_GOURAUD is the current default, preserving fixed-function Coin/GL behavior
  while the DrawList backend migrates to an explicit shading model. */
enum SoShadingModel : uint8_t {
  SO_SHADING_UNLIT = 0,
  SO_SHADING_LEGACY_GOURAUD
};

// --- Texture sampler state ---
// Semantic sampler modes, not OpenGL enum values, so non-OpenGL backends can consume the IR.
enum SoTextureFilter : uint8_t {
  SO_TEXTURE_FILTER_NEAREST = 0,
  SO_TEXTURE_FILTER_LINEAR,
  SO_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST,
  SO_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST,
  SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR,
  SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR
};

enum SoTextureWrap : uint8_t {
  SO_TEXTURE_WRAP_CLAMP_TO_EDGE = 0,
  SO_TEXTURE_WRAP_REPEAT,
  SO_TEXTURE_WRAP_CLAMP_TO_BORDER
};

// Texture environment models retained from SoMultiTextureImageElement (semantic; a backend maps them to its combine API).
enum SoTextureModel : uint8_t {
  SO_TEXTURE_MODEL_MODULATE = 0,
  SO_TEXTURE_MODEL_DECAL,
  SO_TEXTURE_MODEL_BLEND,
  SO_TEXTURE_MODEL_REPLACE
};

// --- Depth state ---------------------------------------------------------

// Semantic comparison functions (not GL enum values: the IR is also consumed by backends lacking GL's enum space).
enum SoDepthFunction : uint8_t {
  SO_DEPTH_NEVER = 0,
  SO_DEPTH_ALWAYS,
  SO_DEPTH_LESS,
  SO_DEPTH_LEQUAL,
  SO_DEPTH_EQUAL,
  SO_DEPTH_GEQUAL,
  SO_DEPTH_GREATER,
  SO_DEPTH_NOTEQUAL
};

// --- Blend state ---------------------------------------------------------

enum SoBlendFactor : uint8_t {
  SO_BLEND_FACTOR_ZERO = 0,
  SO_BLEND_FACTOR_ONE,
  SO_BLEND_FACTOR_SRC_COLOR,
  SO_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
  SO_BLEND_FACTOR_DST_COLOR,
  SO_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
  SO_BLEND_FACTOR_SRC_ALPHA,
  SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
  SO_BLEND_FACTOR_DST_ALPHA,
  SO_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
  SO_BLEND_FACTOR_CONSTANT_COLOR,
  SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
  SO_BLEND_FACTOR_CONSTANT_ALPHA,
  SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
  SO_BLEND_FACTOR_SRC_ALPHA_SATURATE,
  SO_BLEND_FACTOR_SRC1_COLOR,
  SO_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
  SO_BLEND_FACTOR_SRC1_ALPHA,
  SO_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
};

enum SoBlendEquation : uint8_t {
  SO_BLEND_EQUATION_ADD = 0,
  SO_BLEND_EQUATION_SUBTRACT,
  SO_BLEND_EQUATION_REVERSE_SUBTRACT,
  SO_BLEND_EQUATION_MIN,
  SO_BLEND_EQUATION_MAX
};

// --- Stencil state --------------------------------------------------------

// Semantic stencil test/operation functions (not GL enum values).
enum SoStencilFunction : uint8_t {
  SO_STENCIL_FUNC_NEVER = 0,
  SO_STENCIL_FUNC_ALWAYS,
  SO_STENCIL_FUNC_LESS,
  SO_STENCIL_FUNC_LEQUAL,
  SO_STENCIL_FUNC_EQUAL,
  SO_STENCIL_FUNC_GEQUAL,
  SO_STENCIL_FUNC_GREATER,
  SO_STENCIL_FUNC_NOTEQUAL
};

// Semantic stencil buffer update operations.
enum SoStencilOp : uint8_t {
  SO_STENCIL_OP_KEEP = 0,
  SO_STENCIL_OP_ZERO,
  SO_STENCIL_OP_REPLACE,
  SO_STENCIL_OP_INCREMENT,
  SO_STENCIL_OP_DECREMENT,
  SO_STENCIL_OP_INVERT,
  SO_STENCIL_OP_INCREMENT_WRAP,
  SO_STENCIL_OP_DECREMENT_WRAP
};

// --- Alpha-test policy --------------------------------------------------

enum SoAlphaTestFunction : uint8_t {
  SO_ALPHA_TEST_NONE = 0,
  SO_ALPHA_TEST_NEVER,
  SO_ALPHA_TEST_ALWAYS,
  SO_ALPHA_TEST_LESS,
  SO_ALPHA_TEST_LEQUAL,
  SO_ALPHA_TEST_EQUAL,
  SO_ALPHA_TEST_GEQUAL,
  SO_ALPHA_TEST_GREATER,
  SO_ALPHA_TEST_NOTEQUAL
};

enum SoAlphaTestPolicy : uint8_t {
  SO_ALPHA_TEST_POLICY_NONE = 0,
  SO_ALPHA_TEST_POLICY_EXPLICIT,
  SO_ALPHA_TEST_POLICY_LEGACY_THRESHOLD,
  SO_ALPHA_TEST_POLICY_PRESERVE_EDGES
};

// --- Render param flags (SoRenderParams::flags) ---
static constexpr uint32_t SO_PARAM_CLEAR_WINDOW = 1u;
static constexpr uint32_t SO_PARAM_CLEAR_DEPTH  = 4u;  //!< Clear depth buffer before rendering
static constexpr uint32_t SO_PARAM_CLEAR_STENCIL = 8u; //!< Clear stencil buffer before rendering

/*! \struct SoTextureData \brief Embedded texture payload carried directly by a render command.
  Producer-owned; must stay valid until the backend finishes consuming the frame. */
struct SoTextureData {
  const unsigned char * pixels = nullptr;
  int width = 0;
  int height = 0;
  int numComponents = 0; // 1=L, 2=LA, 3=RGB, 4=RGBA

  SoTextureFilter minFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureFilter magFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureWrap wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureWrap wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureModel model = SO_TEXTURE_MODEL_MODULATE;
  SbVec4f blendColor = SbVec4f(0.0f, 0.0f, 0.0f, 1.0f);
};

struct SoPixelTextData {
  int originX = 0;
  int originY = 0;
};

/*! \struct SoMaterialData \brief Logical Inventor material state for one draw call.
  Texture pointers are backend-defined handles; the IR does not own the memory. */
struct SoMaterialData {
  SbVec4f  diffuse = {0.8f, 0.8f, 0.8f, 1.0f};
  SbVec4f  ambient = {0.2f, 0.2f, 0.2f, 1.0f};
  SbVec4f  specular = {0.0f, 0.0f, 0.0f, 1.0f};
  SbVec4f  emissive = {0.0f, 0.0f, 0.0f, 1.0f};
  // No retained lighting setup => explicitly unlit; traversal fills the effective Gouraud model when lighting is present (no invented headlight).
  SoShadingModel shadingModel = SO_SHADING_UNLIT;
  float    shininess = 0.2f;
  float    opacity = 1.0f;

  SoTextureData texture;  //!< Embedded texture data.

  // Some CPU-rasterized textures pre-multiply texel alpha by material opacity; consumers use this to avoid multiplying it twice.
  bool     textureAlphaIncludesOpacity = false;

  // Material-derived per-vertex colors may already carry transparency (e.g. SoMaterial PER_FACE); packed SoVertexProperty colors carry independent alpha.
  bool     vertexColorAlphaIncludesOpacity = false;

  // Opaque backend-integration texture handles; the IR neither interprets nor owns them.
  void *   diffuseTexture = nullptr;
  void *   normalTexture = nullptr;
  void *   emissiveTexture = nullptr;

  float    metalness = 0.0f;
  float    roughness = 0.5f;

  uint32_t flags = 0;
  uint32_t featureFlags = 0;
  bool     twoSidedLighting = false;
};

/*! \struct SoDepthState \brief Depth-test configuration for a draw call. */
struct SoDepthState {
  SbBool  enabled = TRUE;
  SbBool  writeEnabled = TRUE;
  SoDepthFunction func = SO_DEPTH_LEQUAL;
  SbVec2f range = SbVec2f(0.0f, 1.0f);
};

/*! \struct SoBlendState \brief Backend-neutral blending configuration. */
struct SoBlendState {
  SbBool  enabled = FALSE;
  SoBlendFactor srcRGBFactor = SO_BLEND_FACTOR_ONE;
  SoBlendFactor dstRGBFactor = SO_BLEND_FACTOR_ZERO;
  SoBlendFactor srcAlphaFactor = SO_BLEND_FACTOR_ONE;
  SoBlendFactor dstAlphaFactor = SO_BLEND_FACTOR_ZERO;

  // LegacyGL exposes blend factors but not equations, so ADD is the only capturable equation today; separate fields keep the IR ready.
  SoBlendEquation rgbEquation = SO_BLEND_EQUATION_ADD;
  SoBlendEquation alphaEquation = SO_BLEND_EQUATION_ADD;
};

/*! \struct SoStencilState \brief Backend-neutral stencil test/operation configuration.
  Both faces share one configuration (no two-sided stencil yet); a backend maps it to
  its API's front/back stencil state. */
struct SoStencilState {
  SbBool  enabled = FALSE;
  SoStencilFunction function = SO_STENCIL_FUNC_ALWAYS;
  uint8_t reference = 0;
  uint8_t compareMask = 0xFF;
  uint8_t writeMask = 0xFF;
  SoStencilOp failOp = SO_STENCIL_OP_KEEP;
  SoStencilOp zfailOp = SO_STENCIL_OP_KEEP;
  SoStencilOp zpassOp = SO_STENCIL_OP_KEEP;
};

/*! \struct SoAlphaTestState \brief Explicit fragment alpha policy for a render command. */
struct SoAlphaTestState {
  SoAlphaTestPolicy policy = SO_ALPHA_TEST_POLICY_NONE;
  SoAlphaTestFunction function = SO_ALPHA_TEST_NONE;
  float reference = 0.5f;
};

/*! \struct SoRasterState \brief Rasterizer properties (fill mode, culling, polygon offset). */
struct SoRasterState {
  uint8_t fillMode = 0;         // 0=filled, 1=lines (wireframe), 2=points
  SoPointShape pointShape = SO_POINT_SHAPE_SQUARE;
  uint8_t cullMode = 0;
  // GL front face from SoShapeHintsElement/glFrontFace: 1 = CCW (default), 0 = CW. Vulkan must invert on screen (Y-flip reverses winding).
  uint8_t ccwFrontFace = 1;
  SbBool  scissorEnabled = FALSE;
  SbBool  viewportEnabled = FALSE;
  int     viewportX = 0;
  int     viewportY = 0;
  int     viewportWidth = 0;
  int     viewportHeight = 0;
  int     scissorX = 0;
  int     scissorY = 0;
  int     scissorWidth = 0;
  int     scissorHeight = 0;
  float   lineWidth = 1.0f;
  float   pointSize = 1.0f;
  uint16_t linePattern = 0xFFFF;
  int16_t  linePatternScale = 1;
  float   polygonOffsetFactor = 0.0f;
  float   polygonOffsetUnits = 0.0f;
};

/*! \struct SoRenderState \brief Aggregates depth/blend/raster states plus precomputed sort keys. */
struct SoRenderState {
  SoDepthState depth;
  SoBlendState blend;
  SoStencilState stencil;
  SoAlphaTestState alphaTest;
  SoRasterState raster;
  uint32_t opaqueKey = 0;
  uint32_t translucentKey = 0;
};

/*! \enum SoRenderPassType \brief Logical pass identifier for coarse sorting within a stage. */
enum SoRenderPassType : uint8_t {
  SO_RENDERPASS_OPAQUE = 0,
  SO_RENDERPASS_TRANSPARENT,
  //! Screen-space overlay after both prior passes, per-rect viewport, own depth clear (e.g. navigation cube); carries its own view/proj matrices.
  SO_RENDERPASS_OVERLAY,
  SO_RENDERPASS_COUNT
};

/*! \typedef SoLightingHandle \brief Stable 1-based handle into the draw list's deduplicated lighting table. */
typedef uint32_t SoLightingHandle;

/*! \enum SoLightType \brief Light kinds captured in render-backend lighting setups. */
enum SoLightType : uint8_t {
  SO_LIGHT_DIRECTIONAL = 0,
  SO_LIGHT_POINT,
  SO_LIGHT_SPOT
};

/*! \brief Maximum number of lights a retained render-backend shader evaluates.
  All backends (the GL visual program and the Vulkan LightingBlock) evaluate at
  most this many; larger setups are truncated with a one-time consumer warning. */
constexpr int SO_MAX_SHADER_LIGHTS = 8;

/*! \struct SoLightData \brief Scene-space (world) light description for the render backends.
  `direction` is the normalized travel-toward direction (directional: negated node
  direction rotated by its model matrix); `position` is world-space for point/spot.
  View-independent, so a camera-only frame needs no re-derivation: eye-space consumers
  use SoRenderIR::lightToEye(), world-space consumers use the fields directly. */
struct SoLightData {
  SoLightType type = SO_LIGHT_DIRECTIONAL;
  SbVec3f     color = SbVec3f(1.0f, 1.0f, 1.0f);
  SbVec3f     direction = SbVec3f(0.0f, 0.0f, 1.0f); // world, toward the light target
  SbVec3f     position = SbVec3f(0.0f, 0.0f, 1.0f);  // world
  SbVec3f     attenuation = SbVec3f(0.0f, 0.0f, 1.0f);
  float       spotCutoffCos = -1.0f;
  float       spotExponent = 0.0f;
};

/*! \struct SoLightingData \brief Shared (world-space) lighting setup referenced by render commands. */
struct SoLightingData {
  SbVec3f ambient = SbVec3f(0.2f, 0.2f, 0.2f);
  std::vector<SoLightData> lights;

  //! Lights a shader evaluates: stored lights clamped to SO_MAX_SHADER_LIGHTS; the block packer and per-draw-uniform backends share this definition.
  int lightCount() const
  {
    return static_cast<int>(
      this->lights.size() < static_cast<size_t>(SO_MAX_SHADER_LIGHTS)
        ? this->lights.size()
        : static_cast<size_t>(SO_MAX_SHADER_LIGHTS));
  }
};

namespace SoRenderIR {

/*! \brief Transform a world-space light into eye space for \a view.
  Directions are rotated, positions transformed fully; spot cone parameters unchanged. */
COIN_DLL_API SoLightData lightToEye(const SoLightData & world,
                                    const SbMatrix & view);

/*! \brief Transform an eye-space light into world space for \a inverseView.
  Exact inverse of lightToEye(): pass the inverse world-to-eye matrix. Used when
  deriving a camera-anchored light set (Coin GL's view-relative three-point lighting)
  and handing the renderer world-space lights; keeps the convention here, not per call site. */
COIN_DLL_API SoLightData lightToWorld(const SoLightData & eye,
                                      const SbMatrix & inverseView);

} // namespace SoRenderIR

/*! \struct SoLightingBlock \brief Fixed-capacity GPU mirror of one world-space lighting setup.
  Layout is byte-for-byte the std140 `LightingBlock` of the Vulkan visual shaders
  (and the array form the GL program uploads). Producers fill via
  SoRenderIR::fillLightingBlock(); backends memcpy straight in, so per-light packing
  lives in one place. The evaluated light count travels separately (per-draw material
  block on the raster path), keeping the block a pure layout mirror. */
struct COIN_DLL_API SoLightingBlock {
  float ambientLight[4];                            // offset 0
  float lightType[SO_MAX_SHADER_LIGHTS * 4];        // offset 16
  float lightColor[SO_MAX_SHADER_LIGHTS * 4];       // offset 144
  float lightDirection[SO_MAX_SHADER_LIGHTS * 4];   // offset 272
  float lightPosition[SO_MAX_SHADER_LIGHTS * 4];    // offset 400
  float lightAttenuation[SO_MAX_SHADER_LIGHTS * 4]; // offset 528
  float lightSpotParams[SO_MAX_SHADER_LIGHTS * 4];  // offset 656
};
static_assert(sizeof(SoLightingBlock) == 784,
              "SoLightingBlock must match the std140 LightingBlock layout");

namespace SoRenderIR {

/*! \brief Fill a SoLightingBlock from a world-space SoLightingData.
  `toEye` non-NULL transforms each light to eye space (raster/preview), NULL copies
  world fields verbatim; ambient passes unchanged. Returns lights written. */
COIN_DLL_API int fillLightingBlock(SoLightingBlock & block,
                                   const SoLightingData & world,
                                   const SbMatrix * toEye);

} // namespace SoRenderIR

/*! \struct SoLightingRaw \brief Scene-space inputs behind one SoLightingData entry (provenance).
  Setups are world-space/view-independent (no re-derivation on camera moves); the raw
  entry is recorded for provenance and dedup, keeping same-world setups from different
  originating light nodes distinct. */
struct SoLightingRaw {
  struct RawLight {
    SoLightType type = SO_LIGHT_DIRECTIONAL;
    SbVec3f color = SbVec3f(1.0f, 1.0f, 1.0f);       // color * intensity
    SbVec3f sceneDirection = SbVec3f(0.0f, 0.0f, -1.0f); // negated for directional
    SbVec3f scenePosition = SbVec3f(0.0f, 0.0f, 0.0f);
    SbVec3f attenuation = SbVec3f(0.0f, 0.0f, 0.0f);
    float spotCutoffCos = -1.0f;
    float spotExponent = 0.0f;
    SbMatrix sceneMatrix; //!< light model matrix (world <- light local)
  };
  bool hasRaw = false;
  SbVec3f ambient = SbVec3f(0.2f, 0.2f, 0.2f);
  std::vector<RawLight> lights;
};

/*! \struct SoRenderCommand \brief Complete description of a single draw call in the IR. */
struct SoRenderCommand {
  // Pointer-valued fields (geometry, texture pixels) are borrowed; see the lifetime contract on SoGeometryDesc and SoTextureData.
  SoGeometryDesc   geometry;
  SoMaterialData   material;
  SoRenderState    state;

  SbMatrix         modelMatrix;  // default-constructed to identity
  SbMatrix         viewMatrix;
  SbMatrix         projMatrix;

  SoRenderPassType pass = SO_RENDERPASS_OPAQUE;
  SoLightingHandle lightingHandle = 0;
  SoPixelTextData pixelText;
  uint64_t         sortKey = 0; //!< Backend-computed key used by sorting.
  void *           userData = nullptr; //!< Opaque, non-owned producer data.
  //! Producer reports this as the model's B-Rep feature-edge line set (SoShape::isFeatureEdgeSet()); the Vulkan edge overlay redraws only these.
  bool             isFeatureEdge = false;
};

/*! \class SoDrawList \brief Commands and auxiliary tables for one frame.
  Commands keep insertion order; buildSortedOrder() produces a separate index array and
  never reorders the command vector. clear() starts a new frame and invalidates pointers
  into producer-owned frame storage. */
class COIN_DLL_API SoDrawList {
public:
  SoDrawList();

  //! Clear commands and per-frame tables, beginning a new frame generation.
  void clear();
  void reserve(int count);

  //! Return the generation number incremented when clear() starts a new frame.
  uint32_t getGeneration() const { return generation; }

  void addCommand(const SoRenderCommand & cmd);
  SoRenderCommand & emplaceCommand();

  int getNumCommands() const;
  //! Remove commands beyond index count without reordering remaining commands.
  void truncate(int count);
  SoRenderCommand & getCommand(int i);
  const SoRenderCommand & getCommand(int i) const;

  //! Add or reuse a lighting setup and return its stable 1-based handle.
  SoLightingHandle addLightingSetup(const SoLightingData & lighting);

  //! Variant carrying scene-space inputs (SoLightingRaw) alongside the setup; the raw entry
  //! participates in dedup so same-world setups from different light nodes/cameras stay separate.
  SoLightingHandle addLightingSetup(const SoLightingData & lighting,
                                    const SoLightingRaw & raw);

  //! Resolve a handle from addLightingSetup(); NULL for 0 or an invalid handle.
  const SoLightingData * getLighting(SoLightingHandle handle) const;

  SoRenderCommand * begin();
  SoRenderCommand * end();
  const SoRenderCommand * begin() const;
  const SoRenderCommand * end() const;

  //! Build a sorted index array for render ordering; the draw list itself is NOT reordered.
  void buildSortedOrder(const SbMatrix & viewMatrix);

  //! Get the sorted rendering order (indices into the command list).
  const std::vector<int> & getSortedOrder() const { return sortedOrder; }

private:
  std::vector<SoRenderCommand> commands;
  std::vector<SoLightingData> lightingSetups;
  std::vector<SoLightingRaw> lightingRaws;
  std::vector<int> sortedOrder;
  uint32_t generation = 0;
};

#endif // COIN_SORENDERIR_H
