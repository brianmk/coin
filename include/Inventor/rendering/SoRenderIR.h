// include/Inventor/rendering/SoRenderIR.h

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
  \brief Retained intermediate representation for the Vulkan renderer.

  SoIRRenderAction produces a SoDrawList while traversing a scene graph, and
  Coin's Vulkan backends (SoVulkanRenderBackend for raster,
  SoRTXRenderBackend for ray tracing) consume that list to produce pixels.
  The types in this file deliberately use semantic values instead of OpenGL
  enums so the intermediate representation does not require a particular
  graphics API.

  \note This is the Vulkan/retained path, not a universal Coin abstraction.
  The production OpenGL viewport still renders through SoGLRenderAction
  directly and does not traverse the IR. SoGLRenderBackend consumes a
  SoDrawList only as a reference implementation used by the testsuite (see
  testsuite/drawlist-gl-test.cpp); it is not built into libCoin and is not on
  the OpenGL viewport's code path.

  Geometry and embedded texture pointers are borrowed from the producer. They
  normally refer to storage owned by the current SoIRRenderAction frame and
  must not be retained after that frame is cleared, rewound, or replaced.
  Device objects, caches, and other implementation resources belong to the
  consumer, not to the intermediate representation.
*/

/*!
  \enum SoPrimitiveTopology
  \brief Enumerates how primitives referenced by a geometry buffer should be interpreted.
*/
enum SoPrimitiveTopology : uint8_t {
  SO_TOPOLOGY_TRIANGLES = 0,
  SO_TOPOLOGY_LINES,
  SO_TOPOLOGY_POINTS,
  SO_TOPOLOGY_TRIANGLE_STRIP,
  SO_TOPOLOGY_LINE_STRIP,
  SO_TOPOLOGY_COUNT
};

// Semantic point coverage requested by the retained traversal.  Backends may
// emulate this when their native point rasterization cannot provide it.
enum SoPointShape : uint8_t {
  SO_POINT_SHAPE_SQUARE = 0,
  SO_POINT_SHAPE_ROUND
};

/*!
  \struct SoGeometryDesc
  \brief Describes vertex/index data for a single draw call.

  All pointers remain owned by the producer (typically SoIRRenderAction).
  They must remain valid while the backend consumes the frame. They may point
  into the action's frame geometry pool and must not be retained after that
  storage is cleared or rewound. Backends are free to copy the data into
  backend-owned buffers.

  Strides are byte distances between successive entries. A zero position or
  normal stride means three tightly packed floats; a zero texture-coordinate
  stride means four tightly packed floats. normalCount may be smaller than
  vertexCount when only part of a geometry has normals.
*/
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

  //!< Index of this command's first primitive within the source shape's
  //!< primitive stream.  A shape whose material changes partway is split into
  //!< several contiguous commands (see soshape_emit_ir_commands); each carries
  //!< the offset so a backend can map a per-command primitive id back to a
  //!< global primitive index (e.g. a SoBrepFaceSet face via partIndex).  Zero
  //!< when the command covers the shape from its start.
  uint32_t            primitiveOffset = 0;

  //!< Lifetime owners for the borrowed streams above.  When set, this command
  //!< co-owns the storage its raw pointers refer to (the pointers may be
  //!< offsets into these buffers), so the storage outlives the command even
  //!< if the producer (e.g. a shape's retained tessellation cache, which is
  //!< invalidated on any field change via SoShape::notify()) releases its own
  //!< reference before this command is re-emitted or replaced.  Without these
  //!< owners a retained drawlist command can hold a dangling pointer to a
  //!< freed-and-reused chunk and read garbage (a false geometry change).
  //!< Leave null for per-frame arena storage that is only valid for the frame.
  std::shared_ptr<const std::vector<float>>    positionOwner;
  std::shared_ptr<const std::vector<float>>    normalOwner;
  std::shared_ptr<const std::vector<float>>    texcoordOwner;
  std::shared_ptr<const std::vector<uint32_t>> indexOwner;

  //!< Producer guarantees the geometry streams are stable, shape-retained
  //!< buffers whose pointers change exactly when the content changes.  When
  //!< set, backends may rely on pointer/count identity alone to detect a
  //!< change (skipping a per-frame content hash); it must be false for
  //!< per-frame arena pools that rewrite the same pointer in place.
  bool                retained = false;
};

// --- Material flags (SoMaterialData::flags) ---
static constexpr uint32_t SO_MAT_HAS_TEXTURE = 0x1;  //!< Command carries embedded texture data
static constexpr uint32_t SO_MAT_IS_PIXEL_TEXT = 0x2;
static constexpr uint32_t SO_MAT_IS_PIXEL_IMAGE = 0x4;
static constexpr uint32_t SO_MAT_HAS_ROUGHNESS_MAP = 0x8;
static constexpr uint32_t SO_MAT_HAS_NORMAL_MAP = 0x10;
static constexpr uint32_t SO_MAT_HAS_EMISSIVE_MAP = 0x20;

// --- Feature flags (SoMaterialData::featureFlags) ---
static constexpr uint32_t SO_FEAT_BASE_COLOR = 0x1;   //!< Flat/unlit rendering (BASE_COLOR light model)

/*!
  \enum SoShadingModel
  \brief Effective shading contract carried by a render command.

  The legacy-compatible model is the current default. It preserves the
  fixed-function Coin/GL behavior while the DrawList backend is migrated to
  an explicit shading model.
*/
enum SoShadingModel : uint8_t {
  SO_SHADING_UNLIT = 0,
  SO_SHADING_LEGACY_GOURAUD
};

// --- Texture sampler state ---
// These are semantic sampler modes rather than OpenGL enum values so the IR
// can be consumed by non-OpenGL backends as well.
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

// Texture environment models retained from SoMultiTextureImageElement. These
// are semantic values; a backend maps them to its own texture-combine API.
enum SoTextureModel : uint8_t {
  SO_TEXTURE_MODEL_MODULATE = 0,
  SO_TEXTURE_MODEL_DECAL,
  SO_TEXTURE_MODEL_BLEND,
  SO_TEXTURE_MODEL_REPLACE
};

// --- Depth state ---------------------------------------------------------

// Semantic comparison functions. These deliberately do not use GL enum
// values: the IR is also consumed by backends which do not share GL's enum
// space.
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

/*!
  \struct SoTextureData
  \brief Embedded texture payload carried directly by a render command.

  This is used for commands that provide their own embedded image data.
  The memory is owned by the producer of the draw list and must remain valid
  until the backend finishes consuming the frame.
*/
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

/*!
  \struct SoMaterialData
  \brief Snapshot of the logical Inventor material state for one draw call.

  Texture pointers are backend-defined handles; the IR does not own the memory.
*/
struct SoMaterialData {
  SbVec4f  diffuse = {0.8f, 0.8f, 0.8f, 1.0f};
  SbVec4f  ambient = {0.2f, 0.2f, 0.2f, 1.0f};
  SbVec4f  specular = {0.0f, 0.0f, 0.0f, 1.0f};
  SbVec4f  emissive = {0.0f, 0.0f, 0.0f, 1.0f};
  // A command with no retained lighting setup is explicitly unlit. Scene
  // traversal fills this with the effective Gouraud model when lighting is
  // present, so the executor never needs to invent a headlight.
  SoShadingModel shadingModel = SO_SHADING_UNLIT;
  float    shininess = 0.2f;
  float    opacity = 1.0f;

  SoTextureData texture;  //!< Embedded texture data.

  // Some CPU-rasterized textures already multiply their texel alpha by
  // material opacity. Consumers use this to avoid multiplying that opacity a
  // second time while still composing vertex and texture alpha.
  bool     textureAlphaIncludesOpacity = false;

  // Material-derived per-vertex colors can already carry the effective
  // material transparency (for example SoMaterial PER_FACE colors). Packed
  // SoVertexProperty colors carry independent vertex alpha instead.
  bool     vertexColorAlphaIncludesOpacity = false;

  // Optional embedded secondary PBR maps, sampled with the base texture
  // coordinates.  Empty (pixels == nullptr) means the map is not used and the
  // corresponding strength is ignored.
  SoTextureData roughnessTexture;
  SoTextureData normalTexture;
  SoTextureData emissiveTexture;
  float    roughnessStrength = 1.0f;
  float    normalStrength = 1.0f;
  float    emissiveIntensity = 1.0f;

  float    metalness = 0.0f;
  float    roughness = 0.5f;
  // True when metalness/roughness were authored (a physical/PBR material)
  // rather than left at their defaults.  When false the backends use the
  // legacy Blinn-Phong model verbatim, so existing scenes are unchanged.
  bool     physicalMaterial = false;

  // Optical (transmission/glass) model.  transmissionIor and
  // transmissionAbsorption describe the material's dielectric response
  // (refraction index and Beer-Lambert absorption strength).  The transmittance
  // itself is opacity (== 1 - transparency), already carried by diffuse[3], so
  // both backends derive their transmission from one place.  Defaults match
  // air -> window glass.
  float    transmissionIor = 1.5f;
  float    transmissionAbsorption = 0.0f;
  // True when the material authored its own optics rather than leaving them at
  // the struct defaults.  Consumer-side glass settings (the path tracer's
  // global viewer IOR/absorption) are only a fallback: they must not override a
  // material that supplied its own values.  Until per-material authoring lands
  // this stays false, so the viewer setting remains the effective default.
  bool     transmissionAuthored = false;

  uint32_t flags = 0;
  uint32_t featureFlags = 0;
  bool     twoSidedLighting = false;
};

/*!
  \struct SoDepthState
  \brief Depth-test configuration for a draw call.
*/
struct SoDepthState {
  SbBool  enabled = TRUE;
  SbBool  writeEnabled = TRUE;
  SoDepthFunction func = SO_DEPTH_LEQUAL;
  SbVec2f range = SbVec2f(0.0f, 1.0f);
};

/*!
  \struct SoBlendState
  \brief Backend-neutral blending configuration.
*/
struct SoBlendState {
  SbBool  enabled = FALSE;
  SoBlendFactor srcRGBFactor = SO_BLEND_FACTOR_ONE;
  SoBlendFactor dstRGBFactor = SO_BLEND_FACTOR_ZERO;
  SoBlendFactor srcAlphaFactor = SO_BLEND_FACTOR_ONE;
  SoBlendFactor dstAlphaFactor = SO_BLEND_FACTOR_ZERO;

  // Coin's current LegacyGL state API exposes blend factors but not blend
  // equations. ADD is therefore the only equation that can be captured
  // from traversal today; separate fields keep the IR ready for a future
  // state source without pretending that it is currently preserved.
  SoBlendEquation rgbEquation = SO_BLEND_EQUATION_ADD;
  SoBlendEquation alphaEquation = SO_BLEND_EQUATION_ADD;
};

/*!
  \struct SoStencilState
  \brief Backend-neutral stencil test/operation configuration.

  Both faces share the same configuration (no two-sided stencil yet); a
  backend maps this to its API's front and back stencil state.
*/
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

/*!
  \struct SoAlphaTestState
  \brief Explicit fragment alpha policy for a render command.
*/
struct SoAlphaTestState {
  SoAlphaTestPolicy policy = SO_ALPHA_TEST_POLICY_NONE;
  SoAlphaTestFunction function = SO_ALPHA_TEST_NONE;
  float reference = 0.5f;
};

/*!
  \struct SoRasterState
  \brief Rasterizer properties (fill mode, culling, polygon offset).
*/
struct SoRasterState {
  uint8_t fillMode = 0;         // 0=filled, 1=lines (wireframe), 2=points
  SoPointShape pointShape = SO_POINT_SHAPE_SQUARE;
  uint8_t cullMode = 0;
  // GL front face follows the declared vertex ordering (SoShapeHintsElement,
  // glFrontFace): 1 = counterclockwise (the default), 0 = clockwise.  The
  // Vulkan backend must invert this on screen because its pipeline Y-flips
  // clip coordinates (a reflection reverses winding).
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

/*!
  \struct SoRenderState
  \brief Aggregates depth/blend/raster states plus precomputed sort keys.
*/
struct SoRenderState {
  SoDepthState depth;
  SoBlendState blend;
  SoStencilState stencil;
  SoAlphaTestState alphaTest;
  SoRasterState raster;
  uint32_t opaqueKey = 0;
  uint32_t translucentKey = 0;
};

/*!
  \enum SoRenderPassType
  \brief Logical pass identifier used for coarse sorting within a stage.
*/
enum SoRenderPassType : uint8_t {
  SO_RENDERPASS_OPAQUE = 0,
  SO_RENDERPASS_TRANSPARENT,
  //! Screen-space overlay geometry drawn after both previous passes with a
  //! per-rect viewport and its own depth clear (e.g. the navigation cube).
  //! Overlay commands carry their own view/projection matrices.
  SO_RENDERPASS_OVERLAY,
  SO_RENDERPASS_COUNT
};

/*!
  \typedef SoLightingHandle
  \brief Stable 1-based handle into the draw list's deduplicated lighting table.
*/
typedef uint32_t SoLightingHandle;

/*!
  \enum SoLightType
  \brief Light kinds captured in render-backend lighting setups.
*/
enum SoLightType : uint8_t {
  SO_LIGHT_DIRECTIONAL = 0,
  SO_LIGHT_POINT,
  SO_LIGHT_SPOT
};

/*!
  \brief Maximum number of lights a retained render-backend shader evaluates.

  All backends (GL visual program, Vulkan LightingBlock, RTX RTMaterial)
  evaluate at most this many lights; setups carrying more are truncated with
  a one-time warning by the consumer.
*/
constexpr int SO_MAX_SHADER_LIGHTS = 8;

/*!
  \struct SoLightData
  \brief Scene-space (world) light description used by the render backends.

  The geometry fields are expressed in the scene's WORLD space: `direction`
  is the normalized direction the light travels TOWARD (for a directional
  light this is the negated SoDirectionalLight::direction field, rotated by
  the light node's model matrix), and `position` is the light's world-space
  location for point/spot lights.  The data is view-independent -- it does
  not change when the camera moves -- so a draw list kept across a camera-
  only frame needs no light re-derivation.  Consumers that shade in eye
  space derive it with SoRenderIR::lightToEye(); the path tracer consumes
  the world-space fields directly.
*/
struct SoLightData {
  SoLightType type = SO_LIGHT_DIRECTIONAL;
  SbVec3f     color = SbVec3f(1.0f, 1.0f, 1.0f);
  SbVec3f     direction = SbVec3f(0.0f, 0.0f, 1.0f); // world, toward the light target
  SbVec3f     position = SbVec3f(0.0f, 0.0f, 1.0f);  // world
  SbVec3f     attenuation = SbVec3f(0.0f, 0.0f, 1.0f);
  float       spotCutoffCos = -1.0f;
  float       spotExponent = 0.0f;
};

/*!
  \struct SoLightingData
  \brief Shared (world-space) lighting setup referenced by render commands.
*/
struct SoLightingData {
  SbVec3f ambient = SbVec3f(0.2f, 0.2f, 0.2f);
  std::vector<SoLightData> lights;

  //! Number of lights a shader evaluates for this setup: the stored lights
  //! clamped to the fixed shader capacity.  The block packer and every backend
  //! that carries the count in a per-draw uniform share this one definition.
  int lightCount() const
  {
    return static_cast<int>(
      this->lights.size() < static_cast<size_t>(SO_MAX_SHADER_LIGHTS)
        ? this->lights.size()
        : static_cast<size_t>(SO_MAX_SHADER_LIGHTS));
  }
};

namespace SoRenderIR {

/*!
  \brief Transform one world-space light into eye space for \a view.

  Directions are rotated by the view matrix; positions are transformed
  fully.  Spot cone parameters are carried unchanged.
*/
COIN_DLL_API SoLightData lightToEye(const SoLightData & world,
                                    const SbMatrix & view);

/*!
  \brief Transform one eye-space light into world space for \a inverseView.

  The exact inverse of lightToEye(): pass the inverse of the world-to-eye
  view matrix.  Applications use this when they derive a camera-anchored light
  set (Coin GL's view-relative three-point lighting) and must hand the
  renderer world-space lights, so the eye<->world convention lives here rather
  than being re-derived at each call site.  Directions are rotated; positions
  are transformed fully; spot cone parameters are carried unchanged.
*/
COIN_DLL_API SoLightData lightToWorld(const SoLightData & eye,
                                      const SbMatrix & inverseView);

} // namespace SoRenderIR

/*!
  \struct SoLightingBlock
  \brief Standardized fixed-capacity GPU mirror of one world-space lighting
  setup, shared by every retained backend's lighting uniform/staging buffer.

  The layout is the std140 `LightingBlock` uniform layout of the Vulkan
  visual shaders byte-for-byte (and the array form the GL visual program
  uploads).  Producers fill it with SoRenderIR::fillLightingBlock();
  backends memcpy the finished block straight into their buffer, so the
  per-light field packing lives in exactly one place.  The evaluated light
  count travels separately (the raster path carries it in the per-draw
  material block), keeping the block itself a pure layout mirror.
*/
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

/*!
  \brief Fill a SoLightingBlock from a world-space SoLightingData.

  When \a toEye is non-NULL every light is transformed into eye space with
  it (raster/preview consumers); when NULL the world-space fields are copied
  verbatim (path-tracing consumers).  Ambient passes through unchanged.
  Returns the number of lights actually written (<= SO_MAX_SHADER_LIGHTS).
*/
COIN_DLL_API int fillLightingBlock(SoLightingBlock & block,
                                   const SoLightingData & world,
                                   const SbMatrix * toEye);

/*!
  \brief Effective ambient term for one material under a scene ambient.

  The raster and ray-traced backends must agree on how a per-material ambient
  colour combines with the scene ambient.  The raster fragment shader applies
  this in-shader (`ambientLight * materialAmbient`); the path tracer pre-
  multiplies it into the material record.  Both funnel through this single
  definition so the convention cannot drift between the two backends.
*/
COIN_DLL_API SbVec3f effectiveMaterialAmbient(const SbVec3f & sceneAmbient,
                                              const SbVec4f & materialAmbient);

/*!
  \struct SoMaterialBlock
  \brief Canonical mirror of a command's material parameters, shared by every
  retained backend's material staging.

  The raster backend copies these fields into its per-draw DrawBlock UBO and
  the path tracer copies them into its RTMaterial record, so the semantic
  mapping (which SoMaterialData field lands in which slot, and its default)
  lives in exactly one place.  Producers fill it with packMaterialBlock().

  The source diffuse is carried here for the backends that stage it in the
  material record (the path tracer); the raster backend still passes diffuse
  through its per-draw push constant to keep per-vertex/per-face colour
  working, and simply ignores \c diffuse.

  Two slots are context-dependent and are left for the caller after packing:
  \c params[2] (the evaluated light count) and, in the path tracer, an optional
  PBR gate override in \c pbr[2].
*/
struct COIN_DLL_API SoMaterialBlock {
  float diffuse[4];   // offset 0
  float ambient[4];   // offset 16
  float specular[4];  // offset 32
  float emissive[4];  // offset 48
  float params[4];    // offset 64: x=shininess, y=twoSided,
                      //   z=lightCount (caller), w=shadingModel
  float pbr[4];       // offset 80: x=metalness, y=roughness,
                      //   z=physicalMaterial, w=reserved
  float mapParams[4]; // offset 96: x=roughnessStrength, y=normalStrength,
                      //   z=emissiveIntensity, w=mapPresenceFlags
  float optical[4];   // offset 112: x=transmissionIor, y=transmissionAbsorption,
                      //   z=transmission (opacity), w=transmissionAuthored flag
};
static_assert(sizeof(SoMaterialBlock) == 128,
              "SoMaterialBlock must be 8 tightly packed vec4 (std140/std430)");
// Per-field offset locks.  A size-only guard passes even if two same-sized
// fields swap, silently remapping a value into the wrong shader slot; the
// offsets catch that.  Keep in lockstep with the std140 layout the raster
// DrawBlock UBO and the RTMaterial mirror are built from.
static_assert(offsetof(SoMaterialBlock, diffuse) == 0,
              "SoMaterialBlock.diffuse must be at offset 0");
static_assert(offsetof(SoMaterialBlock, ambient) == 16,
              "SoMaterialBlock.ambient must be at offset 16");
static_assert(offsetof(SoMaterialBlock, specular) == 32,
              "SoMaterialBlock.specular must be at offset 32");
static_assert(offsetof(SoMaterialBlock, emissive) == 48,
              "SoMaterialBlock.emissive must be at offset 48");
static_assert(offsetof(SoMaterialBlock, params) == 64,
              "SoMaterialBlock.params must be at offset 64");
static_assert(offsetof(SoMaterialBlock, pbr) == 80,
              "SoMaterialBlock.pbr must be at offset 80");
static_assert(offsetof(SoMaterialBlock, mapParams) == 96,
              "SoMaterialBlock.mapParams must be at offset 96");
static_assert(offsetof(SoMaterialBlock, optical) == 112,
              "SoMaterialBlock.optical must be at offset 112");

/*!
  \brief Fill a SoMaterialBlock from a command's SoMaterialData.

  One definition of the material mapping for every retained backend.  The
  light count (params[2]) and the path tracer's PBR gate override (pbr[2]) are
  left to the caller, which alone knows the effective lighting and the
  backend's gate policy.
*/
COIN_DLL_API void packMaterialBlock(SoMaterialBlock & block,
                                    const SoMaterialData & material);

/*!
  \brief Apply the scene's global glass optics to a packed material.

  \c optical[3] (the transmissionAuthored flag) decides precedence: when the
  material did not author its own optics, the viewer's global IOR/absorption
  are the effective defaults; when it did, the authored \c optical[0..1]
  survive untouched.  Shared by every backend that has a viewer-level glass
  setting, so a per-material dielectric is never silently replaced.
*/
COIN_DLL_API void resolveOptical(SoMaterialBlock & block,
                                 float globalIor,
                                 float globalAbsorption);

} // namespace SoRenderIR

/*!
  \struct SoLightingRaw
  \brief Scene-space inputs behind one SoLightingData entry (provenance).

  The setups themselves are world-space and view-independent, so nothing has
  to be re-derived on camera moves.  The raw entry is recorded for
  provenance and participates in deduplication, keeping setups that differ
  only by their originating light nodes distinct.
*/
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

/*!
  \struct SoRenderCommand
  \brief Complete description of a single draw call in the IR.
*/
struct SoRenderCommand {
  // Geometry, texture pixels, and other pointer-valued fields are borrowed;
  // see the lifetime contract on SoGeometryDesc and SoTextureData.
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
};

/*!
  \class SoDrawList
  \brief Container holding the commands and auxiliary tables for one frame.

  Commands retain their insertion order. buildSortedOrder() produces a
  separate index array for rendering; it never reorders the command vector.
  clear() starts a new frame and invalidates pointers
  into producer-owned frame storage.
*/
class COIN_DLL_API SoDrawList {
public:
  SoDrawList();

  //! Clear commands and per-frame tables, beginning a new frame generation.
  void clear();
  void reserve(int count);

  //! Return a process-unique generation, changed on construction and by
  //! clear().  Backends key GPU resource caches on it, so distinct draw lists
  //! (even at the same address) can never alias each other's cache entries.
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

  //! Variant carrying the scene-space inputs (SoLightingRaw) alongside the
  //! setup.  The raw entry participates in deduplication (two setups equal
  //! in their world-space fields but derived from different light nodes or
  //! cameras must stay separate).
  SoLightingHandle addLightingSetup(const SoLightingData & lighting,
                                    const SoLightingRaw & raw);

  //! Resolve a lighting handle previously returned by addLightingSetup().
  //! Returns NULL for handle 0 or an invalid handle.
  const SoLightingData * getLighting(SoLightingHandle handle) const;

  //! Compatibility no-op: lighting setups are world-space and view-
  //! independent, so a camera-only frame replaying a retained draw list
  //! needs no light re-derivation.  Kept exported (rather than removed) so
  //! already-linked consumers (e.g. the pivy Python bindings) continue to
  //! resolve the symbol; implementations do nothing.
  void restrikeLighting(const SbMatrix & prevView, const SbMatrix & newView);

  SoRenderCommand * begin();
  SoRenderCommand * end();
  const SoRenderCommand * begin() const;
  const SoRenderCommand * end() const;

  //! Build a sorted index array for correct render ordering.
  //! The draw list itself is NOT reordered — command indices stay stable.
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
