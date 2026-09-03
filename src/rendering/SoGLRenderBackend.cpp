// src/rendering/SoGLRenderBackend.cpp

#include "rendering/SoGLRenderBackend.h"

#include <Inventor/C/glue/gl.h>
#include <Inventor/errors/SoDebugError.h>

#include "glue/glp.h"
#include "glue/glslp.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include <data/shaders/gl/visual/Fragment.h>
#include <data/shaders/gl/visual/Vertex.h>
#include <data/shaders/gl/wide-line/Fragment.h>
#include <data/shaders/gl/wide-line/Geometry.h>
#include <data/shaders/gl/wide-line/Vertex.h>
#include <data/shaders/gl/point/Fragment.h>
#include <data/shaders/gl/point/Geometry.h>
#include <data/shaders/gl/point/Vertex.h>
#include <data/shaders/gl/pixel/Fragment.h>
#include <data/shaders/gl/pixel/Vertex.h>

namespace {

static constexpr int MAX_VERTEX_COUNT = 10000000;
static constexpr int MAX_SHADER_LIGHTS = 8;

GLenum
textureWrapToGL(const SoTextureWrap wrap)
{
  switch (wrap) {
  case SO_TEXTURE_WRAP_REPEAT: return GL_REPEAT;
  case SO_TEXTURE_WRAP_CLAMP_TO_BORDER: return GL_CLAMP_TO_BORDER;
  case SO_TEXTURE_WRAP_CLAMP_TO_EDGE:
  default: return GL_CLAMP_TO_EDGE;
  }
}

GLenum
textureMinFilterToGL(const SoTextureFilter filter)
{
  switch (filter) {
  case SO_TEXTURE_FILTER_LINEAR: return GL_LINEAR;
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
    return GL_NEAREST_MIPMAP_NEAREST;
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
    return GL_LINEAR_MIPMAP_NEAREST;
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    return GL_NEAREST_MIPMAP_LINEAR;
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
    return GL_LINEAR_MIPMAP_LINEAR;
  case SO_TEXTURE_FILTER_NEAREST:
  default: return GL_NEAREST;
  }
}

GLenum
textureMagFilterToGL(const SoTextureFilter filter)
{
  return filter == SO_TEXTURE_FILTER_NEAREST ? GL_NEAREST : GL_LINEAR;
}

GLenum
blendFactorToGL(const SoBlendFactor factor)
{
  switch (factor) {
  case SO_BLEND_FACTOR_ZERO: return GL_ZERO;
  case SO_BLEND_FACTOR_ONE: return GL_ONE;
  case SO_BLEND_FACTOR_SRC_COLOR: return GL_SRC_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return GL_ONE_MINUS_SRC_COLOR;
  case SO_BLEND_FACTOR_DST_COLOR: return GL_DST_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return GL_ONE_MINUS_DST_COLOR;
  case SO_BLEND_FACTOR_SRC_ALPHA: return GL_SRC_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return GL_ONE_MINUS_SRC_ALPHA;
  case SO_BLEND_FACTOR_DST_ALPHA: return GL_DST_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return GL_ONE_MINUS_DST_ALPHA;
  case SO_BLEND_FACTOR_CONSTANT_COLOR: return GL_CONSTANT_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
    return GL_ONE_MINUS_CONSTANT_COLOR;
  case SO_BLEND_FACTOR_CONSTANT_ALPHA: return GL_CONSTANT_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
    return GL_ONE_MINUS_CONSTANT_ALPHA;
  case SO_BLEND_FACTOR_SRC_ALPHA_SATURATE: return GL_SRC_ALPHA_SATURATE;
  // The Visual program has no secondary fragment output. Keep the semantic
  // factor in the IR and make the executor's deterministic primary-source
  // fallback only at this API boundary.
  case SO_BLEND_FACTOR_SRC1_COLOR: return GL_SRC_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return GL_ONE_MINUS_SRC_COLOR;
  case SO_BLEND_FACTOR_SRC1_ALPHA: return GL_SRC_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return GL_ONE_MINUS_SRC_ALPHA;
  default: return GL_ONE;
  }
}

bool
isDualSourceBlendFactor(const SoBlendFactor factor)
{
  return factor == SO_BLEND_FACTOR_SRC1_COLOR ||
         factor == SO_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ||
         factor == SO_BLEND_FACTOR_SRC1_ALPHA ||
         factor == SO_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
}

GLenum
blendEquationToGL(const SoBlendEquation equation)
{
  switch (equation) {
  case SO_BLEND_EQUATION_SUBTRACT: return GL_FUNC_SUBTRACT;
  case SO_BLEND_EQUATION_REVERSE_SUBTRACT: return GL_FUNC_REVERSE_SUBTRACT;
  case SO_BLEND_EQUATION_MIN: return GL_MIN;
  case SO_BLEND_EQUATION_MAX: return GL_MAX;
  case SO_BLEND_EQUATION_ADD:
  default: return GL_FUNC_ADD;
  }
}

GLenum
depthFunctionToGL(const SoDepthFunction function)
{
  switch (function) {
  case SO_DEPTH_NEVER: return GL_NEVER;
  case SO_DEPTH_ALWAYS: return GL_ALWAYS;
  case SO_DEPTH_LESS: return GL_LESS;
  case SO_DEPTH_LEQUAL: return GL_LEQUAL;
  case SO_DEPTH_EQUAL: return GL_EQUAL;
  case SO_DEPTH_GEQUAL: return GL_GEQUAL;
  case SO_DEPTH_GREATER: return GL_GREATER;
  case SO_DEPTH_NOTEQUAL: return GL_NOTEQUAL;
  default: return GL_LEQUAL;
  }
}

GLenum
topologyToGL(const SoPrimitiveTopology topology)
{
  switch (topology) {
  case SO_TOPOLOGY_POINTS: return GL_POINTS;
  case SO_TOPOLOGY_LINES: return GL_LINES;
  case SO_TOPOLOGY_TRIANGLES: return GL_TRIANGLES;
  case SO_TOPOLOGY_TRIANGLE_STRIP: return GL_TRIANGLE_STRIP;
  case SO_TOPOLOGY_LINE_STRIP: return GL_LINE_STRIP;
  default: return GL_TRIANGLES;
  }
}

void
applyViewport(const SoRenderParams & params)
{
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();
  glViewport(origin[0], origin[1], size[0], size[1]);
}

void
logShaderSourceMap(const char * source)
{
  const std::string marker = "// coin-source-id: ";
  const std::string sourceText = source ? source : "";
  std::string::size_type position = 0;
  while ((position = sourceText.find(marker, position)) != std::string::npos) {
    const std::string::size_type end = sourceText.find('\n', position);
    const std::string mapping = sourceText.substr(
      position + marker.length(),
      end == std::string::npos ? std::string::npos : end - position - marker.length());
    SoDebugError::postInfo("SoGLRenderBackend::compileShader",
                           "source ID map: %s", mapping.c_str());
    position = end == std::string::npos ? sourceText.length() : end + 1;
  }
}

GLuint
compileShader(const cc_glglue * glue, const GLenum type, const char * source)
{
  GLuint shader = cc_glglue_glCreateShader(glue, type);
  cc_glglue_glShaderSource(glue, shader, 1, &source, nullptr);
  cc_glglue_glCompileShader(glue, shader);

  GLint status = GL_FALSE;
  cc_glglue_glGetShaderiv(glue, shader, GL_COMPILE_STATUS, &status);
  if (status == GL_FALSE) {
    GLint length = 0;
    cc_glglue_glGetShaderiv(glue, shader, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
      std::string log(static_cast<size_t>(length), '\0');
      cc_glglue_glGetShaderInfoLog(glue, shader, length, &length, &log[0]);
      SoDebugError::postInfo("SoGLRenderBackend::compileShader",
                             "%s", log.c_str());
    }
    logShaderSourceMap(source);
    cc_glglue_glDeleteShader(glue, shader);
    return 0;
  }
  return shader;
}

GLuint
linkProgram(const cc_glglue * glue, const char * vertexSource,
            const char * fragmentSource, const char * geometrySource = nullptr)
{
  const GLuint vertex = compileShader(glue, GL_VERTEX_SHADER, vertexSource);
  const GLuint fragment = compileShader(glue, GL_FRAGMENT_SHADER, fragmentSource);
  const GLuint geometry = geometrySource
    ? compileShader(glue, GL_GEOMETRY_SHADER, geometrySource) : 0;
  if (!vertex || !fragment || (geometrySource && !geometry)) {
    if (vertex) cc_glglue_glDeleteShader(glue, vertex);
    if (fragment) cc_glglue_glDeleteShader(glue, fragment);
    if (geometry) cc_glglue_glDeleteShader(glue, geometry);
    return 0;
  }

  const GLuint program = cc_glglue_glCreateProgram(glue);
  cc_glglue_glAttachShader(glue, program, vertex);
  cc_glglue_glAttachShader(glue, program, fragment);
  if (geometry) cc_glglue_glAttachShader(glue, program, geometry);
  cc_glglue_glLinkProgram(glue, program);
  GLint linked = GL_FALSE;
  cc_glglue_glGetGLSLProgramiv(glue, program, GL_LINK_STATUS, &linked);
  if (linked == GL_FALSE) {
    GLint length = 0;
    cc_glglue_glGetGLSLProgramiv(glue, program, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
      std::string log(static_cast<size_t>(length), '\0');
      cc_glglue_glGetProgramInfoLog(glue, program, length, &length, &log[0]);
      SoDebugError::postInfo("SoGLRenderBackend::linkProgram", "%s",
                             log.c_str());
    }
    cc_glglue_glDeleteProgram(glue, program);
  }
  cc_glglue_glDeleteShader(glue, vertex);
  cc_glglue_glDeleteShader(glue, fragment);
  if (geometry) cc_glglue_glDeleteShader(glue, geometry);
  return linked == GL_FALSE ? 0 : program;
}

bool
textureDescriptionMatches(const CachedGPUCommand & entry,
                           const SoRenderCommand & command)
{
  const SoTextureData & texture = command.material.texture;
  return entry.texturePixelsKey == texture.pixels &&
    entry.textureWidth == texture.width &&
    entry.textureHeight == texture.height &&
    entry.textureComponents == texture.numComponents &&
    entry.textureMinFilter == texture.minFilter &&
    entry.textureMagFilter == texture.magFilter &&
    entry.textureWrapS == texture.wrapS &&
    entry.textureWrapT == texture.wrapT;
}

} // namespace

SoGLRenderBackend::SoGLRenderBackend()
{
}

SoGLRenderBackend::~SoGLRenderBackend()
{
  if (this->isInitialized()) this->shutdown();
}

const char *
SoGLRenderBackend::getName() const
{
  return "GLRenderBackend";
}

SbBool
SoGLRenderBackend::initialize(const SoRenderBackendInitParams & params)
{
  if (this->isInitialized()) return TRUE;

  this->setInitParams(params);
  void * context = coin_gl_current_context();
  this->glue = context ? cc_glglue_instance_from_context_ptr(context) : nullptr;
  if (!this->glue || !this->glue->glGenVertexArrays ||
      !this->glue->glBindVertexArray ||
      !this->glue->glDeleteVertexArrays ||
      !this->glue->glGetAttribLocation ||
      !this->glue->glVertexAttribPointer ||
      !this->glue->glEnableVertexAttribArray ||
      !this->glue->glDisableVertexAttribArray ||
      !this->glue->glVertexAttrib4f ||
      !this->glue->glVertexAttrib3f ||
      !this->glue->glVertexAttrib2f ||
      !this->glue->glVertexAttrib1f ||
      !this->glue->glUniform1f || !this->glue->glUniform1i ||
      !this->glue->glUniform2f ||
      !this->glue->glUniform3f || !this->glue->glUniform1iv ||
      !this->glue->glUniform2fv || !this->glue->glUniform3fv ||
      !this->glue->glUniform4f || !this->glue->glUniformMatrix4fv ||
      !this->glue->glBlendFuncSeparate) {
    this->emitError("active context does not provide retained-renderer GL dispatch");
    this->glue = nullptr;
    return FALSE;
  }

  if (!this->createShaders()) {
    this->emitError("failed to create retained core-profile shaders");
    this->glue = nullptr;
    return FALSE;
  }

  this->posLoc = this->glue->glGetAttribLocation(this->shaderProgram,
                                                  "a_position");
  this->colorLoc = this->glue->glGetAttribLocation(this->shaderProgram,
                                                    "a_color");
  this->texcoordLoc = this->glue->glGetAttribLocation(this->shaderProgram,
                                                       "a_texcoord");
  this->normLoc = this->glue->glGetAttribLocation(this->shaderProgram,
                                                   "a_normal");
  this->setInitialized(TRUE);
  return TRUE;
}

void
SoGLRenderBackend::destroyCacheEntry(CachedGPUCommand & entry)
{
  if (entry.posVBO) cc_glglue_glDeleteBuffers(this->glue, 1, &entry.posVBO);
  if (entry.normVBO) cc_glglue_glDeleteBuffers(this->glue, 1, &entry.normVBO);
  if (entry.colorVBO) cc_glglue_glDeleteBuffers(this->glue, 1, &entry.colorVBO);
  if (entry.texcoordVBO) {
    cc_glglue_glDeleteBuffers(this->glue, 1, &entry.texcoordVBO);
  }
  if (entry.lineDistVBO) {
    cc_glglue_glDeleteBuffers(this->glue, 1, &entry.lineDistVBO);
  }
  if (entry.textureId) {
    cc_glglue_glDeleteTextures(this->glue, 1, &entry.textureId);
  }
  if (entry.idxVBO) cc_glglue_glDeleteBuffers(this->glue, 1, &entry.idxVBO);
  if (entry.vao) this->glue->glDeleteVertexArrays(1, &entry.vao);
  entry = CachedGPUCommand();
}

void
SoGLRenderBackend::invalidateCache()
{
  if (this->glue) {
    for (CachedGPUCommand & entry : this->gpuCache) {
      this->destroyCacheEntry(entry);
    }
  }
  this->gpuCache.clear();
  this->commandToCache.clear();
  this->cachedCommandCount = 0;
  this->haveCacheGeneration = false;
}

void
SoGLRenderBackend::shutdown()
{
  if (!this->isInitialized()) return;

  this->invalidateCache();
  if (this->shaderProgram) {
    cc_glglue_glDeleteProgram(this->glue, this->shaderProgram);
    this->shaderProgram = 0;
  }
  if (this->lineShaderProgram) {
    cc_glglue_glDeleteProgram(this->glue, this->lineShaderProgram);
    this->lineShaderProgram = 0;
  }
  if (this->pointShaderProgram) {
    cc_glglue_glDeleteProgram(this->glue, this->pointShaderProgram);
    this->pointShaderProgram = 0;
  }
  if (this->pixelShaderProgram) {
    cc_glglue_glDeleteProgram(this->glue, this->pixelShaderProgram);
    this->pixelShaderProgram = 0;
  }
  this->glue = nullptr;
  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

CachedGPUCommand &
SoGLRenderBackend::getOrCreateCache(const SoRenderCommand * command)
{
  const auto found = this->commandToCache.find(command);
  if (found != this->commandToCache.end()) {
    return this->gpuCache[found->second];
  }

  const size_t index = this->gpuCache.size();
  this->gpuCache.emplace_back();
  this->commandToCache[command] = index;
  return this->gpuCache.back();
}

void
SoGLRenderBackend::uploadGeometry(CachedGPUCommand & entry,
                                  const SoRenderCommand & command)
{
  const SoGeometryDesc & geometry = command.geometry;
  const GLsizei vertexStride = static_cast<GLsizei>(
    geometry.vertexStride ? geometry.vertexStride : sizeof(float) * 3);

  if (!entry.posVBO) cc_glglue_glGenBuffers(this->glue, 1, &entry.posVBO);
  cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.posVBO);
  cc_glglue_glBufferData(this->glue, GL_ARRAY_BUFFER,
                         static_cast<GLsizeiptr>(geometry.vertexCount) *
                         vertexStride,
                         geometry.positions, GL_STATIC_DRAW);

  if (geometry.normals && geometry.normalCount >= geometry.vertexCount) {
    if (!entry.normVBO) cc_glglue_glGenBuffers(this->glue, 1, &entry.normVBO);
    cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.normVBO);
    cc_glglue_glBufferData(this->glue, GL_ARRAY_BUFFER,
                           static_cast<GLsizeiptr>(geometry.vertexCount) *
                           vertexStride,
                           geometry.normals, GL_STATIC_DRAW);
  }
  else if (entry.normVBO) {
    cc_glglue_glDeleteBuffers(this->glue, 1, &entry.normVBO);
    entry.normVBO = 0;
  }

  if (geometry.colors && geometry.vertexCount) {
    if (!entry.colorVBO) cc_glglue_glGenBuffers(this->glue, 1, &entry.colorVBO);
    cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.colorVBO);
    cc_glglue_glBufferData(this->glue, GL_ARRAY_BUFFER,
                           static_cast<GLsizeiptr>(geometry.vertexCount) *
                           sizeof(float) * 4,
                           geometry.colors, GL_STATIC_DRAW);
  }
  else if (entry.colorVBO) {
    cc_glglue_glDeleteBuffers(this->glue, 1, &entry.colorVBO);
    entry.colorVBO = 0;
  }

  const SoTextureData & texture = command.material.texture;
  const bool hasTexture = texture.pixels && texture.width > 0 &&
    texture.height > 0 && (texture.numComponents >= 1 &&
                           texture.numComponents <= 4) &&
    geometry.texcoords && geometry.vertexCount;

  if (hasTexture) {
    if (!entry.texcoordVBO) {
      cc_glglue_glGenBuffers(this->glue, 1, &entry.texcoordVBO);
    }
    cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.texcoordVBO);
    const uint32_t sourceStride = geometry.texcoordStride
      ? geometry.texcoordStride : sizeof(float) * 4;
    std::vector<float> texcoords(static_cast<size_t>(geometry.vertexCount) * 2);
    const char * raw = reinterpret_cast<const char *>(geometry.texcoords);
    for (uint32_t i = 0; i < geometry.vertexCount; ++i) {
      const float * source = reinterpret_cast<const float *>(
        raw + static_cast<size_t>(i) * sourceStride);
      texcoords[static_cast<size_t>(i) * 2] = source[0];
      texcoords[static_cast<size_t>(i) * 2 + 1] = source[1];
    }
    cc_glglue_glBufferData(this->glue, GL_ARRAY_BUFFER,
                           texcoords.size() * sizeof(float),
                           texcoords.data(), GL_STATIC_DRAW);

    if (!entry.textureId) {
      cc_glglue_glGenTextures(this->glue, 1, &entry.textureId);
    }
    cc_glglue_glBindTexture(this->glue, GL_TEXTURE_2D, entry.textureId);

    // Core profiles do not accept the legacy L/LA upload formats. Expand all
    // retained texture data to RGBA at the backend boundary.
    std::vector<unsigned char> rgba(static_cast<size_t>(texture.width) *
                                    static_cast<size_t>(texture.height) * 4);
    const unsigned char * source = texture.pixels;
    const int components = texture.numComponents;
    const size_t pixels = static_cast<size_t>(texture.width) *
      static_cast<size_t>(texture.height);
    for (size_t i = 0; i < pixels; ++i) {
      const unsigned char luminance = source[i * components];
      rgba[i * 4] = components == 3 || components == 4
        ? source[i * components] : luminance;
      rgba[i * 4 + 1] = components == 3 || components == 4
        ? source[i * components + 1] : luminance;
      rgba[i * 4 + 2] = components == 3 || components == 4
        ? source[i * components + 2] : luminance;
      rgba[i * 4 + 3] = components == 2 ? source[i * components + 1]
        : (components == 4 ? source[i * components + 3] : 255);
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    textureMinFilterToGL(texture.minFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    textureMagFilterToGL(texture.magFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    textureWrapToGL(texture.wrapS));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    textureWrapToGL(texture.wrapT));
    const bool mipmapped =
      texture.minFilter == SO_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST ||
      texture.minFilter == SO_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST ||
      texture.minFilter == SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR ||
      texture.minFilter == SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR;
    if (mipmapped && this->glue->glGenerateMipmap) {
      this->glue->glGenerateMipmap(GL_TEXTURE_2D);
    }
    cc_glglue_glBindTexture(this->glue, GL_TEXTURE_2D, 0);
  }
  else {
    if (entry.texcoordVBO) {
      cc_glglue_glDeleteBuffers(this->glue, 1, &entry.texcoordVBO);
      entry.texcoordVBO = 0;
    }
    if (entry.textureId) {
      cc_glglue_glDeleteTextures(this->glue, 1, &entry.textureId);
      entry.textureId = 0;
    }
  }

  const bool lineGeometry = geometry.topology == SO_TOPOLOGY_LINES ||
    geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  if (lineGeometry && geometry.vertexCount) {
    if (!entry.lineDistVBO) {
      cc_glglue_glGenBuffers(this->glue, 1, &entry.lineDistVBO);
    }
    std::vector<float> distances(geometry.vertexCount, 0.0f);
    const uint32_t strideFloats = static_cast<uint32_t>(vertexStride) /
      sizeof(float);
    const uint32_t count = geometry.indexCount && geometry.indices
      ? geometry.indexCount : geometry.vertexCount;
    if (geometry.topology == SO_TOPOLOGY_LINE_STRIP) {
      for (uint32_t i = 1; i < count; ++i) {
        const uint32_t previous = geometry.indices ? geometry.indices[i - 1] : i - 1;
        const uint32_t current = geometry.indices ? geometry.indices[i] : i;
        const float * p0 = geometry.positions + previous * strideFloats;
        const float * p1 = geometry.positions + current * strideFloats;
        const float dx = p1[0] - p0[0];
        const float dy = p1[1] - p0[1];
        const float dz = p1[2] - p0[2];
        distances[current] = distances[previous] +
          std::sqrt(dx * dx + dy * dy + dz * dz);
      }
    }
    else {
      for (uint32_t i = 0; i + 1 < count; i += 2) {
        const uint32_t first = geometry.indices ? geometry.indices[i] : i;
        const uint32_t second = geometry.indices ? geometry.indices[i + 1] : i + 1;
        const float * p0 = geometry.positions + first * strideFloats;
        const float * p1 = geometry.positions + second * strideFloats;
        const float dx = p1[0] - p0[0];
        const float dy = p1[1] - p0[1];
        const float dz = p1[2] - p0[2];
        distances[first] = 0.0f;
        distances[second] = std::sqrt(dx * dx + dy * dy + dz * dz);
      }
    }
    cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.lineDistVBO);
    cc_glglue_glBufferData(this->glue, GL_ARRAY_BUFFER,
                           distances.size() * sizeof(float), distances.data(),
                           GL_STATIC_DRAW);
    entry.lineDistKey = geometry.positions;
  }
  else if (entry.lineDistVBO) {
    cc_glglue_glDeleteBuffers(this->glue, 1, &entry.lineDistVBO);
    entry.lineDistVBO = 0;
    entry.lineDistKey = nullptr;
  }

  if (geometry.indexCount && geometry.indices) {
    if (!entry.idxVBO) cc_glglue_glGenBuffers(this->glue, 1, &entry.idxVBO);
    cc_glglue_glBindBuffer(this->glue, GL_ELEMENT_ARRAY_BUFFER, entry.idxVBO);
    cc_glglue_glBufferData(this->glue, GL_ELEMENT_ARRAY_BUFFER,
                           static_cast<GLsizeiptr>(geometry.indexCount) *
                           sizeof(uint32_t),
                           geometry.indices, GL_STATIC_DRAW);
  }
  else if (entry.idxVBO) {
    cc_glglue_glDeleteBuffers(this->glue, 1, &entry.idxVBO);
    entry.idxVBO = 0;
  }

  cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, 0);
  cc_glglue_glBindBuffer(this->glue, GL_ELEMENT_ARRAY_BUFFER, 0);

  entry.posKey = geometry.positions;
  entry.normKey = geometry.normals;
  entry.colorKey = geometry.colors;
  entry.texcoordKey = geometry.texcoords;
  entry.texturePixelsKey = hasTexture ? texture.pixels : nullptr;
  entry.idxKey = geometry.indices;
  entry.vertexCount = geometry.vertexCount;
  entry.normalCount = geometry.normalCount;
  entry.indexCount = geometry.indexCount;
  entry.vertexStride = static_cast<uint32_t>(vertexStride);
  entry.texcoordStride = geometry.texcoordStride;
  entry.textureWidth = hasTexture ? texture.width : 0;
  entry.textureHeight = hasTexture ? texture.height : 0;
  entry.textureComponents = hasTexture ? texture.numComponents : 0;
  entry.textureMinFilter = hasTexture ? texture.minFilter
                                      : SO_TEXTURE_FILTER_NEAREST;
  entry.textureMagFilter = hasTexture ? texture.magFilter
                                      : SO_TEXTURE_FILTER_NEAREST;
  entry.textureWrapS = hasTexture ? texture.wrapS
                                  : SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  entry.textureWrapT = hasTexture ? texture.wrapT
                                  : SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
}

void
SoGLRenderBackend::setupVisualVAO(CachedGPUCommand & entry)
{
  if (!entry.vao) this->glue->glGenVertexArrays(1, &entry.vao);
  this->glue->glBindVertexArray(entry.vao);

  if (this->posLoc >= 0 && entry.posVBO) {
    cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.posVBO);
    cc_glglue_glEnableVertexAttribArray(this->glue, this->posLoc);
    cc_glglue_glVertexAttribPointer(this->glue, this->posLoc, 3, GL_FLOAT,
                                    GL_FALSE, entry.vertexStride, nullptr);
  }
  if (this->normLoc >= 0) {
    if (entry.normVBO) {
      cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.normVBO);
      cc_glglue_glEnableVertexAttribArray(this->glue, this->normLoc);
      cc_glglue_glVertexAttribPointer(this->glue, this->normLoc, 3, GL_FLOAT,
                                      GL_FALSE, entry.vertexStride, nullptr);
    }
    else {
      cc_glglue_glDisableVertexAttribArray(this->glue, this->normLoc);
      this->glue->glVertexAttrib3f(this->normLoc, 0.0f, 0.0f, 1.0f);
    }
  }
  if (this->colorLoc >= 0) {
    if (entry.colorVBO) {
      cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.colorVBO);
      cc_glglue_glEnableVertexAttribArray(this->glue, this->colorLoc);
      cc_glglue_glVertexAttribPointer(this->glue, this->colorLoc, 4, GL_FLOAT,
                                      GL_FALSE, 0, nullptr);
    }
    else {
      cc_glglue_glDisableVertexAttribArray(this->glue, this->colorLoc);
      cc_glglue_glVertexAttrib4f(this->glue, this->colorLoc,
                                 1.0f, 1.0f, 1.0f, 1.0f);
    }
  }
  if (this->texcoordLoc >= 0) {
    if (entry.texcoordVBO) {
      cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.texcoordVBO);
      cc_glglue_glEnableVertexAttribArray(this->glue, this->texcoordLoc);
      cc_glglue_glVertexAttribPointer(this->glue, this->texcoordLoc, 2,
                                      GL_FLOAT, GL_FALSE, 0, nullptr);
    }
    else {
      cc_glglue_glDisableVertexAttribArray(this->glue, this->texcoordLoc);
      cc_glglue_glVertexAttrib2f(this->glue, this->texcoordLoc, 0.0f, 0.0f);
    }
  }
  if (this->lineDistLoc >= 0) {
    if (entry.lineDistVBO) {
      cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, entry.lineDistVBO);
      cc_glglue_glEnableVertexAttribArray(this->glue, this->lineDistLoc);
      cc_glglue_glVertexAttribPointer(this->glue, this->lineDistLoc, 1,
                                      GL_FLOAT, GL_FALSE, 0, nullptr);
    }
    else {
      cc_glglue_glDisableVertexAttribArray(this->glue, this->lineDistLoc);
      this->glue->glVertexAttrib1f(this->lineDistLoc, 0.0f);
    }
  }
  if (entry.idxVBO) {
    cc_glglue_glBindBuffer(this->glue, GL_ELEMENT_ARRAY_BUFFER, entry.idxVBO);
  }
  this->glue->glBindVertexArray(0);
  cc_glglue_glBindBuffer(this->glue, GL_ARRAY_BUFFER, 0);
  cc_glglue_glBindBuffer(this->glue, GL_ELEMENT_ARRAY_BUFFER, 0);
}

void
SoGLRenderBackend::updateGeometryCache(const SoDrawList & drawlist)
{
  const uint32_t generation = drawlist.getGeneration();
  if ((this->haveCacheGeneration && this->cacheGeneration != generation) ||
      (this->haveCacheGeneration &&
       this->cachedCommandCount != static_cast<size_t>(drawlist.getNumCommands()))) {
    this->invalidateCache();
  }
  this->cacheGeneration = generation;
  this->haveCacheGeneration = true;
  this->cachedCommandCount = static_cast<size_t>(drawlist.getNumCommands());

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) continue;

    CachedGPUCommand & entry = this->getOrCreateCache(&command);
    const uint32_t vertexStride = geometry.vertexStride
      ? geometry.vertexStride : sizeof(float) * 3;
    const bool lineGeometry = geometry.topology == SO_TOPOLOGY_LINES ||
      geometry.topology == SO_TOPOLOGY_LINE_STRIP;
    const bool geometryMatches = entry.posVBO != 0 &&
      entry.cacheGeneration == generation &&
      entry.posKey == geometry.positions &&
      entry.normKey == geometry.normals &&
      entry.colorKey == geometry.colors &&
      entry.texcoordKey == geometry.texcoords &&
      entry.idxKey == geometry.indices &&
      entry.vertexCount == geometry.vertexCount &&
      entry.normalCount == geometry.normalCount &&
      entry.indexCount == geometry.indexCount &&
      entry.vertexStride == vertexStride &&
      entry.texcoordStride == geometry.texcoordStride &&
      entry.lineDistKey == (lineGeometry ? geometry.positions : nullptr) &&
      textureDescriptionMatches(entry, command) &&
      ((entry.texturePixelsKey != nullptr) ==
       (command.material.texture.pixels != nullptr));
    if (!geometryMatches) {
      this->uploadGeometry(entry, command);
      this->setupVisualVAO(entry);
      entry.cacheGeneration = generation;
    }
  }
}

void
SoGLRenderBackend::uploadLighting(const SoDrawList & drawlist,
                                  const SoRenderCommand & command)
{
  const SoLightingData * lighting = drawlist.getLighting(command.lightingHandle);
  static const SoLightingData emptyLighting;
  if (!lighting) {
    lighting = &emptyLighting;
    if (command.lightingHandle != 0) {
      static std::once_flag invalidHandleWarning;
      std::call_once(invalidHandleWarning, []() {
        SoDebugError::postWarning(
          "SoGLRenderBackend::uploadLighting",
          "Draw command references missing lighting data; no headlight is "
          "synthesized.");
      });
    }
  }

  const SbVec3f & ambient = lighting->ambient;
  this->glue->glUniform3f(this->uAmbientLightLocation,
                          ambient[0], ambient[1], ambient[2]);

  GLint types[MAX_SHADER_LIGHTS] = {};
  GLfloat colors[MAX_SHADER_LIGHTS * 3] = {};
  GLfloat directions[MAX_SHADER_LIGHTS * 3] = {};
  GLfloat positions[MAX_SHADER_LIGHTS * 3] = {};
  GLfloat attenuations[MAX_SHADER_LIGHTS * 3] = {};
  GLfloat spotParams[MAX_SHADER_LIGHTS * 2] = {};
  const int count = std::min<int>(static_cast<int>(lighting->lights.size()),
                                  MAX_SHADER_LIGHTS);
  if (static_cast<int>(lighting->lights.size()) > MAX_SHADER_LIGHTS) {
    static std::once_flag lightLimitWarning;
    std::call_once(lightLimitWarning, []() {
      SoDebugError::postWarning(
        "SoGLRenderBackend::uploadLighting",
        "The Visual program supports eight lights; additional retained "
        "lights are ignored by this executor.");
    });
  }
  for (int i = 0; i < count; ++i) {
    const SoLightData & light = lighting->lights[static_cast<size_t>(i)];
    types[i] = static_cast<GLint>(light.type);
    colors[i * 3 + 0] = light.color[0];
    colors[i * 3 + 1] = light.color[1];
    colors[i * 3 + 2] = light.color[2];
    directions[i * 3 + 0] = light.direction[0];
    directions[i * 3 + 1] = light.direction[1];
    directions[i * 3 + 2] = light.direction[2];
    positions[i * 3 + 0] = light.position[0];
    positions[i * 3 + 1] = light.position[1];
    positions[i * 3 + 2] = light.position[2];
    attenuations[i * 3 + 0] = light.attenuation[0];
    attenuations[i * 3 + 1] = light.attenuation[1];
    attenuations[i * 3 + 2] = light.attenuation[2];
    spotParams[i * 2 + 0] = light.spotCutoffCos;
    spotParams[i * 2 + 1] = light.spotExponent;
  }
  this->glue->glUniform1i(this->uLightCountLocation, count);
  this->glue->glUniform1iv(this->uLightTypeLocation, MAX_SHADER_LIGHTS, types);
  this->glue->glUniform3fv(this->uLightColorLocation, MAX_SHADER_LIGHTS,
                           colors);
  this->glue->glUniform3fv(this->uLightDirectionLocation, MAX_SHADER_LIGHTS,
                           directions);
  this->glue->glUniform3fv(this->uLightPositionLocation, MAX_SHADER_LIGHTS,
                           positions);
  this->glue->glUniform3fv(this->uLightAttenuationLocation, MAX_SHADER_LIGHTS,
                           attenuations);
  this->glue->glUniform2fv(this->uLightSpotParamsLocation, MAX_SHADER_LIGHTS,
                           spotParams);
}

void
SoGLRenderBackend::bindPointShader(const SoRenderCommand & command,
                                   const SbMat & viewMat,
                                   const SbMat & projMat,
                                   const SbVec4f & color,
                                   const bool useVertexColor,
                                   const float pointSize,
                                   const SbVec2s & viewportSize)
{
  cc_glglue_glUseProgram(this->glue, this->pointShaderProgram);
  SbMat model;
  command.modelMatrix.getValue(model);
  this->glue->glUniformMatrix4fv(this->pointUViewLocation, 1, GL_FALSE,
                                 &viewMat[0][0]);
  this->glue->glUniformMatrix4fv(this->pointUProjLocation, 1, GL_FALSE,
                                 &projMat[0][0]);
  this->glue->glUniformMatrix4fv(this->pointUModelLocation, 1, GL_FALSE,
                                 &model[0][0]);
  this->glue->glUniform4f(this->pointUColorLocation,
                          color[0], color[1], color[2], color[3]);
  this->glue->glUniform1f(this->pointUUseVertexColorLocation,
                          useVertexColor ? 1.0f : 0.0f);
  this->glue->glUniform1f(this->pointUPointSizeLocation, pointSize);
  this->glue->glUniform1f(
    this->pointURoundPointsLocation,
    command.state.raster.pointShape == SO_POINT_SHAPE_ROUND ? 1.0f : 0.0f);
  this->glue->glUniform2f(this->pointUVpSizeLocation,
                          static_cast<float>(viewportSize[0]),
                          static_cast<float>(viewportSize[1]));
}

void
SoGLRenderBackend::bindLineShader(const SoRenderCommand & command,
                                  const SbMat & viewMat,
                                  const SbMat & projMat,
                                  const SbVec4f & color,
                                  const bool useVertexColor,
                                  const float lineWidth,
                                  const SbVec2s & viewportSize)
{
  cc_glglue_glUseProgram(this->glue, this->lineShaderProgram);
  SbMat model;
  command.modelMatrix.getValue(model);
  this->glue->glUniformMatrix4fv(this->lineUViewLocation, 1, GL_FALSE,
                                 &viewMat[0][0]);
  this->glue->glUniformMatrix4fv(this->lineUProjLocation, 1, GL_FALSE,
                                 &projMat[0][0]);
  this->glue->glUniformMatrix4fv(this->lineUModelLocation, 1, GL_FALSE,
                                 &model[0][0]);
  this->glue->glUniform4f(this->lineUColorLocation,
                          color[0], color[1], color[2], color[3]);
  this->glue->glUniform1f(this->lineUUseVertexColorLocation,
                          useVertexColor ? 1.0f : 0.0f);
  this->glue->glUniform1f(this->lineULineWidthLocation, lineWidth);
  this->glue->glUniform2f(this->lineUVpSizeLocation,
                          static_cast<float>(viewportSize[0]),
                          static_cast<float>(viewportSize[1]));
  this->glue->glUniform1f(this->lineUStipplePeriodLocation, 0.0f);
}

void
SoGLRenderBackend::bindPixelShader(const SoRenderCommand & command,
                                   const SbMat & viewMat,
                                   const SbMat & projMat,
                                   const SbVec2s & viewportSize)
{
  cc_glglue_glUseProgram(this->glue, this->pixelShaderProgram);
  SbMat model;
  command.modelMatrix.getValue(model);
  this->glue->glUniformMatrix4fv(this->pixelUViewLocation, 1, GL_FALSE,
                                 &viewMat[0][0]);
  this->glue->glUniformMatrix4fv(this->pixelUProjLocation, 1, GL_FALSE,
                                 &projMat[0][0]);
  this->glue->glUniformMatrix4fv(this->pixelUModelLocation, 1, GL_FALSE,
                                 &model[0][0]);

  const GLsizei stride = static_cast<GLsizei>(
    command.geometry.vertexStride ? command.geometry.vertexStride : sizeof(float) * 3);
  const char * raw = reinterpret_cast<const char *>(command.geometry.positions);
  SbVec3f center(0.0f, 0.0f, 0.0f);
  for (uint32_t i = 0; i < command.geometry.vertexCount; ++i) {
    const float * position = reinterpret_cast<const float *>(raw + i * stride);
    center += SbVec3f(position[0], position[1], position[2]);
  }
  if (command.geometry.vertexCount) {
    center /= static_cast<float>(command.geometry.vertexCount);
  }
  this->glue->glUniform3f(this->pixelUQuadCenterLocation,
                          center[0], center[1], center[2]);
  this->glue->glUniform2f(this->pixelUTexSizeLocation,
                          static_cast<float>(command.material.texture.width),
                          static_cast<float>(command.material.texture.height));
  this->glue->glUniform2f(this->pixelUVpSizeLocation,
                          static_cast<float>(viewportSize[0]),
                          static_cast<float>(viewportSize[1]));
  this->glue->glUniform2f(this->pixelUPixelOriginLocation,
                          static_cast<float>(command.pixelText.originX),
                          static_cast<float>(command.pixelText.originY));
  this->glue->glUniform1i(this->pixelUTextureLocation, 0);
  this->glue->glUniform4f(this->pixelUTexModColorLocation, 1.0f, 1.0f,
                          1.0f, 1.0f);
  const SbVec4f & color = command.material.diffuse;
  this->glue->glUniform4f(this->pixelUColorLocation,
                          color[0], color[1], color[2], color[3]);
  this->glue->glUniform1f(
    this->pixelUVertexColorAlphaIncludesOpacityLocation,
    command.material.vertexColorAlphaIncludesOpacity ? 1.0f : 0.0f);
  this->glue->glUniform1f(
    this->pixelUTextureAlphaIncludesOpacityLocation,
    command.material.textureAlphaIncludesOpacity ? 1.0f : 0.0f);
  this->glue->glUniform1i(
    this->pixelUAlphaTestFunctionLocation,
    command.state.alphaTest.policy == SO_ALPHA_TEST_POLICY_NONE
      ? 0 : static_cast<GLint>(command.state.alphaTest.function));
  this->glue->glUniform1f(this->pixelUAlphaTestReferenceLocation,
                          command.state.alphaTest.reference);
}

void
SoGLRenderBackend::drawCommand(const SoDrawList & drawlist,
                               const SoRenderCommand & command,
                               const SbMat & viewMat,
                               const SbMat & projMat,
                               const SoRenderParams & params)
{
  if (!command.geometry.positions || command.geometry.vertexCount == 0) return;
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) return;
  const CachedGPUCommand & entry = this->gpuCache[found->second];
  if (!entry.vao) return;

  const GLenum primitive = topologyToGL(command.geometry.topology);
  const bool textured = entry.textureId != 0 && entry.texcoordVBO != 0;
  const bool pixelRaster = textured &&
    (command.material.flags & (SO_MAT_IS_PIXEL_TEXT | SO_MAT_IS_PIXEL_IMAGE));
  const float dpr = params.devicePixelRatio > 0.0f
    ? params.devicePixelRatio : 1.0f;
  const float pointSize = std::max(1.0f, command.state.raster.pointSize) * dpr;
  const float lineWidth = std::max(1.0f, command.state.raster.lineWidth) * dpr;
  const bool patternedLine = command.state.raster.linePattern != 0xFFFF &&
    command.state.raster.linePattern != 0;
  const bool usePointShader = !pixelRaster && primitive == GL_POINTS &&
    this->pointShaderProgram != 0 &&
    (pointSize > 1.0f || command.state.raster.pointShape == SO_POINT_SHAPE_ROUND);
  const bool useLineShader = !pixelRaster &&
    (primitive == GL_LINES || primitive == GL_LINE_STRIP) &&
    this->lineShaderProgram != 0 && (lineWidth > 1.0f || patternedLine);

  applyViewport(params);
  this->glue->glUniformMatrix4fv(this->uViewLocation, 1, GL_FALSE,
                                 &viewMat[0][0]);
  this->glue->glUniformMatrix4fv(this->uProjLocation, 1, GL_FALSE,
                                 &projMat[0][0]);
  SbMat model;
  command.modelMatrix.getValue(model);
  this->glue->glUniformMatrix4fv(this->uModelLocation, 1, GL_FALSE,
                                 &model[0][0]);

  const SbVec4f & color = command.material.diffuse;
  this->glue->glUniform4f(this->uColorLocation,
                          color[0], color[1], color[2], color[3]);
  this->glue->glUniform1f(this->uUseVertexColorLocation,
                          entry.colorVBO ? 1.0f : 0.0f);
  this->glue->glUniform1f(this->uVertexColorAlphaIncludesOpacityLocation,
                          command.material.vertexColorAlphaIncludesOpacity
                            ? 1.0f : 0.0f);
  this->glue->glUniform1f(this->uTextureAlphaIncludesOpacityLocation,
                          command.material.textureAlphaIncludesOpacity
                            ? 1.0f : 0.0f);
  const SoShadingModel shadingModel =
    (command.material.featureFlags & SO_FEAT_BASE_COLOR)
      ? SO_SHADING_UNLIT : command.material.shadingModel;
  this->glue->glUniform1i(this->uShadingModelLocation,
                          static_cast<GLint>(shadingModel));
  const SbVec4f & emissive = command.material.emissive;
  const SbVec4f & ambient = command.material.ambient;
  const SbVec4f & specular = command.material.specular;
  this->glue->glUniform3f(this->uEmissiveColorLocation,
                          emissive[0], emissive[1], emissive[2]);
  this->glue->glUniform3f(this->uMaterialAmbientLocation,
                          ambient[0], ambient[1], ambient[2]);
  this->glue->glUniform3f(this->uMaterialSpecularLocation,
                          specular[0], specular[1], specular[2]);
  this->glue->glUniform1f(this->uMaterialShininessLocation,
                          command.material.shininess);
  this->glue->glUniform1f(this->uTwoSidedLightingLocation,
                          command.material.twoSidedLighting ? 1.0f : 0.0f);
  this->uploadLighting(drawlist, command);

  if (command.state.depth.enabled) {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(depthFunctionToGL(command.state.depth.func));
  }
  else {
    glDisable(GL_DEPTH_TEST);
  }
  // Match LegacyGL's transparent-object contract and keep transparent
  // geometry out of the depth buffer; otherwise triangle order inside a
  // retained command changes visibility and later transparent passes
  // self-occlude unpredictably.
  glDepthMask(command.state.depth.writeEnabled &&
              command.pass != SO_RENDERPASS_TRANSPARENT
                ? GL_TRUE : GL_FALSE);
  glDepthRange(command.state.depth.range[0], command.state.depth.range[1]);

  if (command.state.raster.cullMode) {
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
  }
  else {
    glDisable(GL_CULL_FACE);
  }
  const uint8_t fillMode = command.state.raster.fillMode;
  if (fillMode == 1 &&
      (primitive == GL_TRIANGLES || primitive == GL_TRIANGLE_STRIP)) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
  else if (fillMode == 2 &&
           (primitive == GL_TRIANGLES || primitive == GL_TRIANGLE_STRIP)) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
  }
  if (!usePointShader && (primitive == GL_POINTS || fillMode == 2)) {
    glPointSize(pointSize);
  }
  if (!useLineShader &&
      (primitive == GL_LINES || primitive == GL_LINE_STRIP || fillMode == 1)) {
    glLineWidth(lineWidth);
  }

  const bool blending = command.state.blend.enabled ||
    command.pass == SO_RENDERPASS_TRANSPARENT || color[3] < 0.999f;
  if (blending) {
    glEnable(GL_BLEND);
    if (isDualSourceBlendFactor(command.state.blend.srcRGBFactor) ||
        isDualSourceBlendFactor(command.state.blend.dstRGBFactor) ||
        isDualSourceBlendFactor(command.state.blend.srcAlphaFactor) ||
        isDualSourceBlendFactor(command.state.blend.dstAlphaFactor)) {
      static std::once_flag dualSourceWarning;
      std::call_once(dualSourceWarning, []() {
        SoDebugError::postWarning(
          "SoGLRenderBackend::drawCommand",
          "Dual-source blend factors are not supported by the Visual "
          "program; using primary-source factors for execution.");
      });
    }
    cc_glglue_glBlendFuncSeparate(
      this->glue, blendFactorToGL(command.state.blend.srcRGBFactor),
      blendFactorToGL(command.state.blend.dstRGBFactor),
      blendFactorToGL(command.state.blend.srcAlphaFactor),
      blendFactorToGL(command.state.blend.dstAlphaFactor));
    if (cc_glglue_has_blendequation(this->glue) &&
        command.state.blend.rgbEquation == command.state.blend.alphaEquation) {
      cc_glglue_glBlendEquation(
        this->glue, blendEquationToGL(command.state.blend.rgbEquation));
    }
  }
  else {
    glDisable(GL_BLEND);
  }
  const bool polygonOffset = command.state.raster.polygonOffsetFactor != 0.0f ||
    command.state.raster.polygonOffsetUnits != 0.0f;
  if (polygonOffset) {
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(command.state.raster.polygonOffsetFactor,
                    command.state.raster.polygonOffsetUnits);
  }
  this->glue->glUniform1i(
    this->uAlphaTestFunctionLocation,
    command.state.alphaTest.policy == SO_ALPHA_TEST_POLICY_NONE
      ? 0 : static_cast<GLint>(command.state.alphaTest.function));
  this->glue->glUniform1f(this->uAlphaTestReferenceLocation,
                          command.state.alphaTest.reference);

  this->glue->glUniform1f(this->uTextureEnabledLocation,
                          textured ? 1.0f : 0.0f);
  if (textured) {
    cc_glglue_glActiveTexture(this->glue, GL_TEXTURE0);
    cc_glglue_glBindTexture(this->glue, GL_TEXTURE_2D, entry.textureId);
    this->glue->glUniform1i(this->uTextureLocation, 0);
  }
  this->glue->glUniform1i(
    this->uTextureModelLocation,
    static_cast<GLint>(command.material.texture.model));
  const SbVec4f & textureBlend = command.material.texture.blendColor;
  this->glue->glUniform4f(this->uTextureBlendColorLocation,
                          textureBlend[0], textureBlend[1],
                          textureBlend[2], textureBlend[3]);

  if (pixelRaster) {
    this->bindPixelShader(command, viewMat, projMat,
                          params.viewport.getViewportSizePixels());
  }
  else if (usePointShader) {
    this->bindPointShader(command, viewMat, projMat, color,
                          entry.colorVBO != 0, pointSize,
                          params.viewport.getViewportSizePixels());
  }
  else if (useLineShader) {
    this->bindLineShader(command, viewMat, projMat, color,
                         entry.colorVBO != 0, lineWidth,
                         params.viewport.getViewportSizePixels());
    if (patternedLine) {
      int repeatLength = 16;
      for (int length = 1; length <= 8; ++length) {
        if (16 % length != 0) continue;
        const uint16_t mask = static_cast<uint16_t>((1u << length) - 1u);
        const uint16_t first = command.state.raster.linePattern & mask;
        bool repeats = true;
        for (int offset = length; offset < 16; offset += length) {
          if (((command.state.raster.linePattern >> offset) & mask) != first) {
            repeats = false;
            break;
          }
        }
        if (repeats) {
          repeatLength = length;
          break;
        }
      }
      const float pixelsPerUnit = std::max(1.0f,
        static_cast<float>(params.viewport.getViewportSizePixels()[0]) * 0.5f);
      const float period = static_cast<float>(std::max(
        1, static_cast<int>(command.state.raster.linePatternScale))) *
        static_cast<float>(repeatLength) / pixelsPerUnit;
      this->glue->glUniform1f(this->lineUStipplePeriodLocation, period);
    }
  }

  this->glue->glBindVertexArray(entry.vao);
  if (command.geometry.indexCount && command.geometry.indices) {
    cc_glglue_glDrawElements(this->glue, primitive,
                             static_cast<GLsizei>(command.geometry.indexCount),
                             GL_UNSIGNED_INT, nullptr);
  }
  else {
    cc_glglue_glDrawArrays(this->glue, primitive, 0,
                           static_cast<GLsizei>(command.geometry.vertexCount));
  }
  this->glue->glBindVertexArray(0);
  if (pixelRaster || usePointShader || useLineShader) {
    cc_glglue_glUseProgram(this->glue, this->shaderProgram);
  }
  if (textured) cc_glglue_glBindTexture(this->glue, GL_TEXTURE_2D, 0);
  if (polygonOffset) glDisable(GL_POLYGON_OFFSET_FILL);
  if (fillMode != 0 &&
      (primitive == GL_TRIANGLES || primitive == GL_TRIANGLE_STRIP)) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
  glDepthRange(0.0, 1.0);
  if (!usePointShader) glPointSize(1.0f);
  if (!useLineShader) glLineWidth(1.0f);
}

void
SoGLRenderBackend::renderOpaquePass(const SoDrawList & drawlist,
                                    const SbMat & viewMat,
                                    const SbMat & projMat,
                                    const SoRenderParams & params)
{
  const std::vector<int> & order = drawlist.getSortedOrder();
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const int index = i < static_cast<int>(order.size()) ? order[i] : i;
    const SoRenderCommand & command = drawlist.getCommand(index);
    if (command.pass == SO_RENDERPASS_OPAQUE) {
      this->drawCommand(drawlist, command, viewMat, projMat, params);
    }
  }
}

void
SoGLRenderBackend::renderTransparentPass(const SoDrawList & drawlist,
                                         const SbMat & viewMat,
                                         const SbMat & projMat,
                                         const SoRenderParams & params)
{
  const std::vector<int> & order = drawlist.getSortedOrder();
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const int index = i < static_cast<int>(order.size()) ? order[i] : i;
    const SoRenderCommand & command = drawlist.getCommand(index);
    if (command.pass == SO_RENDERPASS_TRANSPARENT) {
      this->drawCommand(drawlist, command, viewMat, projMat, params);
    }
  }
}

void
SoGLRenderBackend::beginFrame(const SoRenderParams & params)
{
  // Establish a deterministic baseline. These values are not interpretations
  // of retained Coin state; semantic depth/blend/raster execution is layered
  // above this executor.
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDepthMask(GL_TRUE);
  glDisable(GL_CULL_FACE);
  glDisable(GL_SCISSOR_TEST);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glPointSize(1.0f);
  glLineWidth(1.0f);

  if (params.flags & SO_PARAM_CLEAR_WINDOW) {
    const SbColor4f & color = params.clearColor;
    glClearColor(color[0], color[1], color[2], color[3]);
  }
  GLbitfield clearMask = 0;
  if (params.flags & SO_PARAM_CLEAR_WINDOW) clearMask |= GL_COLOR_BUFFER_BIT;
  if (params.flags & SO_PARAM_CLEAR_DEPTH) {
    glClearDepth(params.clearDepth);
    clearMask |= GL_DEPTH_BUFFER_BIT;
  }
  if (clearMask) glClear(clearMask);

  applyViewport(params);
  cc_glglue_glUseProgram(this->glue, this->shaderProgram);
}

bool
SoGLRenderBackend::createShaders()
{
  this->shaderProgram = linkProgram(this->glue,
                                    coin_gl_visual_vertex_shadersource,
                                    coin_gl_visual_fragment_shadersource);
  this->lineShaderProgram = linkProgram(this->glue,
                                        coin_gl_wide_line_vertex_shadersource,
                                        coin_gl_wide_line_fragment_shadersource,
                                        coin_gl_wide_line_geometry_shadersource);
  this->pointShaderProgram = linkProgram(this->glue,
                                         coin_gl_point_vertex_shadersource,
                                         coin_gl_point_fragment_shadersource,
                                         coin_gl_point_geometry_shadersource);
  this->pixelShaderProgram = linkProgram(
    this->glue, coin_gl_pixel_vertex_shadersource,
    coin_gl_pixel_fragment_shadersource);
  if (!this->shaderProgram || !this->lineShaderProgram ||
      !this->pointShaderProgram || !this->pixelShaderProgram) {
    if (this->shaderProgram) cc_glglue_glDeleteProgram(this->glue, this->shaderProgram);
    if (this->lineShaderProgram) cc_glglue_glDeleteProgram(this->glue, this->lineShaderProgram);
    if (this->pointShaderProgram) cc_glglue_glDeleteProgram(this->glue, this->pointShaderProgram);
    if (this->pixelShaderProgram) cc_glglue_glDeleteProgram(this->glue, this->pixelShaderProgram);
    this->shaderProgram = this->lineShaderProgram = this->pointShaderProgram =
      this->pixelShaderProgram = 0;
    return false;
  }

  auto uniform = [this](GLuint program, const char * name) {
    return cc_glglue_glGetUniformLocation(this->glue, program, name);
  };
  const GLuint visual = this->shaderProgram;
  this->uViewLocation = uniform(visual, "u_view");
  this->uProjLocation = uniform(visual, "u_proj");
  this->uModelLocation = uniform(visual, "u_model");
  this->uColorLocation = uniform(visual, "u_color");
  this->uUseVertexColorLocation = uniform(visual, "u_useVertexColor");
  this->uShadingModelLocation = uniform(visual, "u_shadingModel");
  this->uEmissiveColorLocation = uniform(visual, "u_emissiveColor");
  this->uMaterialAmbientLocation = uniform(visual, "u_materialAmbient");
  this->uMaterialSpecularLocation = uniform(visual, "u_materialSpecular");
  this->uMaterialShininessLocation = uniform(visual, "u_materialShininess");
  this->uTwoSidedLightingLocation = uniform(visual, "u_twoSidedLighting");
  this->uVertexColorAlphaIncludesOpacityLocation =
    uniform(visual, "u_vertexColorAlphaIncludesOpacity");
  this->uTextureAlphaIncludesOpacityLocation =
    uniform(visual, "u_textureAlphaIncludesOpacity");
  this->uAmbientLightLocation = uniform(visual, "u_ambientLight");
  this->uLightCountLocation = uniform(visual, "u_lightCount");
  this->uLightTypeLocation = uniform(visual, "u_lightType");
  this->uLightColorLocation = uniform(visual, "u_lightColor");
  this->uLightDirectionLocation = uniform(visual, "u_lightDirection");
  this->uLightPositionLocation = uniform(visual, "u_lightPosition");
  this->uLightAttenuationLocation = uniform(visual, "u_lightAttenuation");
  this->uLightSpotParamsLocation = uniform(visual, "u_lightSpotParams");
  this->uTextureLocation = uniform(visual, "u_texture");
  this->uTextureEnabledLocation = uniform(visual, "u_textureEnabled");
  this->uTextureModelLocation = uniform(visual, "u_textureModel");
  this->uTextureBlendColorLocation = uniform(visual, "u_textureBlendColor");
  this->uAlphaTestFunctionLocation = uniform(visual, "u_alphaTestFunction");
  this->uAlphaTestReferenceLocation = uniform(visual, "u_alphaTestReference");

  const GLuint line = this->lineShaderProgram;
  this->lineUViewLocation = uniform(line, "u_view");
  this->lineUProjLocation = uniform(line, "u_proj");
  this->lineUModelLocation = uniform(line, "u_model");
  this->lineUColorLocation = uniform(line, "u_color");
  this->lineUUseVertexColorLocation = uniform(line, "u_useVertexColor");
  this->lineULineWidthLocation = uniform(line, "u_lineWidth");
  this->lineUVpSizeLocation = uniform(line, "u_vpSize");
  this->lineUStipplePeriodLocation = uniform(line, "u_stipplePeriod");

  const GLuint point = this->pointShaderProgram;
  this->pointUViewLocation = uniform(point, "u_view");
  this->pointUProjLocation = uniform(point, "u_proj");
  this->pointUModelLocation = uniform(point, "u_model");
  this->pointUColorLocation = uniform(point, "u_color");
  this->pointUUseVertexColorLocation = uniform(point, "u_useVertexColor");
  this->pointUPointSizeLocation = uniform(point, "u_pointSize");
  this->pointURoundPointsLocation = uniform(point, "u_roundPoints");
  this->pointUVpSizeLocation = uniform(point, "u_vpSize");

  const GLuint pixel = this->pixelShaderProgram;
  this->pixelUViewLocation = uniform(pixel, "u_view");
  this->pixelUProjLocation = uniform(pixel, "u_proj");
  this->pixelUModelLocation = uniform(pixel, "u_model");
  this->pixelUQuadCenterLocation = uniform(pixel, "u_quadCenter");
  this->pixelUTexSizeLocation = uniform(pixel, "u_texSize");
  this->pixelUVpSizeLocation = uniform(pixel, "u_vpSize");
  this->pixelUPixelOriginLocation = uniform(pixel, "u_pixelOrigin");
  this->pixelUTextureLocation = uniform(pixel, "u_texture");
  this->pixelUTexModColorLocation = uniform(pixel, "u_texModColor");
  this->pixelUColorLocation = uniform(pixel, "u_color");
  this->pixelUVertexColorAlphaIncludesOpacityLocation =
    uniform(pixel, "u_vertexColorAlphaIncludesOpacity");
  this->pixelUTextureAlphaIncludesOpacityLocation =
    uniform(pixel, "u_textureAlphaIncludesOpacity");
  this->pixelUAlphaTestFunctionLocation = uniform(pixel, "u_alphaTestFunction");
  this->pixelUAlphaTestReferenceLocation = uniform(pixel, "u_alphaTestReference");
  return true;
}

SbBool
SoGLRenderBackend::render(const SoDrawList & drawlist,
                          const SoRenderParams & params)
{
  if (!this->isInitialized()) {
    this->emitError("render called before backend initialization");
    return FALSE;
  }

  this->debugValidateDrawList(drawlist);
  this->beginFrame(params);
  this->updateGeometryCache(drawlist);

  SbMat view;
  SbMat projection;
  params.viewMatrix.getValue(view);
  params.projMatrix.getValue(projection);

  this->renderOpaquePass(drawlist, view, projection, params);
  this->renderTransparentPass(drawlist, view, projection, params);
  cc_glglue_glUseProgram(this->glue, 0);
  return TRUE;
}
