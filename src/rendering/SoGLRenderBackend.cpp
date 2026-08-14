// src/rendering/SoGLRenderBackend.cpp

#include "rendering/SoGLRenderBackend.h"

#include <Inventor/C/glue/gl.h>
#include <Inventor/errors/SoDebugError.h>

#include "glue/glp.h"
#include "glue/glslp.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include <data/shaders/gl/visual/Fragment.h>
#include <data/shaders/gl/visual/Vertex.h>

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
      !this->glue->glUniform1f || !this->glue->glUniform1i ||
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
          "Ignoring an invalid retained lighting handle; no headlight is synthesized.");
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
        "The retained GL Visual program supports eight lights; additional "
        "retained lights are not uploaded.");
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
  this->glue->glUniform1i(
    this->uAlphaTestFunctionLocation,
    command.state.alphaTest.policy == SO_ALPHA_TEST_POLICY_NONE
      ? 0 : static_cast<GLint>(command.state.alphaTest.function));
  this->glue->glUniform1f(this->uAlphaTestReferenceLocation,
                          command.state.alphaTest.reference);

  const bool textured = entry.textureId != 0 && entry.texcoordVBO != 0;
  this->glue->glUniform1f(this->uTextureEnabledLocation,
                          textured ? 1.0f : 0.0f);
  if (textured) {
    cc_glglue_glActiveTexture(this->glue, GL_TEXTURE0);
    cc_glglue_glBindTexture(this->glue, GL_TEXTURE_2D, entry.textureId);
    this->glue->glUniform1i(this->uTextureLocation, 0);
  }
  this->glue->glUniform1i(this->uTextureModelLocation,
                          static_cast<GLint>(command.material.texture.model));
  const SbVec4f & textureBlend = command.material.texture.blendColor;
  this->glue->glUniform4f(this->uTextureBlendColorLocation,
                          textureBlend[0], textureBlend[1],
                          textureBlend[2], textureBlend[3]);

  const GLenum primitive = topologyToGL(command.geometry.topology);
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
  if (textured) cc_glglue_glBindTexture(this->glue, GL_TEXTURE_2D, 0);
  glDepthRange(0.0, 1.0);
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
  const GLuint vertex = compileShader(this->glue, GL_VERTEX_SHADER,
                                      coin_gl_visual_vertex_shadersource);
  const GLuint fragment = compileShader(this->glue, GL_FRAGMENT_SHADER,
                                        coin_gl_visual_fragment_shadersource);
  if (!vertex || !fragment) {
    if (vertex) cc_glglue_glDeleteShader(this->glue, vertex);
    if (fragment) cc_glglue_glDeleteShader(this->glue, fragment);
    return false;
  }

  const GLuint program = cc_glglue_glCreateProgram(this->glue);
  cc_glglue_glAttachShader(this->glue, program, vertex);
  cc_glglue_glAttachShader(this->glue, program, fragment);
  cc_glglue_glLinkProgram(this->glue, program);
  GLint linked = GL_FALSE;
  cc_glglue_glGetGLSLProgramiv(this->glue, program, GL_LINK_STATUS, &linked);
  cc_glglue_glDeleteShader(this->glue, vertex);
  cc_glglue_glDeleteShader(this->glue, fragment);
  if (linked == GL_FALSE) {
    cc_glglue_glDeleteProgram(this->glue, program);
    return false;
  }

  this->shaderProgram = program;
  this->uViewLocation = cc_glglue_glGetUniformLocation(this->glue, program,
                                                        "u_view");
  this->uProjLocation = cc_glglue_glGetUniformLocation(this->glue, program,
                                                         "u_proj");
  this->uModelLocation = cc_glglue_glGetUniformLocation(this->glue, program,
                                                          "u_model");
  this->uColorLocation = cc_glglue_glGetUniformLocation(this->glue, program,
                                                         "u_color");
  this->uUseVertexColorLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_useVertexColor");
  this->uShadingModelLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_shadingModel");
  this->uEmissiveColorLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_emissiveColor");
  this->uMaterialAmbientLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_materialAmbient");
  this->uMaterialSpecularLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_materialSpecular");
  this->uMaterialShininessLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_materialShininess");
  this->uTwoSidedLightingLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_twoSidedLighting");
  this->uVertexColorAlphaIncludesOpacityLocation =
    cc_glglue_glGetUniformLocation(this->glue, program,
                                   "u_vertexColorAlphaIncludesOpacity");
  this->uTextureAlphaIncludesOpacityLocation =
    cc_glglue_glGetUniformLocation(this->glue, program,
                                   "u_textureAlphaIncludesOpacity");
  this->uAmbientLightLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_ambientLight");
  this->uLightCountLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightCount");
  this->uLightTypeLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightType");
  this->uLightColorLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightColor");
  this->uLightDirectionLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightDirection");
  this->uLightPositionLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightPosition");
  this->uLightAttenuationLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightAttenuation");
  this->uLightSpotParamsLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_lightSpotParams");
  this->uTextureLocation = cc_glglue_glGetUniformLocation(this->glue, program,
                                                           "u_texture");
  this->uTextureEnabledLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_textureEnabled");
  this->uTextureModelLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_textureModel");
  this->uTextureBlendColorLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_textureBlendColor");
  this->uAlphaTestFunctionLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_alphaTestFunction");
  this->uAlphaTestReferenceLocation = cc_glglue_glGetUniformLocation(
    this->glue, program, "u_alphaTestReference");
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
