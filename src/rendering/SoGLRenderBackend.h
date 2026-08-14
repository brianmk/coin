// src/rendering/SoGLRenderBackend.h

#ifndef COIN_SOGLRENDERBACKEND_H
#define COIN_SOGLRENDERBACKEND_H

#include "rendering/SoRenderBackend.h"

#include <Inventor/system/gl.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct cc_glglue;

struct CachedGPUCommand {
  GLuint posVBO = 0;
  GLuint normVBO = 0;
  GLuint colorVBO = 0;
  GLuint texcoordVBO = 0;
  GLuint textureId = 0;
  GLuint idxVBO = 0;
  GLuint vao = 0;

  const float * posKey = nullptr;
  const float * normKey = nullptr;
  const float * colorKey = nullptr;
  const float * texcoordKey = nullptr;
  const unsigned char * texturePixelsKey = nullptr;
  const uint32_t * idxKey = nullptr;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t vertexStride = 0;
  uint32_t normalCount = 0;
  uint32_t texcoordStride = 0;
  int textureWidth = 0;
  int textureHeight = 0;
  int textureComponents = 0;
  SoTextureFilter textureMinFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureFilter textureMagFilter = SO_TEXTURE_FILTER_NEAREST;
  SoTextureWrap textureWrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  SoTextureWrap textureWrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  uint32_t cacheGeneration = 0;
};

/*! \brief Minimal core-profile OpenGL executor for retained DrawList IR. */
class SoGLRenderBackend : public SoRenderBackend {
public:
  SoGLRenderBackend();
  ~SoGLRenderBackend() override;

  const char * getName() const override;
  SbBool initialize(const SoRenderBackendInitParams & params) override;
  void shutdown() override;
  SbBool render(const SoDrawList & drawlist,
                const SoRenderParams & params) override;

private:
  bool createShaders();
  void beginFrame(const SoRenderParams & params);
  void invalidateCache();
  void updateGeometryCache(const SoDrawList & drawlist);
  void renderOpaquePass(const SoDrawList & drawlist,
                        const SbMat & viewMat,
                        const SbMat & projMat,
                        const SoRenderParams & params);
  void renderTransparentPass(const SoDrawList & drawlist,
                             const SbMat & viewMat,
                             const SbMat & projMat,
                             const SoRenderParams & params);
  void drawCommand(const SoDrawList & drawlist,
                   const SoRenderCommand & command,
                   const SbMat & viewMat,
                   const SbMat & projMat,
                   const SoRenderParams & params);
  void uploadLighting(const SoDrawList & drawlist,
                      const SoRenderCommand & command);

  CachedGPUCommand & getOrCreateCache(const SoRenderCommand * command);
  void uploadGeometry(CachedGPUCommand & entry,
                      const SoRenderCommand & command);
  void setupVisualVAO(CachedGPUCommand & entry);
  void destroyCacheEntry(CachedGPUCommand & entry);

  const cc_glglue * glue = nullptr;
  GLuint shaderProgram = 0;
  GLint uViewLocation = -1;
  GLint uProjLocation = -1;
  GLint uModelLocation = -1;
  GLint uColorLocation = -1;
  GLint uUseVertexColorLocation = -1;
  GLint uShadingModelLocation = -1;
  GLint uEmissiveColorLocation = -1;
  GLint uMaterialAmbientLocation = -1;
  GLint uMaterialSpecularLocation = -1;
  GLint uMaterialShininessLocation = -1;
  GLint uTwoSidedLightingLocation = -1;
  GLint uVertexColorAlphaIncludesOpacityLocation = -1;
  GLint uTextureAlphaIncludesOpacityLocation = -1;
  GLint uAmbientLightLocation = -1;
  GLint uLightCountLocation = -1;
  GLint uLightTypeLocation = -1;
  GLint uLightColorLocation = -1;
  GLint uLightDirectionLocation = -1;
  GLint uLightPositionLocation = -1;
  GLint uLightAttenuationLocation = -1;
  GLint uLightSpotParamsLocation = -1;
  GLint uTextureLocation = -1;
  GLint uTextureEnabledLocation = -1;
  GLint uTextureModelLocation = -1;
  GLint uTextureBlendColorLocation = -1;
  GLint uAlphaTestFunctionLocation = -1;
  GLint uAlphaTestReferenceLocation = -1;
  GLint posLoc = -1;
  GLint normLoc = -1;
  GLint colorLoc = -1;
  GLint texcoordLoc = -1;

  std::vector<CachedGPUCommand> gpuCache;
  std::unordered_map<const SoRenderCommand *, size_t> commandToCache;
  uint32_t cacheGeneration = 0;
  size_t cachedCommandCount = 0;
  bool haveCacheGeneration = false;
};

#endif // COIN_SOGLRENDERBACKEND_H
