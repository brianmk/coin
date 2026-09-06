// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendFrame.cpp
//
// Frame orchestration and teardown.  Provides:
//
//   - shutdown(): release every owned Vulkan object (flush deferred destroys,
//     pipelines, render passes/framebuffers, shaders, buffers, descriptor
//     pools)
//   - renderInternal(): drive the render()/renderOverlaysOnly() entry points
//   - renderExternal()/renderExternalOverlay(): record into a caller-owned
//     command buffer
//   - recordFrame(): opaque/transparent passes + wireframe/point overlay
//     redraws + on-top annotations
//   - recordOverlayBlock() and recordTracedComposite()

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"

#include <Inventor/elements/SoDrawStyleElement.h>
#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace CoinVulkanDetail;

namespace {

long vkBackendRenderNowUs()
{
  return (long)std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

double vkBackendRenderNowMs()
{
  return vkBackendRenderNowUs() * 0.001;
}

// Phase timing for the fcprobe profile harness ([RTDBG] cpuTimingRaster),
// gated by the same FC_VULKAN_FRAME_TIMING flag as the manager and RTX
// [RTDBG] lines.  Cached: the environment does not change mid-process.
bool vkBackendFrameTimingEnabled()
{
  static const bool enabled =
    SoVulkanShared::envFlagEnabled("FC_VULKAN_FRAME_TIMING");
  return enabled;
}

bool vkBackendRenderBreadcrumbEnabled()
{
  static const bool enabled = std::getenv("FC_GUI_OPEN_BREADCRUMB") != nullptr;
  return enabled;
}

void vkBackendRenderBreadcrumbSince(long startUs, long thresholdUs, const char* phase)
{
  if (!vkBackendRenderBreadcrumbEnabled()) {
    return;
  }
  static int logged = 0;
  const long now = vkBackendRenderNowUs();
  if (logged < 30 && now - startUs >= thresholdUs) {
    ++logged;
    std::fprintf(stderr, "[VKBACKEND] %ld %s dur_us=%ld\n", startUs, phase, now - startUs);
    std::fflush(stderr);
  }
}

} // namespace

// True when a command draws through the CPU wide-line expansion path (which is
// not batchable -- each instance needs its own clip-space expansion).
static bool vkIsWideLine(const SoRenderCommand & c)
{
  const bool line =
    c.geometry.topology == SO_TOPOLOGY_LINES ||
    c.geometry.topology == SO_TOPOLOGY_LINE_STRIP;
  if (!line) return false;
  const bool patterned =
    c.state.raster.linePattern != 0xFFFF && c.state.raster.linePattern != 0;
  return c.state.raster.lineWidth > 1.0f || patterned;
}

// Two commands can be drawn as ONE instanced draw only when they share every
// pipeline, descriptor-set and push-constant input and differ solely by their
// model matrix.  `hashA`/`hashB` are the cached geometry content hashes.
// `pass` equality plus the caller rejecting overlay/wide-line commands keeps
// the batch inside the shared frame-camera main pass.
static bool vkCommandBatchable(const SoRenderCommand & a,
                               const SoRenderCommand & b,
                               uint64_t hashA, uint64_t hashB)
{
  if (a.geometry.topology != b.geometry.topology) return false;
  if (a.geometry.vertexCount != b.geometry.vertexCount) return false;
  if (a.geometry.indexCount != b.geometry.indexCount) return false;
  if (a.geometry.vertexStride != b.geometry.vertexStride) return false;
  if (a.geometry.texcoordStride != b.geometry.texcoordStride) return false;
  if (a.lightingHandle != b.lightingHandle) return false;
  if (a.pass != b.pass) return false;
  if (hashA != hashB) return false;
  // Compare the pipeline/push-determining state FIELD BY FIELD.  memcmp of the
  // sub-structs is unsafe: SbBool is an int and the enum fields leave padding
  // bytes that are not deterministically zeroed, so two identical cube states
  // could compare unequal.  The position-dependent sort keys
  // (SoRenderState::opaqueKey/translucentKey) are deliberately not compared --
  // identical geometry at different locations must still batch.
  const SoDepthState & da = a.state.depth, &db = b.state.depth;
  if (da.enabled != db.enabled || da.writeEnabled != db.writeEnabled ||
      da.func != db.func || da.range[0] != db.range[0] ||
      da.range[1] != db.range[1]) return false;
  const SoBlendState & ba = a.state.blend, &bb = b.state.blend;
  if (ba.enabled != bb.enabled ||
      ba.srcRGBFactor != bb.srcRGBFactor ||
      ba.dstRGBFactor != bb.dstRGBFactor ||
      ba.srcAlphaFactor != bb.srcAlphaFactor ||
      ba.dstAlphaFactor != bb.dstAlphaFactor ||
      ba.rgbEquation != bb.rgbEquation ||
      ba.alphaEquation != bb.alphaEquation) return false;
  const SoStencilState & sa = a.state.stencil, &sb = b.state.stencil;
  if (sa.enabled != sb.enabled || sa.function != sb.function ||
      sa.reference != sb.reference || sa.compareMask != sb.compareMask ||
      sa.writeMask != sb.writeMask || sa.failOp != sb.failOp ||
      sa.zfailOp != sb.zfailOp || sa.zpassOp != sb.zpassOp) return false;
  const SoAlphaTestState & aa = a.state.alphaTest, &ab = b.state.alphaTest;
  if (aa.policy != ab.policy || aa.function != ab.function ||
      aa.reference != ab.reference) return false;
  const SoRasterState & ra = a.state.raster, &rb = b.state.raster;
  if (ra.fillMode != rb.fillMode || ra.pointShape != rb.pointShape ||
      ra.cullMode != rb.cullMode || ra.ccwFrontFace != rb.ccwFrontFace ||
      ra.scissorEnabled != rb.scissorEnabled ||
      ra.viewportEnabled != rb.viewportEnabled ||
      ra.viewportX != rb.viewportX || ra.viewportY != rb.viewportY ||
      ra.viewportWidth != rb.viewportWidth ||
      ra.viewportHeight != rb.viewportHeight ||
      ra.scissorX != rb.scissorX || ra.scissorY != rb.scissorY ||
      ra.scissorWidth != rb.scissorWidth ||
      ra.scissorHeight != rb.scissorHeight ||
      ra.lineWidth != rb.lineWidth || ra.pointSize != rb.pointSize ||
      ra.linePattern != rb.linePattern ||
      ra.linePatternScale != rb.linePatternScale ||
      ra.polygonOffsetFactor != rb.polygonOffsetFactor ||
      ra.polygonOffsetUnits != rb.polygonOffsetUnits) return false;
  const SoMaterialData & ma = a.material;
  const SoMaterialData & mb = b.material;
  if (memcmp(&ma.diffuse[0], &mb.diffuse[0], sizeof(SbVec4f)) != 0) return false;
  if (memcmp(&ma.ambient[0], &mb.ambient[0], sizeof(SbVec4f)) != 0) return false;
  if (memcmp(&ma.specular[0], &mb.specular[0], sizeof(SbVec4f)) != 0) return false;
  if (memcmp(&ma.emissive[0], &mb.emissive[0], sizeof(SbVec4f)) != 0) return false;
  if (ma.shininess != mb.shininess) return false;
  if (ma.opacity != mb.opacity) return false;
  if (ma.twoSidedLighting != mb.twoSidedLighting) return false;
  if (ma.shadingModel != mb.shadingModel) return false;
  if (ma.vertexColorAlphaIncludesOpacity != mb.vertexColorAlphaIncludesOpacity)
    return false;
  if (ma.textureAlphaIncludesOpacity != mb.textureAlphaIncludesOpacity)
    return false;
  if ((ma.flags & (SO_MAT_HAS_TEXTURE | SO_MAT_IS_PIXEL_TEXT)) !=
      (mb.flags & (SO_MAT_HAS_TEXTURE | SO_MAT_IS_PIXEL_TEXT))) return false;
  if (ma.texture.pixels != mb.texture.pixels) return false;
  if (ma.texture.width != mb.texture.width) return false;
  if (ma.texture.height != mb.texture.height) return false;
  if (ma.texture.model != mb.texture.model) return false;
  if (ma.texture.minFilter != mb.texture.minFilter) return false;
  if (ma.texture.magFilter != mb.texture.magFilter) return false;
  if (ma.texture.wrapS != mb.texture.wrapS) return false;
  if (ma.texture.wrapT != mb.texture.wrapT) return false;
  if (memcmp(&ma.texture.blendColor[0], &mb.texture.blendColor[0],
             sizeof(SbVec4f)) != 0) return false;
  return true;
}

// Coarse grouping key for the opaque batching pass: two commands with the same
// key MIGHT be batchable (vkCommandBatchable re-verifies and splits any
// collision).  Hashes the fields that determine pipeline/descriptor/push state
// plus the cached geometry content hash.  Not intended to be collision-free;
// the pairwise re-verification is what guarantees correctness.
static uint64_t vkBatchKey(const SoRenderCommand & a, uint64_t contentHash)
{
  uint64_t h = contentHash;
  h ^= (uint64_t)a.geometry.topology << 0;
  h ^= (uint64_t)a.geometry.vertexCount << 8;
  h ^= (uint64_t)a.geometry.indexCount << 24;
  h ^= (uint64_t)a.lightingHandle << 40;
  h ^= (uint64_t)a.material.shadingModel << 56;
  const SoRasterState & r = a.state.raster;
  uint32_t s = (uint32_t)a.state.depth.func |
    ((uint32_t)(a.state.depth.enabled ? 1 : 0) << 3) |
    ((uint32_t)r.fillMode << 4) | ((uint32_t)r.cullMode << 6) |
    ((uint32_t)r.ccwFrontFace << 8) |
    ((uint32_t)r.linePattern << 9) |
    ((uint32_t)(a.state.blend.enabled ? 1 : 0) << 25);
  h ^= (uint64_t)s << 1;
  auto xu = [](uint32_t * out, const float * in) {
    std::memcpy(out, in, sizeof(uint32_t));
  };
  const float * dif = &a.material.diffuse[0];
  uint32_t d0, d1, d2, d3, shin, alphaRef, lw;
  xu(&d0, &dif[0]); xu(&d1, &dif[1]); xu(&d2, &dif[2]); xu(&d3, &dif[3]);
  xu(&shin, &a.material.shininess);
  xu(&alphaRef, &a.state.alphaTest.reference);
  xu(&lw, &r.lineWidth);
  h ^= (uint64_t)(d0 ^ d1 ^ d2 ^ d3 ^ shin ^ alphaRef ^ lw) << 17;
  return h;
}


// --- Lifecycle ------------------------------------------------------------

void
SoVulkanRenderBackend::shutdown()
{
  if (!this->isInitialized()) return;

  vkQueueWaitIdle(this->queue);

  // The queue is idle, so every deferred resource is safe to release now.
  this->flushAllPendingDestroys();

  // Release uploads abandoned by a frame that aborted between the cache
  // update and the flush/finalize step.  Their copies were never recorded,
  // so synchronous destruction is safe here.  The staged pixels live in the
  // shared staging pool, so there is no per-upload staging to free.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index < this->textureCache.size()) {
      this->destroyTextureEntry(this->textureCache[upload.index]);
    }
  }
  this->pendingUploads.clear();

  this->invalidateCache();
  this->destroyAllGeometryBlocks();

  for (auto & entry : this->pipelineCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->pipelineCache.clear();
  if (this->pipelineCacheHandle != VK_NULL_HANDLE) {
    vkDestroyPipelineCache(this->device, this->pipelineCacheHandle,
                           this->allocator);
    this->pipelineCacheHandle = VK_NULL_HANDLE;
  }

  for (auto & entry : this->backgroundPipelineCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->backgroundPipelineCache.clear();

  if (this->renderPassFramebuffer != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(this->device, this->renderPassFramebuffer,
                         this->allocator);
    this->renderPassFramebuffer = VK_NULL_HANDLE;
  }
  for (auto & entry : this->renderPassCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyRenderPass(this->device, entry.second, this->allocator);
    }
  }
  this->renderPassCache.clear();
  this->renderPass = VK_NULL_HANDLE;
  if (this->fragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->fragmentModule, this->allocator);
    this->fragmentModule = VK_NULL_HANDLE;
  }
  if (this->vertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->vertexModule, this->allocator);
    this->vertexModule = VK_NULL_HANDLE;
  }
  if (this->wideLineFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->wideLineFragmentModule,
                          this->allocator);
    this->wideLineFragmentModule = VK_NULL_HANDLE;
  }
  if (this->wideLineVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->wideLineVertexModule,
                          this->allocator);
    this->wideLineVertexModule = VK_NULL_HANDLE;
  }
  if (this->backgroundFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->backgroundFragmentModule, this->allocator);
    this->backgroundFragmentModule = VK_NULL_HANDLE;
  }
  if (this->backgroundVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->backgroundVertexModule, this->allocator);
    this->backgroundVertexModule = VK_NULL_HANDLE;
  }
  if (this->backgroundPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->backgroundPipelineLayout, this->allocator);
    this->backgroundPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->pipelineLayout, this->allocator);
    this->pipelineLayout = VK_NULL_HANDLE;
  }
  if (this->instanceModelMapped != nullptr) {
    vkUnmapMemory(this->device, this->instanceModelMemory);
    this->instanceModelMapped = nullptr;
  }
  if (this->instanceModelBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->instanceModelBuffer, this->allocator);
    this->instanceModelBuffer = VK_NULL_HANDLE;
  }
  if (this->instanceModelMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->instanceModelMemory, this->allocator);
    this->instanceModelMemory = VK_NULL_HANDLE;
  }
  this->instanceModelCapacity = 0;
  if (this->lightingMapped != nullptr) {
    vkUnmapMemory(this->device, this->lightingMemory);
    this->lightingMapped = nullptr;
  }
  if (this->lightingBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->lightingBuffer, this->allocator);
    this->lightingBuffer = VK_NULL_HANDLE;
  }
  if (this->lightingMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->lightingMemory, this->allocator);
    this->lightingMemory = VK_NULL_HANDLE;
  }
  if (this->lightingConstMapped != nullptr) {
    vkUnmapMemory(this->device, this->lightingConstMemory);
    this->lightingConstMapped = nullptr;
  }
  if (this->lightingConstBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->lightingConstBuffer, this->allocator);
    this->lightingConstBuffer = VK_NULL_HANDLE;
  }
  if (this->lightingConstMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->lightingConstMemory, this->allocator);
    this->lightingConstMemory = VK_NULL_HANDLE;
  }
  this->lightingDescriptorSet = VK_NULL_HANDLE;
  if (this->stagingPoolMapped != nullptr) {
    vkUnmapMemory(this->device, this->stagingPoolMemory);
    this->stagingPoolMapped = nullptr;
  }
  if (this->stagingPoolBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->stagingPoolBuffer, this->allocator);
    this->stagingPoolBuffer = VK_NULL_HANDLE;
  }
  if (this->stagingPoolMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->stagingPoolMemory, this->allocator);
    this->stagingPoolMemory = VK_NULL_HANDLE;
  }
  this->stagingPoolCapacity = 0;
  this->stagingPoolCursor = 0;
  for (auto & kv : this->samplerCache) {
    if (kv.second != VK_NULL_HANDLE) {
      vkDestroySampler(this->device, kv.second, this->allocator);
    }
  }
  this->samplerCache.clear();
  if (this->whiteSampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->whiteSampler, this->allocator);
    this->whiteSampler = VK_NULL_HANDLE;
  }
  if (this->whiteImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, this->whiteImageView, this->allocator);
    this->whiteImageView = VK_NULL_HANDLE;
  }
  if (this->whiteImage != VK_NULL_HANDLE) {
    vkDestroyImage(this->device, this->whiteImage, this->allocator);
    this->whiteImage = VK_NULL_HANDLE;
  }
  if (this->whiteImageMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->whiteImageMemory, this->allocator);
    this->whiteImageMemory = VK_NULL_HANDLE;
  }
  this->whiteDescriptorSet = VK_NULL_HANDLE;
  for (VkDescriptorPool pool : this->descriptorPools) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(this->device, pool, this->allocator);
    }
  }
  this->descriptorPools.clear();
  this->descriptorPool = VK_NULL_HANDLE;
  this->descriptorSetCount = 0;
  if (this->descriptorSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->descriptorSetLayout,
                                 this->allocator);
    this->descriptorSetLayout = VK_NULL_HANDLE;
  }
  if (this->lightingSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->lightingSetLayout,
                                 this->allocator);
    this->lightingSetLayout = VK_NULL_HANDLE;
  }
  this->releaseFrameResources();
  if (this->commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->commandPool, this->allocator);
    this->commandPool = VK_NULL_HANDLE;
  }
  if (this->secondaryCommandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->secondaryCommandPool,
                        this->allocator);
    this->secondaryCommandPool = VK_NULL_HANDLE;
  }
  if (this->memPool) {
    // Queue is idle and every deferred destroy has been flushed, so all
    // sub-allocated ranges are free and every block can be released.
    this->memPool->destroyAll();
    this->memPool.reset();
  }

  this->physicalDevice = VK_NULL_HANDLE;
  this->device = VK_NULL_HANDLE;
  this->queue = VK_NULL_HANDLE;
  this->allocator = nullptr;

  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

SbBool
SoVulkanRenderBackend::renderInternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      const bool overlaysOnly)
{
  if (!this->isInitialized()) {
    this->emitError("render called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "render called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  this->debugValidateDrawList(drawlist);

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BLACK_DEBUG")) {
    static int blackFrame = 0;
    int nTri = 0, nLine = 0, nOverlay = 0, nTrans = 0, nTriLit = 0;
    int nTriUnlit = 0;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      const SoRenderCommand & c = drawlist.getCommand(i);
      if (c.pass == SO_RENDERPASS_OVERLAY) nOverlay++;
      else if (c.pass == SO_RENDERPASS_TRANSPARENT) nTrans++;
      if (c.geometry.topology == SO_TOPOLOGY_TRIANGLES) {
        nTri++;
        if (c.material.shadingModel == SO_SHADING_LEGACY_GOURAUD) nTriLit++;
        else nTriUnlit++;
      }
      if (c.geometry.topology == SO_TOPOLOGY_LINES ||
          c.geometry.topology == SO_TOPOLOGY_LINE_STRIP) {
        nLine++;
      }
    }
    fprintf(stderr,
            "[BLACK] frame=%d overlaysOnly=%d flags=0x%x clear=(%.2f,%.2f,%.2f,%.2f) "
            "cmds=%d tri=%d(lit=%d unlit=%d) line=%d overlay=%d trans=%d\n",
            blackFrame++, static_cast<int>(overlaysOnly),
            static_cast<unsigned>(params.flags), params.clearColor[0],
            params.clearColor[1], params.clearColor[2], params.clearColor[3],
            drawlist.getNumCommands(), nTri, nTriLit, nTriUnlit, nLine,
            nOverlay, nTrans);
  }

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  this->cacheFrameMatrices(params);

  if (overlaysOnly) {
    bool hasOverlay = false;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      if (drawlist.getCommand(i).pass == SO_RENDERPASS_OVERLAY) {
        hasOverlay = true;
        break;
      }
    }
    if (!hasOverlay) return TRUE;
  }

  // One frame boundary: advances the ring cursor and releases resources
  // deferred maxFramesInFlight frames ago.
  this->beginFrame();

  // Write the lighting constant block(s) into the ring once per frame so the
  // shared lighting setup is referenced, not re-derived, per draw.
  this->updateLightingSetup(drawlist);

  // Render passes are cached by their attachment identity (formats, sample
  // count, image layouts, load ops), not by the target's images: swapchain
  // targets cycle their images every frame, and pipelines are keyed on the
  // render pass handle, so reusing the pass across image changes keeps the
  // pipeline cache warm.  On the full-target-clear fast path (FC_VULKAN_RP_CLEAR)
  // the color/depth attachments are cleared via their loadOp at pass begin,
  // which is cheaper than a separate vkCmdClearAttachments region clear.
  const bool wantRpClear = COIN_VULKAN_ENV_FLAG("FC_VULKAN_RP_CLEAR");
  const bool fullTargetClear =
    wantRpClear && this->isFullTargetClear(params, *target);
  const bool clearWindow = (params.flags & SO_PARAM_CLEAR_WINDOW) != 0;
  const bool clearDepth = (params.flags & SO_PARAM_CLEAR_DEPTH) != 0;
  const bool hasDepth = target->depthImageView != VK_NULL_HANDLE &&
                        target->depthFormat != VK_FORMAT_UNDEFINED;
  const VkAttachmentLoadOp colorLoadOp =
    (fullTargetClear && clearWindow)
      ? VK_ATTACHMENT_LOAD_OP_CLEAR
      : VK_ATTACHMENT_LOAD_OP_LOAD;
  const VkAttachmentLoadOp depthLoadOp =
    (fullTargetClear && hasDepth && clearDepth)
      ? VK_ATTACHMENT_LOAD_OP_CLEAR
      : VK_ATTACHMENT_LOAD_OP_LOAD;
  this->renderPass = this->getOrCreateRenderPass(*target, colorLoadOp,
                                                 depthLoadOp);
  // Stash whether the pass cleared each attachment so recordClear() can skip
  // the redundant vkCmdClearAttachments, and (below) so the begin info carries
  // the matching clear values.
  this->renderPassColorCleared = (colorLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  this->renderPassDepthCleared = (depthLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  if (this->renderPass == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan render pass");
    return FALSE;
  }

  this->updateGeometryCache(drawlist, overlaysOnly,
                            params.geometryContentUnchanged);

  // Composite renders skip recordFrame(), so reserve the ring slots here;
  // beginFrame() above already advanced the frame cursor.  This covers both
  // the OVERLAY commands and the non-triangle residual geometry.
  if (overlaysOnly &&
      !this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return FALSE;
  }

  if (!this->beginCommandBuffer()) {
    this->emitError("failed to begin Vulkan command buffer");
    return FALSE;
  }

  // The framebuffer is cached for the current target identity (image views +
  // extent + render pass) and recreated whenever any of those change.  The
  // old framebuffer is released through the deferred ring: an older in-flight
  // submission may still reference it (the per-frame vkQueueWaitIdle is gone,
  // so only the current slot's fence has been waited by beginFrame()).
  if (this->renderPassFramebuffer == VK_NULL_HANDLE ||
      this->renderPassFramebufferPass != this->renderPass ||
      this->renderPassFramebufferColorImage != target->colorImage ||
      this->renderPassFramebufferColorView != target->colorImageView ||
      this->renderPassFramebufferDepthImage != target->depthImage ||
      this->renderPassFramebufferDepthView != target->depthImageView ||
      this->renderPassFramebufferExtent.width != target->extent.width ||
      this->renderPassFramebufferExtent.height != target->extent.height) {
    if (this->renderPassFramebuffer != VK_NULL_HANDLE) {
      const VkDevice device = this->device;
      const VkAllocationCallbacks * allocator = this->allocator;
      const VkFramebuffer oldFramebuffer = this->renderPassFramebuffer;
      this->deferDestroy([device, allocator, oldFramebuffer]() {
        if (oldFramebuffer != VK_NULL_HANDLE) {
          vkDestroyFramebuffer(device, oldFramebuffer, allocator);
        }
      });
      this->renderPassFramebuffer = VK_NULL_HANDLE;
    }
    VkFramebufferCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass = this->renderPass;
    fci.attachmentCount =
      (target->depthImageView != VK_NULL_HANDLE &&
       target->depthFormat != VK_FORMAT_UNDEFINED)
        ? 2u : 1u;
    const VkImageView attachments[] = {
      target->colorImageView,
      target->depthImageView,
    };
    fci.pAttachments = attachments;
    fci.width = target->extent.width;
    fci.height = target->extent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(this->device, &fci, this->allocator,
                            &this->renderPassFramebuffer) != VK_SUCCESS) {
      this->emitError("failed to create Vulkan framebuffer");
      // The one-shot command buffer was begun above and never submitted; an
      // implicit reset only happens on submission, so reset it explicitly or
      // every later beginCommandBuffer() will fail.
      vkEndCommandBuffer(this->currentCommandBuffer());
      vkResetCommandBuffer(this->currentCommandBuffer(), 0);
      return FALSE;
    }
    this->renderPassFramebufferPass = this->renderPass;
    this->renderPassFramebufferColorImage = target->colorImage;
    this->renderPassFramebufferColorView = target->colorImageView;
    this->renderPassFramebufferDepthImage = target->depthImage;
    this->renderPassFramebufferDepthView = target->depthImageView;
    this->renderPassFramebufferExtent = target->extent;
  }
  const VkFramebuffer framebuffer = this->renderPassFramebuffer;

  // Record the pending texture copies into the frame command buffer (one
  // submit for the whole frame instead of a separate transfer submit) and
  // finalize the host-side resources.  The draws that sample these textures
  // are recorded below, after the copies, and the descriptor sets they bind
  // must already exist.  Staging buffers are released through the deferred
  // ring once the slot fence signals.
  if (!this->recordPendingTextureUploads()) {
    this->emitError("failed to record texture uploads");
  }
  this->finalizePendingTextureUploads();

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = this->renderPass;
  rpbi.framebuffer = framebuffer;
  rpbi.renderArea.offset = {0, 0};
  rpbi.renderArea.extent = target->extent;
  // When the render pass clears an attachment via its loadOp (full-target
  // clear fast path), the clear value must be supplied here.  clearValueCount
  // maps one-to-one to the attachment indices (0 = color, 1 = depth).
  VkClearValue clearValues[2];
  uint32_t clearValueCount = 0;
  if (this->renderPassColorCleared) {
    clearValues[0].color.float32[0] = params.clearColor[0];
    clearValues[0].color.float32[1] = params.clearColor[1];
    clearValues[0].color.float32[2] = params.clearColor[2];
    clearValues[0].color.float32[3] = params.clearColor[3];
    clearValueCount = 1;
  }
  if (this->renderPassDepthCleared) {
    clearValues[clearValueCount].depthStencil.depth = params.clearDepth;
    clearValues[clearValueCount].depthStencil.stencil = 0;
    ++clearValueCount;
  }
  rpbi.clearValueCount = clearValueCount;
  rpbi.pClearValues = clearValueCount ? clearValues : nullptr;

  vkCmdBeginRenderPass(this->currentCommandBuffer(), &rpbi,
                       VK_SUBPASS_CONTENTS_INLINE);

  this->recordContext.buffer = this->currentCommandBuffer();
  bool recorded = true;
  if (overlaysOnly) {
    this->recordTracedComposite(drawlist, params, *target, this->renderPass,
                                this->recordContext);
    this->recordOverlayBlock(drawlist, params, *target, this->renderPass,
                             this->recordContext);
  }
  else {
    recorded = this->recordFrame(drawlist, params, *target, this->renderPass,
                                 this->recordContext);
  }
  this->recordContext.buffer = VK_NULL_HANDLE;

  vkCmdEndRenderPass(this->currentCommandBuffer());

  // Submit even when recordFrame() failed: an unsubmitted one-shot command
  // buffer cannot be reused, and a partial frame is preferable to a dead
  // backend.
  const bool submitted = this->endAndSubmit();
  if (!submitted) {
    this->emitError("failed to submit Vulkan command buffer");
    return FALSE;
  }
  if (!recorded) {
    this->emitError("recordFrame failed; submitted a partial frame");
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderBackend::render(const SoDrawList & drawlist,
                              const SoRenderParams & params)
{
  return this->renderInternal(drawlist, params, false);
}

SbBool
SoVulkanRenderBackend::renderOverlaysOnly(const SoDrawList & drawlist,
                                          const SoRenderParams & params)
{
  return this->renderInternal(drawlist, params, true);
}

SbBool
SoVulkanRenderBackend::renderExternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass)
{
  if (!this->isInitialized()) {
    this->emitError("renderExternal called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "renderExternal called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    this->emitError(
      "renderExternal called without a command buffer and render pass");
    return FALSE;
  }

  // The external render pass is supplied by the caller (typically created with
  // LOAD loadOps and layered over a pre-existing image), so no attachment is
  // cleared by a loadOp here: recordClear() must emit vkCmdClearAttachments.
  this->renderPassColorCleared = false;
  this->renderPassDepthCleared = false;

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BLACK_DEBUG"))
    fprintf(stderr, "[BLACK] renderExternal ENTER frame=%d cmds=%d\n",
            this->uboFrameIndex, drawlist.getNumCommands());

  this->debugValidateDrawList(drawlist);

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  this->cacheFrameMatrices(params);
  const long externalBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  const bool wantCpuTiming = vkBackendFrameTimingEnabled();
  double setupMs = 0.0, geomMs = 0.0, texMs = 0.0, recordMs = 0.0;
  const double cpuT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->beginFrame();
  this->updateLightingSetup(drawlist);
  const double cpuT1 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  const long geometryBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  this->updateGeometryCache(drawlist, false, params.geometryContentUnchanged);
  vkBackendRenderBreadcrumbSince(geometryBcStart, 5000, "renderExternal updateGeometryCache end");
  const double cpuT2 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  const long textureBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  if (!this->flushPendingTextureUploadsExternal()) {
    this->emitError("renderExternal: texture upload failed");
    return FALSE;
  }
  vkBackendRenderBreadcrumbSince(textureBcStart, 5000, "renderExternal flushPendingTextureUploadsExternal end");

  const double cpuT3 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  const long recordBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  this->recordContext.buffer = commandBuffer;
  const bool recorded = this->recordFrame(drawlist, params, *target, renderPass,
                                          this->recordContext);
  vkBackendRenderBreadcrumbSince(recordBcStart, 5000, "renderExternal recordFrame end");
  this->recordContext.buffer = VK_NULL_HANDLE;
  if (wantCpuTiming) {
    setupMs = cpuT1 - cpuT0;
    geomMs = cpuT2 - cpuT1;
    texMs = cpuT3 - cpuT2;
    recordMs = vkBackendRenderNowMs() - cpuT3;
    std::fprintf(stderr,
                 "[RTDBG] cpuTimingRaster mode=full setup=%.2f geom=%.2f "
                 "tex=%.2f record=%.2f\n",
                 setupMs, geomMs, texMs, recordMs);
    std::fflush(stderr);
  }
  vkBackendRenderBreadcrumbSince(externalBcStart, 5000, "renderExternal end");
  return recorded ? TRUE : FALSE;
}

SbBool
SoVulkanRenderBackend::renderExternalOverlay(const SoDrawList & drawlist,
                                             const SoRenderParams & params,
                                             VkCommandBuffer commandBuffer,
                                             VkRenderPass renderPass)
{
  if (!this->isInitialized()) {
    this->emitError(
      "renderExternalOverlay called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "renderExternalOverlay called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    this->emitError(
      "renderExternalOverlay called without a command buffer and render "
      "pass");
    return FALSE;
  }

  // External passes are caller-supplied LOAD render passes; see renderExternal().
  this->renderPassColorCleared = false;
  this->renderPassDepthCleared = false;

  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BLACK_DEBUG"))
    fprintf(stderr, "[BLACK] renderExternalOverlay ENTER frame=%d cmds=%d\n",
            this->uboFrameIndex, drawlist.getNumCommands());

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  this->cacheFrameMatrices(params);
  const bool wantCpuTiming = vkBackendFrameTimingEnabled();
  double setupMs = 0.0, geomMs = 0.0, texMs = 0.0, recordMs = 0.0;
  const double cpuT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->beginFrame();
  this->updateLightingSetup(drawlist);
  const double cpuT1 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->updateGeometryCache(drawlist, true, params.geometryContentUnchanged);
  const double cpuT2 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;

  // This path never goes through recordFrame(), so reserve the slots it will
  // otherwise the cursor keeps climbing across frames and eventually
  // overflows the lighting UBO.
  if (!this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return FALSE;
  }
  if (!this->flushPendingTextureUploadsExternal()) {
    this->emitError("renderExternalOverlay: texture upload failed");
    return FALSE;
  }

  const double cpuT3 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->recordContext.buffer = commandBuffer;
  this->recordTracedComposite(drawlist, params, *target, renderPass,
                              this->recordContext);
  this->recordOverlayBlock(drawlist, params, *target, renderPass,
                           this->recordContext);
  this->recordContext.buffer = VK_NULL_HANDLE;
  if (wantCpuTiming) {
    setupMs = cpuT1 - cpuT0;
    geomMs = cpuT2 - cpuT1;
    texMs = cpuT3 - cpuT2;
    recordMs = vkBackendRenderNowMs() - cpuT3;
    std::fprintf(stderr,
                 "[RTDBG] cpuTimingRaster mode=overlay setup=%.2f geom=%.2f "
                 "tex=%.2f record=%.2f\n",
                 setupMs, geomMs, texMs, recordMs);
    std::fflush(stderr);
  }
  return TRUE;
}

bool
SoVulkanRenderBackend::buildWorkItems(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      bool wireframeOverlay, bool pointsOverlay,
                                      const float * overlayColor,
                                      std::vector<VulkanWorkItem> & out)
{
  const int overlayFillMode = wireframeOverlay
    ? SoDrawStyleElement::LINES
    : (pointsOverlay ? SoDrawStyleElement::POINTS : -1);
  const std::vector<int> & order = drawlist.getSortedOrder();
  out.clear();

  // Geometry content identity for batching: reuse the cached content hash the
  // geometry cache computed when the buffer was uploaded (a map lookup) instead
  // of re-walking every vertex stream.
  auto contentHashOf = [this](const SoRenderCommand & c) -> uint64_t {
    const auto it = this->commandToCache.find(&c);
    if (it == this->commandToCache.end()) return 0;
    return this->gpuCache[it->second].contentHash;
  };

  uint32_t nextSlot = 0;
  // Opaque then transparent, honoring the draw-list sort order.  Overlay
  // commands are handled by the dedicated overlay block outside the work list.
  for (int passIndex = 0; passIndex < 2; ++passIndex) {
    const bool transparent = passIndex == 1;
    if (transparent) {
      // Transparent geometry must preserve painter's order, so never batch.
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index =
          i < static_cast<int>(order.size()) ? order[i] : i;
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass != SO_RENDERPASS_TRANSPARENT) continue;
        if (!command.geometry.positions || command.geometry.vertexCount == 0)
          continue;
        const auto found = this->commandToCache.find(&command);
        if (found == this->commandToCache.end()) continue;
        if (this->gpuCache[found->second].vertexBuffer == VK_NULL_HANDLE)
          continue;
        VulkanWorkItem item;
        item.single = &command;
        item.count = 1;
        item.transparent = true;
        item.slotBase = nextSlot++;
        out.push_back(item);
      }
    }
    else {
      // Only depth-tested opaque geometry is render-order independent, so it can
      // be reordered and batched.  Depth-off commands go to the on-top
      // annotation pass; the CPU wide-line path must stay per-command.  Bucket
      // by a batch key and re-verify pairwise (splitting any collision).
      std::unordered_map<uint64_t, std::vector<const SoRenderCommand*>> & buckets =
        this->batchBucketScratch;
      buckets.clear();
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index =
          i < static_cast<int>(order.size()) ? order[i] : i;
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        if (!command.state.depth.enabled) continue; // on-top annotation (later)
        if (!command.geometry.positions || command.geometry.vertexCount == 0)
          continue;
        if (vkIsWideLine(command)) continue;        // CPU-expanded per command
        const auto found = this->commandToCache.find(&command);
        if (found == this->commandToCache.end()) continue;
        if (this->gpuCache[found->second].vertexBuffer == VK_NULL_HANDLE)
          continue;
        buckets[vkBatchKey(command, contentHashOf(command))].push_back(&command);
      }
      for (auto & kv : buckets) {
        std::vector<const SoRenderCommand*> & v = kv.second;
        int start = 0;
        while (start < static_cast<int>(v.size())) {
          int end = start + 1;
          const uint64_t hStart = contentHashOf(*v[start]);
          while (end < static_cast<int>(v.size()) &&
                 vkCommandBatchable(*v[start], *v[end], hStart,
                                    contentHashOf(*v[end]))) {
            ++end;
          }
          const int cnt = end - start;
          VulkanWorkItem item;
          if (cnt == 1) {
            item.single = v[start];
          }
          else {
            item.commands = &v[start];
          }
          item.count = cnt;
          item.recordToSecondary = true;
          item.slotBase = nextSlot;
          nextSlot += static_cast<uint32_t>(cnt);
          out.push_back(item);
          start = end;
        }
      }
    }

    // Wireframe/point overlay: re-draw opaque geometry in the requested fill
    // mode using a uniform edge color.
    if (!transparent && overlayFillMode >= 0) {
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index =
          i < static_cast<int>(order.size()) ? order[i] : i;
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        if (!command.geometry.positions || command.geometry.vertexCount == 0)
          continue;
        const auto found = this->commandToCache.find(&command);
        if (found == this->commandToCache.end()) continue;
        if (this->gpuCache[found->second].vertexBuffer == VK_NULL_HANDLE)
          continue;
        VulkanWorkItem item;
        item.single = &command;
        item.count = 1;
        item.fillModeOverride = overlayFillMode;
        item.uniformColorOverride = overlayColor;
        item.slotBase = nextSlot++;
        out.push_back(item);
      }
    }
  }

  // On-top annotations: depth-disabled commands drawn after both passes in
  // insertion order.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    if (command.state.depth.enabled) continue;
    if (!command.geometry.positions || command.geometry.vertexCount == 0)
      continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    if (this->gpuCache[found->second].vertexBuffer == VK_NULL_HANDLE) continue;
    VulkanWorkItem item;
    item.single = &command;
    item.count = 1;
    item.slotBase = nextSlot++;
    out.push_back(item);
  }

  return true;
}

bool
SoVulkanRenderBackend::recordFrame(const SoDrawList & drawlist,
                                   const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target,
                                   VkRenderPass renderPass,
                                   VulkanRecordContext & ctx)
{
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_MATRIX_DUMP")) {
    s_debugFrame++;
    s_dumpCmdCount = 0;
  }
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BLACK_DEBUG")) {
    static int blackFrame = 0;
    int nTri = 0, nLine = 0, nOverlay = 0, nTrans = 0, nTriLit = 0;
    int nTriUnlit = 0;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      const SoRenderCommand & c = drawlist.getCommand(i);
      if (c.pass == SO_RENDERPASS_OVERLAY) nOverlay++;
      else if (c.pass == SO_RENDERPASS_TRANSPARENT) nTrans++;
      if (c.geometry.topology == SO_TOPOLOGY_TRIANGLES) {
        nTri++;
        if (c.material.shadingModel == SO_SHADING_LEGACY_GOURAUD) nTriLit++;
        else nTriUnlit++;
      }
      if (c.geometry.topology == SO_TOPOLOGY_LINES ||
          c.geometry.topology == SO_TOPOLOGY_LINE_STRIP) {
        nLine++;
      }
    }
    fprintf(stderr,
            "[BLACK] recordFrame frame=%d flags=0x%x clear=(%.2f,%.2f,%.2f,%.2f) "
            "cmds=%d tri=%d(lit=%d unlit=%d) line=%d overlay=%d trans=%d\n",
            blackFrame++, static_cast<unsigned>(params.flags),
            params.clearColor[0], params.clearColor[1], params.clearColor[2],
            params.clearColor[3], drawlist.getNumCommands(), nTri, nTriLit,
            nTriUnlit, nLine, nOverlay, nTrans);
  }
  this->applyViewport(params, target, ctx);
  this->recordClear(params, target, this->renderPassColorCleared,
                    this->renderPassDepthCleared, ctx);
  this->recordBackground(params, target, renderPass, ctx);
  // The background pass overrides the viewport/scissor for its own draw;
  // restore the viewport from params before recording geometry so draws
  // land in the requested region.
  this->applyViewport(params, target, ctx);

  // Vulkan-only display options, configured through setWireframeOverlay()/
  // setPointsOverlay()/setEdgeColor().  Environment variables act as a
  // diagnostic fallback for the command line.
  const bool wireframeOverlay =
    this->wireframeOverlay || COIN_VULKAN_ENV_FLAG("FC_VULKAN_WIREFRAME");
  const bool pointsOverlay =
    this->pointsOverlay || COIN_VULKAN_ENV_FLAG("FC_VULKAN_POINTS");
  float overlayColor[4] = {
    this->edgeColor[0], this->edgeColor[1], this->edgeColor[2],
    this->edgeColor[3]
  };
  // Parse the FC_VULKAN_EDGE_COLOR override (a diagnostic switch) once; it is
  // process-lifetime and this runs on the overlay path, so two getenv() calls
  // per frame is pure overhead.  Only RGB is taken from the hex value; alpha
  // keeps the configured edgeColor's.
  struct EdgeColorOverride { bool present; float rgb[3]; };
  static const EdgeColorOverride edgeOverride = []() {
    EdgeColorOverride o{false, {0.0f, 0.0f, 0.0f}};
    const char * hex = getenv("FC_VULKAN_EDGE_COLOR");
    if (hex) {
      unsigned int value = 0;
      if (sscanf(hex, "%x", &value) == 1) {
        o.present = true;
        o.rgb[0] = ((value >> 16) & 0xff) / 255.0f;
        o.rgb[1] = ((value >> 8) & 0xff) / 255.0f;
        o.rgb[2] = (value & 0xff) / 255.0f;
      }
    }
    return o;
  }();
  if (edgeOverride.present) {
    overlayColor[0] = edgeOverride.rgb[0];
    overlayColor[1] = edgeOverride.rgb[1];
    overlayColor[2] = edgeOverride.rgb[2];
  }
  // No overlay when neither is requested; otherwise re-draw opaque geometry
  // in the requested draw style (SoDrawStyleElement encoding: LINES=1,
  // POINTS=2) using a uniform edge color.
  const int overlayFillMode = wireframeOverlay
    ? SoDrawStyleElement::LINES
    : (pointsOverlay ? SoDrawStyleElement::POINTS : -1);
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG")) {
    static int overlayLog = 0;
    if (overlayLog++ < 3) {
      fprintf(stderr,
              "[OVL] wireframe=%d points=%d fillMode=%d edgeColor=(%.2f,%.2f,%.2f,%.2f)\n",
              wireframeOverlay ? 1 : 0, pointsOverlay ? 1 : 0, overlayFillMode,
              overlayColor[0], overlayColor[1], overlayColor[2], overlayColor[3]);
    }
  }

  // Reserve per-draw lighting slots for the worst case (main pass plus
  // overlay redraws) before recording, so slotIndex can never overflow the
  // ring allocation (VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972).
  if (!this->prepareLightingSlots(countDrawCommands(drawlist,
                                                    overlayFillMode))) {
    return FALSE;
  }

  // Build the read-only worklist (bucketed opaque + batched + transparent +
  // overlay-redraw + annotation items), each with a pre-assigned disjoint
  // slotBase, then record it.  buildWorkItems() mirrors the exact order the
  // previous inline loops used, and the slotBase values match what the
  // per-draw uboCmdIndex++ sequence produced, so recording is identical.
  std::vector<VulkanWorkItem> & workItems = this->workItemsScratch;
  this->buildWorkItems(drawlist, params, wireframeOverlay, pointsOverlay,
                       overlayColor, workItems);

  // Record one item into the current ctx.buffer using its pre-assigned slot
  // base.  The pre-assigned disjoint block means we set uboCmdIndex per item
  // rather than letting each record helper increment from the previous value,
  // so workers can later record disjoint ranges without sharing cursor state.
  auto recordItem = [&](const VulkanWorkItem & item) {
    ctx.uboCmdIndex = item.slotBase;
    if (item.count > 1) {
      if (this->recordCommandBatch(drawlist, item.commands, item.count, target,
                                   params, renderPass, item.transparent,
                                   item.fillModeOverride,
                                   item.uniformColorOverride, ctx)) {
        // Batched into one instanced draw; nothing further to do.
      }
      else {
        // Batch rejected (e.g. a non-batchable command slipped in): fall back
        // to per-command draws over the item's slot range.  Re-assign the base
        // so each falls at its own slot, matching the pre-assigned layout.
        for (int k = 0; k < item.count; ++k) {
          ctx.uboCmdIndex = item.slotBase + static_cast<uint32_t>(k);
          this->recordDrawCommand(drawlist, *item.commands[k], target, params,
                                  renderPass, item.transparent,
                                  item.fillModeOverride, item.uniformColorOverride,
                                  false, ctx);
        }
      }
    }
    else {
      this->recordDrawCommand(drawlist, *item.single, target, params,
                              renderPass, item.transparent,
                              item.fillModeOverride, item.uniformColorOverride,
                              false, ctx);
    }
  };

  // M1c: the render-order-independent opaque pass is recorded into a
  // secondary command buffer that we re-play into the already-begun render
  // pass, so the opaque draws can later (M1d) be recorded in parallel by
  // worker threads.  Only paths with a real framebuffer (own-queue render)
  // use a secondary; otherwise fall back to fully-inline record to keep the
  // exact same output.
  if (this->secondaryCommandBuffers.empty() ||
      this->renderPassFramebuffer == VK_NULL_HANDLE) {
    for (const VulkanWorkItem & item : workItems) {
      recordItem(item);
    }
  }
  else {
    VkCommandBuffer secondary = this->currentSecondaryCommandBuffer();
    VkCommandBuffer primary = this->currentCommandBuffer();
    bool hasSecondaryItems = false;
    for (const VulkanWorkItem & item : workItems) {
      if (item.recordToSecondary) { hasSecondaryItems = true; break; }
    }
    if (!hasSecondaryItems) {
      for (const VulkanWorkItem & item : workItems) {
        recordItem(item);
      }
    }
    else {
      // Record the opaque items into the secondary (render-pass-continue).
      // Begin from a clean dedup cache: secondary buffers inherit NOTHING
      // (not dynamic state, pipeline, or descriptors) from the primary, so a
      // stale lastBound* entry would suppress a needed re-bind.
      vkResetCommandBuffer(secondary, 0);
      ctx.reset();
      VkCommandBufferBeginInfo sbi {};
      sbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      sbi.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
                  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VkCommandBufferInheritanceInfo inh {};
      inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
      inh.renderPass = renderPass;
      inh.subpass = 0;
      inh.framebuffer = this->renderPassFramebuffer;
      sbi.pInheritanceInfo = &inh;
      if (vkBeginCommandBuffer(secondary, &sbi) != VK_SUCCESS) {
        this->emitError("failed to begin secondary command buffer");
        // Fall back to fully-inline recording for this frame.
        ctx.buffer = primary;
        ctx.reset();
        for (const VulkanWorkItem & item : workItems) {
          recordItem(item);
        }
        return TRUE;
      }
      ctx.buffer = secondary;
      for (const VulkanWorkItem & item : workItems) {
        if (item.recordToSecondary) recordItem(item);
      }
      ctx.buffer = primary;
      // The primary's bound state was NOT preserved across the secondary, so
      // reset its dedup cache before continuing inline (else the first inline
      // draw would skip binds the primary no longer has).
      ctx.reset();
      vkEndCommandBuffer(secondary);
      vkCmdExecuteCommands(primary, 1, &secondary);
      // Inline the remaining (painter-order, overlay-redraw, annotation) items.
      for (const VulkanWorkItem & item : workItems) {
        if (!item.recordToSecondary) recordItem(item);
      }
    }
  }

  // Screen-space overlay geometry (navigation cube): drawn after both passes
  // into its own viewport, with the overlay rect's depth cleared first so the
  // overlay self-occludes independently of the main scene.
  this->recordOverlayBlock(drawlist, params, target, renderPass, ctx);

  return true;
}

void
SoVulkanRenderBackend::recordOverlayBlock(const SoDrawList & drawlist,
                                          const SoRenderParams & params,
                                          const SoVulkanRenderTarget & target,
                                          VkRenderPass renderPass,
                                          VulkanRecordContext & ctx)
{
  // Overlays are drawn in recorded (insertion) order, matching GL: the
  // draw-list sorted order is a painter's algorithm built from the main
  // scene's camera-space depth, which is meaningless for screen-space
  // overlay geometry and would shuffle the navigation cube's panels
  // relative to each other and to other overlays.
  int lastClearX = -1, lastClearY = -1, lastClearW = -1, lastClearH = -1;
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass != SO_RENDERPASS_OVERLAY) continue;
    const SoRasterState & raster = command.state.raster;
    if (!raster.scissorEnabled || raster.scissorWidth <= 0 ||
        raster.scissorHeight <= 0) {
      continue;
    }
    if (raster.scissorX != lastClearX || raster.scissorY != lastClearY ||
        raster.scissorWidth != lastClearW ||
        raster.scissorHeight != lastClearH) {
      this->recordOverlayDepthClear(command, target, ctx);
      lastClearX = raster.scissorX;
      lastClearY = raster.scissorY;
      lastClearW = raster.scissorWidth;
      lastClearH = raster.scissorHeight;
    }
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, true, ctx);
  }
}

void
SoVulkanRenderBackend::recordTracedComposite(const SoDrawList & drawlist,
                                             const SoRenderParams & params,
                                             const SoVulkanRenderTarget & target,
                                             VkRenderPass renderPass,
                                             VulkanRecordContext & ctx)
{
  // Ray-tracing compositing residue: the RT backend traces only triangles, so
  // the OPAQUE/TRANSPARENT LINES / POINTS / LINE_STRIP commands (BRep edge
  // lines, point markers, polylines) are drawn here as a raster layer on top
  // of the path-traced image.  The present pass wrote the scene depth, so
  // each fragment is depth tested with the overlay's LESS_OR_EQUAL compare
  // (recordDrawCommand's overlay path): a front face's edge lies at the
  // traced surface's depth and passes, while a hidden back-facing edge is
  // farther and is culled -- matching the raster pipeline's silhouette edge
  // look.  Depth is not cleared here and depth write stays off.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    const SoPrimitiveTopology topo = command.geometry.topology;
    if (topo == SO_TOPOLOGY_TRIANGLES) continue;
    if (topo == SO_TOPOLOGY_TRIANGLE_STRIP) continue;

    [[maybe_unused]] const SoRasterState & raster = command.state.raster;
    // Apply the command's own viewport/scissor if it carries one, else the
    // whole-surface viewport (the default for scene geometry).
    this->applyCommandViewport(command, target, ctx);
    this->applyScissor(command, target, ctx);
    // overlayPass=false: these are scene-geometry edge/point commands, so
    // they must draw with the FRAME camera (params.projMatrix), not the
    // command's own projection matrix.  Passing overlayPass=true made them
    // use a stale per-command proj, displacing them off the traced surface
    // (the offset "phantom box").  They inherit the raster depth compare so
    // the present-pass depth occludes hidden edges.
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, false, ctx);
  }
}
