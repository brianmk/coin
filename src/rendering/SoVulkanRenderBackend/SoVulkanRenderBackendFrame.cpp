// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendFrame.cpp
//
// Frame orchestration and teardown:
//   - shutdown(): release every owned Vulkan object
//   - renderInternal(): render()/renderOverlaysOnly() entry points
//   - renderExternal()/renderExternalOverlay(): record into a caller-owned cb
//   - recordFrame() / recordOverlayBlock() / recordTracedComposite()

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanDebug.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanDebugUtils.h"

#include <vk_mem_alloc.h>

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
#include <atomic>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace CoinVulkanDetail;

namespace {

double vkBackendRenderNowMs()
{
  return SoVulkanShared::steadyNowMs();
}

// Phase timing for the fcprobe harness ([RTDBG] cpuTimingRaster); gated by COIN_VULKAN_FRAME_TIMING.
bool vkBackendFrameTimingEnabled()
{
  static const bool enabled =
    SoVulkanShared::envFlagEnabled("COIN_VULKAN_FRAME_TIMING");
  return enabled;
}

} // namespace

// Two commands batch into ONE instanced draw only if they share every pipeline,
// descriptor-set and push-constant input and differ solely by model matrix.
// `hashA`/`hashB` are content hashes; `pass` equality keeps them in the main pass.
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
  // Compare pipeline/push-determining state FIELD BY FIELD: memcmp is unsafe (SbBool
  // is an int, enum padding is not zeroed).  Position-dependent sort keys
  // (opaqueKey/translucentKey) are omitted so identical geometry elsewhere still batches.
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

// Coarse opaque-batching key: same key MIGHT be batchable (vkCommandBatchable
// re-verifies and splits collisions).  Hashes pipeline/descriptor/push state
// plus the geometry content hash; correctness comes from the pairwise re-check.
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

  // Destroy the timestamp query pool while the VkDevice is alive (VUID-vkDestroyQueryPool-device-parameter).
  this->gpuTimers.shutdown();

  // Queue is drained, so the external pre-pass fence is idle; destroy it now.
  if (this->externalPrepassFence != VK_NULL_HANDLE) {
    vkDestroyFence(this->device, this->externalPrepassFence, this->allocator);
    this->externalPrepassFence = VK_NULL_HANDLE;
  }

  // The queue is idle, so every deferred resource is safe to release now.
  this->flushAllPendingDestroys();

  // Release uploads abandoned by a frame that aborted before flush/finalize; their copies
  // were never recorded, so synchronous destruction is safe (shared staging pool).
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.index < this->textureCache.size()) {
      this->destroyTextureEntry(this->textureCache[upload.index]);
    }
  }
  this->pendingUploads.clear();

  this->invalidateCache();
  this->destroyAllGeometryBlocks();
  // invalidateCache()/destroyAllGeometryBlocks() deferDestroy() their cached command buffers
  // (a frame may still reference one); queue idle -> flush now or leak (VUID-vkDestroyDevice-device-05137).
  this->flushAllPendingDestroys();

  // Persist the driver blob and destroy every cached pipeline + VkPipelineCache.
  this->pipelines.shutdown();

  // The render-pass/framebuffer cache owns the current pass + framebuffer.
  this->renderPasses.destroyAll();
  if (this->subPixelCullPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->subPixelCullPipeline, this->allocator);
    this->subPixelCullPipeline = VK_NULL_HANDLE;
  }
  if (this->subPixelCullModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->subPixelCullModule,
                          this->allocator);
    this->subPixelCullModule = VK_NULL_HANDLE;
  }
  if (this->subPixelPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->subPixelPipelineLayout,
                            this->allocator);
    this->subPixelPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->subPixelSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->subPixelSetLayout,
                                 this->allocator);
    this->subPixelSetLayout = VK_NULL_HANDLE;
  }
  for (VkDescriptorPool pool : this->subPixelDescriptorPools) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(this->device, pool, this->allocator);
    }
  }
  this->subPixelDescriptorPools.clear();
  this->subPixelDescriptorSetCount = 0;
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
  if (this->wideLineInstancedVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->wideLineInstancedVertexModule,
                          this->allocator);
    this->wideLineInstancedVertexModule = VK_NULL_HANDLE;
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
  if (this->instanceModelBuffer != VK_NULL_HANDLE) {
    // vmaDestroyBuffer releases buffer, memory and host mapping (no explicit vkUnmapMemory).
    vmaDestroyBuffer(this->vmaAllocator, this->instanceModelBuffer,
                     this->instanceModelMemory);
    this->instanceModelBuffer = VK_NULL_HANDLE;
    this->instanceModelMemory = nullptr;
  }
  this->instanceModelMapped = nullptr;
  this->instanceModelCapacity = 0;
  if (this->lightingBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(this->vmaAllocator, this->lightingBuffer,
                     this->lightingMemory);
    this->lightingBuffer = VK_NULL_HANDLE;
    this->lightingMemory = nullptr;
  }
  this->lightingMapped = nullptr;
  if (this->lightingConstBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(this->vmaAllocator, this->lightingConstBuffer,
                     this->lightingConstMemory);
    this->lightingConstBuffer = VK_NULL_HANDLE;
    this->lightingConstMemory = nullptr;
  }
  this->lightingConstMapped = nullptr;
  this->lightingDescriptorSet = VK_NULL_HANDLE;
  if (this->stagingPoolBuffer != VK_NULL_HANDLE) {
    // vmaDestroyBuffer releases buffer, memory and persistent host mapping together.
    vmaDestroyBuffer(this->vmaAllocator, this->stagingPoolBuffer,
                     this->stagingPoolAllocation);
    this->stagingPoolBuffer = VK_NULL_HANDLE;
    this->stagingPoolAllocation = nullptr;
    this->stagingPoolMapped = nullptr;
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
    vmaDestroyImage(this->vmaAllocator, this->whiteImage,
                    this->whiteImageAllocation);
    this->whiteImage = VK_NULL_HANDLE;
    this->whiteImageAllocation = nullptr;
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
  // Join M1d record workers before freeing their secondary buffers.
  this->shutdownRecordPool();
  this->releaseFrameResources();
  if (this->commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->commandPool, this->allocator);
    this->commandPool = VK_NULL_HANDLE;
  }
  for (VkCommandPool pool : this->secondaryCommandPools) {
    if (pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(this->device, pool, this->allocator);
    }
  }
  this->secondaryCommandPools.clear();
  if (this->vmaAllocator != nullptr) {
    // Queue idle and deferred destroys flushed, so all allocations are free.
    vmaDestroyAllocator(this->vmaAllocator);
    this->vmaAllocator = nullptr;
  }

  this->instance = VK_NULL_HANDLE;
  this->physicalDevice = VK_NULL_HANDLE;
  this->device = VK_NULL_HANDLE;
  this->queue = VK_NULL_HANDLE;
  this->allocator = nullptr;

  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

const SoVulkanRenderTarget *
SoVulkanRenderBackend::validateRenderTarget(const SoRenderParams & params) const
{
  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target == nullptr || target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return nullptr;
  }
  return target;
}

const SoVulkanRenderTarget *
SoVulkanRenderBackend::prepareExternalFrame(
    const SoDrawList & drawlist, const SoRenderParams & params,
    VkCommandBuffer commandBuffer, VkRenderPass renderPass,
    const char * caller, const bool overlaysOnly,
    const bool reserveCompositeSlots, ExternalFrameTiming * timing)
{
  if (!this->isInitialized()) {
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s called before backend initialization",
                  caller);
    this->emitError(msg);
    return nullptr;
  }
  if (!params.renderTarget) {
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "%s called without a SoVulkanRenderTarget in "
                  "SoRenderParams::renderTarget", caller);
    this->emitError(msg);
    return nullptr;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "%s called without a command buffer and render pass",
                  caller);
    this->emitError(msg);
    return nullptr;
  }
  const SoVulkanRenderTarget * target = this->validateRenderTarget(params);
  if (target == nullptr) return nullptr;

  // External passes are caller LOAD passes, so recordClear() must vkCmdClearAttachments.
  this->renderPasses.setClearedByLoad(false, false);

  this->cacheFrameMatrices(params);
  const bool wantCpuTiming = timing != nullptr;
  double t0 = wantCpuTiming ? SoVulkanShared::steadyNowMs() : 0.0;
  this->beginFrame();
  this->updateLightingSetup(drawlist);
  if (wantCpuTiming) {
    const double t1 = SoVulkanShared::steadyNowMs();
    timing->setupMs = t1 - t0;
    t0 = t1;
  }
  this->updateGeometryCache(drawlist, overlaysOnly,
                            params.geometryContentUnchanged);
  if (wantCpuTiming) {
    timing->geomMs = SoVulkanShared::steadyNowMs() - t0;
  }
  // The composite path skips recordFrame(), so reserve its overlay/residual lighting slots here.
  if (reserveCompositeSlots &&
      !this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return nullptr;
  }
  // Changed textures are staged in pendingUploads; the caller's beginExternalPrepass() records
  // the copies (or flushPendingTextureUploadsExternal()); no flush here (caller submits pre-pass).
  return target;
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
  vkBackendTrace(this->uboFrameIndex, "renderInternal.enter",
                 "overlaysOnly=%d cmds=%d",
                 static_cast<int>(overlaysOnly), drawlist.getNumCommands());

  const SoVulkanRenderTarget * target = this->validateRenderTarget(params);
  if (target == nullptr) return FALSE;

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

  // One frame boundary: advance the ring cursor, release resources deferred maxFramesInFlight ago.
  this->beginFrame();

  // Write the lighting constant block(s) into the ring once per frame.
  this->updateLightingSetup(drawlist);

  // Render passes are cached by attachment identity (formats, samples, layouts, load ops),
  // not the target images: swapchain images cycle and pipelines key on the pass handle, so
  // reuse keeps the cache warm.  COIN_VULKAN_RP_CLEAR clears via loadOp, cheaper than clear-cmds.
  const bool wantRpClear = COIN_VULKAN_ENV_FLAG("COIN_VULKAN_RP_CLEAR");
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
  this->renderPasses.getOrCreateRenderPass(*target, colorLoadOp,
                                           depthLoadOp);
  // Stash which attachments the pass cleared so recordClear() skips vkCmdClearAttachments.
  this->renderPasses.setClearedByLoad(
    colorLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR,
    depthLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  if (this->renderPasses.currentRenderPass() == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan render pass");
    return FALSE;
  }

  this->updateGeometryCache(drawlist, overlaysOnly,
                            params.geometryContentUnchanged);

  // Composite skips recordFrame(); reserve slots here: OVERLAY + non-triangle residue.
  if (overlaysOnly &&
      !this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return FALSE;
  }

  if (!this->beginCommandBuffer()) {
    this->emitError("failed to begin Vulkan command buffer");
    return FALSE;
  }
  SoVulkanDebugUtils::beginLabel(this->currentCommandBuffer(),
                                 overlaysOnly ? "Coin raster frame (overlays)"
                                              : "Coin raster frame");
  if (SoVulkanConfig::get().diagnostics.gpuTimestamps &&
      !this->gpuTimers.initialized()) {
    this->gpuTimers.initialize(this->device, this->physicalDevice,
                               this->queueFamilyIndex);
  }

  // The framebuffer is cached per target identity (image views + extent + pass); the old one
  // is defer-destroyed (an older in-flight submission may reference it) -- no per-frame
  // vkQueueWaitIdle, beginFrame() waits only the current slot.
  if (!this->renderPasses.ensureFramebuffer(
        target, this->renderPasses.currentRenderPass())) {
    this->emitError("failed to create Vulkan framebuffer");
    // The one-shot cb was begun but not submitted; reset it explicitly or later begins fail.
    vkEndCommandBuffer(this->currentCommandBuffer());
    vkResetCommandBuffer(this->currentCommandBuffer(), 0);
    return FALSE;
  }
  const VkFramebuffer framebuffer = this->renderPasses.framebuffer();

  // Record pending texture copies into the frame command buffer (one submit for
  // the whole frame instead of a separate transfer submit) and finalize host
  // resources.  Sampling draws are recorded below; their descriptor sets must
  // exist.  Staging buffers release through the deferred ring on the slot fence.
  this->gpuTimers.beginScope(this->currentCommandBuffer(), "textureUploads");
  if (!this->recordPendingTextureUploads()) {
    this->emitError("failed to record texture uploads");
  }
  this->finalizePendingTextureUploads();
  this->gpuTimers.endScope(this->currentCommandBuffer());

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = this->renderPasses.currentRenderPass();
  rpbi.framebuffer = framebuffer;
  rpbi.renderArea.offset = {0, 0};
  rpbi.renderArea.extent = target->extent;
  // On the loadOp clear path the clear value goes here (0 = color, 1 = depth).
  VkClearValue clearValues[2];
  uint32_t clearValueCount = 0;
  if (this->renderPasses.colorClearedByLoad()) {
    clearValues[0].color.float32[0] = params.clearColor[0];
    clearValues[0].color.float32[1] = params.clearColor[1];
    clearValues[0].color.float32[2] = params.clearColor[2];
    clearValues[0].color.float32[3] = params.clearColor[3];
    clearValueCount = 1;
  }
  if (this->renderPasses.depthClearedByLoad()) {
    clearValues[clearValueCount].depthStencil.depth = params.clearDepth;
    clearValues[clearValueCount].depthStencil.stencil = 0;
    ++clearValueCount;
  }
  rpbi.clearValueCount = clearValueCount;
  rpbi.pClearValues = clearValueCount ? clearValues : nullptr;

  // INLINE_AND_SECONDARY: the opaque pass replays a secondary (M1c/M1d), so plain
  // INLINE would violate VUID-vkCmdExecuteCommands-contents-09680 and break the
  // secondary's inheritance of viewport/scissor.  The enum needs
  // VK_EXT_nested_command_buffer; otherwise use the fully-inline fallback.
  vkCmdBeginRenderPass(this->currentCommandBuffer(), &rpbi,
                       VK_SUBPASS_CONTENTS_INLINE_AND_SECONDARY_COMMAND_BUFFERS_EXT);
  SoVulkanDebugUtils::beginLabel(this->currentCommandBuffer(),
                                 overlaysOnly ? "overlay pass" : "opaque pass",
                                 0.9f, 0.6f, 0.2f);
  this->gpuTimers.beginScope(this->currentCommandBuffer(), "renderPass");

  this->recordContext.buffer = this->currentCommandBuffer();
  bool recorded = true;
  if (overlaysOnly) {
    this->recordTracedComposite(drawlist, params, *target,
                                this->renderPasses.currentRenderPass(),
                                this->recordContext);
    this->recordOverlayBlock(drawlist, params, *target,
                             this->renderPasses.currentRenderPass(),
                             this->recordContext);
  }
  else {
    recorded = this->recordFrame(drawlist, params, *target,
                                 this->renderPasses.currentRenderPass(),
                                 this->recordContext,
                                 this->renderPasses.framebuffer());
  }
  this->recordContext.buffer = VK_NULL_HANDLE;

  this->gpuTimers.endScope(this->currentCommandBuffer());
  SoVulkanDebugUtils::endLabel(this->currentCommandBuffer());
  vkCmdEndRenderPass(this->currentCommandBuffer());
  SoVulkanDebugUtils::endLabel(this->currentCommandBuffer());

  // Submit even on recordFrame() failure: an unsubmitted one-shot cb cannot be reused.
  const bool submitted = this->endAndSubmit();
  this->gpuTimers.endFrame();
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

void
SoVulkanRenderBackend::setOverlayCompositeMode(SbBool enabled)
{
  this->overlayCompositeMode = enabled != FALSE;
}

void
SoVulkanRenderBackend::resetExternalGpuQueries(VkCommandBuffer commandBuffer)
{
  // Caller records this on its own cb before vkCmdBeginRenderPass (reset is illegal in-pass).
  this->gpuTimers.resetSlot(commandBuffer);
}

SbBool
SoVulkanRenderBackend::renderExternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass,
                                      VkFramebuffer framebuffer)
{
  this->debugValidateDrawList(drawlist);

  // GPU timestamps on the caller-owned pass: the caller must have recorded the
  // query reset before vkCmdBeginRenderPass (resetExternalGpuQueries()), else an
  // in-pass beginScope() cannot reset the pool.  Guard on the reset.
  if (SoVulkanConfig::get().diagnostics.gpuTimestamps &&
      !this->gpuTimers.initialized()) {
    this->gpuTimers.initialize(this->device, this->physicalDevice,
                               this->queueFamilyIndex);
  }
  const bool gpuScopes =
    this->gpuTimers.initialized() && this->gpuTimers.slotReset();

  const bool wantCpuTiming = vkBackendFrameTimingEnabled();
  const double extT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  ExternalFrameTiming timing;
  const SoVulkanRenderTarget * target = this->prepareExternalFrame(
    drawlist, params, commandBuffer, renderPass, "renderExternal",
    /*overlaysOnly*/ false, /*reserveCompositeSlots*/ false,
    wantCpuTiming ? &timing : nullptr);
  if (target == nullptr) return FALSE;
  const double extPrepareEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;

  // M1c/M1d secondary path records with RENDER_PASS_CONTINUE inheritance, needing the
  // framebuffer matching the caller's pass + swapchain image.  The caller owns that
  // pair (e.g. QVulkanWindow's MSAA pass), so it is threaded in, not fabricated.
  if (framebuffer == VK_NULL_HANDLE) {
    this->emitError("renderExternal called without a framebuffer");
    return FALSE;
  }

  // External pre-pass: transfer/compute are illegal in-pass, so pending texture copies
  // + sub-pixel compaction go into one transient command buffer before recordFrame(),
  // submitted after (overlaps CPU recording with the previous GPU frame); non-fatal.
  VkCommandBuffer prepass = this->beginExternalPrepass(
    drawlist, params, /*lod*/ true, wantCpuTiming ? &timing : nullptr);
  if (prepass == VK_NULL_HANDLE && !this->pendingUploads.empty()) {
  // The transient buffer could not carry the copies; fall back to the one-shot upload so
  // textures land.  A failure leaves entries unstamped and the next frame retries.
    const SoVulkan::Result uploadResult =
      this->flushPendingTextureUploadsExternal();
    if (!uploadResult.isOk()) {
      SoDebugError::postWarning("SoVulkanRenderBackend::renderExternal",
                                "one-shot texture upload fallback failed: %s",
                                uploadResult.message().c_str());
    }
  }

  const double extPreRecEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  const double recordT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->recordContext.buffer = commandBuffer;
  if (gpuScopes) {
    this->gpuTimers.beginScope(commandBuffer, "renderPass");
  }
  const bool recorded = this->recordFrame(drawlist, params, *target, renderPass,
                                          this->recordContext, framebuffer);
  if (gpuScopes) {
    this->gpuTimers.endScope(commandBuffer);
  }
  this->recordContext.buffer = VK_NULL_HANDLE;
  const double recordEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;

  // Submit the pre-pass and wait so copies/compacted writes are visible before the caller's pass.
  this->submitExternalPrepass(prepass, wantCpuTiming ? &timing : nullptr);
  const double extSubmitEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;

  // Advance the GPU-timestamp ring and read back the oldest completed frame; the caller
  // submits after we return, so this frame's writes are read a few frames later (no stall).
  this->gpuTimers.endFrame();

  if (wantCpuTiming) {
    const double recordMs = recordEnd - recordT0;
    const double lodMs = timing.lodRecordMs + timing.lodMs;
    const double totalMs = recordMs + timing.texMs + timing.geomMs +
                           timing.setupMs + lodMs;
    SoVulkanDebug::post("[RTDBG] cpuTimingRaster mode=full setup=%.2f geom=%.2f "
                 "tex=%.2f lod=%.2f record=%.2f total=%.2f\n",
                 timing.setupMs, timing.geomMs, timing.texMs, lodMs,
                 recordMs, totalMs);
    SoVulkanDebug::post("[RTDBG] extPhase prepare=%.2f prepassRecord=%.2f "
                 "record=%.2f prepassSubmit=%.2f wall=%.2f\n",
                 extPrepareEnd - extT0, extPreRecEnd - extPrepareEnd,
                 recordEnd - recordT0, extSubmitEnd - extPreRecEnd,
                 extSubmitEnd - extT0);
  }
  return recorded ? TRUE : FALSE;
}

SbBool
SoVulkanRenderBackend::renderExternalOverlay(const SoDrawList & drawlist,
                                             const SoRenderParams & params,
                                             VkCommandBuffer commandBuffer,
                                             VkRenderPass renderPass)
{
  const bool wantCpuTiming = vkBackendFrameTimingEnabled();
  ExternalFrameTiming timing;
  const SoVulkanRenderTarget * target = this->prepareExternalFrame(
    drawlist, params, commandBuffer, renderPass, "renderExternalOverlay",
    /*overlaysOnly*/ true, /*reserveCompositeSlots*/ true,
    wantCpuTiming ? &timing : nullptr);
  if (target == nullptr) return FALSE;

  // The composite path is a raster overlay inside the caller's (RT) pass: no geometry-LOD
  // pre-pass, but pending texture copies route through the transient pre-pass (transfer is
  // illegal in-pass); fall back to the one-shot upload on failure.
  VkCommandBuffer prepass = this->beginExternalPrepass(
    drawlist, params, /*lod*/ false, wantCpuTiming ? &timing : nullptr);
  if (prepass == VK_NULL_HANDLE && !this->pendingUploads.empty()) {
    const SoVulkan::Result uploadResult =
      this->flushPendingTextureUploadsExternal();
    if (!uploadResult.isOk()) {
      SoDebugError::postWarning("SoVulkanRenderBackend::renderExternalOverlay",
                                "one-shot texture upload fallback failed: %s",
                                uploadResult.message().c_str());
    }
  }

  const double recordT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->recordContext.buffer = commandBuffer;
  this->recordTracedComposite(drawlist, params, *target, renderPass,
                              this->recordContext);
  this->recordOverlayBlock(drawlist, params, *target, renderPass,
                           this->recordContext);
  this->recordContext.buffer = VK_NULL_HANDLE;
  const double recordEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->submitExternalPrepass(prepass, wantCpuTiming ? &timing : nullptr);
  if (wantCpuTiming) {
    const double recordMs = recordEnd - recordT0;
    const double totalMs = recordMs + timing.texMs + timing.geomMs +
                           timing.setupMs + timing.lodMs;
    SoVulkanDebug::post("[RTDBG] cpuTimingRaster mode=overlay setup=%.2f geom=%.2f "
                 "tex=%.2f record=%.2f total=%.2f\n",
                 timing.setupMs, timing.geomMs, timing.texMs, recordMs,
                 totalMs);
  }
  return TRUE;
}

bool
SoVulkanRenderBackend::buildWorkItems(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      bool wireframeOverlay, bool pointsOverlay,
                                      bool tessellationOverlay,
                                      const float * overlayColor,
                                      VkRenderPass renderPass,
                                      std::vector<VulkanWorkItem> & out)
{
  const int wireframeFillMode = wireframeOverlay
    ? SoDrawStyleElement::LINES
    : (pointsOverlay ? SoDrawStyleElement::POINTS : -1);
  const std::vector<int> & order = drawlist.getSortedOrder();
  out.clear();

  // Geometry content identity: reuse the hash computed at upload (a map lookup).
  auto contentHashOf = [this](const SoRenderCommand & c) -> uint64_t {
    const auto it = this->commandToCache.find(&c);
    if (it == this->commandToCache.end()) return 0;
    return this->gpuCache[it->second].contentHash;
  };

  uint32_t nextSlot = 0;
  // A replayed sorted order may be stale/shorter; fall back to identity order.
  const auto orderedIndex = [&order, &drawlist](int i) {
    return (i < static_cast<int>(order.size()) &&
            order[i] < drawlist.getNumCommands())
      ? order[i] : i;
  };
  // Opaque then transparent, honoring sort order; overlays handled outside the work list.
  for (int passIndex = 0; passIndex < 2; ++passIndex) {
    const bool transparent = passIndex == 1;
    if (transparent) {
      // Transparent geometry must preserve painter's order, so never batch.
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index = orderedIndex(i);
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass != SO_RENDERPASS_TRANSPARENT) continue;
        if (!this->findCachedDrawable(command)) continue;
        VulkanWorkItem item;
        item.single = &command;
        item.count = 1;
        item.transparent = true;
        item.slotBase = nextSlot++;
        out.push_back(item);
      }
    }
    else {
      // Only depth-tested opaque geometry is order-independent and batchable.
      // Depth-off commands go to on-top annotations; bucket by key, re-verify
      // pairwise (splitting collisions).  The CPU wide-line path stays per-command.
      std::unordered_map<uint64_t, std::vector<const SoRenderCommand*>> & buckets =
        this->batchBucketScratch;
      buckets.clear();
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index = orderedIndex(i);
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        if (!command.state.depth.enabled) continue; // on-top annotation (later)
        if (isWideLine(command, -1, this->interactionLodActive)) {
          // CPU-expanded per command, so never batched, but still routed to a secondary:
          // prepareWideLineBuffers() grew the quad buffer on the recording thread, so
          // workers only fill/bind the existing mapping.  Expansion dominates per-frame
          // CPU on edge-heavy scenes; routing it through the workers parallelizes it.
          if (!this->findCachedDrawable(command)) continue;
          VulkanWorkItem item;
          item.single = &command;
          item.count = 1;
          item.recordToSecondary = true;
          item.slotBase = nextSlot++;
          out.push_back(item);
          continue;
        }
        const VulkanCachedCommand * cached = this->findCachedDrawable(command);
        if (!cached) continue;
        buckets[vkBatchKey(command, cached->contentHash)].push_back(&command);
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

    // Wireframe/point overlay: re-draw opaque geometry in the requested fill mode with
    // a uniform edge color.  A LINES (edge) overlay re-draws only real B-Rep feature
    // edges (SoBrepEdgeSet emits SO_TOPOLOGY_LINES/LINE_STRIP), not every triangle in
    // polygon-LINES -- that would show the raw tessellation.  Tess overlay does the opposite.
    if (!transparent && (wireframeFillMode >= 0 || tessellationOverlay)) {
      const bool isEdgeOverlay = (wireframeFillMode == SoDrawStyleElement::LINES);
      // Bucket by the same geometry/material key as the opaque pass, then batch
      // consecutive batchable commands into one instanced draw.  The overlay re-draws
      // share one uniform color + fill mode, so a bucket of N identical edges collapses
      // to a single vkCmdDraw(N instances) instead of N individual line draws.
      // Separate map: opaque items hold pointers into batchBucketScratch's vectors, so
      // this must not be the same storage (clearing it would dangle those pointers).
      std::unordered_map<uint64_t, std::vector<const SoRenderCommand*>> & obuckets =
        this->overlayBatchBucketScratch;
      obuckets.clear();
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index = orderedIndex(i);
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        if (!command.geometry.positions || command.geometry.vertexCount == 0)
          continue;
        const SoPrimitiveTopology topo = command.geometry.topology;
        const VulkanCachedCommand * cached = this->findCachedDrawable(command);
        if (!cached) continue;
        const bool lineTopo = topo == SO_TOPOLOGY_LINES ||
          topo == SO_TOPOLOGY_LINE_STRIP;
        const bool triTopo = topo == SO_TOPOLOGY_TRIANGLES ||
          topo == SO_TOPOLOGY_TRIANGLE_STRIP;
        // Enabled overlays form a union (edge = feature-edge lines, points = all as
        // points, tess = triangles), filtered independently so two enabled overlays
        // re-draw both sets instead of cancelling out.  Edge re-draws only real B-Rep
        // feature edges; Draft grid/dimension annotations keep their own colors.
        const bool wantEdge = isEdgeOverlay && lineTopo
          && command.isFeatureEdge;
        const bool wantPoints = wireframeFillMode == SoDrawStyleElement::POINTS;
        const bool wantTess = tessellationOverlay && triTopo;
        if (!wantEdge && !wantPoints && !wantTess) continue;
        obuckets[vkBatchKey(command, cached->contentHash)].push_back(&command);
      }
      for (auto & kv : obuckets) {
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
          // Tess re-draws triangles in polygon-LINES; edge/points keep their fill mode.
          // Uniform within a bucket (depends only on topology, which is part of the key).
          const SoPrimitiveTopology t0 = (*v[start]).geometry.topology;
          const bool tess0 = tessellationOverlay &&
            (t0 == SO_TOPOLOGY_TRIANGLES || t0 == SO_TOPOLOGY_TRIANGLE_STRIP);
          VulkanWorkItem item;
          if (cnt == 1) {
            item.single = v[start];
          }
          else {
            item.commands = &v[start];
          }
          item.count = cnt;
          item.fillModeOverride = tess0
            ? SoDrawStyleElement::LINES
            : wireframeFillMode;
          item.uniformColorOverride = overlayColor;
          item.slotBase = nextSlot;
          nextSlot += static_cast<uint32_t>(cnt);
          out.push_back(item);
          start = end;
        }
      }
    }
  }

  // On-top annotations: depth-disabled commands, after both passes, in insertion order.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    if (command.state.depth.enabled) continue;
    if (!this->findCachedDrawable(command)) continue;
    VulkanWorkItem item;
    item.single = &command;
    item.count = 1;
    item.slotBase = nextSlot++;
    out.push_back(item);
  }

  // M1d: pre-resolve each recordToSecondary item's pipeline here (single-threaded):
  // getOrCreatePipeline() mutates the shared store/gpuCache on its cold path, so
  // warming keys first leaves the parallel recorders on the read-only fast path.
  if (this->parallelRecordEnabled) {
    const auto * tgt = static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
    if (tgt) {
      for (const VulkanWorkItem & item : out) {
        if (!item.recordToSecondary) continue;
        const SoRenderCommand * const cmd =
          item.count > 1 ? item.commands[0] : item.single;
        // Pass the cache entry so the warmed key matches the record path's key.
        VulkanCachedCommand * entry = nullptr;
        const auto found = this->commandToCache.find(cmd);
        if (found != this->commandToCache.end()) {
          entry = &this->gpuCache[found->second];
        }
        VkPipeline warmed = VK_NULL_HANDLE;
        this->getOrCreatePipeline(*cmd, *tgt, renderPass, warmed,
                                  false, -1, false, entry);
      }
    }
  }

  return true;
}

void
SoVulkanRenderBackend::recordWorkItem(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      const SoVulkanRenderTarget & target,
                                      VkRenderPass renderPass,
                                      const VulkanWorkItem & item,
                                      VulkanRecordContext & ctx)
{
  // Plant the pre-assigned disjoint slot block, then record one draw/instanced batch.
  ctx.uboCmdIndex = item.slotBase;
  vkBackendTrace(this->uboFrameIndex, "recordWorkItem.enter",
                 "count=%d slotBase=%u kind=%s",
                 static_cast<int>(item.count), item.slotBase,
                 item.count > 1 ? "batch" : "single");
  if (item.count > 1) {
    if (this->recordCommandBatch(drawlist, item.commands, item.count, target,
                                 params, renderPass, item.transparent,
                                 item.fillModeOverride, item.uniformColorOverride,
                                 ctx)) {
      return;
    }
    // Batch rejected (non-batchable command slipped in): fall back to per-command draws.
    for (int k = 0; k < item.count; ++k) {
      ctx.uboCmdIndex = item.slotBase + static_cast<uint32_t>(k);
      this->recordDrawCommand(drawlist, *item.commands[k], target, params,
                              renderPass, item.transparent,
                              item.fillModeOverride, item.uniformColorOverride,
                              false, ctx);
    }
    return;
  }
  this->recordDrawCommand(drawlist, *item.single, target, params, renderPass,
                          item.transparent, item.fillModeOverride,
                          item.uniformColorOverride, false, ctx);
}

bool
SoVulkanRenderBackend::recordSecondaryChunk(VulkanRecordContext & ctx,
                                            const SoDrawList & drawlist,
                                            const SoRenderParams & params,
                                            const SoVulkanRenderTarget & target,
                                            VkRenderPass renderPass,
                                            const std::vector<const VulkanWorkItem *> & items,
                                            VkCommandBuffer secondary,
                                            VkFramebuffer framebuffer)
{
  if (secondary == VK_NULL_HANDLE || items.empty()) return true;
  vkBackendTrace(this->uboFrameIndex, "recordSecondaryChunk.enter",
                 "secondary=%p items=%zu",
                 reinterpret_cast<const void *>(secondary), items.size());
  if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
    SoVulkanDebug::post("[SEC] begin chunk items=%zu secondary=%p frame=%u\n",
            items.size(), (const void*)secondary, this->uboFrameIndex);
  }
  vkResetCommandBuffer(secondary, 0);
  vkBackendTrace(this->uboFrameIndex, "recordSecondaryChunk.resetDone",
                 "secondary=%p",
                 reinterpret_cast<const void *>(secondary));
  // Clean dedup cache: secondaries inherit no dynamic state, pipeline or descriptors.
  ctx.reset();
  VkCommandBufferBeginInfo sbi {};
  sbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  sbi.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
              VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkCommandBufferInheritanceInfo inh {};
  inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
  inh.renderPass = renderPass;
  inh.subpass = 0;
  inh.framebuffer = framebuffer;
  sbi.pInheritanceInfo = &inh;
  if (vkBeginCommandBuffer(secondary, &sbi) != VK_SUCCESS) {
    vkBackendTrace(this->uboFrameIndex, "recordSecondaryChunk.beginFail",
                   "secondary=%p",
                   reinterpret_cast<const void *>(secondary));
    return false;
  }
  vkBackendTrace(this->uboFrameIndex, "recordSecondaryChunk.beginOk",
                 "secondary=%p", reinterpret_cast<const void *>(secondary));
  ctx.buffer = secondary;
  for (const VulkanWorkItem * item : items) {
    this->recordWorkItem(drawlist, params, target, renderPass, *item, ctx);
  }
  ctx.buffer = VK_NULL_HANDLE;
  vkEndCommandBuffer(secondary);
  vkBackendTrace(this->uboFrameIndex, "recordSecondaryChunk.endOk",
                 "secondary=%p", reinterpret_cast<const void *>(secondary));
  return true;
}

bool
SoVulkanRenderBackend::recordFrame(const SoDrawList & drawlist,
                                   const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target,
                                   VkRenderPass renderPass,
                                   VulkanRecordContext & ctx,
                                   VkFramebuffer inheritFramebuffer)
{
  vkBackendTrace(this->uboFrameIndex, "recordFrame.enter",
                 "cmds=%d", drawlist.getNumCommands());
  // Latch interaction-LOD before any isWideLine() decision so expansion, pipeline key and draw path agree.
  this->interactionLodActive = params.interactionLod == TRUE;
  // Wide-line CPU expansion: grow per-command quad buffers (device-memory allocation
  // is not thread-safe) and compute quads across the worker pool before recording;
  // expansion is the CPU-heavy, parallel part, recording stays single-threaded.
  this->prepareWideLineBuffers(drawlist);
  this->expandWideLinesParallel(drawlist, params);
  this->applyViewport(params, target, ctx);
  this->recordClear(params, target, this->renderPasses.colorClearedByLoad(),
                    this->renderPasses.depthClearedByLoad(), ctx);
  this->recordBackground(params, target, renderPass, ctx);
  // The background pass overrides viewport/scissor; restore from params before geometry.
  this->applyViewport(params, target, ctx);

  // Vulkan-only display options (setWireframeOverlay()/setPointsOverlay()/setTessellationOverlay()/setEdgeColor()).
  const bool wireframeOverlay =
    this->wireframeOverlay || COIN_VULKAN_ENV_FLAG("COIN_VULKAN_WIREFRAME");
  const bool pointsOverlay =
    this->pointsOverlay || COIN_VULKAN_ENV_FLAG("COIN_VULKAN_POINTS");
  const bool tessellationOverlay =
    this->tessellationOverlay || COIN_VULKAN_ENV_FLAG("COIN_VULKAN_TESS");
  float overlayColor[4] = {
    this->edgeColor[0], this->edgeColor[1], this->edgeColor[2],
    this->edgeColor[3]
  };
  // Parse the COIN_VULKAN_EDGE_COLOR diagnostic override once; RGB from hex, alpha kept from edgeColor.
  struct EdgeColorOverride { bool present; float rgb[3]; };
  static const EdgeColorOverride edgeOverride = []() {
    EdgeColorOverride o{false, {0.0f, 0.0f, 0.0f}};
    const char * hex = SoVulkanShared::envString("COIN_VULKAN_EDGE_COLOR");
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
  // Re-draw opaque geometry in the requested draw style (LINES=1, POINTS=2) with a uniform edge color.
  const int wireframeFillMode = wireframeOverlay
    ? SoDrawStyleElement::LINES
    : (pointsOverlay ? SoDrawStyleElement::POINTS : -1);
  if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
    static int overlayLog = 0;
    if (overlayLog++ < 3) {
      SoVulkanDebug::post("[OVL] wireframe=%d points=%d tess=%d fillMode=%d edgeColor=(%.2f,%.2f,%.2f,%.2f)\n",
              wireframeOverlay ? 1 : 0, pointsOverlay ? 1 : 0,
              tessellationOverlay ? 1 : 0, wireframeFillMode,
              overlayColor[0], overlayColor[1], overlayColor[2], overlayColor[3]);
    }
  }

  // Reserve per-draw lighting slots for the worst case (main pass + overlay
  // redraws) before recording so slotIndex never overflows the ring allocation
  // (VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972).
  if (!this->prepareLightingSlots(countDrawCommands(drawlist,
                                                    wireframeFillMode,
                                                    tessellationOverlay))) {
    return FALSE;
  }

  // Build the read-only worklist (bucketed/batched opaque + transparent + overlay
  // redraw + annotation items), each with a pre-assigned disjoint slotBase, then
  // record it.  buildWorkItems() mirrors the old inline order, so recording is identical.
  std::vector<VulkanWorkItem> & workItems = this->workItemsScratch;
  this->buildWorkItems(drawlist, params, wireframeOverlay, pointsOverlay,
                       tessellationOverlay, overlayColor, renderPass,
                       workItems);
  vkBackendTrace(this->uboFrameIndex, "recordFrame.workItemsBuilt",
                 "items=%zu", workItems.size());

  // Secondaries record with RENDER_PASS_CONTINUE inheritance into the frame's pass.
  // INTERNAL is backend-owned; EXTERNAL (FreeCAD/QuarterVulkanWidget) is caller-owned
  // and corrupts NVIDIA driver state, so it stays OFF unless explicitly opted in.
  const bool externalPass = renderPass != this->renderPasses.currentRenderPass();
  const bool canUseSecondary =
    !this->secondaryCommandBuffers.empty() &&
    inheritFramebuffer != VK_NULL_HANDLE &&
    (!externalPass || SoVulkanConfig::get().concurrency.externalSecondary);
  // Count the render-order-independent opaque items that live in a secondary.
  uint64_t secondaryItemCount = 0;
  for (const VulkanWorkItem & item : workItems) {
    if (item.recordToSecondary) ++secondaryItemCount;
  }
  const bool wantParallel =
    canUseSecondary && this->parallelRecordEnabled &&
    secondaryItemCount >= 64 && this->maxRecordWorkers > 1;
  vkBackendTrace(this->uboFrameIndex, "recordFrame.mode",
                 "secondary=%u canSec=%d parEnabled=%d W=%u wantPar=%d",
                 static_cast<unsigned>(secondaryItemCount),
                 canUseSecondary ? 1 : 0, this->parallelRecordEnabled ? 1 : 0,
                 this->maxRecordWorkers, wantParallel ? 1 : 0);

  if (canUseSecondary && secondaryItemCount > 0 && !wantParallel) {
    // M1c serial: one secondary holds the whole opaque pass, replayed in place.
    VkCommandBuffer secondary = this->currentSecondaryCommandBuffer();
    // The primary is the record context's buffer: the caller's cb on the external
    // (FreeCAD) path, the backend's own cb internally. currentCommandBuffer() is
    // wrong on the external path (it returns the backend's one-shot cb, which is
    // not in the caller's render pass), so replay/overlay must target ctx.buffer.
    // Captured before recordSecondaryChunk() re-points ctx.buffer at the secondary.
    VkCommandBuffer primary = ctx.buffer;
    std::vector<const VulkanWorkItem *> & opaqueItems = this->opaqueItemsScratch;
    opaqueItems.clear();
    opaqueItems.reserve(static_cast<size_t>(secondaryItemCount));
    for (const VulkanWorkItem & item : workItems) {
      if (item.recordToSecondary) opaqueItems.push_back(&item);
    }
    if (!this->recordSecondaryChunk(ctx, drawlist, params, target, renderPass,
                                    opaqueItems, secondary,
                                    inheritFramebuffer)) {
      this->emitError("failed to begin secondary command buffer");
      this->recordContext.buffer = primary;
      this->recordContext.reset();
      for (const VulkanWorkItem & item : workItems) {
        this->recordWorkItem(drawlist, params, target, renderPass, item,
                             this->recordContext);
      }
      return TRUE;
    }
    // The primary's bound state is not preserved across the secondary; reset its cache.
    ctx.buffer = primary;
    ctx.reset();
    vkCmdExecuteCommands(primary, 1, &secondary);
    for (const VulkanWorkItem & item : workItems) {
      if (!item.recordToSecondary) {
        this->recordWorkItem(drawlist, params, target, renderPass, item, ctx);
      }
    }
  }
  else if (wantParallel) {
    // M1d parallel: partition opaque items into disjoint chunks (greedy
    // longest-first for load balance), record each into its own worker secondary,
    // replay all in order, then inline the painter-order / overlay / annotation items.
    if (COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG")) {
      static int parLog = 0;
      if (parLog++ < 3) {
        SoVulkanDebug::post("[PAR] parallel record: %u opaque items across %u workers\n",
                static_cast<unsigned>(secondaryItemCount),
                static_cast<unsigned>(this->maxRecordWorkers));
      }
    }
    const uint32_t W = this->maxRecordWorkers;
    for (uint32_t w = 0; w < W; ++w) {
      this->recordJobs[w].expandWideLines = false;
      this->recordJobs[w].wideLineCommands.clear();
      this->recordJobs[w].items.clear();
      this->recordJobs[w].drawlist = &drawlist;
      this->recordJobs[w].params = &params;
      this->recordJobs[w].target = &target;
      this->recordJobs[w].renderPass = renderPass;
      this->recordJobs[w].framebuffer = inheritFramebuffer;
      this->recordJobs[w].secondary = this->parallelCurrentSecondary(w);
      this->recordJobs[w].ctx = &this->workerRecordContexts[w];
      this->recordJobs[w].ok = false;
    }
    // Greedy longest-first: costs = first command's vertex count * item.count.
    std::vector<std::pair<uint64_t, const VulkanWorkItem *>> & heaviest =
      this->heaviestScratch;
    heaviest.clear();
    heaviest.reserve(static_cast<size_t>(secondaryItemCount));
    for (const VulkanWorkItem & item : workItems) {
      if (!item.recordToSecondary) continue;
      const SoRenderCommand * first =
        item.count > 1 ? item.commands[0] : item.single;
      uint64_t cost = static_cast<uint64_t>(first ? first->geometry.vertexCount : 0);
      if (first && first->geometry.indexCount) cost *= first->geometry.indexCount;
      cost *= static_cast<uint64_t>(item.count);
      heaviest.emplace_back(cost, &item);
    }
    std::sort(heaviest.begin(), heaviest.end(),
              [](const std::pair<uint64_t, const VulkanWorkItem *> & a,
                 const std::pair<uint64_t, const VulkanWorkItem *> & b) {
                return a.first > b.first;
              });
    std::vector<uint64_t> & load = this->loadScratch;
    load.assign(W, 0);
    for (const auto & h : heaviest) {
      uint32_t dst = 0;
      for (uint32_t w = 1; w < W; ++w) { if (load[w] < load[dst]) dst = w; }
      this->recordJobs[dst].items.push_back(h.second);
      load[dst] += h.first;
    }
    // Dispatch: main records worker 0, spawned threads record workers 1..W-1.
    uint32_t generation = 0;
    {
      std::lock_guard<std::mutex> lk(this->recordMutex);
      this->recordDoneCount.store(0);
      generation = ++this->recordJobGeneration;
    }
    vkBackendTrace(this->uboFrameIndex, "recordFrame.dispatch",
                   "gen=%u W=%u items=%u", generation, W,
                   static_cast<unsigned>(secondaryItemCount));
    this->recordCvSpawn.notify_all();
    this->recordJobs[0].ok = this->recordSecondaryChunk(
      this->workerRecordContexts[0], drawlist, params, target, renderPass,
      this->recordJobs[0].items, this->recordJobs[0].secondary,
      this->recordJobs[0].framebuffer);
    {
      std::unique_lock<std::mutex> lk(this->recordMutex);
      this->recordCvDone.wait(lk, [this] {
        return this->recordDoneCount.load() >= this->maxRecordWorkers - 1;
      });
    }
    vkBackendTrace(this->uboFrameIndex, "recordFrame.joined",
                   "doneCount=%d ok0=%d",
                   this->recordDoneCount.load(), this->recordJobs[0].ok ? 1 : 0);
    // Replay secondaries in order, then inline non-opaque items; a failed worker records inline.
    // Primary is the record context's buffer (caller's cb externally, backend's cb
    // internally). Workers recorded into their own workerRecordContexts, so ctx.buffer
    // is still the correct primary here.
    VkCommandBuffer primary = ctx.buffer;
    ctx.buffer = primary;
    ctx.reset();
    std::vector<VkCommandBuffer> & execute = this->executeScratch;
    execute.clear();
    execute.reserve(W);
    for (uint32_t w = 0; w < W; ++w) {
      if (this->recordJobs[w].items.empty()) continue;
      if (this->recordJobs[w].ok) {
        execute.push_back(this->recordJobs[w].secondary);
      }
      else {
        this->emitError("parallel record worker failed; recorded inline");
        for (const VulkanWorkItem * item : this->recordJobs[w].items) {
          this->recordWorkItem(drawlist, params, target, renderPass, *item,
                               ctx);
        }
      }
    }
    if (!execute.empty()) {
      vkCmdExecuteCommands(primary, static_cast<uint32_t>(execute.size()),
                           execute.data());
    }
    for (const VulkanWorkItem & item : workItems) {
      if (!item.recordToSecondary) {
        this->recordWorkItem(drawlist, params, target, renderPass, item, ctx);
      }
    }
  }
  else {
    // Fully-inline record (no secondary available or nothing to re-play).
    for (const VulkanWorkItem & item : workItems) {
      this->recordWorkItem(drawlist, params, target, renderPass, item, ctx);
    }
  }

  // Screen-space overlay geometry (navigation cube) into its own viewport, depth cleared first.
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
  // Overlays draw in recorded (insertion) order, matching GL: the draw-list sorted
  // order is a painter's algorithm from main-scene camera-space depth, meaningless
  // for screen-space overlays (it would shuffle the navigation cube's panels).
  int lastClearX = -1, lastClearY = -1, lastClearW = -1, lastClearH = -1;
  const SbVec2s frameSize = params.viewport.getViewportSizePixels();
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass != SO_RENDERPASS_OVERLAY) continue;
    const SoRasterState & raster = command.state.raster;
    if (!raster.scissorEnabled || raster.scissorWidth <= 0 ||
        raster.scissorHeight <= 0) {
      continue;
    }
    // The selection/preselection highlight is a full-frame overlay that must depth-test
    // against scene depth (NOT be forced on top), so its depth is not cleared.  Only
    // sub-viewport widgets (navigation cube, axis cross) clear depth.
    const bool fullFrameOverlay =
      raster.scissorWidth == frameSize[0] && raster.scissorHeight == frameSize[1];
    if (raster.scissorX != lastClearX || raster.scissorY != lastClearY ||
        raster.scissorWidth != lastClearW ||
        raster.scissorHeight != lastClearH) {
      if (!fullFrameOverlay) {
        this->recordOverlayDepthClear(command, target, ctx);
      }
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
  // Same parallel wide-line expansion as recordFrame(): the composite pass draws the residue.
  this->prepareWideLineBuffers(drawlist);
  this->expandWideLinesParallel(drawlist, params);

  // Composite residue: an external renderer may draw only triangles, so
  // LINES/POINTS/LINE_STRIP commands (BRep edges, point markers, polylines) are drawn
  // as a raster layer over that image, depth-tested (LESS_OR_EQUAL) against the
  // present pass's scene depth so hidden back edges are culled.  No depth clear/write.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    const SoPrimitiveTopology topo = command.geometry.topology;
    if (topo == SO_TOPOLOGY_TRIANGLES) continue;
    if (topo == SO_TOPOLOGY_TRIANGLE_STRIP) continue;

    [[maybe_unused]] const SoRasterState & raster = command.state.raster;
    // Apply the command's own viewport/scissor if set, else the whole-surface default.
    this->applyCommandViewport(command, target, ctx);
    this->applyScissor(command, target, ctx);
    // overlayPass=false: these are scene-geometry edge/point commands, so draw with the
    // FRAME camera (params.projMatrix), not the command's own proj -- overlayPass=true
    // used a stale proj, displacing them off the traced surface (the "phantom box").
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, false, ctx);
  }
}
