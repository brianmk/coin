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
#include "rendering/SoVulkanConfig.h"
#include "rendering/SoVulkanDebugUtils.h"

#include "vk_mem_alloc.h"

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

long vkBackendRenderNowUs()
{
  return SoVulkanShared::steadyNowUs();
}

double vkBackendRenderNowMs()
{
  return SoVulkanShared::steadyNowMs();
}

// Phase timing for the fcprobe profile harness ([RTDBG] cpuTimingRaster),
// gated by the same FC_VULKAN_FRAME_TIMING flag as the manager and RTX
// [RTDBG] lines.  Cached: the environment does not change mid-process.
bool vkBackendFrameTimingEnabled()
{
  static const bool enabled = SoVulkanConfig::get().debug.frameTiming;
  return enabled;
}

bool vkBackendRenderBreadcrumbEnabled()
{
  return SoVulkanShared::breadcrumbsEnabled();
}

int vkBackendRenderBreadcrumbLogBudget = 0;

void vkBackendRenderBreadcrumbSince(long startUs, long thresholdUs, const char* phase)
{
  SoVulkanShared::breadcrumbSince(vkBackendRenderBreadcrumbLogBudget,
                                  "[VKBACKEND]", startUs, thresholdUs, phase);
}

} // namespace

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

  // Destroy the timestamp query pool now, while the VkDevice is still alive;
  // the destructor would otherwise run after device teardown
  // (VUID-vkDestroyQueryPool-device-parameter).
  this->gpuTimers.shutdown();

  // The queue is idle, so every deferred resource is safe to release now.
  this->flushAllPendingDestroys();

  // Uploads abandoned by an aborted frame are just entries in the texture
  // cache; textureCache.destroy() below releases them (and the staging pool)
  // synchronously now that the queue is idle.
  this->invalidateCache();
  this->destroyAllGeometryBlocks();
  // invalidateCache()/destroyAllGeometryBlocks() release their cached command
  // buffers (vertex/index/instanced-line/sub-pixel) through deferDestroy(),
  // because a frame may still have referenced them when they were evicted.
  // The queue is idle here, so flush that batch now; without it those buffers
  // and their device memory leak past vkDestroyDevice
  // (VUID-vkDestroyDevice-device-05137).
  this->flushAllPendingDestroys();

  // Persist the driver's blob and destroy every cached pipeline + the
  // VkPipelineCache handle (see SoVulkanPipelineCache).
  this->pipelines.shutdown();

  // The render-pass/framebuffer cache owns the current pass + framebuffer;
  // releasing it after the deferred destroys flush above (queue is idle).
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
    // vmaDestroyBuffer releases the buffer, its memory and the persistent host
    // mapping together, so no explicit vkUnmapMemory is needed.
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
  // Release the texture cache (entries, staging pool, white fallback) while
  // the shared descriptor pools are still valid; destroyEntry() frees each
  // entry's set back to its pool.
  this->textureCache.destroy();
  // Every sampler (including the white fallback's) is owned by the sampler
  // cache; release them once the texture cache has been emptied.
  this->samplerCache.destroyAll();
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
  // Join the M1d record workers before freeing the secondary buffers they
  // record into.
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
    // Queue is idle and every deferred destroy has been flushed, so all
    // allocations are free and the allocator (and its blocks) can be released.
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

bool
SoVulkanRenderBackend::beginFramePlan(
    const SoDrawList & drawlist, const SoRenderParams & params,
    const char * caller, const bool overlaysOnly, const bool external,
    const bool reserveCompositeSlots, FramePlan & plan,
    ExternalFrameTiming * timing)
{
  if (!this->isInitialized()) {
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s called before backend initialization",
                  caller);
    this->emitError(msg);
    return false;
  }
  if (!params.renderTarget) {
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "%s called without a SoVulkanRenderTarget in "
                  "SoRenderParams::renderTarget", caller);
    this->emitError(msg);
    return false;
  }
  const SoVulkanRenderTarget * target = this->validateRenderTarget(params);
  if (target == nullptr) return false;

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
  // The composite path never goes through recordFrame(), so it must reserve
  // the lighting slots its overlay/residual draws consume here; otherwise the
  // cursor keeps climbing across frames and overflows the ring.  The full path
  // reserves its slots inside recordFrame() instead.
  if (reserveCompositeSlots &&
      !this->prepareLightingSlots(countCompositeCommands(drawlist))) {
    this->emitError("failed to reserve lighting UBO slots");
    return false;
  }

  plan.target = target;
  plan.overlaysOnly = overlaysOnly;
  plan.external = external;
  return true;
}

bool
SoVulkanRenderBackend::recordFramePlan(const SoDrawList & drawlist,
                                       const SoRenderParams & params,
                                       const FramePlan & plan,
                                       VkRenderPass renderPass,
                                       VkFramebuffer framebuffer,
                                       VulkanRecordContext & ctx)
{
  if (plan.overlaysOnly) {
    this->recordTracedComposite(drawlist, params, *plan.target, renderPass, ctx);
    this->recordOverlayBlock(drawlist, params, *plan.target, renderPass, ctx);
    return true;
  }
  return this->recordFrame(drawlist, params, *plan.target, renderPass, ctx,
                           framebuffer);
}

bool
SoVulkanRenderBackend::prepareExternalFrame(
    const SoDrawList & drawlist, const SoRenderParams & params,
    VkCommandBuffer commandBuffer, VkRenderPass renderPass,
    const char * caller, const bool overlaysOnly,
    const bool reserveCompositeSlots, FramePlan & plan,
    ExternalFrameTiming * timing)
{
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    char msg[160];
    std::snprintf(msg, sizeof(msg),
                  "%s called without a command buffer and render pass",
                  caller);
    this->emitError(msg);
    return false;
  }
  if (!this->beginFramePlan(drawlist, params, caller, overlaysOnly,
                            /*external*/ true, reserveCompositeSlots, plan,
                            timing)) {
    return false;
  }

  // External passes are caller-supplied LOAD render passes, so no attachment
  // is cleared by a loadOp here: recordClear() must emit vkCmdClearAttachments.
  this->renderPasses.setClearedByLoad(false, false);
  // Changed textures are now staged in the texture cache; the caller's
  // beginExternalPrepass() records the copies into its transient command
  // buffer (or falls back to SoVulkanTextureCache::flushExternal() when that
  // buffer cannot be allocated).  No flush here: it would need its own
  // submission and queue drain, and the caller already submits the pre-pass.
  return true;
}

SbBool
SoVulkanRenderBackend::renderInternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      const bool overlaysOnly)
{
  this->debugValidateDrawList(drawlist);
  vkBackendTrace(this->uboFrameIndex, "renderInternal.enter",
                 "overlaysOnly=%d cmds=%d",
                 static_cast<int>(overlaysOnly), drawlist.getNumCommands());

  if (SoVulkanConfig::get().debug.blackDebug) {
    static int blackFrame = 0;
    logBlackFrameStats(drawlist, params, blackFrame++,
                       overlaysOnly ? 1 : 0);
  }

  // A composite (overlays-only) frame with no overlay commands is a no-op:
  // return before the frame boundary so the ring cursor does not advance.
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
  // deferred maxFramesInFlight frames ago, then writes the lighting setup and
  // updates the geometry cache.  Composite frames also reserve the ring slots
  // their overlay/residual draws consume here; the full path reserves inside
  // recordFrame().
  FramePlan plan;
  if (!this->beginFramePlan(drawlist, params, "render", overlaysOnly,
                            /*external*/ false,
                            /*reserveCompositeSlots*/ overlaysOnly, plan,
                            nullptr)) {
    return FALSE;
  }
  const SoVulkanRenderTarget & target = *plan.target;

  // Render passes are cached by their attachment identity (formats, sample
  // count, image layouts, load ops), not by the target's images: swapchain
  // targets cycle their images every frame, and pipelines are keyed on the
  // render pass handle, so reusing the pass across image changes keeps the
  // pipeline cache warm.  On the full-target-clear fast path (FC_VULKAN_RP_CLEAR)
  // the color/depth attachments are cleared via their loadOp at pass begin,
  // which is cheaper than a separate vkCmdClearAttachments region clear.
  const bool wantRpClear = SoVulkanConfig::get().raster.rpClear;
  const bool fullTargetClear =
    wantRpClear && this->isFullTargetClear(params, target);
  const bool clearWindow = (params.flags & SO_PARAM_CLEAR_WINDOW) != 0;
  const bool clearDepth = (params.flags & SO_PARAM_CLEAR_DEPTH) != 0;
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  const VkAttachmentLoadOp colorLoadOp =
    (fullTargetClear && clearWindow)
      ? VK_ATTACHMENT_LOAD_OP_CLEAR
      : VK_ATTACHMENT_LOAD_OP_LOAD;
  const VkAttachmentLoadOp depthLoadOp =
    (fullTargetClear && hasDepth && clearDepth)
      ? VK_ATTACHMENT_LOAD_OP_CLEAR
      : VK_ATTACHMENT_LOAD_OP_LOAD;
  this->renderPasses.getOrCreateRenderPass(target, colorLoadOp,
                                           depthLoadOp);
  // Stash whether the pass cleared each attachment so recordClear() can skip
  // the redundant vkCmdClearAttachments, and (below) so the begin info carries
  // the matching clear values.
  this->renderPasses.setClearedByLoad(
    colorLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR,
    depthLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  if (this->renderPasses.currentRenderPass() == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan render pass");
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

  // The framebuffer is cached for the current target identity (image views +
  // extent + render pass) and recreated whenever any of those change.  The
  // old framebuffer is released through the deferred ring: an older in-flight
  // submission may still reference it (the per-frame vkQueueWaitIdle is gone,
  // so only the current slot's fence has been waited by beginFrame()).
  if (!this->renderPasses.ensureFramebuffer(
        &target, this->renderPasses.currentRenderPass())) {
    this->emitError("failed to create Vulkan framebuffer");
    // The one-shot command buffer was begun above and never submitted; an
    // implicit reset only happens on submission, so reset it explicitly or
    // every later beginCommandBuffer() will fail.
    vkEndCommandBuffer(this->currentCommandBuffer());
    vkResetCommandBuffer(this->currentCommandBuffer(), 0);
    return FALSE;
  }
  const VkFramebuffer framebuffer = this->renderPasses.framebuffer();

  // Record the pending texture copies into the frame command buffer (one
  // submit for the whole frame instead of a separate transfer submit) and
  // finalize the host-side resources.  The draws that sample these textures
  // are recorded below, after the copies, and the descriptor sets they bind
  // must already exist.  Staging buffers are released through the deferred
  // ring once the slot fence signals.
  this->gpuTimers.beginScope(this->currentCommandBuffer(), "textureUploads");
  if (!this->textureCache.recordPending()) {
    this->emitError("failed to record texture uploads");
  }
  this->textureCache.finalizePending();
  this->gpuTimers.endScope(this->currentCommandBuffer());

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = this->renderPasses.currentRenderPass();
  rpbi.framebuffer = framebuffer;
  rpbi.renderArea.offset = {0, 0};
  rpbi.renderArea.extent = target.extent;
  // When the render pass clears an attachment via its loadOp (full-target
  // clear fast path), the clear value must be supplied here.  clearValueCount
  // maps one-to-one to the attachment indices (0 = color, 1 = depth).
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

  // INLINE_AND_SECONDARY lets the opaque pass replay a secondary command
  // buffer (M1c/M1d) and still record inline commands in the same subpass.
  // That contents enum comes from VK_EXT_nested_command_buffer, so it is only
  // legal when the embedding created the device with the extension +
  // nestedCommandBufferRendering (recorded in nestedCommandBufferEnabled);
  // otherwise fall back to plain INLINE, which forbids
  // vkCmdExecuteCommands() -- recordFrame() disables secondary recording to
  // match.
  const VkSubpassContents subpassContents = this->nestedCommandBufferEnabled
    ? VK_SUBPASS_CONTENTS_INLINE_AND_SECONDARY_COMMAND_BUFFERS_EXT
    : VK_SUBPASS_CONTENTS_INLINE;
  vkCmdBeginRenderPass(this->currentCommandBuffer(), &rpbi, subpassContents);
  SoVulkanDebugUtils::beginLabel(this->currentCommandBuffer(),
                                 overlaysOnly ? "overlay pass" : "opaque pass",
                                 0.9f, 0.6f, 0.2f);
  this->gpuTimers.beginScope(this->currentCommandBuffer(), "renderPass");

  this->recordContext.buffer = this->currentCommandBuffer();
  const bool recorded = this->recordFramePlan(
    drawlist, params, plan, this->renderPasses.currentRenderPass(),
    this->renderPasses.framebuffer(), this->recordContext);
  this->recordContext.buffer = VK_NULL_HANDLE;

  this->gpuTimers.endScope(this->currentCommandBuffer());
  SoVulkanDebugUtils::endLabel(this->currentCommandBuffer());
  vkCmdEndRenderPass(this->currentCommandBuffer());
  SoVulkanDebugUtils::endLabel(this->currentCommandBuffer());

  // Submit even when recordFrame() failed: an unsubmitted one-shot command
  // buffer cannot be reused, and a partial frame is preferable to a dead
  // backend.
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

SbBool
SoVulkanRenderBackend::renderExternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass,
                                      VkFramebuffer framebuffer)
{
  const long externalBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;

  if (SoVulkanConfig::get().debug.blackDebug)
    fprintf(stderr, "[BLACK] renderExternal ENTER frame=%d cmds=%d\n",
            this->uboFrameIndex, drawlist.getNumCommands());

  this->debugValidateDrawList(drawlist);

  const bool wantCpuTiming = vkBackendFrameTimingEnabled();
  ExternalFrameTiming timing;
  FramePlan plan;
  if (!this->prepareExternalFrame(
        drawlist, params, commandBuffer, renderPass, "renderExternal",
        /*overlaysOnly*/ false, /*reserveCompositeSlots*/ false, plan,
        wantCpuTiming ? &timing : nullptr)) {
    return FALSE;
  }

  // The M1c/M1d secondary path records with RENDER_PASS_CONTINUE inheritance,
  // which needs the framebuffer matching the caller's pass + swapchain image.
  // The caller owns the pass/framebuffer pair (e.g. QVulkanWindow's
  // defaultRenderPass()/currentFramebuffer(), whose MSAA pass carries a
  // resolve attachment the backend cannot guess), so the framebuffer is
  // threaded in rather than fabricated here.
  if (framebuffer == VK_NULL_HANDLE) {
    this->emitError("renderExternal called without a framebuffer");
    return FALSE;
  }

  // External pre-pass.  Vulkan forbids transfer and compute inside a render
  // pass and the caller has already begun its pass, so the pending texture
  // copies and the sub-pixel compaction dispatches are recorded into one
  // transient command buffer here, before recordFrame(), so the draw path sees
  // the finalized textures and the compacted slots; it is submitted below,
  // after the frame is recorded.  Recording before and submitting after
  // overlaps the CPU frame recording with the previous GPU frame.  Non-fatal
  // on failure: the caller falls back to the one-shot texture upload and the
  // full-detail draw.
  VkCommandBuffer prepass = this->beginExternalPrepass(
    drawlist, params, /*lod*/ true, wantCpuTiming ? &timing : nullptr);
  if (prepass == VK_NULL_HANDLE && this->textureCache.hasPendingUploads()) {
    // The transient buffer could not carry the copies (allocation/begin/end
    // failure).  Fall back to the legacy one-shot upload so the textures still
    // land this frame; a failure there leaves the entries unstamped and the
    // next frame retries.
    const SoVulkan::Result uploadResult =
      this->textureCache.flushExternal();
    if (!uploadResult.isOk()) {
      SoDebugError::postWarning("SoVulkanRenderBackend::renderExternal",
                                "one-shot texture upload fallback failed: %s",
                                uploadResult.message().c_str());
    }
  }

  const double recordT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  const long recordBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  this->recordContext.buffer = commandBuffer;
  const bool recorded = this->recordFramePlan(
    drawlist, params, plan, renderPass, framebuffer, this->recordContext);
  vkBackendRenderBreadcrumbSince(recordBcStart, 5000, "renderExternal recordFrame end");
  this->recordContext.buffer = VK_NULL_HANDLE;
  const double recordEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;

  // Submit the pre-pass and wait so the copies and the compacted writes are
  // visible before the caller submits its pass.  Only the pre-pass is on the
  // critical path now; the frame recording above overlapped the previous GPU
  // frame.
  this->submitExternalPrepass(prepass, wantCpuTiming ? &timing : nullptr);

  if (wantCpuTiming) {
    const double recordMs = recordEnd - recordT0;
    const double lodMs = timing.lodRecordMs + timing.lodMs;
    const double totalMs = recordMs + timing.texMs + timing.geomMs +
                           timing.setupMs + lodMs;
    std::fprintf(stderr,
                 "[RTDBG] cpuTimingRaster mode=full setup=%.2f geom=%.2f "
                 "tex=%.2f lod=%.2f record=%.2f total=%.2f\n",
                 timing.setupMs, timing.geomMs, timing.texMs, lodMs,
                 recordMs, totalMs);
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
  if (SoVulkanConfig::get().debug.blackDebug)
    fprintf(stderr, "[BLACK] renderExternalOverlay ENTER frame=%d cmds=%d\n",
            this->uboFrameIndex, drawlist.getNumCommands());

  const bool wantCpuTiming = vkBackendFrameTimingEnabled();
  ExternalFrameTiming timing;
  FramePlan plan;
  if (!this->prepareExternalFrame(
        drawlist, params, commandBuffer, renderPass, "renderExternalOverlay",
        /*overlaysOnly*/ true, /*reserveCompositeSlots*/ true, plan,
        wantCpuTiming ? &timing : nullptr)) {
    return FALSE;
  }

  // The composite path is a raster overlay inside the caller's (RT) pass, so
  // it has no geometry-LOD pre-pass; it still routes any pending texture
  // copies through the transient pre-pass, because transfer commands cannot
  // be recorded inside the pass.  Fall back to the one-shot upload when the
  // transient buffer cannot be allocated.
  VkCommandBuffer prepass = this->beginExternalPrepass(
    drawlist, params, /*lod*/ false, wantCpuTiming ? &timing : nullptr);
  if (prepass == VK_NULL_HANDLE && this->textureCache.hasPendingUploads()) {
    const SoVulkan::Result uploadResult =
      this->textureCache.flushExternal();
    if (!uploadResult.isOk()) {
      SoDebugError::postWarning("SoVulkanRenderBackend::renderExternalOverlay",
                                "one-shot texture upload fallback failed: %s",
                                uploadResult.message().c_str());
    }
  }

  const double recordT0 = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->recordContext.buffer = commandBuffer;
  this->recordFramePlan(drawlist, params, plan, renderPass, VK_NULL_HANDLE,
                        this->recordContext);
  this->recordContext.buffer = VK_NULL_HANDLE;
  const double recordEnd = wantCpuTiming ? vkBackendRenderNowMs() : 0.0;
  this->submitExternalPrepass(prepass, wantCpuTiming ? &timing : nullptr);
  if (wantCpuTiming) {
    const double recordMs = recordEnd - recordT0;
    const double totalMs = recordMs + timing.texMs + timing.geomMs +
                           timing.setupMs + timing.lodMs;
    std::fprintf(stderr,
                 "[RTDBG] cpuTimingRaster mode=overlay setup=%.2f geom=%.2f "
                 "tex=%.2f record=%.2f total=%.2f\n",
                 timing.setupMs, timing.geomMs, timing.texMs, recordMs,
                 totalMs);
    std::fflush(stderr);
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

  // Geometry content identity for batching: reuse the cached content hash the
  // geometry cache computed when the buffer was uploaded (a map lookup) instead
  // of re-walking every vertex stream.
  auto contentHashOf = [this](const SoRenderCommand & c) -> uint64_t {
    const auto it = this->commandToCache.find(&c);
    if (it == this->commandToCache.end()) return 0;
    return this->gpuCache[it->second].contentHash;
  };

  uint32_t nextSlot = 0;
  // A replayed list reuses the previous frame's sorted order, which may be
  // shorter than (or hold stale indices for) the current command list; fall
  // back to the identity order for those entries.
  const auto orderedIndex = [&order, &drawlist](int i) {
    return (i < static_cast<int>(order.size()) &&
            order[i] < drawlist.getNumCommands())
      ? order[i] : i;
  };
  // Opaque then transparent, honoring the draw-list sort order.  Overlay
  // commands are handled by the dedicated overlay block outside the work list.
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
      // Only depth-tested opaque geometry is render-order independent, so it can
      // be reordered and batched.  Depth-off commands go to the on-top
      // annotation pass; the CPU wide-line path must stay per-command.  Bucket
      // by a batch key and re-verify pairwise (splitting any collision).
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
          // CPU-expanded per command, so never batched.  It still goes into a
          // secondary: prepareWideLineBuffers() has already grown the
          // per-command quad buffer on the recording thread, so the parallel
          // workers only fill the existing host-visible mapping and bind it.
          // The expansion is the dominant per-frame CPU cost on edge-heavy
          // scenes, and routing it through the workers is what parallelizes
          // it (previously it ran inline on the recording thread).
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

    // Wireframe/point overlay: re-draw opaque geometry in the requested fill
    // mode using a uniform edge color.
    //
    // When the request is a LINES (edge) overlay, re-draw only the actual
    // B-Rep feature-edge commands (SoBrepEdgeSet emits SO_TOPOLOGY_LINES /
    // LINE_STRIP).  Re-drawing every triangle command in polygon-LINES would
    // paint the raw tessellation -- the straight seam meridian on a sphere and
    // the radial fan spokes on a cylinder cap -- instead of the true feature
    // edges (rims, creases, seams).  A CAD edge overlay must show only feature
    // edges; smooth curved surfaces carry no feature edges and read as clean.
    //
    // The debug tessellation overlay is the opposite request: re-draw the
    // TRIANGLE commands in polygon-LINES so the raw triangulation edges are
    // visible on top of the shaded geometry, and skip the line commands so
    // the feature edges are not double-painted.
    if (!transparent && (wireframeFillMode >= 0 || tessellationOverlay)) {
      const bool isEdgeOverlay = (wireframeFillMode == SoDrawStyleElement::LINES);
      const int redrawFillMode = tessellationOverlay
        ? SoDrawStyleElement::LINES
        : wireframeFillMode;
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index = orderedIndex(i);
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        if (!command.geometry.positions || command.geometry.vertexCount == 0)
          continue;
        const SoPrimitiveTopology topo = command.geometry.topology;
        // For the edge overlay, restrict to commands that are themselves line
        // primitives; skip triangles so tessellation edges never render.
        if (isEdgeOverlay) {
          if (topo != SO_TOPOLOGY_LINES &&
              topo != SO_TOPOLOGY_LINE_STRIP) {
            continue;
          }
        }
        // For the debug tessellation overlay, restrict to triangle commands.
        if (tessellationOverlay) {
          if (topo != SO_TOPOLOGY_TRIANGLES &&
              topo != SO_TOPOLOGY_TRIANGLE_STRIP) {
            continue;
          }
        }
        if (!this->findCachedDrawable(command)) continue;
        const bool lineTopo = topo == SO_TOPOLOGY_LINES ||
          topo == SO_TOPOLOGY_LINE_STRIP;
        const bool triTopo = topo == SO_TOPOLOGY_TRIANGLES ||
          topo == SO_TOPOLOGY_TRIANGLE_STRIP;
        const bool tessHit = tessellationOverlay && triTopo;
        if (isEdgeOverlay && !lineTopo && !tessHit) continue;
        if (wireframeFillMode < 0 && !tessHit) continue;
        VulkanWorkItem item;
        item.single = &command;
        item.count = 1;
        item.fillModeOverride = redrawFillMode;
        item.fillModeOverride =
          tessHit ? SoDrawStyleElement::LINES : wireframeFillMode;
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
    if (!this->findCachedDrawable(command)) continue;
    VulkanWorkItem item;
    item.single = &command;
    item.count = 1;
    item.slotBase = nextSlot++;
    out.push_back(item);
  }

  // M1d: when the opaque pass is recorded in parallel, pre-resolve every
  // recordToSecondary item's pipeline here (single-threaded).  getOrCreatePipeline()
  // mutates the shared pipeline store/gpuCache on its cold path, so warming each
  // key ahead of the dispatch means the parallel recorders only hit the read-only
  // warm fast path and never race on the cache.  Opaque items use the default
  // (non-transparent, no fill-mode override, not an overlay) recording state.
  if (this->parallelRecordEnabled) {
    const auto * tgt = static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
    if (tgt) {
      for (const VulkanWorkItem & item : out) {
        if (!item.recordToSecondary) continue;
        const SoRenderCommand * const cmd =
          item.count > 1 ? item.commands[0] : item.single;
        // Pass the cache entry so the warmed key matches the one the record
        // path builds (it depends on the command's wide-line instance buffer).
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
  // Plant the pre-assigned disjoint slot block, then record one draw or one
  // instanced batch.  The record helper advances ctx.uboCmdIndex from here.
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
    // Batch rejected (e.g. a non-batchable command slipped in): fall back to
    // per-command draws over the item's slot range, each at its own slot.
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
  if (SoVulkanConfig::get().debug.backendDebug) {
    fprintf(stderr, "[SEC] begin chunk items=%zu secondary=%p frame=%u\n",
            items.size(), (const void*)secondary, this->uboFrameIndex);
  }
  vkResetCommandBuffer(secondary, 0);
  vkBackendTrace(this->uboFrameIndex, "recordSecondaryChunk.resetDone",
                 "secondary=%p",
                 reinterpret_cast<const void *>(secondary));
  // Start from a clean dedup cache: secondary buffers inherit nothing (not
  // dynamic state, pipeline, or descriptors) from the primary.
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
  // Latch the interaction-LOD state before any isWideLine() decision so the
  // wide-line expansion, pipeline key and draw path all agree for this frame.
  this->interactionLodActive = params.interactionLod == TRUE;
  // Wide-line CPU expansion: grow the per-command quad buffers (device-memory
  // allocation is not thread-safe) and compute the quads across the worker
  // pool before recording, which then only binds the cached buffers.  The
  // expansion is the CPU-heavy part of edge rendering and is embarrassingly
  // parallel; the command-buffer recording itself stays single-threaded.
  this->prepareWideLineBuffers(drawlist);
  this->expandWideLinesParallel(drawlist, params);
  if (SoVulkanConfig::get().debug.matrixDump) {
    s_debugFrame++;
    s_dumpCmdCount = 0;
  }
  if (SoVulkanConfig::get().debug.blackDebug) {
    static int blackFrame = 0;
    logBlackFrameStats(drawlist, params, blackFrame++, -1);
  }
  this->applyViewport(params, target, ctx);
  this->recordClear(params, target, this->renderPasses.colorClearedByLoad(),
                    this->renderPasses.depthClearedByLoad(), ctx);
  this->recordBackground(params, target, renderPass, ctx);
  // The background pass overrides the viewport/scissor for its own draw;
  // restore the viewport from params before recording geometry so draws
  // land in the requested region.
  this->applyViewport(params, target, ctx);

  // Vulkan-only display options, configured through setWireframeOverlay()/
  // setPointsOverlay()/setTessellationOverlay()/setEdgeColor().  Environment
  // variables act as a diagnostic fallback for the command line.
  const bool wireframeOverlay =
    this->wireframeOverlay || SoVulkanConfig::get().raster.wireframe;
  const bool pointsOverlay =
    this->pointsOverlay || SoVulkanConfig::get().raster.points;
  const bool tessellationOverlay =
    this->tessellationOverlay || SoVulkanConfig::get().raster.tessellation;
  float overlayColor[4] = {
    this->edgeColor[0], this->edgeColor[1], this->edgeColor[2],
    this->edgeColor[3]
  };
  // Parse the FC_VULKAN_EDGE_COLOR override (a diagnostic switch) once; it is
  // process-lifetime and this runs on the overlay path, so two env reads
  // per frame is pure overhead.  Only RGB is taken from the hex value; alpha
  // keeps the configured edgeColor's.
  struct EdgeColorOverride { bool present; float rgb[3]; };
  static const EdgeColorOverride edgeOverride = []() {
    EdgeColorOverride o{false, {0.0f, 0.0f, 0.0f}};
    const std::string & edge = SoVulkanConfig::get().raster.edgeColor;
    const char * hex = edge.empty() ? nullptr : edge.c_str();
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
  const int wireframeFillMode = wireframeOverlay
    ? SoDrawStyleElement::LINES
    : (pointsOverlay ? SoDrawStyleElement::POINTS : -1);
  if (SoVulkanConfig::get().debug.backendDebug) {
    static int overlayLog = 0;
    if (overlayLog++ < 3) {
      fprintf(stderr,
              "[OVL] wireframe=%d points=%d tess=%d fillMode=%d edgeColor=(%.2f,%.2f,%.2f,%.2f)\n",
              wireframeOverlay ? 1 : 0, pointsOverlay ? 1 : 0,
              tessellationOverlay ? 1 : 0, wireframeFillMode,
              overlayColor[0], overlayColor[1], overlayColor[2], overlayColor[3]);
    }
  }

  // Reserve per-draw lighting slots for the worst case (main pass plus
  // overlay redraws) before recording, so slotIndex can never overflow the
  // ring allocation (VUID-vkCmdBindDescriptorSets-pDynamicOffsets-01972).
  if (!this->prepareLightingSlots(countDrawCommands(drawlist,
                                                    wireframeFillMode,
                                                    tessellationOverlay))) {
    return FALSE;
  }

  // Build the read-only worklist (bucketed opaque + batched + transparent +
  // overlay-redraw + annotation items), each with a pre-assigned disjoint
  // slotBase, then record it.  buildWorkItems() mirrors the exact order the
  // previous inline loops used, and the slotBase values match what the
  // per-draw uboCmdIndex++ sequence produced, so recording is identical.
  std::vector<VulkanWorkItem> & workItems = this->workItemsScratch;
  this->buildWorkItems(drawlist, params, wireframeOverlay, pointsOverlay,
                       tessellationOverlay, overlayColor, renderPass,
                       workItems);
  vkBackendTrace(this->uboFrameIndex, "recordFrame.workItemsBuilt",
                 "items=%zu", workItems.size());

  // Secondaries are recorded with RENDER_PASS_CONTINUE inheritance into the
  // pass the frame is in.  On the INTERNAL path that pass is backend-owned
  // (renderPass == this->renderPasses.currentRenderPass()) and the combination
  // is exercised by the testsuite.  The EXTERNAL path (FreeCAD's QuarterVulkanWidget) hands us a
  // caller-owned pass/framebuffer/command-buffer triplet (QVulkanWindow's,
  // possibly MSAA); recording secondaries against it has proven to corrupt
  // NVIDIA driver state (crash inside the driver at the first render-pass
  // command after the replay) so it stays OFF unless explicitly opted in
  // while that interaction is investigated.
  const bool externalPass = renderPass != this->renderPasses.currentRenderPass();
  const bool canUseSecondary =
    this->nestedCommandBufferEnabled &&
    !this->secondaryCommandBuffers.empty() &&
    inheritFramebuffer != VK_NULL_HANDLE &&
    (!externalPass || SoVulkanConfig::get().concurrency.externalSecondary);
  const bool debugFlags =
    SoVulkanConfig::get().debug.matrixDump ||
    SoVulkanConfig::get().debug.blackDebug;

  // Count the render-order-independent opaque items that live in a secondary.
  uint64_t secondaryItemCount = 0;
  for (const VulkanWorkItem & item : workItems) {
    if (item.recordToSecondary) ++secondaryItemCount;
  }
  const bool wantParallel =
    canUseSecondary && this->parallelRecordEnabled && !debugFlags &&
    secondaryItemCount >= 64 && this->maxRecordWorkers > 1;
  vkBackendTrace(this->uboFrameIndex, "recordFrame.mode",
                 "secondary=%u canSec=%d parEnabled=%d W=%u wantPar=%d",
                 static_cast<unsigned>(secondaryItemCount),
                 canUseSecondary ? 1 : 0, this->parallelRecordEnabled ? 1 : 0,
                 this->maxRecordWorkers, wantParallel ? 1 : 0);

  if (canUseSecondary && secondaryItemCount > 0 && !wantParallel) {
    // M1c serial: one secondary holds the whole opaque pass, replayed in place.
    VkCommandBuffer secondary = this->currentSecondaryCommandBuffer();
    VkCommandBuffer primary = this->currentCommandBuffer();
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
    // The primary's bound state was NOT preserved across the secondary, so
    // reset its dedup cache before continuing inline.
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
    // longest-first for load balance), record each into its own worker
    // secondary in parallel, then replay all in order followed by the inline
    // painter-order / overlay / annotation items.
    if (SoVulkanConfig::get().debug.backendDebug) {
      static int parLog = 0;
      if (parLog++ < 3) {
        fprintf(stderr,
                "[PAR] parallel record: %u opaque items across %u workers\n",
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
    // Replay the secondaries in order, then inline the non-opaque items.
    // A failed worker's chunk is re-recorded inline (serial fallback) into the
    // primary instead of executing its possibly-invalid secondary.
    VkCommandBuffer primary = this->currentCommandBuffer();
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
  const SbVec2s frameSize = params.viewport.getViewportSizePixels();
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass != SO_RENDERPASS_OVERLAY) continue;
    const SoRasterState & raster = command.state.raster;
    if (!raster.scissorEnabled || raster.scissorWidth <= 0 ||
        raster.scissorHeight <= 0) {
      continue;
    }
    // The selection/preselection highlight is a full-frame overlay: it must
    // depth-test against the scene depth (so a selected face behind other
    // geometry stays hidden), NOT be forced on top.  Clearing the depth over
    // the whole viewport would let it composite over everything.  Only the
    // sub-viewport widgets (navigation cube, axis cross) clear depth so they
    // remain visible over the scene.
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
  // Same parallel wide-line expansion as recordFrame(): the RT composite
  // draws the line/point residue here, so it needs the same pre-expanded
  // buffers.
  this->prepareWideLineBuffers(drawlist);
  this->expandWideLinesParallel(drawlist, params);

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
