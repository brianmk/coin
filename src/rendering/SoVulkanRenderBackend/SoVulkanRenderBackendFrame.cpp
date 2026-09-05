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
#include <vector>

using namespace CoinVulkanDetail;

namespace {

long vkBackendRenderNowUs()
{
  return (long)std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
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
  // so synchronous destruction is safe here.
  for (const PendingTextureUpload & upload : this->pendingUploads) {
    if (upload.staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(this->device, upload.staging, this->allocator);
      vkFreeMemory(this->device, upload.stagingMemory, this->allocator);
    }
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
  // count, image layouts), not by the target's images: swapchain targets
  // cycle their images every frame, and pipelines are keyed on the render
  // pass handle, so reusing the pass across image changes keeps the pipeline
  // cache warm.
  this->renderPass = this->getOrCreateRenderPass(*target);
  if (this->renderPass == VK_NULL_HANDLE) {
    this->emitError("failed to create Vulkan render pass");
    return FALSE;
  }

  this->updateGeometryCache(drawlist, overlaysOnly);

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
  rpbi.clearValueCount = 0;
  rpbi.pClearValues = nullptr;

  vkCmdBeginRenderPass(this->currentCommandBuffer(), &rpbi,
                       VK_SUBPASS_CONTENTS_INLINE);

  this->activeCommandBuffer = this->currentCommandBuffer();
  bool recorded = true;
  if (overlaysOnly) {
    this->recordTracedComposite(drawlist, params, *target, this->renderPass);
    this->recordOverlayBlock(drawlist, params, *target, this->renderPass);
  }
  else {
    recorded = this->recordFrame(drawlist, params, *target, this->renderPass);
  }
  this->activeCommandBuffer = VK_NULL_HANDLE;

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

  const long externalBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  this->beginFrame();
  this->updateLightingSetup(drawlist);
  const long geometryBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  this->updateGeometryCache(drawlist);
  vkBackendRenderBreadcrumbSince(geometryBcStart, 5000, "renderExternal updateGeometryCache end");
  const long textureBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  if (!this->flushPendingTextureUploadsExternal()) {
    this->emitError("renderExternal: texture upload failed");
    return FALSE;
  }
  vkBackendRenderBreadcrumbSince(textureBcStart, 5000, "renderExternal flushPendingTextureUploadsExternal end");

  const long recordBcStart = vkBackendRenderBreadcrumbEnabled() ? vkBackendRenderNowUs() : 0;
  this->activeCommandBuffer = commandBuffer;
  const bool recorded = this->recordFrame(drawlist, params, *target, renderPass);
  vkBackendRenderBreadcrumbSince(recordBcStart, 5000, "renderExternal recordFrame end");
  this->activeCommandBuffer = VK_NULL_HANDLE;
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

  this->beginFrame();
  this->updateLightingSetup(drawlist);
  this->updateGeometryCache(drawlist, true);

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

  this->activeCommandBuffer = commandBuffer;
  this->recordTracedComposite(drawlist, params, *target, renderPass);
  this->recordOverlayBlock(drawlist, params, *target, renderPass);
  this->activeCommandBuffer = VK_NULL_HANDLE;
  return TRUE;
}

bool
SoVulkanRenderBackend::recordFrame(const SoDrawList & drawlist,
                                   const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target,
                                   VkRenderPass renderPass)
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
  this->applyViewport(params, target);
  this->recordClear(params, target);
  this->recordBackground(params, target, renderPass);
  // The background pass overrides the viewport/scissor for its own draw;
  // restore the viewport from params before recording geometry so draws
  // land in the requested region.
  this->applyViewport(params, target);

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

  // Opaque then transparent, honoring the draw-list sort order.  Overlay
  // commands (SO_RENDERPASS_OVERLAY) are handled exclusively by the
  // dedicated overlay block below: they carry their own view/projection
  // matrices and viewport, so recording them here with the main camera
  // matrices would draw garbage into the scene depth buffer.
  const std::vector<int> & order = drawlist.getSortedOrder();
  for (int passIndex = 0; passIndex < 2; ++passIndex) {
    const bool transparent = passIndex == 1;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      const int index =
        i < static_cast<int>(order.size()) ? order[i] : i;
      const SoRenderCommand & command = drawlist.getCommand(index);
      if (command.pass == SO_RENDERPASS_OVERLAY) continue;
      const bool isTransparent = command.pass == SO_RENDERPASS_TRANSPARENT;
      if (isTransparent != transparent) continue;
      this->recordDrawCommand(drawlist, command, target, params, renderPass,
                              transparent);
    }

    // Wireframe/point overlay: re-draw opaque geometry in the requested fill
    // mode using a uniform edge color (no vertex colors, no texture, unlit).
    if (!transparent && overlayFillMode >= 0) {
      for (int i = 0; i < drawlist.getNumCommands(); ++i) {
        const int index =
          i < static_cast<int>(order.size()) ? order[i] : i;
        const SoRenderCommand & command = drawlist.getCommand(index);
        if (command.pass == SO_RENDERPASS_OVERLAY) continue;
        if (command.pass == SO_RENDERPASS_TRANSPARENT) continue;
        this->recordDrawCommand(drawlist, command, target, params, renderPass,
                                false, overlayFillMode, overlayColor);
      }
    }
  }

  // On-top annotations: commands recorded with the depth test disabled are
  // drawn after both passes in insertion order.  This mirrors GL's delayed
  // annotations pass (glClear(GL_DEPTH_BUFFER_BIT) + re-render on top),
  // which clarify selection uses to show a highlighted shape through
  // occluding geometry: the base shape and its highlight are recorded with
  // the depth test off during traversal and deferred here so later-drawn
  // occluders cannot paint over them.
  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (command.pass == SO_RENDERPASS_OVERLAY) continue;
    if (command.state.depth.enabled) continue;
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false);
  }

  // Screen-space overlay geometry (navigation cube): drawn after both passes
  // into its own viewport, with the overlay rect's depth cleared first so the
  // overlay self-occludes independently of the main scene.
  this->recordOverlayBlock(drawlist, params, target, renderPass);

  return true;
}

void
SoVulkanRenderBackend::recordOverlayBlock(const SoDrawList & drawlist,
                                          const SoRenderParams & params,
                                          const SoVulkanRenderTarget & target,
                                          VkRenderPass renderPass)
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
      this->recordOverlayDepthClear(command, target);
      lastClearX = raster.scissorX;
      lastClearY = raster.scissorY;
      lastClearW = raster.scissorWidth;
      lastClearH = raster.scissorHeight;
    }
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, true);
  }
}

void
SoVulkanRenderBackend::recordTracedComposite(const SoDrawList & drawlist,
                                             const SoRenderParams & params,
                                             const SoVulkanRenderTarget & target,
                                             VkRenderPass renderPass)
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

    const SoRasterState & raster = command.state.raster;
    // Apply the command's own viewport/scissor if it carries one, else the
    // whole-surface viewport (the default for scene geometry).
    this->applyCommandViewport(command, target);
    this->applyScissor(command, target);
    // overlayPass=false: these are scene-geometry edge/point commands, so
    // they must draw with the FRAME camera (params.projMatrix), not the
    // command's own projection matrix.  Passing overlayPass=true made them
    // use a stale per-command proj, displacing them off the traced surface
    // (the offset "phantom box").  They inherit the raster depth compare so
    // the present-pass depth occludes hidden edges.
    this->recordDrawCommand(drawlist, command, target, params, renderPass,
                            false, -1, nullptr, false);
  }
}
