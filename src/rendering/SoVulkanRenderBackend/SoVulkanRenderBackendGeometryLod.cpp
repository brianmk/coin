// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendGeometryLod.cpp
//
// GPU sub-pixel primitive culling ("geometry LOD") for the raster backend.  While
// the camera moves (interaction LOD), each eligible triangle command is compacted
// on the GPU: a compute shader applies the visual shader's model*view*proj,
// measures screen-space area, and appends survivors to a dense index buffer drawn
// via vkCmdDrawIndexedIndirect, so culled triangles cost no shading/rasterization.
//
// Vulkan forbids compute/transfer inside a render pass, so the own-queue path
// records the pre-pass before vkCmdBeginRenderPass.  The external path
// (renderExternal()) instead records dispatches plus pending texture copies into a
// transient buffer (beginExternalPrepass()), submitted after frame recording
// (submitExternalPrepass()) to overlap with the previous GPU frame; a trailing
// barrier orders compute writes against DRAW_INDIRECT/VERTEX_INPUT reads.
//
// Resources are per (command, in-flight frame): the compacted buffer is rewritten
// every frame and must not alias a still-executing frame's read; built lazily and
// kept until the content hash changes.  Commands over COIN_VULKAN_GEOM_LOD_MAX_INDEX
// fall back to the full draw to bound memory.  Validation caveat: a nav-cube-only
// run exercises neither the document geometry nor the indexed path;
// tools/fcprobe/vk_geomlod_probe.py checks a real indexed Part.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanDebug.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanConfig.h"

#include <Inventor/errors/SoDebugError.h>

#include <vk_mem_alloc.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace CoinVulkanDetail;

namespace {

// Push-constant block shared with SubPixelCull.glsl.
struct SubPixelPush {
  float mvp[16];      // offset 0
  float params[4];    // offset 64: vp width, vp height, min 2x-area, prim count
  float offsets[4];   // offset 80: vertex base (floats), index base (uints)
};
static_assert(sizeof(SubPixelPush) == 96, "push block must be 96 bytes");

// Minimum projected triangle area (px^2) that survives; 1 px keeps the LOD
// faithful while dropping sub-pixel filler.  From COIN_VULKAN_GEOM_LOD_* config.
float geometryLodMinAreaPixels()
{
  return SoVulkanConfig::get().geometryLod.minAreaPixels;
}

// Largest index count that gets a compacted buffer (indexCount * 4 B per in-flight
// slot).  Default tracks MAX_VERTEX_COUNT so huge meshes are compacted; the old 16M
// default silently excluded e.g. a 23.3M-element Voron face set (COIN_VULKAN_GEOM_LOD_MAX_INDEX).
uint32_t geometryLodMaxIndices()
{
  return SoVulkanConfig::get().geometryLod.maxIndices;
}

// Smallest triangle count worth compacting; below it the per-command fixed cost
// (cursor fill, barrier, dispatch, descriptor bind) exceeds the saving.  Keeps many
// small parts cheap.  COIN_VULKAN_GEOM_LOD_MIN_PRIMS; 0 disables.
uint32_t geometryLodMinPrims()
{
  return SoVulkanConfig::get().geometryLod.minPrims;
}

bool geometryLodEnabled()
{
  return SoVulkanConfig::get().geometryLod.enabled;
}

// Force the pre-pass on even when the camera is not moving (verification aid).
bool geometryLodAlways()
{
  return SoVulkanConfig::get().geometryLod.always;
}

// Print the previous frame's survivor count per compacted command -- the only proof
// on a real mesh.  At COIN_VULKAN_GEOM_LOD_PIXELS=0 all survive; otherwise cull heavily.
bool geometryLodStats()
{
  return SoVulkanConfig::get().geometryLod.stats;
}

// Indexed-ness and element count of a command's triangle stream.  The IR emits
// Voron-class meshes non-indexed and SoBrepFaceSet indexed, so non-indexed vertices
// become sequential indices; shared by all three call sites so the rule cannot drift.
struct SubPixelElementForm {
  bool indexed;
  uint32_t elements;
};

SubPixelElementForm subPixelElementForm(const SoRenderCommand & command)
{
  const bool indexed = command.geometry.indices != nullptr &&
    command.geometry.indexCount >= 3;
  return {indexed, indexed ? command.geometry.indexCount
                           : command.geometry.vertexCount};
}

} // namespace

bool
SoVulkanRenderBackend::isSubPixelEligible(const SoRenderCommand & command)
{
  if (command.pass == SO_RENDERPASS_OVERLAY) return false;
  if (command.geometry.topology != SO_TOPOLOGY_TRIANGLES) return false;
  // Voron-class meshes are non-indexed, so both forms are handled.  Triangle
  // lists carry a multiple of three; anything else stays in the full draw.
  const SubPixelElementForm form = subPixelElementForm(command);
  if (form.elements < 3 || (form.elements % 3) != 0) return false;
  // Small meshes aren't worth a dispatch: draw in full.
  if ((form.elements / 3) < geometryLodMinPrims()) return false;
  return true;
}

const VulkanCachedCommand::VulkanSubPixelSlot *
SoVulkanRenderBackend::subPixelSlotFor(const VulkanCachedCommand & entry) const
{
  if (entry.subPixelSlots.empty()) return nullptr;
  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  if (slot >= entry.subPixelSlots.size()) return nullptr;
  const VulkanCachedCommand::VulkanSubPixelSlot & s = entry.subPixelSlots[slot];
  if (s.readyFrame != this->uboFrameIndex) return nullptr;
  if (s.indexBuffer == VK_NULL_HANDLE || s.indirectBuffer == VK_NULL_HANDLE) {
    return nullptr;
  }
  return &s;
}

void
SoVulkanRenderBackend::destroySubPixelResources(VulkanCachedCommand & entry)
{
  for (VulkanCachedCommand::VulkanSubPixelSlot & s : entry.subPixelSlots) {
    if (s.indexBuffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(this->vmaAllocator, s.indexBuffer, s.indexMemory);
      s.indexBuffer = VK_NULL_HANDLE;
      s.indexMemory = nullptr;
    }
    if (s.indirectBuffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(this->vmaAllocator, s.indirectBuffer, s.indirectMemory);
      s.indirectBuffer = VK_NULL_HANDLE;
      s.indirectMemory = nullptr;
    }
  }
  entry.subPixelSlots.clear();
  entry.subPixelHash = 0;
}

void
SoVulkanRenderBackend::deferDestroySubPixelResources(VulkanCachedCommand & entry)
{
  if (entry.subPixelSlots.empty()) return;
  std::vector<VulkanCachedCommand::VulkanSubPixelSlot> slots =
    std::move(entry.subPixelSlots);
  VmaAllocator vma = this->vmaAllocator;
  this->deferDestroy([vma, slots]() mutable {
    for (VulkanCachedCommand::VulkanSubPixelSlot & s : slots) {
      if (s.indexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, s.indexBuffer, s.indexMemory);
      }
      if (s.indirectBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vma, s.indirectBuffer, s.indirectMemory);
      }
    }
  });
  entry.subPixelHash = 0;
}

bool
SoVulkanRenderBackend::allocateSubPixelDescriptorSet(VkDescriptorSet & set)
{
  if (this->subPixelDescriptorPools.empty() ||
      this->subPixelDescriptorSetCount >= 4096) {
    VkDescriptorPoolSize size {};
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 4096 * 4;
    VkDescriptorPoolCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 4096;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(this->device, &ci, this->allocator, &pool) !=
        VK_SUCCESS) {
      return false;
    }
    this->subPixelDescriptorPools.push_back(pool);
    this->subPixelDescriptorSetCount = 0;
  }

  VkDescriptorSetAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = this->subPixelDescriptorPools.back();
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &this->subPixelSetLayout;
  if (vkAllocateDescriptorSets(this->device, &ai, &set) != VK_SUCCESS) {
    return false;
  }
  ++this->subPixelDescriptorSetCount;
  return true;
}

bool
SoVulkanRenderBackend::ensureSubPixelSlot(VulkanCachedCommand & entry,
                                          const SoRenderCommand & command,
                                          const uint32_t slot)
{
  // Non-indexed lists treat vertices as sequential indices; output is always an index buffer.
  const SubPixelElementForm form = subPixelElementForm(command);
  const bool indexed = form.indexed;
  const uint32_t elementCount = form.elements;
  if (elementCount == 0) return false;
  if (elementCount > geometryLodMaxIndices()) {
    // Log once per command so an oversized mesh's silent LOD skip is visible.
    if (!entry.warnedGeomLodCap) {
      entry.warnedGeomLodCap = true;
      SoVulkanDebug::post("[GEOMLOD] command has %u elements > cap %u; geometry LOD "
              "skipped for it (raise COIN_VULKAN_GEOM_LOD_MAX_INDEX)\n",
              elementCount, geometryLodMaxIndices());
    }
    return false;
  }
  if (entry.vertexBuffer == VK_NULL_HANDLE) return false;
  if (indexed && entry.indexBuffer == VK_NULL_HANDLE) return false;

  if (entry.subPixelSlots.size() < this->maxFramesInFlight) {
    entry.subPixelSlots.resize(this->maxFramesInFlight);
  }
  VulkanCachedCommand::VulkanSubPixelSlot & s = entry.subPixelSlots[slot];

  // (Re)build if empty or too small; content changes already invalidated the slots.
  if (s.indexBuffer == VK_NULL_HANDLE || s.maxIndices < elementCount) {
    if (s.indexBuffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(this->vmaAllocator, s.indexBuffer, s.indexMemory);
      s.indexBuffer = VK_NULL_HANDLE;
      s.indexMemory = nullptr;
    }
    const VkDeviceSize indexBytes =
      static_cast<VkDeviceSize>(elementCount) * sizeof(uint32_t);
    if (!this->createBufferDeviceLocal(
          indexBytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
          s.indexBuffer, s.indexMemory, nullptr)) {
      return false;
    }
    s.maxIndices = elementCount;
  }

  if (s.indirectBuffer == VK_NULL_HANDLE) {
    VkDrawIndexedIndirectCommand icmd {};
    icmd.indexCount = 0;
    icmd.instanceCount = 1;
    icmd.firstIndex = 0;
    icmd.vertexOffset = 0;
    icmd.firstInstance = 0;
    // Host-visible: the command is 20 B and rewritten in place by the compute
    // atomic, so device-local staging would only add a synchronous drain.
    if (!this->createBuffer(sizeof(icmd),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            s.indirectBuffer, s.indirectMemory, &icmd)) {
      return false;
    }
  }

  if (s.descriptorSet == VK_NULL_HANDLE) {
    if (!this->allocateSubPixelDescriptorSet(s.descriptorSet)) {
      return false;
    }
  }

  // Descriptors bind the whole (possibly shared) buffer; the shader applies the
  // per-command base offset.  Explicit ranges (not VK_WHOLE_SIZE) keep within maxStorageBufferRange.
  const VkDeviceSize vertexRange = entry.vertexOffset +
    static_cast<VkDeviceSize>(command.geometry.vertexCount) *
      VULKAN_VERTEX_STRIDE;
  const VkDeviceSize outRange =
    static_cast<VkDeviceSize>(elementCount) * sizeof(uint32_t);

  // A binding cannot span maxStorageBufferRange, which a huge mesh (23M verts * 32 B
  // ~= 745 MB) can exceed, invalidating vkUpdateDescriptorSets.  Fall back, logging once.
  const VkDeviceSize indexRange = indexed
    ? entry.indexOffset +
        static_cast<VkDeviceSize>(command.geometry.indexCount) * sizeof(uint32_t)
    : 0;
  if (this->subPixelMaxStorageRange != 0) {
    const VkDeviceSize worst = std::max(std::max(vertexRange, indexRange),
                                        outRange);
    if (worst > this->subPixelMaxStorageRange) {
      if (!entry.warnedGeomLodRange) {
        entry.warnedGeomLodRange = true;
        SoVulkanDebug::post("[GEOMLOD] command needs a %llu-byte storage range but the "
                "device limit is %llu; geometry LOD skipped for it\n",
                static_cast<unsigned long long>(worst),
                static_cast<unsigned long long>(this->subPixelMaxStorageRange));
      }
      return false;
    }
  }

  VkDescriptorBufferInfo infos[4] {};
  infos[0].buffer = entry.vertexBuffer;
  infos[0].offset = 0;
  infos[0].range = vertexRange;
  // Binding 1 is read only when indexed; non-indexed binds the vertex buffer (unsampled) to stay valid.
  infos[1].buffer = indexed ? entry.indexBuffer : entry.vertexBuffer;
  infos[1].offset = 0;
  infos[1].range = indexed ? indexRange : vertexRange;
  infos[2].buffer = s.indexBuffer;
  infos[2].offset = 0;
  infos[2].range = outRange;
  infos[3].buffer = s.indirectBuffer;
  infos[3].offset = 0;
  infos[3].range = sizeof(VkDrawIndexedIndirectCommand);

  VkWriteDescriptorSet writes[4] {};
  for (uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = s.descriptorSet;
    writes[i].dstBinding = i;
    writes[i].dstArrayElement = 0;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(this->device, 4, writes, 0, nullptr);
  return true;
}

uint32_t
SoVulkanRenderBackend::recordGeometryLodPrepass(VkCommandBuffer cb,
                                                const SoDrawList & drawlist,
                                                const SoRenderParams & params)
{
  const int num = drawlist.getNumCommands();
  if (num == 0) return 0;

  const bool debug = COIN_VULKAN_ENV_FLAG("COIN_VULKAN_BACKEND_DEBUG");

  // Dump the command list once per process; atomic exchange keeps the latch race-free.
  static std::atomic<bool> dumpedCommands {false};
  if (debug && !dumpedCommands.exchange(true)) {
    for (int i = 0; i < num; ++i) {
      const SoRenderCommand & c = drawlist.getCommand(i);
      SoVulkanDebug::post("[GEOMLOD] cmd %d topo=%d vc=%u ic=%u pass=%d idx=%p\n",
              i, static_cast<int>(c.geometry.topology),
              c.geometry.vertexCount, c.geometry.indexCount,
              static_cast<int>(c.pass),
              reinterpret_cast<const void *>(c.geometry.indices));
    }
  }

  const uint32_t slot = this->uboFrameIndex % this->maxFramesInFlight;
  const SbVec2s vpSize = params.viewport.getViewportSizePixels();
  const float vpW = static_cast<float>(vpSize[0]);
  const float vpH = static_cast<float>(vpSize[1]);
  // The shader's cross product is twice the pixel area, so threshold = 2 * minArea.
  const float areaThreshold = 2.0f * geometryLodMinAreaPixels();

  uint32_t compacted = 0;
  uint32_t skipped = 0;
  uint32_t maxElements = 0;
  uint32_t maxPrims = 0;
  // maxVc/maxIc feed only the debug summary, so accumulate them only when it prints.
  uint32_t maxVc = 0;
  uint32_t maxIc = 0;
  for (int i = 0; i < num; ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    if (debug) {
      if (command.geometry.vertexCount > maxVc) {
        maxVc = command.geometry.vertexCount;
      }
      if (command.geometry.indexCount > maxIc) {
        maxIc = command.geometry.indexCount;
      }
    }
    if (!isSubPixelEligible(command)) continue;
    const auto found = this->commandToCache.find(&command);
    if (found == this->commandToCache.end()) continue;
    VulkanCachedCommand & entry = this->gpuCache[found->second];

    if (entry.subPixelHash != entry.contentHash) {
      this->deferDestroySubPixelResources(entry);
      entry.subPixelHash = entry.contentHash;
    }
    if (!this->ensureSubPixelSlot(entry, command, slot)) {
      ++skipped;
      continue;
    }
    VulkanCachedCommand::VulkanSubPixelSlot & s = entry.subPixelSlots[slot];

    const SubPixelElementForm form = subPixelElementForm(command);
    const bool indexed = form.indexed;
    const uint32_t elementCount = form.elements;
    const uint32_t primCount = elementCount / 3;

    // Report the PREVIOUS frame's result before the cursor reset; the indirect buffer is
    // host-visible/coherent and that frame completed (frames-in-flight).  Guarded from normal sessions.
    if (geometryLodStats() && s.readyFrame != 0) {
      void * mapped = nullptr;
      if (vmaMapMemory(this->vmaAllocator, s.indirectMemory, &mapped) ==
          VK_SUCCESS) {
        uint32_t survivors = 0;
        std::memcpy(&survivors, mapped, sizeof(uint32_t));
        vmaUnmapMemory(this->vmaAllocator, s.indirectMemory);
        // The shader appends INDICES, so survivorPrims = indexCount / 3.  At
        // COIN_VULKAN_GEOM_LOD_PIXELS=0 all must survive; otherwise cull heavily.
        const uint32_t survivorPrims = survivors / 3u;
        const float culled = primCount
          ? 100.0f * (1.0f - static_cast<float>(survivorPrims) /
                               static_cast<float>(primCount))
          : 0.0f;
        SoVulkanDebug::post("[GEOMLOD] stats slot=%u prims=%u survivors=%u culled=%.1f%%\n",
                slot, primCount, survivorPrims, culled);
      }
    }

    // Reset the indirect indexCount (append cursor) to zero; only the first 4 bytes are touched.
    vkCmdFillBuffer(cb, s.indirectBuffer, 0, sizeof(uint32_t), 0);

    SoVulkanShared::memoryBarrier(
      cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    SubPixelPush pc {};
    // Same combined transform as the visual shader (which applies u_proj*u_view*model):
    // the CPU composes model*view*proj in row-vector order, packed as mat4 columns.
    SbMatrix mvp =
      command.modelMatrix * params.viewMatrix * params.projMatrix;
    std::memcpy(pc.mvp, &mvp[0][0], sizeof(float) * 16);
    pc.params[0] = vpW;
    pc.params[1] = vpH;
    pc.params[2] = areaThreshold;
    pc.params[3] = static_cast<float>(primCount);
    pc.offsets[0] = static_cast<float>(entry.vertexOffset / sizeof(float));
    pc.offsets[1] = static_cast<float>(entry.indexOffset / sizeof(uint32_t));
    pc.offsets[2] = indexed ? 1.0f : 0.0f;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      this->subPixelCullPipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            this->subPixelPipelineLayout, 0, 1,
                            &s.descriptorSet, 0, nullptr);
    vkCmdPushConstants(cb, this->subPixelPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const uint32_t groups = (primCount + 63u) / 64u;
    vkCmdDispatch(cb, groups, 1, 1);

    s.readyFrame = this->uboFrameIndex;
    ++compacted;
    if (elementCount > maxElements) {
      maxElements = elementCount;
      maxPrims = primCount;
    }
  }

  // One barrier after all dispatches makes compute writes visible to the
  // indirect-command and vertex/index reads.  Skipped when nothing was dispatched.
  if (compacted > 0) {
    SoVulkanShared::memoryBarrier(
      cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
  }

  if (debug) {
    SoVulkanDebug::post("[GEOMLOD] prepass slot=%u compacted=%u skipped=%u "
                    "threshold=%.2fpx2 maxPrims=%u maxVc=%u maxIc=%u\n",
            slot, compacted, skipped, areaThreshold, maxPrims, maxVc, maxIc);
  }
  return compacted;
}

bool
SoVulkanRenderBackend::externalGeometryLodActive(const SoRenderParams & params) const
{
  // Geometry LOD runs only while the camera moves; otherwise use the full-detail draw.
  if (params.interactionLod != TRUE && !geometryLodAlways()) return false;
  if (!geometryLodEnabled()) return false;
  return this->subPixelCullPipeline != VK_NULL_HANDLE;
}

VkCommandBuffer
SoVulkanRenderBackend::beginExternalPrepass(const SoDrawList & drawlist,
                                            const SoRenderParams & params,
                                            const bool lod,
                                            ExternalFrameTiming * timing)
{
  const bool wantTextures = !this->pendingUploads.empty();
  const bool wantLod = lod && this->externalGeometryLodActive(params);
  if (!wantTextures && !wantLod) return VK_NULL_HANDLE;

  const double recordT0 = timing ? SoVulkanShared::steadyNowMs() : 0.0;

  // The caller's external pass is already begun, so neither buffer->image copies nor
  // dispatches can be recorded into it; record both into one transient buffer before
  // recordFrame() and submit after recording (overlaps the previous GPU frame).
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(this->device, &allocInfo, &cb) != VK_SUCCESS) {
    SoDebugError::postWarning("SoVulkanRenderBackend::beginExternalPrepass",
                              "external pre-pass transient buffer allocation "
                              "failed; the caller falls back to the one-shot "
                              "texture upload and the full-detail draw");
    return VK_NULL_HANDLE;
  }
  VkCommandBufferBeginInfo beginInfo {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cb, &beginInfo) != VK_SUCCESS) {
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &cb);
    SoDebugError::postWarning("SoVulkanRenderBackend::beginExternalPrepass",
                              "external pre-pass transient buffer begin failed; "
                              "the caller falls back to the one-shot texture "
                              "upload and the full-detail draw");
    return VK_NULL_HANDLE;
  }

  // Record copies/dispatches now, but defer host-side finalize (view/sampler/descriptor +
  // stamp) until vkEndCommandBuffer() succeeds; on failure pendingUploads must survive.
  if (wantTextures) {
    this->recordPendingTextureUploadsInto(cb);
  }
  const double texEnd = timing ? SoVulkanShared::steadyNowMs() : 0.0;
  if (timing) timing->texMs = texEnd - recordT0;

  uint32_t lodCompacted = 0;
  if (wantLod) {
    lodCompacted = this->recordGeometryLodPrepass(cb, drawlist, params);
  }
  if (timing) timing->lodRecordMs = SoVulkanShared::steadyNowMs() - texEnd;

  // Empty buffer (no texture copies, nothing compacted): end/free it and report "no
  // pre-pass" so the caller skips submitExternalPrepass() -- a full host wait.
  if (!wantTextures && lodCompacted == 0) {
    vkEndCommandBuffer(cb);
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &cb);
    return VK_NULL_HANDLE;
  }

  if (vkEndCommandBuffer(cb) != VK_SUCCESS) {
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &cb);
    SoDebugError::postWarning("SoVulkanRenderBackend::beginExternalPrepass",
                              "external pre-pass transient buffer end failed; "
                              "the caller falls back to the one-shot texture "
                              "upload and the full-detail draw");
    return VK_NULL_HANDLE;
  }
  // Copies are recorded, so the draw path can create its descriptor sets and stamp identity.
  if (wantTextures) {
    this->finalizePendingTextureUploads();
  }
  return cb;
}

void
SoVulkanRenderBackend::submitExternalPrepass(VkCommandBuffer commandBuffer,
                                             ExternalFrameTiming * timing)
{
  if (commandBuffer == VK_NULL_HANDLE) return;
  const double submitT0 = timing ? SoVulkanShared::steadyNowMs() : 0.0;
  VkSubmitInfo submit {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &commandBuffer;
  // Host wait: copies and compacted writes must complete before the caller submits, and
  // its submission is unreachable by semaphore.  A dedicated fence observes only this
  // pre-pass; fall back to the queue drain if it can't be created/reset.
  if (this->externalPrepassFence == VK_NULL_HANDLE) {
    VkFenceCreateInfo fenceInfo {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(this->device, &fenceInfo, this->allocator,
                      &this->externalPrepassFence) != VK_SUCCESS) {
      this->externalPrepassFence = VK_NULL_HANDLE;
    }
  }
  bool submitted = false;
  bool waited = false;
  if (this->externalPrepassFence != VK_NULL_HANDLE &&
      vkResetFences(this->device, 1, &this->externalPrepassFence) ==
        VK_SUCCESS) {
    submitted = vkQueueSubmit(this->queue, 1, &submit,
                              this->externalPrepassFence) == VK_SUCCESS;
    if (submitted) {
      waited = vkWaitForFences(this->device, 1, &this->externalPrepassFence,
                               VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }
  }
  else {
    submitted =
      vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
    waited = submitted && vkQueueWaitIdle(this->queue) == VK_SUCCESS;
  }
  vkFreeCommandBuffers(this->device, this->commandPool, 1, &commandBuffer);
  if (timing) timing->lodMs = SoVulkanShared::steadyNowMs() - submitT0;
  if (!submitted || !waited) {
    SoDebugError::postWarning("SoVulkanRenderBackend::submitExternalPrepass",
                              "external pre-pass transient submit failed; "
                              "falling back to the full-detail draw");
  }
}
