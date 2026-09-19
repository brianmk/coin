// src/rendering/SoVulkanRenderBackend/SoVulkanRenderBackendGeometryLod.cpp
//
// GPU sub-pixel primitive culling ("geometry LOD") for the raster backend.
//
// While the camera moves (interaction LOD), each eligible indexed triangle
// command is compacted on the GPU: a compute shader transforms every triangle
// with the same model*view*projection the visual vertex shader uses, measures
// its screen-space area, and appends the survivors to a dense index buffer.
// The draw is then issued as vkCmdDrawIndexedIndirect, so culled triangles
// cost neither vertex shading nor rasterization.
//
// Vulkan forbids compute (and transfer) commands inside a render pass.  The
// own-queue path records the pre-pass into its own command buffer before
// vkCmdBeginRenderPass; the external path (renderExternal()) cannot, because
// the caller owns and has already begun its pass, so it records the dispatches
// -- together with any pending texture copies -- into one transient command
// buffer (beginExternalPrepass()) and submits it after the frame is recorded
// (submitExternalPrepass()), before the caller submits its pass.  Recording
// before and submitting after the frame recording overlaps the CPU work with
// the previous GPU frame.  A trailing memory barrier orders the compute writes
// against the DRAW_INDIRECT / VERTEX_INPUT reads of the draws.
//
// Resources are per (command, in-flight frame) because the compacted index
// buffer is rewritten every frame and must not alias a buffer a still
// executing frame may read.  They are built lazily and kept until the
// geometry content hash changes.  Commands whose index count exceeds
// FC_VULKAN_GEOM_LOD_MAX_INDEX fall back to the full draw to bound memory.
//
// Validation caveat: a fresh document's main draw list is often just the
// hidden nav cube (a tiny non-indexed list), so a nav-cube-only run exercises
// neither the document geometry nor the indexed path and is not evidence the
// feature works.  tools/fcprobe/vk_geomlod_probe.py asserts the LOD ran on a
// real indexed Part shape and documents the check in full.

#include "rendering/SoVulkanRenderBackend.h"
#include "rendering/SoVulkanRenderBackend/SoVulkanRenderBackendP.h"
#include "rendering/SoVulkanConfig.h"

#include <Inventor/errors/SoDebugError.h>

#include "vk_mem_alloc.h"

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

// Minimum projected triangle area (px^2) that survives.  1 px keeps the LOD
// visually faithful while dropping the sub-pixel filler that dominates a
// zoomed-out CAD mesh.  Resolved once in SoVulkanConfig (FC_VULKAN_GEOM_LOD_*).
float geometryLodMinAreaPixels()
{
  return SoVulkanConfig::get().geometryLod.minAreaPixels;
}

// Largest index count that gets a compacted buffer.  A pathologically large
// mesh would otherwise allocate indexCount * 4 bytes per in-flight slot.
// The default tracks the backend's MAX_VERTEX_COUNT so the huge CAD meshes
// this feature exists for are actually compacted: the previous 16M default
// silently excluded e.g. a 23.3M-element Voron face set, so the LOD did
// nothing on exactly the models that need it.  Override with
// FC_VULKAN_GEOM_LOD_MAX_INDEX (the compacted buffer costs 4 B/element per
// in-flight slot, e.g. 93 MB/slot at 23.3M).
uint32_t geometryLodMaxIndices()
{
  return SoVulkanConfig::get().geometryLod.maxIndices;
}

bool geometryLodEnabled()
{
  return SoVulkanConfig::get().geometryLod.enabled;
}

// Force the pre-pass on even when the camera is not moving.  A verification
// aid: it lets a static screenshot exercise the compacted draw so it can be
// diffed against the full draw.
bool geometryLodAlways()
{
  return SoVulkanConfig::get().geometryLod.always;
}

// Print the previous frame's survivor count per compacted command.  This is
// the only way to prove the compaction is correct on a real, large mesh: a
// nav-cube-only run says nothing.  With FC_VULKAN_GEOM_LOD_PIXELS=0 every
// triangle must survive (survivors == prims); with the default threshold a
// zoomed-out mesh must cull heavily.
bool geometryLodStats()
{
  return SoVulkanConfig::get().geometryLod.stats;
}

// Indexed-ness and element count of a command's triangle stream.  The IR emits
// Voron-class meshes as non-indexed triangle lists and SoBrepFaceSet as
// indexed, so the compaction handles both by treating non-indexed vertices as
// sequential indices.  Shared by isSubPixelEligible(), ensureSubPixelSlot()
// and recordGeometryLodPrepass() so the eligibility rule cannot drift between
// the three call sites.
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
  // The IR emits the Voron-class meshes as non-indexed triangle lists, so the
  // compaction must handle both forms.  Triangle lists always carry a multiple
  // of three elements; anything else is left to the full draw so compaction
  // can never change the visible set.
  const SubPixelElementForm form = subPixelElementForm(command);
  if (form.elements < 3 || (form.elements % 3) != 0) return false;
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
  // Non-indexed triangle lists are compacted by treating the vertices as
  // sequential indices; the output is always an index buffer, so the draw is
  // the same vkCmdDrawIndexedIndirect either way.
  const SubPixelElementForm form = subPixelElementForm(command);
  const bool indexed = form.indexed;
  const uint32_t elementCount = form.elements;
  if (elementCount == 0) return false;
  if (elementCount > geometryLodMaxIndices()) {
    // Log once per command so an oversized mesh does not silently lose the LOD
    // (which is indistinguishable from the feature working, but doing nothing).
    if (!entry.warnedGeomLodCap) {
      entry.warnedGeomLodCap = true;
      fprintf(stderr,
              "[GEOMLOD] command has %u elements > cap %u; geometry LOD "
              "skipped for it (raise FC_VULKAN_GEOM_LOD_MAX_INDEX)\n",
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

  // (Re)build when the slot is empty or the previous allocation was too small.
  // The caller has already invalidated the slots on a content change.
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
    // Host-visible: the command is 20 bytes and rewritten in place by the
    // compute atomic, so a device-local staging copy would only add a
    // synchronous queue drain on first use.
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

  // The input descriptors bind the whole (possibly shared) buffer; the shader
  // applies the per-command base offset from its push constants.  Explicit
  // ranges (base + slice) rather than VK_WHOLE_SIZE keep the range within the
  // device's maxStorageBufferRange when the shared block holds several
  // commands' geometry.
  const VkDeviceSize vertexRange = entry.vertexOffset +
    static_cast<VkDeviceSize>(command.geometry.vertexCount) *
      VULKAN_VERTEX_STRIDE;
  const VkDeviceSize outRange =
    static_cast<VkDeviceSize>(elementCount) * sizeof(uint32_t);

  // A single storage-buffer binding cannot span more than
  // maxStorageBufferRange.  A huge mesh (e.g. 23M vertices * 32 B ~= 745 MB)
  // can exceed it, and vkUpdateDescriptorSets() would then be invalid - the
  // pre-pass must fall back to the full draw for that command.  Log once per
  // command so the fallback is not silent (the whole point of this feature is
  // the huge meshes; a quiet skip looks like it worked).
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
        fprintf(stderr,
                "[GEOMLOD] command needs a %llu-byte storage range but the "
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
  // Binding 1 is only read for indexed geometry; a non-indexed list binds the
  // vertex buffer there (never sampled) so the set stays valid.
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

void
SoVulkanRenderBackend::recordGeometryLodPrepass(VkCommandBuffer cb,
                                                const SoDrawList & drawlist,
                                                const SoRenderParams & params)
{
  const int num = drawlist.getNumCommands();
  if (num == 0) return;

  const bool debug = COIN_VULKAN_ENV_FLAG("FC_VULKAN_BACKEND_DEBUG");

  // Dump the command list once per process.  An atomic exchange makes the
  // once-guard thread-safe (the prepass is single-threaded today, but the
  // backend's record path is not, and a racy latch could double- or never-print).
  static std::atomic<bool> dumpedCommands {false};
  if (debug && !dumpedCommands.exchange(true)) {
    for (int i = 0; i < num; ++i) {
      const SoRenderCommand & c = drawlist.getCommand(i);
      fprintf(stderr,
              "[GEOMLOD] cmd %d topo=%d vc=%u ic=%u pass=%d idx=%p\n",
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
  // The shader's cross product is twice the pixel area, so the pass-through
  // threshold is 2 * minArea.
  const float areaThreshold = 2.0f * geometryLodMinAreaPixels();

  uint32_t compacted = 0;
  uint32_t skipped = 0;
  uint32_t maxElements = 0;
  uint32_t maxPrims = 0;
  // maxVc/maxIc span every command but only feed the debug summary line below,
  // so they are accumulated in the main loop and only when it will print.
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

    // Report the PREVIOUS frame's result before the cursor is reset.  The
    // indirect buffer is host-visible/coherent and this slot's prior frame
    // has completed (frames-in-flight), so the read is valid.  Guarded so it
    // never runs in a normal session (a host read of GPU-written memory
    // stalls the frame).
    if (geometryLodStats() && s.readyFrame != 0) {
      void * mapped = nullptr;
      if (vmaMapMemory(this->vmaAllocator, s.indirectMemory, &mapped) ==
          VK_SUCCESS) {
        uint32_t survivors = 0;
        std::memcpy(&survivors, mapped, sizeof(uint32_t));
        vmaUnmapMemory(this->vmaAllocator, s.indirectMemory);
        // The shader appends INDICES, so the indirect indexCount is 3x the
        // surviving triangles.  Report triangles to compare with primCount:
        // at FC_VULKAN_GEOM_LOD_PIXELS=0 every triangle must survive
        // (survivorPrims == primCount); at the default threshold a zoomed-out
        // mesh must cull heavily.
        const uint32_t survivorPrims = survivors / 3u;
        const float culled = primCount
          ? 100.0f * (1.0f - static_cast<float>(survivorPrims) /
                               static_cast<float>(primCount))
          : 0.0f;
        fprintf(stderr,
                "[GEOMLOD] stats slot=%u prims=%u survivors=%u culled=%.1f%%\n",
                slot, primCount, survivorPrims, culled);
      }
    }

    // Reset the indirect indexCount (the append cursor) to zero.  Only the
    // first 4 bytes are touched, so the fixed fields stay intact.
    vkCmdFillBuffer(cb, s.indirectBuffer, 0, sizeof(uint32_t), 0);

    SoVulkanShared::memoryBarrier(
      cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    SubPixelPush pc {};
    // Same combined transform as the visual vertex shader: the shader applies
    // u_proj * u_view * model, so the CPU composes model * view * proj in
    // row-vector order and packs the rows as mat4 columns.
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

  // One barrier after every dispatch: compute writes become visible to the
  // indirect-command read and the index/vertex-input reads of the draws.
  SoVulkanShared::memoryBarrier(
    cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);

  if (debug) {
    fprintf(stderr, "[GEOMLOD] prepass slot=%u compacted=%u skipped=%u "
                    "threshold=%.2fpx2 maxPrims=%u maxVc=%u maxIc=%u\n",
            slot, compacted, skipped, areaThreshold, maxPrims, maxVc, maxIc);
  }
}

bool
SoVulkanRenderBackend::externalGeometryLodActive(const SoRenderParams & params) const
{
  // Geometry LOD only runs while the camera moves; otherwise the full-detail
  // draw is used.
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

  // The caller's external pass is a LOAD render pass and is already begun, so
  // neither the buffer -> image copies nor the compaction dispatches can be
  // recorded into it.  Record both into one backend-owned transient command
  // buffer now, before recordFrame() so the draw path sees the finalized
  // textures and the compacted slots; submitExternalPrepass() submits it after
  // the frame is recorded, which overlaps the CPU frame recording with the
  // previous GPU frame.
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

  // Record the copies and the dispatches into the transient buffer.  The
  // host-side finalize (view/sampler/descriptor creation + content stamp) is
  // deliberately deferred until after a successful vkEndCommandBuffer(): if
  // the buffer cannot be completed, pendingUploads must stay populated so the
  // caller's one-shot fallback still uploads the textures.  Finalizing first
  // would clear the list and leave the images empty.
  if (wantTextures) {
    this->recordPendingTextureUploadsInto(cb);
  }
  const double texEnd = timing ? SoVulkanShared::steadyNowMs() : 0.0;
  if (timing) timing->texMs = texEnd - recordT0;

  if (wantLod) {
    this->recordGeometryLodPrepass(cb, drawlist, params);
  }
  if (timing) timing->lodRecordMs = SoVulkanShared::steadyNowMs() - texEnd;

  if (vkEndCommandBuffer(cb) != VK_SUCCESS) {
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &cb);
    SoDebugError::postWarning("SoVulkanRenderBackend::beginExternalPrepass",
                              "external pre-pass transient buffer end failed; "
                              "the caller falls back to the one-shot texture "
                              "upload and the full-detail draw");
    return VK_NULL_HANDLE;
  }
  // The copies are now recorded, so the descriptor sets the draw path binds
  // can be created and the content identity stamped.
  if (wantTextures) {
    this->finalizePendingTextureUploads();
  }
  return cb;
}

bool
SoVulkanRenderBackend::submitExternalPrepass(VkCommandBuffer commandBuffer,
                                             ExternalFrameTiming * timing)
{
  if (commandBuffer == VK_NULL_HANDLE) return true;
  const double submitT0 = timing ? SoVulkanShared::steadyNowMs() : 0.0;
  VkSubmitInfo submit {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &commandBuffer;
  const bool submitted =
    vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
  // Host wait: the copies and the compacted writes must be complete and
  // visible before the caller submits its pass, and the caller's submission is
  // out of reach, so no semaphore can be threaded through it.  The queue
  // drains here; because the frame was already recorded above, only the
  // pre-pass itself is on the critical path.
  const bool waited = vkQueueWaitIdle(this->queue) == VK_SUCCESS;
  vkFreeCommandBuffers(this->device, this->commandPool, 1, &commandBuffer);
  if (timing) timing->lodMs = SoVulkanShared::steadyNowMs() - submitT0;
  if (!submitted || !waited) {
    // The copies in the transient buffer never completed, so the texture
    // entries finalizePendingTextureUploads() stamped still hold their
    // (empty) images.  Un-stamp them so prepareGeometryTextures() re-prepares
    // the upload on the next frame instead of sampling the empty image
    // forever.  The already-recorded frame cannot be repaired.
    for (const size_t index : this->finalizedTextureIndices) {
      if (index < this->textureCache.size()) {
        this->textureCache[index].pixelsKey = nullptr;
      }
    }
    SoDebugError::postWarning("SoVulkanRenderBackend::submitExternalPrepass",
                              "external pre-pass transient submit%s failed; "
                              "texture uploads will be retried next frame",
                              submitted ? " wait" : "");
  }
  this->finalizedTextureIndices.clear();
  return submitted && waited;
}
