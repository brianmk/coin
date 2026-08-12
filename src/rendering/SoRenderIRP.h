// src/rendering/SoRenderIRP.h

#ifndef COIN_SORENDERIRP_H
#define COIN_SORENDERIRP_H

#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/rendering/SoRenderIR.h>

#include <cstddef>
#include <memory>

class SoState;

/*!
  \class SoIRBuffer
  \brief Chunk-based CPU scratch allocator for per-frame geometry data.

  Allocations are stable: pointers remain valid until clear() is called.
  Growth allocates new chunks without moving old data.
*/
class SoIRBuffer {
public:
  SoIRBuffer();
  ~SoIRBuffer() = default;

  void clear();
  void reserve(size_t bytes);
  void * allocate(size_t bytes, size_t alignment = alignof(float));

  template <typename T>
  T * allocateArray(size_t count, size_t alignment = alignof(T)) {
    return static_cast<T *>(this->allocate(count * sizeof(T), alignment));
  }

  size_t size() const { return this->totalAllocated; }

private:
  static constexpr size_t MIN_CHUNK_SIZE = 1024 * 1024; // 1 MB
  struct Chunk {
    std::vector<uint8_t> data;
    size_t cursor = 0;
  };
  std::vector<std::unique_ptr<Chunk>> chunks;
  size_t totalAllocated = 0;
  size_t highWaterMark = 0;  // largest total allocation seen across frames
};

/*!
  \brief Compute the coarse/fine sort key used by SoDrawList::buildSortedOrder().
*/
uint64_t SoIRComputeSortKey(const SoRenderCommand & cmd,
                            uint32_t passOrderBits,
                            uint32_t depthBucket);

//! Dump a compact summary of the draw list to Coin's debug output.
void SoIRDumpSummary(const SoDrawList & drawlist);
//! Dump the first \a count render commands to Coin's debug output.
void SoIRDumpFirstN(const SoDrawList & drawlist, int count);
//! Return whether render-backend trace logging is enabled.
SbBool coin_render_ir_trace_enabled();

/*!
  \namespace SoRenderIR
  \brief Helper functions for converting Coin state and caches into render IR.
*/
namespace SoRenderIR {
//! Fill a material snapshot from the current Inventor traversal state.
void fillMaterialFromState(SoState * state, SoMaterialData & material,
                           int materialIndex = 0);
//! Copy the current texture image into action-owned frame storage.
void fillTextureFromState(SoState * state, SoIRRenderAction * action,
                          SoMaterialData & material);
//! Fill render-state fields from the current Inventor traversal state.
void fillRenderStateFromState(SoState * state, SoRenderState & renderState);
//! Complete blend state after material opacity has been captured.
void ensureMaterialBlendState(SoRenderState & renderState,
                              const SoMaterialData & material);
//! Extract the current lighting setup, append/deduplicate it, and return its handle.
SoLightingHandle fillLightingFromState(SoState * state, SoDrawList & drawlist);
//! Return whether the material should be treated as translucent.
bool isMaterialTransparent(const SoMaterialData & material);
}

#endif // COIN_SORENDERIRP_H
