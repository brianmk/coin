// src/rendering/SoFnv1a.h

#ifndef COIN_SOFNV1A_H
#define COIN_SOFNV1A_H

#include <cstdint>

// FNV-1a 64-bit mixing step for the Vulkan backend's content hashes;
// xor, then multiply by the FNV prime.
namespace CoinRenderDetail {

inline void
fnvMix(uint64_t & hash, uint64_t value)
{
  hash ^= value;
  hash *= 1099511628211ULL;
}

} // namespace CoinRenderDetail

#endif // COIN_SOFNV1A_H
