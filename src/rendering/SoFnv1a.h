// src/rendering/SoFnv1a.h

#ifndef COIN_SOFNV1A_H
#define COIN_SOFNV1A_H

#include <cstdint>

// FNV-1a 64-bit mixing step shared by the Vulkan and RTX render backends'
// content hashes.  Each backend keeps its own higher-level sampling/hash
// strategy; only the single-step FNV mix (xor then multiply by the FNV
// prime) is common and was previously duplicated in both private headers.
namespace CoinRenderDetail {

inline void
fnvMix(uint64_t & hash, uint64_t value)
{
  hash ^= value;
  hash *= 1099511628211ULL;
}

} // namespace CoinRenderDetail

#endif // COIN_SOFNV1A_H
