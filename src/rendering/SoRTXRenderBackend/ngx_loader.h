// src/rendering/SoRTXRenderBackend/ngx_loader.h
//
// Loads the proprietary NVIDIA NGX Vulkan runtime (libnvidia-ngx.so.1) at
// runtime via dlopen() and resolves the entry points declared in ngx_abi.h.
// The runtime binary is closed-source and shipped by the NVIDIA driver, never
// by Coin; this loader only binds to it when present so the build has no
// link-time dependency and a missing library degrades gracefully.
//
// SPDX-FileCopyrightText: Copyright (c) 2026 FreeCAD project contributors
// SPDX-License-Identifier: MIT

#ifndef COIN_SORTXRENDERBACKEND_NGX_LOADER_H
#define COIN_SORTXRENDERBACKEND_NGX_LOADER_H

#include <cstddef>
#include <cstdint>

#include "rendering/SoRTXRenderBackend/ngx_abi.h"

namespace SoRTXBackend {

//! Resolved NGX runtime entry points.  Every pointer is null when the library
//! is not loaded or the symbol is absent; callers must guard each use.
struct NgxFunctions {
    NgxFnInit init = nullptr;
    NgxFnGetRequirements getRequirements = nullptr;
    NgxFnAllocateParameters allocateParameters = nullptr;
    NgxFnGetParameters getParameters = nullptr;
    NgxFnDestroyParameters destroyParameters = nullptr;
    NgxFnGetScratchBufferSize getScratchBufferSize = nullptr;
    NgxFnCreateFeature1 createFeature1 = nullptr;
    NgxFnEvaluateFeature evaluateFeature = nullptr;
    NgxFnReleaseFeature releaseFeature = nullptr;
    NgxFnShutdown1 shutdown1 = nullptr;
};

//! Singleton access to the loaded NGX runtime.  Loads and resolves the
//! symbols on first call (thread-safe).  Returns false (and logs a message)
//! when libnvidia-ngx.so.1 is unavailable or a required symbol is missing.
//! The returned pointer is valid for the life of the process.
const NgxFunctions * ngxFunctions();

//! Force a reload attempt (used by tests to emulate a runtime appearing).
//! Does nothing if already loaded.
void ngxReload();

//! True when the NGX runtime resolved all required entry points.
bool ngxAvailable();

// --- Named-parameter helpers -------------------------------------------------
// The NVSDK_NGX_Parameter free-function setters are NOT exported by the Linux
// runtime, so parameters must be driven through the object's index-vtable.
// These helpers route Set()/Get() to the correct overload by the C++ type.

inline void ngxSetParam(NgxParameter * p, const char * name, unsigned int value) {
    if (p) p->Set(name, value);
}
inline void ngxSetParam(NgxParameter * p, const char * name, int value) {
    if (p) p->Set(name, value);
}
inline void ngxSetParam(NgxParameter * p, const char * name, float value) {
    if (p) p->Set(name, value);
}
inline void ngxSetParam(NgxParameter * p, const char * name, unsigned long long value) {
    if (p) p->Set(name, value);
}
inline void ngxSetParam(NgxParameter * p, const char * name, void * value) {
    // Opaque/pointer-valued parameter (e.g. a VkBuffer cast to void*).  The
    // runtime keeps the pointer for the duration of the call.
    if (p) p->Set(name, value);
}
inline void ngxSetParam(NgxParameter * p, const char * name, NgxResourceVk value) {
    // The VK resource is a value struct passed through the void* overload; the
    // runtime keeps a pointer to a copy for the duration of the call.
    if (p) p->Set(name, &value);
}

// Read a parameter through the vtable; returns false on failure.
inline bool ngxGetParam(NgxParameter * p, const char * name, unsigned int * out) {
    return p && NGX_ABI_SUCCEED(p->Get(name, out));
}

} // namespace SoRTXBackend

#endif // COIN_SORTXRENDERBACKEND_NGX_LOADER_H
