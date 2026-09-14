// src/rendering/SoRTXRenderBackend/ngx_loader.cpp
//
// Runtime loader for the NVIDIA NGX Vulkan library.  See ngx_loader.h.
//
// SPDX-FileCopyrightText: Copyright (c) 2026 FreeCAD project contributors
// SPDX-License-Identifier: MIT

#include "rendering/SoRTXRenderBackend/ngx_loader.h"

#include <Inventor/errors/SoDebugError.h>

#include <dlfcn.h>
#include <mutex>

namespace SoRTXBackend {

namespace {

std::once_flag g_loadOnce;
NgxFunctions g_funcs;
bool g_attempted = false;

template <typename Fn>
void
resolveSymbol(void * handle, const char * name, Fn & out)
{
    out = reinterpret_cast<Fn>(dlsym(handle, name));
}

void
loadNgx()
{
    void * handle = dlopen("libnvidia-ngx.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        // Quiet: absence of the NVIDIA runtime simply means no DLSS-RR.  The
        // caller decides whether to log at a debug level.
        g_attempted = true;
        return;
    }

    resolveSymbol(handle, "NVSDK_NGX_VULKAN_Init", g_funcs.init);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_GetFeatureRequirements",
                  g_funcs.getRequirements);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_AllocateParameters",
                  g_funcs.allocateParameters);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_GetParameters",
                  g_funcs.getParameters);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_DestroyParameters",
                  g_funcs.destroyParameters);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_GetScratchBufferSize",
                  g_funcs.getScratchBufferSize);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_CreateFeature1",
                  g_funcs.createFeature1);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_EvaluateFeature",
                  g_funcs.evaluateFeature);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_ReleaseFeature",
                  g_funcs.releaseFeature);
    resolveSymbol(handle, "NVSDK_NGX_VULKAN_Shutdown1",
                  g_funcs.shutdown1);

    if (!g_funcs.init || !g_funcs.getRequirements ||
        !g_funcs.allocateParameters || !g_funcs.createFeature1 ||
        !g_funcs.evaluateFeature) {
        SoDebugError::post("SoRTXBackend::ngxFunctions",
                           "NVIDIA NGX runtime loaded but required entry "
                           "points are missing");
    }
    g_attempted = true;
}

} // namespace

const NgxFunctions *
ngxFunctions()
{
    std::call_once(g_loadOnce, loadNgx);
    return &g_funcs;
}

void
ngxReload()
{
    std::call_once(g_loadOnce, [] {});
    // Already loaded: nothing to do in this minimal loader.
}

bool
ngxAvailable()
{
    (void)ngxFunctions();
    return g_funcs.init != nullptr &&
           g_funcs.getRequirements != nullptr &&
           g_funcs.allocateParameters != nullptr &&
           g_funcs.createFeature1 != nullptr &&
           g_funcs.evaluateFeature != nullptr;
}

} // namespace SoRTXBackend
