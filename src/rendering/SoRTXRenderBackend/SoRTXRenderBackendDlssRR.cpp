// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendDlssRR.cpp
//
// DLSS-RR (NVIDIA NGX Ray Reconstruction) denoiser backend.
//
// The path tracer writes device-local G-buffers:
//   - accumBuffer   (binding 4) rgba: rgb = radiance sum, a = sample count
//   - albedoBuffer  (binding 14) rgba: rgb = first-hit albedo
//   - normalBuffer  (binding 5)  rgba: rgb = world normal
//   - positionBuffer(binding 6)  rgba: rgb = world position, a = hit distance
//
// This backend runs the NVIDIA NGX Ray Reconstruction (feature 13) denoiser as
// a pure on-GPU Vulkan pass: it hands the device-local G-buffers to the NGX
// runtime through NgxResourceVk wrappers, evaluates on the one-shot command
// buffer, and writes the denoised RGBA into denoisedBuffer (present binding
// 5).  No host round-trip.
//
// Everything here is guarded by COIN_BUILD_DLSS_RR_DENOISER (build option,
// default OFF) AND at runtime by the NGX runtime being dlopen-able and a
// registered DLSS App ID (FC_RTX_DLSS_APPID).  When either gate fails the
// caller (createDenoiseBackend) degrades the backend to OIDN.
//
// SPDX-FileCopyrightText: Copyright (c) 2026 FreeCAD project contributors
// SPDX-License-Identifier: MIT

#include "rendering/SoRTXRenderBackend.h"
#include <Inventor/errors/SoDebugError.h>
#include <cstdio>
#include <cstring>
#include <string>

#include <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>

#include "rendering/SoRTXRenderBackend/ngx_abi.h"
#include "rendering/SoRTXRenderBackend/ngx_loader.h"

#if COIN_BUILD_DLSS_RR_DENOISER

using namespace SoRTXBackend;

namespace {

// Skip the first N accumulated frames before denoising; a barely-accumulated
// frame fed to RR produces a blurred/junk result.  Mirrors denoiseMinSamples.
constexpr uint32_t kDlssMinSamples = 8;

// Build a NgxResourceVk buffer wrapper around a device-local buffer.
NgxResourceVk
makeBufferResource(VkBuffer buffer, uint32_t sizeInBytes, bool readWrite)
{
  NgxResourceVk res{};
  res.Resource.BufferInfo.Buffer = buffer;
  res.Resource.BufferInfo.SizeInBytes = sizeInBytes;
  res.Type = NgxResourceVkType_Buffer;
  res.ReadWrite = readWrite;
  return res;
}

} // namespace

// -------------------------------------------------------------------------
// createDlssRrBackend
// -------------------------------------------------------------------------
// Initialize the NGX runtime, query the RR feature, allocate the parameter
// bag and scratch buffer, then CreateFeature1(RR).  Returns true only when
// ngxFeature is created and ready to evaluate.
bool
SoRTXRenderBackend::createDlssRrBackend()
{
  if (this->ngxFeature) return true;
  if (!this->deviceIsNvidia) return false;          // NGX RR is NVIDIA-only
  if (this->denoiseWidth == 0 || this->denoiseHeight == 0) return false;

  const char * appId = std::getenv("FC_RTX_DLSS_APPID");
  if (!appId || !*appId) {
    if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
      fprintf(stderr, "[DENOISE] DLSS-RR disabled: FC_RTX_DLSS_APPID not set\n");
    }
    return false;
  }
  this->ngxAppId = appId;

  const NgxFunctions * ngx = ngxFunctions();
  if (!ngx || !ngx->init || !ngx->getRequirements ||
      !ngx->allocateParameters || !ngx->createFeature1 ||
      !ngx->evaluateFeature) {
    if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
      fprintf(stderr, "[DENOISE] DLSS-RR disabled: NVIDIA NGX runtime "
                      "unavailable\n");
    }
    return false;
  }

  // Feature module search path: the directory containing the feature .so.
  // Without this the runtime scans the application directory only; pointing
  // it at the SDK module dir lets the feature module be found without copying
  // the closed-source .so next to the executable.
  std::wstring moduleDir;
  if (const char * mdir = std::getenv("FC_RTX_DLSS_MODULE_DIR")) {
    for (const char * s = mdir; *s; ++s) {
      moduleDir += static_cast<wchar_t>(static_cast<unsigned char>(*s));
    }
  }
  NgxFeatureCommonInfo fi{};
  const wchar_t * pathList[1];
  if (!moduleDir.empty()) {
    pathList[0] = moduleDir.c_str();
    fi.PathListInfo.Path = pathList;
    fi.PathListInfo.Length = 1;
  }
  const NgxFeatureCommonInfo * fiPtr = moduleDir.empty() ? nullptr : &fi;

  const std::wstring appDataPath = std::wstring(L"./");

  // The NGX runtime expects the Vulkan loader's proc-addr accessors so it can
  // look up the functions it needs.  Reinterpret-cast the real PFNs (whose
  // return type is PFN_vkVoidFunction) to the ABI typedefs whose return type is
  // plain void*; the two are ABI-identical on all supported platforms.
  const NgxVkGetInstanceProcAddr gipa =
    reinterpret_cast<NgxVkGetInstanceProcAddr>(vkGetInstanceProcAddr);
  const NgxVkGetDeviceProcAddr gdpa =
    reinterpret_cast<NgxVkGetDeviceProcAddr>(vkGetDeviceProcAddr);

  // Init.  The feature-info variant carries the module scan path.  NGX Init is
  // subject to a nondeterministic FAIL_OutOfDate race correlated with the
  // internal updater; retry once before giving up.
  NgxResult initRc = ngx->init(0, appDataPath.c_str(), this->instance,
                               this->physicalDevice, this->device,
                               gipa, gdpa, fiPtr, NVSDK_NGX_VERSION_API);
  if (initRc == NgxResult_FAIL_OutOfDate && !this->ngxRetriedInit) {
    this->ngxRetriedInit = true;
    initRc = ngx->init(0, appDataPath.c_str(), this->instance,
                       this->physicalDevice, this->device,
                       gipa, gdpa, fiPtr, NVSDK_NGX_VERSION_API);
  }
  if (NGX_ABI_FAILED(initRc) && initRc != NgxResult_FAIL_OutOfDate) {
    if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
      fprintf(stderr, "[DENOISE] DLSS-RR Init failed 0x%x\n", initRc);
    }
    return false;
  }

  // Query requirements (feature support + min HW arch).
  NgxFeatureDiscoveryInfo discovery{};
  discovery.SDKVersion = NVSDK_NGX_VERSION_API;
  discovery.FeatureID = NgxFeature_RayReconstruction;
  discovery.Identifier.IdentifierType = NgxIdentifier_ProjectId;
  discovery.Identifier.v.ProjectDesc.ProjectId = this->ngxAppId.c_str();
  discovery.Identifier.v.ProjectDesc.EngineType = NgxEngineType_Custom;
  discovery.Identifier.v.ProjectDesc.EngineVersion = "1.0";
  discovery.ApplicationDataPath = appDataPath.c_str();
  discovery.FeatureInfo = fiPtr;

  NgxFeatureRequirement req{};
  NgxResult reqRc = ngx->getRequirements(this->instance, this->physicalDevice,
                                         &discovery, &req);
  if (NGX_ABI_FAILED(reqRc) || req.FeatureSupported !=
        NgxFeatureSupportResult_Supported) {
    if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
      fprintf(stderr, "[DENOISE] DLSS-RR unsupported: rc=0x%x support=0x%x\n",
              reqRc, req.FeatureSupported);
    }
    return false;
  }

  // Parameter bag.
  NgxResult prc = ngx->allocateParameters(&this->ngxParams);
  if (NGX_ABI_FAILED(prc) || !this->ngxParams) return false;

  // Populate create params.
  ngxSetParam(this->ngxParams, NGX_PARAM_CREATION_NODE_MASK, (unsigned int)1);
  ngxSetParam(this->ngxParams, NGX_PARAM_VISIBILITY_NODE_MASK, (unsigned int)1);
  ngxSetParam(this->ngxParams, NGX_PARAM_WIDTH, this->denoiseWidth);
  ngxSetParam(this->ngxParams, NGX_PARAM_HEIGHT, this->denoiseHeight);
  ngxSetParam(this->ngxParams, NGX_PARAM_OUTWIDTH, this->denoiseWidth);
  ngxSetParam(this->ngxParams, NGX_PARAM_OUTHEIGHT, this->denoiseHeight);
  ngxSetParam(this->ngxParams, NGX_PARAM_PERF_QUALITY, (int)5); // DLAA 1:1
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_CREATE_FLAGS, (int)1 | 4);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_OUTPUT_SUBRECTS, (int)0);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_DENOISE_MODE, (int)0); // unified
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_ROUGHNESS_MODE, (unsigned int)2);

  // Scratch buffer size.
  size_t scratchBytes = 0;
  if (ngx->getScratchBufferSize &&
      NGX_ABI_SUCCEED(ngx->getScratchBufferSize(NgxFeature_RayReconstruction,
                                                this->ngxParams,
                                                &scratchBytes)) &&
      scratchBytes > 0) {
    if (!this->createDeviceLocalBuffer(
          scratchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          this->ngxScratch, this->ngxScratchMem)) {
      scratchBytes = 0; // continue without; feature may not need it
    }
    this->ngxScratchBytes = scratchBytes;
  }
  if (this->ngxScratchBytes) {
    ngxSetParam(this->ngxParams, "Dlss.Scratch.Buffer",
                (void *)this->ngxScratch);
    ngxSetParam(this->ngxParams, "Dlss.Scratch.SizeInBytes",
                (unsigned long long)this->ngxScratchBytes);
  }

  // Create the RR output buffer (device-local RGBA) that Evaluate writes.
  const VkDeviceSize outputBytes = static_cast<VkDeviceSize>(
    this->denoiseWidth) * this->denoiseHeight * 16;
  if (this->ngxOutputBuf == VK_NULL_HANDLE) {
    if (!this->createDeviceLocalBuffer(
          outputBytes,
          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          this->ngxOutputBuf, this->ngxOutputMem)) {
      return false;
    }
  }

  // Create the feature on a transient command buffer.
  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) return false;
  NgxResult crc = ngx->createFeature1(this->device, cmd,
                                      NgxFeature_RayReconstruction,
                                      this->ngxParams, &this->ngxFeature);
  this->releaseTransientCommandBuffer();
  if (NGX_ABI_FAILED(crc) || !this->ngxFeature) {
    if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
      fprintf(stderr, "[DENOISE] DLSS-RR CreateFeature1 failed 0x%x\n", crc);
    }
    return false;
  }

  this->ngxReady = true;
  if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
    fprintf(stderr, "[DENOISE] DLSS-RR backend ready (%ux%u)\n",
            this->denoiseWidth, this->denoiseHeight);
  }
  return true;
}

// -------------------------------------------------------------------------
// evaluateDlssRr
// -------------------------------------------------------------------------
// Bind the G-buffers as NgxResourceVk wrappers and run the RR denoiser on
// \a cmd (the one-shot trace command buffer, mid-recording).  Writes the
// denoised RGBA into this->denoisedBuffer so the present pass samples it.
void
SoRTXRenderBackend::evaluateDlssRr(VkCommandBuffer cmd)
{
  if (!this->ngxFeature || !this->ngxParams || !this->ngxReady) return;

  const NgxFunctions * ngx = ngxFunctions();
  if (!ngx || !ngx->evaluateFeature) return;

  if (this->denoiseMinSamples > 0 &&
      this->ptFrameIndex + 1 < kDlssMinSamples) {
    return; // not accumulated enough; leave raw / in-shader edge-stopped
  }

  const uint32_t width = this->denoiseWidth;
  const uint32_t height = this->denoiseHeight;
  const uint32_t bytes = width * height * 16;
  (void)bytes;

  if (this->accumBuffer == VK_NULL_HANDLE ||
      this->denoisedBuffer == VK_NULL_HANDLE) {
    return;
  }

  // Barrier: shader writes -> transfer/host for the RR pass to read.
  VkMemoryBarrier before{};
  before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0,
                       nullptr, 0, nullptr);

  // Bind the G-buffers as the RR inputs.  The accumulated color carries the
  // radiance sum (a = sample count); the albedo and normal buffers feed the
  // guide channels.  All are bound as device-local storage-buffer resources.
  NgxResourceVk color = makeBufferResource(this->accumBuffer, bytes, false);
  NgxResourceVk albedo = makeBufferResource(this->albedoBuffer, bytes, false);
  NgxResourceVk normal = makeBufferResource(this->normalBuffer, bytes, false);
  NgxResourceVk output = makeBufferResource(this->denoisedBuffer, bytes, true);

  ngxSetParam(this->ngxParams, NGX_PARAM_DLSSD_COLOR, color);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSSD_ALBEDO, albedo);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSSD_NORMALS, normal);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSSD_OUTPUT, output);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_RENDER_SUBRECT_X, 0);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_RENDER_SUBRECT_Y, 0);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_RENDER_SUBRECT_W, width);
  ngxSetParam(this->ngxParams, NGX_PARAM_DLSS_RENDER_SUBRECT_H, height);

  NgxResult r = ngx->evaluateFeature(cmd, this->ngxFeature, this->ngxParams,
                                     nullptr);
  if (NGX_ABI_FAILED(r)) {
    if (getenv("FC_VULKAN_PT_DENOISER_DEBUG")) {
      fprintf(stderr, "[DENOISE] DLSS-RR Evaluate failed 0x%x\n", r);
    }
    return;
  }

  // The RR pass wrote denoisedBuffer in place; present binding 5 samples it.
  this->denoiseResultReady = TRUE;
  this->convergeAfterDenoise();
  (void)cmd;
}

// -------------------------------------------------------------------------
// teardownDlssRrBackend
// -------------------------------------------------------------------------
void
SoRTXRenderBackend::teardownDlssRrBackend()
{
  const NgxFunctions * ngx = ngxFunctions();
  if (this->ngxFeature && ngx && ngx->releaseFeature) {
    ngx->releaseFeature(this->ngxFeature);
  }
  this->ngxFeature = nullptr;
  if (this->ngxParams && ngx && ngx->destroyParameters) {
    ngx->destroyParameters(this->ngxParams);
  }
  this->ngxParams = nullptr;
  if (this->ngxOutputBuf != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->ngxOutputBuf, this->allocator);
    vkFreeMemory(this->device, this->ngxOutputMem, this->allocator);
    this->ngxOutputBuf = VK_NULL_HANDLE;
    this->ngxOutputMem = VK_NULL_HANDLE;
  }
  if (this->ngxScratch != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->ngxScratch, this->allocator);
    vkFreeMemory(this->device, this->ngxScratchMem, this->allocator);
    this->ngxScratch = VK_NULL_HANDLE;
    this->ngxScratchMem = VK_NULL_HANDLE;
  }
  this->ngxScratchBytes = 0;
  this->ngxReady = false;
  this->ngxAppId.clear();
#if 0
  if (ngx && ngx->shutdown1) ngx->shutdown1(this->device);
#endif
}

#endif // COIN_BUILD_DLSS_RR_DENOISER
