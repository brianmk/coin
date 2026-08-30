// src/rendering/SoRTXRenderBackend/ngx_abi.h
//
// Minimal self-authored ABI declarations for the NVIDIA NGX (DLSS-RR) Vulkan
// runtime.  This is an interop-only interface declaration written from the
// published NGX API contract; it is NOT derived from NVIDIA's proprietary
// header files and is licensed MIT.  We deliberately do NOT vendor the
// NVIDIA-provided nvsdk_ngx*.h headers into the tree.
//
// The NGX Vulkan runtime (libnvidia-ngx.so.1) is a proprietary, closed-source
// binary loaded at runtime with dlopen().  It is never shipped with Coin.
// This header only reproduces the ABI (function-pointer typedefs, the POD
// enums/structs we touch, and the NVSDK_NGX_Parameter vtable interface we use
// to drive named parameters), which is all that is needed to call into it.
//
// Every value in this file is a declaration of an external ABI.  Keep the
// numeric values and field ordering EXACTLY in sync with the NVIDIA runtime;
// small changes here silently break the call ABI.

// SPDX-FileCopyrightText: Copyright (c) 2026 FreeCAD project contributors
// SPDX-License-Identifier: MIT

#ifndef COIN_SORTXRENDERBACKEND_NGX_ABI_H
#define COIN_SORTXRENDERBACKEND_NGX_ABI_H

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
  #define NGX_ABI_CONV __cdecl
  #define NGX_ABI_API  __declspec(dllimport)
#else
  #define NGX_ABI_CONV
  #define NGX_ABI_API
#endif

// Forward-declare the Vulkan handles we need in signatures.  The backend
// file includes the real Vulkan headers and passes VkInstance/VkDevice etc.
// by value, so these must remain pointer-width trivials regardless of the
// Vulkan typedef (VK_DEFINE_HANDLE -> pointer).
struct VkInstance_T;
typedef struct VkInstance_T * VkInstance;
struct VkPhysicalDevice_T;
typedef struct VkPhysicalDevice_T * VkPhysicalDevice;
struct VkDevice_T;
typedef struct VkDevice_T * VkDevice;
struct VkCommandBuffer_T;
typedef struct VkCommandBuffer_T * VkCommandBuffer;
struct VkBuffer_T;
typedef struct VkBuffer_T * VkBuffer;
struct VkImage_T;
typedef struct VkImage_T * VkImage;
struct VkImageView_T;
typedef struct VkImageView_T * VkImageView;

// SDK API version constant declared by the runtime's headers.  0x15 = NGX 1.5.
#define NVSDK_NGX_VERSION_API 0x15u

// -------------------------------------------------------------------------
// Core result / feature enums (values must match the NVIDIA runtime).
// -------------------------------------------------------------------------

//! NGX call/result code.  Success == 1; all failures share the high
//! fail-flag 0xBAD00000 so NVSDK_NGX_FAILED() can be a cheap mask test.
typedef enum NgxResult {
    NgxResult_Success = 0x1,
    NgxResult_Fail = 0xBAD00000,

    NgxResult_FAIL_FeatureNotSupported = NgxResult_Fail | 1,
    NgxResult_FAIL_PlatformError = NgxResult_Fail | 2,
    NgxResult_FAIL_FeatureAlreadyExists = NgxResult_Fail | 3,
    NgxResult_FAIL_FeatureNotFound = NgxResult_Fail | 4,
    NgxResult_FAIL_InvalidParameter = NgxResult_Fail | 5,
    NgxResult_FAIL_ScratchBufferTooSmall = NgxResult_Fail | 6,
    NgxResult_FAIL_NotInitialized = NgxResult_Fail | 7,
    NgxResult_FAIL_UnsupportedInputFormat = NgxResult_Fail | 8,
    NgxResult_FAIL_RWFlagMissing = NgxResult_Fail | 9,
    NgxResult_FAIL_MissingInput = NgxResult_Fail | 10,
    NgxResult_FAIL_UnableToInitializeFeature = NgxResult_Fail | 11,
    NgxResult_FAIL_OutOfDate = NgxResult_Fail | 12,
    NgxResult_FAIL_OutOfGPUMemory = NgxResult_Fail | 13,
    NgxResult_FAIL_UnsupportedFormat = NgxResult_Fail | 14,
    NgxResult_FAIL_UnableToWriteToAppDataPath = NgxResult_Fail | 15,
    NgxResult_FAIL_UnsupportedParameter = NgxResult_Fail | 16,
    NgxResult_FAIL_Denied = NgxResult_Fail | 17,
    NgxResult_FAIL_NotImplemented = NgxResult_Fail | 18,
} NgxResult;

#define NGX_ABI_SUCCEED(value) (((value) & 0xFFF00000) != NgxResult_Fail)
#define NGX_ABI_FAILED(value)  (((value) & 0xFFF00000) == NgxResult_Fail)

//! Feature identifier used in CreateFeature/EvaluateFeature/requirements.
typedef enum NgxFeature {
    NgxFeature_Reserved0 = 0,
    NgxFeature_SuperSampling = 1,        // DLSS-SR
    NgxFeature_InPainting = 2,
    NgxFeature_ImageSuperResolution = 3,
    NgxFeature_SlowMotion = 4,
    NgxFeature_VideoSuperResolution = 5,
    NgxFeature_ImageSignalProcessing = 9,
    NgxFeature_DeepResolve = 10,
    NgxFeature_FrameGeneration = 11,
    NgxFeature_DeepDVC = 12,
    NgxFeature_RayReconstruction = 13,  // DLSS-RR (denoiser target)
    NgxFeature_Count = 32764,
} NgxFeature;

//! Feature support query result bitmask (0 == fully supported).
typedef enum NgxFeatureSupportResult {
    NgxFeatureSupportResult_Supported = 0,
    NgxFeatureSupportResult_CheckNotPresent = 1,
    NgxFeatureSupportResult_DriverVersionUnsupported = 2,
    NgxFeatureSupportResult_AdapterUnsupported = 4,
    NgxFeatureSupportResult_OSVersionBelowMinimumSupported = 8,
    NgxFeatureSupportResult_NotImplemented = 16,
} NgxFeatureSupportResult;

//! Rendering engine classification for the project-ID identifier path.
typedef enum NgxEngineType {
    NgxEngineType_Custom = 0,
    NgxEngineType_Unreal,
    NgxEngineType_Unity,
    NgxEngineType_Omniverse,
    NgxEngineType_Count,
} NgxEngineType;

//! Vulkan resource kind (image view vs. buffer).
typedef enum NgxResourceVkType {
    NgxResourceVkType_ImageView = 0,
    NgxResourceVkType_Buffer = 1,
} NgxResourceVkType;

// -------------------------------------------------------------------------
// Feature-call common parameters carried by Init / requirement queries.
// -------------------------------------------------------------------------

//! Union used by the application identifier: either an NVIDIA App ID or a
//! project/engine descriptor.
typedef struct NgxProjectIdDescription {
    const char * ProjectId;
    NgxEngineType EngineType;
    const char * EngineVersion;
} NgxProjectIdDescription;

typedef enum NgxIdentifierType {
    NgxIdentifier_AppId = 0,
    NgxIdentifier_ProjectId = 1,
} NgxIdentifierType;

typedef struct NgxApplicationIdentifier {
    NgxIdentifierType IdentifierType;
    union {
        NgxProjectIdDescription ProjectDesc;
        unsigned long long ApplicationId;
    } v;
} NgxApplicationIdentifier;

//! A directory to scan for feature modules.
typedef struct NgxPathListInfo {
    const wchar_t * const * Path;
    int Length;
} NgxPathListInfo;

//! Info common to all features / init calls.
typedef struct NgxFeatureCommonInfo {
    NgxPathListInfo PathListInfo;
} NgxFeatureCommonInfo;

//! Discovery info for a requirements query (feature + app identity + paths).
typedef struct NgxFeatureDiscoveryInfo {
    unsigned int SDKVersion;
    NgxFeature FeatureID;
    NgxApplicationIdentifier Identifier;
    const wchar_t * ApplicationDataPath;
    const NgxFeatureCommonInfo * FeatureInfo;
} NgxFeatureDiscoveryInfo;

//! Requirements result for a feature.
typedef struct NgxFeatureRequirement {
    NgxFeatureSupportResult FeatureSupported;
    unsigned int MinHWArchitecture;
    char MinOSVersion[255];
} NgxFeatureRequirement;

// -------------------------------------------------------------------------
// Vulkan resource wrapper (image view or buffer) handed to EvaluateFeature.
// -------------------------------------------------------------------------

typedef struct NgxBufferInfoVk {
    VkBuffer Buffer;
    unsigned int SizeInBytes;
} NgxBufferInfoVk;

typedef struct NgxImageViewInfoVk {
    VkImageView ImageView;
    VkImage Image;
    // Opaque mirror of VkImageSubresourceRange (aspectMask + 4 uint32 word
    // fields).  We never dereference it in the buffer-only path; keeping the
    // exact bytes preserves the struct layout so image-view resources handed
    // to the runtime are ABI-correct.
    unsigned int SubresourceRange[6];
    unsigned int Format;   // VkFormat value, kept as a plain int
    unsigned int Width;
    unsigned int Height;
} NgxImageViewInfoVk;

typedef struct NgxResourceVk {
    union {
        NgxImageViewInfoVk ImageViewInfo;
        NgxBufferInfoVk BufferInfo;
    } Resource;
    NgxResourceVkType Type;
    bool ReadWrite;
} NgxResourceVk;

// Opaque D3D resource types that appear in the Set/Get overload set.  They are
// part of the vtable layout even though we never call those overloads, so the
// slot indices of the overloads below them stay identical to the runtime.
struct ID3D11Resource; typedef struct ID3D11Resource ID3D11Resource;
struct ID3D12Resource; typedef struct ID3D12Resource ID3D12Resource;

// -------------------------------------------------------------------------
// NVSDK_NGX_Parameter: a polymorphic named-parameter bag.  The runtime
// implements it as an abstract class with an index-vtable; we only need the
// decl so we can call ->Set()/->Get() through that vtable.  The overloads and
// their ORDER must match the NVIDIA ABI exactly (8 Set, 8 Get, Reset) or the
// calls land in the wrong vtable slot.
// -------------------------------------------------------------------------

typedef struct NgxParameter {
    virtual void NGX_ABI_CONV Set(const char * name, unsigned long long value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, float value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, double value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, unsigned int value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, int value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, ID3D11Resource * value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, ID3D12Resource * value) = 0;
    virtual void NGX_ABI_CONV Set(const char * name, void * value) = 0;

    virtual NgxResult NGX_ABI_CONV Get(const char * name, unsigned long long * value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, float * value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, double * value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, unsigned int * value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, int * value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, ID3D11Resource ** value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, ID3D12Resource ** value) const = 0;
    virtual NgxResult NGX_ABI_CONV Get(const char * name, void ** value) const = 0;

    virtual void NGX_ABI_CONV Reset() = 0;
} NgxParameter;

// Opaque feature handle returned by CreateFeature, passed to Evaluate.
typedef struct NgxHandle NgxHandle;

// Vulkan entry-point accessor typedefs, mirroring PFN_vkGetInstanceProcAddr /
// PFN_vkGetDeviceProcAddr.  The backend passes the real PFNs; the NVIDIA
// runtime stores and uses them to look up the handful of Vulkan functions it
// needs (e.g. for scratch/pipeline lowering).
typedef void * (* NgxVkGetInstanceProcAddr)(VkInstance instance, const char * name);
typedef void * (* NgxVkGetDeviceProcAddr)(VkDevice device, const char * name);
// -------------------------------------------------------------------------
// Vulkan entry points (resolved via dlsym at runtime by the loader).  Each
// typedef matches the runtime's exported function signature.
// -------------------------------------------------------------------------

typedef NgxResult (NGX_ABI_CONV * NgxFnInit)(
    unsigned long long applicationId,
    const wchar_t * applicationDataPath,
    VkInstance instance,
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    NgxVkGetInstanceProcAddr getInstanceProcAddr,
    NgxVkGetDeviceProcAddr getDeviceProcAddr,
    const NgxFeatureCommonInfo * featureInfo,
    unsigned int sdkVersion);

typedef NgxResult (NGX_ABI_CONV * NgxFnGetRequirements)(
    VkInstance instance,
    VkPhysicalDevice physicalDevice,
    const NgxFeatureDiscoveryInfo * discoveryInfo,
    NgxFeatureRequirement * outSupported);

typedef NgxResult (NGX_ABI_CONV * NgxFnAllocateParameters)(
    NgxParameter ** outParameters);

typedef NgxResult (NGX_ABI_CONV * NgxFnGetParameters)(
    NgxParameter ** outParameters);

typedef NgxResult (NGX_ABI_CONV * NgxFnDestroyParameters)(
    NgxParameter * parameters);

typedef NgxResult (NGX_ABI_CONV * NgxFnGetScratchBufferSize)(
    NgxFeature feature,
    const NgxParameter * parameters,
    size_t * outSizeInBytes);

typedef NgxResult (NGX_ABI_CONV * NgxFnCreateFeature1)(
    VkDevice device,
    VkCommandBuffer commandBuffer,
    NgxFeature feature,
    const NgxParameter * parameters,
    NgxHandle ** outHandle);

typedef NgxResult (NGX_ABI_CONV * NgxFnEvaluateFeature)(
    VkCommandBuffer commandBuffer,
    const NgxHandle * featureHandle,
    const NgxParameter * parameters,
    void * callback);

typedef NgxResult (NGX_ABI_CONV * NgxFnReleaseFeature)(
    NgxHandle * handle);

typedef NgxResult (NGX_ABI_CONV * NgxFnShutdown1)(
    VkDevice device);

// -------------------------------------------------------------------------
// Common parameter name constants (string literals; independent of ABI).
// -------------------------------------------------------------------------

#define NGX_PARAM_CREATION_NODE_MASK   "CreationNodeMask"
#define NGX_PARAM_VISIBILITY_NODE_MASK "VisibilityNodeMask"
#define NGX_PARAM_WIDTH                "Width"
#define NGX_PARAM_HEIGHT               "Height"
#define NGX_PARAM_OUTWIDTH             "OutWidth"
#define NGX_PARAM_OUTHEIGHT            "OutHeight"
#define NGX_PARAM_PERF_QUALITY         "PerfQualityValue"
#define NGX_PARAM_DLSS_CREATE_FLAGS    "DLSS.Feature.Create.Flags"
#define NGX_PARAM_DLSS_OUTPUT_SUBRECTS "DLSS.Enable.Output.Subrects"
#define NGX_PARAM_DLSS_DENOISE_MODE    "DLSS.Denoise.Mode"
#define NGX_PARAM_DLSS_ROUGHNESS_MODE  "DLSS.Roughness.Mode"

// Evaluate-time inputs.
#define NGX_PARAM_DLSSD_COLOR            "Dlss.Input.Color"
#define NGX_PARAM_DLSSD_ALBEDO           "RTXDI.Out.DiffuseAlbedo"
#define NGX_PARAM_DLSSD_MOTION_VECTORS   "Dlss.Input.MotionVectors"
#define NGX_PARAM_DLSSD_NORMALS          "Dlss.Input.Normals"
#define NGX_PARAM_DLSSD_HITDISTANCE      "Dlss.Input.HitDistance"
#define NGX_PARAM_DLSSD_OUTPUT           "Dlss.Output.Result"
#define NGX_PARAM_DLSS_RENDER_SUBRECT_X  "Dlss.Render.Subrect.OffsetX"
#define NGX_PARAM_DLSS_RENDER_SUBRECT_Y  "Dlss.Render.Subrect.OffsetY"
#define NGX_PARAM_DLSS_RENDER_SUBRECT_W  "Dlss.Render.Subrect.Width"
#define NGX_PARAM_DLSS_RENDER_SUBRECT_H  "Dlss.Render.Subrect.Height"

// DLSS-D (ray reconstruction) denoise mode: unified = albedo+normal guides.
#define NGX_DLSS_DENOISE_MODE_UNIFIED 0

#endif // COIN_SORTXRENDERBACKEND_NGX_ABI_H
