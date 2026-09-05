// src/rendering/SoRTXRenderBackend/SoRTXRenderBackendCore.cpp

// Split from the original monolithic SoRTXRenderBackend.cpp.  Contains the
// member functions for the "Core" concern of the Vulkan RTX backend.

#include "rendering/SoRTXRenderBackend.h"
#include <Inventor/errors/SoDebugError.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <rendering/SoRTXRenderBackend/SoRTXRenderBackendP.h>

using namespace SoRTXBackend;

// Wall-clock milliseconds for the FC_VULKAN_FRAME_TIMING breakdown.
static double vkNowMs()
{
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

SoRTXRenderBackend::SoRTXRenderBackend()
{
}

SoRTXRenderBackend::~SoRTXRenderBackend()
{
  if (this->isInitialized()) this->shutdown();
}

const char *
SoRTXRenderBackend::getName() const
{
  return "RTXRenderBackend";
}

void
SoRTXRenderBackend::setPathTracingEnabled(SbBool enabled)
{
  if (this->ptEnabled == enabled) return;
  this->ptEnabled = enabled;
  // Switching modes invalidates the accumulated image and any in-flight
  // progressive run.
  this->ptAccumulating = FALSE;
  this->ptStartLatch = FALSE;
  this->ptFrameIndex = 0;
  this->ptIdleFrames = 0;
  this->ptWasMoving = FALSE;
  this->ptDenoisePending = FALSE;
  this->ptConverged = FALSE;
  this->denoiseResultReady = FALSE;
  this->haveLastView = FALSE;
  this->haveLastCameraVersion = FALSE;
  this->lastCameraVersion = 0;
}

SbBool
SoRTXRenderBackend::getPathTracingEnabled(void) const
{
  return this->ptEnabled;
}

void
SoRTXRenderBackend::setViewMode(RtxViewMode mode)
{
  if (this->rtxViewMode == mode) return;
  this->rtxViewMode = mode;
  // A view-mode change invalidates any in-flight progressive run.
  this->ptAccumulating = FALSE;
  this->ptStartLatch = FALSE;
  this->ptFrameIndex = 0;
  this->ptIdleFrames = 0;
  this->ptWasMoving = FALSE;
  this->ptDenoisePending = FALSE;
  this->ptConverged = FALSE;
  this->denoiseResultReady = FALSE;
  this->haveLastView = FALSE;
  this->haveLastCameraVersion = FALSE;
  this->lastCameraVersion = 0;
}

SoRTXRenderBackend::RtxViewMode
SoRTXRenderBackend::getViewMode(void) const
{
  return this->rtxViewMode;
}

void
SoRTXRenderBackend::setEnvIntensity(const float intensity)
{
  this->envIntensity = intensity;
}

void
SoRTXRenderBackend::setEnvSunDir(const float x, const float y, const float z)
{
  this->envSunDir[0] = x;
  this->envSunDir[1] = y;
  this->envSunDir[2] = z;
}

void
SoRTXRenderBackend::setEnvSunColor(const float r, const float g, const float b)
{
  this->envSunColor[0] = r;
  this->envSunColor[1] = g;
  this->envSunColor[2] = b;
}

void
SoRTXRenderBackend::setEnvSunPower(const float power)
{
  this->envSunPower = power;
}

void
SoRTXRenderBackend::setEnvSkyBrightness(const float brightness)
{
  this->envSkyBrightness = brightness;
}

// Procedural environment/cubemap presets.  Each is a skin for the analytic
// sky (mode 0: an explicit top/bottom gradient overriding the viewport's
// background colors plus a sun) or a camera-centered room cove (mode 1: a
// colored floor, four walls and ceiling traced in the shader so the cubemap
// reads as a real scene like a desk / table / white lab).  They are defined
// here so the backend owns the palette and the GUI lists them by name without
// carrying data.  Keep in sync with the RTXFrameBlock envRoom members.
struct RtxEnvPreset {
  const char * name;
  int mode;              // 0 = sky gradient, 1 = room cove
  float skyTop[3];       // sky mode: zenith gradient color
  float skyBottom[3];    // sky mode: horizon gradient color
  float sunDir[3];
  float sunColor[3];
  float sunPower;
  float intensity;
  float skyBrightness;
  // Room cove (mode 1): colors + geometry (heights are camera-relative).
  float wallColor[3];
  float floorColor[3];
  float ceilColor[3];
  float roomHalfExtent;
  float roomFloorY;
  float roomCeilY;
};

namespace {
const RtxEnvPreset kRtxEnvPresets[] = {
  // -- Sky-gradient presets ----------------------------------------------
  { "Daylight", 0,
    {0.30f, 0.50f, 0.80f}, {0.85f, 0.88f, 0.90f},
    {0.35f, 0.80f, 0.25f}, {1.00f, 0.95f, 0.85f}, 20.0f, 0.45f, 1.0f,
    {0.75f, 0.76f, 0.78f}, {0.45f, 0.30f, 0.18f}, {0.90f, 0.90f, 0.90f},
    3.0f, -1.2f, 2.5f },
  { "Sunset", 0,
    {0.25f, 0.18f, 0.35f}, {0.95f, 0.55f, 0.30f},
    {0.55f, 0.25f, 0.45f}, {1.00f, 0.55f, 0.25f}, 12.0f, 0.40f, 1.0f,
    {0.75f, 0.76f, 0.78f}, {0.45f, 0.30f, 0.18f}, {0.90f, 0.90f, 0.90f},
    3.0f, -1.2f, 2.5f },
  { "Overcast", 0,
    {0.60f, 0.62f, 0.66f}, {0.82f, 0.83f, 0.85f},
    {0.20f, 0.90f, 0.15f}, {0.90f, 0.90f, 0.90f}, 6.0f, 0.32f, 1.0f,
    {0.75f, 0.76f, 0.78f}, {0.45f, 0.30f, 0.18f}, {0.90f, 0.90f, 0.90f},
    3.0f, -1.2f, 2.5f },
  { "Neutral Studio", 0,
    {0.55f, 0.58f, 0.62f}, {0.80f, 0.81f, 0.83f},
    {0.30f, 0.85f, 0.40f}, {1.00f, 1.00f, 1.00f}, 30.0f, 0.40f, 1.0f,
    {0.75f, 0.76f, 0.78f}, {0.45f, 0.30f, 0.18f}, {0.90f, 0.90f, 0.90f},
    3.0f, -1.2f, 2.5f },
  { "Night", 0,
    {0.02f, 0.03f, 0.06f}, {0.05f, 0.08f, 0.15f},
    {0.40f, 0.85f, 0.60f}, {0.55f, 0.65f, 0.95f}, 100.0f, 0.12f, 1.0f,
    {0.75f, 0.76f, 0.78f}, {0.45f, 0.30f, 0.18f}, {0.90f, 0.90f, 0.90f},
    3.0f, -1.2f, 2.5f },
  // -- Room-cove presets -------------------------------------------------
  // A wooden desk surface under a soft white room; the low floor plane reads
  // as the desk/table top.
  { "Desk", 1,
    {0.30f, 0.50f, 0.80f}, {0.85f, 0.88f, 0.90f},
    {0.30f, 0.45f, 0.30f}, {1.00f, 0.90f, 0.75f}, 20.0f, 0.55f, 1.0f,
    {0.78f, 0.80f, 0.82f}, {0.42f, 0.26f, 0.15f}, {0.93f, 0.93f, 0.94f},
    2.8f, -0.55f, 2.6f },
  { "Table", 1,
    {0.30f, 0.50f, 0.80f}, {0.85f, 0.88f, 0.90f},
    {0.32f, 0.48f, 0.28f}, {1.00f, 0.85f, 0.65f}, 18.0f, 0.50f, 1.0f,
    {0.80f, 0.82f, 0.84f}, {0.48f, 0.30f, 0.17f}, {0.94f, 0.94f, 0.95f},
    3.0f, -0.80f, 2.6f },
  // A bright, clean white lab/shop: all-white walls and ceiling with a pale
  // floor and a soft cool fill.
  { "White Lab", 1,
    {0.30f, 0.50f, 0.80f}, {0.85f, 0.88f, 0.90f},
    {0.20f, 0.50f, 0.35f}, {1.00f, 1.00f, 1.00f}, 10.0f, 0.55f, 1.0f,
    {0.88f, 0.90f, 0.92f}, {0.68f, 0.70f, 0.72f}, {0.94f, 0.95f, 0.96f},
    3.4f, -1.4f, 3.0f },
  // A pure white seamless background: every surface is near-white so objects
  // sit in a neutral studio with even ambient light.
  { "White Background", 1,
    {0.30f, 0.50f, 0.80f}, {0.85f, 0.88f, 0.90f},
    {0.20f, 0.40f, 0.40f}, {1.00f, 1.00f, 1.00f}, 8.0f, 0.45f, 1.0f,
    {0.92f, 0.92f, 0.92f}, {0.90f, 0.90f, 0.90f}, {0.93f, 0.93f, 0.93f},
    4.0f, -2.0f, 3.5f },
};
} // anonymous namespace

const char *
SoRTXRenderBackend::getEnvMapName(const int index)
{
  if (index < 0 || index >= getEnvMapCount()) return nullptr;
  return kRtxEnvPresets[index].name;
}

int
SoRTXRenderBackend::getEnvMapCount(void)
{
  return static_cast<int>(sizeof(kRtxEnvPresets) / sizeof(kRtxEnvPresets[0]));
}

void
SoRTXRenderBackend::setEnvMap(const int index)
{
  if (index == this->envMapId) return;
  // Any environment override invalidates the accumulated image (the sky and
  // its contribution change), so drop back to a fresh run like a view-mode
  // change does.
  this->ptAccumulating = FALSE;
  this->ptStartLatch = FALSE;
  this->ptFrameIndex = 0;
  this->ptIdleFrames = 0;
  this->ptWasMoving = FALSE;
  this->ptDenoisePending = FALSE;
  this->ptConverged = FALSE;
  this->denoiseResultReady = FALSE;
  this->envMapId = index;
  if (index < 0 || index >= getEnvMapCount()) {
    // No (or an invalid) environment preset: the environment is disabled and
    // the viewport gradient is used.  The env member fields must be cleared
    // here -- they are read in updatePathTracingState()'s background-change
    // test alongside the viewport gradient, so a stale intensity/sun/sky from
    // a previously selected preset would keep comparing non-equal to the
    // cleared values and fire backgroundChanged on every frame, resetting a
    // converged progressive run back to the raw preview after every denoise.
    this->envIntensity = 0.0f;
    this->envSunPower = 1.0f;
    this->envSkyBrightness = 1.0f;
    for (int i = 0; i < 3; ++i) {
      this->envSunDir[i] = 0.0f;
      this->envSunColor[i] = 0.0f;
      this->envSkyTop[i] = 0.0f;
      this->envSkyBottom[i] = 0.0f;
      this->envWallColor[i] = 0.0f;
      this->envFloorColor[i] = 0.0f;
      this->envCeilColor[i] = 0.0f;
    }
    this->envMapMode = 0;
    this->envRoomHalfExtent = 0.0f;
    this->envRoomFloorY = 0.0f;
    this->envRoomCeilY = 0.0f;
    return;
  }
  const RtxEnvPreset & p = kRtxEnvPresets[index];
  this->envIntensity = p.intensity;
  this->envSunPower = p.sunPower;
  this->envSkyBrightness = p.skyBrightness;
  for (int i = 0; i < 3; ++i) {
    this->envSunDir[i] = p.sunDir[i];
    this->envSunColor[i] = p.sunColor[i];
    this->envSkyTop[i] = p.skyTop[i];
    this->envSkyBottom[i] = p.skyBottom[i];
    this->envWallColor[i] = p.wallColor[i];
    this->envFloorColor[i] = p.floorColor[i];
    this->envCeilColor[i] = p.ceilColor[i];
  }
  this->envMapMode = p.mode == 1 ? 1 : 0;
  this->envRoomHalfExtent = p.roomHalfExtent;
  this->envRoomFloorY = p.roomFloorY;
  this->envRoomCeilY = p.roomCeilY;
}

int
SoRTXRenderBackend::getEnvMap(void) const
{
  return this->envMapId;
}

void
SoRTXRenderBackend::setPathTracingStart(SbBool start)
{
  if (!this->ptEnabled) {
    this->emitLog("setPathTracingStart ignored: path tracing is disabled");
    return;
  }
  if (start) {
    // Latch: the next frame resets the accumulation and starts a fresh
    // progressive run (even if the camera changed since the last frame).
    this->ptStartLatch = TRUE;
  }
  else {
    this->ptStartLatch = FALSE;
    this->ptAccumulating = FALSE;
    this->ptIdleFrames = 0;
  }
}

SbBool
SoRTXRenderBackend::getPathTracingActive(void) const
{
  return this->ptEnabled && this->ptAccumulating;
}

SbBool
SoRTXRenderBackend::getPathTracingRefining(void) const
{
  // Request continuous frames while working toward a converged image: while
  // accumulating, and during the short post-move settle window (ptIdleFrames
  // below the settle threshold) so the auto-restart has frames to count.
  // After convergence ptIdleFrames is saturated at ptSettleFrames, so this
  // reads FALSE and the viewport can go idle.
  // The single-sample AO and Environment previews never accumulate: they
  // update on demand (camera/scene sensors) like the raster viewport, so
  // they must NOT keep the surface busy-looping.
  if (this->rtxViewMode == RtxViewMode::RtxModeAmbientOcclusion ||
      this->rtxViewMode == RtxViewMode::RtxModeEnvironment) return FALSE;
  return this->ptEnabled &&
    (this->ptAccumulating || this->ptIdleFrames < this->ptSettleFrames);
}

uint32_t
SoRTXRenderBackend::getPathTracingSampleCount(void) const
{
  return this->ptAccumulating ? this->ptFrameIndex + 1 : 0;
}

void
SoRTXRenderBackend::setPathTracingBounces(const uint32_t bounces)
{
  this->ptMaxBounces = std::max(1u, std::min(16u, bounces));
}

void
SoRTXRenderBackend::setPathTracingSettleFrames(const uint32_t frames)
{
  this->ptSettleFrames = std::max(1u, std::min(120u, frames));
}

void
SoRTXRenderBackend::setPathTracingMaxSamples(const uint32_t samples)
{
  this->ptMaxSamples = std::max(1u, std::min(4096u, samples));
}

void
SoRTXRenderBackend::setPathTracingDenoiseEnabled(SbBool enabled)
{
  this->ptDenoise = enabled;
}

void
SoRTXRenderBackend::setDenoiserFilter(const char * denoiser)
{
  if (!denoiser || denoiser[0] == '\0') return;
  // Map the user-facing name onto the private enum; an unknown name leaves
  // the current choice alone so a stale pref value never silently disables
  // the denoiser.  createDenoiseBackend() resolves the kind from this store
  // on the next buffer (re)creation.
  if (std::strcmp(denoiser, "rtx") == 0) this->denoiseKindPref = DenoiseRtx;
  else if (std::strcmp(denoiser, "oidn") == 0) this->denoiseKindPref = DenoiseOidn;
  else if (std::strcmp(denoiser, "fsr") == 0) this->denoiseKindPref = DenoiseFsr;
  else if (std::strcmp(denoiser, "none") == 0) this->denoiseKindPref = DenoiseNone;
  else return;
  this->denoiseKind = this->denoiseKindPref;
  this->denoiseKindDirty = true;
  this->denoiseKindExplicit = true;
}

void
SoRTXRenderBackend::setDenoiserScale(const float scale)
{
  // Clamp to a sane range.  1.0 == native resolution (no upscale).  A factor
  // > 1 runs the denoiser at a reduced internal resolution and the present
  // pass bilinearly upscales it (the shader branch is only taken for scale
  // >= 1.5, so values in (1, 1.5) degrade to native).
  const float s = std::max(1.0f, std::min(8.0f, scale));
  if (s == this->denoiseScale) return;
  this->denoiseScale = s;
  // The denoiser buffers (host staging + device output) are sized from
  // denoiseWidth/Height which createPathTracingBuffers computes from this
  // scale on the next (re)create; force a backend recreate so it picks the
  // new resolution up.
  this->denoiseKindDirty = true;
}

bool
SoRTXRenderBackend::probeComputeQueue(void)
{
  this->computeQueue = VK_NULL_HANDLE;
  this->computeQueueCount = 0;
  this->hasComputeQueue = false;
  if (this->device == VK_NULL_HANDLE || this->physicalDevice == VK_NULL_HANDLE) {
    return false;
  }
  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(this->physicalDevice, &familyCount,
                                            nullptr);
  if (familyCount == 0) return false;
  std::vector<VkQueueFamilyProperties> fams(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(this->physicalDevice, &familyCount,
                                            fams.data());
  // A dedicated compute family (or an extended graphics-family entry) was
  // requested at device creation via setQueueCreateInfoModifier; the family +
  // queue index selected there come through the device context.  UINT32_MAX
  // means none was requested: report unavailable.
  if (this->computeQueueFamilyIndex < static_cast<uint32_t>(fams.size())) {
    const VkQueueFamilyProperties & fp = fams[this->computeQueueFamilyIndex];
    this->computeQueueCount = fp.queueCount;
    if (fp.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      this->computeQueue = VK_NULL_HANDLE;
      vkGetDeviceQueue(this->device, this->computeQueueFamilyIndex,
                       this->computeQueueIndex, &this->computeQueue);
      this->hasComputeQueue = (this->computeQueue != VK_NULL_HANDLE);
    }
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr,
            "[RTDBG] computeCaps family=%u idx=%u req=%d computeQueue=%d "
            "computeCount=%u flags=0x%x\n",
            this->computeQueueFamilyIndex, this->computeQueueIndex,
            this->computeQueueFamilyIndex != ~0u ? 1 : 0,
            this->hasComputeQueue ? 1 : 0, this->computeQueueCount,
            this->computeQueueFamilyIndex < static_cast<uint32_t>(fams.size())
              ? fams[this->computeQueueFamilyIndex].queueFlags : 0u);
  }
  return this->hasComputeQueue;
}

SbBool
SoRTXRenderBackend::initialize(const SoRenderBackendInitParams & params)
{
  if (this->isInitialized()) return TRUE;

  this->setInitParams(params);
  const auto * deviceContext =
    static_cast<const SoVulkanDeviceContext *>(params.userData);
  if (!deviceContext || deviceContext->instance == VK_NULL_HANDLE ||
      deviceContext->physicalDevice == VK_NULL_HANDLE ||
      deviceContext->device == VK_NULL_HANDLE ||
      deviceContext->graphicsQueue == VK_NULL_HANDLE) {
    this->emitError(
      "SoRTXRenderBackend requires a SoVulkanDeviceContext in "
      "SoRenderBackendInitParams::userData");
    return FALSE;
  }
  if (deviceContext->apiVersion < VK_API_VERSION_1_2) {
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "SoRTXRenderBackend requires a Vulkan 1.2+ device (device "
                  "context reports API %u.%u.%u)",
                  VK_API_VERSION_MAJOR(deviceContext->apiVersion),
                  VK_API_VERSION_MINOR(deviceContext->apiVersion),
                  VK_API_VERSION_PATCH(deviceContext->apiVersion));
    this->emitError(buf);
    return FALSE;
  }

  this->instance = deviceContext->instance;
  this->physicalDevice = deviceContext->physicalDevice;
  this->device = deviceContext->device;
  this->queue = deviceContext->graphicsQueue;
  this->queueFamilyIndex = deviceContext->graphicsQueueFamilyIndex;
  this->allocator = deviceContext->allocator;
  this->memProps.setDevice(this->physicalDevice);

  // The async-compute queue requested at device creation (see the widget's
  // setQueueCreateInfoModifier).  probeComputeQueue() retrieves the handle
  // from this family + queue index.  UINT32_MAX family = none requested.
  this->computeQueueFamilyIndex = deviceContext->computeQueueFamilyIndex;
  this->computeQueueIndex = deviceContext->computeQueueIndex;

  // Acquire a compute queue for the optional async-compute path, and report
  // the capability so a probe/check can verify.
  this->probeComputeQueue();

  // Cache the physical-device identity so the denoiser selection can gate the
  // CUDA/OptiX path on NVIDIA hardware (see SoRTXRenderBackend.h).
  VkPhysicalDeviceProperties devProps {};
  vkGetPhysicalDeviceProperties(this->physicalDevice, &devProps);
  this->deviceVendorID = devProps.vendorID;
  this->deviceID = devProps.deviceID;
  this->deviceIsNvidia = (devProps.vendorID == 0x10DE /* NVIDIA */);

  // Query the device UUID (Vulkan 1.1 VkPhysicalDeviceIDProperties) so the
  // CUDA context can be bound to the same GPU on multi-GPU machines.
  this->haveDeviceUUID = false;
  VkPhysicalDeviceIDProperties idProps {};
  idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 idProps2 {};
  idProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  idProps2.pNext = &idProps;
  vkGetPhysicalDeviceProperties2(this->physicalDevice, &idProps2);
  // VkPhysicalDeviceIDProperties always carries the deviceUUID array (filled
  // by the driver once the struct is chained); retain it for CUDA matching.
  // Some drivers expose the array but leave it zero-filled, so only trust it
  // when it actually contains a non-zero identifier.
  std::memcpy(this->deviceUUID, idProps.deviceUUID, sizeof(this->deviceUUID));
  bool deviceUUIDNonZero = false;
  for (const uint8_t byte : this->deviceUUID) {
    if (byte != 0) {
      deviceUUIDNonZero = true;
      break;
    }
  }
  this->haveDeviceUUID = deviceUUIDNonZero;

  // Probe the created device's optional capability extensions once, so the
  // shader/builder paths can select the best available technique at run time
  // instead of querying the extension list every frame.  Only recordings --
  // the features themselves must have been requested by the embedding app
  // (QuarterVulkanWidget) when the device was created.
  {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(this->physicalDevice, nullptr,
                                         &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    if (extCount > 0) {
      vkEnumerateDeviceExtensionProperties(this->physicalDevice, nullptr,
                                           &extCount, exts.data());
    }
    const auto hasExt = [&exts](const char * name) {
      for (const auto & e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
      }
      return false;
    };
    this->hasPositionFetch =
      hasExt("VK_KHR_ray_tracing_position_fetch");
    this->hasOpacityMicromap = hasExt("VK_EXT_opacity_micromap");
    this->hasNvCluster = hasExt("VK_NV_cluster_acceleration_structure");
    this->hasNvPartitioned =
      hasExt("VK_NV_partitioned_acceleration_structure");
    this->hasNvLinearSweptSpheres =
      hasExt("VK_NV_ray_tracing_linear_swept_spheres");
    char capsBuf[192];
    std::snprintf(
      capsBuf, sizeof(capsBuf),
      "[RTDBG] caps positionFetch=%d opacityMicromap=%d "
      "nvCluster=%d nvPartitioned=%d nvLinearSweptSpheres=%d",
      this->hasPositionFetch ? 1 : 0, this->hasOpacityMicromap ? 1 : 0,
      this->hasNvCluster ? 1 : 0, this->hasNvPartitioned ? 1 : 0,
      this->hasNvLinearSweptSpheres ? 1 : 0);
    fprintf(stderr, "%s\n", capsBuf);
  }

  // The system loader only exports core entry points; resolve the ray
  // tracing KHR functions per-device.  Failing here means the device is
  // missing the acceleration-structure/ray-query extensions (or the loader
  // version cannot reach them), and the RT backend cannot function.
  this->vkDestroyAccelerationStructureKHR =
    loadDispatch<PFN_vkDestroyAccelerationStructureKHR>(
      vkGetDeviceProcAddr(this->device, "vkDestroyAccelerationStructureKHR"));
  this->vkGetAccelerationStructureBuildSizesKHR =
    loadDispatch<PFN_vkGetAccelerationStructureBuildSizesKHR>(
      vkGetDeviceProcAddr(this->device, "vkGetAccelerationStructureBuildSizesKHR"));
  this->vkCreateAccelerationStructureKHR =
    loadDispatch<PFN_vkCreateAccelerationStructureKHR>(
      vkGetDeviceProcAddr(this->device, "vkCreateAccelerationStructureKHR"));
  this->vkCmdBuildAccelerationStructuresKHR =
    loadDispatch<PFN_vkCmdBuildAccelerationStructuresKHR>(
      vkGetDeviceProcAddr(this->device, "vkCmdBuildAccelerationStructuresKHR"));
  this->vkGetAccelerationStructureDeviceAddressKHR =
    loadDispatch<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
      vkGetDeviceProcAddr(this->device, "vkGetAccelerationStructureDeviceAddressKHR"));
  this->vkCmdWriteAccelerationStructuresPropertiesKHR =
    loadDispatch<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(
      vkGetDeviceProcAddr(this->device, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
  this->vkCmdCopyAccelerationStructureKHR =
    loadDispatch<PFN_vkCmdCopyAccelerationStructureKHR>(
      vkGetDeviceProcAddr(this->device, "vkCmdCopyAccelerationStructureKHR"));
  if (!this->vkDestroyAccelerationStructureKHR ||
      !this->vkGetAccelerationStructureBuildSizesKHR ||
      !this->vkCreateAccelerationStructureKHR ||
      !this->vkCmdBuildAccelerationStructuresKHR ||
      !this->vkGetAccelerationStructureDeviceAddressKHR ||
      !this->vkCmdWriteAccelerationStructuresPropertiesKHR ||
      !this->vkCmdCopyAccelerationStructureKHR) {
    this->emitError(
      "failed to resolve ray tracing KHR entry points; the device or "
      "loader does not provide VK_KHR_acceleration_structure");
    this->shutdown();
    return FALSE;
  }

  // The ray tracing pipeline (VK_KHR_ray_tracing_pipeline) entry points
  // power the shader binding table dispatch.
  this->vkCreateRayTracingPipelinesKHR =
    loadDispatch<PFN_vkCreateRayTracingPipelinesKHR>(
      vkGetDeviceProcAddr(this->device, "vkCreateRayTracingPipelinesKHR"));
  this->vkGetRayTracingShaderGroupHandlesKHR =
    loadDispatch<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
      vkGetDeviceProcAddr(this->device, "vkGetRayTracingShaderGroupHandlesKHR"));
  this->vkCmdTraceRaysKHR =
    loadDispatch<PFN_vkCmdTraceRaysKHR>(
      vkGetDeviceProcAddr(this->device, "vkCmdTraceRaysKHR"));
  if (!this->vkCreateRayTracingPipelinesKHR ||
      !this->vkGetRayTracingShaderGroupHandlesKHR ||
      !this->vkCmdTraceRaysKHR) {
    this->emitError(
      "failed to resolve VK_KHR_ray_tracing_pipeline entry points; the "
      "device or loader does not provide the ray tracing pipeline");
    this->shutdown();
    return FALSE;
  }

  // Dispatch mode: the SBT pipeline is opt-in (FC_VULKAN_RT_SBT=1); the
  // default ray-query compute path avoids a hang in NVIDIA driver 610.x
  // where triangle hit-group execution stalls the GPU.
  this->useSbtPipeline = COIN_VULKAN_ENV_FLAG("FC_VULKAN_RT_SBT");

  // All entry points are resolved from here on.  Mark the backend
  // initialized before creating resources so that a failure in any
  // create*() below runs the full (null-tolerant) shutdown() cleanup
  // instead of leaking every handle created so far.
  this->setInitialized(TRUE);

  // Query the pipeline properties needed for the SBT record layout, plus the
  // acceleration-structure properties for the scratch buffer alignment
  // (VUID-vkCmdBuildAccelerationStructuresKHR-scratchData-*): the scratch
  // device address must be aligned to
  // minAccelerationStructureScratchOffsetAlignment, which the buffer's own
  // memory requirements do not guarantee.
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps {};
  rtProps.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps {};
  asProps.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  rtProps.pNext = &asProps;
  VkPhysicalDeviceProperties2 rtProps2 {};
  rtProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  rtProps2.pNext = &rtProps;
  vkGetPhysicalDeviceProperties2(this->physicalDevice, &rtProps2);
  this->asScratchAlignment =
    std::max<VkDeviceSize>(asProps.minAccelerationStructureScratchOffsetAlignment, 1u);
  this->sbtGroupHandleSize = rtProps.shaderGroupHandleSize;
  this->sbtGroupBaseAlignment = std::max(rtProps.shaderGroupBaseAlignment, 1u);
  // The record stride must satisfy both the handle alignment and the
  // base alignment, because every strided region address has to be a
  // multiple of shaderGroupBaseAlignment (VUID-vkCmdTraceRaysKHR-*).
  const uint32_t alignment = std::max({
    rtProps.shaderGroupHandleAlignment,
    rtProps.shaderGroupBaseAlignment,
    1u});
  this->sbtRecordSize = this->sbtGroupHandleSize;
  this->sbtRecordSize += alignment - 1;
  this->sbtRecordSize -= this->sbtRecordSize % alignment;

  if (!this->createDescriptorSetLayout()) {
    this->emitError("failed to create RT descriptor set layout");
    this->shutdown();
    return FALSE;
  }
  if (!this->createDescriptorPool()) {
    this->emitError("failed to create RT descriptor pool");
    this->shutdown();
    return FALSE;
  }
  if (!this->createShaderModules()) {
    this->emitError("failed to create RT shader modules");
    this->shutdown();
    return FALSE;
  }
  if (!this->createPipelines()) {
    this->emitError("failed to create ray tracing pipeline");
    this->shutdown();
    return FALSE;
  }
  if (!this->createFrameBuffer()) {
    this->emitError("failed to create RT frame uniform buffer");
    this->shutdown();
    return FALSE;
  }

  // Optional path tracing tuning (kept out of the public API for now).
  if (const char * bounces = getenv("FC_VULKAN_PT_BOUNCES")) {
    const int value = std::atoi(bounces);
    if (value >= 1 && value <= 16) {
      this->ptMaxBounces = static_cast<uint32_t>(value);
    }
  }
  if (const char * settle = getenv("FC_VULKAN_PT_SETTLE")) {
    const int value = std::atoi(settle);
    if (value >= 1 && value <= 120) {
      this->ptSettleFrames = static_cast<uint32_t>(value);
    }
  }
  if (const char * maxsamples = getenv("FC_VULKAN_PT_MAXSAMPLES")) {
    const int value = std::atoi(maxsamples);
    if (value >= 1 && value <= 100000) {
      this->ptMaxSamples = static_cast<uint32_t>(value);
    }
  }
  // Adaptive sampling tuning (see PathTrace.glsl u_adaptive).
  if (const char * adaptive = getenv("FC_VULKAN_PT_ADAPTIVE")) {
    this->ptAdaptiveEnabled = std::atoi(adaptive) != 0 ? TRUE : FALSE;
  }
  if (const char * minsamples = getenv("FC_VULKAN_PT_MIN_SAMPLES")) {
    const int value = std::atoi(minsamples);
    if (value >= 1 && value <= 256) {
      this->ptAdaptiveMinSamples = static_cast<uint32_t>(value);
    }
  }
  if (const char * threshold = getenv("FC_VULKAN_PT_THRESHOLD")) {
    const float value = static_cast<float>(std::atof(threshold));
    if (value > 0.0f && value <= 1.0f) {
      this->ptAdaptiveThreshold = value;
    }
  }
  if (const char * stopfraction = getenv("FC_VULKAN_PT_STOP_FRACTION")) {
    const float value = static_cast<float>(std::atof(stopfraction));
    // 0 disables the fraction-based auto-stop (run to the sample cap only).
    if (value >= 0.0f && value <= 1.0f) {
      this->ptAdaptiveStopFraction = value;
    }
  }
  // Firefly rejection: replace samples far brighter than the pixel's running
  // mean (outlier spikes) with that mean.  FC_VULKAN_PT_FIREFLY is the
  // standard-deviation multiplier; 0 disables it (on by default at 5.0, the
  // member default) so the override only needs to set 0 to turn it off.
  if (const char * firefly = getenv("FC_VULKAN_PT_FIREFLY")) {
    const float value = static_cast<float>(std::atof(firefly));
    if (value >= 0.0f) {
      this->ptFireflySigma = value;
    }
  }
  // Temporal reprojection: carry converged samples across camera moves.
  if (const char * temporal = getenv("FC_VULKAN_PT_TEMPORAL")) {
    this->ptTemporalEnabled = std::atoi(temporal) != 0 ? TRUE : FALSE;
  }

  this->setInitialized(TRUE);
  this->emitLog("initialized (Vulkan ray tracing)");
  return TRUE;
}

// --- Frame recording ------------------------------------------------------

VkCommandBuffer
SoRTXRenderBackend::beginTransientCommandBuffer()
{
  // Persistent transient pool + one-shot command buffer for the AS phase,
  // allocated once instead of per frame.  The caller submits and waits the
  // buffer every frame; resetting it here is safe because the submission is
  // provably complete (vkQueueWaitIdle) by the time the next frame begins.
  if (this->transientPool == VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo pci {};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = this->queueFamilyIndex;
    if (vkCreateCommandPool(this->device, &pci, this->allocator,
                            &this->transientPool) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = this->transientPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(this->device, &ai,
                                 &this->transientCommandBuffer) !=
        VK_SUCCESS) {
      vkDestroyCommandPool(this->device, this->transientPool, this->allocator);
      this->transientPool = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
  }
  vkResetCommandBuffer(this->transientCommandBuffer, 0);
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(this->transientCommandBuffer, &bi) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  return this->transientCommandBuffer;
}

void
SoRTXRenderBackend::releaseTransientCommandBuffer()
{
  if (this->transientCommandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(this->device, this->transientPool, 1,
                         &this->transientCommandBuffer);
    this->transientCommandBuffer = VK_NULL_HANDLE;
  }
  if (this->transientPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->transientPool, this->allocator);
    this->transientPool = VK_NULL_HANDLE;
  }
}

// --- Lifecycle ------------------------------------------------------------

void
SoRTXRenderBackend::shutdown()
{
  if (!this->isInitialized()) return;

  vkQueueWaitIdle(this->queue);

  // The queue is idle: drain both deferred-destruction batches.
  this->flushPendingDestroys();
  this->flushPendingDestroys();

  this->invalidateCache();
  this->freePendingStagingDestroys();

  if (this->tlas != VK_NULL_HANDLE) {
    vkDestroyAccelerationStructureKHR(this->device, this->tlas,
                                      this->allocator);
    this->tlas = VK_NULL_HANDLE;
  }
  if (this->tlasBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->tlasBuffer, this->allocator);
    this->tlasBuffer = VK_NULL_HANDLE;
  }
  if (this->tlasMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->tlasMemory, this->allocator);
    this->tlasMemory = VK_NULL_HANDLE;
  }
  if (this->instanceBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->instanceBuffer, this->allocator);
    this->instanceBuffer = VK_NULL_HANDLE;
  }
  if (this->instanceMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->instanceMemory, this->allocator);
    this->instanceMemory = VK_NULL_HANDLE;
  }
  this->instanceBufferCapacity = 0;
  this->tlasSize = 0;
  if (this->scratchBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->scratchBuffer, this->allocator);
    this->scratchBuffer = VK_NULL_HANDLE;
  }
  if (this->scratchMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->scratchMemory, this->allocator);
    this->scratchMemory = VK_NULL_HANDLE;
  }
  this->scratchSize = 0;
  this->scratchAddress = 0;
  if (this->storageImage != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, this->storageImageView, this->allocator);
    vkDestroyImage(this->device, this->storageImage, this->allocator);
    vkFreeMemory(this->device, this->storageImageMemory, this->allocator);
    this->storageImage = VK_NULL_HANDLE;
    this->storageImageView = VK_NULL_HANDLE;
    this->storageImageMemory = VK_NULL_HANDLE;
  }
  if (this->presentSampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->presentSampler, this->allocator);
    this->presentSampler = VK_NULL_HANDLE;
  }
  if (this->accumBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->accumBuffer, this->allocator);
    this->accumBuffer = VK_NULL_HANDLE;
  }
  if (this->accumMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->accumMemory, this->allocator);
    this->accumMemory = VK_NULL_HANDLE;
  }
  if (this->normalBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->normalBuffer, this->allocator);
    this->normalBuffer = VK_NULL_HANDLE;
  }
  if (this->normalMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->normalMemory, this->allocator);
    this->normalMemory = VK_NULL_HANDLE;
  }
  if (this->positionBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->positionBuffer, this->allocator);
    this->positionBuffer = VK_NULL_HANDLE;
  }
  if (this->positionMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->positionMemory, this->allocator);
    this->positionMemory = VK_NULL_HANDLE;
  }
  if (this->sumSqBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->sumSqBuffer, this->allocator);
    this->sumSqBuffer = VK_NULL_HANDLE;
  }
  if (this->sumSqMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->sumSqMemory, this->allocator);
    this->sumSqMemory = VK_NULL_HANDLE;
  }
  if (this->activeCounterBuffer != VK_NULL_HANDLE) {
    if (this->activeCounterMapped != nullptr) {
      vkUnmapMemory(this->device, this->activeCounterMemory);
      this->activeCounterMapped = nullptr;
    }
    vkDestroyBuffer(this->device, this->activeCounterBuffer, this->allocator);
    this->activeCounterBuffer = VK_NULL_HANDLE;
  }
  if (this->activeCounterMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->activeCounterMemory, this->allocator);
    this->activeCounterMemory = VK_NULL_HANDLE;
  }
  if (this->accumHistoryBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->accumHistoryBuffer, this->allocator);
    this->accumHistoryBuffer = VK_NULL_HANDLE;
  }
  if (this->accumHistoryMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->accumHistoryMemory, this->allocator);
    this->accumHistoryMemory = VK_NULL_HANDLE;
  }
  if (this->sumSqHistoryBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->sumSqHistoryBuffer, this->allocator);
    this->sumSqHistoryBuffer = VK_NULL_HANDLE;
  }
  if (this->sumSqHistoryMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->sumSqHistoryMemory, this->allocator);
    this->sumSqHistoryMemory = VK_NULL_HANDLE;
  }
  if (this->positionHistoryBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->positionHistoryBuffer,
                    this->allocator);
    this->positionHistoryBuffer = VK_NULL_HANDLE;
  }
  if (this->positionHistoryMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->positionHistoryMemory, this->allocator);
    this->positionHistoryMemory = VK_NULL_HANDLE;
  }
  // The screen-space motion-vector G-buffer (read by the denoiser readback)
  // is part of the same PT buffer pool, so it must be destroyed here too.
  if (this->motionBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->motionBuffer, this->allocator);
    this->motionBuffer = VK_NULL_HANDLE;
  }
  if (this->motionMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->motionMemory, this->allocator);
    this->motionMemory = VK_NULL_HANDLE;
  }
  this->ptHistoryValid = FALSE;
  this->ptReprojectFrame = FALSE;
  if (this->positionMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->positionMemory, this->allocator);
    this->positionMemory = VK_NULL_HANDLE;
  }
  this->ptBufferWidth = 0;
  this->ptBufferHeight = 0;
  this->ptAccumulating = FALSE;
  this->ptFrameIndex = 0;
  this->destroyDenoiser();
  this->flushPendingDestroys();
  this->flushPendingDestroys();
  if (this->materialBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->materialBuffer, this->allocator);
    vkFreeMemory(this->device, this->materialMemory, this->allocator);
    this->materialBuffer = VK_NULL_HANDLE;
    this->materialMemory = VK_NULL_HANDLE;
    this->materialMapped = nullptr;
  }
  this->materialCount = 0;
  this->materialBufferBytes = 0;
  if (this->frameBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->frameBuffer, this->allocator);
    vkFreeMemory(this->device, this->frameMemory, this->allocator);
    this->frameBuffer = VK_NULL_HANDLE;
    this->frameMemory = VK_NULL_HANDLE;
    this->frameMapped = nullptr;
  }
  if (this->presentFrameBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->presentFrameBuffer, this->allocator);
    vkFreeMemory(this->device, this->presentFrameMemory, this->allocator);
    this->presentFrameBuffer = VK_NULL_HANDLE;
    this->presentFrameMemory = VK_NULL_HANDLE;
    this->presentFrameMapped = nullptr;
  }
  if (this->presentPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->presentPipeline, this->allocator);
    this->presentPipeline = VK_NULL_HANDLE;
  }
  if (this->rtPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->rtPipeline, this->allocator);
    this->rtPipeline = VK_NULL_HANDLE;
  }
  if (this->computePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->computePipeline, this->allocator);
    this->computePipeline = VK_NULL_HANDLE;
  }
  if (this->denoiseDownsamplePipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(this->device, this->denoiseDownsamplePipeline,
                      this->allocator);
    this->denoiseDownsamplePipeline = VK_NULL_HANDLE;
  }
  this->denoiseDownsampleDescriptorSet = VK_NULL_HANDLE;
  this->denoiseDownsampleValid = false;
  if (this->presentPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->presentPipelineLayout,
                            this->allocator);
    this->presentPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->rtPipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->rtPipelineLayout,
                            this->allocator);
    this->rtPipelineLayout = VK_NULL_HANDLE;
  }
  if (this->denoiseDownsamplePipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->denoiseDownsamplePipelineLayout,
                            this->allocator);
    this->denoiseDownsamplePipelineLayout = VK_NULL_HANDLE;
  }
  if (this->presentVertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->presentVertexModule,
                          this->allocator);
    this->presentVertexModule = VK_NULL_HANDLE;
  }
  if (this->presentFragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->presentFragmentModule,
                          this->allocator);
    this->presentFragmentModule = VK_NULL_HANDLE;
  }
  if (this->pathTraceModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->pathTraceModule,
                          this->allocator);
    this->pathTraceModule = VK_NULL_HANDLE;
  }
  if (this->denoiseDownsampleModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->denoiseDownsampleModule,
                          this->allocator);
    this->denoiseDownsampleModule = VK_NULL_HANDLE;
  }
  if (this->raygenModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->raygenModule, this->allocator);
    this->raygenModule = VK_NULL_HANDLE;
  }
  if (this->missModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->missModule, this->allocator);
    this->missModule = VK_NULL_HANDLE;
  }
  if (this->shadowMissModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->shadowMissModule,
                          this->allocator);
    this->shadowMissModule = VK_NULL_HANDLE;
  }
  if (this->closestHitModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->closestHitModule,
                          this->allocator);
    this->closestHitModule = VK_NULL_HANDLE;
  }
  if (this->shadowClosestHitModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->shadowClosestHitModule,
                          this->allocator);
    this->shadowClosestHitModule = VK_NULL_HANDLE;
  }
  if (this->sbtBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->sbtBuffer, this->allocator);
    this->sbtBuffer = VK_NULL_HANDLE;
  }
  if (this->sbtMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->sbtMemory, this->allocator);
    this->sbtMemory = VK_NULL_HANDLE;
  }
  this->sbtRecordSize = 32;
  this->sbtBaseOffset = 0;
  if (this->normalPoolBuffer != VK_NULL_HANDLE) {
    this->normalPoolMapped = nullptr;
    vkDestroyBuffer(this->device, this->normalPoolBuffer, this->allocator);
    this->normalPoolBuffer = VK_NULL_HANDLE;
    vkFreeMemory(this->device, this->normalPoolMemory, this->allocator);
    this->normalPoolMemory = VK_NULL_HANDLE;
  }
  this->normalPoolCapacity = 0;
  this->normalPoolUsed = 0;
  if (this->neePoolBuffer != VK_NULL_HANDLE) {
    this->neePoolMapped = nullptr;
    vkDestroyBuffer(this->device, this->neePoolBuffer, this->allocator);
    this->neePoolBuffer = VK_NULL_HANDLE;
    vkFreeMemory(this->device, this->neePoolMemory, this->allocator);
    this->neePoolMemory = VK_NULL_HANDLE;
  }
  this->neePoolCapacity = 0;
  this->neePoolUsed = 0;
  this->neePoolCount = 0;
  if (this->descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(this->device, this->descriptorPool,
                            this->allocator);
    this->descriptorPool = VK_NULL_HANDLE;
  }
  if (this->rtSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->rtSetLayout,
                                 this->allocator);
    this->rtSetLayout = VK_NULL_HANDLE;
  }
  if (this->presentSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->presentSetLayout,
                                 this->allocator);
    this->presentSetLayout = VK_NULL_HANDLE;
  }
  if (this->denoiseDownsampleSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->denoiseDownsampleSetLayout,
                                 this->allocator);
    this->denoiseDownsampleSetLayout = VK_NULL_HANDLE;
  }
  if (this->offscreenFramebuffer != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(this->device, this->offscreenFramebuffer,
                         this->allocator);
    this->offscreenFramebuffer = VK_NULL_HANDLE;
  }
  if (this->offscreenRenderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(this->device, this->offscreenRenderPass,
                        this->allocator);
    this->offscreenRenderPass = VK_NULL_HANDLE;
  }
  this->releaseTransientCommandBuffer();
  this->offscreenColorImage = VK_NULL_HANDLE;
  this->offscreenColorView = VK_NULL_HANDLE;
  this->rtDescriptorSets[0] = VK_NULL_HANDLE;
  this->rtDescriptorSets[1] = VK_NULL_HANDLE;
  this->presentDescriptorSets[0] = VK_NULL_HANDLE;
  this->presentDescriptorSets[1] = VK_NULL_HANDLE;
  // The sets are invalid until updateDescriptors() rewrites them in the next
  // engine generation; descriptorSetIndex is intentionally NOT reset here, so
  // the first (possibly non-dirty) frame must repopulate its torn set.
  this->rtSetValid[0] = false;
  this->rtSetValid[1] = false;
  this->presentSetValid[0] = false;
  this->presentSetValid[1] = false;

  this->instance = VK_NULL_HANDLE;
  this->physicalDevice = VK_NULL_HANDLE;
  this->device = VK_NULL_HANDLE;
  this->queue = VK_NULL_HANDLE;
  this->allocator = nullptr;

  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

SbBool
SoRTXRenderBackend::render(const SoDrawList & drawlist,
                           const SoRenderParams & params)
{
  if (!this->isInitialized()) {
    this->emitError("render called before backend initialization");
    return FALSE;
  }
  this->ptLastFrame = params.frame;
  if (!params.renderTarget) {
    this->emitError(
      "render called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  this->debugValidateDrawList(drawlist);
  this->flushPendingDestroys();

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  // Offscreen path: single color attachment render pass + framebuffer.
  // The attachment is the swapchain/MSAA color image, so the render pass
  // and the present pipeline must both use the target's sample count
  // (VUID-VkFramebufferCreateInfo-renderPass-04553 and
  // VUID-VkGraphicsPipelineCreateInfo-renderPass-06082).  Both are cached
  // per target identity so this path does not recreate them every frame.
  const bool targetChanged =
    this->offscreenRenderPass == VK_NULL_HANDLE ||
    this->offscreenColorImage != target->colorImage ||
    this->offscreenColorView != target->colorImageView ||
    this->offscreenColorFormat != target->colorFormat ||
    this->offscreenSampleCount != target->sampleCount ||
    this->offscreenDepthImage != target->depthImage ||
    this->offscreenDepthView != target->depthImageView ||
    this->offscreenExtent.width != target->extent.width ||
    this->offscreenExtent.height != target->extent.height;
  if (targetChanged) {
    if (this->offscreenFramebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(this->device, this->offscreenFramebuffer,
                           this->allocator);
      this->offscreenFramebuffer = VK_NULL_HANDLE;
    }
    if (this->offscreenRenderPass != VK_NULL_HANDLE) {
      // The previous render() completed (it waits idle before returning),
      // so the old pass cannot be referenced by anything still pending.
      vkDestroyRenderPass(this->device, this->offscreenRenderPass,
                          this->allocator);
      this->offscreenRenderPass = VK_NULL_HANDLE;
    }

    VkAttachmentDescription attachment {};
    attachment.format = target->colorFormat;
    attachment.samples = target->sampleCount;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = target->colorLayout;
    attachment.finalLayout = target->colorLayout;

    VkAttachmentReference colorRef {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // The traced scene depth is written here (PresentFragment.glsl sets
    // gl_FragDepth from the first-bounce hit position) so the raster
    // composite overlay that runs afterwards (BRep edge lines, navigation
    // cube) can depth-test against it and cull hidden edges.  Without a depth
    // attachment the write is discarded and overlay edges show through faces.
    const bool hasDepth = target->depthImageView != VK_NULL_HANDLE &&
                          target->depthFormat != VK_FORMAT_UNDEFINED;
    VkAttachmentDescription depthAttachment {};
    VkAttachmentReference depthRef {};
    uint32_t attachmentCount = 1;
    if (hasDepth) {
      depthAttachment.format = target->depthFormat;
      depthAttachment.samples = target->sampleCount;
      depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      depthAttachment.initialLayout = target->depthLayout;
      depthAttachment.finalLayout = target->depthLayout;
      depthRef.attachment = 1;
      depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachmentCount = 2;
    }

    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;
    VkRenderPassCreateInfo rpCI {};
    // C99 compound literals ((Type[]){...}) are invalid in C++, so build the
    // render-pass attachment array on the stack instead of using one on the
    // branch (the framebuffer code below reuses the name 'attachments').
    VkAttachmentDescription renderPassAttachments[2] = {
      attachment,
      depthAttachment
    };
    rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = attachmentCount;
    rpCI.pAttachments = hasDepth ? renderPassAttachments : &attachment;
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    if (vkCreateRenderPass(this->device, &rpCI, this->allocator,
                           &this->offscreenRenderPass) != VK_SUCCESS) {
      this->emitError("failed to create RT render pass");
      return FALSE;
    }

    VkFramebufferCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass = this->offscreenRenderPass;
    fci.attachmentCount = attachmentCount;
    const VkImageView attachments[] = {target->colorImageView,
                                       target->depthImageView};
    fci.pAttachments = attachments;
    fci.width = target->extent.width;
    fci.height = target->extent.height;
    fci.layers = 1;
    if (vkCreateFramebuffer(this->device, &fci, this->allocator,
                            &this->offscreenFramebuffer) != VK_SUCCESS) {
      vkDestroyRenderPass(this->device, this->offscreenRenderPass,
                          this->allocator);
      this->offscreenRenderPass = VK_NULL_HANDLE;
      this->emitError("failed to create RT framebuffer");
      return FALSE;
    }

    this->offscreenColorImage = target->colorImage;
    this->offscreenColorView = target->colorImageView;
    this->offscreenColorFormat = target->colorFormat;
    this->offscreenSampleCount = target->sampleCount;
    this->offscreenDepthImage = target->depthImage;
    this->offscreenDepthView = target->depthImageView;
    this->offscreenExtent = target->extent;
  }
  const VkRenderPass renderPass = this->offscreenRenderPass;
  const VkFramebuffer framebuffer = this->offscreenFramebuffer;

  // Persistent transient command buffer for the AS phase (BLAS/TLAS builds
  // and buffer copies), which is not allowed inside a render pass.  It is
  // recorded first, then the render pass begins and only the trace/present
  // work is recorded inside it.
  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    this->emitError("failed to allocate RT command buffer");
    return FALSE;
  }

  // Phase 1: acceleration structures (outside the render pass).
  const bool asOk =
    this->recordAccelerationStructures(drawlist, params, *target, cmd);
  bool traceOk = false;

  if (asOk) {
    VkRenderPassBeginInfo rpbi {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = framebuffer;
    rpbi.renderArea.offset = {0, 0};
    rpbi.renderArea.extent = target->extent;
    // Depth attachment (when present) is LOAD_OP_CLEAR so the composite
    // overlay's LESS_OR_EQUAL depth test starts from a clean far plane
    // (background = 1.0); the present pass then writes the real scene depth
    // for traced pixels so hidden edges/navcube are culled.
    const bool hasDepthClear =
      target->depthImageView != VK_NULL_HANDLE &&
      target->depthFormat != VK_FORMAT_UNDEFINED;
    VkClearValue clearValues[2] {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    rpbi.clearValueCount = hasDepthClear ? 2u : 1u;
    rpbi.pClearValues = clearValues;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    // The attachment is loaded, not cleared at pass begin: the clear is
    // issued below scoped to the requested viewport region (matching the
    // raster backend's recordClear()).  Clearing the whole attachment here
    // would overwrite content outside a sub-region viewport.
    if (params.flags & SO_PARAM_CLEAR_WINDOW) {
      VkClearAttachment clear {};
      clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      clear.colorAttachment = 0;
      clear.clearValue.color.float32[0] = params.clearColor[0];
      clear.clearValue.color.float32[1] = params.clearColor[1];
      clear.clearValue.color.float32[2] = params.clearColor[2];
      clear.clearValue.color.float32[3] = params.clearColor[3];

      const SbVec2s & origin = params.viewport.getViewportOriginPixels();
      const SbVec2s & size = params.viewport.getViewportSizePixels();
      const int32_t x0 = std::max(0, static_cast<int32_t>(origin[0]));
      const int32_t y0 = std::max(
        0, static_cast<int32_t>(target->extent.height) -
             static_cast<int32_t>(origin[1]) -
             static_cast<int32_t>(size[1]));
      const int32_t x1 = std::min(static_cast<int32_t>(target->extent.width),
                                  static_cast<int32_t>(origin[0]) +
                                    static_cast<int32_t>(size[0]));
      const int32_t y1 = std::min(
        static_cast<int32_t>(target->extent.height),
        static_cast<int32_t>(target->extent.height) -
          static_cast<int32_t>(origin[1]));
      if (x1 > x0 && y1 > y0) {
        VkClearRect rect {};
        rect.rect.offset = {x0, y0};
        rect.rect.extent = {static_cast<uint32_t>(x1 - x0),
                            static_cast<uint32_t>(y1 - y0)};
        rect.baseArrayLayer = 0;
        rect.layerCount = 1;
        vkCmdClearAttachments(cmd, 1, &clear, 1, &rect);
      }
    }

    // Phase 2: trace + present (inside the render pass).
    traceOk =
      this->recordTraceAndPresent(params, *target, cmd, renderPass);

    vkCmdEndRenderPass(cmd);
  }
  vkEndCommandBuffer(cmd);

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  const VkResult submitResult = vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE);
  const VkResult waitResult = submitResult == VK_SUCCESS
    ? vkQueueWaitIdle(this->queue) : VK_SUCCESS;
  if (submitResult == VK_SUCCESS && waitResult == VK_SUCCESS) {
    this->updateAdaptiveStats();
    // Denoise must read the just-written accumBuffer before the ping-pong
    // swap for the next frame's reprojection (see renderExternal).
    this->updateDenoise();
    this->swapPathTracingHistory();
  }
  if (getenv("FC_VULKAN_RT_DEBUG")) {
    fprintf(stderr, "[RTDBG] submit=%d wait=%d asOk=%d traceOk=%d\n",
            static_cast<int>(submitResult), static_cast<int>(waitResult),
            asOk ? 1 : 0, traceOk ? 1 : 0);
  }
  const bool submitted = submitResult == VK_SUCCESS && waitResult == VK_SUCCESS;

  // Staging buffers are only referenced by the private submission; release
  // them after it provably completed (or never ran).  The transient command
  // buffer stays pooled for the next frame.
  if (submitted) {
    this->freePendingStagingDestroys();
    // The frame is fully complete here: shrink any BLAS that was built with
    // ALLOW_COMPACTION and has not yet been compacted (FC_VULKAN_AS_COMPACT).
    this->compactPendingBlases();
  }

  if (!asOk || !traceOk || !submitted) {
    this->emitError("render: RT frame failed");
    return FALSE;
  }
  return TRUE;
}

void
SoRTXRenderBackend::dumpStorageImageIfRequested()
{
  const char * path = getenv("FC_VULKAN_PT_DUMP");
  if (!path) return;
  if (this->storageImage == VK_NULL_HANDLE ||
      this->storageWidth == 0 || this->storageHeight == 0) return;

  // FC_VULKAN_PT_DUMP_EVERY=N: dump every N frames (frame-index-suffixed),
  // letting one run capture the whole progressive sequence.  Otherwise dump
  // one frame at FC_VULKAN_PT_DUMP_FRAME (default = the sample cap).
  const char * everystr = getenv("FC_VULKAN_PT_DUMP_EVERY");
  const char * atstr = getenv("FC_VULKAN_PT_DUMP_FRAME");
  const uint32_t dumpAt =
    atstr ? static_cast<uint32_t>(std::atoi(atstr)) : this->ptMaxSamples;
  bool ok = false;
  if (everystr) {
    const uint32_t every = static_cast<uint32_t>(std::atoi(everystr));
    if (every == 0 || this->ptFrameIndex % every != 0) return;
    ok = true;
  } else if (this->ptFrameIndex != dumpAt || this->ptDumpDone) {
    return;
  }
  if (ok) this->ptDumpDone = TRUE;

  char fullpath[4096];
  if (everystr) {
    std::snprintf(fullpath, sizeof(fullpath), "%s.%03u.ppm", path,
                  this->ptFrameIndex);
  } else {
    std::snprintf(fullpath, sizeof(fullpath), "%s", path);
  }

  const uint32_t w = this->storageWidth;
  const uint32_t h = this->storageHeight;
  const VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  if (!this->createHostVisibleBuffer(
        size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, staging, stagingMem)) {
    fprintf(stderr, "[RTDBG] dumpStorageImage: createHostVisibleBuffer failed\n");
    return;
  }

  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, staging, this->allocator);
    vkFreeMemory(this->device, stagingMem, this->allocator);
    return;
  }

  VkImageMemoryBarrier toTfr {};
  toTfr.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toTfr.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toTfr.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  toTfr.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  toTfr.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toTfr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toTfr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toTfr.image = this->storageImage;
  toTfr.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  toTfr.subresourceRange.levelCount = 1;
  toTfr.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toTfr);

  VkBufferImageCopy region {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {w, h, 1};
  vkCmdCopyImageToBuffer(cmd, this->storageImage,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                         &region);

  VkImageMemoryBarrier toGeneral {};
  toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGeneral.image = this->storageImage;
  toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  toGeneral.subresourceRange.levelCount = 1;
  toGeneral.subresourceRange.layerCount = 1;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toGeneral);

  if (vkEndCommandBuffer(cmd) == VK_SUCCESS) {
    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS &&
        vkQueueWaitIdle(this->queue) == VK_SUCCESS) {
      void * mapped = nullptr;
      if (vkMapMemory(this->device, stagingMem, 0, size, 0, &mapped) !=
            VK_SUCCESS ||
          mapped == nullptr) {
        fprintf(stderr, "[RTDBG] dumpStorageImage: vkMapMemory failed\n");
      } else {
        const unsigned char * src = static_cast<const unsigned char *>(mapped);
        FILE * f = fopen(fullpath, "wb");
        if (f) {
          fprintf(f, "P6\n%u %u\n255\n", w, h);
          const size_t rowbytes = static_cast<size_t>(w) * 4;
          for (uint32_t y = 0; y < h; ++y) {
            const unsigned char * r = src + (static_cast<size_t>(y) * rowbytes);
            fwrite(r, 1, static_cast<size_t>(w) * 3, f);
          }
          fclose(f);
          fprintf(stderr, "[RTDBG] dumpStorageImage: wrote %s %ux%u\n",
                  fullpath, w, h);
        } else {
          fprintf(stderr, "[RTDBG] dumpStorageImage: fopen %s failed\n",
                  fullpath);
        }
        vkUnmapMemory(this->device, stagingMem);
      }
    }
  }

  vkDestroyBuffer(this->device, staging, this->allocator);
  vkFreeMemory(this->device, stagingMem, this->allocator);
}

SbBool
  SoRTXRenderBackend::renderExternal(const SoDrawList & drawlist,
                                     const SoRenderParams & params,
                                     VkCommandBuffer commandBuffer,
                                     VkRenderPass renderPass)
{
  if (!this->isInitialized()) {
    this->emitError("renderExternal called before backend initialization");
    return FALSE;
  }
  this->ptLastFrame = params.frame;
  if (COIN_VULKAN_ENV_FLAG("FC_VULKAN_BLACK_DEBUG")) {
    fprintf(stderr,
            "[BLACKRT] rtx renderExternal frame=%d acc=%d frameIndex=%u "
            "idleFrames=%u settle=%u enabled=%d samples=%u\n",
            params.frame, static_cast<int>(this->ptAccumulating),
            this->ptFrameIndex, this->ptIdleFrames, this->ptSettleFrames,
            static_cast<int>(this->ptEnabled),
            this->getPathTracingSampleCount());
  }
  if (!params.renderTarget) {
    this->emitError(
      "renderExternal called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    this->emitError("renderExternal called without command buffer/render pass");
    return FALSE;
  }
  this->debugValidateDrawList(drawlist);
  this->flushPendingDestroys(true);

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  // The caller's command buffer is already inside an active render pass, so
  // the acceleration-structure phase (BLAS/TLAS builds and buffer copies) is
  // recorded on the persistent transient command buffer which is submitted
  // and waited on here.  Queue submission is strictly ordered and the wait
  // makes the AS writes visible to the trace recorded below, so no explicit
  // synchronization with the caller's buffer is required.
  const bool wantTiming = getenv("FC_VULKAN_FRAME_TIMING") != nullptr;
  const double t0 = wantTiming ? vkNowMs() : 0.0;
  VkCommandBuffer cmd = this->beginTransientCommandBuffer();
  if (cmd == VK_NULL_HANDLE) {
    this->emitError("renderExternal: failed to allocate AS command buffer");
    return FALSE;
  }
  const bool asOk =
    this->recordAccelerationStructures(drawlist, params, *target, cmd);
  vkEndCommandBuffer(cmd);
  const double t1 = wantTiming ? vkNowMs() : 0.0;
  bool submitted = FALSE;
  if (asOk) {
    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    submitted = vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE) ==
      VK_SUCCESS && vkQueueWaitIdle(this->queue) == VK_SUCCESS;
  }
  const double t2 = wantTiming ? vkNowMs() : 0.0;
  if (submitted) {
    this->updateAdaptiveStats();
    // The denoiser readback (recordDenoiseReadback) reads this->accumBuffer,
    // which the raygen wrote this frame.  swapPathTracingHistory() below
    // swaps accumBuffer<->accumHistoryBuffer for the next frame's
    // reprojection, so the denoise must read the just-written data BEFORE the
    // swap; otherwise it denoises the stale/empty history buffer.
    this->updateDenoise();
    this->swapPathTracingHistory();
  }
  const double t3 = wantTiming ? vkNowMs() : 0.0;
  // Staging buffers are only referenced by the private submission; release
  // them after it provably completed (or never ran).  The command buffer
  // and pool are released in either case.
  if (submitted) {
    this->freePendingStagingDestroys();
    // The AS phase submission is complete here: shrink any BLAS built with
    // ALLOW_COMPACTION that has not yet been compacted (FC_VULKAN_AS_COMPACT).
    this->compactPendingBlases();
  }

  if (!asOk || !submitted) {
    this->emitError("renderExternal: AS phase failed");
    return FALSE;
  }

  // The trace ran in the AS phase above and the queue is idle, so storageImage
  // holds the current ray-traced result (if the tracer is accumulating).
  this->dumpStorageImageIfRequested();

  // The present pass is recorded into the caller's buffer (inside its
  // render pass); the trace ran in the AS phase above.  The descriptor set
  // was refreshed by recordAccelerationStructures() after the TLAS
  // (re)build, so binding 0 references the current TLAS.
  const bool presentOk =
    this->recordTraceAndPresent(params, *target, commandBuffer, renderPass);
  if (wantTiming) {
    // interval = host frame pacing (1/FPS, includes the Qt-submitted trace +
    // present + vsync).  asRecord = CPU time to record the AS builds;
    // asGpu = AS-phase GPU time (the submit+waitIdle blocks until it is done);
    // denoise = updateDenoise; traceRecord = present-pass record (CPU).
    const double t4 = vkNowMs();
    const double interval = this->lastFrameStartMs > 0.0
      ? t0 - this->lastFrameStartMs : 0.0;
    this->lastFrameStartMs = t4;
    fprintf(stderr,
            "[RTDBG] frameTiming interval=%.2f asRecord=%.2f asGpu=%.2f "
            "denoise=%.2f traceRecord=%.2f\n",
            interval, t1 - t0, t2 - t1, t3 - t2, t4 - t3);
  }
  return presentOk ? TRUE : FALSE;
}
