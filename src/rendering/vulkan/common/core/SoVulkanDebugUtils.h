// src/rendering/vulkan/common/core/SoVulkanDebugUtils.h
//
// Internal VK_EXT_debug_utils helpers: object names and command-buffer labels
// so RenderDoc / Nsight / validation captures are readable.  Internal to Coin
// (not installed, not public API).
//
// Gated by SoVulkanConfig::get().diagnostics.debugUtils (FC_VULKAN_DEBUG_UTILS).
// The entry points are resolved through vkGetDeviceProcAddr, so a build/run
// where the instance did not enable VK_EXT_debug_utils degrades to a no-op
// instead of dereferencing a null pointer.
//
// Single-device assumption: the renderer shares one VkDevice across the raster
// and ray-tracing backends.  setDevice() resolves the entry points at the
// device-creation site, before any frame recording; call it again if a second
// device ever appears.

#ifndef COIN_SOVULKANDEBUGUTILS_H
#define COIN_SOVULKANDEBUGUTILS_H

#include <vulkan/vulkan.h>

#include "rendering/vulkan/common/core/SoVulkanConfig.h"

namespace SoVulkanDebugUtils {

inline bool
enabled()
{
  return SoVulkanConfig::get().diagnostics.debugUtils;
}

// The device whose vkGetDeviceProcAddr resolves the entry points.  Set once
// after device creation; a null device keeps every helper a no-op.
inline VkDevice &
deviceRef()
{
  static VkDevice device = VK_NULL_HANDLE;
  return device;
}

struct Functions {
  PFN_vkSetDebugUtilsObjectNameEXT setName = nullptr;
  PFN_vkCmdBeginDebugUtilsLabelEXT beginLabel = nullptr;
  PFN_vkCmdEndDebugUtilsLabelEXT endLabel = nullptr;
};

inline Functions &
functionsRef()
{
  static Functions fns;
  return fns;
}

// Resolve the device-level entry points here, at the single-threaded
// device-creation site, instead of lazily on first use: a lazy resolve raced
// concurrent first calls and could latch an all-null table.  Calling this again
// for a second device re-resolves for that device.
inline void
setDevice(VkDevice device)
{
  deviceRef() = device;
  Functions fns {};
  if (device != VK_NULL_HANDLE) {
    fns.setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
      vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    fns.beginLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT"));
    fns.endLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT"));
  }
  functionsRef() = fns;
}

inline const Functions &
functions()
{
  return functionsRef();
}

inline void
nameObject(VkDevice device, VkObjectType type, uint64_t handle,
           const char * name)
{
  if (!enabled() || handle == 0 || name == nullptr) {
    return;
  }
  const Functions & f = functions();
  if (f.setName == nullptr) {
    return;
  }
  VkDebugUtilsObjectNameInfoEXT info {};
  info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  info.objectType = type;
  info.objectHandle = handle;
  info.pObjectName = name;
  f.setName(device, &info);
}

inline void
beginLabel(VkCommandBuffer commandBuffer, const char * name,
           float r = 0.25f, float g = 0.55f, float b = 0.95f, float a = 1.0f)
{
  if (!enabled() || commandBuffer == VK_NULL_HANDLE || name == nullptr) {
    return;
  }
  const Functions & f = functions();
  if (f.beginLabel == nullptr) {
    return;
  }
  VkDebugUtilsLabelEXT label {};
  label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  label.pLabelName = name;
  label.color[0] = r;
  label.color[1] = g;
  label.color[2] = b;
  label.color[3] = a;
  f.beginLabel(commandBuffer, &label);
}

inline void
endLabel(VkCommandBuffer commandBuffer)
{
  if (!enabled() || commandBuffer == VK_NULL_HANDLE) {
    return;
  }
  const Functions & f = functions();
  if (f.endLabel != nullptr) {
    f.endLabel(commandBuffer);
  }
}

} // namespace SoVulkanDebugUtils

#endif // COIN_SOVULKANDEBUGUTILS_H
