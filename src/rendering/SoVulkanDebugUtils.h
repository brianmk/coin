// src/rendering/SoVulkanDebugUtils.h
//
// Internal VK_EXT_debug_utils helpers (object names + command-buffer labels) for
// readable RenderDoc/Nsight captures.  Not public API.
//
// Gated by diagnostics.debugUtils (COIN_VULKAN_DEBUG_UTILS); when unavailable the
// vkGetDeviceProcAddr-resolved entry points stay null and every helper is a no-op.
// Cached once for the shared VkDevice; call setDevice() for a second device.

#ifndef COIN_SOVULKANDEBUGUTILS_H
#define COIN_SOVULKANDEBUGUTILS_H

#include "rendering/SoVulkanPlatform.h"
#include <vulkan/vulkan.h>

#include "rendering/SoVulkanConfig.h"

namespace SoVulkanDebugUtils {

inline bool
enabled()
{
  return SoVulkanConfig::get().diagnostics.debugUtils;
}

// Device whose vkGetDeviceProcAddr resolves the entry points; null = no-op.
inline VkDevice &
deviceRef()
{
  static VkDevice device = VK_NULL_HANDLE;
  return device;
}

inline void
setDevice(VkDevice device)
{
  deviceRef() = device;
}

struct Functions {
  PFN_vkSetDebugUtilsObjectNameEXT setName = nullptr;
  PFN_vkCmdBeginDebugUtilsLabelEXT beginLabel = nullptr;
  PFN_vkCmdEndDebugUtilsLabelEXT endLabel = nullptr;
  PFN_vkCmdInsertDebugUtilsLabelEXT insertLabel = nullptr;
};

inline const Functions &
functions()
{
  static const Functions fns = [] {
    Functions f;
    const VkDevice device = deviceRef();
    if (device == VK_NULL_HANDLE) {
      return f;
    }
    f.setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
      vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    f.beginLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT"));
    f.endLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT"));
    f.insertLabel = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
      vkGetDeviceProcAddr(device, "vkCmdInsertDebugUtilsLabelEXT"));
    return f;
  }();
  return fns;
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
