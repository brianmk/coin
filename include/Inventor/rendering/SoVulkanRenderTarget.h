// include/Inventor/rendering/SoVulkanRenderTarget.h

#ifndef COIN_SOVULKANRENDERTARGET_H
#define COIN_SOVULKANRENDERTARGET_H

/*!
  \file SoVulkanRenderTarget.h
  \brief Backend-neutral Vulkan device and render-target contracts.

  These structures describe the Vulkan resources a concrete SoRenderBackend
  needs from the embedding application (typically a QVulkanWindow inside
  FreeCAD's Gui module, or an offscreen device for tests and exporters).

  Coin itself has no Qt or window-system knowledge, so the application owns
  the VkInstance, physical/logical device, and graphics queue, and hands them
  to the backend through SoRenderBackendInitParams::userData.  Render targets
  are delivered per frame through SoRenderParams::renderTarget and are never
  retained by the backend beyond the current render() call.
*/

#include <cstdint>

// Pull in the Vulkan declarations.  This header is only compiled when
// COIN_BUILD_VULKAN_RENDERER is enabled, so the Vulkan SDK must be available.
#include <vulkan/vulkan.h>

/*!
  struct SoVulkanDeviceContext
  \brief Application-owned Vulkan device state required by the backend.

  The backend borrows these handles for the lifetime of the backend.  The
  application must keep the instance, device, and queue valid until the
  backend has been shut down.
*/
struct SoVulkanDeviceContext {
  VkInstance instance = VK_NULL_HANDLE;               //!< Owning instance.
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;   //!< Selected GPU.
  VkDevice device = VK_NULL_HANDLE;                   //!< Logical device.
  VkQueue graphicsQueue = VK_NULL_HANDLE;             //!< Submission queue.
  uint32_t graphicsQueueFamilyIndex = 0;              //!< Queue family index.
  uint32_t apiVersion = VK_API_VERSION_1_0;           //!< Negotiated API version.
  const VkAllocationCallbacks * allocator = nullptr;  //!< Optional host allocator.
};

/*!
  struct SoVulkanRenderTarget
  \brief Per-frame destination framebuffer for a retained render.

  The application guarantees the images are already in the layouts declared
  here (or in VK_IMAGE_LAYOUT_PRESENT_SRC_KHR when a swapchain image is used
  and the backend is expected to transition it).  The backend records a render
  pass that loads the existing attachment contents and conditionally clears
  them according to SoRenderParams::flags, so partial-viewport and overlay
  rendering compose correctly.
*/
struct SoVulkanRenderTarget {
  VkImage colorImage = VK_NULL_HANDLE;             //!< Destination color image.
  VkImageView colorImageView = VK_NULL_HANDLE;     //!< Destination color view.
  VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM; //!< Color attachment format.
  VkImageLayout colorLayout =
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;      //!< Incoming color layout.

  VkImage depthImage = VK_NULL_HANDLE;             //!< Optional depth image.
  VkImageView depthImageView = VK_NULL_HANDLE;     //!< Optional depth view.
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;      //!< Optional depth format.
  VkImageLayout depthLayout =
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; //!< Incoming depth layout.

  VkExtent2D extent {0, 0};                        //!< Attachment extent in pixels.
  VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT; //!< MSAA samples.
};

#endif // COIN_SOVULKANRENDERTARGET_H
