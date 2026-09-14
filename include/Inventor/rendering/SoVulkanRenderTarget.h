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

  The whole contract is compiled only when COIN_BUILD_VULKAN_RENDERER is set
  (by Coin's own build and by applications that opt in).  Without it the
  header expands to nothing, so an installed Coin built without the Vulkan
  renderer does not force a Vulkan SDK dependency on its consumers.
*/

#ifndef COIN_BUILD_VULKAN_RENDERER
#define COIN_BUILD_VULKAN_RENDERER 0
#endif

#if COIN_BUILD_VULKAN_RENDERER

#include <cstdint>

// Pull in the Vulkan declarations.  This header is only compiled when
// COIN_BUILD_VULKAN_RENDERER is enabled, so the Vulkan SDK must be available.
#include <vulkan/vulkan.h>

/*!
  struct SoVulkanDeviceCaps
  \brief Physical-device capabilities probed once by the embedding application.

  The renderer needs to know which optional extensions/features the device
  advertises to select its best technique.  The application already probes the
  device to decide which extensions/features to request at vkCreateDevice, so
  it hands the result through SoVulkanDeviceContext::caps instead of the
  renderer re-enumerating the device extension list (the extension-name list
  then lives in exactly one place).
*/
struct SoVulkanDeviceCaps {
  bool rayTracing = false;             //!< AS + ray_tracing_pipeline + ray_query.
  bool positionFetch = false;          //!< VK_KHR_ray_tracing_position_fetch.
  bool opacityMicromap = false;        //!< VK_EXT_opacity_micromap.
  bool nvCluster = false;              //!< VK_NV_cluster_acceleration_structure.
  bool nvPartitioned = false;          //!< VK_NV_partitioned_acceleration_structure.
  bool nvLinearSweptSpheres = false;   //!< VK_NV_ray_tracing_linear_swept_spheres.
  bool externalSemaphoreFd = false;    //!< VK_KHR_external_semaphore_fd.
  bool externalMemoryFd = false;       //!< VK_KHR_external_memory_fd.
  bool fillModeNonSolid = false;       //!< VK_POLYGON_MODE_LINE/POINT.
  bool fullDrawIndexUint32 = false;    //!< 32-bit vertex indices.
  bool dualSrcBlend = false;           //!< SRC1_* blend factors.
  bool timelineSemaphore = false;      //!< Vulkan 1.2 timeline semaphores.
  bool synchronization2 = false;       //!< VK_KHR_synchronization2.
};

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
  // Optional async-compute queue.  The embedding app requests it at device
  // creation (e.g. via QVulkanWindow::setQueueCreateInfoModifier) and the
  // backend retrieves the handle with vkGetDeviceQueue() using the family +
  // index below.  computeQueueFamilyIndex is UINT32_MAX when none was created.
  uint32_t computeQueueFamilyIndex = ~0u;             //!< Compute family index.
  uint32_t computeQueueIndex = 0;                     //!< Queue index (default 0).
  uint32_t apiVersion = VK_API_VERSION_1_0;           //!< Negotiated API version.
  const VkAllocationCallbacks * allocator = nullptr;  //!< Optional host allocator.
  //! Probed capabilities (see SoVulkanDeviceCaps).  When capsValid is false
  //! (e.g. an offscreen/test context) the renderer probes the device itself.
  SoVulkanDeviceCaps caps {};
  bool capsValid = false;
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

#endif // COIN_BUILD_VULKAN_RENDERER

#endif // COIN_SOVULKANRENDERTARGET_H
