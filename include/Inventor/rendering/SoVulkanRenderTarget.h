// include/Inventor/rendering/SoVulkanRenderTarget.h

#ifndef COIN_SOVULKANRENDERTARGET_H
#define COIN_SOVULKANRENDERTARGET_H

/*!
  \file SoVulkanRenderTarget.h
  \brief Backend-neutral Vulkan device and render-target contracts.

  Structures describing the Vulkan resources a concrete SoRenderBackend needs
  from the embedding application (a QVulkanWindow in FreeCAD's Gui module, or an
  offscreen device for tests/exporters): the app owns the VkInstance, device and
  queue and hands them via SoRenderBackendInitParams::userData; render targets
  arrive per frame via SoRenderParams::renderTarget and are never retained past
  the current render() call.

  Compiles only when the installed Coin exports COIN_HAVE_VULKAN_RENDERER, so
  an installed non-Vulkan Coin does not force a Vulkan SDK dependency on its
  consumers.
*/

#include <Inventor/C/basic.h>

/* Honour the capability the installed header exports; fall back to off for a
   pre-existing basic.h that predates it. */
#ifndef COIN_HAVE_VULKAN_RENDERER
#define COIN_HAVE_VULKAN_RENDERER 0
#endif

#if COIN_HAVE_VULKAN_RENDERER

#include <cstdint>

// On Windows <vulkan/vulkan.h> pulls in <windows.h>, whose min/max macros break
// std::min/std::max; suppress them for this translation unit only.
#if defined(_WIN32) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

// Vulkan declarations; only compiled with COIN_HAVE_VULKAN_RENDERER.
#include <vulkan/vulkan.h>

/*!
  struct SoVulkanDeviceCaps
  \brief Physical-device capabilities probed once by the embedding application.

  The renderer needs which optional extensions/features the device advertises to
  select its best technique.  The app already probes the device for
  vkCreateDevice, so it hands the result through SoVulkanDeviceContext::caps
  rather than the renderer re-enumerating the extension list (kept in one place).
*/
struct SoVulkanDeviceCaps {
  bool externalSemaphoreFd = false;    //!< VK_KHR_external_semaphore_fd.
  bool externalMemoryFd = false;       //!< VK_KHR_external_memory_fd.
  bool fillModeNonSolid = false;       //!< VK_POLYGON_MODE_LINE/POINT.
  bool fullDrawIndexUint32 = false;    //!< 32-bit vertex indices.
  bool dualSrcBlend = false;           //!< SRC1_* blend factors.
  bool timelineSemaphore = false;      //!< Vulkan 1.2 timeline semaphores.
  bool synchronization2 = false;       //!< VK_KHR_synchronization2.
  //! VK_KHR_synchronization2 advertised as an extension (not core 1.3), required to add it.
  bool synchronization2Extension = false;
  //! VK_EXT_descriptor_indexing update-after-bind, so a set may be updated while
  //! an in-flight command buffer references it (VUID-vkUpdateDescriptorSets-None-03047).
  bool descriptorIndexingUpdateAfterBind = false;
  //! VK_EXT_pipeline_creation_feedback (COIN_VULKAN_PIPELINE_FEEDBACK).
  bool pipelineCreationFeedback = false;
  //! VK_EXT_debug_printf + VK_KHR_shader_non_semantic_info for COIN_ENABLE_DEBUG_PRINTF
  //! shaders to emit via the validation layer (COIN_VULKAN_DEBUG_PRINTF).
  bool debugPrintf = false;
};

//! Application-owned device state; borrowed for the backend's lifetime, so keep valid until shutdown.
struct SoVulkanDeviceContext {
  VkInstance instance = VK_NULL_HANDLE;               //!< Owning instance.
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;   //!< Selected GPU.
  VkDevice device = VK_NULL_HANDLE;                   //!< Logical device.
  VkQueue graphicsQueue = VK_NULL_HANDLE;             //!< Submission queue.
  uint32_t graphicsQueueFamilyIndex = 0;              //!< Queue family index.
  // Optional async-compute queue, requested by the app at device creation (e.g.
  // QVulkanWindow::setQueueCreateInfoModifier); UINT32_MAX when absent.
  uint32_t computeQueueFamilyIndex = ~0u;             //!< Compute family index.
  uint32_t computeQueueIndex = 0;                     //!< Queue index (default 0).
  uint32_t apiVersion = VK_API_VERSION_1_0;           //!< Negotiated API version.
  const VkAllocationCallbacks * allocator = nullptr;  //!< Optional host allocator.
  //! Probed caps; when capsValid is false the renderer probes the device itself.
  SoVulkanDeviceCaps caps {};
  bool capsValid = false;
};

/*!
  struct SoVulkanRenderTarget
  \brief Per-frame destination framebuffer for a retained render.

  The app guarantees the images are already in the layouts declared here (or
  PRESENT_SRC_KHR for a swapchain image the backend should transition).  The
  backend's pass loads the contents and clears conditionally per
  SoRenderParams::flags, so partial-viewport and overlay rendering compose.
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

#endif // COIN_HAVE_VULKAN_RENDERER

#endif // COIN_SOVULKANRENDERTARGET_H
