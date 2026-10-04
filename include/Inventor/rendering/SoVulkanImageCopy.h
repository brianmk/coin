// include/Inventor/rendering/SoVulkanImageCopy.h

#ifndef COIN_SOVULKANIMAGECOPY_H
#define COIN_SOVULKANIMAGECOPY_H

/*!
  \file SoVulkanImageCopy.h
  \brief Shared "copy a color image into a host-visible buffer" primitive.

  Used by the two debug frame dumps:

    - the renderer's storage-image dump (SoVulkanShared::dumpImageToHost), and
    - the application's swapchain dump (FreeCAD's VulkanFrameDumper).

  They differ only in who owns the command buffer: the renderer submits a
  one-shot buffer, the app records into the frame's own.  The copy sequence is
  therefore shared here.  The helpers call the raw Vulkan entry points, so both
  callers must link a Vulkan loader.

  The Vulkan declarations are compiled in only when the installed Coin exports
  COIN_HAVE_VULKAN_RENDERER, so a non-Vulkan Coin does not force a Vulkan SDK
  dependency on its consumers.
*/

#include <Inventor/C/basic.h>

/* Honour the capability the installed header exports; fall back to off for a
   pre-existing basic.h that predates it. */
#ifndef COIN_HAVE_VULKAN_RENDERER
#define COIN_HAVE_VULKAN_RENDERER 0
#endif

#if COIN_HAVE_VULKAN_RENDERER

// On Windows <vulkan/vulkan.h> pulls in <windows.h>, whose min/max macros break
// std::min/std::max; suppress them for this translation unit only.
#if defined(_WIN32) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

// Vulkan declarations; only compiled with COIN_HAVE_VULKAN_RENDERER.
#include <vulkan/vulkan.h>

namespace SoVulkanImageCopy {

//! Record an image -> buffer copy, with the layout transitions it requires.
//!
//! The image is transitioned out of its current state
//! (\a srcLayout / \a srcAccess / \a srcStage) into TRANSFER_SRC_OPTIMAL, copied,
//! then transitioned into the caller's desired final state
//! (\a restoreLayout / \a restoreAccess / \a restoreStage).
//! \a restoreSrcAccess is the access the copy leaves pending on the image
//! (TRANSFER_READ): the source access mask of that trailing transition.
inline void recordToBuffer(VkCommandBuffer cmd, VkImage image, VkBuffer buffer,
                           VkImageLayout srcLayout, VkAccessFlags srcAccess,
                           VkPipelineStageFlags srcStage,
                           VkImageLayout restoreLayout,
                           VkAccessFlags restoreSrcAccess,
                           VkAccessFlags restoreAccess,
                           VkPipelineStageFlags restoreStage,
                           uint32_t width, uint32_t height)
{
  auto transition = [&](VkImageLayout oldLayout, VkImageLayout newLayout,
                        VkAccessFlags fromAccess, VkAccessFlags toAccess,
                        VkPipelineStageFlags fromStage,
                        VkPipelineStageFlags toStage) {
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = fromAccess;
    barrier.dstAccessMask = toAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, fromStage, toStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
  };

  transition(srcLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, srcAccess,
             VK_ACCESS_TRANSFER_READ_BIT, srcStage,
             VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkBufferImageCopy region {};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {width, height, 1};
  vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         buffer, 1, &region);

  transition(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, restoreLayout,
             restoreSrcAccess, restoreAccess, VK_PIPELINE_STAGE_TRANSFER_BIT,
             restoreStage);
}

} // namespace SoVulkanImageCopy

#endif // COIN_HAVE_VULKAN_RENDERER

#endif // COIN_SOVULKANIMAGECOPY_H
