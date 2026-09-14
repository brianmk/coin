// include/Inventor/rendering/SoVulkanImageCopy.h

#ifndef COIN_SOVULKANIMAGECOPY_H
#define COIN_SOVULKANIMAGECOPY_H

#include <vulkan/vulkan.h>

/*!
  \file SoVulkanImageCopy.h
  \brief Shared "copy a color image into a host-visible buffer" primitive.

  Both debug frame dumps -- the renderer's storage-image dump (Coin's internal
  SoVulkanShared::dumpImageToHost) and the embedding application's swapchain
  dump (FreeCAD's VulkanFrameDumper) -- move an image into a host-visible
  buffer wrapped in the same TRANSFER_SRC_OPTIMAL layout transitions.  They
  differ only in who owns the command buffer (the renderer submits a one-shot
  buffer, the application records into the frame's own buffer), so the copy
  sequence lives here and both call it.

  The helper uses the raw Vulkan entry points; both callers link the loader.
*/
namespace SoVulkanImageCopy {

//! Record an image -> buffer copy, wrapping it in the TRANSFER_SRC_OPTIMAL
//! layout transitions.  \a srcLayout / \a srcAccess / \a srcStage describe the
//! image's current state; \a restoreLayout / \a restoreAccess / \a restoreStage
//! the state to leave it in.  \a restoreSrcAccess is the access that the copy
//! itself leaves pending on the image (TRANSFER_READ for a read-back, but kept
//! a parameter so callers can mirror their previous barrier exactly).  The
//! copy reads at VK_ACCESS_TRANSFER_READ_BIT /
//! VK_PIPELINE_STAGE_TRANSFER_BIT.
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

#endif // COIN_SOVULKANIMAGECOPY_H
