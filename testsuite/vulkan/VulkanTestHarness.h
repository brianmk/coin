// testsuite/VulkanTestHarness.h
//
// Shared header-only harness for SoVulkanRenderBackend smoke tests.  Creates a
// headless Vulkan instance/device plus an offscreen 32x32 color+depth target,
// and provides helpers to transition layouts, render, and read back pixels.
// No window system or swapchain is involved.

#ifndef COIN_VULKANTESTHARNESS_H
#define COIN_VULKANTESTHARNESS_H

#include "rendering/SoVulkanRenderBackend.h"

#include <Inventor/SoDB.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

namespace vulkan_test {

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 32;
constexpr uint32_t kPixelBytes = 4;

inline int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

inline uint32_t findMemoryType(VkPhysicalDevice physicalDevice,
                               uint32_t typeBits,
                               VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((typeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return 0;
}

inline bool createImage(VkDevice device,
                        VkPhysicalDevice physicalDevice,
                        VkFormat format,
                        VkImageUsageFlags usage,
                        VkImage & image,
                        VkDeviceMemory & memory)
{
  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {kWidth, kHeight, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(device, &ci, nullptr, &image) != VK_SUCCESS) return false;

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(device, image, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = findMemoryType(physicalDevice,
                                      requirements.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyImage(device, image, nullptr);
    image = VK_NULL_HANDLE;
    return false;
  }
  vkBindImageMemory(device, image, memory, 0);
  return true;
}

inline VkImageView createImageView(VkDevice device,
                                   VkImage image,
                                   VkFormat format,
                                   VkImageAspectFlags aspect)
{
  VkImageViewCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ci.image = image;
  ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ci.format = format;
  ci.subresourceRange.aspectMask = aspect;
  ci.subresourceRange.baseMipLevel = 0;
  ci.subresourceRange.levelCount = 1;
  ci.subresourceRange.baseArrayLayer = 0;
  ci.subresourceRange.layerCount = 1;
  VkImageView view = VK_NULL_HANDLE;
  vkCreateImageView(device, &ci, nullptr, &view);
  return view;
}

inline void transitionImage(VkCommandBuffer commandBuffer,
                            VkImage image,
                            VkImageAspectFlags aspect,
                            VkImageLayout oldLayout,
                            VkImageLayout newLayout)
{
  VkAccessFlags srcAccess = 0;
  VkAccessFlags dstAccess = 0;
  VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

  if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    srcStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }

  if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }
  else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
    dstAccess = VK_ACCESS_TRANSFER_READ_BIT;
    dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = aspect;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = srcAccess;
  barrier.dstAccessMask = dstAccess;

  vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

/*!
  Owns the full headless Vulkan context used by the smoke tests: instance,
  device, graphics queue, command pool, offscreen color+depth target, and the
  SoVulkanRenderBackend under test.  Call init() then exercise backend/render,
  and shutdown() once.
*/
struct Harness
{
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;

  VkCommandPool commandPool = VK_NULL_HANDLE;

  VkImage colorImage = VK_NULL_HANDLE;
  VkDeviceMemory colorMemory = VK_NULL_HANDLE;
  VkImageView colorView = VK_NULL_HANDLE;

  VkImage depthImage = VK_NULL_HANDLE;
  VkDeviceMemory depthMemory = VK_NULL_HANDLE;
  VkImageView depthView = VK_NULL_HANDLE;
  bool haveDepth = false;

  SoVulkanDeviceContext deviceContext;
  SoVulkanRenderTarget target;
  SoVulkanRenderBackend backend;
  bool backendInitialized = false;

  // Returns 0 on success, 77 to skip (no usable device), 1 on hard failure.
  int init()
  {
    SoDB::init();

    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "coin-vulkan-smoke";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instanceInfo, nullptr, &this->instance) !=
        VK_SUCCESS) {
      return skip("could not create a Vulkan instance");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(this->instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      vkDestroyInstance(this->instance, nullptr);
      return skip("no Vulkan physical devices");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(this->instance, &deviceCount, devices.data());

    bool found = false;
    for (VkPhysicalDevice candidate : devices) {
      uint32_t familyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount,
                                               nullptr);
      std::vector<VkQueueFamilyProperties> families(familyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount,
                                               families.data());
      for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
          this->physicalDevice = candidate;
          this->queueFamily = i;
          found = true;
          break;
        }
      }
      if (found) break;
    }
    if (!found) {
      vkDestroyInstance(this->instance, nullptr);
      return skip("no Vulkan device with a graphics queue");
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = this->queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (vkCreateDevice(this->physicalDevice, &deviceInfo, nullptr,
                       &this->device) != VK_SUCCESS) {
      vkDestroyInstance(this->instance, nullptr);
      return skip("could not create a Vulkan logical device");
    }
    vkGetDeviceQueue(this->device, this->queueFamily, 0, &this->queue);

    if (!createImage(this->device, this->physicalDevice,
                     VK_FORMAT_B8G8R8A8_UNORM,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     this->colorImage, this->colorMemory)) {
      this->shutdown();
      return skip("could not allocate color image");
    }
    this->colorView =
      createImageView(this->device, this->colorImage,
                      VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

    this->haveDepth = createImage(
      this->device, this->physicalDevice, VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, this->depthImage,
      this->depthMemory);
    if (this->haveDepth) {
      this->depthView =
        createImageView(this->device, this->depthImage,
                        VK_FORMAT_D32_SFLOAT_S8_UINT,
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    }

    VkCommandPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = this->queueFamily;
    vkCreateCommandPool(this->device, &poolInfo, nullptr, &this->commandPool);

    // Transition attachments into their attachment layouts once, up front.
    this->oneShot([&](VkCommandBuffer buffer) {
      transitionImage(buffer, this->colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      if (this->haveDepth) {
        transitionImage(buffer, this->depthImage,
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      }
    });

    this->deviceContext.instance = this->instance;
    this->deviceContext.physicalDevice = this->physicalDevice;
    this->deviceContext.device = this->device;
    this->deviceContext.graphicsQueue = this->queue;
    this->deviceContext.graphicsQueueFamilyIndex = this->queueFamily;

    this->target.colorImage = this->colorImage;
    this->target.colorImageView = this->colorView;
    this->target.colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    this->target.colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (this->haveDepth) {
      this->target.depthImage = this->depthImage;
      this->target.depthImageView = this->depthView;
      this->target.depthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
      this->target.depthLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    this->target.extent = {kWidth, kHeight};
    this->target.sampleCount = VK_SAMPLE_COUNT_1_BIT;

    SoRenderBackendInitParams initParams;
    initParams.userData = &this->deviceContext;
    if (!this->backend.initialize(initParams)) {
      this->shutdown();
      return skip("Vulkan backend could not initialize");
    }
    this->backendInitialized = true;
    return 0;
  }

  void shutdown()
  {
    if (this->backendInitialized) {
      this->backend.shutdown();
      this->backendInitialized = false;
    }
    if (this->commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(this->device, this->commandPool, nullptr);
      this->commandPool = VK_NULL_HANDLE;
    }
    if (this->colorView != VK_NULL_HANDLE) {
      vkDestroyImageView(this->device, this->colorView, nullptr);
      this->colorView = VK_NULL_HANDLE;
    }
    if (this->colorImage != VK_NULL_HANDLE) {
      vkDestroyImage(this->device, this->colorImage, nullptr);
      this->colorImage = VK_NULL_HANDLE;
    }
    if (this->colorMemory != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, this->colorMemory, nullptr);
      this->colorMemory = VK_NULL_HANDLE;
    }
    if (this->depthView != VK_NULL_HANDLE) {
      vkDestroyImageView(this->device, this->depthView, nullptr);
      this->depthView = VK_NULL_HANDLE;
    }
    if (this->depthImage != VK_NULL_HANDLE) {
      vkDestroyImage(this->device, this->depthImage, nullptr);
      this->depthImage = VK_NULL_HANDLE;
    }
    if (this->depthMemory != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, this->depthMemory, nullptr);
      this->depthMemory = VK_NULL_HANDLE;
    }
    if (this->device != VK_NULL_HANDLE) {
      vkDestroyDevice(this->device, nullptr);
      this->device = VK_NULL_HANDLE;
    }
    if (this->instance != VK_NULL_HANDLE) {
      vkDestroyInstance(this->instance, nullptr);
      this->instance = VK_NULL_HANDLE;
    }
  }

  void oneShot(const std::function<void(VkCommandBuffer)> & record)
  {
    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = this->commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(this->device, &ai, &buffer);

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(buffer, &bi);
    record(buffer);
    vkEndCommandBuffer(buffer);

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &buffer;
    vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(this->queue);
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &buffer);
  }

  SoRenderParams renderParams()
  {
    SoRenderParams params;
    params.viewport = SbViewportRegion(kWidth, kHeight);
    params.viewport.setViewportPixels(SbVec2s(0, 0),
                                      SbVec2s(kWidth, kHeight));
    params.viewMatrix.makeIdentity();
    params.projMatrix.makeIdentity();
    params.clearColor.setValue(0.0f, 0.0f, 0.0f, 1.0f);
    params.clearDepth = 1.0f;
    params.flags = SO_PARAM_CLEAR_WINDOW | SO_PARAM_CLEAR_DEPTH;
    params.renderTarget = &this->target;
    return params;
  }

  // Reads back the color attachment and returns it in BGRA byte order.
  std::vector<uint8_t> readback()
  {
    const VkDeviceSize bufferSize = kWidth * kHeight * kPixelBytes;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    {
      VkBufferCreateInfo bi {};
      bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bi.size = bufferSize;
      bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      vkCreateBuffer(this->device, &bi, nullptr, &staging);
      VkMemoryRequirements requirements;
      vkGetBufferMemoryRequirements(this->device, staging, &requirements);
      VkMemoryAllocateInfo ai {};
      ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      ai.allocationSize = requirements.size;
      ai.memoryTypeIndex = findMemoryType(
        this->physicalDevice, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      vkAllocateMemory(this->device, &ai, nullptr, &stagingMemory);
      vkBindBufferMemory(this->device, staging, stagingMemory, 0);
    }

    this->oneShot([&](VkCommandBuffer buffer) {
      transitionImage(buffer, this->colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      VkBufferImageCopy region {};
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.layerCount = 1;
      region.imageExtent = {kWidth, kHeight, 1};
      vkCmdCopyImageToBuffer(buffer, this->colorImage,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                             &region);
      transitionImage(buffer, this->colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    });

    std::vector<uint8_t> pixels(static_cast<size_t>(bufferSize));
    void * mapped = nullptr;
    vkMapMemory(this->device, stagingMemory, 0, bufferSize, 0, &mapped);
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(bufferSize));
    vkUnmapMemory(this->device, stagingMemory);

    vkDestroyBuffer(this->device, staging, nullptr);
    vkFreeMemory(this->device, stagingMemory, nullptr);
    return pixels;
  }
};

inline const uint8_t * pixelAt(const std::vector<uint8_t> & pixels,
                               int x,
                               int y)
{
  return &pixels[static_cast<size_t>(y * kWidth + x) * kPixelBytes];
}

// BGRA target: byte order is B,G,R,A.
inline bool nearColor(const uint8_t * pixel, int red, int green, int blue)
{
  return std::abs(static_cast<int>(pixel[2]) - red) < 40 &&
    std::abs(static_cast<int>(pixel[1]) - green) < 40 &&
    std::abs(static_cast<int>(pixel[0]) - blue) < 40;
}

// Counts pixels within tolerance of (red, green, blue) across the whole
// framebuffer.  Robust for rasterization checks where the exact footprint is
// driver-dependent (line/point width, edge coverage).
inline int countNear(const std::vector<uint8_t> & pixels,
                     int red,
                     int green,
                     int blue)
{
  int count = 0;
  for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
    if (nearColor(&pixels[static_cast<size_t>(i) * kPixelBytes], red, green,
                  blue)) {
      ++count;
    }
  }
  return count;
}

inline SoRenderCommand makeTriangle(const float * positions,
                                    SoPrimitiveTopology topology =
                                      SO_TOPOLOGY_TRIANGLES,
                                    uint32_t vertexCount = 3)
{
  SoRenderCommand command;
  command.modelMatrix.makeIdentity();
  command.geometry.topology = topology;
  command.geometry.vertexCount = vertexCount;
  command.geometry.positions = positions;
  command.geometry.vertexStride = sizeof(float) * 3;
  return command;
}

} // namespace vulkan_test

#endif // COIN_VULKANTESTHARNESS_H
