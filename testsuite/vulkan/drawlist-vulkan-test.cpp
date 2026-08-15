// testsuite/drawlist-vulkan-test.cpp
//
// Headless runtime smoke test for SoVulkanRenderBackend.  Exercises a full
// offscreen render into an application-owned image, readback, repeated
// execution, and generation-based cache invalidation.  No window system or
// swapchain is required: the test creates its own instance/device/images.

#include "rendering/SoVulkanRenderBackend.h"

#include <Inventor/SoDB.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 32;

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT severity,
  VkDebugUtilsMessageTypeFlagsEXT,
  const VkDebugUtilsMessengerCallbackDataEXT * data,
  void *)
{
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cerr << "[VK] " << data->pMessage << std::endl;
  }
  return VK_FALSE;
}

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return 77;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice,
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

bool createImage(VkDevice device,
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
  ai.memoryTypeIndex = findMemoryType(
    physicalDevice, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyImage(device, image, nullptr);
    image = VK_NULL_HANDLE;
    return false;
  }
  vkBindImageMemory(device, image, memory, 0);
  return true;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
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

void transitionImage(VkCommandBuffer commandBuffer,
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

void oneShot(VkDevice device, VkCommandPool pool, VkQueue queue,
             const std::function<void(VkCommandBuffer)> & record)
{
  VkCommandBufferAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer buffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(device, &ai, &buffer);

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
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, pool, 1, &buffer);
}

SoRenderParams renderParams(SoVulkanRenderTarget * target)
{
  SoRenderParams params;
  params.viewport = SbViewportRegion(kWidth, kHeight);
  params.viewport.setViewportPixels(SbVec2s(0, 0), SbVec2s(kWidth, kHeight));
  params.viewMatrix.makeIdentity();
  params.projMatrix.makeIdentity();
  params.clearColor.setValue(0.0f, 0.0f, 0.0f, 1.0f);
  params.clearDepth = 1.0f;
  params.flags = SO_PARAM_CLEAR_WINDOW | SO_PARAM_CLEAR_DEPTH;
  params.renderTarget = target;
  return params;
}

const uint8_t * pixelAt(const std::vector<uint8_t> & pixels, int x, int y)
{
  return &pixels[static_cast<size_t>(y * kWidth + x) * 4];
}

bool nearColor(const uint8_t * pixel, int red, int green, int blue)
{
  // The offscreen target is B8G8R8A8, so byte order is B,G,R,A.
  return std::abs(static_cast<int>(pixel[2]) - red) < 40 &&
    std::abs(static_cast<int>(pixel[1]) - green) < 40 &&
    std::abs(static_cast<int>(pixel[0]) - blue) < 40;
}

} // namespace

static int
runTest()
{
  SoDB::init();

  // --- Instance -----------------------------------------------------------
  VkApplicationInfo appInfo {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "drawlist-vulkan-test";
  appInfo.apiVersion = VK_API_VERSION_1_0;

  const char * validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
  const char * debugExtensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

  VkInstanceCreateInfo instanceInfo {};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;
  if (vkEnumerateInstanceVersion != nullptr) {
    uint32_t availableLayers = 0;
    vkEnumerateInstanceLayerProperties(&availableLayers, nullptr);
    std::vector<VkLayerProperties> layers(availableLayers);
    vkEnumerateInstanceLayerProperties(&availableLayers, layers.data());
    for (const auto & layer : layers) {
      if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
        instanceInfo.enabledLayerCount = 1;
        instanceInfo.ppEnabledLayerNames = validationLayers;
        instanceInfo.enabledExtensionCount = 1;
        instanceInfo.ppEnabledExtensionNames = debugExtensions;
        break;
      }
    }
  }

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
    return skip("could not create a Vulkan instance");
  }

  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  if (instanceInfo.enabledExtensionCount) {
    const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (create) {
      VkDebugUtilsMessengerCreateInfoEXT ci {};
      ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      ci.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      ci.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      ci.pfnUserCallback = debugCallback;
      create(instance, &ci, nullptr, &messenger);
    }
  }

  // --- Device -------------------------------------------------------------
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    vkDestroyInstance(instance, nullptr);
    return skip("no Vulkan physical devices");
  }
  std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  uint32_t graphicsQueueFamily = 0;
  for (VkPhysicalDevice candidate : physicalDevices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, families.data());
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        physicalDevice = candidate;
        graphicsQueueFamily = i;
        break;
      }
    }
    if (physicalDevice != VK_NULL_HANDLE) break;
  }
  if (physicalDevice == VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
    return skip("no Vulkan device with a graphics queue");
  }

  const float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo {};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex = graphicsQueueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceInfo {};
  deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;

  VkDevice device = VK_NULL_HANDLE;
  if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) !=
      VK_SUCCESS) {
    vkDestroyInstance(instance, nullptr);
    return skip("could not create a Vulkan logical device");
  }

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, graphicsQueueFamily, 0, &queue);

  // --- Offscreen target ----------------------------------------------------
  VkImage colorImage = VK_NULL_HANDLE;
  VkDeviceMemory colorMemory = VK_NULL_HANDLE;
  if (!createImage(device, physicalDevice, VK_FORMAT_B8G8R8A8_UNORM,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   colorImage, colorMemory)) {
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return skip("could not allocate color image");
  }
  VkImageView colorView =
    createImageView(device, colorImage, VK_FORMAT_B8G8R8A8_UNORM,
                    VK_IMAGE_ASPECT_COLOR_BIT);

  VkImage depthImage = VK_NULL_HANDLE;
  VkDeviceMemory depthMemory = VK_NULL_HANDLE;
  const bool haveDepth = createImage(
    device, physicalDevice, VK_FORMAT_D32_SFLOAT,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImage, depthMemory);
  VkImageView depthView = VK_NULL_HANDLE;
  if (haveDepth) {
    depthView = createImageView(device, depthImage, VK_FORMAT_D32_SFLOAT,
                                VK_IMAGE_ASPECT_DEPTH_BIT);
  }

  // --- Setup command pool --------------------------------------------------
  VkCommandPoolCreateInfo poolInfo {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = graphicsQueueFamily;
  VkCommandPool pool = VK_NULL_HANDLE;
  vkCreateCommandPool(device, &poolInfo, nullptr, &pool);

  // Transition attachments into their attachment layouts.
  oneShot(device, pool, queue, [&](VkCommandBuffer buffer) {
    transitionImage(buffer, colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (haveDepth) {
      transitionImage(buffer, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
  });

  // --- Backend -------------------------------------------------------------
  SoVulkanDeviceContext context;
  context.instance = instance;
  context.physicalDevice = physicalDevice;
  context.device = device;
  context.graphicsQueue = queue;
  context.graphicsQueueFamilyIndex = graphicsQueueFamily;

  SoVulkanRenderBackend backend;
  SoRenderBackendInitParams initParams;
  initParams.userData = &context;
  if (!backend.initialize(initParams)) {
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyImageView(device, colorView, nullptr);
    vkDestroyImage(device, colorImage, nullptr);
    vkFreeMemory(device, colorMemory, nullptr);
    if (haveDepth) {
      vkDestroyImageView(device, depthView, nullptr);
      vkDestroyImage(device, depthImage, nullptr);
      vkFreeMemory(device, depthMemory, nullptr);
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return skip("Vulkan draw-list backend could not initialize");
  }

  SoVulkanRenderTarget target;
  target.colorImage = colorImage;
  target.colorImageView = colorView;
  target.colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
  target.colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  if (haveDepth) {
    target.depthImage = depthImage;
    target.depthImageView = depthView;
    target.depthFormat = VK_FORMAT_D32_SFLOAT;
    target.depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  }
  target.extent = {kWidth, kHeight};
  target.sampleCount = VK_SAMPLE_COUNT_1_BIT;

  // --- Build draw list -----------------------------------------------------
  const uint32_t indices[] = {0, 1, 2, 0, 2, 3};
  const float quad[] = {
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
     1.0f,  1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f
  };
  const float triangle[] = {
    -0.8f, -0.8f, 0.0f,
     0.8f, -0.8f, 0.0f,
     0.0f,  0.8f, 0.0f
  };
  const float triangleColors[] = {
    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f
  };

  SoDrawList drawlist;
  SoRenderCommand blueQuad;
  blueQuad.modelMatrix.makeIdentity();
  blueQuad.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  blueQuad.geometry.vertexCount = 4;
  blueQuad.geometry.indexCount = 6;
  blueQuad.geometry.positions = quad;
  blueQuad.geometry.indices = indices;
  blueQuad.geometry.vertexStride = sizeof(float) * 3;
  blueQuad.material.diffuse = SbVec4f(0.0f, 0.0f, 1.0f, 1.0f);
  drawlist.addCommand(blueQuad);

  SoRenderCommand greenTriangle;
  greenTriangle.modelMatrix.makeIdentity();
  greenTriangle.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  greenTriangle.geometry.vertexCount = 3;
  greenTriangle.geometry.positions = triangle;
  greenTriangle.geometry.vertexStride = sizeof(float) * 3;
  greenTriangle.geometry.colors = triangleColors;
  greenTriangle.material.diffuse = SbVec4f(1.0f, 0.0f, 0.0f, 1.0f);
  drawlist.addCommand(greenTriangle);

  const SoRenderParams params = renderParams(&target);
  int result = 0;

  // Empty draw list is a valid (no-op) frame.
  SoDrawList empty;
  if (!backend.render(empty, params)) {
    std::cerr << "FAIL: empty draw list was not accepted" << std::endl;
    result = 1;
  }

  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: draw-list execution failed" << std::endl;
    result = 1;
  }

  // Read back and validate.
  VkDeviceSize bufferSize = kWidth * kHeight * 4;
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  {
    VkBufferCreateInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bufferSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bi, nullptr, &staging);
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, staging, &requirements);
    VkMemoryAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = requirements.size;
    ai.memoryTypeIndex = findMemoryType(
      physicalDevice, requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &ai, nullptr, &stagingMemory);
    vkBindBufferMemory(device, staging, stagingMemory, 0);
  }

  oneShot(device, pool, queue, [&](VkCommandBuffer buffer) {
    transitionImage(buffer, colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy region {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {kWidth, kHeight, 1};
    vkCmdCopyImageToBuffer(buffer, colorImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                           &region);
    transitionImage(buffer, colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  });

  std::vector<uint8_t> pixels(static_cast<size_t>(kWidth * kHeight * 4));
  {
    void * mapped = nullptr;
    vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &mapped);
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, stagingMemory);
  }

  if (!nearColor(pixelAt(pixels, 30, 16), 0, 0, 255)) {
    const uint8_t * p = pixelAt(pixels, 30, 16);
    std::cerr << "FAIL: indexed blue quad produced unexpected pixels (B,G,R,A = "
              << static_cast<int>(p[0]) << "," << static_cast<int>(p[1]) << ","
              << static_cast<int>(p[2]) << "," << static_cast<int>(p[3]) << ")"
              << std::endl;
    result = 1;
  }
  if (!nearColor(pixelAt(pixels, 16, 16), 0, 255, 0)) {
    const uint8_t * p = pixelAt(pixels, 16, 16);
    std::cerr << "FAIL: vertex-color triangle produced unexpected pixels (B,G,R,A = "
              << static_cast<int>(p[0]) << "," << static_cast<int>(p[1]) << ","
              << static_cast<int>(p[2]) << "," << static_cast<int>(p[3]) << ")"
              << std::endl;
    result = 1;
  }

  // Repeated execution with an unchanged draw list must be safe.
  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: repeated draw-list execution failed" << std::endl;
    result = 1;
  }

  // clear() changes the generation; replacing geometry must not reuse stale
  // GPU buffers.
  const float replacement[] = {
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
     1.0f,  1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f
  };
  drawlist.clear();
  SoRenderCommand replaced;
  replaced.modelMatrix.makeIdentity();
  replaced.geometry.topology = SO_TOPOLOGY_TRIANGLES;
  replaced.geometry.vertexCount = 4;
  replaced.geometry.indexCount = 6;
  replaced.geometry.positions = replacement;
  replaced.geometry.indices = indices;
  replaced.geometry.vertexStride = sizeof(float) * 3;
  replaced.material.diffuse = SbVec4f(0.0f, 1.0f, 1.0f, 1.0f);
  drawlist.addCommand(replaced);
  if (!backend.render(drawlist, params)) {
    std::cerr << "FAIL: generation-invalidated draw-list execution failed" << std::endl;
    result = 1;
  }

  // Validate the replacement frame: the whole viewport should be cyan, with
  // no leftover green triangle or blue quad.
  {
    oneShot(device, pool, queue, [&](VkCommandBuffer buffer) {
      transitionImage(buffer, colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      VkBufferImageCopy region {};
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.layerCount = 1;
      region.imageExtent = {kWidth, kHeight, 1};
      vkCmdCopyImageToBuffer(buffer, colorImage,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                             &region);
    });
    void * mapped = nullptr;
    vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &mapped);
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, stagingMemory);
    if (!nearColor(pixelAt(pixels, 16, 16), 0, 255, 255)) {
      std::cerr << "FAIL: clear()/generation change reused stale GPU data" << std::endl;
      result = 1;
    }
  }

  // --- Teardown -------------------------------------------------------------
  backend.shutdown();
  if (backend.isInitialized()) {
    std::cerr << "FAIL: backend remained initialized after shutdown" << std::endl;
    result = 1;
  }

  vkDestroyBuffer(device, staging, nullptr);
  vkFreeMemory(device, stagingMemory, nullptr);
  vkDestroyCommandPool(device, pool, nullptr);
  vkDestroyImageView(device, colorView, nullptr);
  vkDestroyImage(device, colorImage, nullptr);
  vkFreeMemory(device, colorMemory, nullptr);
  if (haveDepth) {
    vkDestroyImageView(device, depthView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthMemory, nullptr);
  }
  vkDestroyDevice(device, nullptr);
  if (messenger != VK_NULL_HANDLE) {
    const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy) destroy(instance, messenger, nullptr);
  }
  vkDestroyInstance(instance, nullptr);

  SoDB::finish();
  return result;
}

int
main()
{
  return runTest();
}
