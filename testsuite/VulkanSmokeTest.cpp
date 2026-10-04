// testsuite/VulkanSmokeTest.cpp
//
// Headless smoke test for Coin's Vulkan retained-renderer path.  Creates a
// bare Vulkan device (no window system or swapchain), initializes the public
// SoVulkanRenderManager against it, records one offscreen frame, and checks
// that the retained-IR front-end (SoIRRenderAction -> SoDrawList) produces
// commands.  Follows the EGLBindingTest convention: returns
// COIN_TEST_SKIP_RETURN_CODE (77) when no Vulkan driver is available so CTest
// reports the test as skipped rather than failed.

#include <Inventor/SoDB.h>
#include <Inventor/actions/SoIRRenderAction.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/rendering/SoRenderIR.h>
#include <Inventor/rendering/SoVulkanRenderManager.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#if !defined(COIN_HAVE_VULKAN_RENDERER) || !COIN_HAVE_VULKAN_RENDERER
#error "VulkanSmokeTest must only be built with COIN_HAVE_VULKAN_RENDERER"
#endif

#ifndef COIN_TEST_SKIP_RETURN_CODE
#error COIN_TEST_SKIP_RETURN_CODE must match the CTest SKIP_RETURN_CODE property
#endif

namespace {

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 64;

int skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return COIN_TEST_SKIP_RETURN_CODE;
}

int fail(const char * message)
{
  std::cerr << "FAIL: " << message << std::endl;
  return 1;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits,
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

bool createImage(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkFormat format, VkImageUsageFlags usage, VkImage & image,
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
  if (vkCreateImage(device, &ci, nullptr, &image) != VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(device, image, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
  ci.subresourceRange.levelCount = 1;
  ci.subresourceRange.layerCount = 1;
  VkImageView view = VK_NULL_HANDLE;
  vkCreateImageView(device, &ci, nullptr, &view);
  return view;
}

void transitionImage(VkCommandBuffer commandBuffer, VkImage image,
                     VkImageAspectFlags aspect, VkImageLayout newLayout)
{
  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = aspect;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Owns the headless device, the offscreen color+depth target and the manager
// under test; init() returns 0 on success, 77 to skip, 1 on hard failure.
struct Smoke {
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

  void oneShot(const std::function<void(VkCommandBuffer)> & fn)
  {
    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = this->commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(this->device, &ai, &cb);

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    fn(cb);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(this->queue);
    vkFreeCommandBuffers(this->device, this->commandPool, 1, &cb);
  }

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
    if (vkCreateInstance(&instanceInfo, nullptr, &this->instance) != VK_SUCCESS) {
      return skip("could not create a Vulkan instance");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(this->instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      return skip("no Vulkan physical devices");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(this->instance, &deviceCount, devices.data());

    bool found = false;
    for (VkPhysicalDevice candidate : devices) {
      uint32_t familyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
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
      return skip("no Vulkan device with a graphics queue");
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = this->queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    // Enable fillModeNonSolid when advertised: the backend's wireframe/point
    // pipelines render in VK_POLYGON_MODE_LINE/POINT, exactly like the widget.
    VkPhysicalDeviceFeatures supported {};
    vkGetPhysicalDeviceFeatures(this->physicalDevice, &supported);
    VkPhysicalDeviceFeatures enabled {};
    enabled.fillModeNonSolid = supported.fillModeNonSolid;

    VkDeviceCreateInfo deviceInfo {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &enabled;
    if (vkCreateDevice(this->physicalDevice, &deviceInfo, nullptr,
                       &this->device) != VK_SUCCESS) {
      return skip("could not create a Vulkan logical device");
    }
    vkGetDeviceQueue(this->device, this->queueFamily, 0, &this->queue);

    if (!createImage(this->device, this->physicalDevice,
                     VK_FORMAT_B8G8R8A8_UNORM,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     this->colorImage, this->colorMemory)) {
      return skip("could not allocate a color image");
    }
    this->colorView = createImageView(this->device, this->colorImage,
                                      VK_FORMAT_B8G8R8A8_UNORM,
                                      VK_IMAGE_ASPECT_COLOR_BIT);

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

    // Move the attachments into their attachment layouts once, up front, as the
    // embedding app must before handing a target to the manager.
    this->oneShot([&](VkCommandBuffer cb) {
      transitionImage(cb, this->colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      if (this->haveDepth) {
        transitionImage(cb, this->depthImage,
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      }
    });

    this->deviceContext = {};
    this->deviceContext.instance = this->instance;
    this->deviceContext.physicalDevice = this->physicalDevice;
    this->deviceContext.device = this->device;
    this->deviceContext.graphicsQueue = this->queue;
    this->deviceContext.graphicsQueueFamilyIndex = this->queueFamily;
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(this->physicalDevice, &props);
    this->deviceContext.apiVersion = props.apiVersion;
    this->deviceContext.capsValid = false;

    this->target = {};
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
    return 0;
  }

  void shutdown()
  {
    if (this->commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(this->device, this->commandPool, nullptr);
      this->commandPool = VK_NULL_HANDLE;
    }
    if (this->colorView != VK_NULL_HANDLE) {
      vkDestroyImageView(this->device, this->colorView, nullptr);
    }
    if (this->colorImage != VK_NULL_HANDLE) {
      vkDestroyImage(this->device, this->colorImage, nullptr);
    }
    if (this->colorMemory != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, this->colorMemory, nullptr);
    }
    if (this->depthView != VK_NULL_HANDLE) {
      vkDestroyImageView(this->device, this->depthView, nullptr);
    }
    if (this->depthImage != VK_NULL_HANDLE) {
      vkDestroyImage(this->device, this->depthImage, nullptr);
    }
    if (this->depthMemory != VK_NULL_HANDLE) {
      vkFreeMemory(this->device, this->depthMemory, nullptr);
    }
    if (this->device != VK_NULL_HANDLE) {
      vkDeviceWaitIdle(this->device);
      vkDestroyDevice(this->device, nullptr);
    }
    if (this->instance != VK_NULL_HANDLE) {
      vkDestroyInstance(this->instance, nullptr);
    }
    this->device = VK_NULL_HANDLE;
    this->instance = VK_NULL_HANDLE;
  }
};

} // namespace

int main()
{
  Smoke smoke;
  const int initResult = smoke.init();
  if (initResult != 0) {
    smoke.shutdown();
    return initResult;
  }

  int result = 0;

  SoSeparator * root = new SoSeparator;
  root->ref();
  SoPerspectiveCamera * camera = new SoPerspectiveCamera;
  root->addChild(camera);
  root->addChild(new SoDirectionalLight);
  root->addChild(new SoCube);

  const SbViewportRegion viewport(kWidth, kHeight);

  // The retained-IR front-end must turn the scene into draw commands without a
  // device -- this is the shader-independent half of the renderer.
  {
    SoIRRenderAction action(viewport);
    action.apply(root);
    if (action.getDrawList().getNumCommands() <= 0) {
      result = fail("SoIRRenderAction produced an empty draw list");
    }
  }

  if (result == 0) {
    SoVulkanRenderManager manager;
    manager.setSceneGraph(root);
    manager.setCamera(camera);
    manager.setViewportRegion(viewport);
    manager.setRenderTarget(&smoke.target);

    if (!manager.initialize(&smoke.deviceContext)) {
      result = fail("SoVulkanRenderManager::initialize failed");
    }
    else {
      camera->viewAll(root, viewport);
      if (!manager.render(TRUE, TRUE)) {
        result = fail("SoVulkanRenderManager::render returned FALSE");
      }
      else if (manager.getRenderFrameCount() == 0) {
        result = fail("render did not advance the frame counter");
      }
      manager.shutdown();
    }
  }

  root->unref();
  SoDB::finish();
  smoke.shutdown();

  if (result == 0) {
    std::cout << "OK: Vulkan retained-renderer smoke test passed" << std::endl;
  }
  return result;
}
