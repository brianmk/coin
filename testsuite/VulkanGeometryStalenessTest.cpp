// testsuite/VulkanGeometryStalenessTest.cpp
//
// Headless end-to-end regression test for the Vulkan-only "stale geometry while
// dragging a sketch" bug.
//
// The bug had two roots and both are user-visible through the rendered image:
//
//   * the retained-IR cache in SoShape did not notice a sibling SoCoordinate3
//     rewrite (the shape is never notified), so the draw list kept replaying the
//     previous vertex stream, and
//   * the Vulkan backend's pointer-identity fast path accepted a rebuilt stream
//     whose address happened to equal the just-freed one, so the GPU buffer was
//     never re-uploaded.
//
// This test renders two offscreen frames and asserts the drawn line's screen row
// actually moves when the only change is (1) a sibling SoCoordinate3 rewrite,
// (2) a shape notify plus a sibling coordinate rewrite (the pointer-ABAb case),
// and (3) a pure shape-field rewrite (an SoIndexedLineSet's coordIndex).  It
// returns COIN_TEST_SKIP_RETURN_CODE (77) when no Vulkan device is available,
// mirroring VulkanSmokeTest.cpp, so CTest reports a skip instead of a failure.

#include <Inventor/SoDB.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/rendering/SoVulkanRenderManager.h>
#include <Inventor/rendering/SoVulkanRenderTarget.h>

#include "rendering/SoVulkanShared.h"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#if !defined(COIN_HAVE_VULKAN_RENDERER) || !COIN_HAVE_VULKAN_RENDERER
#error "VulkanGeometryStalenessTest must only be built with COIN_HAVE_VULKAN_RENDERER"
#endif

#ifndef COIN_TEST_SKIP_RETURN_CODE
#error COIN_TEST_SKIP_RETURN_CODE must match the CTest SKIP_RETURN_CODE property
#endif

namespace {

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 128;

int
skip(const char * reason)
{
  std::cout << "SKIP: " << reason << std::endl;
  return COIN_TEST_SKIP_RETURN_CODE;
}

int
fail(const char * message)
{
  std::cerr << "FAIL: " << message << std::endl;
  return 1;
}

uint32_t
findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits,
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

bool
createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkFormat format,
            VkImageUsageFlags usage, VkImage & image, VkDeviceMemory & memory)
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

VkImageView
createImageView(VkDevice device, VkImage image, VkFormat format,
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

void
transitionImage(VkCommandBuffer commandBuffer, VkImage image,
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

// Statistics of the lit pixels in one host image: weighted centroid row plus the
// raw pixel count, so a missing/black line is distinguishable from a moved one.
struct LineStats {
  bool visible = false;
  double row = 0.0;
  long lit = 0;
};

LineStats
measureLine(const std::vector<uint8_t> & pixels)
{
  LineStats stats;
  double weighted = 0.0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const size_t i = (static_cast<size_t>(y) * kWidth + x) * 4;
      const int b = pixels[i + 0];
      const int g = pixels[i + 1];
      const int r = pixels[i + 2];
      if (r + g + b > 40) {
        ++stats.lit;
        weighted += static_cast<double>(y);
      }
    }
  }
  if (stats.lit > 0) {
    stats.visible = true;
    stats.row = weighted / static_cast<double>(stats.lit);
  }
  return stats;
}

// Owns the headless device, the offscreen color+depth target and the host
// readback.  init() returns 0 on success, 77 to skip, 1 on hard failure.
struct Harness {
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

  void
  oneShot(const std::function<void(VkCommandBuffer)> & fn)
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

  bool
  pickMemory(const VkMemoryRequirements & requirements,
             VkMemoryPropertyFlags desired, uint32_t & index) const
  {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1u << i)) == 0) continue;
      if ((props.memoryTypes[i].propertyFlags & desired) == desired) {
        index = i;
        return true;
      }
    }
    return false;
  }

  int
  init()
  {
    SoDB::init();

    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "coin-vulkan-staleness";
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

  // Copy the offscreen color attachment back into host memory (BGRA order).
  bool
  readback(std::vector<uint8_t> & pixels)
  {
    pixels.assign(static_cast<size_t>(kWidth) * kHeight * 4, 0);
    return SoVulkanShared::dumpImageToHost(
      this->device, this->queue, this->commandPool, nullptr, this->colorImage,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kWidth, kHeight,
      [this](const VkMemoryRequirements & requirements,
             VkMemoryPropertyFlags desired, uint32_t & index) {
        return this->pickMemory(requirements, desired, index);
      },
      [&pixels](const void * mapped) {
        std::memcpy(pixels.data(), mapped, pixels.size());
      });
  }

  void
  shutdown()
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

// What the regression assertion needs per rendered frame.
LineStats
renderAndMeasure(SoVulkanRenderManager & manager, Harness & harness, int & result)
{
  if (result != 0) return {};
  if (!manager.render(TRUE, TRUE)) {
    result = fail("SoVulkanRenderManager::render returned FALSE");
    return {};
  }
  std::vector<uint8_t> pixels;
  if (!harness.readback(pixels)) {
    result = fail("offscreen readback (dumpImageToHost) failed");
    return {};
  }
  return measureLine(pixels);
}

// The line's y moved by 1.0 world units in a 2.0-unit-high orthographic view, so
// the row delta is ~half the image; 16 px is a generous lower bound that still
// fails for a frozen frame.
constexpr double kMinimumRowDelta = 16.0;

void
expectMoved(LineStats before, LineStats after, const char * what, int & result)
{
  if (result != 0) return;
  if (!before.visible || !after.visible) {
    std::cerr << "FAIL: " << what << ": drawn line not visible (before "
              << before.lit << " px, after " << after.lit << " px)"
              << std::endl;
    result = 1;
    return;
  }
  const double delta = std::fabs(after.row - before.row);
  std::cout << "  " << what << ": row " << before.row << " -> " << after.row
            << " (delta " << delta << ")" << std::endl;
  if (delta < kMinimumRowDelta) {
    result = fail(what);
    std::cerr << "  drawn line did not move: row " << before.row << " -> "
              << after.row << " (delta " << delta << " < " << kMinimumRowDelta
              << ")" << std::endl;
  }
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) {
    harness.shutdown();
    return initResult;
  }

  int result = 0;
  // Scope every Coin object -- nodes, render managers -- so they are destructed
  // before SoDB::finish(); a manager outliving the database touches freed
  // statics (cc_mutex_lock assertion).
  {
  // --- Shared scene pieces -------------------------------------------------
  SoSeparator * root = new SoSeparator;
  root->ref();
  SoOrthographicCamera * camera = new SoOrthographicCamera;
  camera->position = SbVec3f(0.0f, 0.0f, 5.0f);
  camera->orientation = SbRotation(SbVec3f(0.0f, 0.0f, 1.0f), 0.0f);
  camera->nearDistance = 0.1f;
  camera->farDistance = 20.0f;
  camera->focalDistance = 5.0f;
  camera->aspectRatio = 1.0f;
  camera->height = 2.0f;
  root->addChild(camera);

  SoMaterial * material = new SoMaterial;
  material->diffuseColor.setValue(0.0f, 0.0f, 0.0f);
  material->emissiveColor.setValue(1.0f, 0.0f, 0.0f);
  root->addChild(material);

  SoCoordinate3 * coords = new SoCoordinate3;
  coords->point.set1Value(0, -0.8f, -0.5f, 0.0f);
  coords->point.set1Value(1, 0.8f, -0.5f, 0.0f);
  root->addChild(coords);

  SoLineSet * lines = new SoLineSet;
  lines->numVertices.set1Value(0, 2);
  root->addChild(lines);

  const SbViewportRegion viewport(kWidth, kHeight);

  SoVulkanRenderManager manager;
  manager.setSceneGraph(root);
  manager.setCamera(camera);
  manager.setViewportRegion(viewport);
  manager.setRenderTarget(&harness.target);
  manager.setBackgroundColor(SbColor4f(0.0f, 0.0f, 0.0f, 1.0f));
  manager.setAutoClipping(SoVulkanRenderManager::NO_AUTO_CLIPPING);
  manager.setClearEnabled(TRUE, TRUE);

  if (!manager.initialize(&harness.deviceContext)) {
    result = fail("SoVulkanRenderManager::initialize failed");
  }

  if (result == 0) {
    // Warm-up: the first frame may still contain init-time state changes.
    renderAndMeasure(manager, harness, result);
    const LineStats baseline = renderAndMeasure(manager, harness, result);

    // ---- Case 1: sibling SoCoordinate3 rewrite only -----------------------
    // This is the Sketcher drag: the shape is never notified, only the
    // inherited coordinate element changes.
    coords->point.set1Value(0, -0.8f, 0.5f, 0.0f);
    coords->point.set1Value(1, 0.8f, 0.5f, 0.0f);
    const LineStats afterSiblingEdit = renderAndMeasure(manager, harness, result);
    expectMoved(baseline, afterSiblingEdit,
                "sibling SoCoordinate3 rewrite",
                result);

    // ---- Case 2: shape field write plus a sibling coordinate rewrite ------
    // The shape *is* notified (SoShape::notify drops the retained runs), so the
    // IR rebuilds legitimately; the rebuild commonly gets the just-freed stream
    // address back, which is exactly what the backend's pointer-identity ABA
    // guard (retainedGeneration) has to reject.
    coords->point.set1Value(0, -0.8f, -0.5f, 0.0f);
    coords->point.set1Value(1, 0.8f, -0.5f, 0.0f);
    lines->touch();
    const LineStats resetFrame = renderAndMeasure(manager, harness, result);
    expectMoved(afterSiblingEdit, resetFrame,
                "shape-notified reset", result);

    coords->point.set1Value(0, -0.8f, 0.5f, 0.0f);
    coords->point.set1Value(1, 0.8f, 0.5f, 0.0f);
    lines->touch();
    const LineStats afterNotifiedEdit = renderAndMeasure(manager, harness, result);
    expectMoved(resetFrame, afterNotifiedEdit,
                "shape field write + sibling coordinate rewrite", result);
  }

  if (result == 0) {
    manager.shutdown();

    // ---- Case 3: pure shape-field rewrite (SoIndexedLineSet coordIndex) ---
    // The coordinates are never touched: only the shape's own field selects a
    // different pair, yet a stale backend buffer keeps the old row.
    SoSeparator * indexedRoot = new SoSeparator;
    indexedRoot->ref();
    SoOrthographicCamera * indexedCamera = new SoOrthographicCamera;
    indexedCamera->position = SbVec3f(0.0f, 0.0f, 5.0f);
    indexedCamera->orientation = SbRotation(SbVec3f(0.0f, 0.0f, 1.0f), 0.0f);
    indexedCamera->nearDistance = 0.1f;
    indexedCamera->farDistance = 20.0f;
    indexedCamera->focalDistance = 5.0f;
    indexedCamera->aspectRatio = 1.0f;
    indexedCamera->height = 2.0f;
    indexedRoot->addChild(indexedCamera);

    SoMaterial * indexedMaterial = new SoMaterial;
    indexedMaterial->diffuseColor.setValue(0.0f, 0.0f, 0.0f);
    indexedMaterial->emissiveColor.setValue(1.0f, 0.0f, 0.0f);
    indexedRoot->addChild(indexedMaterial);

    SoCoordinate3 * indexedCoords = new SoCoordinate3;
    indexedCoords->point.set1Value(0, -0.8f, -0.5f, 0.0f);
    indexedCoords->point.set1Value(1, 0.8f, -0.5f, 0.0f);
    indexedCoords->point.set1Value(2, -0.8f, 0.5f, 0.0f);
    indexedCoords->point.set1Value(3, 0.8f, 0.5f, 0.0f);
    indexedRoot->addChild(indexedCoords);

    SoIndexedLineSet * indexed = new SoIndexedLineSet;
    indexed->coordIndex.set1Value(0, 0);
    indexed->coordIndex.set1Value(1, 1);
    indexed->coordIndex.set1Value(2, -1);
    indexedRoot->addChild(indexed);

    SoVulkanRenderManager indexedManager;
    indexedManager.setSceneGraph(indexedRoot);
    indexedManager.setCamera(indexedCamera);
    indexedManager.setViewportRegion(viewport);
    indexedManager.setRenderTarget(&harness.target);
    indexedManager.setBackgroundColor(SbColor4f(0.0f, 0.0f, 0.0f, 1.0f));
    indexedManager.setAutoClipping(SoVulkanRenderManager::NO_AUTO_CLIPPING);
    indexedManager.setClearEnabled(TRUE, TRUE);
    if (!indexedManager.initialize(&harness.deviceContext)) {
      result = fail("SoVulkanRenderManager::initialize failed (indexed scene)");
    }
    else {
      renderAndMeasure(indexedManager, harness, result);
      const LineStats low = renderAndMeasure(indexedManager, harness, result);

      indexed->coordIndex.set1Value(0, 2);
      indexed->coordIndex.set1Value(1, 3);
      const LineStats high = renderAndMeasure(indexedManager, harness, result);
      expectMoved(low, high,
                  "shape-field (coordIndex) rewrite",
                  result);
      indexedManager.shutdown();
    }

    indexedRoot->unref();
  }

  manager.shutdown();
  root->unref();
  } // Coin objects destroyed here, before SoDB::finish()

  SoDB::finish();
  harness.shutdown();

  if (result == 0) {
    std::cout << "OK: Vulkan retained-geometry staleness test passed"
              << std::endl;
  }
  return result;
}
