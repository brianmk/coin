// src/rendering/SoVulkanRenderBackend.cpp

#include "rendering/SoVulkanRenderBackend.h"

#include <Inventor/C/glue/gl.h>
#include <Inventor/errors/SoDebugError.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#include "vulkan/visual/Fragment.spv.h"
#include "vulkan/visual/Vertex.spv.h"

namespace {

// Fixed interleaved vertex layout shared by every retained command.
//
//   offset 0 : vec3 position
//   offset 12: vec3 normal
//   offset 24: vec4 color
//   offset 40: vec2 texcoord
//
// This keeps a single static vertex-input description usable across all
// pipelines, mirroring the GL backend's VAO-per-command bookkeeping without
// any per-command vertex-state objects.
constexpr uint32_t VULKAN_VERTEX_STRIDE = 48;
constexpr int MAX_VERTEX_COUNT = 10000000;
constexpr int MAX_SHADER_LIGHTS = 8;

struct alignas(16) VulkanPushConstants {
  float proj[16];       // projection matrix (view/model live in the UBO)
  float color[4];       // uniform diffuse color
  float flags[4];       // x = useVertexColor
                        // y = vertexColorAlphaIncludesOpacity
                        // z = textureEnabled
                        // w = textureAlphaIncludesOpacity
  float texParams[4];   // x = textureModel, y = alphaTestFunction,
                        // z = alphaTestReference
  float texBlend[4];    // texture blend color
};

// std140 mirror of the VisualBlock uniform in Vertex.glsl.  The layout must
// match the shader byte-for-byte; the C++ side uses plain float arrays with
// alignas(16) so vec3 members consume 16 bytes like std140 vec4s.
struct alignas(16) VulkanVisualUbo {
  float view[16];                 // offset 0
  float model[16];                // offset 64
  float emissive[4];              // offset 128
  float ambientLight[4];          // offset 144
  float materialAmbient[4];       // offset 160
  float materialSpecular[4];      // offset 176
  float materialParams[4];        // offset 192
  float lightType[MAX_SHADER_LIGHTS * 4];        // offset 208
  float lightColor[MAX_SHADER_LIGHTS * 4];       // offset 336
  float lightDirection[MAX_SHADER_LIGHTS * 4];   // offset 464
  float lightPosition[MAX_SHADER_LIGHTS * 4];    // offset 592
  float lightAttenuation[MAX_SHADER_LIGHTS * 4]; // offset 720
  float lightSpotParams[MAX_SHADER_LIGHTS * 4];  // offset 848
};
static_assert(sizeof(VulkanVisualUbo) == 976,
              "VulkanVisualUbo must match VisualBlock std140 layout");

VkCompareOp
depthFunctionToVk(const SoDepthFunction function)
{
  switch (function) {
  case SO_DEPTH_NEVER: return VK_COMPARE_OP_NEVER;
  case SO_DEPTH_ALWAYS: return VK_COMPARE_OP_ALWAYS;
  case SO_DEPTH_LESS: return VK_COMPARE_OP_LESS;
  case SO_DEPTH_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
  case SO_DEPTH_EQUAL: return VK_COMPARE_OP_EQUAL;
  case SO_DEPTH_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case SO_DEPTH_GREATER: return VK_COMPARE_OP_GREATER;
  case SO_DEPTH_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
  default: return VK_COMPARE_OP_LESS_OR_EQUAL;
  }
}

VkCompareOp
stencilFunctionToVk(const SoStencilFunction function)
{
  switch (function) {
  case SO_STENCIL_FUNC_NEVER: return VK_COMPARE_OP_NEVER;
  case SO_STENCIL_FUNC_ALWAYS: return VK_COMPARE_OP_ALWAYS;
  case SO_STENCIL_FUNC_LESS: return VK_COMPARE_OP_LESS;
  case SO_STENCIL_FUNC_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
  case SO_STENCIL_FUNC_EQUAL: return VK_COMPARE_OP_EQUAL;
  case SO_STENCIL_FUNC_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case SO_STENCIL_FUNC_GREATER: return VK_COMPARE_OP_GREATER;
  case SO_STENCIL_FUNC_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
  default: return VK_COMPARE_OP_ALWAYS;
  }
}

VkStencilOp
stencilOpToVk(const SoStencilOp op)
{
  switch (op) {
  case SO_STENCIL_OP_ZERO: return VK_STENCIL_OP_ZERO;
  case SO_STENCIL_OP_REPLACE: return VK_STENCIL_OP_REPLACE;
  case SO_STENCIL_OP_INCREMENT: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
  case SO_STENCIL_OP_DECREMENT: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
  case SO_STENCIL_OP_INVERT: return VK_STENCIL_OP_INVERT;
  case SO_STENCIL_OP_INCREMENT_WRAP: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
  case SO_STENCIL_OP_DECREMENT_WRAP: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
  case SO_STENCIL_OP_KEEP:
  default: return VK_STENCIL_OP_KEEP;
  }
}

VkBlendFactor
blendFactorToVk(const SoBlendFactor factor)
{
  switch (factor) {
  case SO_BLEND_FACTOR_ZERO: return VK_BLEND_FACTOR_ZERO;
  case SO_BLEND_FACTOR_ONE: return VK_BLEND_FACTOR_ONE;
  case SO_BLEND_FACTOR_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
  case SO_BLEND_FACTOR_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_DST_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case SO_BLEND_FACTOR_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case SO_BLEND_FACTOR_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  case SO_BLEND_FACTOR_CONSTANT_COLOR:
    return VK_BLEND_FACTOR_CONSTANT_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
  case SO_BLEND_FACTOR_CONSTANT_ALPHA:
    return VK_BLEND_FACTOR_CONSTANT_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
  case SO_BLEND_FACTOR_SRC_ALPHA_SATURATE:
    return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
  case SO_BLEND_FACTOR_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
  case SO_BLEND_FACTOR_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA;
  case SO_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  default: return VK_BLEND_FACTOR_ONE;
  }
}

VkBlendOp
blendEquationToVk(const SoBlendEquation equation)
{
  switch (equation) {
  case SO_BLEND_EQUATION_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
  case SO_BLEND_EQUATION_REVERSE_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
  case SO_BLEND_EQUATION_MIN: return VK_BLEND_OP_MIN;
  case SO_BLEND_EQUATION_MAX: return VK_BLEND_OP_MAX;
  case SO_BLEND_EQUATION_ADD:
  default: return VK_BLEND_OP_ADD;
  }
}

VkPrimitiveTopology
topologyToVk(const SoPrimitiveTopology topology)
{
  switch (topology) {
  case SO_TOPOLOGY_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  case SO_TOPOLOGY_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  case SO_TOPOLOGY_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case SO_TOPOLOGY_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  case SO_TOPOLOGY_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
}

VkFormat
textureFormatToVk(const int numComponents)
{
  switch (numComponents) {
  case 1: return VK_FORMAT_R8_UNORM;
  case 2: return VK_FORMAT_R8G8_UNORM;
  case 3: return VK_FORMAT_R8G8B8_UNORM;
  case 4:
  default: return VK_FORMAT_R8G8B8A8_UNORM;
  }
}

VkFilter
textureFilterToVk(const SoTextureFilter filter)
{
  switch (filter) {
  case SO_TEXTURE_FILTER_NEAREST:
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
  case SO_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    return VK_FILTER_NEAREST;
  case SO_TEXTURE_FILTER_LINEAR:
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
  case SO_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
  default:
    return VK_FILTER_LINEAR;
  }
}

VkSamplerAddressMode
textureWrapToVk(const SoTextureWrap wrap)
{
  switch (wrap) {
  case SO_TEXTURE_WRAP_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case SO_TEXTURE_WRAP_CLAMP_TO_EDGE:
  case SO_TEXTURE_WRAP_CLAMP_TO_BORDER:
  default:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

VkImageView
createImageView(VkDevice device,
                VkImage image,
                VkFormat format,
                VkImageAspectFlags aspect,
                const VkAllocationCallbacks * allocator)
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
  vkCreateImageView(device, &ci, allocator, &view);
  return view;
}

} // namespace

SoVulkanRenderBackend::SoVulkanRenderBackend()
{
}

SoVulkanRenderBackend::~SoVulkanRenderBackend()
{
  if (this->isInitialized()) this->shutdown();
}

const char *
SoVulkanRenderBackend::getName() const
{
  return "VulkanRenderBackend";
}

SbBool
SoVulkanRenderBackend::initialize(const SoRenderBackendInitParams & params)
{
  if (this->isInitialized()) return TRUE;

  this->setInitParams(params);
  const auto * deviceContext =
    static_cast<const SoVulkanDeviceContext *>(params.userData);
  if (!deviceContext || deviceContext->instance == VK_NULL_HANDLE ||
      deviceContext->physicalDevice == VK_NULL_HANDLE ||
      deviceContext->device == VK_NULL_HANDLE ||
      deviceContext->graphicsQueue == VK_NULL_HANDLE) {
    this->emitError(
      "SoVulkanRenderBackend requires a SoVulkanDeviceContext in "
      "SoRenderBackendInitParams::userData");
    return FALSE;
  }

  this->instance = deviceContext->instance;
  this->physicalDevice = deviceContext->physicalDevice;
  this->device = deviceContext->device;
  this->queue = deviceContext->graphicsQueue;
  this->queueFamilyIndex = deviceContext->graphicsQueueFamilyIndex;
  this->allocator = deviceContext->allocator;

  if (!this->createCommandPool()) {
    this->emitError("failed to create Vulkan command pool");
    this->shutdown();
    return FALSE;
  }

  if (!this->createDescriptorSetLayout()) {
    this->emitError("failed to create Vulkan descriptor set layout");
    this->shutdown();
    return FALSE;
  }

  if (!this->createDescriptorPool()) {
    this->emitError("failed to create Vulkan descriptor pool");
    this->shutdown();
    return FALSE;
  }

  if (!this->createLightingUniformBuffer()) {
    this->emitError("failed to create Vulkan lighting uniform buffer");
    this->shutdown();
    return FALSE;
  }

  if (!this->createWhiteTexture()) {
    this->emitError("failed to create Vulkan white fallback texture");
    this->shutdown();
    return FALSE;
  }

  if (!this->createPipelineLayout()) {
    this->emitError("failed to create Vulkan pipeline layout");
    this->shutdown();
    return FALSE;
  }

  if (!this->createShaders(this->vertexModule, this->fragmentModule)) {
    this->emitError("failed to create Vulkan shader modules");
    this->shutdown();
    return FALSE;
  }

  this->setInitialized(TRUE);
  this->emitLog("initialized");
  return TRUE;
}

bool
SoVulkanRenderBackend::createCommandPool()
{
  VkCommandPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
             VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  ci.queueFamilyIndex = this->queueFamilyIndex;
  if (vkCreateCommandPool(this->device, &ci, this->allocator,
                          &this->commandPool) != VK_SUCCESS) {
    return false;
  }

  VkCommandBufferAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = this->commandPool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  return vkAllocateCommandBuffers(this->device, &ai,
                                  &this->commandBuffer) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createDescriptorSetLayout()
{
  VkDescriptorSetLayoutBinding bindings[2] {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  bindings[0].pImmutableSamplers = nullptr;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[1].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = 2;
  ci.pBindings = bindings;
  return vkCreateDescriptorSetLayout(this->device, &ci, this->allocator,
                                     &this->descriptorSetLayout) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createDescriptorPool()
{
  VkDescriptorPoolSize poolSizes[2] {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = 1024;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = 1024;

  VkDescriptorPoolCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  ci.flags = 0;
  ci.maxSets = 1024;
  ci.poolSizeCount = 2;
  ci.pPoolSizes = poolSizes;
  return vkCreateDescriptorPool(this->device, &ci, this->allocator,
                                &this->descriptorPool) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::allocateTextureDescriptorSet(VkImageView view,
                                                    VkSampler sampler,
                                                    VkDescriptorSet & set)
{
  VkDescriptorSetAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = this->descriptorPool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &this->descriptorSetLayout;
  if (vkAllocateDescriptorSets(this->device, &ai, &set) != VK_SUCCESS) {
    return false;
  }

  VkDescriptorBufferInfo bufferInfo {};
  bufferInfo.buffer = this->lightingBuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = sizeof(VulkanVisualUbo);

  VkDescriptorImageInfo imageInfo {};
  imageInfo.sampler = sampler;
  imageInfo.imageView = view;
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkWriteDescriptorSet writes[2] {};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = set;
  writes[0].dstBinding = 0;
  writes[0].dstArrayElement = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[0].pBufferInfo = &bufferInfo;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = set;
  writes[1].dstBinding = 1;
  writes[1].dstArrayElement = 0;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(this->device, 2, writes, 0, nullptr);
  return true;
}

bool
SoVulkanRenderBackend::createLightingUniformBuffer()
{
  if (!this->createBuffer(sizeof(VulkanVisualUbo),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          this->lightingBuffer, this->lightingMemory,
                          nullptr)) {
    return false;
  }
  if (vkMapMemory(this->device, this->lightingMemory, 0, sizeof(VulkanVisualUbo),
                  0, &this->lightingMapped) != VK_SUCCESS) {
    this->emitError("createLightingUniformBuffer: vkMapMemory failed");
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createWhiteTexture()
{
  const uint8_t white = 255;
  const uint32_t extent = 1;

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = VK_FORMAT_R8G8B8A8_UNORM;
  ci.extent = {extent, extent, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(this->device, &ci, this->allocator, &this->whiteImage) !=
      VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(this->device, this->whiteImage, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = 0;
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &props);
  bool found = false;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      ai.memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    this->emitError("createWhiteTexture: no device-local memory type");
    return false;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator,
                       &this->whiteImageMemory) != VK_SUCCESS) {
    return false;
  }
  vkBindImageMemory(this->device, this->whiteImage, this->whiteImageMemory, 0);

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging,
                          stagingMemory, &white)) {
    return false;
  }

  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer uploadBuffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(this->device, &allocInfo, &uploadBuffer);
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(uploadBuffer, &bi);

  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = this->whiteImage;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(uploadBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region {};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {extent, extent, 1};
  vkCmdCopyBufferToImage(uploadBuffer, staging, this->whiteImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(uploadBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);

  vkEndCommandBuffer(uploadBuffer);
  VkSubmitInfo submit {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &uploadBuffer;
  vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(this->queue);
  vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);
  vkDestroyBuffer(this->device, staging, this->allocator);
  vkFreeMemory(this->device, stagingMemory, this->allocator);

  this->whiteImageView =
    createImageView(this->device, this->whiteImage, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (this->whiteImageView == VK_NULL_HANDLE) {
    return false;
  }

  SoTextureData fallback;
  fallback.minFilter = SO_TEXTURE_FILTER_NEAREST;
  fallback.magFilter = SO_TEXTURE_FILTER_NEAREST;
  fallback.wrapS = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  fallback.wrapT = SO_TEXTURE_WRAP_CLAMP_TO_EDGE;
  if (!this->createSampler(fallback, this->whiteSampler)) {
    return false;
  }
  return this->allocateTextureDescriptorSet(this->whiteImageView,
                                            this->whiteSampler,
                                            this->whiteDescriptorSet);
}

bool
SoVulkanRenderBackend::createPipelineLayout()
{
  constexpr VkPushConstantRange range {
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    0,
    sizeof(VulkanPushConstants)
  };

  VkPipelineLayoutCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  ci.setLayoutCount = this->descriptorSetLayout != VK_NULL_HANDLE ? 1u : 0u;
  ci.pSetLayouts = &this->descriptorSetLayout;
  ci.pushConstantRangeCount = 1;
  ci.pPushConstantRanges = &range;
  return vkCreatePipelineLayout(this->device, &ci, this->allocator,
                                &this->pipelineLayout) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::createShaders(VkShaderModule & vertex,
                                     VkShaderModule & fragment)
{
  auto load = [this](const uint32_t * code, size_t count,
                     VkShaderModule & module) {
    VkShaderModuleCreateInfo ci {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = count * sizeof(uint32_t);
    ci.pCode = code;
    return vkCreateShaderModule(this->device, &ci, this->allocator,
                                &module) == VK_SUCCESS;
  };

  vertex = VK_NULL_HANDLE;
  fragment = VK_NULL_HANDLE;
  if (!load(coin_vulkan_visual_vertex_spirv,
            coin_vulkan_visual_vertex_spirv_count, vertex)) {
    return false;
  }
  if (!load(coin_vulkan_visual_fragment_spirv,
            coin_vulkan_visual_fragment_spirv_count, fragment)) {
    vkDestroyShaderModule(this->device, vertex, this->allocator);
    vertex = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool
SoVulkanRenderBackend::createRenderPass(const SoVulkanRenderTarget & target,
                                        VkRenderPass & pass)
{
  VkAttachmentDescription attachments[2];
  uint32_t attachmentCount = 1;

  attachments[0].flags = 0;
  attachments[0].format = target.colorFormat;
  attachments[0].samples = target.sampleCount;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = target.colorLayout;
  attachments[0].finalLayout = target.colorLayout;

  VkAttachmentReference colorRef {};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef {};
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;
  if (hasDepth) {
    attachments[1].flags = 0;
    attachments[1].format = target.depthFormat;
    attachments[1].samples = target.sampleCount;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = target.depthLayout;
    attachments[1].finalLayout = target.depthLayout;
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachmentCount = 2;
  }

  VkSubpassDescription subpass {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

  VkRenderPassCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = attachmentCount;
  ci.pAttachments = attachments;
  ci.subpassCount = 1;
  ci.pSubpasses = &subpass;
  ci.dependencyCount = 0;
  ci.pDependencies = nullptr;

  return vkCreateRenderPass(this->device, &ci, this->allocator, &pass) ==
         VK_SUCCESS;
}

bool
SoVulkanRenderBackend::getOrCreatePipeline(const SoRenderCommand & command,
                                           const SoVulkanRenderTarget & target,
                                           VkRenderPass pass,
                                           VkPipeline & pipeline,
                                           const bool transparent)
{
  // Pipelines are immutable in Vulkan.  Key the cache on every retained
  // state value that changes the created pipeline so commands of different
  // topology, fill mode, depth/blend state, or sample count never reuse an
  // incompatible pipeline.  Shading model, vertex-color, texture, and
  // lighting remain uniform/push-constant concerns in this milestone and do
  // not need to participate in the key yet.
  const bool blending = transparent || command.state.blend.enabled ||
                        command.material.diffuse[3] < 0.999f;
  PipelineKey key;
  key.renderPass = pass;
  key.topology = command.geometry.topology;
  key.fillMode = command.state.raster.fillMode;
  key.cullMode = command.state.raster.cullMode;
  key.depthTestEnable = command.state.depth.enabled;
  key.depthWriteEnable = !transparent && command.state.depth.writeEnabled;
  key.depthFunction = command.state.depth.func;
  key.blendEnable = blending;
  key.sampleCount = target.sampleCount;
  if (blending) {
    key.blendSrcRGB = command.state.blend.srcRGBFactor;
    key.blendDstRGB = command.state.blend.dstRGBFactor;
    key.blendSrcAlpha = command.state.blend.srcAlphaFactor;
    key.blendDstAlpha = command.state.blend.dstAlphaFactor;
    key.blendEquationRGB = command.state.blend.rgbEquation;
    key.blendEquationAlpha = command.state.blend.alphaEquation;
  }
  const SoStencilState & stencil = command.state.stencil;
  key.stencilEnable = stencil.enabled;
  if (stencil.enabled) {
    key.stencilFunction = stencil.function;
    key.stencilReference = stencil.reference;
    key.stencilCompareMask = stencil.compareMask;
    key.stencilWriteMask = stencil.writeMask;
    key.stencilFailOp = stencil.failOp;
    key.stencilZFailOp = stencil.zfailOp;
    key.stencilZPassOp = stencil.zpassOp;
  }

  const auto found = this->pipelineCache.find(key);
  if (found != this->pipelineCache.end()) {
    pipeline = found->second;
    return pipeline != VK_NULL_HANDLE;
  }

  VkPipelineShaderStageCreateInfo stages[2] {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = this->vertexModule;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = this->fragmentModule;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding {};
  binding.binding = 0;
  binding.stride = VULKAN_VERTEX_STRIDE;
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attributes[4] {};
  attributes[0].location = 0;
  attributes[0].binding = 0;
  attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[0].offset = 0;
  attributes[1].location = 1;
  attributes[1].binding = 0;
  attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributes[1].offset = 12;
  attributes[2].location = 2;
  attributes[2].binding = 0;
  attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  attributes[2].offset = 24;
  attributes[3].location = 3;
  attributes[3].binding = 0;
  attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
  attributes[3].offset = 40;

  VkPipelineVertexInputStateCreateInfo vertexInput {};
  vertexInput.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInput.vertexBindingDescriptionCount = 1;
  vertexInput.pVertexBindingDescriptions = &binding;
  vertexInput.vertexAttributeDescriptionCount = 4;
  vertexInput.pVertexAttributeDescriptions = attributes;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
  inputAssembly.sType =
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = topologyToVk(command.geometry.topology);
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState {};
  viewportState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization {};
  rasterization.sType =
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization.depthClampEnable = VK_FALSE;
  rasterization.rasterizerDiscardEnable = VK_FALSE;
  const uint8_t fillMode = command.state.raster.fillMode;
  rasterization.polygonMode =
    fillMode == 1 ? VK_POLYGON_MODE_LINE
                  : (fillMode == 2 ? VK_POLYGON_MODE_POINT
                                   : VK_POLYGON_MODE_FILL);
  // The vertex shader flips Y to match Coin's bottom-left origin; compensate
  // the winding so back-face culling matches the GL pipeline.
  rasterization.cullMode =
    command.state.raster.cullMode ? VK_CULL_MODE_BACK_BIT
                                  : VK_CULL_MODE_NONE;
  rasterization.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterization.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample {};
  multisample.sType =
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = target.sampleCount;

  VkPipelineDepthStencilStateCreateInfo depthStencil {};
  depthStencil.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable =
    command.state.depth.enabled ? VK_TRUE : VK_FALSE;
  depthStencil.depthWriteEnable =
    (!transparent && command.state.depth.writeEnabled) ? VK_TRUE : VK_FALSE;
  depthStencil.depthCompareOp = depthFunctionToVk(command.state.depth.func);
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = stencil.enabled ? VK_TRUE : VK_FALSE;
  VkStencilOpState stencilState {};
  if (stencil.enabled) {
    stencilState.failOp = stencilOpToVk(stencil.failOp);
    stencilState.passOp = stencilOpToVk(stencil.zpassOp);
    stencilState.depthFailOp = stencilOpToVk(stencil.zfailOp);
    stencilState.compareOp = stencilFunctionToVk(stencil.function);
    stencilState.compareMask = stencil.compareMask;
    stencilState.writeMask = stencil.writeMask;
    stencilState.reference = stencil.reference;
  }
  depthStencil.front = stencilState;
  depthStencil.back = stencilState;

  VkPipelineColorBlendAttachmentState blendAttachment {};
  blendAttachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendAttachment.blendEnable = blending ? VK_TRUE : VK_FALSE;
  if (command.state.blend.enabled) {
    blendAttachment.srcColorBlendFactor =
      blendFactorToVk(command.state.blend.srcRGBFactor);
    blendAttachment.dstColorBlendFactor =
      blendFactorToVk(command.state.blend.dstRGBFactor);
    blendAttachment.colorBlendOp =
      blendEquationToVk(command.state.blend.rgbEquation);
    blendAttachment.srcAlphaBlendFactor =
      blendFactorToVk(command.state.blend.srcAlphaFactor);
    blendAttachment.dstAlphaBlendFactor =
      blendFactorToVk(command.state.blend.dstAlphaFactor);
    blendAttachment.alphaBlendOp =
      blendEquationToVk(command.state.blend.alphaEquation);
  }
  else {
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  }

  VkPipelineColorBlendStateCreateInfo colorBlend {};
  colorBlend.sType =
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlend.logicOpEnable = VK_FALSE;
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &blendAttachment;

  const VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicState {};
  dynamicState.sType =
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkGraphicsPipelineCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  ci.stageCount = 2;
  ci.pStages = stages;
  ci.pVertexInputState = &vertexInput;
  ci.pInputAssemblyState = &inputAssembly;
  ci.pViewportState = &viewportState;
  ci.pRasterizationState = &rasterization;
  ci.pMultisampleState = &multisample;
  ci.pDepthStencilState = &depthStencil;
  ci.pColorBlendState = &colorBlend;
  ci.pDynamicState = &dynamicState;
  ci.layout = this->pipelineLayout;
  ci.renderPass = pass;
  ci.subpass = 0;

  VkPipeline created = VK_NULL_HANDLE;
  const VkResult result =
    vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &ci,
                              this->allocator, &created);
  if (result != VK_SUCCESS) {
    this->emitError("failed to create Vulkan graphics pipeline");
    this->pipelineCache[key] = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    return false;
  }
  this->pipelineCache[key] = created;
  pipeline = created;
  return true;
}

// --- Geometry cache -------------------------------------------------------

VulkanCachedCommand &
SoVulkanRenderBackend::getOrCreateCache(const SoRenderCommand * command)
{
  const auto found = this->commandToCache.find(command);
  if (found != this->commandToCache.end()) {
    return this->gpuCache[found->second];
  }
  const size_t index = this->gpuCache.size();
  this->gpuCache.emplace_back();
  this->commandToCache[command] = index;
  return this->gpuCache.back();
}

bool
SoVulkanRenderBackend::createBuffer(VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkBuffer & buffer,
                                    VkDeviceMemory & memory,
                                    const void * data)
{
  VkBufferCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(this->device, &ci, this->allocator, &buffer) !=
      VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(this->device, buffer, &requirements);

  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  // Milestone: no per-memory-type discovery; assume a host-visible coherent
  // heap.  A follow-up replaces this with a memory-type index lookup.
  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &memoryProperties);
  uint32_t memoryTypeIndex = 0;
  const VkMemoryPropertyFlags desired =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  bool found = false;
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (memoryProperties.memoryTypes[i].propertyFlags & desired) == desired) {
      memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  ai.memoryTypeIndex = memoryTypeIndex;

  if (vkAllocateMemory(this->device, &ai, this->allocator, &memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(this->device, buffer, this->allocator);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(this->device, buffer, memory, 0);

  if (data) {
    void * mapped = nullptr;
    if (vkMapMemory(this->device, memory, 0, size, 0, &mapped) != VK_SUCCESS) {
      this->emitError("createBuffer: vkMapMemory failed");
      vkDestroyBuffer(this->device, buffer, this->allocator);
      vkFreeMemory(this->device, memory, this->allocator);
      buffer = VK_NULL_HANDLE;
      memory = VK_NULL_HANDLE;
      return false;
    }
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(this->device, memory);
  }
  return true;
}

void
SoVulkanRenderBackend::uploadGeometry(VulkanCachedCommand & entry,
                                      const SoRenderCommand & command)
{
  const SoGeometryDesc & geometry = command.geometry;
  const uint32_t vertexCount = geometry.vertexCount;

  // Pack interleaved vertices with deterministic defaults for absent streams.
  std::vector<float> vertices(static_cast<size_t>(vertexCount) * 12);
  const uint32_t posStride = geometry.vertexStride
    ? geometry.vertexStride : sizeof(float) * 3;
  const uint32_t posStrideFloats = posStride / sizeof(float);
  const uint32_t normalStrideFloats =
    (geometry.normals ? posStrideFloats : 0);
  const uint32_t texcoordStride = geometry.texcoordStride
    ? geometry.texcoordStride : sizeof(float) * 4;
  const uint32_t texcoordStrideFloats = texcoordStride / sizeof(float);

  for (uint32_t i = 0; i < vertexCount; ++i) {
    float * out = vertices.data() + static_cast<size_t>(i) * 12;

    const float * pos = geometry.positions + static_cast<size_t>(i) * posStrideFloats;
    out[0] = pos[0];
    out[1] = pos[1];
    out[2] = pos[2];

    if (geometry.normals && i < geometry.normalCount) {
      const float * normal = geometry.normals + static_cast<size_t>(i) * normalStrideFloats;
      out[3] = normal[0];
      out[4] = normal[1];
      out[5] = normal[2];
    }
    else {
      out[3] = 0.0f;
      out[4] = 0.0f;
      out[5] = 1.0f;
    }

    if (geometry.colors) {
      const float * color = geometry.colors + static_cast<size_t>(i) * 4;
      out[6] = color[0];
      out[7] = color[1];
      out[8] = color[2];
      out[9] = color[3];
    }
    else {
      out[6] = 1.0f;
      out[7] = 1.0f;
      out[8] = 1.0f;
      out[9] = 1.0f;
    }

    if (geometry.texcoords) {
      const float * uv = geometry.texcoords + static_cast<size_t>(i) * texcoordStrideFloats;
      out[10] = uv[0];
      out[11] = uv[1];
    }
    else {
      out[10] = 0.0f;
      out[11] = 0.0f;
    }
  }

  const VkDeviceSize vertexBytes =
    static_cast<VkDeviceSize>(vertexCount) * VULKAN_VERTEX_STRIDE;
  if (!this->createBuffer(vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          entry.vertexBuffer, entry.vertexMemory,
                          vertices.data())) {
    this->emitError("uploadGeometry: failed to create vertex buffer");
    return;
  }

  if (geometry.indexCount && geometry.indices) {
    const VkDeviceSize indexBytes =
      static_cast<VkDeviceSize>(geometry.indexCount) * sizeof(uint32_t);
    if (!this->createBuffer(indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            entry.indexBuffer, entry.indexMemory,
                            geometry.indices)) {
      this->emitError("uploadGeometry: failed to create index buffer");
      return;
    }
  }

  entry.posKey = geometry.positions;
  entry.normalKey = geometry.normals;
  entry.colorKey = geometry.colors;
  entry.texcoordKey = geometry.texcoords;
  entry.idxKey = geometry.indices;
  entry.vertexCount = vertexCount;
  entry.indexCount = geometry.indexCount;
  entry.vertexStride = posStride;
  entry.texcoordStride = geometry.texcoordStride;
  entry.normalCount = geometry.normalCount;
}

void
SoVulkanRenderBackend::destroyCacheEntry(VulkanCachedCommand & entry)
{
  if (entry.indexBuffer) {
    vkDestroyBuffer(this->device, entry.indexBuffer, this->allocator);
    entry.indexBuffer = VK_NULL_HANDLE;
  }
  if (entry.indexMemory) {
    vkFreeMemory(this->device, entry.indexMemory, this->allocator);
    entry.indexMemory = VK_NULL_HANDLE;
  }
  if (entry.vertexBuffer) {
    vkDestroyBuffer(this->device, entry.vertexBuffer, this->allocator);
    entry.vertexBuffer = VK_NULL_HANDLE;
  }
  if (entry.vertexMemory) {
    vkFreeMemory(this->device, entry.vertexMemory, this->allocator);
    entry.vertexMemory = VK_NULL_HANDLE;
  }
  entry = VulkanCachedCommand();
}

void
SoVulkanRenderBackend::invalidateCache()
{
  for (VulkanCachedCommand & entry : this->gpuCache) {
    this->destroyCacheEntry(entry);
  }
  this->gpuCache.clear();
  this->commandToCache.clear();
  this->invalidateTextureCache();
  this->cachedCommandCount = 0;
  this->haveCacheGeneration = false;
}

void
SoVulkanRenderBackend::updateGeometryCache(const SoDrawList & drawlist)
{
  const uint32_t generation = drawlist.getGeneration();
  if ((this->haveCacheGeneration && this->cacheGeneration != generation) ||
      (this->haveCacheGeneration &&
       this->cachedCommandCount !=
         static_cast<size_t>(drawlist.getNumCommands()))) {
    this->invalidateCache();
  }
  this->cacheGeneration = generation;
  this->haveCacheGeneration = true;
  this->cachedCommandCount = static_cast<size_t>(drawlist.getNumCommands());

  for (int i = 0; i < drawlist.getNumCommands(); ++i) {
    const SoRenderCommand & command = drawlist.getCommand(i);
    const SoGeometryDesc & geometry = command.geometry;
    if (!geometry.positions || geometry.vertexCount == 0 ||
        geometry.vertexCount > MAX_VERTEX_COUNT) {
      continue;
    }

    VulkanCachedCommand & entry = this->getOrCreateCache(&command);
    const uint32_t vertexStride = geometry.vertexStride
      ? geometry.vertexStride : sizeof(float) * 3;
    const bool geometryMatches = entry.vertexBuffer != VK_NULL_HANDLE &&
      entry.cacheGeneration == generation &&
      entry.posKey == geometry.positions &&
      entry.normalKey == geometry.normals &&
      entry.colorKey == geometry.colors &&
      entry.texcoordKey == geometry.texcoords &&
      entry.idxKey == geometry.indices &&
      entry.vertexCount == geometry.vertexCount &&
      entry.indexCount == geometry.indexCount &&
      entry.normalCount == geometry.normalCount &&
      entry.vertexStride == vertexStride &&
      entry.texcoordStride == geometry.texcoordStride;
    if (!geometryMatches) {
      this->destroyCacheEntry(entry);
      this->uploadGeometry(entry, command);
      entry.cacheGeneration = generation;
    }

    const SoTextureData & texture = command.material.texture;
    if (texture.pixels && texture.width > 0 && texture.height > 0) {
      VulkanCachedTexture & texEntry = this->getOrCreateTexture(&command);
      const bool textureMatches = texEntry.image != VK_NULL_HANDLE &&
        texEntry.cacheGeneration == generation &&
        texEntry.pixelsKey == texture.pixels &&
        texEntry.width == texture.width &&
        texEntry.height == texture.height &&
        texEntry.numComponents == texture.numComponents &&
        texEntry.minFilter == texture.minFilter &&
        texEntry.magFilter == texture.magFilter &&
        texEntry.wrapS == texture.wrapS &&
        texEntry.wrapT == texture.wrapT &&
        texEntry.model == texture.model;
      if (!textureMatches) {
        this->destroyTextureEntry(texEntry);
        this->uploadTexture(texEntry, texture);
        texEntry.pixelsKey = texture.pixels;
        texEntry.width = texture.width;
        texEntry.height = texture.height;
        texEntry.numComponents = texture.numComponents;
        texEntry.minFilter = texture.minFilter;
        texEntry.magFilter = texture.magFilter;
        texEntry.wrapS = texture.wrapS;
        texEntry.wrapT = texture.wrapT;
        texEntry.model = texture.model;
        texEntry.cacheGeneration = generation;
      }
    }
  }
}

// --- Texture cache --------------------------------------------------------

void
SoVulkanRenderBackend::destroyTextureEntry(VulkanCachedTexture & entry)
{
  if (entry.sampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, entry.sampler, this->allocator);
    entry.sampler = VK_NULL_HANDLE;
  }
  if (entry.view != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, entry.view, this->allocator);
    entry.view = VK_NULL_HANDLE;
  }
  if (entry.image != VK_NULL_HANDLE) {
    vkDestroyImage(this->device, entry.image, this->allocator);
    entry.image = VK_NULL_HANDLE;
  }
  if (entry.memory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, entry.memory, this->allocator);
    entry.memory = VK_NULL_HANDLE;
  }
  entry = VulkanCachedTexture();
}

void
SoVulkanRenderBackend::invalidateTextureCache()
{
  for (VulkanCachedTexture & entry : this->textureCache) {
    this->destroyTextureEntry(entry);
  }
  this->textureCache.clear();
  this->commandToTexture.clear();
}

VulkanCachedTexture &
SoVulkanRenderBackend::getOrCreateTexture(const SoRenderCommand * command)
{
  const auto found = this->commandToTexture.find(command);
  if (found != this->commandToTexture.end()) {
    return this->textureCache[found->second];
  }
  const size_t index = this->textureCache.size();
  this->textureCache.emplace_back();
  this->commandToTexture[command] = index;
  return this->textureCache.back();
}

bool
SoVulkanRenderBackend::createSampler(const SoTextureData & texture,
                                     VkSampler & sampler)
{
  VkSamplerCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  ci.magFilter = textureFilterToVk(texture.magFilter);
  ci.minFilter = textureFilterToVk(texture.minFilter);
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  ci.addressModeU = textureWrapToVk(texture.wrapS);
  ci.addressModeV = textureWrapToVk(texture.wrapT);
  ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.mipLodBias = 0.0f;
  ci.anisotropyEnable = VK_FALSE;
  ci.maxAnisotropy = 1.0f;
  ci.compareEnable = VK_FALSE;
  ci.minLod = 0.0f;
  ci.maxLod = 0.0f;
  ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  ci.unnormalizedCoordinates = VK_FALSE;
  return vkCreateSampler(this->device, &ci, this->allocator, &sampler) ==
         VK_SUCCESS;
}

bool
SoVulkanRenderBackend::uploadTexture(VulkanCachedTexture & entry,
                                     const SoTextureData & texture)
{
  const VkFormat format = textureFormatToVk(texture.numComponents);
  const VkDeviceSize byteSize =
    static_cast<VkDeviceSize>(texture.width) * texture.height *
    texture.numComponents;

  VkImageCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {static_cast<uint32_t>(texture.width),
               static_cast<uint32_t>(texture.height), 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(this->device, &ci, this->allocator, &entry.image) !=
      VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements requirements;
  vkGetImageMemoryRequirements(this->device, entry.image, &requirements);
  VkMemoryAllocateInfo ai {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = requirements.size;
  ai.memoryTypeIndex = 0;
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(this->physicalDevice, &props);
  bool found = false;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) &&
        (props.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      ai.memoryTypeIndex = i;
      found = true;
      break;
    }
  }
  if (!found) {
    this->emitError("uploadTexture: no device-local memory type");
    this->destroyTextureEntry(entry);
    return false;
  }
  if (vkAllocateMemory(this->device, &ai, this->allocator, &entry.memory) !=
      VK_SUCCESS) {
    this->destroyTextureEntry(entry);
    return false;
  }
  vkBindImageMemory(this->device, entry.image, entry.memory, 0);

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  if (!this->createBuffer(byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          staging, stagingMemory, texture.pixels)) {
    this->emitError("uploadTexture: staging buffer creation failed");
    this->destroyTextureEntry(entry);
    return false;
  }

  // Upload via a temporary command buffer so the copy is retired before the
  // texture is sampled in the render command buffer.
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = this->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer uploadBuffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(this->device, &allocInfo, &uploadBuffer);

  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(uploadBuffer, &bi);

  VkImageMemoryBarrier barrier {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = entry.image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(uploadBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {static_cast<uint32_t>(texture.width),
                        static_cast<uint32_t>(texture.height), 1};
  vkCmdCopyBufferToImage(uploadBuffer, staging, entry.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(uploadBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);

  vkEndCommandBuffer(uploadBuffer);
  VkSubmitInfo submit {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &uploadBuffer;
  vkQueueSubmit(this->queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(this->queue);
  vkFreeCommandBuffers(this->device, this->commandPool, 1, &uploadBuffer);

  vkDestroyBuffer(this->device, staging, this->allocator);
  vkFreeMemory(this->device, stagingMemory, this->allocator);

  entry.view = createImageView(this->device, entry.image, format,
                               VK_IMAGE_ASPECT_COLOR_BIT, this->allocator);
  if (entry.view == VK_NULL_HANDLE ||
      !this->createSampler(texture, entry.sampler) ||
      !this->allocateTextureDescriptorSet(entry.view, entry.sampler,
                                          entry.descriptorSet)) {
    this->emitError("uploadTexture: view/sampler/descriptor creation failed");
    this->destroyTextureEntry(entry);
    return false;
  }
  return true;
}

VkDescriptorSet
SoVulkanRenderBackend::resolveTextureSet(const SoRenderCommand & command)
{
  const auto found = this->commandToTexture.find(&command);
  if (found != this->commandToTexture.end() &&
      this->textureCache[found->second].descriptorSet != VK_NULL_HANDLE) {
    return this->textureCache[found->second].descriptorSet;
  }
  return this->whiteDescriptorSet;
}

// --- Render recording -----------------------------------------------------

void
SoVulkanRenderBackend::applyViewport(const SoRenderParams & params,
                                     const SoVulkanRenderTarget & target)
{
  const SbVec2s & origin = params.viewport.getViewportOriginPixels();
  const SbVec2s & size = params.viewport.getViewportSizePixels();

  VkViewport viewport {};
  viewport.x = static_cast<float>(origin[0]);
  viewport.y = static_cast<float>(origin[1]);
  viewport.width = static_cast<float>(size[0]);
  viewport.height = static_cast<float>(size[1]);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(this->activeCommandBuffer, 0, 1, &viewport);

  VkRect2D scissor {};
  scissor.offset = {0, 0};
  scissor.extent = target.extent;
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

void
SoVulkanRenderBackend::applyScissor(const SoRenderCommand & command,
                                    const SoVulkanRenderTarget & target)
{
  VkRect2D scissor {};
  const SoRasterState & raster = command.state.raster;
  if (raster.scissorEnabled && raster.scissorWidth > 0 &&
      raster.scissorHeight > 0) {
    scissor.offset = {static_cast<int32_t>(raster.scissorX),
                      static_cast<int32_t>(raster.scissorY)};
    scissor.extent = {static_cast<uint32_t>(raster.scissorWidth),
                      static_cast<uint32_t>(raster.scissorHeight)};
  }
  else {
    scissor.offset = {0, 0};
    scissor.extent = target.extent;
  }
  vkCmdSetScissor(this->activeCommandBuffer, 0, 1, &scissor);
}

void
SoVulkanRenderBackend::recordClear(const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target)
{
  const bool hasDepth = target.depthImageView != VK_NULL_HANDLE &&
                        target.depthFormat != VK_FORMAT_UNDEFINED;

  VkClearAttachment attachments[3];
  uint32_t attachmentCount = 0;

  if (params.flags & SO_PARAM_CLEAR_WINDOW) {
    const SbColor4f & color = params.clearColor;
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.color.float32[0] = color[0];
    clear.clearValue.color.float32[1] = color[1];
    clear.clearValue.color.float32[2] = color[2];
    clear.clearValue.color.float32[3] = color[3];
    attachments[attachmentCount++] = clear;
  }

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_DEPTH)) {
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.depthStencil.depth = params.clearDepth;
    clear.clearValue.depthStencil.stencil = 0;
    attachments[attachmentCount++] = clear;
  }

  if (hasDepth && (params.flags & SO_PARAM_CLEAR_STENCIL)) {
    VkClearAttachment clear {};
    clear.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    clear.colorAttachment = 0;
    clear.clearValue.depthStencil.depth = 0;
    clear.clearValue.depthStencil.stencil = params.clearStencil;
    attachments[attachmentCount++] = clear;
  }

  if (attachmentCount == 0) return;

  VkClearRect rect {};
  rect.rect.offset = {0, 0};
  rect.rect.extent = target.extent;
  rect.baseArrayLayer = 0;
  rect.layerCount = 1;
  vkCmdClearAttachments(this->activeCommandBuffer, attachmentCount, attachments, 1,
                        &rect);
}

void
SoVulkanRenderBackend::updateLightingUniforms(const SoDrawList & drawlist,
                                              const SoRenderCommand & command,
                                              const SoRenderParams & params)
{
  VulkanVisualUbo ubo {};

  SbMat m;
  params.viewMatrix.getValue(m);
  std::memcpy(ubo.view, &m[0][0], sizeof(float) * 16);
  command.modelMatrix.getValue(m);
  std::memcpy(ubo.model, &m[0][0], sizeof(float) * 16);

  const SoMaterialData & material = command.material;
  ubo.emissive[0] = material.emissive[0];
  ubo.emissive[1] = material.emissive[1];
  ubo.emissive[2] = material.emissive[2];
  ubo.emissive[3] = 1.0f;
  ubo.materialAmbient[0] = material.ambient[0];
  ubo.materialAmbient[1] = material.ambient[1];
  ubo.materialAmbient[2] = material.ambient[2];
  ubo.materialAmbient[3] = 1.0f;
  ubo.materialSpecular[0] = material.specular[0];
  ubo.materialSpecular[1] = material.specular[1];
  ubo.materialSpecular[2] = material.specular[2];
  ubo.materialSpecular[3] = 1.0f;
  ubo.materialParams[0] = material.shininess;
  ubo.materialParams[1] = material.twoSidedLighting ? 1.0f : 0.0f;
  ubo.materialParams[3] =
    material.shadingModel == SO_SHADING_LEGACY_GOURAUD ? 1.0f : 0.0f;

  const SoLightingData * lighting = drawlist.getLighting(command.lightingHandle);
  static const SoLightingData emptyLighting;
  if (!lighting) {
    lighting = &emptyLighting;
    if (command.lightingHandle != 0) {
      static std::once_flag invalidHandleWarning;
      std::call_once(invalidHandleWarning, []() {
        SoDebugError::postWarning(
          "SoVulkanRenderBackend::updateLightingUniforms",
          "Draw command references missing lighting data; no headlight is "
          "synthesized.");
      });
    }
  }

  ubo.ambientLight[0] = lighting->ambient[0];
  ubo.ambientLight[1] = lighting->ambient[1];
  ubo.ambientLight[2] = lighting->ambient[2];
  ubo.ambientLight[3] = 1.0f;

  const int count = std::min<int>(
    static_cast<int>(lighting->lights.size()), MAX_SHADER_LIGHTS);
  if (static_cast<int>(lighting->lights.size()) > MAX_SHADER_LIGHTS) {
    static std::once_flag lightLimitWarning;
    std::call_once(lightLimitWarning, []() {
      SoDebugError::postWarning(
        "SoVulkanRenderBackend::updateLightingUniforms",
        "The Visual program supports eight lights; additional retained "
        "lights are ignored by this executor.");
    });
  }
  ubo.materialParams[2] = static_cast<float>(count);

  for (int i = 0; i < count; ++i) {
    const SoLightData & light = lighting->lights[static_cast<size_t>(i)];
    float * type = ubo.lightType + i * 4;
    type[0] = static_cast<float>(light.type);
    type[1] = type[2] = 0.0f;
    type[3] = 1.0f;

    float * color = ubo.lightColor + i * 4;
    color[0] = light.color[0];
    color[1] = light.color[1];
    color[2] = light.color[2];
    color[3] = 1.0f;

    float * direction = ubo.lightDirection + i * 4;
    direction[0] = light.direction[0];
    direction[1] = light.direction[1];
    direction[2] = light.direction[2];
    direction[3] = 1.0f;

    float * position = ubo.lightPosition + i * 4;
    position[0] = light.position[0];
    position[1] = light.position[1];
    position[2] = light.position[2];
    position[3] = 1.0f;

    float * attenuation = ubo.lightAttenuation + i * 4;
    attenuation[0] = light.attenuation[0];
    attenuation[1] = light.attenuation[1];
    attenuation[2] = light.attenuation[2];
    attenuation[3] = 1.0f;

    float * spot = ubo.lightSpotParams + i * 4;
    spot[0] = light.spotCutoffCos;
    spot[1] = light.spotExponent;
    spot[2] = 0.0f;
    spot[3] = 1.0f;
  }

  if (this->lightingMapped) {
    std::memcpy(this->lightingMapped, &ubo, sizeof(ubo));
  }
}

void
SoVulkanRenderBackend::recordDrawCommand(const SoDrawList & drawlist,
                                         const SoRenderCommand & command,
                                         const SoVulkanRenderTarget & target,
                                         const SoRenderParams & params,
                                         VkRenderPass pass,
                                         const bool transparent)
{
  if (!command.geometry.positions || command.geometry.vertexCount == 0) return;
  const auto found = this->commandToCache.find(&command);
  if (found == this->commandToCache.end()) return;
  const VulkanCachedCommand & entry = this->gpuCache[found->second];
  if (entry.vertexBuffer == VK_NULL_HANDLE) return;

  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!this->getOrCreatePipeline(command, target, pass, pipeline, transparent) ||
      pipeline == VK_NULL_HANDLE) {
    return;
  }
  vkCmdBindPipeline(this->activeCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline);
  this->applyScissor(command, target);

  VkDescriptorSet textureSet = this->resolveTextureSet(command);
  if (textureSet != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(this->activeCommandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            this->pipelineLayout, 0, 1, &textureSet, 0,
                            nullptr);
  }

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(this->activeCommandBuffer, 0, 1, &entry.vertexBuffer,
                         &offset);
  const bool indexed =
    entry.indexBuffer != VK_NULL_HANDLE && command.geometry.indexCount &&
    command.geometry.indices;
  if (indexed) {
    vkCmdBindIndexBuffer(this->activeCommandBuffer, entry.indexBuffer, 0,
                         VK_INDEX_TYPE_UINT32);
  }

  this->updateLightingUniforms(drawlist, command, params);

  VulkanPushConstants push {};
  SbMat projValue;
  params.projMatrix.getValue(projValue);
  std::memcpy(push.proj, &projValue[0][0], sizeof(float) * 16);
  const SbVec4f & color = command.material.diffuse;
  push.color[0] = color[0];
  push.color[1] = color[1];
  push.color[2] = color[2];
  push.color[3] = color[3];
  push.flags[0] = entry.colorKey ? 1.0f : 0.0f;
  push.flags[1] =
    command.material.vertexColorAlphaIncludesOpacity ? 1.0f : 0.0f;
  const bool textured = command.material.texture.pixels &&
                        command.material.texture.width > 0 &&
                        command.material.texture.height > 0;
  push.flags[2] = textured ? 1.0f : 0.0f;
  push.flags[3] = command.material.textureAlphaIncludesOpacity
                    ? 1.0f : 0.0f;
  push.texParams[0] =
    static_cast<float>(command.material.texture.model);
  push.texParams[1] =
    static_cast<float>(command.state.alphaTest.function);
  push.texParams[2] = command.state.alphaTest.reference;
  push.texParams[3] =
    (command.material.flags & SO_MAT_IS_PIXEL_TEXT) ? 1.0f : 0.0f;
  const SbVec4f & blendColor = command.material.texture.blendColor;
  push.texBlend[0] = blendColor[0];
  push.texBlend[1] = blendColor[1];
  push.texBlend[2] = blendColor[2];
  push.texBlend[3] = blendColor[3];

  vkCmdPushConstants(this->activeCommandBuffer, this->pipelineLayout,
                     VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(push), &push);

  if (indexed) {
    vkCmdDrawIndexed(this->activeCommandBuffer, command.geometry.indexCount, 1, 0,
                     0, 0);
  }
  else {
    vkCmdDraw(this->activeCommandBuffer, command.geometry.vertexCount, 1, 0, 0);
  }
}

bool
SoVulkanRenderBackend::beginCommandBuffer()
{
  VkCommandBufferBeginInfo bi {};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return vkBeginCommandBuffer(this->commandBuffer, &bi) == VK_SUCCESS;
}

bool
SoVulkanRenderBackend::endAndSubmit()
{
  if (vkEndCommandBuffer(this->commandBuffer) != VK_SUCCESS) return false;

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &this->commandBuffer;
  if (vkQueueSubmit(this->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
    return false;
  }
  return vkQueueWaitIdle(this->queue) == VK_SUCCESS;
}

// --- Lifecycle ------------------------------------------------------------

void
SoVulkanRenderBackend::shutdown()
{
  if (!this->isInitialized()) return;

  vkQueueWaitIdle(this->queue);

  this->invalidateCache();

  for (auto & entry : this->pipelineCache) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(this->device, entry.second, this->allocator);
    }
  }
  this->pipelineCache.clear();

  if (this->renderPass != VK_NULL_HANDLE) {
    vkDestroyRenderPass(this->device, this->renderPass, this->allocator);
    this->renderPass = VK_NULL_HANDLE;
  }
  if (this->fragmentModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->fragmentModule, this->allocator);
    this->fragmentModule = VK_NULL_HANDLE;
  }
  if (this->vertexModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(this->device, this->vertexModule, this->allocator);
    this->vertexModule = VK_NULL_HANDLE;
  }
  if (this->pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(this->device, this->pipelineLayout, this->allocator);
    this->pipelineLayout = VK_NULL_HANDLE;
  }
  if (this->lightingBuffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device, this->lightingBuffer, this->allocator);
    this->lightingBuffer = VK_NULL_HANDLE;
  }
  if (this->lightingMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->lightingMemory, this->allocator);
    this->lightingMemory = VK_NULL_HANDLE;
  }
  this->lightingMapped = nullptr;
  if (this->whiteSampler != VK_NULL_HANDLE) {
    vkDestroySampler(this->device, this->whiteSampler, this->allocator);
    this->whiteSampler = VK_NULL_HANDLE;
  }
  if (this->whiteImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device, this->whiteImageView, this->allocator);
    this->whiteImageView = VK_NULL_HANDLE;
  }
  if (this->whiteImage != VK_NULL_HANDLE) {
    vkDestroyImage(this->device, this->whiteImage, this->allocator);
    this->whiteImage = VK_NULL_HANDLE;
  }
  if (this->whiteImageMemory != VK_NULL_HANDLE) {
    vkFreeMemory(this->device, this->whiteImageMemory, this->allocator);
    this->whiteImageMemory = VK_NULL_HANDLE;
  }
  this->whiteDescriptorSet = VK_NULL_HANDLE;
  if (this->descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(this->device, this->descriptorPool, this->allocator);
    this->descriptorPool = VK_NULL_HANDLE;
  }
  if (this->descriptorSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(this->device, this->descriptorSetLayout,
                                 this->allocator);
    this->descriptorSetLayout = VK_NULL_HANDLE;
  }
  if (this->commandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(this->device, this->commandPool, 1,
                         &this->commandBuffer);
    this->commandBuffer = VK_NULL_HANDLE;
  }
  if (this->commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(this->device, this->commandPool, this->allocator);
    this->commandPool = VK_NULL_HANDLE;
  }

  this->instance = VK_NULL_HANDLE;
  this->physicalDevice = VK_NULL_HANDLE;
  this->device = VK_NULL_HANDLE;
  this->queue = VK_NULL_HANDLE;
  this->allocator = nullptr;

  this->setInitialized(FALSE);
  this->emitLog("shutdown");
}

SbBool
SoVulkanRenderBackend::render(const SoDrawList & drawlist,
                              const SoRenderParams & params)
{
  if (!this->isInitialized()) {
    this->emitError("render called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "render called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }

  this->debugValidateDrawList(drawlist);

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  // (Re)create the render pass when the destination changes identity.
  const bool targetChanged =
    this->renderPass == VK_NULL_HANDLE ||
    this->renderPassColorImage != target->colorImage ||
    this->renderPassColorView != target->colorImageView ||
    this->renderPassDepthImage != target->depthImage ||
    this->renderPassDepthView != target->depthImageView ||
    this->renderPassExtent.width != target->extent.width ||
    this->renderPassExtent.height != target->extent.height;
  if (targetChanged) {
    if (this->renderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(this->device, this->renderPass, this->allocator);
      this->renderPass = VK_NULL_HANDLE;
    }
    if (!this->createRenderPass(*target, this->renderPass)) {
      this->emitError("failed to create Vulkan render pass");
      return FALSE;
    }
    this->renderPassColorImage = target->colorImage;
    this->renderPassColorView = target->colorImageView;
    this->renderPassDepthImage = target->depthImage;
    this->renderPassDepthView = target->depthImageView;
    this->renderPassExtent = target->extent;
  }

  this->updateGeometryCache(drawlist);

  if (!this->beginCommandBuffer()) {
    this->emitError("failed to begin Vulkan command buffer");
    return FALSE;
  }

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkFramebufferCreateInfo fci {};
  fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fci.renderPass = this->renderPass;
  fci.attachmentCount =
    (target->depthImageView != VK_NULL_HANDLE &&
     target->depthFormat != VK_FORMAT_UNDEFINED)
      ? 2u : 1u;
  const VkImageView attachments[] = {
    target->colorImageView,
    target->depthImageView,
  };
  fci.pAttachments = attachments;
  fci.width = target->extent.width;
  fci.height = target->extent.height;
  fci.layers = 1;
  if (vkCreateFramebuffer(this->device, &fci, this->allocator, &framebuffer) !=
      VK_SUCCESS) {
    this->emitError("failed to create Vulkan framebuffer");
    vkEndCommandBuffer(this->commandBuffer);
    return FALSE;
  }

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = this->renderPass;
  rpbi.framebuffer = framebuffer;
  rpbi.renderArea.offset = {0, 0};
  rpbi.renderArea.extent = target->extent;
  rpbi.clearValueCount = 0;
  rpbi.pClearValues = nullptr;

  vkCmdBeginRenderPass(this->commandBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

  this->activeCommandBuffer = this->commandBuffer;
  this->recordFrame(drawlist, params, *target, this->renderPass);
  this->activeCommandBuffer = VK_NULL_HANDLE;

  vkCmdEndRenderPass(this->commandBuffer);

  const bool submitted = this->endAndSubmit();
  vkDestroyFramebuffer(this->device, framebuffer, this->allocator);
  if (!submitted) {
    this->emitError("failed to submit Vulkan command buffer");
    return FALSE;
  }
  return TRUE;
}

SbBool
SoVulkanRenderBackend::renderExternal(const SoDrawList & drawlist,
                                      const SoRenderParams & params,
                                      VkCommandBuffer commandBuffer,
                                      VkRenderPass renderPass)
{
  if (!this->isInitialized()) {
    this->emitError("renderExternal called before backend initialization");
    return FALSE;
  }
  if (!params.renderTarget) {
    this->emitError(
      "renderExternal called without a SoVulkanRenderTarget in "
      "SoRenderParams::renderTarget");
    return FALSE;
  }
  if (commandBuffer == VK_NULL_HANDLE || renderPass == VK_NULL_HANDLE) {
    this->emitError(
      "renderExternal called without a command buffer and render pass");
    return FALSE;
  }

  this->debugValidateDrawList(drawlist);

  const auto * target =
    static_cast<const SoVulkanRenderTarget *>(params.renderTarget);
  if (target->colorImageView == VK_NULL_HANDLE ||
      target->colorImage == VK_NULL_HANDLE || target->extent.width == 0 ||
      target->extent.height == 0) {
    this->emitError("invalid Vulkan render target");
    return FALSE;
  }

  this->updateGeometryCache(drawlist);

  this->activeCommandBuffer = commandBuffer;
  const bool recorded = this->recordFrame(drawlist, params, *target, renderPass);
  this->activeCommandBuffer = VK_NULL_HANDLE;
  return recorded ? TRUE : FALSE;
}

bool
SoVulkanRenderBackend::recordFrame(const SoDrawList & drawlist,
                                   const SoRenderParams & params,
                                   const SoVulkanRenderTarget & target,
                                   VkRenderPass renderPass)
{
  this->applyViewport(params, target);
  this->recordClear(params, target);

  // Opaque then transparent, honoring the draw-list sort order.
  const std::vector<int> & order = drawlist.getSortedOrder();
  for (int passIndex = 0; passIndex < 2; ++passIndex) {
    const bool transparent = passIndex == 1;
    for (int i = 0; i < drawlist.getNumCommands(); ++i) {
      const int index =
        i < static_cast<int>(order.size()) ? order[i] : i;
      const SoRenderCommand & command = drawlist.getCommand(index);
      const bool isTransparent = command.pass == SO_RENDERPASS_TRANSPARENT;
      if (isTransparent != transparent) continue;
      this->recordDrawCommand(drawlist, command, target, params, renderPass,
                              transparent);
    }
  }
  return true;
}
