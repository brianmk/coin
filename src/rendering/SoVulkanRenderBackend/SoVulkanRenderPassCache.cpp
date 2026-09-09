// src/rendering/SoVulkanRenderBackend/SoVulkanRenderPassCache.cpp
//
// Render-pass and framebuffer caching for the Vulkan raster backend.  See
// SoVulkanRenderPassCache.h for the design contract.

#include "rendering/SoVulkanRenderBackend/SoVulkanRenderPassCache.h"

SoVulkanRenderPassCache::SoVulkanRenderPassCache() = default;

SoVulkanRenderPassCache::~SoVulkanRenderPassCache()
{
  this->destroyAll();
}

void
SoVulkanRenderPassCache::setDevice(VkDevice device,
                                   const VkAllocationCallbacks * allocator)
{
  this->device_ = device;
  this->allocator_ = allocator;
}

void
SoVulkanRenderPassCache::setDeferredDestroy(
  std::function<void(std::function<void()> &&)> fn)
{
  this->deferDestroyFn_ = std::move(fn);
}

SoVulkanRenderPassCache::RenderPassIdentity
SoVulkanRenderPassCache::identityFor(const SoVulkanRenderTarget & target) const
{
  RenderPassIdentity identity;
  identity.colorFormat = target.colorFormat;
  identity.sampleCount = target.sampleCount;
  identity.colorLayout = target.colorLayout;
  // createRenderPass() only adds a depth attachment when a depth view is
  // present, so a configured-but-viewless depth format must not be part of
  // the identity.
  identity.depthFormat =
    (target.depthImageView != VK_NULL_HANDLE) ? target.depthFormat
                                              : VK_FORMAT_UNDEFINED;
  identity.depthLayout = target.depthLayout;
  return identity;
}

bool
SoVulkanRenderPassCache::createRenderPass(const SoVulkanRenderTarget & target,
                                          VkAttachmentLoadOp colorLoadOp,
                                          VkAttachmentLoadOp depthLoadOp,
                                          VkRenderPass & pass)
{
  VkAttachmentDescription attachments[2];
  uint32_t attachmentCount = 1;

  attachments[0].flags = 0;
  attachments[0].format = target.colorFormat;
  attachments[0].samples = target.sampleCount;
  attachments[0].loadOp = colorLoadOp;
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
    attachments[1].loadOp = depthLoadOp;
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

  return vkCreateRenderPass(this->device_, &ci, this->allocator_, &pass) ==
         VK_SUCCESS;
}

VkRenderPass
SoVulkanRenderPassCache::getOrCreateRenderPass(
  const SoVulkanRenderTarget & target,
  VkAttachmentLoadOp colorLoadOp,
  VkAttachmentLoadOp depthLoadOp)
{
  RenderPassIdentity identity = this->identityFor(target);
  identity.colorLoadOp = colorLoadOp;
  identity.depthLoadOp = depthLoadOp;
  const auto found = this->passCache_.find(identity);
  if (found != this->passCache_.end()) {
    this->renderPass_ = found->second;
    return found->second;
  }

  VkRenderPass pass = VK_NULL_HANDLE;
  if (!this->createRenderPass(target, colorLoadOp, depthLoadOp, pass)) {
    this->renderPass_ = VK_NULL_HANDLE;
    return VK_NULL_HANDLE;
  }
  this->passCache_.emplace(identity, pass);
  this->renderPass_ = pass;
  return pass;
}

bool
SoVulkanRenderPassCache::ensureFramebuffer(const SoVulkanRenderTarget * target,
                                           VkRenderPass renderPass)
{
  if (this->framebuffer_ != VK_NULL_HANDLE &&
      this->framebufferPass_ == renderPass &&
      this->framebufferColorImage_ == target->colorImage &&
      this->framebufferColorView_ == target->colorImageView &&
      this->framebufferDepthImage_ == target->depthImage &&
      this->framebufferDepthView_ == target->depthImageView &&
      this->framebufferExtent_.width == target->extent.width &&
      this->framebufferExtent_.height == target->extent.height) {
    return true;
  }
  if (this->framebuffer_ != VK_NULL_HANDLE) {
    const VkDevice device = this->device_;
    const VkAllocationCallbacks * allocator = this->allocator_;
    const VkFramebuffer oldFramebuffer = this->framebuffer_;
    if (this->deferDestroyFn_) {
      this->deferDestroyFn_([device, allocator, oldFramebuffer]() {
        if (oldFramebuffer != VK_NULL_HANDLE) {
          vkDestroyFramebuffer(device, oldFramebuffer, allocator);
        }
      });
    }
    else {
      vkDestroyFramebuffer(device, oldFramebuffer, allocator);
    }
    this->framebuffer_ = VK_NULL_HANDLE;
  }
  VkFramebufferCreateInfo fci {};
  fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fci.renderPass = renderPass;
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
  if (vkCreateFramebuffer(this->device_, &fci, this->allocator_,
                          &this->framebuffer_) != VK_SUCCESS) {
    return false;
  }
  this->framebufferPass_ = renderPass;
  this->framebufferColorImage_ = target->colorImage;
  this->framebufferColorView_ = target->colorImageView;
  this->framebufferDepthImage_ = target->depthImage;
  this->framebufferDepthView_ = target->depthImageView;
  this->framebufferExtent_ = target->extent;
  return true;
}

void
SoVulkanRenderPassCache::destroyAll()
{
  if (this->framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(this->device_, this->framebuffer_, this->allocator_);
    this->framebuffer_ = VK_NULL_HANDLE;
  }
  for (auto & entry : this->passCache_) {
    if (entry.second != VK_NULL_HANDLE) {
      vkDestroyRenderPass(this->device_, entry.second, this->allocator_);
    }
  }
  this->passCache_.clear();
  this->renderPass_ = VK_NULL_HANDLE;
  this->framebufferPass_ = VK_NULL_HANDLE;
  this->framebufferColorImage_ = VK_NULL_HANDLE;
  this->framebufferColorView_ = VK_NULL_HANDLE;
  this->framebufferDepthImage_ = VK_NULL_HANDLE;
  this->framebufferDepthView_ = VK_NULL_HANDLE;
  this->framebufferExtent_ = {0, 0};
}
