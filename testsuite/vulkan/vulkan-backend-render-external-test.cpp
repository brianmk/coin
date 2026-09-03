// testsuite/vulkan-backend-render-external-test.cpp
//
// Exercises SoVulkanRenderBackend::renderExternal(): recording the retained
// draw list into a caller-owned command buffer and render pass.  Unlike the
// owned render() path, the backend must not begin/end a command buffer,
// begin/end the render pass, create a framebuffer, or submit to the queue.

#include "VulkanTestHarness.h"

using namespace vulkan_test;

namespace {

const float kTrianglePositions[] = {
  -0.8f, -0.8f, 0.0f,
   0.8f, -0.8f, 0.0f,
   0.0f,  0.8f, 0.0f
};

SoRenderCommand solidTriangle()
{
  SoRenderCommand command = makeTriangle(kTrianglePositions);
  command.material.diffuse = SbVec4f(1.0f, 1.0f, 1.0f, 1.0f);
  return command;
}

VkRenderPass createExternalRenderPass(const Harness & harness)
{
  const bool hasDepth = harness.haveDepth;

  VkAttachmentDescription attachments[2] {};
  attachments[0].format = harness.target.colorFormat;
  attachments[0].samples = harness.target.sampleCount;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = harness.target.colorLayout;
  attachments[0].finalLayout = harness.target.colorLayout;

  if (hasDepth) {
    attachments[1].format = harness.target.depthFormat;
    attachments[1].samples = harness.target.sampleCount;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = harness.target.depthLayout;
    attachments[1].finalLayout = harness.target.depthLayout;
  }

  VkAttachmentReference colorRef {};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef {};
  depthRef.attachment = 1;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

  VkRenderPassCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  ci.attachmentCount = hasDepth ? 2u : 1u;
  ci.pAttachments = attachments;
  ci.subpassCount = 1;
  ci.pSubpasses = &subpass;

  VkRenderPass pass = VK_NULL_HANDLE;
  vkCreateRenderPass(harness.device, &ci, nullptr, &pass);
  return pass;
}

VkFramebuffer createExternalFramebuffer(const Harness & harness,
                                        VkRenderPass pass)
{
  const bool hasDepth = harness.haveDepth;
  const VkImageView attachments[] = {
    harness.target.colorImageView,
    harness.target.depthImageView,
  };
  VkFramebufferCreateInfo ci {};
  ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  ci.renderPass = pass;
  ci.attachmentCount = hasDepth ? 2u : 1u;
  ci.pAttachments = attachments;
  ci.width = harness.target.extent.width;
  ci.height = harness.target.extent.height;
  ci.layers = 1;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  vkCreateFramebuffer(harness.device, &ci, nullptr, &framebuffer);
  return framebuffer;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  int failures = 0;

  SoDrawList drawlist;
  drawlist.addCommand(solidTriangle());
  SoRenderParams params = harness.renderParams();

  VkRenderPass pass = createExternalRenderPass(harness);
  if (pass == VK_NULL_HANDLE) {
    std::cerr << "FAIL: could not create external render pass" << std::endl;
    harness.shutdown();
    SoDB::finish();
    return 1;
  }
  VkFramebuffer framebuffer = createExternalFramebuffer(harness, pass);
  if (framebuffer == VK_NULL_HANDLE) {
    std::cerr << "FAIL: could not create external framebuffer" << std::endl;
    vkDestroyRenderPass(harness.device, pass, nullptr);
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  // renderExternal must be rejected without a command buffer or render pass.
  {
    if (harness.backend.renderExternal(drawlist, params, VK_NULL_HANDLE, pass)) {
      std::cerr << "FAIL: renderExternal accepted a null command buffer"
                << std::endl;
      ++failures;
    }
    if (harness.backend.renderExternal(drawlist, params, nullptr, VK_NULL_HANDLE)) {
      std::cerr << "FAIL: renderExternal accepted a null render pass"
                << std::endl;
      ++failures;
    }
  }

  // Record into a caller-owned command buffer + render pass, then submit and
  // verify the triangle was drawn.
  {
    VkCommandBufferAllocateInfo ai {};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = harness.commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(harness.device, &ai, &buffer);

    VkCommandBufferBeginInfo bi {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(buffer, &bi);

    VkRenderPassBeginInfo rpbi {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = pass;
    rpbi.framebuffer = framebuffer;
    rpbi.renderArea.offset = {0, 0};
    rpbi.renderArea.extent = harness.target.extent;
    vkCmdBeginRenderPass(buffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    const bool recorded = harness.backend.renderExternal(drawlist, params,
                                                         buffer, pass);

    vkCmdEndRenderPass(buffer);
    vkEndCommandBuffer(buffer);

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &buffer;
    vkQueueSubmit(harness.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(harness.queue);
    vkFreeCommandBuffers(harness.device, harness.commandPool, 1, &buffer);

    if (!recorded) {
      std::cerr << "FAIL: renderExternal returned false" << std::endl;
      ++failures;
    }

    const std::vector<uint8_t> pixels = harness.readback();
    const int colored = countNear(pixels, 255, 255, 255);
    if (colored == 0) {
      std::cerr << "FAIL: renderExternal produced no fragments" << std::endl;
      ++failures;
    }
  }

  vkDestroyFramebuffer(harness.device, framebuffer, nullptr);
  vkDestroyRenderPass(harness.device, pass, nullptr);

  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
