// testsuite/vulkan/vulkan-backend-hdr-test.cpp
//
// Exercises SoVulkanRenderBackend::renderExternalHdr() and the output display
// transform: the sRGB-to-linear decode and the exposure (diffuse-white) gain
// that produce the scRGB extended-linear output.
//
// The scene is rendered into the backend's RGBA16F intermediate (display-
// referred sRGB, as the visual shaders write it) and then presented into a plain
// 8-bit caller-owned pass, so the display transform is testable headlessly (no
// FP16 swapchain needed).  The expected 8-bit output is computed with the
// reference math mirrored from data/shaders/vulkan/output/OutputFragment.glsl
// (sRGB decode + exposure gain) and compared within a small tolerance (float
// precision + UNORM rounding).

#include "VulkanTestHarness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace vulkan_test;

namespace {

// --- reference display transform (mirrors OutputFragment.glsl) -------------
//
// The HDR output path presents the scene intermediate as scRGB extended-linear
// sRGB: the display-referred sRGB code value is decoded to linear light and
// scaled by the exposure (diffuse-white gain).  There is deliberately no tone
// mapping or PQ encode in this path (the compositor maps the extended range),
// so the 8-bit caller target simply shows the clamped linear value.

float clamp01(float v) { return std::min(std::max(v, 0.0f), 1.0f); }

// sRGB inverse EOTF (IEC 61966-2-1), mirroring srgb_to_linear() in
// OutputFragment.glsl.  The scene intermediate is display-referred sRGB, so the
// HDR transform decodes it to linear light before the exposure gain.
float srgb_to_linear(float c)
{
  const float v = clamp01(c);
  return v <= 0.04045f ? v / 12.92f
                       : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

// Expected 8-bit output of the scRGB display transform for display-referred
// sRGB scene value c: decode to linear, apply the exposure gain, clamp.
int expectedScrgbByte(float c, float exposure)
{
  const float mapped = clamp01(srgb_to_linear(c) * exposure);
  return static_cast<int>(std::lround(mapped * 255.0f));
}

// --- caller-owned output pass / framebuffer (same shape as the render pass
//     QVulkanWindow hands the backend) ---------------------------------------

VkRenderPass createOutputRenderPass(const Harness & harness)
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

VkFramebuffer createOutputFramebuffer(const Harness & harness,
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

// Fullscreen unlit quad: the scene radiance is the diffuse color, so the
// intermediate holds a known, uniform gray.
SoRenderCommand grayQuad(float value)
{
  return makeLitQuad(SbVec4f(value, value, value, 1.0f), SO_SHADING_UNLIT);
}

struct Rgb
{
  int b = 0;
  int g = 0;
  int r = 0;
};

// Runs one renderExternalHdr frame and returns the centre pixel (BGRA bytes).
Rgb runHdrFrame(Harness & harness,
                const SoDrawList & drawlist,
                const SoRenderParams & params,
                VkRenderPass pass,
                VkFramebuffer framebuffer,
                bool hdr,
                float exposure,
                int toneMap,
                bool & recorded)
{
  harness.backend.setHdrOutput(hdr ? TRUE : FALSE, exposure, toneMap);

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

  recorded = harness.backend.renderExternalHdr(drawlist, params, buffer, pass,
                                               framebuffer) != FALSE;

  vkEndCommandBuffer(buffer);

  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &buffer;
  vkQueueSubmit(harness.queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(harness.queue);
  vkFreeCommandBuffers(harness.device, harness.commandPool, 1, &buffer);

  const std::vector<uint8_t> pixels = harness.readback();
  const uint8_t * center = pixelAt(pixels, kWidth / 2, kHeight / 2);
  return Rgb { center[0], center[1], center[2] };
}

bool within(int got, int expected, int tolerance)
{
  return std::abs(got - expected) <= tolerance;
}

} // namespace

int
main()
{
  Harness harness;
  const int initResult = harness.init();
  if (initResult != 0) return initResult;

  // renderExternalHdr's output pass carries a depth attachment (the begin info
  // supplies two clear values), so the test needs a depth-capable device.
  if (!harness.haveDepth) {
    harness.shutdown();
    SoDB::finish();
    return skip("HDR output pass needs a depth attachment");
  }

  VkRenderPass pass = createOutputRenderPass(harness);
  VkFramebuffer framebuffer =
    pass != VK_NULL_HANDLE ? createOutputFramebuffer(harness, pass)
                           : VK_NULL_HANDLE;
  if (pass == VK_NULL_HANDLE || framebuffer == VK_NULL_HANDLE) {
    std::cerr << "FAIL: could not create the HDR output pass/framebuffer"
              << std::endl;
    if (framebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(harness.device, framebuffer, nullptr);
    }
    if (pass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(harness.device, pass, nullptr);
    }
    harness.shutdown();
    SoDB::finish();
    return 1;
  }

  SoRenderParams params = harness.renderParams();

  SoDrawList grayList;
  grayList.addCommand(grayQuad(0.5f));

  SoDrawList emptyList;

  const int kTolerance = 3;
  bool recorded = false;

  // Measure the scene radiance from the SDR clamp (the display transform is a
  // no-op there), so the HDR expectations do not hard-code the scene color.
  const Rgb sdr1 =
    runHdrFrame(harness, grayList, params, pass, framebuffer,
                /*hdr*/ false, 1.0f, 1, recorded);
  const float scene = static_cast<float>(sdr1.r) / 255.0f;

  CaseRunner cases;

  cases.add("renderExternalHdr rejects a null command buffer", [&] {
    harness.backend.setHdrOutput(TRUE, 1.0f, 1);
    VK_CHECK(
      harness.backend.renderExternalHdr(grayList, params, VK_NULL_HANDLE, pass,
                                        framebuffer) == FALSE,
      "renderExternalHdr accepted a null command buffer");
  });

  cases.add("SDR clamp is exposure- and tone-map-independent", [&] {
    const Rgb a = runHdrFrame(harness, grayList, params, pass, framebuffer,
                              /*hdr*/ false, 1.0f, 1, recorded);
    const Rgb b = runHdrFrame(harness, grayList, params, pass, framebuffer,
                              /*hdr*/ false, 16.0f, 3, recorded);
    VK_CHECK(a.r > 0 && a.g > 0 && a.b > 0, "SDR render produced a black pixel");
    VK_CHECK(a.r == b.r && a.g == b.g && a.b == b.b,
             "SDR clamp changed with exposure/tone map: "
               << a.r << " vs " << b.r);
  });

  cases.add("HDR pure black stays black", [&] {
    const Rgb p = runHdrFrame(harness, emptyList, params, pass, framebuffer,
                              /*hdr*/ true, 1.0f, 1, recorded);
    VK_CHECK(p.r == 0 && p.g == 0 && p.b == 0,
             "HDR black was not 0: " << p.b << "," << p.g << "," << p.r);
  });

  cases.add("HDR decodes the sRGB scene to linear scRGB", [&] {
    const Rgb p = runHdrFrame(harness, grayList, params, pass, framebuffer,
                              /*hdr*/ true, 1.0f, 0, recorded);
    const int expected = expectedScrgbByte(scene, 1.0f);
    VK_CHECK(within(p.r, expected, kTolerance),
             "scRGB byte " << p.r << " != expected " << expected);
  });

  cases.add("HDR exposure pre-scales radiance", [&] {
    const Rgb one = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ true, 1.0f, 0, recorded);
    const Rgb two = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ true, 2.0f, 0, recorded);
    const int expected = expectedScrgbByte(scene, 2.0f);
    VK_CHECK(two.r > one.r, "2x exposure did not brighten: "
                              << one.r << " -> " << two.r);
    VK_CHECK(within(two.r, expected, kTolerance),
             "2x exposure scRGB byte " << two.r << " != expected " << expected);
  });

  cases.add("HDR output differs from the SDR clamp midtone", [&] {
    const Rgb sdr = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ false, 1.0f, 0, recorded);
    const Rgb hdr = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ true, 1.0f, 0, recorded);
    VK_CHECK(sdr.r != hdr.r, "HDR output matched the SDR clamp: "
                               << sdr.r << " == " << hdr.r);
  });

  const int failures = cases.run();

  vkDestroyFramebuffer(harness.device, framebuffer, nullptr);
  vkDestroyRenderPass(harness.device, pass, nullptr);
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
