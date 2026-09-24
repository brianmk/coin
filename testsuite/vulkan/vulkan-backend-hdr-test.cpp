// testsuite/vulkan/vulkan-backend-hdr-test.cpp
//
// Exercises SoVulkanRenderBackend::renderExternalHdr() and the output display
// transform: the exposure pre-scale, the four tone-mapping operators
// (clip / Reinhard / ACES / Hable) and the SMPTE ST 2084 (PQ) HDR10 encode.
//
// The scene is rendered into the backend's RGBA16F intermediate (display-
// referred sRGB, as the visual shaders write it) and then presented into a plain
// 8-bit caller-owned pass, so the whole HDR pipeline is testable headlessly (no
// 10-bit swapchain needed).  The expected 8-bit output is computed with the
// reference math mirrored from data/shaders/vulkan/output/OutputFragment.glsl
// (sRGB decode + exposure + tone map + PQ) and compared within a small tolerance
// (float precision + UNORM rounding).

#include "VulkanTestHarness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace vulkan_test;

namespace {

// --- reference display transform (mirrors OutputFragment.glsl) -------------

float linear_to_pq(float L)
{
  const float m1 = 2610.0f / 16384.0f;
  const float m2 = 2523.0f / 4096.0f * 128.0f;
  const float c1 = 3424.0f / 4096.0f;
  const float c2 = 2413.0f / 4096.0f * 32.0f;
  const float c3 = 2392.0f / 4096.0f * 32.0f;
  const float Lp = std::pow(std::max(L, 0.0f), m1);
  return std::pow((c1 + c2 * Lp) / (1.0f + c3 * Lp), m2);
}

float clamp01(float v) { return std::min(std::max(v, 0.0f), 1.0f); }

// sRGB inverse EOTF (IEC 61966-2-1), mirroring srgb_to_linear() in
// OutputFragment.glsl.  The scene intermediate is display-referred sRGB, so the
// HDR transform decodes it to linear light before the exposure/PQ.
float srgb_to_linear(float c)
{
  const float v = clamp01(c);
  return v <= 0.04045f ? v / 12.92f
                       : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

float tonemap(float L, int mode)
{
  switch (mode) {
    case 1:  // Reinhard
      return L / (1.0f + L);
    case 2: {  // ACES (Narkowicz filmic fit)
      const float a = 2.51f;
      const float b = 0.03f;
      const float c = 2.43f;
      const float d = 0.59f;
      const float e = 0.14f;
      return clamp01((L * (a * L + b)) / (L * (c * L + d) + e));
    }
    case 3: {  // Hable (Uncharted 2), normalized at the 11.2 white point
      const float A = 0.15f;
      const float B = 0.50f;
      const float C = 0.10f;
      const float D = 0.20f;
      const float E = 0.02f;
      const float F = 0.30f;
      const float W = 11.2f;
      const float v = L * 2.0f;
      const float num = v * (A * v + C * B) + D * E;
      const float den = v * (A * v + B) + D * F;
      const float wnum = W * (A * W + C * B) + D * E;
      const float wden = W * (A * W + B) + D * F;
      return clamp01((num / den - E / F) / (wnum / wden - E / F));
    }
    default:  // clip
      return clamp01(L);
  }
}

// Expected 8-bit output of the HDR display transform for display-referred sRGB
// scene value c (decoded to linear light first, like the shader).
int expectedHdrByte(float c, float exposure, int toneMap)
{
  const float mapped = tonemap(srgb_to_linear(c) * exposure, toneMap);
  return static_cast<int>(std::lround(linear_to_pq(mapped) * 255.0f));
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

  cases.add("HDR clip matches the PQ reference", [&] {
    const Rgb p = runHdrFrame(harness, grayList, params, pass, framebuffer,
                              /*hdr*/ true, 1.0f, 0, recorded);
    const int expected = expectedHdrByte(scene, 1.0f, 0);
    VK_CHECK(within(p.r, expected, kTolerance),
             "clip PQ byte " << p.r << " != expected " << expected);
  });

  cases.add("HDR exposure pre-scales radiance", [&] {
    const Rgb one = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ true, 1.0f, 0, recorded);
    const Rgb two = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ true, 2.0f, 0, recorded);
    const int expected = expectedHdrByte(scene, 2.0f, 0);
    VK_CHECK(two.r > one.r, "2x exposure did not brighten: "
                              << one.r << " -> " << two.r);
    VK_CHECK(within(two.r, expected, kTolerance),
             "2x exposure PQ byte " << two.r << " != expected " << expected);
  });

  cases.add("HDR Reinhard darkens midtones vs clip", [&] {
    const Rgb clip = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                 /*hdr*/ true, 1.0f, 0, recorded);
    const Rgb rein = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                 /*hdr*/ true, 1.0f, 1, recorded);
    const int expected = expectedHdrByte(scene, 1.0f, 1);
    VK_CHECK(rein.r < clip.r, "Reinhard did not darken: clip="
                                << clip.r << " reinhard=" << rein.r);
    VK_CHECK(within(rein.r, expected, kTolerance),
             "Reinhard PQ byte " << rein.r << " != expected " << expected);
  });

  cases.add("HDR ACES lifts midtones vs clip", [&] {
    const Rgb clip = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                 /*hdr*/ true, 1.0f, 0, recorded);
    const Rgb aces = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                 /*hdr*/ true, 1.0f, 2, recorded);
    const int expected = expectedHdrByte(scene, 1.0f, 2);
    VK_CHECK(aces.r > clip.r, "ACES did not lift midtones: clip="
                                << clip.r << " aces=" << aces.r);
    VK_CHECK(within(aces.r, expected, kTolerance),
             "ACES PQ byte " << aces.r << " != expected " << expected);
  });

  cases.add("HDR Hable shoulder matches the reference", [&] {
    const Rgb hable = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                  /*hdr*/ true, 1.0f, 3, recorded);
    const int expected = expectedHdrByte(scene, 1.0f, 3);
    VK_CHECK(within(hable.r, expected, kTolerance),
             "Hable PQ byte " << hable.r << " != expected " << expected);
  });

  cases.add("HDR tone-map operators are distinct", [&] {
    int bytes[4];
    for (int mode = 0; mode < 4; ++mode) {
      const Rgb p = runHdrFrame(harness, grayList, params, pass, framebuffer,
                                /*hdr*/ true, 1.0f, mode, recorded);
      bytes[mode] = p.r;
    }
    const bool allEqual =
      bytes[0] == bytes[1] && bytes[1] == bytes[2] && bytes[2] == bytes[3];
    VK_CHECK(!allEqual, "all four tone maps produced " << bytes[0]);
    VK_CHECK(bytes[0] != bytes[1] && bytes[1] != bytes[3],
             "clip/Reinhard/Hable collapsed: " << bytes[0] << "," << bytes[1]
                                               << "," << bytes[2] << ","
                                               << bytes[3]);
  });

  const int failures = cases.run();

  vkDestroyFramebuffer(harness.device, framebuffer, nullptr);
  vkDestroyRenderPass(harness.device, pass, nullptr);
  harness.shutdown();
  SoDB::finish();
  return failures == 0 ? 0 : 1;
}
