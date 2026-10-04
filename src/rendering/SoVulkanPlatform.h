// src/rendering/SoVulkanPlatform.h
//
// Windows macro hygiene for the Vulkan backend.  On Windows <vulkan/vulkan.h>
// includes <windows.h> (when VK_USE_PLATFORM_WIN32_KHR is set), and that header
// defines the min()/max() function-like macros which break std::min/std::max.
// Suppress them only for the translation units that pull in Vulkan, instead of
// defining NOMINMAX globally for the whole build.  Include before
// <vulkan/vulkan.h>.

#ifndef COIN_SOVULKANPLATFORM_H
#define COIN_SOVULKANPLATFORM_H

#if defined(_WIN32) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

#endif // COIN_SOVULKANPLATFORM_H
