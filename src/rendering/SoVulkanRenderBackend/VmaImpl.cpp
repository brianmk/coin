// src/rendering/SoVulkanRenderBackend/VmaImpl.cpp
//
// Single translation unit compiling the header-only Vulkan Memory Allocator.
// VMA is not bundled with Coin; a Vulkan build needs it installed on the
// system (COIN_VMA_INCLUDE_DIR in the top-level CMakeLists.txt).  Coin calls
// vk* entry points directly (Vulkan::Vulkan), so VMA uses statically linked
// functions rather than vkGetInstanceProcAddr; both macros are pinned so the
// choice does not depend on VK_NO_PROTOTYPES being defined elsewhere.

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>
