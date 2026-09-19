// src/rendering/third_party/vma/vk_mem_alloc_impl.cpp
//
// Compile the Vulkan Memory Allocator exactly once.  VMA is header-only by
// default; this translation unit provides the single implementation.
//
// Coin links the Vulkan loader directly (find_package(Vulkan) + Vulkan::Vulkan)
// and calls the vk* entry points directly, so VMA uses the statically linked
// functions rather than resolving them through vkGetInstanceProcAddr at
// runtime.  Both macros are pinned here so the choice does not depend on
// whether VK_NO_PROTOTYPES happens to be defined elsewhere in the build.

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include "vk_mem_alloc.h"
