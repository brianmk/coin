// include/Inventor/rendering/SoVulkanViewMode.h

#ifndef COIN_SOVULKANVIEWMODE_H
#define COIN_SOVULKANVIEWMODE_H

/*!
  \file SoVulkanViewMode.h
  \brief Public ray-traced view-mode selector shared by the renderer and its
  embedding application.

  The ray-traced view mode used to exist as three different encodings: an
  application-side enum, a raw int passed down the widget stack, and this
  renderer-side enum.  They are now one type, so the mode crosses the
  application/renderer boundary with the compiler checking it instead of
  relying on matching magic ints.

  The values are stable and match the historical int encoding:
    0 = RtxModeOff          raster (no ray tracing)
    1 = RtxModeAmbientOcclusion  single-sample AO preview, no accumulation
    2 = RtxModePathTrace    accumulating multi-bounce path tracer
    3 = RtxModeEnvironment  single-sample procedural IBL preview

  This header is always available (it has no Vulkan dependency) so an
  application can name the type even in a build without the Vulkan renderer.
*/
enum class SoVulkanViewMode : int {
  RtxModeOff = 0,
  RtxModeAmbientOcclusion = 1,
  RtxModePathTrace = 2,
  RtxModeEnvironment = 3,
};

#endif // COIN_SOVULKANVIEWMODE_H
