// include/Inventor/rendering/SoVulkanViewSettings.h

#ifndef COIN_SOVULKANVIEWSETTINGS_H
#define COIN_SOVULKANVIEWSETTINGS_H

/*!
  \file SoVulkanViewSettings.h
  \brief The Vulkan viewport's display/tuning settings as one value type.

  Before this type the same settings were plumbed as a dozen independent
  setters through every layer (application -> widget -> renderer -> render
  manager), each layer keeping its own "last applied" mirror.  Passing one
  struct lets the manager diff once and lets callers hand the whole display
  state over in a single call.

  Structural per-frame state (scene graph, camera, viewport region, render
  target) and the stateful path-tracing enable/start latch are deliberately
  NOT part of this struct; they are passed through their own calls.
*/

#include <Inventor/SbColor4f.h>
#include <Inventor/rendering/SoVulkanViewMode.h>

#include <string>

struct SoVulkanViewSettings {
  //! Ray-traced view mode (see SoVulkanViewMode).
  SoVulkanViewMode viewMode = SoVulkanViewMode::RtxModeOff;
  //! Cubemap environment preset (-1 = viewport background gradient).
  int envMap = -1;

  //! Path-tracing tuning.  The denoiser is required by the path tracer; this
  //! only selects the filter (empty = backend default).
  bool pathTracingDenoise = true;
  int pathTracingBounces = 4;
  int pathTracingSettleFrames = 6;
  int pathTracingMaxSamples = 256;
  std::string pathTracingDenoiser;
  float pathTracingDenoiserScale = 1.0f;
  //! Physically-based glass (RtxModePathTraceMax only): index of refraction
  //! for dielectric surfaces (transparency > 0) and the Beer-Lambert
  //! absorption strength derived from the material colour.
  float pathTracingGlassIor = 1.5f;
  float pathTracingGlassAbsorption = 0.2f;

  //! Viewport background (solid or gradient).
  SbColor4f backgroundColor {0.0f, 0.0f, 0.0f, 1.0f};
  bool backgroundGradient = false;
  SbColor4f backgroundTop {0.0f, 0.0f, 0.0f, 1.0f};
  SbColor4f backgroundBottom {0.0f, 0.0f, 0.0f, 1.0f};

  //! Raster overlays.
  bool wireframeOverlay = false;
  bool pointsOverlay = false;
  //! Debug overlay: re-draw the triangle commands in polygon-LINES so the
  //! raw tessellation is visible on top of the shaded geometry.
  bool tessellationOverlay = false;
  SbColor4f edgeColor {0.05f, 0.05f, 0.05f, 1.0f};

  //! HDR10 output: encode the final swapchain write as SMPTE ST 2084 (PQ) with
  //! BT.2020 primaries.  Set only when the application both requested HDR and
  //! the swapchain actually came up with a 10-bit HDR format (the application
  //! owns that decision; the backend just encodes).  A linear pre-scale of the
  //! scene radiance is applied before the PQ transfer function.
  bool hdrOutput = false;
  //! Linear exposure/gain applied to the scene radiance before the PQ encode.
  //! The default matches the backends' reference-white convention: 0.02 maps
  //! scene-white (radiance 1.0) to ~200 cd/m² rather than to the PQ peak
  //! (10000 cd/m²), so a caller that pushes a default-constructed settings
  //! blob does not silently render ~50x too bright.  Ignored when hdrOutput is
  //! false.
  float hdrExposure = 0.02f;
  //! HDR tone-mapping operator applied between the exposure and the PQ encode
  //! (ignored when hdrOutput is false): 0 = clip (no tone map, the pre-tone-map
  //! behavior), 1 = Reinhard, 2 = ACES, 3 = Hable.  The exposure is a plain
  //! linear pre-scale (the reference convention), so the operators differ in
  //! brightness/contrast as documented by their sources.  See tonemap() in
  //! data/shaders/vulkan/output/OutputFragment.glsl and rt/PresentFragment.glsl.
  int hdrToneMap = 1;

  bool operator==(const SoVulkanViewSettings & other) const
  {
    return viewMode == other.viewMode && envMap == other.envMap
      && pathTracingDenoise == other.pathTracingDenoise
      && pathTracingBounces == other.pathTracingBounces
      && pathTracingSettleFrames == other.pathTracingSettleFrames
      && pathTracingMaxSamples == other.pathTracingMaxSamples
      && pathTracingDenoiser == other.pathTracingDenoiser
      && pathTracingDenoiserScale == other.pathTracingDenoiserScale
      && pathTracingGlassIor == other.pathTracingGlassIor
      && pathTracingGlassAbsorption == other.pathTracingGlassAbsorption
      && backgroundColor == other.backgroundColor
      && backgroundGradient == other.backgroundGradient
      && backgroundTop == other.backgroundTop
      && backgroundBottom == other.backgroundBottom
      && wireframeOverlay == other.wireframeOverlay
      && pointsOverlay == other.pointsOverlay
      && tessellationOverlay == other.tessellationOverlay
      && edgeColor == other.edgeColor
      && hdrOutput == other.hdrOutput
      && hdrExposure == other.hdrExposure
      && hdrToneMap == other.hdrToneMap;
  }
  bool operator!=(const SoVulkanViewSettings & other) const
  {
    return !(*this == other);
  }
};

#endif // COIN_SOVULKANVIEWSETTINGS_H
