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

  //! Viewport background (solid or gradient).
  SbColor4f backgroundColor {0.0f, 0.0f, 0.0f, 1.0f};
  bool backgroundGradient = false;
  SbColor4f backgroundTop {0.0f, 0.0f, 0.0f, 1.0f};
  SbColor4f backgroundBottom {0.0f, 0.0f, 0.0f, 1.0f};

  //! Raster overlays.
  bool wireframeOverlay = false;
  bool pointsOverlay = false;
  SbColor4f edgeColor {0.05f, 0.05f, 0.05f, 1.0f};

  bool operator==(const SoVulkanViewSettings & other) const
  {
    return viewMode == other.viewMode && envMap == other.envMap
      && pathTracingDenoise == other.pathTracingDenoise
      && pathTracingBounces == other.pathTracingBounces
      && pathTracingSettleFrames == other.pathTracingSettleFrames
      && pathTracingMaxSamples == other.pathTracingMaxSamples
      && pathTracingDenoiser == other.pathTracingDenoiser
      && pathTracingDenoiserScale == other.pathTracingDenoiserScale
      && backgroundColor == other.backgroundColor
      && backgroundGradient == other.backgroundGradient
      && backgroundTop == other.backgroundTop
      && backgroundBottom == other.backgroundBottom
      && wireframeOverlay == other.wireframeOverlay
      && pointsOverlay == other.pointsOverlay
      && edgeColor == other.edgeColor;
  }
  bool operator!=(const SoVulkanViewSettings & other) const
  {
    return !(*this == other);
  }
};

#endif // COIN_SOVULKANVIEWSETTINGS_H
