// include/Inventor/rendering/vulkan/SoVulkanViewSettings.h

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
#include <Inventor/rendering/vulkan/SoVulkanViewMode.h>

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
  float pathTracingGlassAbsorption = 0.1f;

  //! Viewport background (solid or gradient).
  SbColor4f backgroundColor {0.0f, 0.0f, 0.0f, 1.0f};
  bool backgroundGradient = false;
  SbColor4f backgroundTop {0.0f, 0.0f, 0.0f, 1.0f};
  SbColor4f backgroundBottom {0.0f, 0.0f, 0.0f, 1.0f};

  //! Raster overlays.  wireframeOverlay and tessellationOverlay are debug
  //! views (the tessellation drawn in polygon-LINES) that FreeCAD does not
  //! expose as preferences: they are set programmatically or through the
  //! renderer's FC_VULKAN_WIREFRAME / FC_VULKAN_TESSELLATION environment hooks
  //! (SoVulkanConfig), which the backend ORs with these flags.  Only
  //! pointsOverlay and edgeOverlay are driven by the FreeCAD view preferences.
  bool wireframeOverlay = false;
  bool pointsOverlay = false;
  //! Show the model's feature-edge lines (BRep edges / polylines).  When
  //! false the non-triangle line residue is skipped in the raster main pass
  //! and in the ray-tracing composite, leaving only the shaded faces.  The
  //! navigation cube and other SO_RENDERPASS_OVERLAY geometry are unaffected.
  bool edgeOverlay = true;
  //! Debug overlay: re-draw the triangle commands in polygon-LINES so the
  //! raw tessellation is visible on top of the shaded geometry.
  bool tessellationOverlay = false;
  SbColor4f edgeColor {0.05f, 0.05f, 0.05f, 1.0f};

  //! HDR output for the Vulkan viewport, presented as scRGB: an FP16
  //! extended-linear sRGB swapchain (VK_FORMAT_R16G16B16A16_SFLOAT) that the
  //! compositor color-manages through the wp_color_manager_v1 protocol.  This
  //! is the representation Blender and gamescope use for Wayland HDR.  Diffuse
  //! white is linear 1.0 (the compositor's reference white) and values above
  //! 1.0 carry HDR highlights.  Set only when the application both requested
  //! HDR and the swapchain actually came up with the FP16 extended-linear
  //! format (the application owns that decision; the backend just encodes).
  //! scRGB is deliberately NOT PQ: the surface is tagged extended-linear sRGB,
  //! so no BT.2020/PQ encode is ever applied here.
  bool hdrOutput = false;
  //! Diffuse-white gain applied to the linear scene radiance before the scRGB
  //! write.  1.0 presents linear white 1.0 at the compositor's reference white
  //! (the default); > 1 brightens the whole viewport, < 1 darkens it.  The
  //! default matches the raster/RTX backends and the FreeCAD preference default
  //! so a default-constructed settings blob renders at the reference white
  //! instead of the retired HDR10/PQ white-gain convention (0.02).  Ignored
  //! when hdrOutput is false.
  float hdrExposure = 1.0f;
  //! Highlight compression for the ray-traced path, applied between the
  //! exposure and the scRGB write (ignored when hdrOutput is false): 0 = clip
  //! (no tone map, highlights pass through above 1.0; the default), 1 =
  //! Reinhard, 2 = ACES, 3 = Hable.  The filmic operators compress the result
  //! into [0,1].  The raster path is display-referred and never tone-maps.
  //! See tonemap() in data/shaders/vulkan/rt/PresentFragment.glsl.
  int hdrToneMap = 0;

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
      && edgeOverlay == other.edgeOverlay
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
