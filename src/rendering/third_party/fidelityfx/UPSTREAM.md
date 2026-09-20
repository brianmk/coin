# Vendored source: AMD FidelityFX Denoiser (DNSR)

Upstream: https://github.com/AMD/FidelityFX-Denoiser
Commit:   d7dfecbabe7b9523b14e7b067216e06b86e8d189
License:  MIT (see `LICENSE.txt`)

`dnsr/` holds the unmodified upstream shader headers:

- `ffx_denoiser_reflections_{common,config,prefilter,reproject,resolve_temporal}.h`
  - the reflection (ray-regeneration) denoiser.
- `ffx_denoiser_shadows_{util,prepare,tileclassification,filter}.h`
  - the shadow denoiser (vendored for reference; not yet ported).

These are HLSL shader headers: they are not compiled by Coin.  The Coin port
lives in `data/shaders/vulkan/rt/FsrPrefilter.glsl` (plus `FsrCommon.glsl`),
which reimplements the reflection prefilter stage in Vulkan GLSL compute and
feeds it the path tracer's G-buffers.  `COIN_BUILD_FSR_DENOISER` (default OFF)
selects whether that backend is built into `SoRTXRenderBackend`'s "fsr"
denoiser slot.
