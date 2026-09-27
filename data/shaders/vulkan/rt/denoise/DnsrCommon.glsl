// data/shaders/vulkan/rt/DnsrCommon.glsl
//
// Shared helpers for the AMD FidelityFX DNSR denoiser port.  Ported from the
// MIT-licensed AMD/FidelityFX-Denoiser sources vendored at
// src/rendering/third_party/fidelityfx/dnsr/:
//
//   ffx_denoiser_reflections_config.h  -> the DNSR_* tuning constants
//   ffx_denoiser_reflections_common.h  -> the half-precision storage helpers
//
// The upstream shaders are HLSL written with min16float.  This port widens
// min16float to float (the half-precision arithmetic is a GPU throughput
// optimization, not a result change) but keeps the pack/unpack round trip so
// the shared-memory layout and filter weights match the reference filter.

#ifndef COIN_DNSR_COMMON_GLSL
#define COIN_DNSR_COMMON_GLSL

// --- ffx_denoiser_reflections_config.h --------------------------------------
#define DNSR_GAUSSIAN_K 3.0
#define DNSR_RADIANCE_WEIGHT_BIAS 0.6
#define DNSR_RADIANCE_WEIGHT_VARIANCE_K 0.1
#define DNSR_AVG_RADIANCE_LUMINANCE_WEIGHT 0.3
#define DNSR_PREFILTER_VARIANCE_WEIGHT 4.4
#define DNSR_REPROJECT_SURFACE_DISCARD_VARIANCE_WEIGHT 1.5
#define DNSR_PREFILTER_VARIANCE_BIAS 0.1
#define DNSR_PREFILTER_NORMAL_SIGMA 512.0
#define DNSR_PREFILTER_DEPTH_SIGMA 4.0
#define DNSR_DISOCCLUSION_NORMAL_WEIGHT 1.4
#define DNSR_DISOCCLUSION_DEPTH_WEIGHT 1.0
#define DNSR_DISOCCLUSION_THRESHOLD 0.9
#define DNSR_REPROJECTION_NORMAL_SIMILARITY_THRESHOLD 0.9999
#define DNSR_SAMPLES_FOR_ROUGHNESS(r) (1.0 - exp(-(r) * 100.0))

// --- ffx_denoiser_reflections_common.h (half-precision storage helpers) -----
// The shared-memory tiles store radiance/normal/variance packed as fp16 pairs,
// exactly as the reference filter does.
uint dnsrPackFloat16(vec2 v)
{
    return packHalf2x16(v);
}

vec2 dnsrUnpackFloat16(uint a)
{
    return unpackHalf2x16(a);
}

uvec2 dnsrPackFloat16x4(vec4 v)
{
    return uvec2(dnsrPackFloat16(v.xy), dnsrPackFloat16(v.zw));
}

vec4 dnsrUnpackFloat16x4(uvec2 a)
{
    return vec4(dnsrUnpackFloat16(a.x), dnsrUnpackFloat16(a.y));
}

// --- ffx_denoiser_reflections_common.h (shared temporal helpers) ------------
// Luminance used by the temporal variance/weight terms (clamped away from zero
// exactly as the reference does).
float dnsrLuminance(vec3 color)
{
    return max(dot(color, vec3(0.299, 0.587, 0.114)), 0.001);
}

// Relative luminance difference of two samples, squared (the reference's
// per-pixel temporal variance estimate).
float dnsrComputeTemporalVariance(vec3 historyRadiance, vec3 radiance)
{
    float historyLuminance = dnsrLuminance(historyRadiance);
    float luminance = dnsrLuminance(radiance);
    float diff = abs(historyLuminance - luminance)
        / max(max(historyLuminance, luminance), 0.5);
    return diff * diff;
}

// Clip the history sample towards the center of the local color AABB, which is
// what stops stale history from ghosting (from "Temporal Reprojection
// Anti-Aliasing", playdeadgames/temporal, as used by the reference filter).
vec3 dnsrClipAABB(vec3 aabbMin, vec3 aabbMax, vec3 prevSample)
{
    vec3 aabbCenter = 0.5 * (aabbMax + aabbMin);
    vec3 extentClip = 0.5 * (aabbMax - aabbMin) + 0.001;
    vec3 colorVector = prevSample - aabbCenter;
    vec3 colorVectorClip = colorVector / extentClip;
    colorVectorClip = abs(colorVectorClip);
    float maxAbsUnit = max(max(colorVectorClip.x, colorVectorClip.y),
                           colorVectorClip.z);
    if (maxAbsUnit > 1.0) {
        return aabbCenter + colorVector / maxAbsUnit;
    }
    return prevSample;
}

#define DNSR_LOCAL_NEIGHBORHOOD_RADIUS 2
float dnsrLocalNeighborhoodKernelWeight(float i)
{
    const float radius = DNSR_LOCAL_NEIGHBORHOOD_RADIUS + 1.0;
    return exp(-DNSR_GAUSSIAN_K * (i * i) / (radius * radius));
}

#endif // COIN_DNSR_COMMON_GLSL
