// data/shaders/vulkan/rt/FsrResolveTemporal.glsl
//
// AMD FidelityFX DNSR denoiser - temporal reproject + resolve stage, ported to
// Vulkan GLSL compute from the MIT-licensed AMD/FidelityFX-Denoiser source
// src/rendering/third_party/fidelityfx/dnsr/ (ffx_denoiser_reflections_reproject.h
// and _resolve_temporal.h; see FsrCommon.glsl for the shared helpers/license).
//
// This backend uses the *surface* reprojection model: the history is sampled at
// the motion-vector-reprojected UV and rejected where normal/depth disagree
// with the current frame (the reference's disocclusion factor).  The
// reflection-specific parallax "hit position" reprojection and the 3x3 search
// / 2x2 bilinear fallback are omitted: this pass denoises the path tracer's
// beauty radiance (not a separate mirror signal), so the surface model is the
// correct one.  History is then clipped to the current local color AABB and
// blended by the reference's 1/numSamples accumulation speed.
//
// Inputs come from FsrPrefilter.glsl (prefiltered radiance/variance), the path
// tracer's G-buffers (position/normal/motion) and the previous frame's history.
// Output is the denoised radiance (present binding 5) plus the new history.

#version 460
#extension GL_GOOGLE_include_directive : require

#include "FsrCommon.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer PrefilterRadiance { vec4 prefiltered[]; };
layout(set = 0, binding = 1, std430) readonly buffer PrefilterVariance { vec4 prefilteredVar[]; };
layout(set = 0, binding = 2, std430) readonly buffer PositionBuffer { vec4 positions[]; };
layout(set = 0, binding = 3, std430) readonly buffer NormalBuffer { vec4 normals[]; };
layout(set = 0, binding = 4, std430) readonly buffer MotionBuffer { vec4 motions[]; };
// The history sample count is packed into HistoryRadiance.w (radiance only
// uses .rgb) so the separate per-pixel count buffer is not needed -- two fewer
// full-resolution history buffers (~148 MB at 2758x1681).  The first frame
// reads a zero-filled history (num = 0), so the current frame wins outright.
layout(set = 0, binding = 5, std430) readonly buffer HistoryRadiance { vec4 histRadiance[]; };
layout(set = 0, binding = 6, std430) readonly buffer HistoryVariance { vec4 histVariance[]; };
layout(set = 0, binding = 7, std430) readonly buffer HistoryPosition { vec4 histPosition[]; };
layout(set = 0, binding = 8, std430) readonly buffer HistoryNormal { vec4 histNormal[]; };
layout(set = 0, binding = 9, std430) buffer DenoisedBuffer { vec4 denoised[]; };
layout(set = 0, binding = 10, std430) buffer NewHistoryRadiance { vec4 newRadiance[]; };
layout(set = 0, binding = 11, std430) buffer NewHistoryVariance { vec4 newVariance[]; };

layout(push_constant) uniform FsrTemporalPush {
    vec4  cameraPos;    // xyz = world-space camera origin
    uvec2 screen;       // full path-tracing resolution
    float maxSamples;   // accumulation cap
    float pad;
} pc;

// Nearest-texel read of a full-resolution storage buffer at UV, clamped to the
// image.  The reference filter uses bilinear history sampling; nearest is used
// here because the G-buffers/history are storage buffers, not sampled images.
ivec2 fsrUvToTexel(vec2 uv, uvec2 screen)
{
    ivec2 t = ivec2(floor(uv * vec2(screen)));
    return clamp(t, ivec2(0), ivec2(screen) - 1);
}

void main()
{
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    uvec2 screen = pc.screen;
    if (px.x >= int(screen.x) || px.y >= int(screen.y)) {
        return;
    }
    uint idx = uint(px.y) * screen.x + uint(px.x);
    vec2 uv = (vec2(px) + 0.5) / vec2(screen);

    vec3  curRadiance = prefiltered[idx].rgb;
    float curVariance = prefilteredVar[idx].x;
    vec4  pos4 = positions[idx];
    vec3  curPos = pos4.rgb;
    vec3  curNormal = normals[idx].xyz;
    bool  valid = pos4.a > 0.0 && dot(curNormal, curNormal) > 0.0;

    // --- Reproject the history (surface model + disocclusion rejection) -----
    vec3  historyRadiance = vec3(0.0);
    float historyVariance = 1.0;
    float historyNum = 0.0;

    vec4 motion = motions[idx];
    vec2 historyUv = uv - motion.xy;
    bool inBounds = motion.z > 0.5 && valid
        && all(greaterThan(historyUv, vec2(0.0)))
        && all(lessThan(historyUv, vec2(1.0)));

    if (inBounds) {
        ivec2 ht = fsrUvToTexel(historyUv, screen);
        uint hidx = uint(ht.y) * screen.x + uint(ht.x);
        vec3  hPos = histPosition[hidx].rgb;
        vec3  hNormal = histNormal[hidx].xyz;
        vec3  hRadiance = histRadiance[hidx].rgb;
        float hVariance = histVariance[hidx].x;
        float hNum = histRadiance[hidx].w;

        float depth = distance(pc.cameraPos.xyz, curPos);
        float hDepth = distance(pc.cameraPos.xyz, hPos);
        float disocclusion = 1.0
            * exp(-abs(1.0 - max(0.0, dot(normalize(curNormal),
                                           normalize(hNormal))))
                  * FSR_DISOCCLUSION_NORMAL_WEIGHT)
            * exp(-abs(hDepth - depth) / max(depth, 1.0e-3)
                  * FSR_DISOCCLUSION_DEPTH_WEIGHT);

        if (disocclusion >= FSR_DISOCCLUSION_THRESHOLD) {
            historyRadiance = hRadiance;
            historyVariance = hVariance;
            historyNum = hNum;
        }
    }

    // --- Local color AABB from the prefiltered radiance (3x3 neighborhood) ---
    vec3  sum = vec3(0.0);
    vec3  sumSq = vec3(0.0);
    float wsum = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            ivec2 q = clamp(px + ivec2(i, j), ivec2(0), ivec2(screen) - 1);
            vec3 r = prefiltered[uint(q.y) * screen.x + uint(q.x)].rgb;
            sum += r;
            sumSq += r * r;
            wsum += 1.0;
        }
    }
    vec3 mean = sum / wsum;
    vec3 variance = max(sumSq / wsum - mean * mean, vec3(0.0));
    vec3 colorStd = sqrt(variance) + length(mean - curRadiance);
    vec3 clippedHistory = fsrClipAABB(mean - colorStd, mean + colorStd,
                                      historyRadiance);

    // --- Temporal accumulation ---------------------------------------------
    float accumSpeed = 1.0 / max(historyNum, 1.0);
    float weight = 1.0 - accumSpeed;
    float outNum = min(pc.maxSamples, historyNum + 1.0);
    vec3  outRadiance = mix(curRadiance, clippedHistory, weight);
    float outVariance = mix(fsrComputeTemporalVariance(curRadiance,
                                                       clippedHistory),
                            curVariance, weight);
    if (any(isnan(outRadiance)) || any(isinf(outRadiance)) ||
        any(isnan(vec3(outVariance))) || any(isinf(vec3(outVariance)))) {
        outRadiance = curRadiance;
        outVariance = curVariance;
        outNum = 1.0;
    }

    denoised[idx] = vec4(outRadiance, valid ? 1.0 : 0.0);
    newRadiance[idx] = vec4(outRadiance, outNum);
    newVariance[idx] = vec4(outVariance, 0.0, 0.0, 0.0);
}
