// data/shaders/vulkan/rt/FsrPrefilter.glsl
//
// AMD FidelityFX DNSR denoiser - spatial prefilter stage, ported to Vulkan
// GLSL compute from the MIT-licensed AMD/FidelityFX-Denoiser source
// src/rendering/third_party/fidelityfx/dnsr/ffx_denoiser_reflections_prefilter.h
// (see FsrCommon.glsl for the shared helpers and the license note).
//
// The reference filter denoises a *reflection* radiance signal using a 15-tap
// edge-stopping spatial filter (normal + depth + radiance weights) over a
// 16x16 shared-memory tile.  This backend reuses that exact filter math as the
// path tracer's "fsr" denoiser: the accumulated (temporally converged) first-
// bounce radiance is the filtered signal, and the albedo/normal/motion G-buffers
// the reference denoiser would consume are supplied by the existing
// NormalBuffer/PositionBuffer G-buffers.  The reference filter's per-8x8
// average-radiance mip is approximated by the center sample (documented at the
// call site); the temporal reproject/resolve stages are not part of this pass.
//
// One 8x8 workgroup filters an 8x8 output tile and reads a 12x12 neighborhood
// through a 16x16 shared tile, so the group boundary is seamless.

#version 460
#extension GL_GOOGLE_include_directive : require

#include "FsrCommon.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Path-tracer G-buffers (full path-tracing resolution).
//   accum : rgb = radiance sum, a = sample count
//   sq    : rgb = sum(radiance^2), a = sample count (adaptive-sampling moments)
//   normal: rgb = first-bounce world normal
//   position: rgb = first-bounce world position, a = hit distance
layout(set = 0, binding = 0, std430) readonly buffer AccumBuffer { vec4 accum[]; };
layout(set = 0, binding = 1, std430) readonly buffer SumSqBuffer { vec4 sq[]; };
layout(set = 0, binding = 2, std430) readonly buffer NormalBuffer { vec4 normals[]; };
layout(set = 0, binding = 3, std430) readonly buffer PositionBuffer { vec4 positions[]; };

// Denoised output of this stage: rgb = prefiltered radiance, a = validity,
// consumed by the temporal resolve pass.  The prefiltered variance is a
// separate scalar buffer.
layout(set = 0, binding = 4, std430) buffer PrefilterRadiance { vec4 prefiltered[]; };
layout(set = 0, binding = 5, std430) buffer PrefilterVariance { vec4 prefilteredVar[]; };

layout(push_constant) uniform FsrPush {
    uvec2 screen;   // full path-tracing resolution
    uvec2 pad;
} pc;

shared uint  gShared0[16][16];
shared uint  gShared1[16][16];
shared uint  gShared2[16][16];
shared uint  gShared3[16][16];
shared float gSharedDepth[16][16];

struct FsrNeighborhoodSample {
    vec3  radiance;
    float variance;
    vec3  normal;
    float depth;
};

float fsrLuminanceUnclamped(vec3 color)
{
    return max(dot(color, vec3(0.299, 0.587, 0.114)), 0.0);
}

// Reads one G-buffer texel into the filter's neighborhood sample.  The sample
// count in accum.w gates validity: an empty pixel reports zero variance so the
// caller takes the copy path, and zero normal so it contributes no edge weight
// to its neighbours.
void fsrLoadNeighborhood(ivec2 pixel, out vec3 radiance, out float variance,
                         out vec3 normal, out float depth, uvec2 screen)
{
    ivec2 q = clamp(pixel, ivec2(0), ivec2(screen) - 1);
    uint  idx = uint(q.y) * screen.x + uint(q.x);

    vec4  a = accum[idx];
    float n = a.a;
    vec3  mean = n > 0.0 ? a.rgb / n : vec3(0.0);
    vec3  meanSq = n > 0.0 ? sq[idx].rgb / n : vec3(0.0);
    float meanL = fsrLuminanceUnclamped(mean);
    float meanSqL = fsrLuminanceUnclamped(meanSq);

    radiance = mean;
    variance = max(0.0, meanSqL - meanL * meanL);
    normal = n > 0.0 ? normals[idx].xyz : vec3(0.0);
    depth = positions[idx].a;
}

FsrNeighborhoodSample fsrLoadFromGroupSharedMemory(ivec2 idx)
{
    uvec2 packedRadiance = uvec2(gShared0[idx.y][idx.x], gShared1[idx.y][idx.x]);
    vec4  unpackedRadiance = fsrUnpackFloat16x4(packedRadiance);
    uvec2 packedNormalVariance = uvec2(gShared2[idx.y][idx.x], gShared3[idx.y][idx.x]);
    vec4  unpackedNormalVariance = fsrUnpackFloat16x4(packedNormalVariance);

    FsrNeighborhoodSample result;
    result.radiance = unpackedRadiance.xyz;
    result.normal = unpackedNormalVariance.xyz;
    result.variance = unpackedNormalVariance.w;
    result.depth = gSharedDepth[idx.y][idx.x];
    return result;
}

void fsrStoreInGroupSharedMemory(ivec2 groupThreadId, vec3 radiance, float variance,
                                 vec3 normal, float depth)
{
    gShared0[groupThreadId.y][groupThreadId.x] = fsrPackFloat16(radiance.xy);
    gShared1[groupThreadId.y][groupThreadId.x] = fsrPackFloat16(radiance.zz);
    gShared2[groupThreadId.y][groupThreadId.x] = fsrPackFloat16(normal.xy);
    gShared3[groupThreadId.y][groupThreadId.x] = fsrPackFloat16(vec2(normal.z, variance));
    gSharedDepth[groupThreadId.y][groupThreadId.x] = depth;
}

void fsrInitializeGroupSharedMemory(ivec2 dispatchThreadId, ivec2 groupThreadId,
                                    uvec2 screen)
{
    // Load a 16x16 region into shared memory using four 8x8 blocks.
    ivec2 offset[4] = ivec2[4](ivec2(0, 0), ivec2(8, 0), ivec2(0, 8), ivec2(8, 8));

    vec3  radiance[4];
    float variance[4];
    vec3  normal[4];
    float depth[4];

    // Start in the upper left corner of the 16x16 region.
    dispatchThreadId -= 4;

    for (int i = 0; i < 4; ++i) {
        fsrLoadNeighborhood(dispatchThreadId + offset[i], radiance[i], variance[i],
                            normal[i], depth[i], screen);
    }
    for (int j = 0; j < 4; ++j) {
        fsrStoreInGroupSharedMemory(groupThreadId + offset[j], radiance[j],
                                    variance[j], normal[j], depth[j]);
    }
}

float fsrGetEdgeStoppingNormalWeight(vec3 normalP, vec3 normalQ)
{
    return pow(max(dot(normalP, normalQ), 0.0), FSR_PREFILTER_NORMAL_SIGMA);
}

float fsrGetEdgeStoppingDepthWeight(float centerDepth, float neighborDepth)
{
    return exp(-abs(centerDepth - neighborDepth) * centerDepth * FSR_PREFILTER_DEPTH_SIGMA);
}

float fsrGetRadianceWeight(vec3 centerRadiance, vec3 neighborRadiance, float variance)
{
    return max(exp(-(FSR_RADIANCE_WEIGHT_BIAS + variance * FSR_RADIANCE_WEIGHT_VARIANCE_K)
                    * length(centerRadiance - neighborRadiance)),
               1.0e-2);
}

void fsrResolve(ivec2 groupThreadId, vec3 avgRadiance, FsrNeighborhoodSample center,
                out vec3 resolvedRadiance, out float resolvedVariance)
{
    // The initial weight removes fireflies at the cost of some energy.
    float accumulatedWeight = fsrGetRadianceWeight(avgRadiance, center.radiance, center.variance);
    vec3  accumulatedRadiance = center.radiance * accumulatedWeight;
    float accumulatedVariance = center.variance * accumulatedWeight * accumulatedWeight;

    // First 15 numbers of Halton(2,3) stretched to [-3,3], skipping the center.
    const int sampleCount = 15;
    ivec2 sampleOffsets[15] = ivec2[15](
        ivec2(0, 1),  ivec2(-2, 1),  ivec2(2, -3), ivec2(-3, 0),  ivec2(1, 2),
        ivec2(-1, -2), ivec2(3, 0), ivec2(-3, 3),  ivec2(0, -3),  ivec2(-1, -1),
        ivec2(2, 1),  ivec2(-2, -2), ivec2(1, 0),  ivec2(0, 2),   ivec2(3, -1));

    float varianceWeight = max(FSR_PREFILTER_VARIANCE_BIAS,
                               1.0 - exp(-(center.variance * FSR_PREFILTER_VARIANCE_WEIGHT)));

    for (int i = 0; i < sampleCount; ++i) {
        ivec2 newIdx = groupThreadId + sampleOffsets[i];
        FsrNeighborhoodSample neighbor = fsrLoadFromGroupSharedMemory(newIdx);

        float weight = 1.0;
        weight *= fsrGetEdgeStoppingNormalWeight(center.normal, neighbor.normal);
        weight *= fsrGetEdgeStoppingDepthWeight(center.depth, neighbor.depth);
        weight *= fsrGetRadianceWeight(avgRadiance, neighbor.radiance, center.variance);
        weight *= varianceWeight;

        accumulatedWeight += weight;
        accumulatedRadiance += weight * neighbor.radiance;
        accumulatedVariance += weight * weight * neighbor.variance;
    }

    accumulatedRadiance /= accumulatedWeight;
    accumulatedVariance /= (accumulatedWeight * accumulatedWeight);
    resolvedRadiance = accumulatedRadiance;
    resolvedVariance = accumulatedVariance;
}

void main()
{
    ivec2 dispatchThreadId = ivec2(gl_GlobalInvocationID.xy);
    ivec2 groupThreadId = ivec2(gl_LocalInvocationID.xy);
    uvec2 screen = pc.screen;

    fsrInitializeGroupSharedMemory(dispatchThreadId, groupThreadId, screen);
    barrier();

    groupThreadId += 4; // Center threads in shared memory

    FsrNeighborhoodSample center = fsrLoadFromGroupSharedMemory(groupThreadId);
    vec3  resolvedRadiance = center.radiance;
    float resolvedVariance = center.variance;

    // The reference filter gates on roughness (glossy but not mirror).  The
    // path tracer has no roughness G-buffer, so the signal is always treated as
    // a denoisable glossy reflection: only an empty pixel (zero variance) skips
    // the filter.
    bool needsDenoiser = center.variance > 0.0;
    if (needsDenoiser) {
        // Approximate the reference filter's per-8x8 average-radiance mip with
        // the center sample.  It only biases the radiance-weight reference; a
        // local center value keeps the filter stable without a mip chain.
        vec3 avgRadiance = center.radiance;
        fsrResolve(groupThreadId, avgRadiance, center, resolvedRadiance, resolvedVariance);
    }

    // Bounds guard: the shared-tile loads clamp, but only real pixels write.
    if (dispatchThreadId.x >= int(screen.x) || dispatchThreadId.y >= int(screen.y)) {
        return;
    }
    uint outIdx = uint(dispatchThreadId.y) * screen.x + uint(dispatchThreadId.x);
    float valid = accum[outIdx].a > 0.0 ? 1.0 : 0.0;
    prefiltered[outIdx] = vec4(resolvedRadiance, valid);
    prefilteredVar[outIdx] = vec4(resolvedVariance, 0.0, 0.0, 0.0);
}
