// data/shaders/vulkan/rt/DenoiseDownsample.glsl
// Compute pass that fuses the host denoiser's input preparation into the GPU:
// reads the full-resolution G-buffers (accum/albedo/normal/motion) written by
// the path tracer, normalizes the accumradiance sum -> per-sample average in
// the color channel, and writes the (optionally downsampled) working set that
// OIDN/FSR read from the mapped host staging block.
//
// Historically the host worker (SoRTXRenderBackendDenoise.cpp) normalized the
// color sum and downsample the full-res G-buffers on the CPU, a serial
// pass over 2M fragments that dominated the denoiser latency in a debug/-O0
// build.  This shader does that work in parallel on the GPU and writes only
// the small low-res working set (or the full-res set when scale == 1), leaving
// the worker to run OIDN/FSR alone.
//
// One invocation per denoiser-working pixel.  The source G-buffer texel is
// picked with the same nearest-neighbour subsample the CPU used
// (gx = min(fullW-1, x*sx)), so the working set is bit-identical to the old
// host path -- only the location (GPU) and order (parallel) change.

#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Full-resolution G-buffers (device-local, written by the path tracer).
layout(set = 0, binding = 0, std430) readonly buffer AccumBuf { vec4 accum[]; };
layout(set = 0, binding = 1, std430) readonly buffer AlbedoBuf { vec4 albedos[]; };
layout(set = 0, binding = 2, std430) readonly buffer NormalBuf { vec4 normals[]; };
layout(set = 0, binding = 3, std430) readonly buffer MotionBuf { vec4 motions[]; };

// The single host-visible staging allocation.  outData covers the whole
// buffer; the four working-set regions are addressed by push-constant vec4
// element offsets (byteOffset / 16), exactly the regions the host worker's
// cIn/aIn/nIn/mIn point at.
layout(set = 0, binding = 4, std430) buffer DownOut { vec4 outData[]; };

// Push constants, laid out as raw 32-bit words matching the C++
// DenoiseDownsamplePush struct:
//    0: fullW,  4: fullH,   8: sx,     12: sy
//   16: colorElem, 20: albedoElem, 24: normalElem, 28: motionElem
//   32: outPixels (outW*outH), 36: unused
layout(push_constant) uniform Push {
    uvec4 pcFull;   // x = fullW, y = fullH, z = sx, w = sy
    uvec4 pcReg;    // x = colorElem, y = albedoElem, z = normalElem, w = motionElem
    uvec4 pcNum;    // x = outPixels, yzw unused
} pc;

void main()
{
    uvec2 o = gl_GlobalInvocationID.xy;
    const uint fullW = max(pc.pcFull.x, 1u);
    const uint fullH = max(pc.pcFull.y, 1u);
    const uint sx = max(pc.pcFull.z, 1u);
    const uint sy = max(pc.pcFull.w, 1u);
    const uint outW = (fullW + sx - 1u) / sx;
    const uint outH = (fullH + sy - 1u) / sy;
    if (o.x >= outW || o.y >= outH) { return; }

    // Nearest subsample, matching the host loop (gx = min(gbW-1, x*sx)).
    const uint gx = min(fullW - 1u, o.x * sx);
    const uint gy = min(fullH - 1u, o.y * sy);
    const uint gi = gy * fullW + gx;
    const uint oi = o.y * outW + o.x;

    const vec4 c = accum[gi];
    vec4 cn;
    if (c.a > 1.0e-5) {
        const float inv = 1.0 / c.a;
        cn = vec4(c.rgb * inv, 1.0);
    }
    else {
        cn = vec4(c.rgb, 0.0);
    }

    // Guides are copied verbatim (they are first-bounce per-pixel values, not
    // sample sums, so no normalization applies).
    const vec4 alb = albedos[gi];
    const vec4 nrm = normals[gi];
    const vec4 mot = motions[gi];

    outData[pc.pcReg.x + oi] = cn;
    outData[pc.pcReg.y + oi] = vec4(alb.rgb, alb.a);
    outData[pc.pcReg.z + oi] = vec4(nrm.rgb, nrm.a);
    outData[pc.pcReg.w + oi] = vec4(mot.rgb, mot.a);
}
