// data/shaders/vulkan/rt/PresentFragment.glsl
// Present pass for the ray-tracing backend.
//
// Two modes, selected by the push constant:
//
//  - Preview (u_present.z < 0.5): samples the ray-traced storage image
//    (set 0, binding 1) as in the v1 backend.
//
//  - Path tracing (u_present.z >= 0.5): reads the per-pixel accumulation
//    buffer (set 0, binding 2: rgb = radiance sum, a = sample count) and
//    applies an edge-stopping denoise filter guided by the first-bounce
//    world-normal and hit-distance G-buffers (bindings 3 and 4).  Neighbor
//    weights favor similar normals and similar relative depth so the blur
//    does not bleed across silhouette and crease edges, and are scaled by
//    each neighbor's sample count so early frames still average correctly.

#version 450

layout(set = 0, binding = 1) uniform sampler2D u_rtImage;

layout(set = 0, binding = 2, std430) readonly buffer AccumBuffer { vec4 accum[]; };
layout(set = 0, binding = 3, std430) readonly buffer NormalBuffer { vec4 normals[]; };
layout(set = 0, binding = 4, std430) readonly buffer PositionBuffer { vec4 positions[]; };
layout(set = 0, binding = 5, std430) readonly buffer DenoisedBuffer { vec4 denoised[]; };

layout(push_constant) uniform PresentPush {
    vec4 u_present;  // x = width, y = height, z = denoiseOn, w = frameIndex
    vec4 u_origin;   // x = viewport origin x, y = viewport origin y (pixels)
    vec4 u_denoise;  // x = OIDN result available (sample denoised buffer)
} pc;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec2 viewportCoord = gl_FragCoord.xy - pc.u_origin.xy;
    if (pc.u_present.z < 0.5) {
        fragColor =
          texture(u_rtImage, viewportCoord / textureSize(u_rtImage, 0));
        return;
    }

    ivec2 px = ivec2(viewportCoord);
    const int width = int(max(pc.u_present.x, 1.0));
    const int height = int(max(pc.u_present.y, 1.0));
    const int idx = px.y * width + px.x;

    // OIDN host-side denoise result: when ready it replaces the raw
    // accumulation entirely (the filter already smoothed the noise).  A
    // result scale > 1 is the low-resolution motion preview: bilinearly
    // upsample it from the leading entries of the denoised buffer.
    if (pc.u_denoise.x > 0.5) {
        float scale = max(pc.u_denoise.y, 1.0);
        if (scale < 1.5) {
            vec4 d = denoised[idx];
            if (d.a > 0.5) {
                fragColor = vec4(clamp(d.rgb, 0.0, 1.0), 1.0);
                return;
            }
        }
        else {
            int lw = max((width + int(scale) - 1) / int(scale), 1);
            int lh = max((height + int(scale) - 1) / int(scale), 1);
            vec2 q = (vec2(px) + vec2(0.5)) / scale - vec2(0.5);
            q = clamp(q, vec2(0.0), vec2(lw - 1, lh - 1));
            ivec2 q0 = ivec2(q);
            vec2 f = q - vec2(q0);
            ivec2 q1 = min(q0 + ivec2(1, 1), ivec2(lw - 1, lh - 1));
            vec4 d00 = denoised[q0.y * lw + q0.x];
            vec4 d10 = denoised[q0.y * lw + q1.x];
            vec4 d01 = denoised[q1.y * lw + q0.x];
            vec4 d11 = denoised[q1.y * lw + q1.x];
            vec4 d = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
            if (d.a > 0.5) {
                fragColor = vec4(clamp(d.rgb, 0.0, 1.0), 1.0);
                return;
            }
        }
    }

    vec4 c0 = accum[idx];
    if (c0.a <= 0.0) {
        fragColor = vec4(0.0);
        return;
    }
    vec3 col0 = c0.rgb / c0.a;
    vec3 n0 = normals[idx].xyz;
    float d0 = positions[idx].w;

    vec3 sum = col0 * c0.a;
    float wsum = c0.a;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            ivec2 p = px + ivec2(dx, dy);
            if (p.x < 0 || p.y < 0 || p.x >= width || p.y >= height) continue;
            int i = p.y * width + p.x;
            vec4 c = accum[i];
            if (c.a <= 0.0) continue;
            // Normal weight: strongly reject large orientation changes.
            float wN = pow(clamp(dot(normals[i].xyz, n0), 0.0, 1.0), 32.0);
            // Depth weight: relative distance, scale-invariant.
            float dd = abs(positions[i].w - d0);
            float wP = exp(-dd / max(d0 * 0.02, 1.0e-3));
            float w = wN * wP * c.a;
            sum += (c.rgb / c.a) * w;
            wsum += w;
        }
    }
    fragColor = vec4(clamp(sum / max(wsum, 1.0e-6), 0.0, 1.0), 1.0);
}
