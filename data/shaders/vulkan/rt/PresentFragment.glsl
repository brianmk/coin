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

// Stable occlusion depth for the raster edge-overlay composite: the first-
// bounce hit of the path tracer's un-jittered centre sample (world position in
// xyz, ray distance in w; w > 1.0e6 marks a miss).  Separate from the ping-ponged
// position G-buffer (binding 4) so it is constant across the accumulation run.
layout(set = 0, binding = 7, std430) readonly buffer StableDepthBuffer { vec4 stableDepth[]; };

// View -> clip projection (forward) of the traced camera.  The present
// pass writes the scene depth from the first-bounce hit position so the
// raster composite overlay (BRep edge lines / point markers) can be depth
// tested against the traced surface, occluding hidden edges like the raster
// pipeline does.  The compound is std140: two mat4 at offsets 0 and 64.
layout(set = 0, binding = 6, std140) uniform PresentFrame {
    mat4 u_view;  // world -> view (offset 0)
    mat4 u_proj;  // view -> clip (offset 64)
} frame;

layout(push_constant) uniform PresentPush {
    vec4 u_present;  // x = width, y = height, z = denoiseOn, w = frameIndex
    vec4 u_origin;   // x = viewport origin x, y = viewport origin y (pixels)
    vec4 u_denoise;  // x = OIDN result available (sample denoised buffer)
                     // y = denoise upscale factor, z = HDR output, w = exposure
    vec4 u_tone;     // x = tone-mapping operator (0 = clip), yzw reserved
} pc;

layout(location = 0) out vec4 fragColor;

// Tone-mapping operators for the HDR path, mirroring
// data/shaders/vulkan/output/OutputFragment.glsl (kept in sync by hand: the two
// backends own separate shaders and glslangValidator does not resolve shared
// includes here).  Input and output are PQ-normalized linear luminance
// (1.0 = 10000 cd/m^2) with the exposure already applied; the operators are the
// published matrix-free forms (0 = clip, 1 = Reinhard, 2 = ACES, 3 = Hable).
vec3 tonemap_clip(vec3 L)
{
    return clamp(L, 0.0, 1.0);
}

// https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf
vec3 tonemap_reinhard(vec3 L)
{
    return L / (1.0 + L);
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 tonemap_aces(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Uncharted 2 (Hable), http://filmicworlds.com/blog/filmic-tonemapping-operators/.
vec3 tonemap_hable(vec3 x)
{
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    const float W = 11.2;
    vec3 v = x * 2.0;
    vec3 num = v * (A * v + C * B) + D * E;
    vec3 den = v * (A * v + B) + D * F;
    float wnum = W * (A * W + C * B) + D * E;
    float wden = W * (A * W + B) + D * F;
    return clamp((num / den - E / F) / (wnum / wden - E / F), 0.0, 1.0);
}

vec3 tonemap(vec3 L, int mode)
{
    if (mode == 1) return tonemap_reinhard(L);
    if (mode == 2) return tonemap_aces(L);
    if (mode == 3) return tonemap_hable(L);
    return tonemap_clip(L);
}

// Final output transform.  With HDR off (pc.u_denoise.z < 0.5) the color is
// clamped to [0,1] exactly as before, so SDR output is unchanged.  With HDR on
// the linear scene radiance is scaled by pc.u_denoise.w (which maps scene-white
// to the PQ peak of 10000 cd/m^2; 0.02 ~= 200 cd/m^2 reference white),
// tone-mapped by pc.u_tone.x and encoded with the SMPTE ST 2084 (PQ) transfer
// function, BT.2020 primaries.  The swapchain is
// VK_FORMAT_A2B10G10R10_UNORM_PACK32 with the surface tagged Bt2100Pq, so the
// compositor maps it onto the HDR output.
vec4 presentColor(vec3 linearColor)
{
    if (pc.u_denoise.z < 0.5) {
        return vec4(clamp(linearColor, 0.0, 1.0), 1.0);
    }
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 4096.0 * 128.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 4096.0 * 32.0;
    const float c3 = 2392.0 / 4096.0 * 32.0;
    vec3 L = max(linearColor, vec3(0.0)) * pc.u_denoise.w;
    vec3 mapped = tonemap(L, int(pc.u_tone.x + 0.5));
    vec3 Lp = pow(mapped, vec3(m1));
    vec3 pq = pow((c1 + c2 * Lp) / (1.0 + c3 * Lp), vec3(m2));
    return vec4(clamp(pq, 0.0, 1.0), 1.0);
}

// Scene depth (Vulkan [0,1]) of the first-bounce hit at the current pixel.
// The raygen stores the stable (un-jittered centre-sample) hit world position
// in stableDepth[].xyz with the ray distance in .w (a 1e7 sentinel means
// "miss", i.e. background).  Using the stable buffer instead of the jittered
// positions[] G-buffer keeps the edge-overlay depth test from flickering along
// silhouettes.  Project it
// through the traced camera exactly like the visual vertex shader
// (clip.y is flipped but that does not affect Z), then apply the same
// OpenGL->Vulkan depth remap: z_ndc = 0.5*(z_clip/w + 1).  The raster
// composite draws BRep edge lines / point markers at *their own* clip depth,
// so a front face's edge matches this value and passes LEQUAL, while an
// edge of a hidden back face is farther and is culled -- the silhouette
// edge look the raster pipeline produces.
float sceneDepth(ivec2 px, int idx)
{
    vec4 wpos = stableDepth[idx];
    if (wpos.w > 1.0e6) {
        return 1.0; // no hit: background, edge geometry is unoccluded
    }
    vec4 eye = frame.u_view * vec4(wpos.xyz, 1.0);
    vec4 clip = frame.u_proj * eye;
    return clamp(0.5 * (clip.z / clip.w + 1.0), 0.0, 1.0);
}

void main()
{
    vec2 viewportCoord = gl_FragCoord.xy - pc.u_origin.xy;
    ivec2 px = ivec2(viewportCoord);
    const int width = int(max(pc.u_present.x, 1.0));
    const int height = int(max(pc.u_present.y, 1.0));
    const int idx = px.y * width + px.x;
    gl_FragDepth = sceneDepth(px, idx);

    if (pc.u_present.z < 0.5) {
        fragColor = presentColor(
          texture(u_rtImage, viewportCoord / textureSize(u_rtImage, 0)).rgb);
        return;
    }

    // OIDN host-side denoise result: when ready it replaces the raw
    // accumulation entirely (the filter already smoothed the noise).  A
    // result scale > 1 is the low-resolution motion preview: bilinearly
    // upsample it from the leading entries of the denoised buffer.
    if (pc.u_denoise.x > 0.5) {
        float scale = max(pc.u_denoise.y, 1.0);
        if (scale < 1.5) {
            vec4 d = denoised[idx];
            if (d.a > 0.5) {
                fragColor = presentColor(d.rgb);
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
                fragColor = presentColor(d.rgb);
                return;
            }
        }
    }

    vec4 c0 = accum[idx];
    if (c0.a <= 0.0) {
        fragColor = presentColor(vec3(0.0));
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
    fragColor = presentColor(sum / max(wsum, 1.0e-6));
}
