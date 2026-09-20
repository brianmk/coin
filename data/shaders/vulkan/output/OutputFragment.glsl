// data/shaders/vulkan/output/OutputFragment.glsl
// Final output / display transform for the raster HDR path.
//
// Samples the linear scene-radiance intermediate (RGBA16F) and writes it to the
// caller's swapchain framebuffer, which is either:
//   - an HDR10 swapchain (VK_FORMAT_A2B10G10R10_UNORM_PACK32) tagged with the
//     BT.2020 + ST 2084 (PQ) color space, or
//   - an SDR swapchain (8-bit), in which case the output is clamped to [0,1].
//
// Doing the transfer function here, once, after all geometry/transparency has
// blended in linear light, is what makes the pipeline color-correct: blending
// and MSAA resolve happen on linear radiance, and only the final displayed
// value is encoded.

#version 450

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(push_constant) uniform OutputPush {
    // x = HDR output (0 = clamp to [0,1], 1 = PQ encode)
    // y = linear exposure/gain applied before the PQ encode
    // z, w = reserved
    vec4 u_params;
} pc;

layout(location = 0) out vec4 fragColor;

// SMPTE ST 2084 (PQ) inverse EOTF: linear luminance (normalized so 1.0 = the
// 10000 cd/m^2 peak) -> non-linear PQ code value in [0,1].
vec3 linear_to_pq(vec3 L)
{
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 4096.0 * 128.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 4096.0 * 32.0;
    const float c3 = 2392.0 / 4096.0 * 32.0;
    vec3 Lp = pow(max(L, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * Lp) / (1.0 + c3 * Lp), vec3(m2));
}

void main()
{
    // gl_FragCoord is in framebuffer pixels; the intermediate has the same
    // extent, so the normalized coordinate is a direct lookup.  Vulkan's
    // top-left origin matches the render-target convention, so no Y flip.
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_source, 0));
    vec3 c = texture(u_source, uv).rgb;

    if (pc.u_params.x < 0.5) {
        fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
        return;
    }
    vec3 L = max(c, vec3(0.0)) * pc.u_params.y;
    fragColor = vec4(clamp(linear_to_pq(L), 0.0, 1.0), 1.0);
}
