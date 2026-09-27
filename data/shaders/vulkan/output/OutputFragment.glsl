// data/shaders/vulkan/output/OutputFragment.glsl
// Final output / display transform for the raster HDR path.
//
// Samples the scene intermediate (RGBA16F) and writes it to the caller's
// swapchain framebuffer, which is either:
//   - an FP16 scRGB swapchain (VK_FORMAT_R16G16B16A16_SFLOAT) tagged with an
//     extended-linear sRGB (scRGB) image description through the compositor's
//     color-management protocol, or
//   - an SDR swapchain (8-bit), in which case the output is clamped to [0,1].
//
// The intermediate is *display-referred sRGB*, not linear light: the visual and
// background shaders write Coin/FreeCAD colors verbatim (see visual/Fragment.glsl
// and visual/BackgroundFragment.glsl) and the SDR swapchain is a plain UNORM
// format, so nothing in the scene pass ever linearizes.  The HDR branch below
// therefore decodes sRGB to linear (BT.709 primaries) and writes linear light
// into the FP16 surface.
//
// scRGB is deliberately NOT PQ.  The surface is tagged extended-linear sRGB, so
// its diffuse white is linear 1.0 (the compositor's reference white) and values
// above 1.0 are HDR highlights; the compositor anchors the reference white and
// maps the extended range onto the actual output.  This is the representation
// Blender and gamescope use for Wayland HDR.  Encoding PQ into a surface the
// compositor treats as SDR is exactly what washed the image out: the SDR band
// of a PQ signal (roughly 0..0.58) then renders as a compressed grey ramp.
//
// The SDR branch (HDR off) deliberately stays a plain clamp: it must reproduce
// the pre-HDR, byte-identical SDR output.

#version 450

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(push_constant) uniform OutputPush {
    // x = HDR output (0 = clamp to [0,1], 1 = scRGB extended-linear)
    // y = diffuse-white gain (1.0 presents linear 1.0 at the reference white)
    // z = reserved
    // w = reserved
    vec4 u_params;
} pc;

layout(location = 0) out vec4 fragColor;

// sRGB inverse EOTF (IEC 61966-2-1): display-referred sRGB code value ->
// linear light.  The scene intermediate carries sRGB code values, so this must
// run to turn the display-referred frame into the linear scRGB signal.
vec3 srgb_to_linear(vec3 c)
{
    bvec3 low = lessThanEqual(c, vec3(0.04045));
    vec3 lo = c / 12.92;
    vec3 hi = pow((max(c, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return mix(hi, lo, low);
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
    // Decode the display-referred sRGB scene to linear light and apply the
    // diffuse-white gain.  The FP16 scRGB surface carries values above 1.0 as
    // HDR highlights; the compositor anchors diffuse white (1.0) and maps the
    // extended range onto the output.  No PQ, no BT.2020 gamut conversion: the
    // surface is tagged extended-linear sRGB.
    vec3 lin = max(srgb_to_linear(clamp(c, 0.0, 1.0)) * pc.u_params.y, vec3(0.0));
    fragColor = vec4(lin, 1.0);
}
