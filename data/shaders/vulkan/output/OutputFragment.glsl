// data/shaders/vulkan/output/OutputFragment.glsl
// Final output / display transform for the raster HDR path.
//
// Samples the scene intermediate (RGBA16F) and writes it to the caller's
// swapchain framebuffer, which is either:
//   - an HDR10 swapchain (VK_FORMAT_A2B10G10R10_UNORM_PACK32) tagged with the
//     BT.2020 + ST 2084 (PQ) color space, or
//   - an SDR swapchain (8-bit), in which case the output is clamped to [0,1].
//
// The intermediate is *display-referred sRGB*, not linear light: the visual and
// background shaders write Coin/FreeCAD colors verbatim (see visual/Fragment.glsl
// and visual/BackgroundFragment.glsl) and the SDR swapchain is a plain UNORM
// format, so nothing in the scene pass ever linearizes.  The HDR branch below
// therefore decodes sRGB to linear (BT.709 primaries + sRGB transfer) and
// converts the primaries to BT.2020 before applying the exposure, tone map and
// PQ encode.  Feeding the PQ inverse EOTF gamma-encoded values instead would
// lift every midtone (while pinning white) and wash the image out.
//
// The SDR branch (HDR off) deliberately stays a plain clamp: it must reproduce
// the pre-HDR, byte-identical SDR output.

#version 450

layout(set = 0, binding = 0) uniform sampler2D u_source;

layout(push_constant) uniform OutputPush {
    // x = HDR output (0 = clamp to [0,1], 1 = exposure + tone map + PQ encode)
    // y = linear exposure/gain applied before the tone map
    // z = tone-mapping operator (see tonemap(); 0 = clip)
    // w = reserved
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

// sRGB inverse EOTF (IEC 61966-2-1): display-referred sRGB code value ->
// linear light.  The scene intermediate carries sRGB code values, so this
// must run before any radiometric operation (exposure/PQ).
vec3 srgb_to_linear(vec3 c)
{
    bvec3 low = lessThanEqual(c, vec3(0.04045));
    vec3 lo = c / 12.92;
    vec3 hi = pow((max(c, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return mix(hi, lo, low);
}

// Linear BT.709 (the sRGB primaries) -> linear BT.2020, which is the gamut the
// HDR10 swapchain is tagged with (Bt2100Pq).  Column-major GLSL mat3; the
// matrix is the Linear709->Linear2020 matrix from ITU-R BT.2087.  White is
// preserved (every row sums to 1), so it does not perturb the exposure.
const mat3 kL709ToL2020 = mat3(
    0.6274038959, 0.0690972894, 0.0163914389,
    0.3292830384, 0.9195403951, 0.0880133079,
    0.0433130657, 0.0113623156, 0.8955952532);

// --- Tone-mapping operators ------------------------------------------------
// Applied to the exposure-scaled linear luminance, exactly the reference
// convention (a plain linear pre-scale, then the curve), with input and output
// in PQ-normalized linear units (1.0 = 10000 cd/m^2).  All four are the
// published, matrix-free forms: the HDR swapchain is tagged BT.2020 without a
// primaries conversion, so operators that bake in a gamut transform (the ACES
// RRT/ODT sRGB<->AP1 matrices, AgX's Rec.2020 matrices) would be inconsistent
// with the pipeline.
//
//   0 = Clip      hard clamp at the PQ peak (the pre-tone-map behavior)
//   1 = Reinhard  simple Reinhard (Reinhard et al. 2002); neutral, linear at 0
//   2 = ACES      Narkowicz's ACES filmic fit (2016); lifts midtones
//   3 = Hable     Uncharted 2 filmic curve (Hable 2010); strong shoulder

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

// Uncharted 2 (Hable), http://filmicworlds.com/blog/filmic-tonemapping-operators/:
// the raw curve, normalized by its value at the 11.2 white point.
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
    // Decode the display-referred sRGB scene to linear light and convert the
    // sRGB/BT.709 primaries to the BT.2020 gamut, then expose + tone map + PQ.
    vec3 lin = kL709ToL2020 * srgb_to_linear(clamp(c, 0.0, 1.0));
    vec3 L = lin * pc.u_params.y;
    vec3 mapped = tonemap(L, int(pc.u_params.z + 0.5));
    fragColor = vec4(clamp(linear_to_pq(mapped), 0.0, 1.0), 1.0);
}
