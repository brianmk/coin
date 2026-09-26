// data/shaders/vulkan/common/MaterialCommon.glsl
// Shared material-shading primitives for every retained backend.
//
// The raster fragment shader (visual/Fragment.glsl) and the compute path tracer
// (rt/PathTrace.glsl, via rt/RTShadingCommon.glsl) evaluate the same
// metallic-roughness microfacet BRDF.  Single-sourcing the primitive terms here
// keeps the two backends from drifting apart, exactly as SoRenderIR::packMaterialBlock
// single-sources the CPU-side material mapping.
//
// Textual #include only: this file has no #version and must be included after
// the host shader's #version and GL_GOOGLE_include_directive extension.  It is
// pulled in once per translation unit.

const float COIN_PI = 3.14159265358979323846;

// Perceptual roughness -> GGX alpha (roughness^2), guarded so a degenerate
// roughness (0 or >1) never produces a singular distribution.
float coin_pbr_alpha(float roughness)
{
    float a = clamp(roughness, 0.0, 1.0);
    return clamp(a * a, 0.001, 1.0);
}

// Dielectric base reflectance blended toward the albedo for metals.
vec3 coin_pbr_f0(vec3 baseColor, float metalness)
{
    return mix(vec3(0.04), baseColor, clamp(metalness, 0.0, 1.0));
}

// GGX / Trowbridge-Reitz normal distribution.
float coin_pbr_d_ggx(float NdotH, float a)
{
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(COIN_PI * d * d, 1e-7);
}

// Schlick-GGX with k = a/2 (Smith height-correlated visibility term).
float coin_pbr_g_schlick(float NdotX, float a)
{
    float k = a * 0.5;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-7);
}

float coin_pbr_g_smith(float NdotV, float NdotL, float a)
{
    return coin_pbr_g_schlick(NdotV, a) * coin_pbr_g_schlick(NdotL, a);
}

// Schlick Fresnel approximation.
vec3 coin_pbr_f_schlick(float VdotH, vec3 F0)
{
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}
