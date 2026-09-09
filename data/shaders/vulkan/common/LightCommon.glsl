// data/shaders/vulkan/common/LightCommon.glsl
// Shared Vulkan lighting container + space-agnostic evaluators.
//
// Textual #include consumed by BOTH the raster visual pipeline
// (visual/Vertex.glsl, visual/Fragment.glsl) and the ray-tracing pipeline
// (rt/PathTrace.glsl, rt/RTRayTrace.glsl, rt/Raygen.glsl, rt/ClosestHit.glsl).
// It carries no #version so it can be textually spliced into a wider shader.
//
// SPACE CONVENTION
// ---------------
// Every light field (direction/position) is consumed in the SAME space the
// sample point and the normal/view vectors are supplied in.  The retained
// raster backend pushes eye-space lights (via SoRenderIR::lightToEye) and
// evaluates in eye space; the ray-tracing backend pushes world-space lights
// (the standard IR convention) and evaluates in world space.  Because the
// shading terms below only use dot products (and a reflection-dependent
// specular split), the same functions are correct in either space: rotating
// all of {N, V, L, hit point} together leaves every dot product unchanged.
//
// So the "fork" between raster and ray tracing is NOT in this file: it is the
// choice of which space the producer fills the light set for, made once at the
// call site.  This module is deliberately space-agnostic so both pipelines can
// share it verbatim instead of maintaining two copies of the Blinn-Phong loop.
//
// LAYOUT
// ------
// The container below is byte-compatible with both the std140 LightingBlock
// UBO of the raster pipeline AND the std430 RTMaterial SSBO of the RT pipeline
// (verified via SPIR-V member offsets: light arrays land at 16/144/272/... in
// std140 and at 80/208/336/... in the nested std430 RTMaterial).  That is why
// embedding one `CoinLightSet` field in each is safe: no C++ producer change
// is required, only the GLSL struct layout is unified.

#ifndef COIN_LIGHT_COMMON_GLSL
#define COIN_LIGHT_COMMON_GLSL

const int COIN_MAX_LIGHTS = 8;

// One light in the set is a full "type/color/direction/position/attenuation/
// spot" tuple spread across parallel arrays.  Helper functions index the same
// slot `i` in every array, matching the SoLightingBlock/RTMaterial packing.
struct CoinLightSet {
  vec4  lightType[COIN_MAX_LIGHTS];       // x = type (0 dir, 1 point, 2 spot)
  vec4  lightColor[COIN_MAX_LIGHTS];
  vec4  lightDirection[COIN_MAX_LIGHTS];
  vec4  lightPosition[COIN_MAX_LIGHTS];
  vec4  lightAttenuation[COIN_MAX_LIGHTS];
  vec4  lightSpot[COIN_MAX_LIGHTS];       // x = spot cutoff cos, y = exponent
};

// Per-light shading direction, attenuation and spot factor, in the space the
// light fields and \a point are expressed in.  Fills \a L (surface -> light),
// \a attenuation and \a spotFactor for slot \a i.  Returns false when the
// light is degenerate (zero point-to-light distance) or the surface lies
// outside the spot cone, so the caller can skip it without writing output.
bool coinResolveLight(const CoinLightSet cl, int i, vec3 point,
                      out vec3 L, out float attenuation, out float spotFactor)
{
  attenuation = 1.0f;
  spotFactor = 1.0f;
  if (cl.lightType[i].x > 0.5f) {
    vec3 lightVector = cl.lightPosition[i].xyz - point;
    float distanceToLight = length(lightVector);
    if (distanceToLight <= 0.0001f) return false;
    L = lightVector / distanceToLight;
    vec3 att = cl.lightAttenuation[i].xyz;
    attenuation = 1.0f / max(att.z + att.y * distanceToLight +
                             att.x * distanceToLight * distanceToLight, 0.0001f);
    if (cl.lightType[i].x > 1.5f) {
      vec3 coneDir = normalize(cl.lightDirection[i].xyz);
      vec3 fromLight = normalize(point - cl.lightPosition[i].xyz);
      float spotCos = dot(coneDir, fromLight);
      if (spotCos < cl.lightSpot[i].x) return false;
      spotFactor = pow(max(spotCos, 0.0f), cl.lightSpot[i].y);
    }
  } else {
    // Directional light: the direction IS the surface -> light vector.
    L = cl.lightDirection[i].xyz;
  }
  return true;
}

// Blinn-Phong diffuse + specular for a single light slot \a i, given the
// resolved surface -> light direction.  \a N and \a V are unit vectors in the
// same space as \a L; \a shininess is the exponent (may be 0 for pure rough).
// Space-agnostic (dot products only).
vec3 coinShadeLightCls(const CoinLightSet cl, int i, vec3 N, vec3 V, vec3 L,
                       vec3 baseColor, vec3 specularColor, float shininess)
{
  vec3 Ln = normalize(L);
  float NdotL = max(dot(N, Ln), 0.0f);
  vec3 H = normalize(Ln + V);
  float NdotH = max(dot(N, H), 0.0f);
  float specularFactor = shininess > 0.0f ? pow(NdotH, shininess) : 0.0f;
  vec3 diffuse = baseColor * NdotL;
  vec3 specular = specularColor * specularFactor;
  return cl.lightColor[i].rgb * (diffuse + specular);
}

// Evaluate the whole light set at \a point with Blinn-Phong shading.  Returns
// rgb += ambient (the caller's pre-folded ambient term).  \a toEval (the
// "participating" predicate) and the per-light shadow term are NOT handled
// here: it accumulates only the analytic diffuse/specular contribution over
// the COIN_MAX_LIGHTS slots.  \a N and \a V are unit vectors in the same space.
// Space-agnostic (dot products only).
vec3 coinShadeLightsCls(const CoinLightSet cl, int lightCount, vec3 point,
                        vec3 N, vec3 V, vec3 baseColor, vec3 specularColor,
                        float shininess)
{
  vec3 rgb = vec3(0.0f);
  for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
    if (i >= lightCount) break;
    vec3 L;
    float attenuation;
    float spotFactor;
    if (!coinResolveLight(cl, i, point, L, attenuation, spotFactor)) continue;
    rgb += coinShadeLightCls(cl, i, N, V, L, baseColor, specularColor,
                             shininess) * attenuation * spotFactor;
  }
  return rgb;
}

// Full Gouraud/Blinn-Phong surface shading, matching the raster visual program
// and the RT preview mode.  \a ambientTerm is the pre-folded ambient
// contribution (scene ambient * material ambient).  Returns the clamped lit
// color including \a emissive.  \a baseColor is the surface diffuse.
// Space-agnostic.
vec3 coinGouraudCls(const CoinLightSet cl, int lightCount, vec3 point,
                    vec3 N, vec3 V, vec3 baseColor, vec3 specularColor,
                    float shininess, vec3 ambientTerm, vec3 emissive)
{
  vec3 litColor = ambientTerm + coinShadeLightsCls(cl, lightCount, point,
                                                   N, V, baseColor,
                                                   specularColor, shininess);
  return clamp(litColor + emissive, 0.0f, 1.0f);
}

#endif // COIN_LIGHT_COMMON_GLSL
