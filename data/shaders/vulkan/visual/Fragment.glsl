// data/shaders/vulkan/visual/Fragment.glsl
// Vulkan visual-pass fragment shader for the retained render backend.
//
// Receives the vertex-lit (Gouraud) color computed by the vertex stage.  The
// base color is either the per-vertex color or the uniform diffuse color,
// with material opacity applied to the alpha channel.  An optional embedded
// texture (set 1, binding 1) modulates, replaces, or blends the base color
// according to the command's SoTextureModel.

#version 450
#extension GL_GOOGLE_include_directive : require

#include "../common/MaterialCommon.glsl"

layout(push_constant) uniform PushConstants {
    vec4  u_color;        // offset 0, 16 bytes
    vec4  u_flags;        // offset 16, 16 bytes
    vec4  u_texParams;    // offset 32, 16 bytes
    vec4  u_texBlend;     // offset 48, 16 bytes
    float u_pointSize;    // offset 64, 16 bytes (pad[3])
    vec4  u_lineParams;   // offset 80, 16 bytes: x = stipple factor,
                          //   y = round points, z = line primitive,
                          //   w = point primitive
} pc;

// Lighting constant block (written once per lighting setup per frame).
layout(set = 0, binding = 0, std140) uniform LightingBlock {
    vec4  u_ambientLight;         // offset 0
    vec4  u_lightType[8];         // offset 16
    vec4  u_lightColor[8];        // offset 144
    vec4  u_lightDirection[8];    // offset 272
    vec4  u_lightPosition[8];     // offset 400
    vec4  u_lightAttenuation[8];  // offset 528
    vec4  u_lightSpotParams[8];   // offset 656
} lighting;

// Per-draw block (material), selected by a dynamic offset.
layout(set = 1, binding = 0, std140) uniform DrawBlock {
    mat4  u_view;                 // offset 0
    mat4  u_model;                // offset 64
    vec4  u_emissiveColor;        // offset 128
    vec4  u_materialAmbient;      // offset 144
    vec4  u_materialSpecular;     // offset 160
    vec4  u_materialParams;       // offset 176: x=shininess, y=twoSided,
                                  //            z=lightCount, w=shadingModel
    mat4  u_proj;                 // offset 192: projection (view/model above)
    vec4  u_materialPbr;          // offset 256: x=metalness, y=roughness,
                                  //            z=physical-material enabled
    vec4  u_materialMapParams;    // offset 272: x=roughness strength,
                                  //            y=normal strength,
                                  //            z=emissive intensity
} draw;

layout(set = 1, binding = 1) uniform sampler2D u_texture;

// Optional secondary PBR maps (set 2).  When a map is absent the corresponding
// default texture is bound (white roughness, flat normal, black emissive), so
// sampling is always safe and the arithmetic is a no-op.
layout(set = 2, binding = 0) uniform sampler2D u_roughnessMap;
layout(set = 2, binding = 1) uniform sampler2D u_normalMap;
layout(set = 2, binding = 2) uniform sampler2D u_emissiveMap;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_eyePos;
layout(location = 2) in vec3 v_eyeNormal;
layout(location = 3) in vec2 v_texcoord;

layout(location = 0) out vec4 fragColor;

const int COIN_MAX_LIGHTS = 8;

// Emissive contribution: the scalar emissive colour plus the optional emissive
// map (set 2, binding 2) scaled by its authored intensity.  The presence
// bitmask gates the sample so an absent map costs nothing.
vec3 coin_vulkan_emissive()
{
    vec3 emissive = draw.u_emissiveColor.rgb;
    if ((int(draw.u_materialMapParams.w) & 4) != 0) {
        emissive += texture(u_emissiveMap, v_texcoord).rgb
            * max(draw.u_materialMapParams.z, 0.0);
    }
    return emissive;
}

// Perturb an eye-space normal by the optional tangent-space normal map
// (set 2, binding 1).  The tangent basis is derived per-fragment from the
// screen-space derivatives of the eye-space position and the texture
// coordinate (no per-vertex tangent stream), which is adequate for the CAD
// preview and needs no geometry changes.  The map is decoded from [0,1] to
// [-1,1] and its xy scaled by the authored normal strength.
vec3 coin_vulkan_perturb_normal(vec3 N)
{
    if ((int(draw.u_materialMapParams.w) & 2) == 0) {
        return N;
    }
    vec3 sampled = texture(u_normalMap, v_texcoord).xyz * 2.0 - 1.0;
    sampled.xy *= max(draw.u_materialMapParams.y, 0.0);

    vec3 dp1 = dFdx(v_eyePos);
    vec3 dp2 = dFdy(v_eyePos);
    vec2 duv1 = dFdx(v_texcoord);
    vec2 duv2 = dFdy(v_texcoord);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1.0e-8);
    mat3 tbn = mat3(T * invmax, B * invmax, N);
    return normalize(tbn * sampled);
}

// Per-fragment Blinn-Phong (matches the GL model's terms, but evaluated
// here instead of per vertex): interpolated normals give a smooth diffuse
// gradient and a soft specular highlight even on coarse tessellations.
vec3 coin_vulkan_lighting(vec3 eyePos, vec3 eyeNormal, vec3 baseColor)
{
    vec3 N = normalize(eyeNormal);
    // View vector to the viewer.  For a perspective camera the viewer is the
    // eye-space origin, so -eyePos is correct.  For an orthographic camera the
    // viewer is at infinity: the view direction is the constant eye-space +Z.
    // Using -eyePos there made dot(N, V) cross zero inside the silhouette (at
    // r/R = sqrt(1 - (R/D)^2) instead of at the silhouette), so the two-sided
    // normal flip below triggered over the front surface and drew a hard-edged
    // dark ring.  u_proj[2][3] is the perspective-divide term: 0 for an
    // orthographic projection, -1 for a perspective one.
    vec3 V = (draw.u_proj[2][3] == 0.0) ? vec3(0.0, 0.0, 1.0)
                                      : normalize(-eyePos);
    if (draw.u_materialParams.y > 0.5 && dot(N, V) < 0.0) {
        N = -N;
    }
    N = coin_vulkan_perturb_normal(N);
    // sceneAmbient * materialAmbient; keep in sync with
    // SoRenderIR::effectiveMaterialAmbient (the path tracer's definition, used
    // for the same term so raster and RT agree).
    vec3 litColor = lighting.u_ambientLight.rgb * draw.u_materialAmbient.rgb;

    for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
        if (i >= int(draw.u_materialParams.z)) break;

        vec3 L = lighting.u_lightDirection[i].xyz;
        float attenuation = 1.0;
        float spotFactor = 1.0;
        if (lighting.u_lightType[i].x > 0.5) {
            vec3 lightVector = lighting.u_lightPosition[i].xyz - eyePos;
            float distanceToLight = length(lightVector);
            if (distanceToLight <= 0.0001) continue;
            L = lightVector / distanceToLight;
            vec3 att = lighting.u_lightAttenuation[i].xyz;
            attenuation = 1.0 / max(att.z + att.y * distanceToLight +
                                    att.x * distanceToLight * distanceToLight,
                                    0.0001);
            if (lighting.u_lightType[i].x > 1.5) {
                vec3 coneDir = normalize(lighting.u_lightDirection[i].xyz);
                vec3 fromLight =
                    normalize(eyePos - lighting.u_lightPosition[i].xyz);
                float spotCos = dot(coneDir, fromLight);
                if (spotCos < lighting.u_lightSpotParams[i].x) continue;
                spotFactor = pow(max(spotCos, 0.0),
                                 lighting.u_lightSpotParams[i].y);
            }
        }

        vec3 Ln = normalize(L);
        float NdotL = max(dot(N, Ln), 0.0);
        vec3 H = normalize(Ln + V);
        float NdotH = max(dot(N, H), 0.0);
        float shininess = max(draw.u_materialParams.x * 128.0, 0.0);
        float specularFactor = shininess > 0.0 ? pow(NdotH, shininess) : 0.0;
        vec3 diffuse = baseColor * NdotL;
        vec3 specular = draw.u_materialSpecular.rgb * specularFactor;
        litColor += lighting.u_lightColor[i].rgb * attenuation * spotFactor *
                    (diffuse + specular);
    }
    return clamp(litColor + coin_vulkan_emissive(), 0.0, 1.0);
}

// Metallic-roughness (GGX) direct lighting.  Selected when the command
// carries an authored physical material (u_materialPbr.z > 0.5); otherwise
// the legacy Blinn-Phong path above is used, so default materials render
// exactly as before.
//
//   metalness 0 = dielectric (plastic/concrete), 1 = conductor (metal)
//   roughness 0 = mirror, 1 = fully diffuse
//
// Direct lights only.  Environment/IBL is approximated by the ambient term
// scaled by the diffuse albedo, which keeps the fast viewport cheap.
vec3 coin_vulkan_pbr_lighting(vec3 eyePos, vec3 eyeNormal, vec3 baseColor)
{
    vec3 N = normalize(eyeNormal);
    vec3 V = (draw.u_proj[2][3] == 0.0) ? vec3(0.0, 0.0, 1.0)
                                      : normalize(-eyePos);
    if (draw.u_materialParams.y > 0.5 && dot(N, V) < 0.0) {
        N = -N;
    }
    N = coin_vulkan_perturb_normal(N);

    float metalness = clamp(draw.u_materialPbr.x, 0.0, 1.0);
    // Optional roughness map: sampled with the base texture coordinates and
    // blended in by the authored strength (0 ignores the map, 1 multiplies the
    // scalar roughness by the sampled value).  Gated by the presence bitmask.
    float roughnessFactor = 1.0;
    if ((int(draw.u_materialMapParams.w) & 1) != 0) {
        float sampledRoughness = texture(u_roughnessMap, v_texcoord).r;
        roughnessFactor =
            mix(1.0, sampledRoughness, clamp(draw.u_materialMapParams.x, 0.0, 1.0));
    }
    float roughness =
        clamp(draw.u_materialPbr.y * roughnessFactor, 0.045, 1.0);
    float alpha = coin_pbr_alpha(roughness);

    vec3 F0 = coin_pbr_f0(baseColor, metalness);
    vec3 diffuseColor = baseColor * (1.0 - metalness);
    float NdotV = max(dot(N, V), 1.0e-4);

    vec3 litColor = lighting.u_ambientLight.rgb * draw.u_materialAmbient.rgb *
                    diffuseColor;

    for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
        if (i >= int(draw.u_materialParams.z)) break;

        vec3 L = lighting.u_lightDirection[i].xyz;
        float attenuation = 1.0;
        float spotFactor = 1.0;
        if (lighting.u_lightType[i].x > 0.5) {
            vec3 lightVector = lighting.u_lightPosition[i].xyz - eyePos;
            float distanceToLight = length(lightVector);
            if (distanceToLight <= 0.0001) continue;
            L = lightVector / distanceToLight;
            vec3 att = lighting.u_lightAttenuation[i].xyz;
            attenuation = 1.0 / max(att.z + att.y * distanceToLight +
                                    att.x * distanceToLight * distanceToLight,
                                    0.0001);
            if (lighting.u_lightType[i].x > 1.5) {
                vec3 coneDir = normalize(lighting.u_lightDirection[i].xyz);
                vec3 fromLight =
                    normalize(eyePos - lighting.u_lightPosition[i].xyz);
                float spotCos = dot(coneDir, fromLight);
                if (spotCos < lighting.u_lightSpotParams[i].x) continue;
                spotFactor = pow(max(spotCos, 0.0),
                                 lighting.u_lightSpotParams[i].y);
            }
        }

        vec3 Ln = normalize(L);
        float NdotL = max(dot(N, Ln), 0.0);
        if (NdotL <= 0.0) continue;
        vec3 H = normalize(Ln + V);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        // GGX/Trowbridge-Reitz normal distribution.
        float D = coin_pbr_d_ggx(NdotH, alpha);

        // Smith height-correlated visibility (Schlick-GGX with k = r^2/2).
        float G = coin_pbr_g_smith(NdotV, NdotL, alpha);

        // Schlick Fresnel.
        vec3 F = coin_pbr_f_schlick(VdotH, F0);

        vec3 specular = (D * G) * F / max(4.0 * NdotV * NdotL, 1.0e-4);
        vec3 kd = (vec3(1.0) - F) * (1.0 - metalness);
        vec3 diffuse = kd * diffuseColor / COIN_PI;

        litColor += lighting.u_lightColor[i].rgb * attenuation * spotFactor *
                    (diffuse + specular) * NdotL;
    }
    return clamp(litColor + coin_vulkan_emissive(), 0.0, 1.0);
}

bool coin_vulkan_alpha_test_pass(float alpha, int function, float reference)
{
    if (function == 1) return false;  // NEVER
    if (function == 2) return true;   // ALWAYS
    if (function == 3) return alpha < reference;
    if (function == 4) return alpha <= reference;
    if (function == 5) return abs(alpha - reference) < 0.0001;
    if (function == 6) return alpha >= reference;
    if (function == 7) return alpha > reference;
    if (function == 8) return abs(alpha - reference) >= 0.0001;
    return true;                       // NONE
}

void main()
{
    // Round point glyphs (SO_POINT_SHAPE_ROUND): discard the fragment outside
    // the circle inscribed in the point square, mirroring the GL point shader.
    if (pc.u_lineParams.y > 0.5) {
        vec2 pointCoord = gl_PointCoord * 2.0 - 1.0;
        if (dot(pointCoord, pointCoord) > 1.0) {
            discard;
        }
    }

    // Mirror the retained GL visual program: vertex alpha already carries the
    // material transparency for PER_FACE vertex colors (flagged on the
    // command); otherwise the uniform material opacity multiplies the vertex
    // alpha.
    float materialAlpha = pc.u_color.a;
    if (pc.u_flags.x > 0.5 && pc.u_flags.y > 0.5) {
        materialAlpha = 1.0;
    }

    vec3 rgb = draw.u_materialParams.w < 0.5
        ? v_color.rgb
        : (draw.u_materialPbr.z > 0.5
            ? coin_vulkan_pbr_lighting(v_eyePos, v_eyeNormal, v_color.rgb)
            : coin_vulkan_lighting(v_eyePos, v_eyeNormal, v_color.rgb));
    float primaryAlpha = v_color.a;
    float alpha = primaryAlpha * materialAlpha;

    if (pc.u_flags.z > 0.5) {
        vec4 texel = texture(u_texture, v_texcoord);

        // Pixel text is CPU-rasterized by the producer (SoText2-style
        // overlays): the texture already carries the final RGBA, including
        // opacity and the text color.  Emit it verbatim instead of modulating
        // it by material diffuse (which would double-tint and darken text
        // versus the legacy glDrawPixels path).
        if (pc.u_texParams.w > 0.5) {
            // Match the legacy GL_ALPHA_TEST(GL_GREATER, 0.3f) used for
            // glDrawPixels so fully-transparent glyph padding stays clean.
            if (texel.a <= 0.3) {
                discard;
            }
            fragColor = texel;
            return;
        }

        float textureAlpha = pc.u_flags.w > 0.5
            ? texel.a : texel.a * materialAlpha;

        int model = int(pc.u_texParams.x);
        if (model == 1) {
            // DECAL
            rgb = mix(rgb, texel.rgb, texel.a);
        }
        else if (model == 2) {
            // BLEND
            rgb = mix(rgb, pc.u_texBlend.rgb, texel.rgb);
            alpha = primaryAlpha * textureAlpha;
        }
        else if (model == 3) {
            // REPLACE
            rgb = texel.rgb;
            alpha = primaryAlpha * textureAlpha;
        }
        else {
            // MODULATE (default)
            rgb = rgb * texel.rgb;
            alpha = primaryAlpha * textureAlpha;
        }
    }

    if (!coin_vulkan_alpha_test_pass(alpha, int(pc.u_texParams.y),
                                     pc.u_texParams.z)) {
        discard;
    }
    fragColor = vec4(rgb, alpha);
}
