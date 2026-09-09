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

// Shared lighting container + Blinn-Phong evaluators (see LightCommon.glsl).
#include "../common/LightCommon.glsl"

layout(push_constant) uniform PushConstants {
    mat4  u_proj;         // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes
    vec4  u_texParams;    // offset 96, 16 bytes
    vec4  u_texBlend;     // offset 112, 16 bytes
    float u_pointSize;    // offset 128, 16 bytes (pad[3])
    vec4  u_lineParams;   // offset 144, 16 bytes: x = stipple factor,
                        // y = round points
                          //   y = round points, z = line primitive,
                          //   w = point primitive
} pc;

// Lighting constant block (written once per lighting setup per frame).  The
// light arrays match the shared CoinLightSet layout byte-for-byte (std140
// offsets land at 16/144/272/400/528/656), so the container is shared with
// the RT backend via LightCommon.glsl.
layout(set = 0, binding = 0, std140) uniform LightingBlock {
    vec4  u_ambientLight;         // offset 0
    CoinLightSet lights;          // offset 16
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
} draw;

layout(set = 1, binding = 1) uniform sampler2D u_texture;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_eyePos;
layout(location = 2) in vec3 v_eyeNormal;
layout(location = 3) in vec2 v_texcoord;

layout(location = 0) out vec4 fragColor;

// Per-fragment Blinn-Phong (matches the GL model's terms, but evaluated
// here instead of per vertex): interpolated normals give a smooth diffuse
// gradient and a soft specular highlight even on coarse tessellations.
// The light container and the Blinn-Phong loop live in LightCommon.glsl and
// are shared with the ray-tracing backend; this wrapper supplies the eye-space
// vectors (raster evaluates in eye space) and the raster two-sided test.
vec3 coin_vulkan_lighting(vec3 eyePos, vec3 eyeNormal, vec3 baseColor)
{
    vec3 N = normalize(eyeNormal);
    vec3 V = normalize(-eyePos);
    if (draw.u_materialParams.y > 0.5 && gl_FrontFacing) {
        N = -N;
    }
    vec3 ambientTerm = lighting.u_ambientLight.rgb * draw.u_materialAmbient.rgb;
    int lightCount = int(draw.u_materialParams.z);
    float shininess = max(draw.u_materialParams.x * 128.0, 0.0);
    return coinGouraudCls(lighting.lights, lightCount, eyePos, N, V,
                          baseColor, draw.u_materialSpecular.rgb, shininess,
                          ambientTerm, draw.u_emissiveColor.rgb);
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
        : coin_vulkan_lighting(v_eyePos, v_eyeNormal, v_color.rgb);
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
