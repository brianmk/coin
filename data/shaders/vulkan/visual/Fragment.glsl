// data/shaders/vulkan/visual/Fragment.glsl
// Vulkan visual-pass fragment shader for the retained render backend.
//
// Receives the vertex-lit (Gouraud) color computed by the vertex stage.  The
// base color is either the per-vertex color or the uniform diffuse color,
// with material opacity applied to the alpha channel.  An optional embedded
// texture (set 0, binding 1) modulates, replaces, or blends the base color
// according to the command's SoTextureModel.

#version 450

layout(push_constant) uniform PushConstants {
    mat4  u_proj;         // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes
    vec4  u_texParams;    // offset 96, 16 bytes
    vec4  u_texBlend;     // offset 112, 16 bytes
} pc;

layout(set = 0, binding = 1) uniform sampler2D u_texture;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_litColor;
layout(location = 2) in vec2 v_texcoord;

layout(location = 0) out vec4 fragColor;

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
    // Mirror the retained GL visual program: vertex alpha already carries the
    // material transparency for PER_FACE vertex colors (flagged on the
    // command); otherwise the uniform material opacity multiplies the vertex
    // alpha.
    float materialAlpha = pc.u_color.a;
    if (pc.u_flags.x > 0.5 && pc.u_flags.y > 0.5) {
        materialAlpha = 1.0;
    }

    vec3 rgb = v_litColor;
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
