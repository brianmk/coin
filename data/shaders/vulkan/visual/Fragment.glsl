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
    float u_pointSize;    // offset 128, 16 bytes (pad[3])
} pc;

layout(set = 0, binding = 0, std140) uniform VisualBlock {
    mat4  u_view;                 // offset 0
    mat4  u_model;                // offset 64
    vec4  u_emissiveColor;        // offset 128
    vec4  u_ambientLight;         // offset 144
    vec4  u_materialAmbient;      // offset 160
    vec4  u_materialSpecular;     // offset 176
    vec4  u_materialParams;       // offset 192: x=shininess, y=twoSided,
                                  //            z=lightCount, w=shadingModel
    vec4  u_lightType[8];         // offset 208
    vec4  u_lightColor[8];        // offset 336
    vec4  u_lightDirection[8];    // offset 464
    vec4  u_lightPosition[8];     // offset 592
    vec4  u_lightAttenuation[8];  // offset 720
    vec4  u_lightSpotParams[8];   // offset 848
} visual;

layout(set = 0, binding = 1) uniform sampler2D u_texture;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_eyePos;
layout(location = 2) in vec3 v_eyeNormal;
layout(location = 3) in vec2 v_texcoord;

layout(location = 0) out vec4 fragColor;

const int COIN_MAX_LIGHTS = 8;

// Per-fragment Blinn-Phong (matches the GL model's terms, but evaluated
// here instead of per vertex): interpolated normals give a smooth diffuse
// gradient and a soft specular highlight even on coarse tessellations.
vec3 coin_vulkan_lighting(vec3 eyePos, vec3 eyeNormal, vec3 baseColor)
{
    vec3 N = normalize(eyeNormal);
    vec3 V = normalize(-eyePos);
    if (visual.u_materialParams.y > 0.5 && dot(N, V) < 0.0) {
        N = -N;
    }
    vec3 litColor = visual.u_ambientLight.rgb * visual.u_materialAmbient.rgb;

    for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
        if (i >= int(visual.u_materialParams.z)) break;

        vec3 L = visual.u_lightDirection[i].xyz;
        float attenuation = 1.0;
        float spotFactor = 1.0;
        if (visual.u_lightType[i].x > 0.5) {
            vec3 lightVector = visual.u_lightPosition[i].xyz - eyePos;
            float distanceToLight = length(lightVector);
            if (distanceToLight <= 0.0001) continue;
            L = lightVector / distanceToLight;
            vec3 att = visual.u_lightAttenuation[i].xyz;
            attenuation = 1.0 / max(att.z + att.y * distanceToLight +
                                    att.x * distanceToLight * distanceToLight,
                                    0.0001);
            if (visual.u_lightType[i].x > 1.5) {
                vec3 coneDir = normalize(visual.u_lightDirection[i].xyz);
                vec3 fromLight =
                    normalize(eyePos - visual.u_lightPosition[i].xyz);
                float spotCos = dot(coneDir, fromLight);
                if (spotCos < visual.u_lightSpotParams[i].x) continue;
                spotFactor = pow(max(spotCos, 0.0),
                                 visual.u_lightSpotParams[i].y);
            }
        }

        vec3 Ln = normalize(L);
        float NdotL = max(dot(N, Ln), 0.0);
        if (NdotL <= 0.0) continue;
        vec3 H = normalize(Ln + V);
        float NdotH = max(dot(N, H), 0.0);
        float shininess = max(visual.u_materialParams.x * 128.0, 0.0);
        float specularFactor = shininess > 0.0 ? pow(NdotH, shininess) : 0.0;
        vec3 diffuse = baseColor * NdotL;
        vec3 specular = visual.u_materialSpecular.rgb * specularFactor;
        litColor += visual.u_lightColor[i].rgb * attenuation * spotFactor *
                    (diffuse + specular);
    }
    return clamp(litColor + visual.u_emissiveColor.rgb, 0.0, 1.0);
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
    // Mirror the retained GL visual program: vertex alpha already carries the
    // material transparency for PER_FACE vertex colors (flagged on the
    // command); otherwise the uniform material opacity multiplies the vertex
    // alpha.
    float materialAlpha = pc.u_color.a;
    if (pc.u_flags.x > 0.5 && pc.u_flags.y > 0.5) {
        materialAlpha = 1.0;
    }

    vec3 rgb = visual.u_materialParams.w < 0.5
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
