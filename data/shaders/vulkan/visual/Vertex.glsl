// data/shaders/vulkan/visual/Vertex.glsl
// Vulkan visual-pass vertex shader for the retained render backend.
//
// The vertex buffer uses a fixed interleaved layout:
//   location 0: vec3 a_position
//   location 1: vec3 a_normal
//   location 2: vec4 a_color
//   location 3: vec2 a_texcoord
//
// A single push-constant block carries the projection matrix, the uniform
// diffuse color, and scalar feature flags.  The view/model matrices plus the
// per-draw lighting/material state live in a std140 uniform buffer (set 0,
// binding 0) whose layout mirrors SoGLRenderBackend's uniform set.  Lighting
// is evaluated per-vertex (Gouraud) in eye space, matching the retained GL
// visual program.

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

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec2 a_texcoord;

// Lighting is evaluated per fragment (see Fragment.glsl): Gouraud shading
// hardens the light gradient into linear bands between vertices, which on
// coarse CAD tessellations (FreeCAD's default angular deflection) reads as a
// sharp, faceted gradient and a hard-edged specular highlight.  Interpolated
// eye-space position/normal give a smooth gradient and a soft highlight.
layout(location = 0) out vec4 v_color;
layout(location = 1) out vec3 v_eyePos;
layout(location = 2) out vec3 v_eyeNormal;
layout(location = 3) out vec2 v_texcoord;

void main()
{
    vec4 worldPos = visual.u_model * vec4(a_position, 1.0);
    vec4 eyePos = visual.u_view * worldPos;
    mat3 normalMatrix = transpose(inverse(mat3(visual.u_view * visual.u_model)));
    vec3 eyeNormal = normalMatrix * a_normal;

    vec4 clip = pc.u_proj * eyePos;
    // Coin/OpenGL uses a bottom-left origin; Vulkan uses top-left.  Flip Y so
    // the two pipelines produce identical output for the same viewport.
    clip.y = -clip.y;
    // Coin's projection matrices are OpenGL-style: clip/NDC depth is in
    // [-1, 1].  Vulkan's depth range is [0, 1] and clips Z/w against it,
    // so remap the depth component.  W must stay untouched so the remap
    // survives perspective division: z_ndc = 0.5*(z_clip/w + 1)
    // => z_clip' = 0.5*(z_clip + w).
    clip.z = 0.5 * clip.z + 0.5 * clip.w;
    // Vulkan has no implicit point size: carry the retained
    // SoDrawStyle/SoPointSizeElement value in the push constants.  Applies
    // to point primitives and to the point-list overlay pipeline
    // (FC_VULKAN_POINTS); other topologies ignore the write.
    gl_PointSize = pc.u_pointSize;
    gl_Position = clip;

    v_color = pc.u_flags.x > 0.5 ? a_color : vec4(pc.u_color.rgb, 1.0);
    v_eyePos = eyePos.xyz;
    v_eyeNormal = eyeNormal;
    v_texcoord = a_texcoord;
}
