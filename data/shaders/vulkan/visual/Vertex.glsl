// data/shaders/vulkan/visual/Vertex.glsl
// Vulkan visual-pass vertex shader for the retained render backend.
//
// The vertex buffer uses a fixed interleaved layout:
//   location 0: vec3 a_position
//   location 1: vec3 a_normal
//   location 2: vec4 a_color
//   location 3: vec2 a_texcoord
//
// A single push-constant block carries the premultiplied model-view-projection
// matrix, the uniform diffuse color, and scalar feature flags.  This keeps the
// push-constant size under the guaranteed 128-byte minimum.

#version 450

layout(push_constant) uniform PushConstants {
    mat4  u_mvp;          // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes (x = useVertexColor)
} pc;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec2 a_texcoord;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec3 v_normal;

void main()
{
    vec4 clip = pc.u_mvp * vec4(a_position, 1.0);
    // Coin/OpenGL uses a bottom-left origin; Vulkan uses top-left.  Flip Y so
    // the two pipelines produce identical output for the same viewport.
    clip.y = -clip.y;
    gl_Position = clip;
    v_color = a_color;
    v_normal = a_normal;
}
