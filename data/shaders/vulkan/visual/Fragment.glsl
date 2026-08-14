// data/shaders/vulkan/visual/Fragment.glsl
// Vulkan visual-pass fragment shader for the retained render backend.
//
// Milestone scope: unlit base color.  When per-vertex colors are present
// (u_flags.x > 0.5) they win over the uniform diffuse color.  Gouraud
// lighting and texturing are layered on in later milestones.

#version 450

layout(push_constant) uniform PushConstants {
    mat4  u_mvp;          // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes (x = useVertexColor)
} pc;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 fragColor;

void main()
{
    vec4 base = pc.u_flags.x > 0.5 ? v_color : pc.u_color;
    fragColor = base;
}
