// data/shaders/vulkan/wide-line/WideLineVertex.glsl
// Vulkan wide-line vertex shader for the retained render backend.
//
// The wide-line pipeline expands line segments into quads on the CPU (no
// geometry shader): a_position carries the CLIP-SPACE quad corner (xyz with
// the original clip w in the fourth component), with the Coin Y-flip and the
// OpenGL->Vulkan depth remap already applied by the producer.  Each vertex
// also carries the accumulated distance along the polyline in object units
// for screen-space stippling.  The push-constant layout matches the visual
// pass so both pipelines share one layout; only the fields below are read.

#version 450

layout(push_constant) uniform PushConstants {
    mat4  u_proj;         // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes
    vec4  u_texParams;    // offset 96, 16 bytes
    vec4  u_texBlend;     // offset 112, 16 bytes
    float u_pointSize;    // offset 128, 16 bytes (pad[3])
    vec4  u_lineParams;   // offset 144, 16 bytes: x = stipple factor (px/bit),
                        // y = 16-bit stipple pattern
} pc;

layout(location = 0) in vec4 a_position;
layout(location = 2) in vec4 a_color;
layout(location = 4) in float a_lineDistance;

layout(location = 0) out vec4 v_color;
layout(location = 1) out float v_lineDistance;

void main()
{
    gl_Position = a_position;
    v_color = pc.u_flags.x > 0.5 ? a_color : vec4(pc.u_color.rgb, 1.0);
    v_lineDistance = a_lineDistance;
}
