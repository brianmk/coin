// data/shaders/vulkan/rt/Miss.glsl
// Primary-ray miss shader: fills the payload with the background gradient
// and marks the ray as unhit.

#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 2, std140) uniform FrameBlock {
    mat4  u_view;
    mat4  u_viewInverse;
    mat4  u_projInverse;
    vec4  u_cameraPos;
    vec4  u_viewport;
    vec4  u_bgTop;
    vec4  u_bgBottom;
    vec4  u_state;
} frame;

// The payload struct must match Raygen.glsl verbatim (all stages share
// the same ray payload type at location 0).

struct Payload {
    vec4  color;
    vec4  normal;
    vec4  posT;
    uvec4 info;
    uint  occluded;
};
layout(location = 0) rayPayloadEXT Payload payload;

void main()
{
    float t = clamp(float(gl_LaunchIDEXT.y) / max(frame.u_viewport.y, 1.0),
                    0.0, 1.0);
    payload.color = mix(frame.u_bgTop, frame.u_bgBottom, t);
    payload.info.y = 0u;
}
