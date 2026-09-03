// data/shaders/vulkan/rt/ShadowClosestHit.glsl
// Shadow-ray closest-hit shader: any opaque surface occludes the light.
//
// The payload declaration order must match Raygen.glsl exactly (glslang
// assigns ray payload locations by declaration order).

#version 460
#extension GL_EXT_ray_tracing : require

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
    // The reference to gl_InstanceCustomIndexEXT keeps a non-payload
    // built-in in the entry-point interface.  NVIDIA driver 610.x hangs
    // the GPU when a closest-hit entry point's interface contains only the
    // ray payload, so every hit shader must reference at least one other
    // interface variable.  The comparison is intentionally vacuous.
    if (gl_InstanceCustomIndexEXT == 0x80000000u) {
        payload.occluded = 1u;
    }
    else {
        payload.occluded = 1u;
    }
}
