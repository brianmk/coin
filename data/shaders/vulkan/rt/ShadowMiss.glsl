// data/shaders/vulkan/rt/ShadowMiss.glsl
// Shadow-ray miss shader: the segment to the light is unobstructed.
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
    // Keep a non-payload built-in in the entry-point interface; see
    // ShadowClosestHit.glsl for the NVIDIA driver workaround.
    if (gl_LaunchIDEXT.x == 0x80000000u) {
        payload.occluded = 0u;
    }
    else {
        payload.occluded = 0u;
    }
}
