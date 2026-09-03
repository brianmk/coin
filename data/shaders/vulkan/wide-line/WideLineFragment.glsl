// data/shaders/vulkan/wide-line/WideLineFragment.glsl
// Vulkan wide-line fragment shader for the retained render backend.
//
// Line stipple mirroring classic GL (glLineStipple): each bit of the
// 16-bit pattern covers linePatternScaleFactor PIXELS in screen space.
// The vertex attribute carries the polyline distance in window pixels;
// the fragment selects bit floor(distance / factor) % 16 and discards
// pixels whose bit is not set.

#version 450

layout(push_constant) uniform PushConstants {
    mat4  u_proj;         // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes
    vec4  u_texParams;    // offset 96, 16 bytes
    vec4  u_texBlend;     // offset 112, 16 bytes
    float u_pointSize;    // offset 128, 16 bytes (pad[3])
    vec4  u_lineParams;   // offset 144, 16 bytes: x = stipple factor (px/bit),
                          // y = 16-bit pattern as floatBitsToUint
} pc;

layout(location = 0) in vec4 v_color;
layout(location = 1) in float v_lineDistance;

layout(location = 0) out vec4 fragColor;

void main()
{
    if (pc.u_lineParams.x > 0.0) {
        uint pattern = floatBitsToUint(pc.u_lineParams.y);
        float bitSize = max(pc.u_lineParams.x, 1.0);
        float bitIndex = floor(v_lineDistance / bitSize);
        uint bit = uint(mod(bitIndex, 16.0));
        if ((pattern & (1u << bit)) == 0u) {
            discard;
        }
    }
    fragColor = v_color;
}
