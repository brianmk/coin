// data/shaders/vulkan/visual/BackgroundFragment.glsl
// Background gradient pass.  Linear top-to-bottom interpolation between
// topColor and bottomColor in screen space (gl_FragCoord.y == 0 is the top
// edge in Vulkan).  The colors arrive via a small push-constant block so the
// backend can keep this pipeline free of descriptor sets.

#version 450

layout(push_constant) uniform BackgroundPush {
    vec4 topColor;       // offset 0, 16 bytes
    vec4 bottomColor;    // offset 16, 16 bytes
    vec4 viewport;       // offset 32, 16 bytes: x=width, y=height
} bg;

layout(location = 0) out vec4 fragColor;

void main()
{
    float h = max(bg.viewport.y, 1.0);
    float t = clamp(gl_FragCoord.y / h, 0.0, 1.0);
    fragColor = mix(bg.topColor, bg.bottomColor, t);
}
