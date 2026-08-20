// data/shaders/vulkan/rt/PresentVertex.glsl
// Fullscreen-triangle vertex shader for the ray-tracing present pass.
// No vertex inputs: gl_VertexIndex drives a unit triangle covering the
// whole viewport (same technique as the background pass).

#version 450

void main()
{
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
