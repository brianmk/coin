// data/shaders/vulkan/visual/BackgroundVertex.glsl
// Fullscreen triangle for the background gradient pass.  No vertex inputs or
// descriptor sets: gl_VertexIndex drives a unit triangle covering the whole
// viewport, and the fragment shader uses gl_FragCoord to compute the gradient.

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
