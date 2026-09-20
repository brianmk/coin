// data/shaders/vulkan/output/OutputVertex.glsl
// Fullscreen-triangle vertex stage for the HDR output pass.
//
// The output pass presents the linear HDR intermediate (an RGBA16F image the
// raster backend rendered the scene into) to the caller's swapchain framebuffer,
// applying the exposure + transfer-function encode.  No vertex buffer is bound:
// the three vertices are derived from gl_VertexIndex and cover the whole
// viewport with one oversized triangle (no diagonal seam, one fewer vertex than
// a quad).

#version 450

void main()
{
    // (0,0), (2,0), (0,2) in normalized device coordinates -> a triangle that
    // covers [-1,1]^2.  gl_VertexIndex is 0,1,2 for the three vertices.
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
