// data/shaders/vulkan/wide-line/WideLineInstancedVertex.glsl
// Vulkan wide-line vertex shader (GPU instanced) for the retained backend.
//
// Replaces the CPU quad expansion: one instance per line segment, six
// vertices per instance (two triangles).  The instance vertex buffer
// (binding 0, VK_VERTEX_INPUT_RATE_INSTANCE) carries the segment's two
// object-space endpoints and their vertex colors; the per-instance model
// matrix rides in binding 1, exactly as in the visual pipeline.  The
// expansion math mirrors SoVulkanRenderBackendWideLine.cpp's producer so the
// rasterized result is the same:
//
//   - transform both endpoints with u_proj * u_view * model
//   - apply the Coin Y-flip and the OpenGL->Vulkan depth remap
//   - near-clip the segment, interpolating the hidden endpoint onto z = 0
//   - offset each corner perpendicular to the segment in NDC, scaled by the
//     endpoint clip w so the width is perspective-correct
//
// The push-constant layout matches the visual pass (both pipelines share the
// layout); u_lineGeom is appended after lineParams and read only here.

#version 450

layout(push_constant) uniform PushConstants {
    mat4  u_proj;         // offset 0, 64 bytes
    vec4  u_color;        // offset 64, 16 bytes
    vec4  u_flags;        // offset 80, 16 bytes
    vec4  u_texParams;    // offset 96, 16 bytes
    vec4  u_texBlend;     // offset 112, 16 bytes
    float u_pointSize;    // offset 128, 16 bytes (pad[3])
    vec4  u_lineParams;   // offset 144, 16 bytes
    vec4  u_lineGeom;     // offset 160, 16 bytes: x = line width (device px),
                          // y = viewport width, z = viewport height,
                          // w = device pixel ratio
} pc;

// Per-draw view matrix (set 1, binding 0).  Only u_view is read; declaring the
// leading member is enough (the block is std140 and u_view sits at offset 0).
layout(set = 1, binding = 0, std140) uniform DrawBlock {
    mat4  u_view;
} draw;

// Instance-rate attributes: the segment endpoints/colors (binding 0) and the
// per-instance model matrix (binding 1, same locations as the visual pass).
layout(location = 0) in vec4 a_p0;
layout(location = 1) in vec4 a_p1;
layout(location = 2) in vec4 a_c0;
layout(location = 3) in vec4 a_c1;
layout(location = 4) in vec4 a_iModelRow0;
layout(location = 5) in vec4 a_iModelRow1;
layout(location = 6) in vec4 a_iModelRow2;
layout(location = 7) in vec4 a_iModelRow3;

layout(location = 0) out vec4 v_color;
// Declared for interface compatibility with WideLineFragment.glsl, which always
// reads location 1.  The instanced path is only used for non-stippled lines
// (stippled lines stay on the serial CPU-expansion path because their
// per-vertex distance is order-dependent), so the fragment's stipple branch
// (u_lineParams.x > 0) is never taken here.  Still write a meaningful
// per-segment screen distance so the varying is not undefined.
layout(location = 1) out float v_lineDistance;

// Same triangle order as the CPU producer: corners [0]=p0+off, [1]=p0-off,
// [2]=p1+off, [3]=p1-off.
const int kTriOrder[6] = int[](0, 1, 2, 2, 1, 3);

void main()
{
    const float kNearEps = 1.0e-5;
    v_lineDistance = 0.0;

    mat4 model = mat4(a_iModelRow0, a_iModelRow1, a_iModelRow2, a_iModelRow3);
    mat4 mvp = pc.u_proj * draw.u_view * model;

    vec4 c0 = mvp * vec4(a_p0.xyz, 1.0);
    vec4 c1 = mvp * vec4(a_p1.xyz, 1.0);
    // Coin/OpenGL bottom-left origin -> Vulkan top-left.
    c0.y = -c0.y;
    c1.y = -c1.y;
    // OpenGL [-1,1] depth -> Vulkan [0,1] (keep w for the perspective divide).
    c0.z = 0.5 * c0.z + 0.5 * c0.w;
    c1.z = 0.5 * c1.z + 0.5 * c1.w;

    // Near-clip, mirroring the producer: a hidden endpoint is moved onto the
    // plane z = 0 by interpolating with t = z0 / (z0 - z1).
    bool visible0 = (c0.w > kNearEps) && (c0.z >= 0.0);
    bool visible1 = (c1.w > kNearEps) && (c1.z >= 0.0);
    if (!visible0 && !visible1) {
        // Fully clipped: collapse to a degenerate primitive.
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        v_color = vec4(0.0);
        return;
    }
    float tA = 0.0;
    float tB = 1.0;
    if (!(visible0 && visible1)) {
        float denom = c0.z - c1.z;
        float tclip = (denom != 0.0) ? c0.z / denom : 0.0;
        tA = visible0 ? 0.0 : tclip;
        tB = visible1 ? 1.0 : tclip;
    }
    vec4 cA = mix(c0, c1, tA);
    vec4 cB = mix(c0, c1, tB);
    if (cA.w <= kNearEps || cB.w <= kNearEps) {
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        v_color = vec4(0.0);
        return;
    }

    vec2 ndc0 = cA.xy / cA.w;
    vec2 ndc1 = cB.xy / cB.w;
    vec2 d = ndc1 - ndc0;
    float len = length(d);
    if (len < 1.0e-8) {
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        v_color = vec4(0.0);
        return;
    }
    vec2 dir = d / len;
    // Half-width offset in NDC, one axis per viewport dimension (as produced).
    float offx = -dir.y * pc.u_lineGeom.x / max(pc.u_lineGeom.y, 1.0);
    float offy =  dir.x * pc.u_lineGeom.x / max(pc.u_lineGeom.z, 1.0);

    int corner = kTriOrder[gl_VertexIndex];
    bool endpoint1 = corner >= 2;
    float sign = ((corner & 1) == 0) ? 1.0 : -1.0;
    // Per-segment screen distance (pixels): 0 at the p0 corners, the segment
    // length at the p1 corners, linearly interpolated across the quad.
    v_lineDistance = endpoint1
        ? length(d * vec2(max(pc.u_lineGeom.y, 1.0),
                          max(pc.u_lineGeom.z, 1.0)) * 0.5)
        : 0.0;
    vec4 base = endpoint1 ? cB : cA;
    float w = base.w;
    // The offset is scaled by w so the perspective divide yields a constant
    // screen-space width.
    vec4 pos = base;
    pos.x += sign * offx * w;
    pos.y += sign * offy * w;
    gl_Position = pos;

    vec4 colA = mix(a_c0, a_c1, tA);
    vec4 colB = mix(a_c0, a_c1, tB);
    vec4 col = endpoint1 ? colB : colA;
    v_color = pc.u_flags.x > 0.5 ? col : vec4(pc.u_color.rgb, 1.0);
}
