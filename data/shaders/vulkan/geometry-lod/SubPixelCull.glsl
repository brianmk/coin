// data/shaders/vulkan/geometry-lod/SubPixelCull.glsl
// GPU sub-pixel primitive culling (geometry LOD) for the raster Vulkan backend.
//
// One invocation per indexed TRIANGLE.  The triangle's three object-space
// positions are read through the same interleaved vertex buffer the draw uses,
// transformed with the same combined model*view*projection the visual vertex
// shader applies, projected to device pixels, and its screen-space area is
// compared against a threshold.  Triangles below the threshold are dropped;
// the survivors are compacted into a dense output index buffer and the draw is
// issued as vkCmdDrawIndexedIndirect, so culled primitives cost neither vertex
// shading nor rasterization.
//
// The pipeline runs OUTSIDE the render pass (Vulkan forbids compute inside
// one): the backend records it into the caller's command buffer before
// vkCmdBeginRenderPass and a memory barrier orders it against the draws.
//
// Layout notes:
//   - The interleaved vertex stride is 32 bytes (8 floats): position at
//     floats 0..2, normal 3..5, color 6, texcoord 7 (see VULKAN_VERTEX_STRIDE).
//   - u_offsets.x/y are the float/uint base offsets of the command's geometry
//     inside its (possibly shared) vertex/index buffers; the descriptors bind
//     the whole buffer, so the shader applies the offset itself and no
//     descriptor offset-alignment constraint applies.
//   - indirect.indexCount is the atomic append cursor AND the field
//     vkCmdDrawIndexedIndirect reads back; the CPU zeroes just that 4-byte
//     field per frame with vkCmdFillBuffer, leaving the other fields fixed.

#version 450

layout(local_size_x = 64) in;

layout(push_constant) uniform PushConstants {
    mat4 u_mvp;        // offset 0:  combined model*view*projection (row-major
                       //            SbMat packed as mat4 columns)
    vec4 u_params;     // offset 64: x = viewport width  (px),
                       //            y = viewport height (px),
                       //            z = minimum screen area (px^2 * 2),
                       //            w = primitive (triangle) count
    vec4 u_offsets;    // offset 80: x = vertex base (floats),
                       //            y = index base (uints),
                       //            z = 1 when the command is indexed,
                       //                0 for a non-indexed triangle list
} pc;

layout(set = 0, binding = 0) readonly buffer VertexData {
    float data[];
} verts;

layout(set = 0, binding = 1) readonly buffer IndexData {
    uint data[];
} idx;

layout(set = 0, binding = 2) writeonly buffer OutIndexData {
    uint data[];
} outIdx;

// The indirect command.  indexCount is the append cursor; the remaining fields
// are initialized once at buffer creation and never rewritten.
layout(set = 0, binding = 3) buffer IndirectData {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
} indirect;

const uint VERTEX_FLOATS = 8u;

vec3 loadPosition(const uint vertexIndex)
{
    const uint base = uint(pc.u_offsets.x) + vertexIndex * VERTEX_FLOATS;
    return vec3(verts.data[base], verts.data[base + 1u], verts.data[base + 2u]);
}

void main()
{
    const uint prim = gl_GlobalInvocationID.x;
    if (prim >= uint(pc.u_params.w)) return;

    // Indexed geometry reads the command's index buffer; a non-indexed
    // triangle list uses its vertices sequentially (vertex 3i, 3i+1, 3i+2).
    // Either way the survivors are written as indices, so the draw always uses
    // the compacted index buffer.
    const bool indexed = pc.u_offsets.z > 0.5;
    uint i0;
    uint i1;
    uint i2;
    if (indexed) {
        const uint ibase = uint(pc.u_offsets.y) + prim * 3u;
        i0 = idx.data[ibase];
        i1 = idx.data[ibase + 1u];
        i2 = idx.data[ibase + 2u];
    }
    else {
        i0 = prim * 3u;
        i1 = i0 + 1u;
        i2 = i0 + 2u;
    }

    const vec4 c0 = pc.u_mvp * vec4(loadPosition(i0), 1.0);
    const vec4 c1 = pc.u_mvp * vec4(loadPosition(i1), 1.0);
    const vec4 c2 = pc.u_mvp * vec4(loadPosition(i2), 1.0);

    // Keep any triangle that touches or crosses the near plane (w <= eps) or
    // is partly behind the camera: the rasterizer clips it correctly, and
    // culling on an undefined projection would pop it in and out.
    const float kNearEps = 1.0e-6;
    bool keep = false;
    if (c0.w <= kNearEps || c1.w <= kNearEps || c2.w <= kNearEps) {
        keep = true;
    }
    else {
        // NDC -> device pixels.  The exact origin (+1) cancels in the edge
        // differences, and the 0.5 makes the differences true pixel deltas, so
        // the cross product is twice the triangle's area in px^2.
        const vec2 vp = pc.u_params.xy * 0.5;
        const vec2 p0 = (c0.xy / c0.w) * vp;
        const vec2 p1 = (c1.xy / c1.w) * vp;
        const vec2 p2 = (c2.xy / c2.w) * vp;
        const float cross = abs((p1.x - p0.x) * (p2.y - p0.y) -
                                (p2.x - p0.x) * (p1.y - p0.y));
        keep = cross >= pc.u_params.z;
    }
    if (!keep) return;

    const uint slot = atomicAdd(indirect.indexCount, 3u);
    outIdx.data[slot] = i0;
    outIdx.data[slot + 1u] = i1;
    outIdx.data[slot + 2u] = i2;
}
