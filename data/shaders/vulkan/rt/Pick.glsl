// data/shaders/vulkan/rt/Pick.glsl
//
// Single-ray GPU pick for the Vulkan/RTX viewport.  Traces one ray against the
// frame's TLAS (the same acceleration structures the path tracer uses) and
// writes the closest triangle hit to a host-readable buffer: hit distance,
// world-space position, the TLAS instance custom index (the draw-list command
// index, see SoRTXRenderBackend::buildTlas) and the primitive (triangle) id.
//
// The host maps the command index back to the originating SoShape
// (SoRenderCommand::userData) and the primitive id back to a sub-element -- for
// a SoBrepFaceSet, partIndex maps the triangle to its topological face.
//
// This is Vulkan/RTX only.  The GL renderer and the raster Vulkan backend do
// not compile or use this shader; they keep the CPU SoRayPickAction path.

#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

// std430-flat hit record (48 bytes): two vec4 plus one uvec4 so the C++ mirror
// needs no implicit padding guesswork.
struct PickHit {
    vec4  data0;  // x = t, y = hit flag, zw = unused
    vec4  worldPos;
    uvec4 ids;    // x = instance/command index, y = primitive id, zw = unused
};

layout(set = 0, binding = 1) buffer PickResult {
    PickHit hit;
} pickResult;

layout(push_constant) uniform PickPush {
    vec4  origin;     // xyz = world-space ray origin
    vec4  direction;  // xyz = world-space ray direction (normalized)
    float tMax;
} pc;

void main()
{
    rayQueryEXT q;
    rayQueryInitializeEXT(q, tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                          pc.origin.xyz, 0.001, pc.direction.xyz, pc.tMax);
    while (rayQueryProceedEXT(q)) {
        // Advance to the closest committed triangle intersection.
    }

    PickHit h;
    h.data0 = vec4(-1.0, 0.0, 0.0, 0.0);
    h.worldPos = vec4(0.0);
    h.ids = uvec4(0u);

    if (rayQueryGetIntersectionTypeEXT(q, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        const float t = rayQueryGetIntersectionTEXT(q, true);
        h.data0 = vec4(t, 1.0, 0.0, 0.0);
        h.worldPos = vec4(pc.origin.xyz + pc.direction.xyz * t, 0.0);
        h.ids.x = rayQueryGetIntersectionInstanceCustomIndexEXT(q, true);
        h.ids.y = rayQueryGetIntersectionPrimitiveIndexEXT(q, true);
    }
    pickResult.hit = h;
}
