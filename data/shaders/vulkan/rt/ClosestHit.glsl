// data/shaders/vulkan/rt/ClosestHit.glsl
// Primary-ray closest-hit shader.
//
// Two modes, selected by the raygen through payloadInfo.w:
//
//  - Preview (w == 0): shade the surface with the same Gouraud model as
//    the raster visual program and return the color in payloadColor.
//
//  - Path tracing (w != 0): return the world normal, world position with
//    hit distance, material index and emissive color; the raygen performs
//    direct lighting and bounce sampling.

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

// Shared lighting container + Blinn-Phong evaluators (see LightCommon.glsl).
#include "../common/LightCommon.glsl"

layout(set = 0, binding = 2, std140) uniform FrameBlock {
    mat4  u_view;
    mat4  u_viewInverse;
    mat4  u_projInverse;
    vec4  u_cameraPos;
    vec4  u_viewport;
    vec4  u_bgTop;
    vec4  u_bgBottom;
    vec4  u_state;
} frame;

// std430 mirror of the C++ RTMaterial record; the light block is the shared
// CoinLightSet (byte-identical to the C++ RTMaterial light arrays).
struct RTMaterial {
    vec4  diffuse;
    vec4  ambient;
    vec4  specular;
    vec4  emissive;
    vec4  params;
    CoinLightSet lights;
    vec4  triangleData;
    vec4  pbr;             // x = metalness, y = roughness, z = usePbr
};

layout(set = 0, binding = 3, std430) buffer Materials {
    RTMaterial materials[];
} matBuffer;

// Object-space per-triangle geometric normals (one vec4 per triangle of the
// whole scene, indexed per command via RTMaterial::triangleData).
layout(set = 0, binding = 7, std430) readonly buffer NormalPool {
    vec4 triangleNormals[];
} normalPoolBuffer;

struct Payload {
    vec4  color;
    vec4  normal;
    vec4  posT;
    uvec4 info;
    uint  occluded;
};
layout(location = 0) rayPayloadEXT Payload payload;

// Same Gouraud evaluation as the raster visual program.  The producer's
// light data is world-space (the standard IR convention), so the evaluation
// runs directly in world space; dot-product shading makes this identical to
// the eye-space form.  The Blinn-Phong loop is the shared LightCommon helper
// (space-agnostic); this wrapper supplies the world-space vectors.
vec3 coin_rtx_gouraud(vec3 worldPos, vec3 worldNormal, vec3 baseColor,
                      RTMaterial mat)
{
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(frame.u_cameraPos.xyz - worldPos);
    if (mat.params.y > 0.5 && dot(N, V) < 0.0) {
        N = -N;
    }
    int lightCount = int(mat.params.z);
    float shininess = max(mat.params.x * 128.0, 0.0);
    return coinGouraudCls(mat.lights, lightCount, worldPos, N, V, baseColor,
                          mat.specular.rgb, shininess, mat.ambient.rgb,
                          mat.emissive.rgb);
}

void main()
{
    const uint materialIndex = gl_InstanceCustomIndexEXT;
    RTMaterial mat = matBuffer.materials[materialIndex];
    const uint prim = gl_PrimitiveID;
    // Smooth shading: barycentric-interpolate the three object-space vertex
    // normals.  The pool stores 6 vec4 per triangle (3 normals + 3 positions,
    // see appendTriangleNormals).  This toolchain exposes no portable
    // barycentric built-in, so recover the hit's barycentric coordinates by
    // solving for the weights of the object-space hit point against the
    // triangle's three stored vertices.
    const uint triBase = uint(mat.triangleData.x) + prim * 6u;
    const vec3 p0 = normalPoolBuffer.triangleNormals[triBase + 3u].xyz;
    const vec3 p1 = normalPoolBuffer.triangleNormals[triBase + 4u].xyz;
    const vec3 p2 = normalPoolBuffer.triangleNormals[triBase + 5u].xyz;
    const vec3 hitP =
      gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;
    const vec3 v0 = p1 - p0;
    const vec3 v1 = p2 - p0;
    const vec3 v2 = hitP - p0;
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    float w1;
    float w2;
    if (abs(denom) > 1e-12) {
      w1 = (d11 * d20 - d01 * d21) / denom;  // weight of vertex 1
      w2 = (d00 * d21 - d01 * d20) / denom;  // weight of vertex 2
    }
    else {
      w1 = w2 = 1.0 / 3.0;  // degenerate triangle: average the vertices
    }
    const float w0 = 1.0 - w1 - w2;
    const vec3 objN =
      normalPoolBuffer.triangleNormals[triBase + 0u].xyz * w0 +
      normalPoolBuffer.triangleNormals[triBase + 1u].xyz * w1 +
      normalPoolBuffer.triangleNormals[triBase + 2u].xyz * w2;
    if (dot(objN, objN) < 1e-12) {
        // Degenerate triangle: treat the ray as unhit.  The miss shader did
        // not run, so every payload field the raygen may read afterwards
        // must be written here or the previous trace's stale data leaks
        // into this one (garbage pixels).
        payload.color = vec4(0.0);
        payload.normal = vec4(0.0);
        payload.posT = vec4(0.0);
        payload.info = uvec4(materialIndex, 0u, 0u, payload.info.w);
        payload.occluded = 0u;
        return;
    }

    const vec3 worldPos =
        gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Flat shading: the object-space face normal from the pool is
    // transformed to world space by the instance's object-to-world matrix.
    mat4x3 objToWorld = gl_ObjectToWorldEXT;
    vec3 worldN = normalize(mat3(transpose(inverse(mat3(objToWorld)))) * objN);
    // The pool normals follow the producer's triangle winding, whose
    // orientation is not guaranteed to face the ray; for closed solids the
    // outward normal always points toward the ray origin, so flip when
    // needed (this also covers two-sided materials).
    vec3 toRay = normalize(gl_WorldRayOriginEXT - worldPos);
    if (dot(worldN, toRay) < 0.0) {
        worldN = -worldN;
    }

    payload.normal = vec4(worldN, 1.0);
    payload.posT = vec4(worldPos, gl_HitTEXT);
    payload.info = uvec4(materialIndex, 1u,
                         mat.params.x > 0.0 ? 1u : 0u, payload.info.w);

    if (payload.info.w == 0u) {
        // Preview mode: full Gouraud shading (matching the raster viewport
        // and the old compute tracer); world-space light data is consumed
        // directly.
        vec3 rgb = mat.diffuse.rgb;
        if (mat.params.w > 0.5) {
            rgb = coin_rtx_gouraud(worldPos, worldN, mat.diffuse.rgb, mat);
        }
        payload.color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
    }
    else {
        // Path tracing mode: return the emissive term; the raygen adds
        // direct light and samples the next bounce.
        payload.color = vec4(mat.emissive.rgb, 1.0);
    }
}
