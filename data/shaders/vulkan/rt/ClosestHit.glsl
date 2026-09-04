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

struct RTMaterial {
    vec4  diffuse;
    vec4  ambient;
    vec4  specular;
    vec4  emissive;
    vec4  params;
    vec4  lightType[8];
    vec4  lightColor[8];
    vec4  lightDirection[8];
    vec4  lightPosition[8];
    vec4  lightAttenuation[8];
    vec4  lightSpot[8];
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

const int COIN_MAX_LIGHTS = 8;

// Same Gouraud evaluation as the raster visual program.
vec3 coin_rtx_gouraud(vec3 eyePos, vec3 eyeNormal, vec3 baseColor,
                      RTMaterial mat)
{
    vec3 N = normalize(eyeNormal);
    vec3 V = normalize(-eyePos);
    if (mat.params.y > 0.5 && dot(N, V) < 0.0) {
        N = -N;
    }
    vec3 litColor = mat.ambient.rgb; // ambient light folded in by producer
    int lightCount = int(mat.params.z);
    for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
        if (i >= lightCount) break;

        vec3 L = mat.lightDirection[i].xyz;
        float attenuation = 1.0;
        float spotFactor = 1.0;
        if (mat.lightType[i].x > 0.5) {
            vec3 lightVector = mat.lightPosition[i].xyz - eyePos;
            float distanceToLight = length(lightVector);
            if (distanceToLight <= 0.0001) continue;
            L = lightVector / distanceToLight;
            vec3 att = mat.lightAttenuation[i].xyz;
            attenuation = 1.0 / max(att.z + att.y * distanceToLight +
                                    att.x * distanceToLight * distanceToLight,
                                    0.0001);
            if (mat.lightType[i].x > 1.5) {
                vec3 coneDir = normalize(mat.lightDirection[i].xyz);
                vec3 fromLight = normalize(eyePos - mat.lightPosition[i].xyz);
                float spotCos = dot(coneDir, fromLight);
                if (spotCos < mat.lightSpot[i].x) continue;
                spotFactor = pow(max(spotCos, 0.0), mat.lightSpot[i].y);
            }
        }

        vec3 Ln = normalize(L);
        float NdotL = max(dot(N, Ln), 0.0);
        vec3 H = normalize(Ln + V);
        float NdotH = max(dot(N, H), 0.0);
        float shininess = max(mat.params.x * 128.0, 0.0);
        float specularFactor = shininess > 0.0 ? pow(NdotH, shininess) : 0.0;
        vec3 diffuse = baseColor * NdotL;
        vec3 specular = mat.specular.rgb * specularFactor;
        litColor += mat.lightColor[i].rgb * attenuation * spotFactor *
                    (diffuse + specular);
    }
    return clamp(litColor + mat.emissive.rgb, 0.0, 1.0);
}

void main()
{
    const uint materialIndex = gl_InstanceCustomIndexEXT;
    RTMaterial mat = matBuffer.materials[materialIndex];
    const uint prim = gl_PrimitiveID;
    const uint normalIndex = uint(mat.triangleData.x) + prim;
    const vec3 objN = normalPoolBuffer.triangleNormals[normalIndex].xyz;
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
        // and the old compute tracer).
        vec3 eyePos = (frame.u_view * vec4(worldPos, 1.0)).xyz;
        vec3 eyeN = normalize(mat3(frame.u_view) * worldN);
        vec3 rgb = mat.diffuse.rgb;
        if (mat.params.w > 0.5) {
            rgb = coin_rtx_gouraud(eyePos, eyeN, mat.diffuse.rgb, mat);
        }
        payload.color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
    }
    else {
        // Path tracing mode: return the emissive term; the raygen adds
        // direct light and samples the next bounce.
        payload.color = vec4(mat.emissive.rgb, 1.0);
    }
}
