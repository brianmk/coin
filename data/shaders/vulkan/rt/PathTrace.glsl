// data/shaders/vulkan/rt/PathTrace.glsl
// Compute-based path tracer for the retained ray-tracing backend.
//
// One invocation per pixel.  Uses VK_KHR_ray_query (no ray tracing
// pipeline/SBT involved), which keeps the whole tracer inside a single
// compute shader:
//
//  - Preview mode (frame.u_state.y == 0): one primary ray, Gouraud-style
//    direct lighting, written straight to the storage image.
//
//  - Path tracing (frame.u_state.y == 1): multi-bounce loop with
//    next-event-estimation direct lighting (shadow ray queries), cosine
//    hemisphere sampling for indirect light, perfect specular reflection
//    for shiny materials and Russian-roulette termination.  While the
//    start flag state (frame.u_state.z) is set, one jittered sample per
//    frame is accumulated into the accumulation buffer; otherwise a single
//    sample is stored as a live preview.  The first bounce's world normal
//    and hit distance feed the denoising present pass.

#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D storageImage;

layout(set = 0, binding = 2, std140) uniform FrameBlock {
    mat4  u_view;          // world -> view
    mat4  u_viewInverse;   // view -> world
    mat4  u_projInverse;   // clip -> view
    vec4  u_cameraPos;     // world-space camera origin (viewInv[3])
    vec4  u_viewport;      // x = width, y = height, z = orthographic
    vec4  u_bgTop;         // gradient top color
    vec4  u_bgBottom;      // gradient bottom color
    vec4  u_state;         // x = frameIndex, y = pathTracing, z = accumulating, w = maxBounces
} frame;

// std430 mirror of the C++ RTMaterial record; one per draw command, indexed
// by the instance custom index (the draw-list command index).
struct RTMaterial {
    vec4  diffuse;
    vec4  ambient;
    vec4  specular;
    vec4  emissive;
    vec4  params;          // x = shininess, y = twoSided, z = lightCount,
                           // w = shadingModel (0 = unlit, 1 = gouraud)
    vec4  lightType[8];
    vec4  lightColor[8];
    vec4  lightDirection[8];
    vec4  lightPosition[8];
    vec4  lightAttenuation[8];
    vec4  lightSpot[8];
    vec4  triangleData;    // x = triangle-normal pool offset
    vec4  pbr;             // x = metalness, y = roughness, z = usePbr,
                           // w = unused
};

layout(set = 0, binding = 3, std430) buffer Materials {
    RTMaterial materials[];
} matBuffer;

// Object-space per-triangle geometric normals (one vec4 per triangle of the
// whole scene, indexed per command via RTMaterial::triangleData).  Computed
// on the CPU when the acceleration structures are built.
layout(set = 0, binding = 7, std430) readonly buffer NormalPool {
    vec4 triangleNormals[];
} normalPoolBuffer;

// Per-pixel accumulation buffer: rgb = radiance sum, a = sample count.
layout(set = 0, binding = 4, std430) buffer AccumBuffer { vec4 accum[]; };

// First-bounce G-buffers for the denoising present pass.
layout(set = 0, binding = 5, std430) buffer NormalBuffer { vec4 normals[]; };
layout(set = 0, binding = 6, std430) buffer PositionBuffer { vec4 positions[]; };

const int COIN_MAX_LIGHTS = 8;

// Small xorshift-style hash: pixel + seed -> two values in [0,1)^2.
vec2 hash2(uint px, uint py, uint seed)
{
    uint s = px * 747796405u + py * 2891336453u + seed * 2654435761u + 149857u;
    s = (s ^ (s >> 16)) * 2246822519u;
    s = (s ^ (s >> 13)) * 3266489917u;
    s ^= s >> 16;
    return vec2(float(s & 0xffffu), float((s >> 16) & 0xffffu)) * (1.0 / 65536.0);
}

// Cosine-weighted hemisphere sample around +Z.
vec3 sampleCosine(vec2 u)
{
    float a = sqrt(max(u.x, 0.0));
    float phi = 6.28318530718 * u.y;
    return vec3(a * cos(phi), a * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

// --- PBR (metallic-roughness) BRDF helpers --------------------------------
float pbrAlpha(RTMaterial mat)
{
    return clamp(mat.pbr.y * mat.pbr.y, 0.001, 1.0);
}

vec3 pbrF0(RTMaterial mat)
{
    return mix(vec3(0.04), mat.diffuse.rgb, clamp(mat.pbr.x, 0.0, 1.0));
}

float pbrD_GGX(float NdotH, float a)
{
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-7);
}

float pbrG_SchlickGGX(float NdotX, float a)
{
    float k = a * 0.5;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-7);
}

float pbrG_Smith(float NdotV, float NdotL, float a)
{
    return pbrG_SchlickGGX(NdotV, a) * pbrG_SchlickGGX(NdotL, a);
}

vec3 pbrF_Schlick(float VdotH, vec3 F0)
{
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// Full BRDF evaluation for unit N, V, L (NdotV > 0, NdotL >= 0).  The
// caller multiplies by NdotL and the incoming radiance.
vec3 pbrEval(vec3 N, vec3 V, vec3 L, RTMaterial mat)
{
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-6);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float a = pbrAlpha(mat);
    vec3 F = pbrF_Schlick(VdotH, pbrF0(mat));
    vec3 kd = (vec3(1.0) - F) * (1.0 - clamp(mat.pbr.x, 0.0, 1.0));
    return kd * mat.diffuse.rgb / 3.14159265 +
           F * pbrD_GGX(NdotH, a) * pbrG_Smith(NdotV, NdotL, a) /
             max(4.0 * NdotV * NdotL, 1e-6);
}

// GGX (VNDF) half-vector sample around +Z.
vec3 sampleGGX(vec2 u, float a)
{
    float phi = 6.28318530718 * u.y;
    float cosTheta =
      sqrt(max((1.0 - u.x) / (1.0 + (a * a - 1.0) * u.x), 0.0));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

struct HitInfo {
    bool hit;
    float t;
    vec3 pos;
    vec3 normal;
    int materialIndex;
};

// Closest-hit query against the TLAS.  All geometry is opaque triangles.
HitInfo traceClosest(vec3 origin, vec3 dir, float tMax)
{
    rayQueryEXT q;
    rayQueryInitializeEXT(q, tlas, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001,
                          dir, tMax);
    while (rayQueryProceedEXT(q)) {
        // Advance to the closest committed triangle intersection.
    }
    HitInfo h;
    h.hit = rayQueryGetIntersectionTypeEXT(q, true) ==
      gl_RayQueryCommittedIntersectionTriangleEXT;
    h.t = -1.0;
    h.pos = vec3(0.0);
    h.normal = vec3(0.0);
    h.materialIndex = 0;
    if (!h.hit) return h;

    h.t = rayQueryGetIntersectionTEXT(q, true);
    h.pos = origin + dir * h.t;
    h.materialIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(q, true);

    // Flat shading: the object-space face normal comes from the triangle-
    // normal pool (computed on the CPU at BLAS build time) and is
    // transformed to world space via the instance's object-to-world
    // transform.
    RTMaterial mat = matBuffer.materials[h.materialIndex];
    uint prim = rayQueryGetIntersectionPrimitiveIndexEXT(q, true);
    uint normalIndex = uint(mat.triangleData.x) + prim;
    vec3 objN = normalPoolBuffer.triangleNormals[normalIndex].xyz;
    if (dot(objN, objN) < 1e-12) {
        h.hit = false;
        return h;
    }
    mat4x3 objToWorld = rayQueryGetIntersectionObjectToWorldEXT(q, true);
    h.normal = normalize(mat3(transpose(inverse(mat3(objToWorld)))) * objN);
    // The pool normals follow the producer's triangle winding, whose
    // orientation is not guaranteed to face the ray; for closed solids the
    // outward normal always points toward the ray origin, so flip when
    // needed (this also covers two-sided materials).
    vec3 toRay = normalize(origin - h.pos);
    if (dot(h.normal, toRay) < 0.0) {
        h.normal = -h.normal;
    }
    return h;
}

// Shadow query: true when something blocks the segment to the light.
bool traceShadow(vec3 origin, vec3 dir, float tMax)
{
    rayQueryEXT q;
    rayQueryInitializeEXT(q, tlas, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.001,
                          dir, tMax);
    while (rayQueryProceedEXT(q)) {
        // First candidate terminates the traversal.
        return true;
    }
    return rayQueryGetIntersectionTypeEXT(q, true) ==
      gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Same Gouraud evaluation as the raster visual program and the v1 chit.
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
        if (NdotL <= 0.0) continue;
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

// Next-event-estimation direct lighting with shadow rays.  Matches the
// raster Gouraud convention: the producer's light data is used verbatim
// against eye-space normals (no view-space conversion), so the direct term
// looks like the raster viewport.  The producer stores lights in eye space
// (the light's node matrix is ModelMatrix * ViewingMatrix), so shadow ray
// directions are converted back to world space before the ray query.
vec3 coin_rtx_directLighting(vec3 worldPos, vec3 worldN, vec3 rayDir,
                             RTMaterial mat)
{
    vec3 eyePos = (frame.u_view * vec4(worldPos, 1.0)).xyz;
    vec3 N = normalize(mat3(frame.u_view) * worldN);
    vec3 V = normalize(-eyePos);
    vec3 lit = mat.ambient.rgb; // ambient light folded in by producer
    int lightCount = int(mat.params.z);
    for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
        if (i >= lightCount) break;

        vec3 L;
        float attenuation = 1.0;
        float spotFactor = 1.0;
        float distToLight = 1e30;
        if (mat.lightType[i].x > 0.5) {
            vec3 lightVector = mat.lightPosition[i].xyz - eyePos;
            distToLight = length(lightVector);
            if (distToLight <= 0.0001) continue;
            L = lightVector / distToLight;
            vec3 att = mat.lightAttenuation[i].xyz;
            attenuation = 1.0 / max(att.z + att.y * distToLight +
                                    att.x * distToLight * distToLight,
                                    0.0001);
            if (mat.lightType[i].x > 1.5) {
                vec3 coneDir = normalize(mat.lightDirection[i].xyz);
                vec3 fromLight = normalize(eyePos - mat.lightPosition[i].xyz);
                float spotCos = dot(coneDir, fromLight);
                if (spotCos < mat.lightSpot[i].x) continue;
                spotFactor = pow(max(spotCos, 0.0), mat.lightSpot[i].y);
            }
        }
        else {
            L = mat.lightDirection[i].xyz;
        }

        float NdotL = dot(N, normalize(L));
        if (NdotL <= 0.0) continue;

        // Shadow ray: the light data is in eye space (see above), so
        // convert the direction back to world space for the shadow query
        // against the world-space TLAS.
        vec3 worldL = normalize((frame.u_viewInverse *
                                 vec4(normalize(L), 0.0)).xyz);
        if (traceShadow(worldPos + worldN * 0.001, worldL,
                        distToLight - 0.001)) {
            continue;
        }

        vec3 contribution;
        if (mat.pbr.z > 0.5) {
            contribution = pbrEval(N, V, normalize(L), mat) * NdotL;
        }
        else {
            vec3 H = normalize(normalize(L) + V);
            float NdotH = max(dot(N, H), 0.0);
            float shininess = max(mat.params.x * 128.0, 0.0);
            float specularFactor = shininess > 0.0 ? pow(NdotH, shininess) : 0.0;
            contribution = mat.diffuse.rgb * NdotL +
                           mat.specular.rgb * specularFactor;
        }
        lit += mat.lightColor[i].rgb * attenuation * spotFactor * contribution;
    }
    return clamp(lit, 0.0, 1.0);
}

void main()
{
    uvec2 px = gl_GlobalInvocationID.xy;
    if (px.x >= uint(max(frame.u_viewport.x, 1.0)) ||
        px.y >= uint(max(frame.u_viewport.y, 1.0))) {
        return;
    }
    const uint frameIndex = uint(max(frame.u_state.x, 0.0));
    const float ptEnabled = frame.u_state.y;
    const float accumulating = frame.u_state.z;
    const int maxBounces = int(clamp(frame.u_state.w, 1.0, 16.0));
    const int index = int(px.y * uint(max(frame.u_viewport.x, 1.0)) + px.x);

    // Primary ray with per-frame sub-pixel jitter while accumulating.
    vec2 jitter = vec2(0.5);
    if (ptEnabled > 0.5 && accumulating > 0.5) {
        jitter = hash2(px.x, px.y, frameIndex);
    }
    vec2 uv = (vec2(px) + jitter) / max(frame.u_viewport.xy, vec2(1.0));
    vec2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    vec4 viewTarget = frame.u_projInverse * vec4(ndc, -1.0, 1.0);
    vec3 dirView = viewTarget.xyz / viewTarget.w;
    vec3 origin;
    vec3 dir;
    if (frame.u_viewport.z > 0.5) {
        // Orthographic camera (u_viewport.z == 1): all rays are parallel to
        // the view axis and the per-pixel ray origin sits on the view plane.
        // Fanning the directions out from the camera position (the
        // perspective-style code below) is what makes ortho views stretch
        // into infinity while orbiting.
        origin = (frame.u_viewInverse * vec4(dirView.xy, 0.0, 1.0)).xyz;
        dir = normalize((frame.u_viewInverse * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    }
    else {
        origin = frame.u_cameraPos.xyz;
        dir = normalize((frame.u_viewInverse * vec4(dirView, 0.0)).xyz);
    }

    if (frame.u_state.y > 2.5) {
        // Debug path (u_state.y == 3): trace, then write the payload.
        HitInfo h = traceClosest(origin, dir, 1e30);
        imageStore(storageImage, ivec2(px),
                   h.hit ? vec4(0.9, 0.2, 0.1, 1.0) : vec4(0.1, 0.8, 0.3, 1.0));
        return;
    }

    if (ptEnabled < 0.5) {
        // Preview mode: single sample, fully shaded.
        HitInfo h = traceClosest(origin, dir, 1e30);
        vec3 rgb = vec3(0.0);
        if (h.hit) {
            RTMaterial mat = matBuffer.materials[h.materialIndex];
            if (mat.pbr.z > 0.5) {
                rgb = coin_rtx_directLighting(h.pos, h.normal, dir, mat) +
                      mat.emissive.rgb;
            }
            else if (mat.params.w > 0.5) {
                vec3 eyePos = (frame.u_view * vec4(h.pos, 1.0)).xyz;
                vec3 eyeN = normalize(mat3(frame.u_view) * h.normal);
                rgb = coin_rtx_gouraud(eyePos, eyeN, mat.diffuse.rgb, mat);
            }
            else {
                rgb = mat.diffuse.rgb;
            }
        }
        else {
            float t = clamp(float(px.y) / max(frame.u_viewport.y, 1.0), 0.0, 1.0);
            rgb = mix(frame.u_bgTop.rgb, frame.u_bgBottom.rgb, t);
        }
        imageStore(storageImage, ivec2(px), clamp(vec4(rgb, 1.0), 0.0, 1.0));
        return;
    }

    // --- Path tracing loop -----------------------------------------------
    vec3 radiance = vec3(0.0);
    vec3 weight = vec3(1.0);
    vec3 rayOrigin = origin;
    vec3 rayDir = dir;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        HitInfo h = traceClosest(rayOrigin, rayDir, 1e30);

        if (!h.hit) {
            // Miss: background gradient contributes and the path ends.
            float t = clamp(float(px.y) / max(frame.u_viewport.y, 1.0), 0.0, 1.0);
            radiance += weight * mix(frame.u_bgTop.rgb, frame.u_bgBottom.rgb, t);
            if (bounce == 0) {
                normals[index] = vec4(-rayDir, 1.0);
                positions[index] = vec4(0.0, 0.0, 0.0, 1.0e7);
            }
            break;
        }

        if (bounce == 0) {
            // G-buffer for the denoiser (visible surface only).
            normals[index] = vec4(h.normal, 1.0);
            positions[index] = vec4(h.pos, h.t);
        }

        RTMaterial mat = matBuffer.materials[h.materialIndex];

        // Emissive surfaces terminate the path.
        radiance += weight * mat.emissive.rgb;
        if (dot(mat.emissive.rgb, mat.emissive.rgb) > 0.0) {
            break;
        }

        // Direct lighting (NEE with shadow rays).
        radiance += weight * coin_rtx_directLighting(h.pos, h.normal, rayDir, mat);

        vec3 n = normalize(h.normal);
        vec3 albedo = mat.diffuse.rgb;
        const float spec = mat.params.x > 0.0 ? 1.0 : 0.0;

        vec3 newDir;
        if (mat.pbr.z > 0.5) {
            // PBR: importance-sample both lobes (0.5/0.5 split).  The GGX
            // lobe weight is the standard F*G*VdotH/(NdotV*NdotH) form
            // (the D and the 4s cancel against the VNDF pdf).
            vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                       : vec3(1.0, 0.0, 0.0);
            vec3 tangent = normalize(cross(up, n));
            vec3 bitangent = cross(n, tangent);
            vec3 V = -rayDir;
            float NdotV = max(dot(n, V), 1e-6);
            float metallic = clamp(mat.pbr.x, 0.0, 1.0);
            float a = pbrAlpha(mat);
            vec3 F0 = pbrF0(mat);
            vec3 Fv = pbrF_Schlick(NdotV, F0);
            vec2 u2 = hash2(px.x, px.y,
                            frameIndex + uint(bounce) * 919u + 1u);
            if (u2.x < 0.5) {
                // Diffuse lobe (cosine-weighted).
                vec3 s = sampleCosine(hash2(px.x, px.y,
                                            frameIndex + uint(bounce) * 919u +
                                              3u));
                newDir = normalize(tangent * s.x + bitangent * s.y + n * s.z);
                float pdf = 0.5 * s.z / 3.14159265;
                vec3 kd = (vec3(1.0) - Fv) * (1.0 - metallic);
                weight *= kd * albedo * s.z / max(pdf, 1e-7);
            }
            else {
                // Specular lobe (GGX half-vector sample).
                vec3 Vt = vec3(dot(V, tangent), dot(V, bitangent), dot(V, n));
                vec3 Ht = sampleGGX(
                  hash2(px.x, px.y, frameIndex + uint(bounce) * 919u + 5u), a);
                vec3 H = normalize(tangent * Ht.x + bitangent * Ht.y +
                                   n * Ht.z);
                vec3 L = normalize(reflect(-V, H));
                float NdotL = max(dot(n, L), 0.0);
                float VdotH = max(dot(V, H), 0.0);
                float NdotH = max(dot(n, H), 0.0);
                if (NdotL <= 0.0) {
                    newDir = normalize(reflect(rayDir, n));
                }
                else {
                    newDir = L;
                }
                vec3 F = pbrF_Schlick(VdotH, F0);
                float G = pbrG_Smith(NdotV, NdotL, a);
                weight *= (F * G * VdotH) / max(NdotV * NdotH, 1e-6) / 0.5;
            }
        }
        else if (spec > 0.5) {
            // Perfect specular reflection.
            newDir = reflect(rayDir, n);
            weight *= mix(albedo, mat.specular.rgb, 0.5);
        }
        else {
            vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                       : vec3(1.0, 0.0, 0.0);
            vec3 tangent = normalize(cross(up, n));
            vec3 bitangent = cross(n, tangent);
            vec3 s = sampleCosine(hash2(px.x, px.y,
                                        frameIndex + uint(bounce) * 919u + 1u));
            newDir = normalize(tangent * s.x + bitangent * s.y + n * s.z);
            weight *= albedo;
        }

        rayOrigin = h.pos + n * 0.001;
        rayDir = normalize(newDir);

        // Russian roulette on the throughput.
        if (bounce > 0) {
            float p = clamp(max(max(weight.r, weight.g), weight.b), 0.1, 0.95);
            if (hash2(px.x, px.y, frameIndex + uint(bounce) * 503u + 7u).x > p) {
                break;
            }
            weight /= p;
        }
    }

    vec4 outColor = vec4(clamp(radiance, 0.0, 1.0), 1.0);
    if (accumulating > 0.5) {
        vec4 acc = accum[index];
        acc.rgb += outColor.rgb;
        acc.a += 1.0;
        accum[index] = acc;
    }
    else {
        accum[index] = outColor;
    }
    imageStore(storageImage, ivec2(px), outColor);
}
