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
    vec4  u_adaptive;      // x = minSamples, y = relErrorThreshold (0 = off)
    mat4  u_prevViewProj;  // world -> clip of the previous frame's camera
    vec4  u_temporal;      // x = reproject the history this frame
    vec4  u_nee;           // x = emissive-triangle count, y = NEE enabled,
                           // z = MIS balance enabled
    vec4  u_env;           // procedural IBL: x = intensity, yzw = sun dir (world)
    vec4  u_envColor;      // procedural IBL: rgb = sun color, w = sky brightness
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
    vec4  triangleData;    // x = triangle-normal pool offset, y = normal count,
                           // z = NEE pool offset, w = NEE entry count
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

// Per-pixel sums-of-squares for the adaptive-sampling variance test
// (rgb = sum(radiance^2), a = sample count, mirroring the accumulation
// buffer).  Cleared with the accumulation buffer on every fresh run.
layout(set = 0, binding = 8, std430) buffer SumSqBuffer { vec4 sq[]; };

// Per-frame active-pixel counter (pixels that actually traced this frame;
// converged pixels early-out and do not add).  Cleared every frame and read
// back by the host after the queue wait.  counts[1] counts the pixels that
// accepted reprojected history.
layout(set = 0, binding = 9, std430) buffer ActiveCounter {
    uint counts[];
};

// Temporal reprojection history: the previous traced frame's accumulation,
// sums-of-squares and world positions (the host swaps the live buffers into
// these slots after every traced frame).
layout(set = 0, binding = 10, std430) buffer AccumHistoryBuffer {
    vec4 accumHist[];
};
layout(set = 0, binding = 11, std430) buffer SumSqHistoryBuffer {
    vec4 sqHist[];
};
layout(set = 0, binding = 12, std430) buffer PositionHistoryBuffer {
    vec4 posHist[];
};

// Emissive-triangle pool for next-event estimation: one entry per triangle
// of the scene whose material emits.  Vertices are object space, with the
// command's model matrix baked in so the pool is valid without a per-frame
// BLAS rebuild; v0.w carries the triangle area.
struct NeeTriangle {
    vec4  v0;
    vec4  v1;
    vec4  v2;
    vec4  color;   // rgb = emission
    mat4  xform;   // object -> world (column-major)
};
layout(set = 0, binding = 13, std430) readonly buffer NeePool {
    NeeTriangle triangles[];
} neePool;

// First-bounce surface albedo G-buffer for the denoiser (rgb = albedo).
layout(set = 0, binding = 14, std430) buffer AlbedoBuffer { vec4 albedos[]; };

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

// Three values in [0,1)^3 for multi-dimensional sampling decisions.
vec3 hash3(uint px, uint py, uint seed)
{
    return vec3(hash2(px, py, seed), hash2(px, py, seed + 1u).x);
}

// Cosine-weighted hemisphere sample around +Z.
vec3 sampleCosine(vec2 u)
{
    float a = sqrt(max(u.x, 0.0));
    float phi = 6.28318530718 * u.y;
    return vec3(a * cos(phi), a * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

// --- Procedural environment / IBL ----------------------------------------
// The environment is an analytic sky: a vertical gradient between the
// background bottom (horizon) and top (zenith) colors, plus a sun disk that
// falls off with a user-set power around the sun direction.  This gives
// image-based lighting without an HDR cubemap: the radiance at a direction
// is a closed-form function, so it can be evaluated for the primary-ray miss
// (background/sky reflections), as a diffuse irradiance term (sky ambient)
// and as a specular IBL term (polished surfaces picking up the sun/sky).
vec3 envSkyColor(vec3 dir)
{
    // dir.y in [-1,1]; -1 = straight down = horizon color, +1 = zenith.
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(frame.u_bgBottom.rgb, frame.u_bgTop.rgb, t);
}

vec3 envSunDir()
{
    return normalize(frame.u_env.yzw);
}

// Environment radiance along a world direction.
vec3 envRadiance(vec3 dir)
{
    vec3 n = normalize(dir);
    vec3 sky = envSkyColor(n) * frame.u_env.x;
    float d = max(dot(n, envSunDir()), 0.0);
    vec3 sun = frame.u_envColor.rgb * pow(d, max(frame.u_envColor.w, 1.0));
    return sky + sun * frame.u_env.x;
}

// Diffuse irradiance of the environment at a surface normal (sky ambient).
// Approximates the cosine-weighted hemisphere integral of envSkyColor by
// sampling the sky at N fixed cosine-hemisphere directions, deterministic so
// the single-sample preview does not flicker frame to frame.
vec3 envIrradiance(vec3 n)
{
    vec3 up = (abs(n.y) < 0.999) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);
    const int N = 4;
    float phase = 0.0;
    vec3 irr = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        float fi = (float(i) + phase) * 2.39996; // golden-angle spiral
        float cosT = sqrt(max((float(i) + 0.5) / float(N), 0.0));
        float sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
        vec3 dir = tangent * (cos(fi) * sinT) + bitangent * (sin(fi) * sinT) +
                   n * cosT;
        irr += envSkyColor(dir);
    }
    return (irr / float(N)) * 3.14159265 * frame.u_env.x;
}

// Specular IBL at a reflection direction with a roughness blur estimate.
// Samples the sky (and sun) at the reflected direction; a higher roughness
// darkens and desaturates the sky term while the sun falls off through the
// reflection (a cheap prefilter-free approximation).
vec3 envSpecular(vec3 r, float roughness)
{
    vec3 rn = normalize(r);
    float rough = clamp(roughness, 0.0, 1.0);
    vec3 radiance = envRadiance(rn);
    // Sky term softens and fades with roughness; the sun stays only for
    // glossy reflections (low roughness) so it reads as a hot spot.
    float sunAlign = max(dot(rn, envSunDir()), 0.0);
    radiance *= mix(1.0, (0.3 + 0.7 * sunAlign), rough);
    return radiance;
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
    int primitiveId;
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
    h.primitiveId = rayQueryGetIntersectionPrimitiveIndexEXT(q, true);

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

// Ambient-occlusion visibility at a hit point: N rays over the cosine-
// weighted hemisphere distributed by the golden-angle spiral, returned as the
// fraction that reach the background unoccluded (1 = fully open).  Single-
// sample, no accumulation -- the seed only stabilises the spiral phase so the
// real-time preview does not flicker frame to frame.
float coin_rtx_ao(vec3 worldPos, vec3 worldN, vec2 seed)
{
    vec3 N = normalize(worldN);
    // Tangent basis around N.
    vec3 up = (abs(N.y) < 0.999) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    float phase = seed.x * 6.2831853;
    vec3 P = worldPos + N * 0.001;
    float visible = 0.0;
    const int N_SAMPLES = 8;
    for (int i = 0; i < N_SAMPLES; ++i) {
        float fi = float(i) + phase;
        float phi = fi * 2.39996;                 // golden angle (radians)
        float cosT = 1.0 - (float(i) + 0.5) / float(N_SAMPLES);
        float sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
        vec3 dir = T * (cos(phi) * sinT) + B * (sin(phi) * sinT) + N * cosT;
        if (!traceShadow(P, dir, 1e30)) {
            visible += 1.0;
        }
    }
    return visible / float(N_SAMPLES);
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

// Emissive-surface next-event estimation: sample one emissive triangle
// uniformly, shadow-ray it and evaluate the same shading models as the
// analytic-light path.  pdf = 1 / (triangleCount * area).
vec3 coin_rtx_neeEmissive(vec3 worldPos, vec3 worldN, RTMaterial mat,
                          vec3 sampleXi)
{
    const float N = frame.u_nee.x;
    if (N < 0.5) return vec3(0.0);
    int t = int(clamp(sampleXi.x * N, 0.0, N - 1.0));
    NeeTriangle nt = neePool.triangles[t];
    const float area = max(nt.v0.w, 1e-8);
    float b0 = sampleXi.y;
    float b1 = sampleXi.z;
    if (b0 + b1 > 1.0) { b0 = 1.0 - b0; b1 = 1.0 - b1; }
    vec3 pObj = nt.v0.xyz * (1.0 - b0 - b1) + nt.v1.xyz * b0 + nt.v2.xyz * b1;
    vec3 pWorld = (nt.xform * vec4(pObj, 1.0)).xyz;
    vec3 e1 = (nt.xform * vec4(nt.v1.xyz - nt.v0.xyz, 0.0)).xyz;
    vec3 e2 = (nt.xform * vec4(nt.v2.xyz - nt.v0.xyz, 0.0)).xyz;
    vec3 lightN = cross(e1, e2);
    float lightNLen = length(lightN);
    if (lightNLen < 1e-9) return vec3(0.0);
    lightN /= lightNLen;
    vec3 toLight = pWorld - worldPos;
    float dist = length(toLight);
    if (dist < 1e-4) return vec3(0.0);
    vec3 L = toLight / dist;
    float cosL = dot(lightN, -L);
    float cosS = dot(worldN, L);
    if (cosL <= 0.0 || cosS <= 0.0) return vec3(0.0);
    if (traceShadow(worldPos + worldN * 0.001, L, dist - 0.001)) {
        return vec3(0.0);
    }
    // Evaluate the shading model in eye space, matching
    // coin_rtx_directLighting so an emissive surface reads like an
    // analytic light of the same brightness.
    vec3 eyePos = (frame.u_view * vec4(worldPos, 1.0)).xyz;
    vec3 eyeN = normalize(mat3(frame.u_view) * worldN);
    vec3 eyeV = normalize(-eyePos);
    vec3 eyeL = normalize((frame.u_view * vec4(L, 0.0)).xyz);
    vec3 contribution;
    if (mat.pbr.z > 0.5) {
        contribution = pbrEval(eyeN, eyeV, eyeL, mat) * cosS;
    }
    else {
        vec3 H = normalize(eyeL + eyeV);
        float NdotH = max(dot(eyeN, H), 0.0);
        float shininess = max(mat.params.x * 128.0, 0.0);
        float specularFactor = shininess > 0.0 ? pow(NdotH, shininess) : 0.0;
        contribution = mat.diffuse.rgb * cosS +
                       mat.specular.rgb * specularFactor;
    }
    const float pdf = 1.0 / (N * area);
    return nt.color.rgb * contribution * cosL /
           max(dist * dist * pdf, 1e-8);
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

    // Adaptive sampling: once enough samples accumulated, a pixel whose
    // relative variance fell below the threshold is converged and skips
    // tracing entirely (its mean is shown as-is).  The early-out must come
    // before the ray setup so the entire path loop is skipped.  On
    // reprojection frames it is disabled: the buffer contents are stale
    // until the history rewrite below has run.
    if (ptEnabled > 0.5 && accumulating > 0.5 &&
        frame.u_adaptive.y > 0.0 && frame.u_temporal.x < 0.5 &&
        frame.u_adaptive.w > 0.5) {
        const uint nPrev = uint(max(accum[index].a, 0.0));
        const uint minSamples = uint(max(frame.u_adaptive.x, 1.0));
        if (nPrev >= minSamples) {
            vec3 mean = accum[index].rgb / float(nPrev);
            // Variance per component: E[x^2] - E[x]^2, where accum.rgb is the
            // running sum of samples and sq.rgb is the running sum of squared
            // samples.  The scalar relative-error test below compares the
            // magnitude of the per-component variance against the mean.
            vec3 v2 = max(sq[index].rgb / float(nPrev) - mean * mean,
                          vec3(0.0));
            float relVar = max(v2.x, max(v2.y, v2.z));
            float m2 = max(dot(mean, mean), 1e-10);
            if (relVar < (frame.u_adaptive.y * frame.u_adaptive.y) * m2) {
                imageStore(storageImage, ivec2(px),
                           clamp(vec4(mean, 1.0), 0.0, 1.0));
                return;
            }
        }
    }

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

    // --- Environment / IBL preview (u_state.y == 3) ----------------------
    // Single primary ray, one sample, no accumulation: a real-time preview
    // lit by the procedural environment (sky gradient + sun).  The surface
    // receives the sky as a diffuse irradiance term (replacing the constant
    // material ambient) plus a specular reflection of the sun/sky, and the
    // analytic lights still contribute so the view reads like a shaded CAD
    // view.  On miss the environment radiance is shown in the direction of
    // the ray, so the sky/sun is visible *through* the scene, giving the
    // polished-surface reflections something to sample.
    if (ptEnabled > 2.5 && ptEnabled < 3.5) {
        HitInfo h = traceClosest(origin, dir, 1e30);
        vec3 rgb;
        if (h.hit) {
            RTMaterial mat = matBuffer.materials[h.materialIndex];
            vec3 N = normalize(h.normal);
            vec3 V = -dir;
            vec3 diffuse = mat.diffuse.rgb;
            // Diffuse IBL: sky irradiance, softly tinting the surface toward
            // the sky at its normal.  PBR materials keep their metallic tint.
            vec3 kd = (mat.pbr.z > 0.5)
              ? (vec3(1.0) - pbrF0(mat)) * (1.0 - clamp(mat.pbr.x, 0.0, 1.0))
              : vec3(1.0);
            vec3 iblDiffuse = kd * diffuse * envIrradiance(N) / 3.14159265;
            // Specular IBL: reflect the view about the normal and sample the
            // environment (sun disk + sky).  Only glossy/mirror surfaces get
            // a strong response.
            float roughness = (mat.pbr.z > 0.5)
              ? clamp(mat.pbr.y, 0.0, 1.0) : 1.0;
            vec3 R = normalize(reflect(V, N));
            vec3 F0 = (mat.pbr.z > 0.5) ? pbrF0(mat)
                                        : mix(vec3(0.04), mat.specular.rgb, 0.5);
            float NdotV = max(dot(N, V), 0.0);
            vec3 Fres = F0 + (vec3(1.0) - F0) * pow(1.0 - NdotV, 5.0);
            vec3 iblSpec = Fres * envSpecular(R, roughness);
            rgb = iblDiffuse + iblSpec;
            // Analytic lights (direct, shadowed) + emissive on top, matching
            // the shading-model conventions used everywhere else.
            if (mat.pbr.z > 0.5) {
                rgb += coin_rtx_directLighting(h.pos, h.normal, dir, mat);
            }
            else if (mat.params.w > 0.5) {
                vec3 eyePos = (frame.u_view * vec4(h.pos, 1.0)).xyz;
                vec3 eyeN = normalize(mat3(frame.u_view) * h.normal);
                rgb += coin_rtx_gouraud(eyePos, eyeN, mat.diffuse.rgb, mat);
            }
            rgb += mat.emissive.rgb;
            normals[index] = vec4(h.normal, 1.0);
            positions[index] = vec4(h.pos, h.t);
            albedos[index] = vec4(mat.diffuse.rgb, 1.0);
        }
        else {
            rgb = envRadiance(dir);
        }
        imageStore(storageImage, ivec2(px), clamp(vec4(rgb, 1.0), 0.0, 1.0));
        // Mirror into the accumulation G-buffer so the edge-stopping present
        // path (which reads accum when the denoise toggle is up) shows the
        // environment image too instead of stale/empty accumulation data.
        accum[index] = vec4(clamp(rgb, 0.0, 1.0), 1.0);
        return;
    }

    // --- Ambient-occlusion preview (u_state.y == 2) ----------------------
    // Single primary ray + a hemisphere of occlusion rays, one sample, no
    // accumulation: a real-time AO preview that recomputes on every camera
    // move like the raster viewport.  Diffuse/ambient/emissive are darkened
    // by the visibility factor; direct lights are left as-is so the preview
    // reads like a shaded CAD view rather than a flat grey AO map.
    if (ptEnabled > 1.5 && ptEnabled < 2.5) {
        HitInfo h = traceClosest(origin, dir, 1e30);
        vec3 rgb;
        if (h.hit) {
            RTMaterial mat = matBuffer.materials[h.materialIndex];
            float ao = coin_rtx_ao(h.pos, h.normal, hash2(px.x, px.y, frameIndex));
            if (mat.pbr.z > 0.5) {
                rgb = (mat.diffuse.rgb * ao) +
                      coin_rtx_directLighting(h.pos, h.normal, dir, mat) +
                      mat.emissive.rgb;
            }
            else if (mat.params.w > 0.5) {
                vec3 eyePos = (frame.u_view * vec4(h.pos, 1.0)).xyz;
                vec3 eyeN = normalize(mat3(frame.u_view) * h.normal);
                rgb = (mat.diffuse.rgb * ao) +
                      coin_rtx_gouraud(eyePos, eyeN, mat.diffuse.rgb, mat) +
                      mat.emissive.rgb;
            }
            else {
                rgb = mat.diffuse.rgb * ao + mat.emissive.rgb;
            }
            normals[index] = vec4(h.normal, 1.0);
            positions[index] = vec4(h.pos, h.t);
            albedos[index] = vec4(mat.diffuse.rgb, 1.0);
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
    float lastPdf = 1.0; // pdf of the direction that brought us to this hit

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        HitInfo h = traceClosest(rayOrigin, rayDir, 1e30);

        if (!h.hit) {
            // Miss: background gradient contributes and the path ends.
            float t = clamp(float(px.y) / max(frame.u_viewport.y, 1.0), 0.0, 1.0);
            radiance += weight * mix(frame.u_bgTop.rgb, frame.u_bgBottom.rgb, t);
            if (bounce == 0) {
                normals[index] = vec4(-rayDir, 1.0);
                positions[index] = vec4(0.0, 0.0, 0.0, 1.0e7);
                albedos[index] = vec4(0.0);
            }
            break;
        }

        if (bounce == 0) {
            // G-buffer for the denoiser (visible surface only).
            normals[index] = vec4(h.normal, 1.0);
            positions[index] = vec4(h.pos, h.t);
        }

        RTMaterial mat = matBuffer.materials[h.materialIndex];

        if (bounce == 0) {
            albedos[index] = vec4(mat.diffuse.rgb, 1.0);
        }

        // Emissive surfaces terminate the path.  With NEE enabled the
        // emission arrives through the emissive-triangle sampling below, so
        // a BSDF hit here is double counting unless the balance heuristic
        // (MIS) splits the weight by the sampling pdfs.
        float emissionWeight = 1.0;
        if (bounce > 0 && frame.u_nee.y > 0.5 && frame.u_nee.x > 0.5) {
            if (frame.u_nee.z > 0.5) {
                int poolIdx = int(mat.triangleData.z) + h.primitiveId;
                if (poolIdx >= 0 && poolIdx < int(frame.u_nee.x)) {
                    float areaHit = max(neePool.triangles[poolIdx].v0.w, 1e-8);
                    float pNee = 1.0 / (frame.u_nee.x * areaHit);
                    float pBsdf = max(lastPdf, 1e-8);
                    emissionWeight =
                      (pBsdf * pBsdf) / (pBsdf * pBsdf + pNee * pNee);
                }
            }
            else {
                emissionWeight = 0.0;
            }
        }
        radiance += weight * mat.emissive.rgb * emissionWeight;
        if (dot(mat.emissive.rgb, mat.emissive.rgb) > 0.0) {
            break;
        }

        // Direct lighting (NEE with shadow rays) plus emissive-surface
        // sampling.  The emissive term also covers the primary ray: the
        // directly visible surface still receives area-light light.
        radiance += weight * coin_rtx_directLighting(h.pos, h.normal, rayDir, mat);
        if (frame.u_nee.y > 0.5) {
            radiance += weight * coin_rtx_neeEmissive(
              h.pos, h.normal, mat,
              hash3(px.x, px.y, frameIndex + uint(bounce) * 727u + 11u));
        }

        vec3 n = normalize(h.normal);
        vec3 albedo = mat.diffuse.rgb;

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
                lastPdf = pdf;
                vec3 kd = (vec3(1.0) - Fv) * (1.0 - metallic);
                // f_r = kd*albedo/pi for the cosine lobe; with the 0.5
                // lobe probability the weight is kd*albedo*cos/pdf =
                // 2*kd*albedo (the missing 1/pi made the diffuse bounce
                // pi times too bright).
                weight *= kd * albedo * s.z /
                           max(pdf * 3.14159265, 1e-7);
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
                lastPdf = 0.5 * pbrD_GGX(NdotH, a) * NdotH /
                          max(4.0 * VdotH, 1e-8);
                weight *= (F * G * VdotH) / max(NdotV * NdotH, 1e-6) / 0.5;
            }
        }
        else {
            // Legacy specular model: shininess (0..1) is the probability of
            // a mirror bounce; the rest follows the cosine diffuse lobe.
            // A binary threshold here turned the default Coin shininess of
            // 0.2 into a perfect mirror for every material, whose image
            // swept across surfaces as the camera/object moved and looked
            // like lighting attached to the camera.
            float specProb = clamp(mat.params.x, 0.0, 1.0);
            vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                       : vec3(1.0, 0.0, 0.0);
            vec3 tangent = normalize(cross(up, n));
            vec3 bitangent = cross(n, tangent);
            if (hash2(px.x, px.y,
                      frameIndex + uint(bounce) * 431u + 9u).x < specProb) {
                newDir = reflect(rayDir, n);
                lastPdf = max(specProb, 1e-4);
                weight *= mix(albedo, mat.specular.rgb, 0.5) /
                           max(specProb, 1e-4);
            }
            else {
                vec3 s = sampleCosine(hash2(px.x, px.y,
                                            frameIndex + uint(bounce) * 919u +
                                              1u));
                newDir = normalize(tangent * s.x + bitangent * s.y + n * s.z);
                lastPdf = s.z / 3.14159265;
                weight *= albedo / max(1.0 - specProb, 1e-4);
            }
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
        vec4 s = sq[index];
        // Firefly rejection (FC_VULKAN_PT_FIREFLY = standard-deviation
        // multiplier, 0 = off): before adding this sample, if it is many
        // sigma brighter than the pixel's running mean it is an outlier
        // (firefly) spike.  Replace it with the running mean so it does not
        // poison the average.  On flat surfaces which the denoiser leaves
        // grainy this removes the extreme high-variance draws that read as
        // isolated bright speckles, without clipping real highlights (only
        // points far from the mean are touched).
        float fireflySigma = frame.u_adaptive.z;
        if (fireflySigma > 0.0 && acc.a > 1.0) {
            vec3 mean = acc.rgb / acc.a;
            // Per-component variance: E[x^2] - E[x]^2 (s.rgb holds the sum of
            // squared samples).  Take the largest-component sigma so a firefly
            // on any channel is caught.
            vec3 v2 = max(s.rgb / acc.a - mean * mean, vec3(0.0));
            float sigma = sqrt(max(v2.x, max(v2.y, v2.z)));
            float sampleLum = max(max(outColor.r, outColor.g), outColor.b);
            float meanLum = max(max(mean.r, mean.g), mean.b);
            // Reject if the sample is > fireflySigma sigma above the mean.
            if (sigma > 1e-5 && sampleLum > meanLum + fireflySigma * sigma) {
                outColor = vec4(mean, 1.0);
            }
        }
        if (frame.u_temporal.x > 0.5) {
            // Temporal reprojection: carry the previous frame's history
            // forward where this pixel's world point was also visible to
            // the previous camera (mapped through u_prevViewProj).  A
            // position mismatch means disocclusion or a surface change,
            // so the pixel restarts from zero samples instead.
            acc = vec4(0.0);
            s = vec4(0.0);
            if (positions[index].w < 1.0e7) {
                vec4 oldClip = frame.u_prevViewProj *
                    vec4(positions[index].xyz, 1.0);
                if (oldClip.w > 0.0) {
                    vec2 oldNdc = oldClip.xy / oldClip.w;
                    vec2 oldPxF = vec2(oldNdc.x * 0.5 + 0.5,
                                       0.5 - oldNdc.y * 0.5) *
                        frame.u_viewport.xy;
                    const ivec2 vpSize = ivec2(frame.u_viewport.xy);
                    ivec2 oldPixel = ivec2(floor(oldPxF));
                    if (oldPixel.x >= 0 && oldPixel.y >= 0 &&
                        oldPixel.x < vpSize.x && oldPixel.y < vpSize.y) {
                        const int q = oldPixel.y * vpSize.x + oldPixel.x;
                        const vec3 delta =
                            posHist[q].xyz - positions[index].xyz;
                        if (posHist[q].w < 1.0e7 &&
                            dot(delta, delta) <
                            max(4.0e-4,
                                4.0e-6 * dot(positions[index].xyz,
                                              positions[index].xyz))) {
                            acc = accumHist[q];
                            s = sqHist[q];
                            atomicAdd(counts[1], 1u);
                        }
                    }
                }
            }
        }
        acc.rgb += outColor.rgb;
        acc.a += 1.0;
        accum[index] = acc;
        s.rgb += outColor.rgb * outColor.rgb;
        s.a += 1.0;
        sq[index] = s;
        atomicAdd(counts[0], 1u);
    }
    else {
        accum[index] = outColor;
    }
    imageStore(storageImage, ivec2(px), outColor);
}
