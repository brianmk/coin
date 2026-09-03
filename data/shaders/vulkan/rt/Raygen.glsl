// data/shaders/vulkan/rt/Raygen.glsl
// Ray-generation stage of the retained ray-tracing backend.
//
// One invocation per pixel (the same contract as the previous compute
// tracer, but executed through VK_KHR_ray_tracing_pipeline with a five-
// group shader binding table: raygen, miss, shadow miss, closest hit and
// shadow closest hit).
//
//  - Preview mode (push flags bit 0 clear): one primary ray; the closest
//    hit shader shades it with the raster Gouraud model and the miss
//    shader fills the background gradient.
//
//  - Path tracing (bit 0 set): multi-bounce loop with next-event-estimation
//    direct lighting (shadow rays through the shadow hit/miss pair),
//    cosine hemisphere sampling for indirect light, perfect specular
//    reflection for shiny materials and Russian-roulette termination.
//    While the accumulate flag (bit 1) is set, one jittered sample per
//    frame is accumulated into the accumulation buffer; otherwise a single
//    sample is stored as a live preview.  The first bounce's world normal
//    and hit distance feed the denoising present pass.

#version 460
#extension GL_EXT_ray_tracing : require

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
    vec4  u_state;         // unused by the raygen (state rides in push constants)
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
    vec4  pbr;             // x = metalness, y = roughness, z = usePbr
};

layout(set = 0, binding = 3, std430) buffer Materials {
    RTMaterial materials[];
} matBuffer;

// Per-pixel accumulation buffer: rgb = radiance sum, a = sample count.
layout(set = 0, binding = 4, std430) buffer AccumBuffer { vec4 accum[]; };

// First-bounce G-buffers for the denoising present pass.
layout(set = 0, binding = 5, std430) buffer NormalBuffer { vec4 normals[]; };
layout(set = 0, binding = 6, std430) buffer PositionBuffer { vec4 positions[]; };

// Per-frame state for this stage (see SoRTXRenderBackend).
layout(push_constant) uniform RaygenPush {
    uint u_frameIndex;  // progressive sample index (jitter seed)
    uint u_flags;       // bit 0 = path tracing, bit 1 = accumulating,
                        // bit 2 = debug fill
    uint u_maxBounces;
    uint u_pad;
} pc;

// Ray payload shared by every stage of the pipeline.  All five shaders
// declare this struct verbatim: the closest hit fills the primary data,
// the miss writes color + hit flag, and the shadow pair touches only
// occluded.
struct Payload {
    vec4  color;   // chit: shaded/emissive, miss: bg
    vec4  normal;  // world normal
    vec4  posT;    // world position + hit distance
    uvec4 info;    // x = materialIndex, y = hit, z = specular, w = shading mode
    uint  occluded;
};
layout(location = 0) rayPayloadEXT Payload payload;

const int COIN_MAX_LIGHTS = 8;

// SBT record indices (see createShaderBindingTable()).
const uint SBT_HIT_PRIMARY = 0;
const uint SBT_HIT_SHADOW  = 1;
const uint SBT_MISS_PRIMARY = 0;
const uint SBT_MISS_SHADOW  = 1;

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

// Next-event-estimation direct lighting with shadow rays.  Matches the
// raster Gouraud convention: the producer's light data is used verbatim
// against eye-space normals, so the direct term looks like the raster
// viewport.  The producer stores lights in eye space (the light's node
// matrix is ModelMatrix * ViewingMatrix), so the shadow ray direction is
// converted back to world space before the TLAS trace.
vec3 coin_rtx_directLighting(vec3 worldPos, vec3 worldN, RTMaterial mat)
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
                                    att.x * distToLight * distToLight, 0.0001);
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

        // Shadow ray: the shadow pair writes only payload.occluded.  The
        // TLAS is world space, so convert the eye-space direction L back to
        // world space before tracing.
        payload.occluded = 0u;
        vec3 worldL = normalize((frame.u_viewInverse *
                                 vec4(normalize(L), 0.0)).xyz);
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, SBT_HIT_SHADOW, 0,
                    SBT_MISS_SHADOW, worldPos + worldN * 0.001, 0.001,
                    worldL, distToLight - 0.001, 0);
        if (payload.occluded != 0u) continue;

        vec3 H = normalize(normalize(L) + V);
        float NdotH = max(dot(N, H), 0.0);
        float shininess = max(mat.params.x * 128.0, 0.0);
        float specularFactor = shininess > 0.0 ? pow(NdotH, shininess) : 0.0;
        lit += mat.lightColor[i].rgb * attenuation * spotFactor *
               (mat.diffuse.rgb * NdotL + mat.specular.rgb * specularFactor);
    }
    return clamp(lit, 0.0, 1.0);
}

void main()
{
    uvec2 px = gl_LaunchIDEXT.xy;
    if (px.x >= uint(max(frame.u_viewport.x, 1.0)) ||
        px.y >= uint(max(frame.u_viewport.y, 1.0))) {
        return;
    }
    const uint frameIndex = pc.u_frameIndex;
    const bool ptEnabled = (pc.u_flags & 1u) != 0u;
    const bool accumulating = (pc.u_flags & 2u) != 0u;
    const bool debugFill = (pc.u_flags & 4u) != 0u;
    const int maxBounces = int(clamp(pc.u_maxBounces, 1u, 16u));
    const int index = int(px.y * uint(max(frame.u_viewport.x, 1.0)) + px.x);

    // Primary ray with per-frame sub-pixel jitter while accumulating.
    vec2 jitter = vec2(0.5);
    if (ptEnabled && accumulating) {
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
        origin = (frame.u_viewInverse * vec4(dirView.xy, 0.0, 1.0)).xyz;
        dir = normalize((frame.u_viewInverse * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    }
    else {
        origin = frame.u_cameraPos.xyz;
        dir = normalize((frame.u_viewInverse * vec4(dirView, 0.0)).xyz);
    }

    if (debugFill) {
        // Debug path: trace, then color by hit/miss.
        payload.info.y = 0u;
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, SBT_HIT_PRIMARY, 0,
                    SBT_MISS_PRIMARY, origin, 0.001, dir, 1e30, 0);
        imageStore(storageImage, ivec2(px),
                   payload.info.y != 0u ? vec4(0.9, 0.2, 0.1, 1.0)
                                       : vec4(0.1, 0.8, 0.3, 1.0));
        return;
    }

    if (!ptEnabled) {
        // Preview mode: single sample, shaded by the closest hit shader.
        payload.info.y = 0u;
        payload.info.w = 0u; // request Gouraud shading from the chit
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, SBT_HIT_PRIMARY, 0,
                    SBT_MISS_PRIMARY, origin, 0.001, dir, 1e30, 0);
        imageStore(storageImage, ivec2(px), clamp(payload.color, 0.0, 1.0));
        return;
    }

    // --- Path tracing loop -----------------------------------------------
    vec3 radiance = vec3(0.0);
    vec3 weight = vec3(1.0);
    vec3 rayOrigin = origin;
    vec3 rayDir = dir;

    for (int bounce = 0; bounce < maxBounces; ++bounce) {
        payload.info.y = 0u;
        payload.info.w = 1u; // path-tracing mode: chit returns hit data only
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, SBT_HIT_PRIMARY, 0,
                    SBT_MISS_PRIMARY, rayOrigin, 0.001, rayDir, 1e30, 0);

        if (payload.info.y == 0u) {
            // Miss: background gradient (payload.color from the miss shader)
            // contributes and the path ends.
            if (bounce == 0) {
                normals[index] = vec4(-rayDir, 1.0);
                positions[index] = vec4(0.0, 0.0, 0.0, 1.0e7);
            }
            radiance += weight * payload.color.rgb;
            break;
        }

        if (bounce == 0) {
            // G-buffer for the denoiser (visible surface only).
            normals[index] = payload.normal;
            positions[index] = payload.posT;
        }

        RTMaterial mat = matBuffer.materials[payload.info.x];

        // Emissive surfaces terminate the path.
        radiance += weight * payload.color.rgb; // chit returned the emissive
        if (dot(payload.color.rgb, payload.color.rgb) > 0.0) {
            break;
        }

        // Direct lighting (NEE with shadow rays).
        radiance += weight * coin_rtx_directLighting(payload.posT.xyz,
                                                     payload.normal.xyz, mat);

        vec3 n = normalize(payload.normal.xyz);
        vec3 albedo = mat.diffuse.rgb;

        vec3 newDir;
        // Legacy specular model: shininess (0..1) is the probability of a
        // mirror bounce; the rest follows the cosine diffuse lobe.  A binary
        // threshold here turned the default Coin shininess of 0.2 into a
        // perfect mirror for every material, whose image swept across
        // surfaces as the camera/object moved and looked like lighting
        // attached to the camera.
        float specProb = clamp(mat.params.x, 0.0, 1.0);
        vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                   : vec3(1.0, 0.0, 0.0);
        vec3 tangent = normalize(cross(up, n));
        vec3 bitangent = cross(n, tangent);
        if (hash2(px.x, px.y,
                  frameIndex + uint(bounce) * 431u + 9u).x < specProb) {
            newDir = reflect(rayDir, n);
            weight *= mix(albedo, mat.specular.rgb, 0.5) / max(specProb, 1e-4);
        }
        else {
            vec3 s = sampleCosine(hash2(px.x, px.y,
                                        frameIndex + uint(bounce) * 919u + 1u));
            newDir = normalize(tangent * s.x + bitangent * s.y + n * s.z);
            weight *= albedo / max(1.0 - specProb, 1e-4);
        }

        rayOrigin = payload.posT.xyz + n * 0.001;
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
    if (accumulating) {
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
