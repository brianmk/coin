// data/shaders/vulkan/rt/RTShadingCommon.glsl
// Shading math / environment IBL / BRDF helpers, included by PathTrace.glsl.
// Behavior-neutral module split: no #version here (textual #include).
// Depends on the globals declared in PathTrace.glsl (frame, matBuffer, ...).

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
// The environment is either an analytic sky (mode 0): a vertical gradient
// between the background bottom (horizon) and top (zenith) colors, plus a sun
// disk that falls off with a user-set power around the sun direction; or a
// camera-centered "room cove" (mode 1): a back-facing box the viewer sits
// inside with a colored floor, four walls and a ceiling, so the cubemap reads
// as a real scene (desk / table / white lab) whose reflections carry the room.
// In both cases radiance is a closed-form function of direction, so it can be
// evaluated for the primary-ray miss (background/reflections), as a diffuse
// irradiance term (ambient) and as a specular IBL term (glossy surfaces).
vec3 envSunDir()
{
    return normalize(frame.u_env.yzw);
}

// Surface color of the environment along a world direction, independent of
// the intensity/lighting scale.  Sky mode returns the vertical gradient;
// room mode returns the face of the camera-aligned cove box the ray leaves.
vec3 envSurfaceColor(vec3 dir)
{
    if (frame.u_envRoom.w > 0.5) {
        // Room cove: a box centered on the camera.  floorY/ceilY are
        // camera-relative offsets; the box spans +/-E on X and Z.  Trace a
        // ray from the box center (camera) along dir and take the nearest
        // back-facing plane the ray exits through.
        const float E = max(frame.u_envRoomScale.x, 0.1);
        const float floorY = frame.u_envRoomFloor.w; // camera-relative
        const float ceilY = frame.u_envRoomCeil.w;   // camera-relative
        vec3 n = normalize(dir);
        float t = 1e30;
        vec3 color = vec3(0.0);
        // X walls (+/-E).
        if (n.x > 1e-5) {
            float tx = E / n.x;
            if (tx < t) { t = tx; color = frame.u_envRoom.rgb; }
        }
        if (n.x < -1e-5) {
            float tx = -E / n.x;
            if (tx < t) { t = tx; color = frame.u_envRoom.rgb; }
        }
        // Z walls (+/-E).
        if (n.z > 1e-5) {
            float tz = E / n.z;
            if (tz < t) { t = tz; color = frame.u_envRoom.rgb; }
        }
        if (n.z < -1e-5) {
            float tz = -E / n.z;
            if (tz < t) { t = tz; color = frame.u_envRoom.rgb; }
        }
        // Floor / ceiling planes (ray from the box center, so t = off / n.y).
        if (n.y < -1e-5) {
            float ty = floorY / n.y;
            if (ty >= 0.0 && ty < t) { t = ty; color = frame.u_envRoomFloor.rgb; }
        }
        if (n.y > 1e-5) {
            float ty = ceilY / n.y;
            if (ty >= 0.0 && ty < t) { t = ty; color = frame.u_envRoomCeil.rgb; }
        }
        return color;
    }
    // Sky gradient: dir.y in [-1,1]; -1 = horizon color, +1 = zenith.
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(frame.u_bgBottom.rgb, frame.u_bgTop.rgb, t);
}

// Environment radiance along a world direction (surface color * sky/room
// intensity, plus the sun disk).
vec3 envRadiance(vec3 dir)
{
    vec3 n = normalize(dir);
    vec3 env = envSurfaceColor(n) * frame.u_env.x;
    float d = max(dot(n, envSunDir()), 0.0);
    vec3 sun = frame.u_envColor.rgb * pow(d, max(frame.u_envColor.w, 1.0));
    return env + sun * frame.u_env.x;
}

// Diffuse irradiance of the environment at a surface normal (ambient).
// Approximates the cosine-weighted hemisphere integral of envSurfaceColor by
// sampling at N fixed cosine-hemisphere directions, deterministic so the
// single-sample preview does not flicker frame to frame.
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
        irr += envSurfaceColor(dir);
    }
    return (irr / float(N)) * 3.14159265 * frame.u_env.x;
}

// Specular IBL at a reflection direction with a roughness blur estimate.
// Samples the environment (and sun) at the reflected direction; a higher
// roughness darkens and desaturates the surface term while the sun stays only
// for glossy reflections so it reads as a hot spot.
vec3 envSpecular(vec3 r, float roughness)
{
    vec3 rn = normalize(r);
    float rough = clamp(roughness, 0.0, 1.0);
    vec3 radiance = envRadiance(rn);
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
