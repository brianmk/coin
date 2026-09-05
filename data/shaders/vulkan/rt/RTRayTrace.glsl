// data/shaders/vulkan/rt/RTRayTrace.glsl
// Ray-query trace + surface-shading helpers (NEE/MIS/direct), included by PathTrace.glsl.
// Behavior-neutral module split: no #version here (textual #include).
// Depends on RTShadingCommon.glsl (pbrEval) and the PathTrace.glsl globals.

struct HitInfo {
    bool hit;
    float t;
    vec3 pos;
    vec3 normal;
    int materialIndex;
    int primitiveId;
};

// Closest-hit query against the TLAS.  All geometry is opaque triangles
// (translucency is handled in the path-tracing loop as a deterministic
// transmission term, not here as a stochastic hit test).
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

// Shadow-query transmittance at a point: the fraction of the light's radiance
// that survives the segment to the light after passing through any
// transparent surfaces.  An opaque (alpha >= 1) surface blocks the light
// entirely (returns 0).  A translucent surface (alpha < 1, FreeCAD
// Transparency) attenuates the ray by (1 - alpha) and lets the remainder
// continue so the ray still sees surfaces behind it; the returned transmittance
// is the product over every transparent surface crossed.  This makes a glass
// pane cast a proportionally lighter shadow instead of the fully-opaque
// silhouette produced by the old binary test.
//
// The ray is walked as a sequence of closest-hit queries (not a
// TerminateOnFirstHit query, which stops at the first triangle whatever its
// material): the nearest surface is found, its material alpha decides whether
// the ray stops (opaque) or is re-launched just past it (transparent), up to
// MAX_TRANSPARENT_HITS surfaces.  For the common opaque case the first query
// returns an opaque surface and the loop breaks after one iteration, so there
// is no steady-state cost over the previous binary shadow ray.
float shadowTransmittance(vec3 origin, vec3 dir, float tMax)
{
    const int MAX_TRANSPARENT_HITS = 8;
    float transmittance = 1.0;
    vec3 curOrigin = origin;
    float curMax = tMax;
    for (int i = 0; i < MAX_TRANSPARENT_HITS; ++i) {
        rayQueryEXT q;
        rayQueryInitializeEXT(q, tlas, gl_RayFlagsOpaqueEXT, 0xFF, curOrigin,
                              0.001, dir, curMax);
        while (rayQueryProceedEXT(q)) {
            // Walk to the closest committed triangle intersection.
        }
        if (rayQueryGetIntersectionTypeEXT(q, true) !=
            gl_RayQueryCommittedIntersectionTriangleEXT) {
            break;  // nothing between the point and the light here
        }
        RTMaterial mat = matBuffer.materials[
          rayQueryGetIntersectionInstanceCustomIndexEXT(q, true)];
        const float alpha = clamp(mat.diffuse.a, 0.0, 1.0);
        if (alpha >= 1.0) {
            return 0.0;   // opaque surface: fully blocks the light
        }
        transmittance *= (1.0 - alpha);
        if (transmittance < 1e-4) {
            return 0.0;   // negligible remainder
        }
        const float hitT = rayQueryGetIntersectionTEXT(q, true);
        const float moved = hitT + 0.001;
        curOrigin += dir * moved;
        curMax -= moved;
    }
    return transmittance;
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
        // Transmittance-weighted visibility: a transparent surface partially
        // occludes, so it contributes (1 - alpha) toward the visible fraction.
        visible += shadowTransmittance(P, dir, 1e30);
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
        float transm = shadowTransmittance(worldPos + worldN * 0.001, worldL,
                                           distToLight - 0.001);
        if (transm <= 1e-4) {
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
        lit += transm * mat.lightColor[i].rgb * attenuation * spotFactor *
               contribution;
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
    float transm = shadowTransmittance(worldPos + worldN * 0.001, L,
                                       dist - 0.001);
    if (transm <= 1e-4) {
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
    return transm * nt.color.rgb * contribution * cosL /
           max(dist * dist * pdf, 1e-8);
}
