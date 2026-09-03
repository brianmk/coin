// Gouraud lighting helpers for the retained material pipeline.

const int COIN_MAX_LIGHTS = 8;

vec3 coin_material_compute_gouraud_color(vec3 eyePos, vec3 eyeNormal,
                                         vec3 baseColor)
{
  vec3 N = normalize(eyeNormal);
  vec3 V = normalize(-eyePos);
  if (u_twoSidedLighting > 0.5 && dot(N, V) < 0.0) {
    N = -N;
  }
  vec3 litColor = u_ambientLight * u_materialAmbient;

  for (int i = 0; i < COIN_MAX_LIGHTS; ++i) {
    if (i >= u_lightCount) break;

    vec3 L = u_lightDirection[i];
    float attenuation = 1.0;
    float spotFactor = 1.0;
    if (u_lightType[i] != 0) {
      vec3 lightVector = u_lightPosition[i] - eyePos;
      float distanceToLight = length(lightVector);
      if (distanceToLight <= 0.0001) continue;
      L = lightVector / distanceToLight;
      vec3 att = u_lightAttenuation[i];
      attenuation = 1.0 / max(att.z + att.y * distanceToLight +
                               att.x * distanceToLight * distanceToLight,
                               0.0001);
      if (u_lightType[i] == 2) {
        vec3 coneDir = normalize(u_lightDirection[i]);
        vec3 fromLight = normalize(eyePos - u_lightPosition[i]);
        float spotCos = dot(coneDir, fromLight);
        if (spotCos < u_lightSpotParams[i].x) continue;
        spotFactor = pow(max(spotCos, 0.0), u_lightSpotParams[i].y);
      }
    }

    vec3 Ln = normalize(L);
    float NdotL = max(dot(N, Ln), 0.0);
    if (NdotL <= 0.0) continue;
    vec3 H = normalize(Ln + V);
    float NdotH = max(dot(N, H), 0.0);
    float shininess = max(u_materialShininess * 128.0, 0.0);
    float specularFactor = shininess > 0.0
      ? pow(NdotH, shininess) : 0.0;
    vec3 diffuse = baseColor * NdotL;
    vec3 specular = u_materialSpecular * specularFactor;
    litColor += u_lightColor[i] * attenuation * spotFactor *
                (diffuse + specular);
  }
  return clamp(litColor + u_emissiveColor, 0.0, 1.0);
}
