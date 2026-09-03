#version 410 core

#include "Common.glsl"
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec2 a_texcoord;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
uniform vec4 u_color;
uniform float u_useVertexColor;
uniform int u_shadingModel;
uniform vec3 u_emissiveColor;
uniform vec3 u_ambientLight;
uniform vec3 u_materialAmbient;
uniform vec3 u_materialSpecular;
uniform float u_materialShininess;
uniform float u_twoSidedLighting;
uniform int u_lightCount;
uniform int u_lightType[8];
uniform vec3 u_lightColor[8];
uniform vec3 u_lightDirection[8];
uniform vec3 u_lightPosition[8];
uniform vec3 u_lightAttenuation[8];
uniform vec2 u_lightSpotParams[8];

#include "../material/Lighting.glsl"

out vec4 v_color;
out vec3 v_litColor;
out vec2 v_texcoord;

void main()
{
  v_color = coin_visual_color(a_color, vec4(u_color.rgb, 1.0),
                              u_useVertexColor);
  v_texcoord = a_texcoord;

  vec4 worldPos = u_model * vec4(a_position, 1.0);
  vec4 eyePos = u_view * worldPos;
  mat3 normalMatrix = transpose(inverse(mat3(u_view * u_model)));
  vec3 eyeNormal = normalMatrix * a_normal;
  v_litColor = u_shadingModel == 0
    ? v_color.rgb
    : coin_material_compute_gouraud_color(eyePos.xyz, eyeNormal,
                                          v_color.rgb);
  gl_Position = u_proj * eyePos;
}
