#version 410 core

layout(location = 0) in vec3 a_position;
layout(location = 3) in vec2 a_texcoord;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
uniform vec3 u_quadCenter;
uniform vec2 u_texSize;
uniform vec2 u_vpSize;
uniform vec2 u_pixelOrigin;

out vec2 v_texcoord;

void main()
{
  vec2 pixelPosition = u_pixelOrigin + a_texcoord * u_texSize;
  vec2 ndcPosition = 2.0 * pixelPosition / u_vpSize - 1.0;
  vec4 centerClip = u_proj * u_view * u_model * vec4(u_quadCenter, 1.0);
  gl_Position = vec4(ndcPosition * centerClip.w, centerClip.z, centerClip.w);
  v_texcoord = a_texcoord;
}
