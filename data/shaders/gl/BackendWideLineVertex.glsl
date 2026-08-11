#version 410 core
layout(location = 0) in vec3 a_position;
layout(location = 2) in vec4 a_color;
layout(location = 4) in float a_lineDistance;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
uniform vec4 u_color;
uniform float u_useVertexColor;

out vec4 vs_color;
out float vs_lineDistance;

void main()
{
  gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
  vs_color = (u_useVertexColor > 0.5) ? a_color : u_color;
  vs_lineDistance = a_lineDistance;
}
