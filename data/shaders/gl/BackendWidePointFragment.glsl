#version 410 core

in vec4 v_color;
in vec2 v_uv;

uniform float u_pointSize;
uniform float u_roundPoints;

out vec4 fragColor;

void main()
{
  if (u_roundPoints > 0.5 && u_pointSize > 2.5) {
    vec2 delta = v_uv - vec2(0.5);
    if (dot(delta, delta) > 0.25) discard;
  }
  fragColor = v_color;
}
