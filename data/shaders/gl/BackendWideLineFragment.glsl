#version 410 core

uniform float u_stipplePeriod;

in vec4 v_color;
in float v_lineDistance;

out vec4 fragColor;

void main()
{
  if (u_stipplePeriod > 0.0) {
    if (mod(v_lineDistance, u_stipplePeriod) > u_stipplePeriod * 0.5) discard;
  }

  fragColor = v_color;
}
