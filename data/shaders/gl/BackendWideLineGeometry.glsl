#version 410 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 u_vpSize;
uniform float u_lineWidth;

in vec4 vs_color[];
in float vs_lineDistance[];

out vec4 v_color;
out float v_lineDistance;

void main()
{
  vec4 p0 = gl_in[0].gl_Position;
  vec4 p1 = gl_in[1].gl_Position;
  vec2 ndc0 = p0.xy / p0.w;
  vec2 ndc1 = p1.xy / p1.w;
  vec2 dir = normalize(ndc1 - ndc0);
  vec2 perp = vec2(-dir.y, dir.x);
  vec2 offset = perp * u_lineWidth / u_vpSize;

  v_color = vs_color[0];
  v_lineDistance = vs_lineDistance[0];
  gl_Position = p0 + vec4(offset * p0.w, 0.0, 0.0);
  EmitVertex();
  gl_Position = p0 - vec4(offset * p0.w, 0.0, 0.0);
  EmitVertex();

  v_color = vs_color[1];
  v_lineDistance = vs_lineDistance[1];
  gl_Position = p1 + vec4(offset * p1.w, 0.0, 0.0);
  EmitVertex();
  gl_Position = p1 - vec4(offset * p1.w, 0.0, 0.0);
  EmitVertex();

  EndPrimitive();
}
