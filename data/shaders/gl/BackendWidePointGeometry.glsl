#version 410 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 u_vpSize;
uniform float u_pointSize;

in vec4 vs_color[];

out vec4 v_color;
out vec2 v_uv;

void emitCorner(vec2 uv)
{
  vec4 center = gl_in[0].gl_Position;
  vec2 pixelOffset = (uv - vec2(0.5)) * u_pointSize;
  vec2 ndcOffset = 2.0 * pixelOffset / u_vpSize;
  gl_Position = center + vec4(ndcOffset * center.w, 0.0, 0.0);
  v_color = vs_color[0];
  v_uv = uv;
  EmitVertex();
}

void main()
{
  emitCorner(vec2(0.0, 1.0));
  emitCorner(vec2(0.0, 0.0));
  emitCorner(vec2(1.0, 1.0));
  emitCorner(vec2(1.0, 0.0));
  EndPrimitive();
}
