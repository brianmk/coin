#version 410 core

uniform sampler2D u_texture;
uniform vec4 u_texModColor;
uniform vec4 u_color;
uniform float u_vertexColorAlphaIncludesOpacity;
uniform float u_textureAlphaIncludesOpacity;
uniform int u_alphaTestFunction;
uniform float u_alphaTestReference;
uniform vec2 u_pixelOrigin;

in vec2 v_texcoord;
out vec4 fragColor;

#include "Common.glsl"
#include "../material/AlphaTest.glsl"

void main()
{
  ivec2 pixel = ivec2(floor(gl_FragCoord.xy - u_pixelOrigin));
  ivec2 size = textureSize(u_texture, 0);
  if (pixel.x < 0 || pixel.y < 0 || pixel.x >= size.x || pixel.y >= size.y) {
    discard;
  }
  vec4 texel = coin_pixel_fetch(u_texture, pixel);
  float materialAlpha = u_textureAlphaIncludesOpacity > 0.5
    ? 1.0 : u_color.a;
  vec4 color = vec4(texel.rgb * u_texModColor.rgb,
                    texel.a * materialAlpha);
  if (u_vertexColorAlphaIncludesOpacity > 0.5) color.a = texel.a;
  if (!coin_material_alpha_test_pass(color.a, u_alphaTestFunction,
                                     u_alphaTestReference)) discard;
  fragColor = color;
}
