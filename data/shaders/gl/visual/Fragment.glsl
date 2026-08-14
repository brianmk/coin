#version 410 core

#include "Common.glsl"
#include "../material/Texture.glsl"
#include "../material/AlphaTest.glsl"

uniform sampler2D u_texture;
uniform float u_textureEnabled;
uniform int u_textureModel;
uniform vec4 u_textureBlendColor;
uniform vec4 u_color;
uniform float u_useVertexColor;
uniform float u_vertexColorAlphaIncludesOpacity;
uniform float u_textureAlphaIncludesOpacity;
uniform int u_alphaTestFunction;
uniform float u_alphaTestReference;

in vec4 v_color;
in vec3 v_litColor;
in vec2 v_texcoord;

out vec4 fragColor;

void main()
{
  float materialAlpha = u_color.a;
  if (u_useVertexColor > 0.5 &&
      u_vertexColorAlphaIncludesOpacity > 0.5) {
    materialAlpha = 1.0;
  }
  vec4 color;
  if (u_textureEnabled > 0.5) {
    color = coin_material_textured_color(u_texture, v_texcoord,
                                          vec4(v_litColor, v_color.a),
                                          materialAlpha,
                                          u_textureAlphaIncludesOpacity,
                                          u_textureModel,
                                          u_textureBlendColor);
  }
  else {
    color = vec4(v_litColor, v_color.a * materialAlpha);
  }
  if (!coin_material_alpha_test_pass(color.a, u_alphaTestFunction,
                                     u_alphaTestReference)) discard;
  fragColor = color;
}
