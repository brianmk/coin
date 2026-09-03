// Shared helpers for the minimal visual OpenGL pipeline.

vec4 coin_visual_color(vec4 vertexColor, vec4 uniformColor,
                       float useVertexColor)
{
  return useVertexColor > 0.5 ? vertexColor : uniformColor;
}

vec4 coin_visual_texture(vec4 color, sampler2D textureSampler,
                         vec2 texcoord, float textureEnabled,
                         vec4 textureModulation)
{
  return textureEnabled > 0.5
    ? color * texture(textureSampler, texcoord) * textureModulation
    : color;
}
