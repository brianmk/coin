// Helpers for screen-space image and text sampling.

vec4 coin_pixel_fetch(sampler2D textureSampler, ivec2 pixel)
{
  return texelFetch(textureSampler, pixel, 0);
}
