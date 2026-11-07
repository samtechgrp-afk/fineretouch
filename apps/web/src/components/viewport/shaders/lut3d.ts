/**
 * 3D look LUT lookup (preview of color_engine.apply_lut).
 *
 * The table is uploaded as an RGBA16F 3D texture with x = red, y = green, z = blue (the
 * .cube order, see @fineretouch/shared lut.ts) and hardware trilinear filtering. Texel
 * centres sit at (i + 0.5) / n, so input 0..1 is remapped onto them; that makes the
 * GPU lookup identical to the CPU trilinear interpolation of the export.
 */
export const LUT_GLSL = /* glsl */ `
vec3 applyLut(highp sampler3D lut, float n, vec3 c, float amount) {
  vec3 uvw = clamp(c, 0.0, 1.0) * ((n - 1.0) / n) + 0.5 / n;
  return mix(c, texture(lut, uvw).rgb, amount);
}
`;
