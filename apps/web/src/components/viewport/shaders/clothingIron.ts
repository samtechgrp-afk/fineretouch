/**
 * Clothing "virtual iron" and recolour (real-time preview of
 * retouch_engine.remove_clothing_wrinkles / shift_clothing_color).
 *
 * Like the export, it works on log-luminance only, so chromaticity, prints and dyed
 * patterns are untouched: the weave (finer than the s0 band) and the drape (coarser than
 * the s3 band) are kept, the crease and fold shading in between is attenuated, and
 * strong steps (seams, garment edges, bold prints) are protected. The export splits the
 * bands at face-scaled radii inside the garment mask; the preview reuses the four
 * precomputed long-edge bands. Recolour rotates YCbCr chroma here and CIELAB on export.
 */
export const IRON_GLSL = /* glsl */ `
float logY(vec3 c) { return log(max(dot(c, vec3(0.299, 0.587, 0.114)), 1e-3)); }
// Multiplicative luminance correction for one pixel. garment: mask weight; strength 0..1.
float ironRatio(vec3 low0, vec3 low1, vec3 low3, float garment, float strength) {
  float creases = logY(low0) - logY(low1);
  float folds = logY(low1) - logY(low3);
  float keep = clamp(smoothstep(0.35, 0.6, abs(creases)) + smoothstep(0.45, 0.8, abs(folds)), 0.0, 1.0);
  float k = strength * garment * (1.0 - keep);
  return exp(-(0.9 * creases + 0.45 * folds) * k);
}
vec3 recolour(vec3 c, float garment, float hueDeg, float sat) {
  if (garment <= 1e-3 || (hueDeg == 0.0 && sat == 0.0)) return c;
  float y = dot(c, vec3(0.299, 0.587, 0.114));
  vec2 ch = vec2(-0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b, 0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b);
  float t = radians(hueDeg);
  ch = mat2(cos(t), sin(t), -sin(t), cos(t)) * ch * max(0.0, 1.0 + sat);
  vec3 o = vec3(y + 1.402 * ch.y, y - 0.344136 * ch.x - 0.714136 * ch.y, y + 1.772 * ch.x);
  return mix(c, clamp(o, 0.0, 1.0), garment);
}
`;
