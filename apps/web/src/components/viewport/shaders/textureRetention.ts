/**
 * Texture-retention overlay: shows how much pore and fine-detail energy the skin keeps
 * after retouching, as a QA view against "plastic" skin.
 *
 * Detail energy is the mean absolute difference between a pixel and an 8-tap ring around
 * it at pore scale (the detail layer natural_skin_retouch re-injects), measured the same
 * way on the original and on the retouched result. The ratio is painted on skin: green
 * where at least 85% of the original texture survived, amber where more than 15% was lost,
 * red where less than 60% is left and the skin has gone flat.
 */
export const TEXTURE_RETENTION_GLSL = /* glsl */ `
float detailEnergy(sampler2D t, vec2 uv, vec2 r) {
  vec3 c = texture(t, uv).rgb;
  float e = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.7853982;
    e += length(c - texture(t, uv + r * vec2(cos(a), sin(a))).rgb);
  }
  return e / 8.0;
}
// Overlay colour (rgb) and opacity (a) for one pixel; skin: skin weight.
vec4 textureRetention(sampler2D orig, sampler2D edited, vec2 uv, vec2 r, float skin) {
  if (skin <= 0.05) return vec4(0.0);
  float e0 = detailEnergy(orig, uv, r);
  if (e0 < 0.006) return vec4(0.0);
  float kept = clamp(detailEnergy(edited, uv, r) / e0, 0.0, 1.0);
  vec3 col = kept > 0.85 ? vec3(0.25, 0.85, 0.45) : (kept > 0.6 ? vec3(0.95, 0.7, 0.2) : vec3(0.95, 0.25, 0.3));
  return vec4(col, 0.5 * skin);
}
`;
