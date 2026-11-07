/**
 * Glasses glare reduction (real-time preview of retouch_engine.reduce_glasses_glare).
 *
 * The export inverts a reflection model, observed = scene * (1 - a) + a * G, with the
 * opacity `a` measured against the scene around each lens and blown-out cores inpainted.
 * The preview uses the same model with cheaper estimates: the scene's brightness is the
 * broadest precomputed blur band (s3, ~3% of the long edge) and the reflection's lift is
 * the s1 band above it, inside the lens region built from the eye landmarks
 * (masks.ts buildLensRoiMask). The reflection colour is taken as neutral white, and `a`
 * is capped at GLARE_PREVIEW_MAX_VEIL because the preview cannot inpaint cores.
 */
export const GLARE_PREVIEW_MAX_VEIL = 0.6;

export const GLARE_GLSL = /* glsl */ `
// lens: lens-region weight; strength: 0..1; low1 / low3: blur bands of the original.
float glareVeil(vec3 low1, vec3 low3, float lens, float strength) {
  float lift = dot(low1 - low3, vec3(0.299, 0.587, 0.114));
  float thr = 0.11 - 0.07 * strength;
  float bg = dot(low3, vec3(0.299, 0.587, 0.114));
  float a = clamp(lift / max(1.0 - bg, 0.05), 0.0, 1.0) * smoothstep(thr, thr + 0.06, lift);
  return min(a * lens, ${GLARE_PREVIEW_MAX_VEIL});
}
vec3 removeGlare(vec3 c, float a, float strength) {
  if (a <= 1e-3) return c;
  vec3 scene = clamp((c - vec3(a)) / (1.0 - a), 0.0, 1.0);
  return mix(c, scene, (0.5 + 0.5 * strength) * clamp(a * 4.0, 0.0, 1.0));
}
`;
