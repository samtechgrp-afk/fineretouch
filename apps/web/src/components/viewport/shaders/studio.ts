/**
 * Pro studio passes (v0.6), run after the main RETOUCH pass and before bokeh:
 *
 *   STUDIO   glasses glare -> clothing iron -> clothing recolour -> look LUT. Writes the
 *            image that is displayed and exported.
 *   OVERLAY  optional QA overlay (glare veil, texture retention, garment mask), composited
 *            by the display pass only, so it never reaches an export.
 *
 * Inputs: the RETOUCH output, the original and its blur bands (s0, s1, s3; see BLUR_SIGMAS),
 * the face mask (r = skin), the lens region (masks.ts buildLensRoiMask) and the
 * segmentation from the engine (r = clothes, g = hair, b = person; /v1/segment).
 */
import { GLARE_GLSL } from './glareReduction';
import { IRON_GLSL } from './clothingIron';
import { LUT_GLSL } from './lut3d';
import { TEXTURE_RETENTION_GLSL } from './textureRetention';

export type StudioOverlay = 'none' | 'glare' | 'texture' | 'clothing';
export const OVERLAY_MODES: Record<StudioOverlay, number> = { none: 0, glare: 1, texture: 2, clothing: 3 };

export const STUDIO_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uImg, uLow0, uLow1, uLow3, uLens, uSeg;
uniform sampler3D uLut;
uniform bool uGlare, uIron, uRecolour, uLutOn;
uniform float uGlareStrength, uIronStrength, uHue, uSatShift, uLutSize, uLutAmount;
${GLARE_GLSL}
${IRON_GLSL}
${LUT_GLSL}
void main() {
  vec3 c = texture(uImg, vUv).rgb;
  vec3 l0 = texture(uLow0, vUv).rgb, l1 = texture(uLow1, vUv).rgb, l3 = texture(uLow3, vUv).rgb;
  if (uGlare) c = removeGlare(c, glareVeil(l1, l3, texture(uLens, vUv).r, uGlareStrength), uGlareStrength);
  float garment = texture(uSeg, vUv).r;
  if (uIron && garment > 1e-3) c = clamp(c * ironRatio(l0, l1, l3, garment, uIronStrength), 0.0, 1.0);
  if (uRecolour) c = recolour(c, garment, uHue, uSatShift);
  if (uLutOn) c = applyLut(uLut, uLutSize, c, uLutAmount);
  outColor = vec4(c, 1.0);
}`;

export const OVERLAY_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc, uEdited, uLow1, uLow3, uLens, uSeg, uMask;
uniform int uMode;          // OVERLAY_MODES
uniform float uGlareStrength;
uniform vec2 uRing;         // texture-retention sampling radius in UV
${GLARE_GLSL}
${TEXTURE_RETENTION_GLSL}
void main() {
  vec4 o = vec4(0.0);
  if (uMode == 1) {
    float lens = texture(uLens, vUv).r;
    float a = glareVeil(texture(uLow1, vUv).rgb, texture(uLow3, vUv).rgb, lens, max(uGlareStrength, 0.5));
    o = vec4(0.2, 0.85, 1.0, clamp(a * 1.6, 0.0, 0.8));
    if (lens > 0.02 && lens < 0.98) o = vec4(1.0, 1.0, 1.0, 0.25);  // lens outline
  } else if (uMode == 2) {
    o = textureRetention(uSrc, uEdited, vUv, uRing, texture(uMask, vUv).r);
  } else if (uMode == 3) {
    o = vec4(0.65, 0.4, 1.0, texture(uSeg, vUv).r * 0.5);
  }
  outColor = o;
}`;
