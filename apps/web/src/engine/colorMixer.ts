/**
 * CPU side of the HSL mixer and colour grading: uniform packing for RETOUCH_FRAG and a
 * per-pixel JavaScript reference of the exact shader maths (used by tests, the colour
 * wheels' swatches, and anything that needs the graded colour without a GPU).
 *
 * Twin of services/ai/app/color_engine.py; keep all three in step.
 */
import {
  GRADE_BALANCE_SHIFT,
  GRADE_LUM,
  GRADE_PIVOTS,
  GRADE_TINT,
  GRADE_WIDTH_MIN,
  GRADE_WIDTH_RANGE,
  GRADE_ZONES,
  HSL_CENTERS,
  HSL_CHANNELS,
  HSL_HUE_RANGE,
  HSL_LUM_RANGE,
  HSL_NEUTRAL_EDGE,
  gradingIsDefault,
  hslIsDefault,
  type ColorGrading,
  type HslMixer,
} from '@fineretouch/shared';

export type RGB = [number, number, number];

const LUMA: RGB = [0.299, 0.587, 0.114];
const luma = (c: RGB) => c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const mod = (x: number, y: number) => x - y * Math.floor(x / y);
const CENTERS = [...HSL_CHANNELS.map((c) => HSL_CENTERS[c]), 360];

export function rgbToHsv(c: RGB): RGB {
  const [r, g, b] = c;
  const v = Math.max(r, g, b);
  const d = v - Math.min(r, g, b);
  const s = v > 1e-6 ? d / v : 0;
  const dd = Math.max(d, 1e-6);
  const h = v === r ? mod((g - b) / dd, 6) : v === g ? (b - r) / dd + 2 : (r - g) / dd + 4;
  return [d > 1e-6 ? h * 60 : 0, s, v];
}

export function hsvToRgb(h: number, s: number, v: number): RGB {
  const hh = mod(h, 360) / 60;
  const f = (n: number) => {
    const k = mod(n + hh, 6);
    return v * (1 - s * clamp01(Math.min(k, 4 - k)));
  };
  return [f(5), f(3), f(1)];
}

/** Lower channel index, upper channel index and smoothstepped blend for a hue in degrees. */
export function hueSegment(h: number): [number, number, number] {
  const hue = mod(h, 360);
  let lo = 0;
  for (let i = 1; i < 8; i++) if (hue >= CENTERS[i]) lo = i;
  let t = (hue - CENTERS[lo]) / (CENTERS[lo + 1] - CENTERS[lo]);
  t = t * t * (3 - 2 * t);
  return [lo, (lo + 1) % 8, t];
}

/** uHslK: 8 x (hue, saturation, luminance) in -1..1, in HSL_CHANNELS order. */
export function hslUniform(hsl: HslMixer): Float32Array {
  const out = new Float32Array(24);
  HSL_CHANNELS.forEach((ch, i) => {
    out[i * 3] = hsl[ch].hue / 100;
    out[i * 3 + 1] = hsl[ch].saturation / 100;
    out[i * 3 + 2] = hsl[ch].luminance / 100;
  });
  return out;
}

export function applyHslPixel(c: RGB, hsl: HslMixer): RGB {
  if (hslIsDefault(hsl)) return c;
  const k = hslUniform(hsl);
  const cc: RGB = [clamp01(c[0]), clamp01(c[1]), clamp01(c[2])];
  const [h, s, v] = rgbToHsv(cc);
  const [lo, hi, t] = hueSegment(h);
  const mixK = (j: number) => k[lo * 3 + j] * (1 - t) + k[hi * 3 + j] * t;
  const g = smoothstep(0, HSL_NEUTRAL_EDGE, s);
  const o = hsvToRgb(h + mixK(0) * HSL_HUE_RANGE * g, clamp01(s * (1 + mixK(1))), v);
  const y1 = luma(o);
  const keep = y1 > 1e-4 ? luma(cc) / Math.max(y1, 1e-4) : 1; // hue / saturation keep luma
  const f = keep * (1 + HSL_LUM_RANGE * mixK(2) * g * s);
  return [clamp01(o[0] * f), clamp01(o[1] * f), clamp01(o[2] * f)];
}

/** (shadow pivot, highlight pivot, crossover width) in luma. */
export function gradeBounds(g: ColorGrading): RGB {
  const b = g.balance / 100;
  return [GRADE_PIVOTS[0] - GRADE_BALANCE_SHIFT * b, GRADE_PIVOTS[1] - GRADE_BALANCE_SHIFT * b, GRADE_WIDTH_MIN + GRADE_WIDTH_RANGE * (g.blending / 100)];
}

/** RGB offset of one wheel: zero-luma chroma of its hue times saturation, plus luminance. */
export function wheelOffset(hue: number, saturation: number, luminance: number): RGB {
  const pure = hsvToRgb(hue, 1, 1);
  const y = luma(pure);
  const k = (saturation / 100) * GRADE_TINT;
  const l = (luminance / 100) * GRADE_LUM;
  return [(pure[0] - y) * k + l, (pure[1] - y) * k + l, (pure[2] - y) * k + l];
}

/** uGradeOff: shadows, midtones, highlights offsets (9 floats). */
export function gradeUniform(g: ColorGrading): Float32Array {
  const out = new Float32Array(9);
  GRADE_ZONES.forEach((z, i) => out.set(wheelOffset(g[z].hue, g[z].saturation, g[z].luminance), i * 3));
  return out;
}

export function zoneWeights(y: number, g: ColorGrading): RGB {
  const [lo, hi, width] = gradeBounds(g);
  const ws = 1 - smoothstep(lo - width / 2, lo + width / 2, y);
  const wh = smoothstep(hi - width / 2, hi + width / 2, y);
  return [ws, clamp01(1 - ws - wh), wh];
}

export function applyGradingPixel(c: RGB, g: ColorGrading): RGB {
  if (gradingIsDefault(g)) return c;
  const cc: RGB = [clamp01(c[0]), clamp01(c[1]), clamp01(c[2])];
  const w = zoneWeights(clamp01(luma(cc)), g);
  const off = gradeUniform(g);
  return cc.map((v, ch) => clamp01(v + w[0] * off[ch] + w[1] * off[3 + ch] + w[2] * off[6 + ch])) as RGB;
}
