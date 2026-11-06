import type { ColorStats } from '@fineretouch/shared';

/** TypeScript port of services/ai/app/color.py (sRGB D65 -> CIE L*a*b*). */

const EPS = 216 / 24389;
const KAPPA = 24389 / 27;
const WHITE = [0.95047, 1.0, 1.08883];

const LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  LIN[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function rgb8ToLab(r: number, g: number, b: number): [number, number, number] {
  const lr = LIN[r];
  const lg = LIN[g];
  const lb = LIN[b];
  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / WHITE[0];
  const y = (0.2126729 * lr + 0.7151522 * lg + 0.072175 * lb) / WHITE[1];
  const z = (0.0193339 * lr + 0.119192 * lg + 0.9503041 * lb) / WHITE[2];
  const f = (t: number) => (t > EPS ? Math.cbrt(t) : (KAPPA * t + 16) / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** Mean/std of L*a*b* over RGBA pixels, subsampled to at most `maxSamples`. */
export function colorStatsFromRgba(px: Uint8ClampedArray | Uint8Array, maxSamples = 262_144): ColorStats {
  const count = px.length / 4;
  const step = Math.max(1, Math.ceil(count / maxSamples));
  const sum = [0, 0, 0];
  const sq = [0, 0, 0];
  let n = 0;
  for (let i = 0; i < count; i += step) {
    const lab = rgb8ToLab(px[i * 4], px[i * 4 + 1], px[i * 4 + 2]);
    for (let k = 0; k < 3; k++) {
      sum[k] += lab[k];
      sq[k] += lab[k] * lab[k];
    }
    n++;
  }
  const mean = sum.map((s) => s / n) as [number, number, number];
  const std = sq.map((s, k) => Math.max(Math.sqrt(Math.max(s / n - mean[k] ** 2, 0)), 1e-3)) as [number, number, number];
  return { mean, std };
}
