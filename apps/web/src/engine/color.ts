import { TONE_QUANTILE_COUNT, type ColorStats } from '@fineretouch/shared';

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

const CLIP_LOW = 2;
const CLIP_HIGH = 98;
const MIN_CONTENT_SHARE = 0.05;
/** Largest per-knot tone-curve shift in L* (MAX_CURVE_SHIFT in color.py). */
export const MAX_CURVE_SHIFT = 25;

/** Mean/std/covariance of L*a*b* and L* quantiles over RGBA pixels, subsampled to at most `maxSamples`. */
export function colorStatsFromRgba(px: Uint8ClampedArray | Uint8Array, maxSamples = 262_144): ColorStats {
  const count = px.length / 4;
  const step = Math.max(1, Math.ceil(count / maxSamples));
  const sum = [0, 0, 0];
  const cross = new Float64Array(9);
  // L* histogram at 0.05 resolution for quantiles (exact enough for a 17-knot tone curve).
  const BINS = 2001;
  const hist = new Uint32Array(BINS);
  let n = 0;
  // Ignore clipped pixels (studio white, crushed black) unless they are nearly everything,
  // exactly as color_stats in services/ai/app/color.py.
  let content = 0;
  let total = 0;
  for (let i = 0; i < count; i += step) {
    const L = rgb8ToLab(px[i * 4], px[i * 4 + 1], px[i * 4 + 2])[0];
    total++;
    if (L > CLIP_LOW && L < CLIP_HIGH) content++;
  }
  const skipClipped = content >= MIN_CONTENT_SHARE * total;
  for (let i = 0; i < count; i += step) {
    const lab = rgb8ToLab(px[i * 4], px[i * 4 + 1], px[i * 4 + 2]);
    if (skipClipped && !(lab[0] > CLIP_LOW && lab[0] < CLIP_HIGH)) continue;
    for (let a = 0; a < 3; a++) {
      sum[a] += lab[a];
      for (let b = 0; b < 3; b++) cross[a * 3 + b] += lab[a] * lab[b];
    }
    hist[Math.min(BINS - 1, Math.max(0, Math.round(lab[0] * 20)))]++;
    n++;
  }
  const mean = sum.map((s) => s / n) as [number, number, number];
  const cov = Array.from(cross, (v, k) => v / n - mean[Math.floor(k / 3)] * mean[k % 3]);
  const std = [0, 4, 8].map((k) => Math.max(Math.sqrt(Math.max(cov[k], 0)), 1e-3)) as [number, number, number];
  const lQuantiles: number[] = [];
  let acc = 0;
  let bin = 0;
  for (let q = 0; q < TONE_QUANTILE_COUNT; q++) {
    const target = (q / (TONE_QUANTILE_COUNT - 1)) * (n - 1);
    while (bin < BINS - 1 && acc + hist[bin] <= target) acc += hist[bin++];
    lQuantiles.push(bin / 20);
  }
  return { mean, std, cov, lQuantiles };
}

// --- colour transfer ---------------------------------------------------------

type M3 = number[]; // row-major 3x3

const mul = (a: M3, b: M3): M3 => {
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return o;
};

/** Eigen-decomposition of a symmetric 3x3 matrix (cyclic Jacobi). Returns eigenvalues and column eigenvectors. */
export function symEigen(m: M3): { values: number[]; vectors: M3 } {
  const a = [...m];
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[1] ** 2 + a[2] ** 2 + a[5] ** 2;
    if (off < 1e-18) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-15) continue;
      const theta = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p];
        const akq = a[k * 3 + q];
        a[k * 3 + p] = c * akp - s * akq;
        a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k];
        const aqk = a[q * 3 + k];
        a[p * 3 + k] = c * apk - s * aqk;
        a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p];
        const vkq = v[k * 3 + q];
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0], a[4], a[8]], vectors: v };
}

function symPow(m: M3, p: number): M3 {
  const { values, vectors: v } = symEigen(m);
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) o[i * 3 + j] += v[i * 3 + k] * Math.max(values[k], 1e-9) ** p * v[j * 3 + k];
  return o;
}

/** Same constant as services/ai/app/color.py COV_EPS. */
export const COV_EPS = 1.0;

/**
 * Linear L*a*b* colour-transfer matrix (row-major), applied to (lab - source.mean).
 * Port of match_matrix in services/ai/app/color.py: Monge-Kantorovich when both sides
 * carry covariance, per-channel std scaling otherwise.
 */
export function matchMatrix(source: ColorStats, reference: ColorStats): M3 {
  if (source.cov && reference.cov) {
    const reg = (c: number[]) => c.map((v, k) => (k % 4 === 0 ? v + COV_EPS : v));
    const cs = reg(source.cov);
    const cr = reg(reference.cov);
    const csH = symPow(cs, 0.5);
    const csIH = symPow(cs, -0.5);
    return mul(mul(csIH, symPow(mul(mul(csH, cr), csH), 0.5)), csIH);
  }
  const d = [0, 1, 2].map((i) => reference.std[i] / Math.max(source.std[i], 1e-3));
  return [d[0], 0, 0, 0, d[1], 0, 0, 0, d[2]];
}

/** Quantiles made strictly increasing (port of monotone_quantiles). */
export function monotoneQuantiles(q: number[]): number[] {
  const out = [...q];
  for (let i = 1; i < out.length; i++) out[i] = Math.max(out[i], out[i - 1] + 1e-3);
  return out;
}

/** Tone-curve knots for histogram matching with each shift limited (tone_curve_knots in color.py). */
export function toneCurveKnots(source: number[], reference: number[]): [number[], number[]] {
  const qs = monotoneQuantiles(source);
  const qr = monotoneQuantiles(reference).map((v, i) => qs[i] + Math.min(MAX_CURVE_SHIFT, Math.max(-MAX_CURVE_SHIFT, v - qs[i])));
  return [qs, monotoneQuantiles(qr)];
}

/** k-means (k colours) in L*a*b* over RGBA pixels: the reference's dominant palette, most common first. */
export function extractPalette(px: Uint8ClampedArray | Uint8Array, k = 5, maxSamples = 4096): Array<{ rgb: [number, number, number]; share: number }> {
  const count = px.length / 4;
  const step = Math.max(1, Math.floor(count / maxSamples));
  const pts: number[][] = [];
  const rgbs: number[][] = [];
  for (let i = 0; i < count; i += step) {
    pts.push(rgb8ToLab(px[i * 4], px[i * 4 + 1], px[i * 4 + 2]));
    rgbs.push([px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]);
  }
  if (!pts.length) return [];
  // Deterministic init: points spread along sorted lightness.
  const order = pts.map((_, i) => i).sort((a, b) => pts[a][0] - pts[b][0]);
  let centres = Array.from({ length: k }, (_, j) => [...pts[order[Math.floor(((j + 0.5) / k) * (pts.length - 1))]]]);
  const assign = new Int32Array(pts.length);
  for (let it = 0; it < 10; it++) {
    for (let i = 0; i < pts.length; i++) {
      let best = 0;
      let bd = Infinity;
      for (let j = 0; j < k; j++) {
        const d = (pts[i][0] - centres[j][0]) ** 2 + (pts[i][1] - centres[j][1]) ** 2 + (pts[i][2] - centres[j][2]) ** 2;
        if (d < bd) {
          bd = d;
          best = j;
        }
      }
      assign[i] = best;
    }
    centres = centres.map((c, j) => {
      const acc = [0, 0, 0];
      let m = 0;
      for (let i = 0; i < pts.length; i++)
        if (assign[i] === j) {
          acc[0] += pts[i][0];
          acc[1] += pts[i][1];
          acc[2] += pts[i][2];
          m++;
        }
      return m ? acc.map((v) => v / m) : c;
    });
  }
  const out = centres.map((_, j) => {
    const acc = [0, 0, 0];
    let m = 0;
    for (let i = 0; i < pts.length; i++)
      if (assign[i] === j) {
        acc[0] += rgbs[i][0];
        acc[1] += rgbs[i][1];
        acc[2] += rgbs[i][2];
        m++;
      }
    return { rgb: acc.map((v) => Math.round(v / Math.max(m, 1))) as [number, number, number], share: m / pts.length };
  });
  return out.filter((c) => c.share > 0).sort((a, b) => b.share - a.share);
}
