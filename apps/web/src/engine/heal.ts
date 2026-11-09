/**
 * Pro tools on the CPU (v0.10, Phase 5): healing and clone strokes, the automatic heal source,
 * and the local retouch layer masks. A port of services/ai/app/engines/local.py: any change to
 * one must be mirrored in the other.
 *
 * Heal / clone run on the preview pixels (and on the full-size pixels for a browser export)
 * before the shader chain, exactly where the engine runs them (stage 0b), so every automatic
 * tool sees the cleaned photo.
 */

import type { HealStroke, LayerStroke, RetouchLayer } from '@fineretouch/shared';
import { gaussianApprox, STROKE_FEATHER_MAX } from './masks';

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
/** Coverage above which a pixel is filled rather than known (engines/local.HEAL_UNKNOWN). */
export const HEAL_UNKNOWN = 0.002;

// --- membrane fill ---------------------------------------------------------------------------

/**
 * Fill unknown pixels (known[i] == 0) of a C-channel float image smoothly from the known ones
 * (push-pull: weighted 2x2 pyramid down, bilinear back up). Same as engines/local.push_pull.
 */
export function pushPull(values: Float32Array, known: Float32Array, w: number, h: number, C: number): Float32Array {
  const V: Float32Array[] = [];
  const W: Float32Array[] = [];
  const dims: [number, number][] = [];
  let v = new Float32Array(w * h * C);
  let wt = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    wt[i] = known[i];
    for (let k = 0; k < C; k++) v[i * C + k] = values[i * C + k] * known[i];
  }
  V.push(v);
  W.push(wt);
  dims.push([w, h]);
  let cw = w;
  let ch = h;
  while (cw > 1 || ch > 1) {
    const w2 = (cw + 1) >> 1;
    const h2 = (ch + 1) >> 1;
    const v2 = new Float32Array(w2 * h2 * C);
    const wt2 = new Float32Array(w2 * h2);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const i = y * cw + x;
        const j = (y >> 1) * w2 + (x >> 1);
        wt2[j] += wt[i];
        for (let k = 0; k < C; k++) v2[j * C + k] += v[i * C + k];
      }
    }
    for (let j = 0; j < w2 * h2; j++) {
      if (wt2[j] > 1) {
        for (let k = 0; k < C; k++) v2[j * C + k] /= wt2[j];
        wt2[j] = 1;
      }
    }
    v = v2;
    wt = wt2;
    cw = w2;
    ch = h2;
    V.push(v);
    W.push(wt);
    dims.push([cw, ch]);
  }
  let c = new Float32Array(C);
  if (wt[0] > 0) for (let k = 0; k < C; k++) c[k] = v[k] / wt[0];
  for (let level = V.length - 2; level >= 0; level--) {
    const [lw, lh] = dims[level];
    const [pw, ph] = dims[level + 1];
    const lv = V[level];
    const lwt = W[level];
    const next = new Float32Array(lw * lh * C);
    for (let y = 0; y < lh; y++) {
      const sy = Math.min(ph - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
      const y0 = Math.floor(sy);
      const y1 = Math.min(y0 + 1, ph - 1);
      const fy = sy - y0;
      for (let x = 0; x < lw; x++) {
        const sx = Math.min(pw - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
        const x0 = Math.floor(sx);
        const x1 = Math.min(x0 + 1, pw - 1);
        const fx = sx - x0;
        const i = y * lw + x;
        const rest = 1 - lwt[i];
        for (let k = 0; k < C; k++) {
          const top = c[(y0 * pw + x0) * C + k] * (1 - fx) + c[(y0 * pw + x1) * C + k] * fx;
          const bot = c[(y1 * pw + x0) * C + k] * (1 - fx) + c[(y1 * pw + x1) * C + k] * fx;
          next[i * C + k] = lv[i * C + k] + rest * (top * (1 - fy) + bot * fy);
        }
      }
    }
    c = next;
  }
  return c;
}

// --- brush geometry --------------------------------------------------------------------------

function polyline(points: readonly number[], w: number, h: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i + 1 < points.length; i += 2) out.push([points[i] * w, points[i + 1] * h]);
  return out;
}

/** Distance (px) from each pixel centre of the crop to the polyline (Infinity beyond reach). */
export function distanceToPolyline(pts: [number, number][], x0: number, y0: number, cw: number, ch: number, reach: number): Float32Array {
  const d = new Float32Array(cw * ch).fill(Infinity);
  const segs: [[number, number], [number, number]][] = pts.length === 1 ? [[pts[0], pts[0]]] : pts.slice(1).map((b, i) => [pts[i], b]);
  for (const [a, b] of segs) {
    const sx0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - reach) - x0);
    const sx1 = Math.min(cw, Math.ceil(Math.max(a[0], b[0]) + reach) - x0 + 1);
    const sy0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - reach) - y0);
    const sy1 = Math.min(ch, Math.ceil(Math.max(a[1], b[1]) + reach) - y0 + 1);
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    for (let y = sy0; y < sy1; y++) {
      const py = y + y0 + 0.5;
      for (let x = sx0; x < sx1; x++) {
        const px = x + x0 + 0.5;
        const t = len2 > 0 ? clamp01(((px - a[0]) * dx + (py - a[1]) * dy) / len2) : 0;
        const ex = px - (a[0] + t * dx);
        const ey = py - (a[1] + t * dy);
        const dist = Math.sqrt(ex * ex + ey * ey);
        const i = y * cw + x;
        if (dist < d[i]) d[i] = dist;
      }
    }
  }
  return d;
}

/** Soft brush tip: 1 inside hardness * r, smooth falloff to r (engines/local.brush_coverage). */
export function brushCoverage(d: Float32Array, r: number, hardness: number): Float32Array {
  const inner = r * clamp01(hardness / 100);
  const span = Math.max(r + 0.5 - inner, 1e-3);
  const out = new Float32Array(d.length);
  for (let i = 0; i < d.length; i++) {
    const t = clamp01((d[i] - inner) / span);
    out[i] = 1 - t * t * (3 - 2 * t);
  }
  return out;
}

/** RGB at pixel-index coordinates (x, y), bilinear, edges clamped; values 0..1. */
function sample(px: Uint8ClampedArray, w: number, h: number, x: number, y: number, out: Float32Array, o: number) {
  x = Math.min(w - 1, Math.max(0, x));
  y = Math.min(h - 1, Math.max(0, y));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const fx = x - x0;
  const fy = y - y0;
  for (let k = 0; k < 3; k++) {
    const top = px[(y0 * w + x0) * 4 + k] * (1 - fx) + px[(y0 * w + x1) * 4 + k] * fx;
    const bot = px[(y1 * w + x0) * 4 + k] * (1 - fx) + px[(y1 * w + x1) * 4 + k] * fx;
    out[o + k] = (top * (1 - fy) + bot * fy) / 255;
  }
}

// --- healing and clone -----------------------------------------------------------------------

/** One heal / clone stroke on RGBA pixels (modified in place). Same as engines/local.apply_heal_stroke. */
export function applyHealStroke(px: Uint8ClampedArray, w: number, h: number, s: HealStroke): void {
  const L = Math.max(w, h);
  const r = Math.max(1, s.size * L);
  const pts = polyline(s.points, w, h);
  if (!pts.length) return;
  const pad = r + Math.max(3, 0.5 * r);
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad));
  const x1 = Math.min(w, Math.ceil(Math.max(...xs) + pad) + 1);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - pad));
  const y1 = Math.min(h, Math.ceil(Math.max(...ys) + pad) + 1);
  if (x0 >= x1 || y0 >= y1) return;
  const cw = x1 - x0;
  const ch = y1 - y0;
  const cov = brushCoverage(distanceToPolyline(pts, x0, y0, cw, ch, r + 1), r, s.hardness);
  const op = s.opacity / 100;
  const ox = s.offset[0] * w;
  const oy = s.offset[1] * h;
  const n = cw * ch;
  const source = new Float32Array(n * 3);
  const dest = new Float32Array(n * 3);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = y * cw + x;
      sample(px, w, h, x + x0 + ox, y + y0 + oy, source, i * 3);
      const q = ((y + y0) * w + x + x0) * 4;
      dest[i * 3] = px[q] / 255;
      dest[i * 3 + 1] = px[q + 1] / 255;
      dest[i * 3 + 2] = px[q + 2] / 255;
    }
  }
  if (s.mode === 'heal') {
    const known = new Float32Array(n);
    let any = false;
    for (let i = 0; i < n; i++) {
      known[i] = cov[i] <= HEAL_UNKNOWN ? 1 : 0;
      any ||= known[i] > 0;
    }
    if (any) {
      const diff = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) diff[i] = dest[i] - source[i];
      const filled = pushPull(diff, known, cw, ch, 3);
      for (let i = 0; i < n * 3; i++) source[i] += filled[i];
    }
  }
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = y * cw + x;
      const a = cov[i] * op;
      if (a <= 0) continue;
      const q = ((y + y0) * w + x + x0) * 4;
      for (let k = 0; k < 3; k++) px[q + k] = Math.round(255 * clamp01(dest[i * 3 + k] + (source[i * 3 + k] - dest[i * 3 + k]) * a));
    }
  }
}

/** All strokes in order on a copy of the pixels (each stroke sees the previous ones). */
export function applyHealStrokes(px: Uint8ClampedArray, w: number, h: number, strokes: readonly HealStroke[]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(px);
  for (const s of strokes) applyHealStroke(out, w, h, s);
  return out;
}

/**
 * Spot healing: pick a heal source near the stroke automatically. Candidates ring the stroke
 * at 1.5-3x its extent in 16 directions; the best one matches the destination's surroundings
 * (border ring) and has no blemish of its own (its inside close to that ring's tone). Returns
 * the offset normalised to the image (as HealStroke.offset), or null if nothing fits.
 */
export function autoHealSource(px: Uint8ClampedArray, w: number, h: number, size: number, points: readonly number[]): [number, number] | null {
  const L = Math.max(w, h);
  const r = Math.max(1, size * L);
  const pts = polyline(points, w, h);
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const ring = r * 0.6 + 2;
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - r - ring));
  const x1 = Math.min(w, Math.ceil(Math.max(...xs) + r + ring) + 1);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - r - ring));
  const y1 = Math.min(h, Math.ceil(Math.max(...ys) + r + ring) + 1);
  const cw = x1 - x0;
  const ch = y1 - y0;
  if (cw <= 0 || ch <= 0) return null;
  const d = distanceToPolyline(pts, x0, y0, cw, ch, r + ring + 1);
  const ringIdx: number[] = [];
  const innerIdx: number[] = [];
  const step = Math.max(1, Math.round(Math.sqrt((cw * ch) / 3000)));
  for (let y = 0; y < ch; y += step) {
    for (let x = 0; x < cw; x += step) {
      const v = d[y * cw + x];
      if (v <= r) innerIdx.push(y * cw + x);
      else if (v <= r + ring) ringIdx.push(y * cw + x);
    }
  }
  if (!ringIdx.length || !innerIdx.length) return null;
  const at = (i: number, dx: number, dy: number, out: Float32Array) => sample(px, w, h, (i % cw) + x0 + dx, Math.floor(i / cw) + y0 + dy, out, 0);
  const a = new Float32Array(3);
  const b = new Float32Array(3);
  const ringMean = [0, 0, 0];
  for (const i of ringIdx) {
    at(i, 0, 0, a);
    for (let k = 0; k < 3; k++) ringMean[k] += a[k] / ringIdx.length;
  }
  const extent = Math.max((Math.max(...xs) - Math.min(...xs)) / 2, (Math.max(...ys) - Math.min(...ys)) / 2) + r;
  let best: [number, number] | null = null;
  let bestScore = Infinity;
  for (const dist of [1.5, 2.2, 3.0]) {
    for (let k = 0; k < 16; k++) {
      const ang = (k / 16) * Math.PI * 2;
      const dx = Math.cos(ang) * dist * extent;
      const dy = Math.sin(ang) * dist * extent;
      // The source must lie inside the photo (no clamped edge pixels).
      if (x0 + dx < 0 || y0 + dy < 0 || x1 + dx > w || y1 + dy > h) continue;
      let border = 0;
      for (const i of ringIdx) {
        at(i, 0, 0, a);
        at(i, dx, dy, b);
        border += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
      }
      let inside = 0;
      for (const i of innerIdx) {
        at(i, dx, dy, b);
        inside += (b[0] - ringMean[0]) ** 2 + (b[1] - ringMean[1]) ** 2 + (b[2] - ringMean[2]) ** 2;
      }
      const score = border / ringIdx.length + 0.5 * (inside / innerIdx.length) + 1e-4 * dist;
      if (score < bestScore) {
        bestScore = score;
        best = [dx / w, dy / h];
      }
    }
  }
  return best;
}

// --- layer masks -----------------------------------------------------------------------------

/** Coverage of the strokes with this mode; flow builds up across strokes (engines/local.paint_strokes). */
export function paintStrokes(strokes: readonly LayerStroke[], mode: LayerStroke['mode'], w: number, h: number, feather: number): Float32Array | null {
  const group = strokes.filter((s) => s.mode === mode);
  if (!group.length) return null;
  const L = Math.max(w, h);
  const canvas = new Float32Array(w * h);
  const f = clamp01(feather / 100) * STROKE_FEATHER_MAX;
  let sigma = 0;
  for (const s of group) {
    const r = Math.max(0.5, s.size * L);
    const pts = polyline(s.points, w, h);
    sigma = Math.max(sigma, f * r * 0.5);
    if (!pts.length) continue;
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const x0 = Math.max(0, Math.floor(Math.min(...xs) - r - 2));
    const x1 = Math.min(w, Math.ceil(Math.max(...xs) + r + 2) + 1);
    const y0 = Math.max(0, Math.floor(Math.min(...ys) - r - 2));
    const y1 = Math.min(h, Math.ceil(Math.max(...ys) + r + 2) + 1);
    if (x0 >= x1 || y0 >= y1) continue;
    const cw = x1 - x0;
    const d = distanceToPolyline(pts, x0, y0, cw, y1 - y0, r + 1);
    const flow = s.flow / 100;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const c = clamp01(r + 0.5 - d[(y - y0) * cw + (x - x0)]);
        if (c <= 0) continue;
        const i = y * w + x;
        canvas[i] = Math.min(1, canvas[i] + c * flow);
      }
    }
  }
  if (sigma >= 0.5) gaussianApprox(canvas, w, h, sigma);
  for (let i = 0; i < canvas.length; i++) canvas[i] = clamp01(canvas[i]);
  return canvas;
}

/**
 * RGBA texture of one layer's mask for the LOCAL_FRAG pass: r = coverage (base, add and erase
 * strokes, invert), g = dodge, b = burn. The luminance range, density and opacity are applied
 * in the shader, since they depend on the pixels under the layer. `skin` (0..1 per pixel, same
 * size) is the skin mask for the 'skin' base, or null when there is none.
 */
export function buildLayerMask(layer: RetouchLayer, w: number, h: number, skin: Float32Array | null): Uint8Array {
  const m = layer.mask;
  const add = paintStrokes(m.strokes, 'add', w, h, m.feather);
  const erase = paintStrokes(m.strokes, 'erase', w, h, m.feather);
  const dodge = layer.kind === 'dodgeBurn' ? paintStrokes(m.strokes, 'dodge', w, h, m.feather) : null;
  const burn = layer.kind === 'dodgeBurn' ? paintStrokes(m.strokes, 'burn', w, h, m.feather) : null;
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const keep = erase ? 1 - erase[i] : 1;
    const base = m.base === 'all' ? 1 : m.base === 'skin' && skin ? skin[i] : 0;
    let cov = Math.max(base, add ? add[i] : 0) * keep;
    if (m.invert) cov = 1 - cov;
    const gate = m.base === 'skin' && skin ? skin[i] : 1;
    out[i * 4] = Math.round(cov * 255);
    out[i * 4 + 1] = dodge ? Math.round(dodge[i] * keep * gate * 255) : 0;
    out[i * 4 + 2] = burn ? Math.round(burn[i] * keep * gate * 255) : 0;
    out[i * 4 + 3] = 255;
  }
  return out;
}
