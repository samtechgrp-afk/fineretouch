import { describe, expect, it } from 'vitest';
import { createLayer, type HealStroke } from '@fineretouch/shared';
import { applyHealStrokes, autoHealSource, buildLayerMask, paintStrokes, pushPull } from './heal';

const N = 128;
const SKIN = [200, 150, 122];

/** Deterministic skin with fine texture and a gentle left-right gradient. */
function skin(seed = 7): Uint8ClampedArray {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) - 0.5;
  const px = new Uint8ClampedArray(N * N * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const n = rnd() * 14 + (x - N / 2) * 0.08;
      const i = (y * N + x) * 4;
      for (let k = 0; k < 3; k++) px[i + k] = SKIN[k] + n;
      px[i + 3] = 255;
    }
  }
  return px;
}

function stamp(px: Uint8ClampedArray, cx: number, cy: number, r: number, rgb: number[]) {
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) for (let k = 0; k < 3; k++) px[(y * N + x) * 4 + k] = rgb[k];
  }
}

const luma = (px: Uint8ClampedArray, i: number) => 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
function region(cx: number, cy: number, r: number): number[] {
  const out: number[] = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) out.push(y * N + x);
  return out;
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const std = (v: number[]) => Math.sqrt(mean(v.map((x) => (x - mean(v)) ** 2)));
/** Fine detail: pixel minus the mean of its 3x3 neighbourhood. */
function detail(px: Uint8ClampedArray, idx: number[]): number[] {
  return idx.map((i) => {
    const x = i % N;
    const y = Math.floor(i / N);
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += luma(px, Math.min(N - 1, Math.max(0, y + dy)) * N + Math.min(N - 1, Math.max(0, x + dx)));
    return luma(px, i) - s / 9;
  });
}

describe('push-pull fill', () => {
  it('keeps known pixels and continues a ramp across a hole', () => {
    const w = 64;
    const v = new Float32Array(w * w);
    const known = new Float32Array(w * w);
    for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) {
      v[y * w + x] = x / w;
      known[y * w + x] = (x - 32) ** 2 + (y - 32) ** 2 > 100 ? 1 : 0;
    }
    const f = pushPull(v, known, w, w, 1);
    for (let i = 0; i < w * w; i++) {
      if (known[i]) expect(f[i]).toBeCloseTo(v[i], 5);
      else expect(Math.abs(f[i] - v[i])).toBeLessThan(0.06);
    }
  });
});

describe('healing and clone', () => {
  it('heals a dark blemish to the surrounding tone, keeps texture, leaves the rest alone', () => {
    const clean = skin();
    const img = new Uint8ClampedArray(clean);
    stamp(img, 40, 64, 5, [120, 80, 70]);
    const s: HealStroke = { mode: 'heal', size: 8 / N, hardness: 40, opacity: 100, offset: [40 / N, 0], points: [40 / N, 64 / N] };
    const out = applyHealStrokes(img, N, N, [s]);
    const inner = region(40, 64, 4);
    expect(Math.abs(mean(inner.map((i) => luma(out, i))) - mean(inner.map((i) => luma(clean, i))))).toBeLessThan(4);
    expect(std(detail(out, inner))).toBeGreaterThan(0.5 * std(detail(clean, inner)));
    for (const i of region(100, 100, 10)) expect(out[i * 4]).toBe(img[i * 4]);
  });

  it('clones the source exactly at full opacity', () => {
    const img = skin();
    stamp(img, 90, 64, 6, [20, 40, 230]);
    const out = applyHealStrokes(img, N, N, [{ mode: 'clone', size: 4 / N, hardness: 100, opacity: 100, offset: [50 / N, 0], points: [40 / N, 64 / N] }]);
    expect(Array.from(out.slice((64 * N + 40) * 4, (64 * N + 40) * 4 + 3))).toEqual([20, 40, 230]);
  });

  it('spot healing picks a clean source, not another blemish', () => {
    const img = skin();
    stamp(img, 64, 64, 4, [90, 60, 50]);
    // Blemishes on most candidate spots to the right; the left stays clean.
    for (const dy of [-18, -9, 0, 9, 18]) stamp(img, 64 + 18, 64 + dy, 4, [90, 60, 50]);
    const off = autoHealSource(img, N, N, 5 / N, [64 / N, 64 / N]);
    expect(off).not.toBeNull();
    const out = applyHealStrokes(img, N, N, [{ mode: 'heal', size: 5 / N, hardness: 50, opacity: 100, offset: off!, points: [64 / N, 64 / N] }]);
    expect(mean(region(64, 64, 3).map((i) => luma(out, i)))).toBeGreaterThan(150);
  });
});

describe('layer masks', () => {
  it('builds up flow and erases', () => {
    const a = { mode: 'add' as const, size: 6 / N, flow: 40, points: [0.5, 0.5] };
    expect(paintStrokes([a], 'add', N, N, 0)![64 * N + 64]).toBeCloseTo(0.4, 5);
    expect(paintStrokes([a, a], 'add', N, N, 0)![64 * N + 64]).toBeCloseTo(0.8, 5);
    const l = createLayer('correction', 'x');
    l.mask.feather = 0;
    l.mask.strokes = [a, { mode: 'erase', size: 3 / N, flow: 100, points: [0.5, 0.5] }];
    const m = buildLayerMask(l, N, N, null);
    expect(m[(64 * N + 64) * 4]).toBe(0);
    expect(m[(64 * N + 67) * 4]).toBe(Math.round(0.4 * 255));
  });

  it('starts from the skin mask, inverts, and splits dodge from burn', () => {
    const skinW = new Float32Array(N * N).map((_, i) => (i % N < 64 ? 1 : 0));
    const f = createLayer('frequency', 'f');
    const m = buildLayerMask(f, N, N, skinW);
    expect(m[(10 * N + 10) * 4]).toBe(255);
    expect(m[(10 * N + 100) * 4]).toBe(0);
    f.mask.invert = true;
    expect(buildLayerMask(f, N, N, skinW)[(10 * N + 10) * 4]).toBe(0);
    const db = createLayer('dodgeBurn', 'd');
    db.mask.feather = 0;
    db.mask.strokes = [
      { mode: 'dodge', size: 5 / N, flow: 100, points: [0.25, 0.5] },
      { mode: 'burn', size: 5 / N, flow: 50, points: [0.75, 0.5] },
    ];
    const d = buildLayerMask(db, N, N, null);
    expect(d[(64 * N + 32) * 4 + 1]).toBe(255);
    expect(d[(64 * N + 32) * 4 + 2]).toBe(0);
    expect(d[(64 * N + 96) * 4 + 2]).toBe(128);
  });
});

describe('parity with the engine (services/ai/tests/test_local_tools.py HEAL_VECTORS)', () => {
  it('heals and clones the same pixels as engines/local.py', () => {
    const w = 48;
    const h = 40;
    const px = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const b = 120 + ((x * 37 + y * 91) % 61);
      const dark = (x - 14) ** 2 + (y - 20) ** 2 <= 16;
      const i = (y * w + x) * 4;
      px.set(dark ? [60, 40, 30, 255] : [b + 40, b, b - 20, 255], i);
    }
    const strokes: HealStroke[] = [
      { mode: 'heal', size: 0.12, hardness: 30, opacity: 100, offset: [0.4, 0.05], points: [0.29, 0.5, 0.31, 0.52] },
      { mode: 'clone', size: 0.08, hardness: 70, opacity: 60, offset: [-0.3, 0.1], points: [0.75, 0.3] },
    ];
    const out = applyHealStrokes(px, w, h, strokes);
    const vectors: Array<[number, number, number[]]> = [
      [14, 20, [206.48, 166.48, 146.48]],
      [12, 22, [168.12, 131.09, 112.57]],
      [16, 19, [187.13, 147.47, 127.64]],
      [36, 12, [191.56, 151.56, 131.56]],
      [34, 13, [183.9, 143.9, 123.9]],
      [5, 5, [190, 150, 130]],
    ];
    for (const [x, y, want] of vectors) {
      for (let k = 0; k < 3; k++) expect(Math.abs(out[(y * w + x) * 4 + k] - want[k])).toBeLessThanOrEqual(1);
    }
  });
});
