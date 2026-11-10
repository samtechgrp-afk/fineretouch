import { readFileSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import type { HealStroke } from '@fineretouch/shared';
import { applyHealStrokes } from './heal';

/**
 * Browser / Python healing parity. The fixture is written by
 * services/ai/scripts/heal_parity_fixture.py: a crop of a Skin Perfection result with repairs
 * in it, four heal and clone strokes across those repairs, and what the Python engine
 * (engines/local.apply_heal_strokes, the export path) makes of them. The preview runs this
 * module on the same pixels, so the two must agree. services/ai/tests/test_heal_parity.py
 * checks the Python side against the same file.
 */
interface Fixture {
  width: number;
  height: number;
  strokes: HealStroke[];
  input: string;
  expected: string;
  repairs: { x: number; y: number; r: number }[];
}

const fx: Fixture = JSON.parse(readFileSync(new URL('./__fixtures__/heal-parity.json', import.meta.url), 'utf8'));
const W = fx.width;
const H = fx.height;
const rgb = (s: string) => new Uint8Array(inflateSync(Buffer.from(s, 'base64')));

function rgba(src: Uint8Array): Uint8ClampedArray {
  const px = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    px[i * 4] = src[i * 3];
    px[i * 4 + 1] = src[i * 3 + 1];
    px[i * 4 + 2] = src[i * 3 + 2];
    px[i * 4 + 3] = 255;
  }
  return px;
}

function compare(out: Uint8ClampedArray, want: Uint8Array, within: (x: number, y: number) => boolean = () => true) {
  let max = 0;
  let sum = 0;
  let n = 0;
  let off = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (!within(x, y)) continue;
    const i = y * W + x;
    for (let k = 0; k < 3; k++) {
      const d = Math.abs(out[i * 4 + k] - want[i * 3 + k]);
      max = Math.max(max, d);
      sum += d;
      n++;
      if (d > 0) off++;
    }
  }
  return { max, mean: sum / n, off, n };
}

describe('healing parity with the Python engine', () => {
  const input = rgb(fx.input);
  const want = rgb(fx.expected);
  const out = applyHealStrokes(rgba(input), W, H, fx.strokes);

  it('the strokes change the repairs (the fixture is not trivial)', () => {
    const before = compare(rgba(input), want);
    expect(before.off).toBeGreaterThan(1000);
    expect(before.max).toBeGreaterThan(5);
  });

  it('gives the same pixels as the Python engine, within 8-bit rounding', () => {
    const all = compare(out, want);
    if (process.env.HEAL_PARITY_REPORT) console.log('parity', JSON.stringify(all));
    // The browser rounds to 8 bits after each stroke, Python keeps floats between strokes and
    // rounds once. Where strokes overlap that can move a channel by one level, never more.
    expect(all.max).toBeLessThanOrEqual(1);
    expect(all.mean).toBeLessThan(0.01);
  });

  it('agrees exactly-or-within-one inside the Skin Perfection repairs the strokes cross', () => {
    for (const m of fx.repairs) {
      const r = compare(out, want, (x, y) => (x + 0.5 - m.x) ** 2 + (y + 0.5 - m.y) ** 2 <= (m.r + 2) ** 2);
      expect(r.n).toBeGreaterThan(100);
      expect(r.max).toBeLessThanOrEqual(1);
    }
  });

  it('pixels beyond the reach of every stroke are left exactly as they were', () => {
    const L = Math.max(W, H);
    const reach = fx.strokes.map((s) => {
      const r = Math.max(1, s.size * L);
      const pts: [number, number][] = [];
      for (let i = 0; i + 1 < s.points.length; i += 2) pts.push([s.points[i] * W, s.points[i + 1] * H]);
      return { pts, r: r + 1 };
    });
    const near = (x: number, y: number) => reach.some(({ pts, r }) => pts.some(([px, py], j) => {
      const [qx, qy] = pts[Math.min(j + 1, pts.length - 1)];
      const vx = qx - px, vy = qy - py;
      const t = vx || vy ? Math.max(0, Math.min(1, ((x + 0.5 - px) * vx + (y + 0.5 - py) * vy) / (vx * vx + vy * vy))) : 0;
      return (x + 0.5 - px - t * vx) ** 2 + (y + 0.5 - py - t * vy) ** 2 <= r * r;
    }));
    const src = rgba(input);
    const far = (x: number, y: number) => !near(x, y);
    expect(compare(out, input, far).max).toBe(0);
    expect(compare(src, want, far).max).toBe(0);
  });
});
