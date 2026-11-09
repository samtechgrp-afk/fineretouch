import type { FaceAnalysis } from '@fineretouch/shared';
import { describe, expect, it } from 'vitest';
import { buildEditMask, buildFaceMask, rasterizeMaskStrokes } from './masks';

const W = 200;
const H = 200;

/** A face in the middle of the frame (oval 0.35..0.65 x 0.3..0.7), eyes and lips cut out. */
function face(): FaceAnalysis {
  const oval = Array.from({ length: 36 }, (_, i) => {
    const a = (i / 36) * Math.PI * 2;
    return [0.5 + 0.15 * Math.cos(a), 0.5 + 0.2 * Math.sin(a)] as [number, number];
  });
  const sq = (x0: number, y0: number, x1: number, y1: number): [number, number][] => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  return {
    box: [0.35, 0.3, 0.3, 0.4],
    landmarkCount: 478,
    eyeOpenness: 0.8,
    regions: {
      faceOval: oval,
      leftEye: sq(0.53, 0.44, 0.6, 0.47),
      rightEye: sq(0.4, 0.44, 0.47, 0.47),
      leftUnderEye: sq(0.53, 0.48, 0.6, 0.51),
      rightUnderEye: sq(0.4, 0.48, 0.47, 0.51),
      innerMouth: sq(0.46, 0.6, 0.54, 0.62),
      lips: sq(0.44, 0.59, 0.56, 0.63),
      leftBrow: [],
      rightBrow: [],
    },
  };
}

/** Parse map RGBA: r face skin, g body skin, b hair. */
function parseMap(fill: (x: number, y: number) => [number, number, number]): Uint8Array {
  const d = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const [f, b, h] = fill(x, y);
      d.set([f, b, h, 255], (y * W + x) * 4);
    }
  return d;
}
const skinAt = (m: Uint8Array, x: number, y: number) => m[(y * W + x) * 4] / 255;
const count = (m: Uint8Array) => {
  let n = 0;
  for (let i = 0; i < W * H; i++) if (m[i * 4] > 128) n++;
  return n;
};

describe('semantic skin mask (face parsing)', () => {
  it('extends skin to parsed face skin beyond the landmark oval, keeping eyes and lips out', () => {
    const base = buildFaceMask([face()], W, H);
    // Parsed face skin reaches the hairline above the oval (y 45..60).
    const parsed = buildFaceMask([face()], W, H, { data: parseMap((x, y) => [x > 70 && x < 130 && y > 45 && y < 140 ? 255 : 0, 0, 0]), includeBody: false });
    expect(count(parsed)).toBeGreaterThan(count(base));
    expect(skinAt(base, 100, 55)).toBeLessThan(0.2);
    expect(skinAt(parsed, 100, 55)).toBeGreaterThan(0.8);
    expect(skinAt(parsed, 87, 91)).toBeLessThan(0.3); // right eye stays out
    expect(skinAt(parsed, 100, 122)).toBeLessThan(0.3); // lips stay out
  });

  it('ignores parsed skin far from any face (a poster in the background)', () => {
    const parsed = buildFaceMask([face()], W, H, { data: parseMap((x, y) => [x < 20 && y < 20 ? 255 : 0, 0, 0]), includeBody: false });
    expect(skinAt(parsed, 8, 8)).toBe(0);
  });

  it('removes skin under hair lying across the face', () => {
    const parsed = buildFaceMask([face()], W, H, { data: parseMap((_, y) => [0, 0, y > 70 && y < 80 ? 255 : 0]), includeBody: false });
    expect(skinAt(buildFaceMask([face()], W, H), 100, 75)).toBeGreaterThan(0.8);
    expect(skinAt(parsed, 100, 75)).toBeLessThan(0.2);
  });

  it('adds neck and body skin only when asked', () => {
    const body = parseMap((_, y) => [0, y > 160 ? 255 : 0, 0]);
    expect(skinAt(buildFaceMask([face()], W, H, { data: body, includeBody: false }), 20, 190)).toBe(0);
    expect(skinAt(buildFaceMask([face()], W, H, { data: body, includeBody: true }), 20, 190)).toBeGreaterThan(0.9);
  });
});

describe('mask brush strokes', () => {
  it('rasterises where drawn, with a radius relative to the long edge', () => {
    const { add, erase } = rasterizeMaskStrokes([{ mode: 'add', size: 0.05, points: [0.2, 0.5, 0.8, 0.5] }, { mode: 'erase', size: 0.05, points: [0.5, 0.2] }], 100, 100, 0);
    expect(add[50 * 100 + 50]).toBe(1);
    expect(add[50 * 100 + 5]).toBe(0);
    expect(add[10 * 100 + 50]).toBe(0);
    expect(erase[20 * 100 + 50]).toBe(1);
    expect(erase[80 * 100 + 50]).toBe(0);
  });

  it('feathers edges', () => {
    const hard = rasterizeMaskStrokes([{ mode: 'add', size: 0.05, points: [0.2, 0.5, 0.8, 0.5] }], 100, 100, 0).add;
    const soft = rasterizeMaskStrokes([{ mode: 'add', size: 0.05, points: [0.2, 0.5, 0.8, 0.5] }], 100, 100, 100).add;
    expect(hard[44 * 100 + 50]).toBe(0);
    expect(soft[44 * 100 + 50]).toBeGreaterThan(0);
    expect(soft[44 * 100 + 50]).toBeLessThan(1);
  });

  it('packs add in red and erase in green, null without strokes', () => {
    expect(buildEditMask([], 10, 10, 50)).toBeNull();
    const t = buildEditMask([{ mode: 'erase', size: 0.2, points: [0.5, 0.5] }], 10, 10, 0)!;
    const i = (5 * 10 + 5) * 4;
    expect([t[i], t[i + 1]]).toEqual([0, 255]);
  });
});
