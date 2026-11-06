import type { FaceAnalysis } from '@fineretouch/shared';
import { describe, expect, it } from 'vitest';
import { buildFaceMask, fillPolygon, gaussianApprox } from './masks';

const square = (x0: number, y0: number, x1: number, y1: number): [number, number][] => [
  [x0, y0],
  [x1, y0],
  [x1, y1],
  [x0, y1],
];

function face(): FaceAnalysis {
  const oval = Array.from({ length: 36 }, (_, i) => {
    const a = (i / 36) * Math.PI * 2;
    return [0.5 + 0.4 * Math.cos(a), 0.5 + 0.45 * Math.sin(a)] as [number, number];
  });
  return {
    box: [0.1, 0.05, 0.8, 0.9],
    landmarkCount: 478,
    eyeOpenness: 0.8,
    regions: {
      faceOval: oval,
      leftEye: square(0.6, 0.38, 0.7, 0.42),
      rightEye: square(0.3, 0.38, 0.4, 0.42),
      leftUnderEye: square(0.6, 0.43, 0.7, 0.48),
      rightUnderEye: square(0.3, 0.43, 0.4, 0.48),
      innerMouth: square(0.42, 0.7, 0.58, 0.76),
      lips: square(0.4, 0.68, 0.6, 0.78),
      leftBrow: [],
      rightBrow: [],
    },
  };
}

describe('mask rasterisation', () => {
  it('fills polygons at pixel centres', () => {
    const buf = new Float32Array(10 * 10);
    fillPolygon(buf, 10, 10, square(2, 2, 6, 6), 1);
    expect(buf.reduce((a, b) => a + b, 0)).toBe(16);
    expect(buf[2 * 10 + 2]).toBe(1);
    expect(buf[6 * 10 + 6]).toBe(0);
  });

  it('blur preserves mass and softens edges', () => {
    const w = 64;
    const buf = new Float32Array(w * w);
    fillPolygon(buf, w, w, square(24, 24, 40, 40), 1);
    const before = buf.reduce((a, b) => a + b, 0);
    gaussianApprox(buf, w, w, 3);
    expect(buf.reduce((a, b) => a + b, 0)).toBeCloseTo(before, 0);
    expect(buf[32 * w + 32]).toBeGreaterThan(0.95);
    expect(buf[32 * w + 24]).toBeGreaterThan(0.3);
    expect(buf[32 * w + 24]).toBeLessThan(0.7);
  });

  it('builds skin/eye/under-eye/mouth channels and cuts features out of the skin', () => {
    const w = 200;
    const h = 200;
    const m = buildFaceMask([face()], w, h);
    const at = (x: number, y: number, c: number) => m[(Math.round(y * h) * w + Math.round(x * w)) * 4 + c];
    expect(at(0.5, 0.55, 0)).toBeGreaterThan(240); // cheek/nose skin
    expect(at(0.02, 0.02, 0)).toBe(0); // background
    expect(at(0.35, 0.4, 0)).toBeLessThan(at(0.5, 0.55, 0)); // eye cut out of skin
    expect(at(0.35, 0.4, 1)).toBeGreaterThan(100); // eye channel
    expect(at(0.35, 0.455, 2)).toBeGreaterThan(100); // under-eye channel
    expect(at(0.5, 0.73, 3)).toBeGreaterThan(100); // mouth channel
  });

  it('is empty without faces', () => {
    expect(buildFaceMask([], 8, 8).every((v) => v === 0)).toBe(true);
  });
});
