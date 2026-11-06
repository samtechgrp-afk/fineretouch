import type { FaceAnalysis } from '@fineretouch/shared';
import { describe, expect, it } from 'vitest';
import { colorStatsFromRgba, extractPalette, matchMatrix, symEigen } from './color';
import { buildDepthPrior, buildHairRoiMask } from './masks';
import { apertureKernel } from './shaders';
import { profileFromRgba } from './style';

const mul = (a: number[], b: number[]) =>
  Array.from({ length: 9 }, (_, k) => [0, 1, 2].reduce((s, j) => s + a[Math.floor(k / 3) * 3 + j] * b[j * 3 + (k % 3)], 0));
const tr = (a: number[]) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];

function noise(w: number, h: number, base: [number, number, number], amp: number, seed = 1): Uint8ClampedArray {
  let x = seed;
  const rnd = () => ((x = (x * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
  const px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const n = rnd() * amp;
    px[i * 4] = base[0] + n + rnd() * amp * 0.3;
    px[i * 4 + 1] = base[1] + n * 0.8;
    px[i * 4 + 2] = base[2] + n * 0.5 + rnd() * amp * 0.3;
    px[i * 4 + 3] = 255;
  }
  return px;
}

describe('colour transfer maths', () => {
  it('symEigen reconstructs a symmetric matrix', () => {
    const m = [4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2];
    const { values, vectors: v } = symEigen(m);
    const d = [values[0], 0, 0, 0, values[1], 0, 0, 0, values[2]];
    mul(mul(v, d), tr(v)).forEach((x, i) => expect(x).toBeCloseTo(m[i], 6));
  });

  it('Monge-Kantorovich matrix maps the source covariance onto the reference covariance', () => {
    const src = colorStatsFromRgba(noise(64, 64, [120, 100, 90], 60, 3));
    const ref = colorStatsFromRgba(noise(64, 64, [200, 140, 80], 30, 9));
    const M = matchMatrix(src, ref);
    const reg = (c: number[]) => c.map((v, k) => (k % 4 === 0 ? v + 1 : v));
    const mapped = mul(mul(M, reg(src.cov!)), tr(M));
    reg(ref.cov!).forEach((x, i) => expect(mapped[i]).toBeCloseTo(x, 3));
  });

  it('stats carry monotone L* quantiles and a symmetric covariance', () => {
    const s = colorStatsFromRgba(noise(32, 32, [128, 128, 128], 80));
    expect(s.lQuantiles).toHaveLength(17);
    for (let i = 1; i < 17; i++) expect(s.lQuantiles![i]).toBeGreaterThanOrEqual(s.lQuantiles![i - 1]);
    expect(s.cov![1]).toBeCloseTo(s.cov![3], 8);
    expect(Math.sqrt(s.cov![0])).toBeCloseTo(s.std[0], 6);
  });

  it('extracts the dominant palette colours', () => {
    const px = new Uint8ClampedArray(100 * 4);
    for (let i = 0; i < 100; i++) px.set(i < 70 ? [220, 40, 40, 255] : [30, 60, 200, 255], i * 4);
    const pal = extractPalette(px, 2);
    expect(pal[0].rgb).toEqual([220, 40, 40]);
    expect(pal[0].share).toBeCloseTo(0.7);
  });
});

describe('style profile', () => {
  it('finds skin and measures its texture; sky has none', () => {
    const skin = profileFromRgba(noise(64, 64, [214, 160, 130], 20), 64, 64);
    expect(skin.skinTexture).not.toBeNull();
    expect(skin.warmth).toBeGreaterThan(5);
    const sky = profileFromRgba(noise(64, 64, [90, 140, 220], 10), 64, 64);
    expect(sky.skinTexture).toBeNull();
    expect(sky.warmth).toBeLessThan(0);
    const smooth = profileFromRgba(noise(64, 64, [214, 160, 130], 4), 64, 64);
    expect(smooth.skinTexture!).toBeLessThan(skin.skinTexture!);
    expect(smooth.sharpness).toBeLessThan(skin.sharpness);
  });
});

const face: FaceAnalysis = {
  box: [0.3, 0.2, 0.4, 0.5],
  landmarkCount: 478,
  eyeOpenness: 0.8,
  regions: {
    faceOval: Array.from({ length: 36 }, (_, i) => [0.5 + 0.2 * Math.cos((i / 36) * 2 * Math.PI), 0.45 + 0.25 * Math.sin((i / 36) * 2 * Math.PI)]),
    leftEye: [], rightEye: [], leftUnderEye: [], rightUnderEye: [], innerMouth: [],
    lips: [[0.45, 0.58], [0.55, 0.58], [0.55, 0.62], [0.45, 0.62]],
    leftBrow: [], rightBrow: [],
  },
};

describe('v0.2 masks', () => {
  it('stray-hair band hugs the upper face outline and stops at the mouth', () => {
    const m = buildHairRoiMask([face], 100, 100);
    expect(m[18 * 100 + 50]).toBeGreaterThan(128); // just above the oval top (forehead/hairline)
    expect(m[45 * 100 + 50]).toBeLessThan(20); // face centre
    expect(m[90 * 100 + 50]).toBe(0); // below the mouth
    expect(m[45 * 100 + 5]).toBe(0); // far background
  });

  it('depth prior keeps the subject at 0 and the background far', () => {
    const d = buildDepthPrior([face], 100, 100);
    expect(d[45 * 100 + 50]).toBeLessThan(20);
    expect(d[10 * 100 + 3]).toBeGreaterThan(200);
    const none = buildDepthPrior([], 50, 50);
    expect(none[22 * 50 + 25]).toBeLessThan(40);
    expect(none[0]).toBeGreaterThan(200);
  });
});

describe('aperture kernels', () => {
  it('stay inside the unit disk and the hexagon has flat sides', () => {
    for (const shape of ['circle', 'hex'] as const) {
      const k = apertureKernel(48, shape);
      for (let i = 0; i < 48; i++) expect(Math.hypot(k[i * 2], k[i * 2 + 1])).toBeLessThanOrEqual(1.0001);
    }
    const hex = apertureKernel(400, 'hex');
    let maxAt30 = 0;
    for (let i = 0; i < 400; i++) {
      const a = Math.atan2(hex[i * 2 + 1], hex[i * 2]);
      if (Math.abs(Math.abs(a) - Math.PI / 6) < 0.05) maxAt30 = Math.max(maxAt30, Math.hypot(hex[i * 2], hex[i * 2 + 1]));
    }
    expect(maxAt30).toBeLessThan(0.88); // apothem cos(30deg) = 0.866
  });
});
