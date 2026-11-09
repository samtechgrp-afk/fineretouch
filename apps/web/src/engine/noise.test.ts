import { describe, expect, it } from 'vitest';
import { estimateNoiseLevel } from './noise';

function image(w: number, h: number, f: (x: number, y: number, c: number) => number): Uint8Array {
  const d = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) d[(y * w + x) * 4 + c] = Math.max(0, Math.min(255, Math.round(f(x, y, c))));
      d[(y * w + x) * 4 + 3] = 255;
    }
  return d;
}

// Deterministic pseudo-random normal noise.
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const SKIN = [204, 153, 128];

describe('noise level estimate', () => {
  it('rises with per-channel noise', () => {
    const clean = estimateNoiseLevel(image(96, 96, (_x, _y, c) => SKIN[c]), 96, 96);
    const noisy = estimateNoiseLevel(image(96, 96, (_x, _y, c) => SKIN[c] + 6 * gauss()), 96, 96);
    expect(clean).toBe(0.002);
    expect(noisy).toBeGreaterThan(5 * clean);
  });

  it('is not raised by luma-only texture such as pores', () => {
    const tex = Array.from({ length: 96 * 96 }, () => 14 * gauss());
    const pores = estimateNoiseLevel(image(96, 96, (x, y, c) => SKIN[c] + tex[y * 96 + x]), 96, 96);
    const noisy = estimateNoiseLevel(image(96, 96, (_x, _y, c) => SKIN[c] + 6 * gauss()), 96, 96);
    expect(pores).toBeLessThan(0.3 * noisy);
  });
});
