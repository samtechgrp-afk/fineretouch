import { describe, expect, it } from 'vitest';
import { colorStatsFromRgba, rgb8ToLab } from './color';

describe('L*a*b* conversion (must match services/ai/app/color.py)', () => {
  it('matches reference values', () => {
    const white = rgb8ToLab(255, 255, 255);
    expect(white[0]).toBeCloseTo(100, 1);
    expect(Math.abs(white[1])).toBeLessThan(0.05);
    const red = rgb8ToLab(255, 0, 0);
    expect(red[0]).toBeCloseTo(53.24, 1);
    expect(red[1]).toBeCloseTo(80.09, 1);
    expect(red[2]).toBeCloseTo(67.2, 1);
  });

  it('computes mean and std over pixels', () => {
    const px = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]);
    const s = colorStatsFromRgba(px);
    expect(s.mean[0]).toBeCloseTo(50, 1);
    expect(s.std[0]).toBeCloseTo(50, 1);
  });
});
