import { DEFAULT_COLOR_GRADING, DEFAULT_HSL, sanitizeColorGrading, sanitizeHsl } from '@fineretouch/shared';
import { describe, expect, it } from 'vitest';
import { applyGradingPixel, applyHslPixel, gradeBounds, hsvToRgb, hueSegment, rgbToHsv, wheelOffset, zoneWeights, type RGB } from './colorMixer';
import { RETOUCH_FRAG } from './shaders';

const swatch = (h: number, s = 0.8, v = 0.8): RGB => hsvToRgb(h, s, v);

describe('colour mixer reference (twin of color_engine.py)', () => {
  it('round-trips HSV', () => {
    for (const c of [[0.2, 0.5, 0.9], [0.9, 0.1, 0.1], [0.5, 0.5, 0.5], [0.3, 0.8, 0.2]] as RGB[]) {
      const [h, s, v] = rgbToHsv(c);
      hsvToRgb(h, s, v).forEach((x, i) => expect(x).toBeCloseTo(c[i], 5));
    }
  });

  it('splits hue between neighbouring channels', () => {
    expect(hueSegment(90)).toEqual([2, 3, 0.5]);
    expect(hueSegment(0)).toEqual([0, 1, 0]);
    const [lo, hi, t] = hueSegment(330);
    expect([lo, hi]).toEqual([7, 0]);
    expect(t).toBeCloseTo(0.5);
  });

  it('shifts green toward aqua by 30 degrees at +100', () => {
    const hsl = sanitizeHsl({ green: { hue: 100 } });
    expect(rgbToHsv(applyHslPixel(swatch(120), hsl))[0]).toBeCloseTo(150, 0);
  });

  it('desaturates one band and leaves others and greys alone', () => {
    const hsl = sanitizeHsl({ blue: { saturation: -100, luminance: -100 } });
    const out = applyHslPixel(swatch(240), hsl);
    expect(Math.max(...out) - Math.min(...out)).toBeLessThan(1e-3);
    applyHslPixel(swatch(0), hsl).forEach((v, i) => expect(v).toBeCloseTo(swatch(0)[i], 9));
    expect(applyHslPixel([0.5, 0.5, 0.5], hsl)).toEqual([0.5, 0.5, 0.5]);
  });

  it('identity settings return the input', () => {
    const c: RGB = [0.3, 0.6, 0.2];
    expect(applyHslPixel(c, DEFAULT_HSL)).toBe(c);
    expect(applyGradingPixel(c, DEFAULT_COLOR_GRADING)).toBe(c);
  });

  it('grading zones, balance and zero-luma tints', () => {
    const g = sanitizeColorGrading({ shadows: { hue: 220, saturation: 60 }, highlights: { hue: 40, saturation: 60 } });
    expect(zoneWeights(0.05, g)[0]).toBeGreaterThan(0.99);
    expect(zoneWeights(0.5, g)[1]).toBeGreaterThan(0.99);
    expect(zoneWeights(0.95, g)[2]).toBeGreaterThan(0.99);
    const [lo, hi] = gradeBounds(sanitizeColorGrading({ balance: 100 }));
    expect(lo).toBeCloseTo(0.05);
    expect(hi).toBeCloseTo(0.55);
    const off = wheelOffset(45, 100, 0);
    expect(off[0] * 0.299 + off[1] * 0.587 + off[2] * 0.114).toBeCloseTo(0, 6);
    const dark = applyGradingPixel([0.12, 0.12, 0.12], g);
    const bright = applyGradingPixel([0.9, 0.9, 0.9], g);
    expect(dark[2]).toBeGreaterThan(dark[0]);
    expect(bright[0]).toBeGreaterThan(bright[2]);
  });

  it('matches services/ai/app/color_engine.py on reference colours', () => {
    // Expected values printed by color_engine.apply_color_engine with these settings.
    const hsl = sanitizeHsl({
      red: { hue: 20, saturation: -40, luminance: 10 }, orange: { hue: -30, saturation: 25, luminance: -20 }, green: { hue: 60, saturation: -80 },
      blue: { saturation: -100, luminance: 30 }, purple: { hue: -50, luminance: -40 }, aqua: { hue: 45, saturation: 50 },
    });
    const g = sanitizeColorGrading({
      shadows: { hue: 200, saturation: 40, luminance: -10 }, midtones: { hue: 30, saturation: 15 }, highlights: { hue: 50, saturation: 30, luminance: 5 },
      blending: 70, balance: -20,
    });
    const cases: Array<[RGB, RGB]> = [
      [[0.8, 0.3, 0.25], [0.64651, 0.38797, 0.31754]],
      [[0.9, 0.6, 0.2], [1.0, 0.46252, 0.01323]],
      [[0.3, 0.7, 0.35], [0.50582, 0.5668, 0.49603]],
      [[0.15, 0.2, 0.5], [0.18822, 0.23923, 0.27347]],
      [[0.5, 0.3, 0.7], [0.366, 0.27495, 0.64008]],
      [[0.2, 0.75, 0.8], [0.01834, 0.78545, 0.97334]],
      [[0.5, 0.5, 0.5], [0.51834, 0.49584, 0.47334]],
      [[0.05, 0.06, 0.1], [0.0, 0.06067, 0.10522]],
      [[0.95, 0.9, 0.85], [0.99125, 0.90233, 0.77785]],
      [[0.6, 0.45, 0.4], [0.60562, 0.44278, 0.37954]],
    ];
    for (const [input, expected] of cases) {
      // Python bins hue in 0.1 degree steps; allow for that and float32.
      applyGradingPixel(applyHslPixel(input, hsl), g).forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(4e-3));
    }
  });

  it('desaturating keeps brightness (luma)', () => {
    const navy: RGB = [0.1, 0.13, 0.3];
    const out = applyHslPixel(navy, sanitizeHsl({ blue: { saturation: -100 } }));
    const y = (c: RGB) => c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114;
    expect(y(out)).toBeCloseTo(y(navy), 4);
  });

  it('the shader carries the same channel centres and ranges', () => {
    expect(RETOUCH_FRAG).toContain('float[9](0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0, 360.0)');
    expect(RETOUCH_FRAG).toContain('k.x * 30.0 * g');
    expect(RETOUCH_FRAG).toContain('1.0 + 0.6 * k.z * g * hsv.y');
  });
});
