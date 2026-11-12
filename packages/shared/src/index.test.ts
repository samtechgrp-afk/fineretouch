import { describe, expect, it } from 'vitest';
import {
  BUILT_IN_PRESETS,
  DEFAULT_ADJUSTMENTS,
  GLASSES_CONFIDENT,
  glareFaces,
  HSL_CHANNELS,
  SLIDERS,
  applyPreset,
  isDefault,
  sanitizeAdjustments,
  type Adjustments,
} from './index';

describe('sanitizeAdjustments', () => {
  it('fills defaults, clamps, and drops unknown keys and bad types', () => {
    const a = sanitizeAdjustments({ exposure: 500, skinSmoothing: -5, contrast: '20', evil: 1, tint: NaN });
    expect(a.exposure).toBe(100);
    expect(a.skinSmoothing).toBe(0);
    expect(a.contrast).toBe(0);
    expect(a.tint).toBe(0);
    expect('evil' in a).toBe(false);
  });

  it('validates colour match payloads', () => {
    const ok = sanitizeAdjustments({ colorMatch: { reference: { mean: [50, 1, 2], std: [20, 5, 5] }, amount: 150 } });
    expect(ok.colorMatch?.amount).toBe(100);
    expect(sanitizeAdjustments({ colorMatch: { reference: { mean: [1, 2] } } }).colorMatch).toBeNull();
  });

  it('handles garbage input', () => {
    expect(sanitizeAdjustments(null)).toEqual(DEFAULT_ADJUSTMENTS);
    expect(isDefault(sanitizeAdjustments('x'))).toBe(true);
  });
});

describe('presets', () => {
  it('only use known slider keys within range', () => {
    const keys = new Set(SLIDERS.map((s) => s.key));
    for (const p of BUILT_IN_PRESETS) {
      for (const [k, v] of Object.entries(p.adjustments)) {
        if (k === 'hsl' || k === 'colorGrading') {
          // Structured settings must survive sanitising unchanged (i.e. be complete and in range).
          expect(sanitizeAdjustments({ [k]: v })[k as 'hsl' | 'colorGrading']).toEqual(v);
          continue;
        }
        expect(keys.has(k as never)).toBe(true);
        const spec = SLIDERS.find((s) => s.key === k)!;
        expect(v).toBeGreaterThanOrEqual(spec.min);
        expect(v).toBeLessThanOrEqual(spec.max);
      }
    }
  });

  it('merge keeps unrelated edits, replace resets them', () => {
    const cur = sanitizeAdjustments({ exposure: 30 });
    const p = BUILT_IN_PRESETS[0];
    expect(applyPreset(cur, p, 'merge').exposure).toBe(30);
    expect(applyPreset(cur, p, 'replace').exposure).toBe(0);
  });
});

describe('v0.5 colour mixer, grading and spot healing settings', () => {
  it('defaults are neutral and keep beauty marks', () => {
    const d = sanitizeAdjustments({});
    expect(d.protectBeautyMarks).toBe(true);
    expect(d.skinToneGuard).toBe(true);
    expect(Object.keys(d.hsl)).toEqual([...HSL_CHANNELS]);
    expect(d.colorGrading.blending).toBe(50);
    expect(isDefault(d)).toBe(true);
  });

  it('clamps and fills the 8-channel mixer', () => {
    const a = sanitizeAdjustments({ hsl: { red: { hue: 400, saturation: -20 }, blue: 'x', neon: { hue: 1 } } });
    expect(a.hsl.red).toEqual({ hue: 100, saturation: -20, luminance: 0 });
    expect(a.hsl.blue).toEqual({ hue: 0, saturation: 0, luminance: 0 });
    expect('neon' in a.hsl).toBe(false);
    expect(isDefault(a)).toBe(false);
  });

  it('clamps grading wheels, blending and balance', () => {
    const g = sanitizeAdjustments({ colorGrading: { shadows: { hue: 720, saturation: 150, luminance: -300 }, blending: -5, balance: 9e9 } }).colorGrading;
    expect(g.shadows).toEqual({ hue: 360, saturation: 100, luminance: -100 });
    expect(g.blending).toBe(0);
    expect(g.balance).toBe(100);
    expect(g.midtones).toEqual({ hue: 0, saturation: 0, luminance: 0 });
  });

  it('a grading wheel with hue but no saturation is still neutral', () => {
    expect(isDefault(sanitizeAdjustments({ colorGrading: { highlights: { hue: 40 } } }))).toBe(true);
  });

  it('beauty-mark protection is a plain boolean', () => {
    expect(sanitizeAdjustments({ protectBeautyMarks: false }).protectBeautyMarks).toBe(false);
    expect(sanitizeAdjustments({ skinToneGuard: false }).skinToneGuard).toBe(false);
    expect(sanitizeAdjustments({ skinToneGuard: 'no' } as unknown as Partial<Adjustments>).skinToneGuard).toBe(true);
    expect(sanitizeAdjustments({ protectBeautyMarks: 0 } as unknown as Partial<Adjustments>).protectBeautyMarks).toBe(true);
  });
});

describe('phase 3b: glasses gate and skin brightening', () => {
  const ring = (cx: number) => Array.from({ length: 8 }, (_, i) => [cx + 0.03 * Math.cos(i), 0.4 + 0.01 * Math.sin(i)] as [number, number]);
  const face = (glasses?: number | null) => ({
    box: [0.2, 0.1, 0.6, 0.8] as [number, number, number, number], landmarkCount: 478, eyeOpenness: 1, glasses,
    regions: { faceOval: [], leftEye: ring(0.6), rightEye: ring(0.4), leftUnderEye: [], rightUnderEye: [], innerMouth: [], lips: [], leftBrow: [], rightBrow: [] },
  });
  it('glare runs only on confidently detected glasses unless forced', () => {
    expect(glareFaces([face(0.9), face(0.3), face(null), face()], false)).toHaveLength(1);
    expect(glareFaces([face(0.9), face(0.3), face(null), face()], true)).toHaveLength(4);
    expect(GLASSES_CONFIDENT).toBe(0.6);
  });
  it('defaults are off and survive sanitising', () => {
    expect(DEFAULT_ADJUSTMENTS.glareForce).toBe(false);
    expect(DEFAULT_ADJUSTMENTS.skinBrighten).toBe(0);
    expect(sanitizeAdjustments({ glareForce: true }).glareForce).toBe(true);
    expect(sanitizeAdjustments({ glareForce: 1 } as unknown as Partial<Adjustments>).glareForce).toBe(false);
    expect(sanitizeAdjustments({ skinBrighten: 250 }).skinBrighten).toBe(100);
  });
});
