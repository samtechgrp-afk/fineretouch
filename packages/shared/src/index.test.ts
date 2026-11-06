import { describe, expect, it } from 'vitest';
import { BUILT_IN_PRESETS, DEFAULT_ADJUSTMENTS, SLIDERS, applyPreset, isDefault, sanitizeAdjustments } from './index';

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
