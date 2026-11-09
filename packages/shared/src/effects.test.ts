import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ADJUSTMENTS,
  DEFAULT_EFFECTS,
  EFFECT_CURVES,
  EFFECT_IDS,
  boostStrength,
  isDefault,
  resolveAdjustments,
  sanitizeAdjustments,
  type Adjustments,
} from './index.js';

/** Same vectors as services/ai/tests/test_effects.py: the two resolvers must agree. */
export const CURVE_VECTORS: Array<[number, number, number, number, number]> = [
  // value, multiplier, knee, cap, expected
  [30, 1, 0.8, 100, 30],
  [30, 2, 0.8, 100, 60],
  [30, 3, 0.8, 100, 87.86938680575],
  [40, 2, 0.8, 100, 80],
  [40, 3, 0.8, 100, 97.29329433527],
  [90, 1, 0.8, 100, 90],
  [90, 2, 0.8, 100, 99.86524106002],
  [50, 3, 0.6, 85, 83.15109084593],
  [95, 2, 0.6, 85, 95],
  [0, 3, 0.8, 100, 0],
];

const adj = (patch: Partial<Adjustments>): Adjustments => sanitizeAdjustments({ ...DEFAULT_ADJUSTMENTS, ...patch });

describe('effect strength curves', () => {
  it.each(CURVE_VECTORS)('boost(%d, %dx, knee %d, cap %d) = %d', (v, m, knee, cap, want) => {
    expect(boostStrength(v, m, { knee, cap })).toBeCloseTo(want, 8);
  });

  it('matches the spec examples: 30% -> ~60% at 2x and ~90% at 3x; 40% at 3x caps near 100%', () => {
    const c = EFFECT_CURVES.skinSmoothing;
    expect(boostStrength(30, 2, c)).toBe(60);
    expect(boostStrength(30, 3, c)).toBeGreaterThan(85);
    expect(boostStrength(40, 3, c)).toBeGreaterThan(95);
    expect(boostStrength(40, 3, c)).toBeLessThanOrEqual(100);
  });

  it('is monotonic in both slider and multiplier, and never exceeds the cap when boosted', () => {
    for (const id of EFFECT_IDS) {
      const c = EFFECT_CURVES[id];
      let prev = -1;
      for (let v = 0; v <= 100; v += 5) {
        const b1 = boostStrength(v, 1, c), b2 = boostStrength(v, 2, c), b3 = boostStrength(v, 3, c);
        expect(b1).toBe(v);
        expect(b2).toBeGreaterThanOrEqual(b1);
        expect(b3).toBeGreaterThanOrEqual(b2);
        expect(b3).toBeLessThanOrEqual(Math.max(v, c.cap));
        expect(b2).toBeGreaterThanOrEqual(prev);
        prev = b2;
      }
    }
  });
});

describe('resolveAdjustments', () => {
  it('leaves old edits (no effect controls) exactly as they were', () => {
    const a = adj({ skinSmoothing: 42, blemishRemoval: 70, sharpen: 33, strayHairRemoval: { enabled: true, strength: 55 } });
    const r = resolveAdjustments(a);
    expect(r.skinSmoothing).toBe(42);
    expect(r.blemishRemoval).toBe(70);
    expect(r.sharpen).toBe(33);
    expect(r.strayHairRemoval.strength).toBe(55);
    expect(r.spotMultiplier).toBe(1);
  });

  it('applies multiplier, opacity, enable and global strength per effect only', () => {
    const a = adj({
      skinSmoothing: 30, darkCircles: 40, sharpen: 50, globalStrength: 50,
      effects: { ...DEFAULT_EFFECTS, skinSmoothing: { enabled: true, multiplier: 2, opacity: 100 }, darkCircles: { enabled: true, multiplier: 1, opacity: 50 }, sharpen: { enabled: false, multiplier: 3, opacity: 100 } },
    });
    const r = resolveAdjustments(a);
    expect(r.skinSmoothing).toBeCloseTo(30, 9); // 60 at 2x, halved by global strength
    expect(r.darkCircles).toBeCloseTo(10, 9); // 40 * 50% opacity * 50% global
    expect(r.sharpen).toBe(0); // disabled
    expect(r.exposure).toBe(0); // light / colour untouched by global strength
    expect(a.skinSmoothing).toBe(30); // the user's slider is not rewritten
    expect(r.effects).toEqual(DEFAULT_EFFECTS);
    expect(r.globalStrength).toBe(100);
  });

  it('passes the spot healing multiplier to the detector, and keeps the skin boost', () => {
    const on = adj({ blemishRemoval: 40, effects: { ...DEFAULT_EFFECTS, blemishRemoval: { enabled: true, multiplier: 3, opacity: 100 } } });
    expect(resolveAdjustments(on).spotMultiplier).toBe(3);
    const boost = adj({ blemishRemoval: 40, skinMultiplier: 2 });
    expect(resolveAdjustments(boost).spotMultiplier).toBe(2);
  });

  it('scales structured effects (stray hair, clothing iron)', () => {
    const a = adj({
      strayHairRemoval: { enabled: true, strength: 30 },
      clothing: { ...DEFAULT_ADJUSTMENTS.clothing, enabled: true, wrinkleStrength: 30 },
      effects: { ...DEFAULT_EFFECTS, strayHair: { enabled: true, multiplier: 2, opacity: 100 }, clothingIron: { enabled: false, multiplier: 1, opacity: 100 } },
    });
    const r = resolveAdjustments(a);
    expect(r.strayHairRemoval.strength).toBe(60);
    expect(r.clothing.wrinkleStrength).toBe(0);
    expect(a.strayHairRemoval.strength).toBe(30);
  });
});

describe('sanitising effect controls', () => {
  it('fills defaults, clamps opacity and rejects bad multipliers', () => {
    const a = sanitizeAdjustments({ effects: { sharpen: { multiplier: 5, opacity: 250, enabled: 'yes' } }, globalStrength: -4 });
    expect(a.effects.sharpen).toEqual({ enabled: true, multiplier: 1, opacity: 100 });
    expect(a.effects.skinSmoothing).toEqual(DEFAULT_EFFECTS.skinSmoothing);
    expect(a.globalStrength).toBe(0);
  });

  it('counts effect controls and global strength in isDefault', () => {
    expect(isDefault(DEFAULT_ADJUSTMENTS)).toBe(true);
    expect(isDefault(adj({ globalStrength: 80 }))).toBe(false);
    expect(isDefault(adj({ effects: { ...DEFAULT_EFFECTS, sharpen: { enabled: true, multiplier: 2, opacity: 100 } } }))).toBe(false);
  });
});
