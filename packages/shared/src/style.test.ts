import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ADJUSTMENTS,
  autoRetouch,
  exposureForLightness,
  isDefault,
  mapStyleToAdjustments,
  sanitizeAdjustments,
  smartSyncAdjustments,
  type StyleProfile,
} from './index';

const profile = (over: Partial<StyleProfile> = {}): StyleProfile => ({
  stats: { mean: [55, 8, 12], std: [20, 6, 8] },
  meanL: 55,
  contrast: 20,
  warmth: 12,
  tintAxis: 8,
  chroma: 20,
  sharpness: 200,
  skinLuma: 62,
  skinTexture: 3,
  ...over,
});

describe('v0.2 adjustments', () => {
  it('defaults are off and sanitise nested settings', () => {
    expect(isDefault(DEFAULT_ADJUSTMENTS)).toBe(true);
    const a = sanitizeAdjustments({
      skinMultiplier: 5,
      strayHairRemoval: { enabled: 'yes', strength: 250 },
      depthBokeh: { enabled: true, blurRadius: -4, bokehMultiplier: 900, apertureShape: 'star', highlightBoost: 30 },
    });
    expect(a.skinMultiplier).toBe(1);
    expect(a.strayHairRemoval).toEqual({ enabled: false, strength: 100 });
    expect(a.depthBokeh).toEqual({ enabled: true, blurRadius: 0, bokehMultiplier: 300, apertureShape: 'circle', highlightBoost: 30 });
    expect(isDefault(a)).toBe(false);
    expect(sanitizeAdjustments({ skinMultiplier: 3 }).skinMultiplier).toBe(3);
  });

  it('keeps optional covariance and quantiles on colour match references', () => {
    const q = Array.from({ length: 17 }, (_, i) => i * 6);
    const a = sanitizeAdjustments({
      colorMatch: { reference: { mean: [50, 0, 0], std: [20, 5, 5], cov: [400, 0, 0, 0, 25, 0, 0, 0, 25], lQuantiles: q }, amount: 80 },
    });
    expect(a.colorMatch?.reference.cov).toHaveLength(9);
    expect(a.colorMatch?.reference.lQuantiles).toEqual(q);
    const bad = sanitizeAdjustments({ colorMatch: { reference: { mean: [50, 0, 0], std: [20, 5, 5], cov: [1, 2] } } });
    expect(bad.colorMatch?.reference.cov).toBeUndefined();
  });
});

describe('style mapping', () => {
  it('exposure maps lightness in both directions and is zero for equal images', () => {
    expect(exposureForLightness(50, 50)).toBeCloseTo(0);
    expect(exposureForLightness(40, 60)).toBeGreaterThan(20);
    expect(exposureForLightness(60, 40)).toBeLessThan(-20);
  });

  it('palette mode carries the reference stats; sliders mode moves tone toward the reference', () => {
    const ref = profile({ meanL: 65, warmth: 20, contrast: 26, chroma: 26, skinTexture: 1.2 });
    const tgt = profile();
    const pal = mapStyleToAdjustments(ref, tgt, 'palette', 70);
    expect(pal.colorMatch).toEqual({ reference: ref.stats, amount: 70 });
    expect(pal.skinSmoothing).toBeGreaterThan(40); // smoother reference skin
    expect(pal.skinMultiplier).toBe(2);
    const sl = mapStyleToAdjustments(ref, tgt, 'sliders');
    expect(sl.colorMatch).toBeNull();
    expect(sl.exposure).toBeGreaterThan(0);
    expect(sl.temperature).toBeGreaterThan(0);
    expect(sl.contrast).toBeGreaterThan(0);
    expect(sl.saturation).toBeGreaterThan(0);
    for (const v of Object.values(sanitizeAdjustments({ ...DEFAULT_ADJUSTMENTS, ...sl }))) expect(v).not.toBeNaN();
  });

  it('smart sync compensates a darker, cooler frame and keeps the rest of the hero edit', () => {
    const hero = sanitizeAdjustments({ skinSmoothing: 60, exposure: 10, temperature: 5 });
    const out = smartSyncAdjustments(hero, { mean: [60, 5, 15], std: [20, 5, 5] }, { mean: [50, 5, 8], std: [20, 5, 5] });
    expect(out.exposure).toBeGreaterThan(10);
    expect(out.temperature).toBeGreaterThan(5);
    expect(out.skinSmoothing).toBe(60);
    expect(smartSyncAdjustments(hero, { mean: [60, 5, 15], std: [1, 1, 1] }, { mean: [60, 5, 15], std: [1, 1, 1] })).toEqual(hero);
  });

  it('auto retouch stays within gentle bounds', () => {
    const dark = autoRetouch(profile({ meanL: 30, skinTexture: 9 }), true);
    expect(dark.exposure).toBeGreaterThan(0);
    expect(dark.exposure).toBeLessThanOrEqual(60);
    expect(dark.skinMultiplier).toBe(2);
    expect(dark.eyeBrighten).toBe(20);
    const noSkin = autoRetouch(profile({ skinTexture: null, skinLuma: null }), false);
    expect(noSkin.skinSmoothing).toBeUndefined();
    expect(noSkin.eyeBrighten).toBeUndefined();
  });
});
