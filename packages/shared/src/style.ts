/**
 * Style sync: turn measurements of a reference photo into edits for other photos.
 *
 * A `StyleProfile` is a handful of numbers measured the same way in the browser
 * (apps/web/src/engine/style.ts) and on the server (services/ai/app/retouch_engine.py).
 * The mapping functions below are pure, so the editor, the batch planner and tests
 * all agree on what "make it look like the reference" means.
 *
 * Two ways to apply a reference:
 *  - `palette`: an exact statistical colour transfer (L*a*b* covariance + tone curve)
 *    carried in `colorMatch`, plus retouch intensity (skin smoothing, sharpening).
 *  - `sliders`: the same intent expressed only through ordinary, editable sliders
 *    (exposure, contrast, temperature, tint, saturation, skin, sharpen).
 */

import { clamp, type Adjustments, type ColorStats } from './index.js';

export interface StyleProfile {
  /** L*a*b* statistics including covariance and L* quantiles. */
  stats: ColorStats;
  /** Mean L* (0..100). */
  meanL: number;
  /** Standard deviation of L* (global contrast). */
  contrast: number;
  /** Mean b* (positive = warm). */
  warmth: number;
  /** Mean a* (positive = magenta). */
  tintAxis: number;
  /** Mean chroma sqrt(a*^2 + b*^2). */
  chroma: number;
  /** Variance of the Laplacian of an 8-bit grey image at 512 px long edge. */
  sharpness: number;
  /** Mean L* of skin-coloured pixels, or null when too few were found. */
  skinLuma: number | null;
  /** Mean absolute high-pass luma (0..255 scale) on skin pixels: pore/blotch energy. */
  skinTexture: number | null;
}

export type StyleSyncMode = 'palette' | 'sliders';

const r1 = (v: number) => Math.round(v);

/** Exposure slider value that moves mean L* from `fromL` to `toL` (slider = 50 * log2(linear gain)). */
export function exposureForLightness(fromL: number, toL: number): number {
  // L* ~ 116 Y^(1/3) - 16, so a linear gain k scales (L* + 16) by k^(1/3).
  const k = ((toL + 16) / Math.max(fromL + 16, 1)) ** 3;
  return clamp(50 * Math.log2(Math.max(k, 1e-3)), -100, 100);
}

/** Temperature slider shifting mean b* by `db` at lightness L (calibrated against the shader's WB gains). */
export function temperatureForWarmth(db: number, L: number): number {
  return clamp((100 * db) / (0.15 * Math.max(L, 10)), -100, 100);
}

/** Tint slider shifting mean a* by `da` at lightness L. */
export function tintForMagenta(da: number, L: number): number {
  return clamp((100 * da) / (0.078 * Math.max(L, 10)), -100, 100);
}

/** Retouch intensity implied by the reference: how smooth its skin is and how crisp the image. */
export function retouchIntensityFrom(ref: StyleProfile, target: StyleProfile): Partial<Adjustments> {
  const out: Partial<Adjustments> = {};
  if (ref.skinTexture !== null && target.skinTexture !== null && target.skinTexture > 0) {
    const ratio = ref.skinTexture / target.skinTexture;
    out.skinSmoothing = r1(clamp((1 - ratio) * 130, 0, 85));
    out.skinTexture = r1(clamp(40 + ratio * 30, 35, 80));
    // Much smoother reference skin usually means blemishes were cleaned too.
    if (ratio < 0.75) out.blemishRemoval = r1(clamp((1 - ratio) * 120, 0, 80));
    if (ratio < 0.45) out.skinMultiplier = 2;
  }
  if (target.sharpness > 0) {
    const s = ref.sharpness / target.sharpness;
    out.sharpen = r1(clamp((s - 1) * 35, 0, 60));
  }
  return out;
}

/** Tone and colour sliders that move `target`'s global look toward `ref`'s. */
export function toneSlidersFrom(ref: StyleProfile, target: StyleProfile): Partial<Adjustments> {
  const L = (ref.meanL + target.meanL) / 2;
  return {
    exposure: r1(exposureForLightness(target.meanL, ref.meanL)),
    contrast: r1(clamp(((ref.contrast / Math.max(target.contrast, 1) - 1) / 0.5) * 100, -60, 60)),
    temperature: r1(temperatureForWarmth(ref.warmth - target.warmth, L)),
    tint: r1(tintForMagenta(ref.tintAxis - target.tintAxis, L)),
    saturation: r1(clamp((ref.chroma / Math.max(target.chroma, 1) - 1) * 100, -80, 60)),
  };
}

/**
 * Edits that make `target` look like `ref`. In `palette` mode the colour transfer is
 * exact and carried by `colorMatch` (the renderer re-derives it per photo from that
 * photo's own statistics, so one patch works for a whole batch).
 */
export function mapStyleToAdjustments(
  ref: StyleProfile,
  target: StyleProfile,
  mode: StyleSyncMode = 'palette',
  amount = 100,
): Partial<Adjustments> {
  const retouch = retouchIntensityFrom(ref, target);
  if (mode === 'palette') {
    return { ...retouch, colorMatch: { reference: ref.stats, amount: clamp(amount, 0, 100) } };
  }
  const tone = toneSlidersFrom(ref, target) as Record<string, number>;
  const k = clamp(amount, 0, 100) / 100;
  const scaled = Object.fromEntries(Object.entries(tone).map(([key, v]) => [key, r1(v * k)])) as Partial<Adjustments>;
  return { ...retouch, ...scaled, colorMatch: null };
}

/**
 * Smart Sync: copy the hero photo's edits to another photo, but compensate exposure
 * and white balance for the difference between the two originals, so a batch shot in
 * changing light lands on one consistent look instead of copying the same numbers.
 */
export function smartSyncAdjustments(
  hero: Adjustments,
  heroStats: ColorStats,
  targetStats: ColorStats,
): Adjustments {
  const [hL, ha, hb] = heroStats.mean;
  const [tL, ta, tb] = targetStats.mean;
  const L = (hL + tL) / 2;
  // Only correct moderate differences; a big jump is likely a different scene, not drift.
  const dExp = clamp(exposureForLightness(tL, hL), -60, 60);
  const dTemp = clamp(temperatureForWarmth(hb - tb, L), -50, 50);
  const dTint = clamp(tintForMagenta(ha - ta, L), -50, 50);
  return {
    ...hero,
    exposure: r1(clamp(hero.exposure + dExp, -100, 100)),
    temperature: r1(clamp(hero.temperature + dTemp, -100, 100)),
    tint: r1(clamp(hero.tint + dTint, -100, 100)),
  };
}

/** A well-exposed, gently retouched portrait: the target for one-click Auto Retouch. */
export const AUTO_TARGET = {
  meanL: 56,
  contrast: 22,
  warmth: 9,
  tintAxis: 6,
  skinTexture: 2.2,
} as const;

/** Before phase 3b: most Auto Retouch's exposure could lift measured skin (L*). Kept for reference. */
export const SKIN_LIFT_MAX = 10;
/**
 * Subject this many L* brighter than the frame's mean: a lit person against a darker scene (a night or
 * low-key portrait). The frame mean is then low by design, not because the photo is underexposed.
 */
export const LOW_KEY_SUBJECT_MARGIN = 6;

/** Auto Retouch's skin brightening for each exposure step the frame alone would ask for. */
export const AUTO_SKIN_PER_EXPOSURE = 1.5;
/** Most skin brightening Auto Retouch sets (of 100): conservative, the user can raise it. */
export const AUTO_SKIN_BRIGHTEN_MAX = 35;

export interface AutoExposure {
  /** Exposure slider value Auto Retouch applies: always 0, the photo's exposure is kept. */
  exposure: number;
  /** Skin brightening (0..100) Auto Retouch applies instead: skin only, by the skin mask. */
  skinBrighten: number;
  /** The brightening was limited to AUTO_SKIN_BRIGHTEN_MAX. */
  capped: boolean;
  /** Low-key portrait: the subject is not brightened at all. */
  lowKey: boolean;
}

/**
 * Auto Retouch never changes the exposure of the whole photograph (phase 3b): the background,
 * clothing, hair and the photo's lighting stay as shot. When the frame looks underexposed and a
 * face is found, it lifts the skin only, conservatively, through the Skin brightening slider:
 *  - a low-key portrait (subject clearly brighter than the frame) gets no lift at all;
 *  - the lift follows the damped frame correction and is capped at AUTO_SKIN_BRIGHTEN_MAX;
 *  - an overexposed frame is left as it is (no darkening).
 */
export function autoExposure(p: StyleProfile, hasFace: boolean, damp = 0.6): AutoExposure {
  const raw = clamp(exposureForLightness(p.meanL, AUTO_TARGET.meanL) * damp, -60, 60);
  const none = { exposure: 0, skinBrighten: 0, capped: false, lowKey: false };
  if (raw <= 0 || p.skinLuma == null || !hasFace) return none;
  if (p.skinLuma - p.meanL >= LOW_KEY_SUBJECT_MARGIN) return { ...none, lowKey: true };
  const want = raw * AUTO_SKIN_PER_EXPOSURE;
  return want > AUTO_SKIN_BRIGHTEN_MAX
    ? { ...none, skinBrighten: AUTO_SKIN_BRIGHTEN_MAX, capped: true }
    : { ...none, skinBrighten: r1(want) };
}

/**
 * One-click Auto Retouch: conservative corrections toward AUTO_TARGET plus skin
 * work scaled to how textured the skin actually is. Corrections are damped (x0.6)
 * so the result is a starting point, never a jarring jump. Global exposure and contrast are not
 * touched (the photo's lighting, highlights, shadows and background are kept, and so are the
 * user's own Light sliders); skin brightening follows autoExposure.
 */
export function autoRetouch(p: StyleProfile, hasFace: boolean): Partial<Adjustments> {
  const damp = 0.6;
  const L = p.meanL;
  const out: Partial<Adjustments> = {
    temperature: r1(clamp(temperatureForWarmth(AUTO_TARGET.warmth - p.warmth, L) * damp * 0.5, -40, 40)),
    tint: r1(clamp(tintForMagenta(AUTO_TARGET.tintAxis - p.tintAxis, L) * damp * 0.3, -25, 25)),
    vibrance: p.chroma < 18 ? 15 : 0,
    sharpen: p.sharpness < 120 ? 20 : 10,
  };
  if (p.skinTexture !== null) {
    const rough = clamp(p.skinTexture / AUTO_TARGET.skinTexture, 0.5, 3);
    out.skinSmoothing = r1(clamp(20 + (rough - 1) * 25, 15, 60));
    out.skinTexture = 65;
    out.blemishRemoval = r1(clamp(30 + (rough - 1) * 30, 20, 80));
    out.skinToneUnify = 25;
    out.skinMultiplier = rough > 2 ? 2 : 1;
  }
  const lift = autoExposure(p, hasFace, damp).skinBrighten;
  if (lift > 0) out.skinBrighten = lift;
  if (hasFace) {
    out.eyeBrighten = 20;
    out.darkCircles = 30;
    out.teethWhiten = 15;
  }
  return out;
}
