/**
 * Per-effect retouch controls (v0.7): enable, 1x / 2x / 3x multiplier, opacity, reset, and a
 * global retouch strength.
 *
 * The user's slider values are never rewritten. Instead, every renderer first calls
 * `resolveAdjustments()`, which turns (slider, multiplier, opacity, global strength, enabled)
 * into the *effective* engine value of each effect, then renders those. The same function is
 * mirrored in services/ai/app/effects.py, and both are pinned to the same numbers by tests
 * (packages/shared/src/effects.test.ts and services/ai/tests/test_effects.py).
 *
 * The multiplier scales the effect's *parameter*, not pixel values, through a per-effect
 * response curve with a soft knee and a safety cap:
 *
 *   x    = slider * multiplier
 *   soft = x                                                   if x <= knee * cap
 *        = k + (cap - k) * (1 - exp(-(x - k) / (cap - k)))     otherwise (k = knee * cap)
 *   boosted = max(slider, soft)        (so 1x is exactly the slider: old edits render unchanged)
 *   effective = enabled ? boosted * opacity / 100 * globalStrength / 100 : 0
 *
 * Each engine stage then interprets its value in its own way (spot healing widens detection,
 * smoothing works on the tone layer only so pores are re-injected, sharpening is capped lower
 * to avoid halos). Spot healing additionally passes its multiplier to the detector
 * (`spotMultiplier`), so 2x / 3x find fainter and larger blemishes, not just repair harder.
 */

import type { Adjustments } from './index.js';

export type EffectMultiplier = 1 | 2 | 3;
export const EFFECT_MULTIPLIERS: readonly EffectMultiplier[] = [1, 2, 3];

export const EFFECT_IDS = [
  'skinSmoothing',
  'blemishRemoval',
  'skinToneUnify',
  'darkCircles',
  'eyeBrighten',
  'teethWhiten',
  'teethStainRemoval',
  'glareReduction',
  'sharpen',
  'strayHair',
  'clothingIron',
] as const;
export type EffectId = (typeof EFFECT_IDS)[number];

export interface EffectControl {
  /** Off = the effect contributes nothing, but its slider value is kept. */
  enabled: boolean;
  multiplier: EffectMultiplier;
  /** 0..100: share of the effect's contribution that is applied. */
  opacity: number;
}

export type EffectControls = Record<EffectId, EffectControl>;

export interface EffectCurve {
  label: string;
  /** Effective value at which the curve starts to bend (share of cap). */
  knee: number;
  /** Maximum safe effective value a multiplier can reach (slider units, <= 100). */
  cap: number;
  /** Why this curve: shown as a tooltip next to the multiplier. */
  note: string;
}

/** Keep in sync with EFFECT_CURVES in services/ai/app/effects.py (pinned by tests). */
export const EFFECT_CURVES: Readonly<Record<EffectId, EffectCurve>> = Object.freeze({
  skinSmoothing: { label: 'Smoothing', knee: 0.8, cap: 100, note: 'Evens the tone layer harder; pores come back through Texture retention.' },
  blemishRemoval: { label: 'Spot healing', knee: 0.85, cap: 100, note: '2x / 3x also widen detection to fainter and larger spots.' },
  skinToneUnify: { label: 'Even skin tone', knee: 0.75, cap: 100, note: 'Pulls more chroma toward the local skin average; the tone guard keeps the undertone.' },
  darkCircles: { label: 'Dark circles', knee: 0.75, cap: 100, note: 'Lifts more of the under-eye shadow, never above the cheek.' },
  eyeBrighten: { label: 'Brighten eyes', knee: 0.6, cap: 85, note: 'Capped at 85 so eyes never glow.' },
  teethWhiten: { label: 'Whiten teeth', knee: 0.6, cap: 85, note: 'Capped at 85 so teeth keep their enamel colour.' },
  teethStainRemoval: { label: 'Teeth stains', knee: 0.8, cap: 100, note: 'Evens more stains toward the teeth\'s own colour.' },
  glareReduction: { label: 'Glasses glare', knee: 0.85, cap: 100, note: 'Removes fainter reflections.' },
  sharpen: { label: 'Sharpen', knee: 0.6, cap: 90, note: 'Bends early and caps at 90 to avoid halos.' },
  strayHair: { label: 'Stray hair', knee: 0.8, cap: 100, note: 'Removes fainter strands.' },
  clothingIron: { label: 'Clothing iron', knee: 0.8, cap: 100, note: 'Irons deeper folds; weave is kept by Texture preservation.' },
});

export const DEFAULT_EFFECT_CONTROL: Readonly<EffectControl> = Object.freeze({ enabled: true, multiplier: 1, opacity: 100 });
export const DEFAULT_EFFECTS: Readonly<EffectControls> = Object.freeze(
  Object.fromEntries(EFFECT_IDS.map((id) => [id, DEFAULT_EFFECT_CONTROL])) as EffectControls,
);

/** The slider an effect reads, as [top-level key] or [structured key, field]. */
export const EFFECT_SOURCE: Readonly<Record<EffectId, readonly [keyof Adjustments] | readonly [keyof Adjustments, string]>> = Object.freeze({
  skinSmoothing: ['skinSmoothing'],
  blemishRemoval: ['blemishRemoval'],
  skinToneUnify: ['skinToneUnify'],
  darkCircles: ['darkCircles'],
  eyeBrighten: ['eyeBrighten'],
  teethWhiten: ['teethWhiten'],
  teethStainRemoval: ['teethStainRemoval'],
  glareReduction: ['glareReduction'],
  sharpen: ['sharpen'],
  strayHair: ['strayHairRemoval', 'strength'],
  clothingIron: ['clothing', 'wrinkleStrength'],
});

/** Effect id driven by a plain slider key, if any. */
export const effectForSlider = (key: string): EffectId | null =>
  (EFFECT_IDS as readonly string[]).includes(key) ? (key as EffectId) : null;

/** Slider * multiplier through the effect's soft-knee curve; 1x returns the slider exactly. */
export function boostStrength(value: number, multiplier: number, curve: Pick<EffectCurve, 'knee' | 'cap'>): number {
  const s = Math.min(100, Math.max(0, value));
  if (multiplier <= 1 || s === 0) return s;
  const x = s * multiplier;
  const k = curve.knee * curve.cap;
  const room = curve.cap - k;
  const soft = x <= k || room <= 0 ? Math.min(x, curve.cap) : k + room * (1 - Math.exp(-(x - k) / room));
  return Math.max(s, soft);
}

/** Effective engine value of one effect (0 when disabled). */
export function effectiveStrength(value: number, control: EffectControl, curve: EffectCurve, globalStrength: number): number {
  if (!control.enabled) return 0;
  return (boostStrength(value, control.multiplier, curve) * control.opacity * globalStrength) / 10000;
}

export function sanitizeEffects(input: unknown): EffectControls {
  const src = input && typeof input === 'object' ? (input as Record<string, unknown>) : {};
  const out = {} as EffectControls;
  for (const id of EFFECT_IDS) {
    const r = src[id] && typeof src[id] === 'object' ? (src[id] as Record<string, unknown>) : {};
    const m = r.multiplier;
    const o = r.opacity;
    out[id] = {
      enabled: typeof r.enabled === 'boolean' ? r.enabled : true,
      multiplier: m === 2 || m === 3 ? m : 1,
      opacity: typeof o === 'number' && Number.isFinite(o) ? Math.min(100, Math.max(0, o)) : 100,
    };
  }
  return out;
}

export function effectIsDefault(c: EffectControl): boolean {
  return c.enabled && c.multiplier === 1 && c.opacity === 100;
}

export function effectsAreDefault(e: EffectControls): boolean {
  return EFFECT_IDS.every((id) => effectIsDefault(e[id]));
}

export interface ResolvedAdjustments extends Adjustments {
  /** Spot detector reach: max(skin boost, spot healing multiplier) when spot healing is on. */
  spotMultiplier: EffectMultiplier;
}

/**
 * The adjustments the engines actually render: each effect's slider replaced by its effective
 * value, controls reset to neutral and global strength folded in. Pure and idempotent on values.
 */
export function resolveAdjustments(adj: Adjustments): ResolvedAdjustments {
  const effects = adj.effects ?? DEFAULT_EFFECTS;
  const global = typeof adj.globalStrength === 'number' ? Math.min(100, Math.max(0, adj.globalStrength)) : 100;
  const out = { ...adj, strayHairRemoval: { ...adj.strayHairRemoval }, clothing: { ...adj.clothing } } as ResolvedAdjustments;
  for (const id of EFFECT_IDS) {
    const control = effects[id] ?? DEFAULT_EFFECT_CONTROL;
    const src = EFFECT_SOURCE[id];
    if (src.length === 1) {
      const key = src[0] as 'skinSmoothing';
      out[key] = effectiveStrength(adj[key], control, EFFECT_CURVES[id], global);
    } else {
      const [key, field] = src as ['strayHairRemoval', 'strength'];
      const holder = out[key] as unknown as Record<string, number>;
      holder[field] = effectiveStrength(holder[field], control, EFFECT_CURVES[id], global);
    }
  }
  const spot = effects.blemishRemoval ?? DEFAULT_EFFECT_CONTROL;
  out.spotMultiplier = Math.max(adj.skinMultiplier, spot.enabled ? spot.multiplier : 1) as EffectMultiplier;
  out.effects = DEFAULT_EFFECTS;
  out.globalStrength = 100;
  return out;
}
