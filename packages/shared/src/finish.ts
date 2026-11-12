/**
 * Professional Skin Finish (v0.14). Mirrors SkinFinishSettings / FinishControl in
 * services/ai/app/schemas.py; the engine is services/ai/app/engines/finish.py (texture finish,
 * phase 1) and finish_local.py (local corrections, phase 2). It runs in the local AI engine only:
 * the browser shows the engine's finished photo as its source, like Skin Perfection.
 *
 * Every effect has its own switch and amount; `strength` (Overall) scales all of them. A style
 * (Natural, Professional, Advanced) sets every amount at once; moving any control after that
 * makes the style "custom".
 */

export const FINISH_CONTROLS = [
  'smoothing',
  'texturePreservation',
  'porePreservation',
  'shine',
  'darkSpots',
  'toneEvenness',
  'underEye',
  'dodgeBurn',
] as const;
export type FinishControlName = (typeof FINISH_CONTROLS)[number];

export const FINISH_CONTROL_LABELS: Record<FinishControlName, string> = {
  smoothing: 'Skin smoothing',
  texturePreservation: 'Texture preservation',
  porePreservation: 'Pore preservation',
  shine: 'Shine and glare reduction',
  darkSpots: 'Dark spots and post-acne marks',
  toneEvenness: 'Skin tone evenness',
  underEye: 'Under-eye correction',
  dodgeBurn: 'Dodge and burn',
};

export const FINISH_CONTROL_HINTS: Record<FinishControlName, string> = {
  smoothing: 'Softens bumps and blotches between pores and patches, away from real contours. Pores and grain stay.',
  texturePreservation: 'How much fine grain is kept, and restored where smoothing took it.',
  porePreservation: 'How much pore detail is kept.',
  shine: 'Lowers oily shine and glare that washes out the skin colour; better-lit skin and natural highlights stay.',
  darkSpots: 'Lifts dark spots, post-acne marks and hyperpigmentation toward the skin around them. Moles, nostrils, creases and the folds beside the nose stay.',
  toneEvenness: 'Evens blotchy colour (red or brown patches) without changing the lighting or the skin tone.',
  underEye: 'Lifts the under-eye shadow toward your own cheek skin, keeping texture and part of the natural shadow.',
  dodgeBurn: 'Evens small light and dark irregularities, like a retoucher’s dodge and burn. The face’s shape and contours stay.',
};

export interface FinishControl {
  enabled: boolean;
  /** 0..100 */
  amount: number;
}

export type FinishStyle = 'natural' | 'professional' | 'advanced' | 'custom';

export const FINISH_STYLE_LABELS: Record<Exclude<FinishStyle, 'custom'>, string> = {
  natural: 'Natural',
  professional: 'Professional',
  advanced: 'Advanced',
};

export interface SkinFinishSettings extends Record<FinishControlName, FinishControl> {
  enabled: boolean;
  /** 0..100 Overall Skin Finish Strength: scales every effect. */
  strength: number;
  style: FinishStyle;
  /** Neck, shoulders, arms, hands, legs (face parsing). */
  bodyCoverage: boolean;
}

/** Every amount of each style; the same as the engine's measurements in docs/SKIN_FINISH.md. */
export const FINISH_STYLES: Readonly<Record<Exclude<FinishStyle, 'custom'>, Readonly<Record<FinishControlName, number>>>> = Object.freeze({
  natural: { smoothing: 35, texturePreservation: 85, porePreservation: 80, shine: 25, darkSpots: 45, toneEvenness: 25, underEye: 25, dodgeBurn: 15 },
  professional: { smoothing: 55, texturePreservation: 75, porePreservation: 65, shine: 45, darkSpots: 70, toneEvenness: 45, underEye: 45, dodgeBurn: 30 },
  advanced: { smoothing: 80, texturePreservation: 60, porePreservation: 45, shine: 60, darkSpots: 95, toneEvenness: 65, underEye: 60, dodgeBurn: 45 },
});

function styleControls(style: Exclude<FinishStyle, 'custom'>): Record<FinishControlName, FinishControl> {
  const out = {} as Record<FinishControlName, FinishControl>;
  for (const k of FINISH_CONTROLS) out[k] = { enabled: true, amount: FINISH_STYLES[style][k] };
  return out;
}

/** Off by default; switched on it starts as the Natural style. */
export const DEFAULT_SKIN_FINISH: Readonly<SkinFinishSettings> = Object.freeze({
  enabled: false,
  strength: 100,
  style: 'natural',
  bodyCoverage: true,
  ...styleControls('natural'),
});

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const num = (v: unknown, lo: number, hi: number, d: number) => (finite(v) ? clamp(v, lo, hi) : d);
const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d);

export function sanitizeSkinFinish(input: unknown): SkinFinishSettings {
  const r = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>;
  const d = DEFAULT_SKIN_FINISH;
  const out = {
    enabled: bool(r.enabled, d.enabled),
    strength: num(r.strength, 0, 100, d.strength),
    style: (['natural', 'professional', 'advanced', 'custom'] as const).includes(r.style as FinishStyle) ? (r.style as FinishStyle) : 'custom',
    bodyCoverage: bool(r.bodyCoverage, d.bodyCoverage),
  } as SkinFinishSettings;
  if (!('style' in r)) out.style = d.style;
  for (const k of FINISH_CONTROLS) {
    const c = (r[k] && typeof r[k] === 'object' ? r[k] : {}) as Record<string, unknown>;
    out[k] = { enabled: bool(c.enabled, true), amount: num(c.amount, 0, 100, d[k].amount) };
  }
  return out;
}

export function skinFinishIsDefault(s: SkinFinishSettings): boolean {
  return !s.enabled;
}

/** Every amount from the style, every effect switched on; Overall strength and body kept. */
export function applyFinishStyle(s: SkinFinishSettings, style: Exclude<FinishStyle, 'custom'>): SkinFinishSettings {
  return { ...s, ...styleControls(style), style };
}

/** One effect changed by the user: the settings are now a custom style. */
export function setFinishControl(s: SkinFinishSettings, k: FinishControlName, patch: Partial<FinishControl>): SkinFinishSettings {
  return { ...s, [k]: { ...s[k], ...patch }, style: 'custom' };
}

/** The value Reset returns an effect to: the picked style's amount (Natural when custom). */
export function finishResetValue(s: SkinFinishSettings, k: FinishControlName): number {
  return FINISH_STYLES[s.style === 'custom' ? 'natural' : s.style][k];
}

/** Reset one effect: back on, at its style's amount. The style is kept. */
export function resetFinishControl(s: SkinFinishSettings, k: FinishControlName): SkinFinishSettings {
  return { ...s, [k]: { enabled: true, amount: finishResetValue(s, k) } };
}

/** Whether an effect does anything (switched on, amount above 0, Overall above 0). */
export function finishControlActive(s: SkinFinishSettings, k: FinishControlName): boolean {
  return s.enabled && s.strength > 0 && s[k].enabled && s[k].amount > 0;
}
