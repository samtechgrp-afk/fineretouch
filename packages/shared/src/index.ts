/**
 * Shared editing model for FineRetouch.
 *
 * The same `Adjustments` object drives three renderers that must agree:
 *  - the WebGL2 proxy viewport in apps/web (live preview),
 *  - the Python full-resolution renderer in services/ai (export),
 *  - the batch worker in services/worker (which forwards presets to services/ai).
 *
 * Every slider is a plain number with a documented range so a preset is just JSON.
 */

import { DEFAULT_EFFECTS, effectsAreDefault, sanitizeEffects, type EffectControls } from './effects.js';
import { DEFAULT_SKIN_MASK, sanitizeSkinMask, skinMaskIsDefault, withOwnStrokes, type SkinMaskSettings } from './skinMask.js';

export interface SliderSpec {
  key: AdjustmentKey;
  label: string;
  min: number;
  max: number;
  step: number;
  group: AdjustmentGroup;
  /** Needs face landmarks from the AI service to have any effect. */
  needsFace?: boolean;
  /** Shown after the value, e.g. '%'. */
  unit?: string;
  /** Tooltip on the label. */
  hint?: string;
}

export type AdjustmentGroup = 'light' | 'color' | 'skin' | 'face' | 'detail';

export interface ColorStats {
  /** Mean of CIE L*, a*, b* (D65, L 0-100, a/b signed). */
  mean: [number, number, number];
  /** Standard deviation of L, a, b. */
  std: [number, number, number];
  /**
   * Optional full 3x3 L*a*b* covariance (row-major). When both the photo and the
   * reference carry it, colour match uses a covariance-aligning (Monge-Kantorovich)
   * linear transform instead of per-channel scaling, so cross-channel casts transfer too.
   */
  cov?: number[];
  /**
   * Optional L* values at TONE_QUANTILES. When both sides carry them, colour match
   * replaces L* with a histogram-matched tone curve (the reference's exact contrast shape).
   */
  lQuantiles?: number[];
}

/** 17 evenly spaced quantiles (0, 1/16, ..., 1) used for tone-curve matching. */
export const TONE_QUANTILE_COUNT = 17;

export interface ColorMatch {
  /** Statistics of the reference image the look is copied from. */
  reference: ColorStats;
  /** 0-100 blend between the original and the fully transferred look. */
  amount: number;
}

/** Strength curve for the boosted blemish / dark-spot / tone-evening pass. */
export type SkinMultiplier = 1 | 2 | 3;
export const SKIN_MULTIPLIERS: readonly SkinMultiplier[] = [1, 2, 3];

export interface StrayHairSettings {
  enabled: boolean;
  strength: number; // 0..100
}

export type ApertureShape = 'circle' | 'hex';

/** Clothing "virtual iron" and recolour (services/ai retouch_engine.remove_clothing_wrinkles). */
export interface ClothingSettings {
  enabled: boolean;
  wrinkleStrength: number; // 0..100 how much of the crease / fold shading is ironed out
  texturePreservation: number; // 0..100 share of the weave / thread layer kept at full strength
  hueShift: number; // -180..180 degrees, CIELAB hue rotation of the garment
  saturation: number; // -100..100 garment chroma
}

/** Largest 3D LUT accepted (33^3 entries); same as LUT_MAX_SIZE in services/ai/app/schemas.py. */
export const LUT_MAX_SIZE = 33;

/**
 * A 3D colour lookup table ("look"): fitted from a graded reference by the engine, or loaded
 * from a .cube file. `data` is base64 of size^3 RGB triples as little-endian uint16, red index
 * fastest (the .cube order); see lut.ts.
 */
export interface LookLut {
  name: string;
  size: number; // 2..LUT_MAX_SIZE
  data: string;
  amount: number; // 0..100
}

export interface DepthBokehSettings {
  enabled: boolean;
  /** 0..100: base background blur, as a share of 2% of the long edge. */
  blurRadius: number;
  /** 100..300 (%): multiplies the base radius; 200 = "double blur", 300 = triple. */
  bokehMultiplier: number;
  apertureShape: ApertureShape;
  /** 0..100: how much bright background points bloom into aperture-shaped discs. */
  highlightBoost: number;
}

// ---------------------------------------------------------------------------
// v0.5: 8-channel HSL mixer and 3-way colour grading
// (maths in apps/web/src/engine/shaders.ts and services/ai/app/color_engine.py)
// ---------------------------------------------------------------------------

export const HSL_CHANNELS = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'] as const;
export type HslChannel = (typeof HSL_CHANNELS)[number];
/** Hue (degrees) where each channel has full weight; pixels in between are split by a smoothstep. */
export const HSL_CENTERS: Readonly<Record<HslChannel, number>> = Object.freeze({
  red: 0, orange: 30, yellow: 60, green: 120, aqua: 180, blue: 240, purple: 270, magenta: 300,
});
export const HSL_LABELS: Readonly<Record<HslChannel, string>> = Object.freeze({
  red: 'Red', orange: 'Orange', yellow: 'Yellow', green: 'Green', aqua: 'Aqua', blue: 'Blue', purple: 'Purple', magenta: 'Magenta',
});
/** Hue shift in degrees at hue = +/-100. */
export const HSL_HUE_RANGE = 30;
/** Brightness change of a fully saturated colour at luminance = +/-100. */
export const HSL_LUM_RANGE = 0.6;
/** Saturation where the mixer reaches full strength (greys are left alone). */
export const HSL_NEUTRAL_EDGE = 0.2;

/**
 * Skin-tone guard (CIELAB), same values as TONE_GUARD_* in services/ai/app/retouch_engine.py.
 * Retouched skin keeps at least `chromaKeep` of the lower of its own and its surroundings'
 * chroma, and its hue may leave the arc between those two by at most `hueTolerance` degrees.
 * The hue lock fades in between the two `minChroma` values (a*b* units).
 */
export const SKIN_TONE_GUARD = Object.freeze({ chromaKeep: 0.9, hueTolerance: 6, minChroma: [2, 6] as const });

export interface HslValue {
  hue: number; // -100..100
  saturation: number; // -100..100
  luminance: number; // -100..100
}
export type HslMixer = Record<HslChannel, HslValue>;

export type GradeZone = 'shadows' | 'midtones' | 'highlights';
export const GRADE_ZONES: readonly GradeZone[] = ['shadows', 'midtones', 'highlights'];

export interface GradeWheel {
  hue: number; // 0..360
  saturation: number; // 0..100 tint strength
  luminance: number; // -100..100
}

export interface ColorGrading {
  shadows: GradeWheel;
  midtones: GradeWheel;
  highlights: GradeWheel;
  /** 0..100: width of the crossovers between zones (50 = default soft blend). */
  blending: number;
  /** -100..100: moves both pivots; positive lets the highlight tint reach further down. */
  balance: number;
}

/** Shadow / highlight pivots (luma), their shift at balance +/-100, and crossover widths. */
export const GRADE_PIVOTS = [0.25, 0.75] as const;
export const GRADE_BALANCE_SHIFT = 0.2;
export const GRADE_WIDTH_MIN = 0.05;
export const GRADE_WIDTH_RANGE = 0.4;
/** Chroma offset at wheel saturation 100, and additive offset at wheel luminance +/-100. */
export const GRADE_TINT = 0.3;
export const GRADE_LUM = 0.25;

const ZERO_HSL: Readonly<HslValue> = Object.freeze({ hue: 0, saturation: 0, luminance: 0 });
export const DEFAULT_HSL: Readonly<HslMixer> = Object.freeze(
  Object.fromEntries(HSL_CHANNELS.map((c) => [c, ZERO_HSL])) as HslMixer,
);
const ZERO_WHEEL: Readonly<GradeWheel> = Object.freeze({ hue: 0, saturation: 0, luminance: 0 });
export const DEFAULT_COLOR_GRADING: Readonly<ColorGrading> = Object.freeze({
  shadows: ZERO_WHEEL,
  midtones: ZERO_WHEEL,
  highlights: ZERO_WHEEL,
  blending: 50,
  balance: 0,
});

export function hslIsDefault(h: HslMixer): boolean {
  return HSL_CHANNELS.every((c) => h[c].hue === 0 && h[c].saturation === 0 && h[c].luminance === 0);
}

export function gradingIsDefault(g: ColorGrading): boolean {
  return GRADE_ZONES.every((z) => g[z].saturation === 0 && g[z].luminance === 0);
}

export interface Adjustments {
  // Light
  exposure: number; // -100..100 => -2EV..+2EV
  contrast: number; // -100..100
  highlights: number; // -100..100
  shadows: number; // -100..100
  // Color
  temperature: number; // -100..100 (blue..amber)
  tint: number; // -100..100 (green..magenta)
  saturation: number; // -100..100
  vibrance: number; // -100..100
  // Skin
  skinSmoothing: number; // 0..100 frequency-separation strength on skin
  skinTexture: number; // 0..100 texture retention: share of the pore / detail layer re-injected after smoothing
  blemishRemoval: number; // 0..100
  skinToneUnify: number; // 0..100 pulls skin chroma toward the face average
  /** v0.7 Acne removal: red spots, pimples, raised bumps, then post-acne redness. 0..100 */
  acneRemoval: number;
  /** v0.7 Dark spot removal: isolated dark marks, then uneven pigmentation on the tone layer. 0..100 */
  darkSpotRemoval: number;
  /** Dark spot and blemish removal leave small, faint freckles alone. */
  keepFreckles: boolean;
  // Face (landmark driven)
  eyeBrighten: number; // 0..100
  darkCircles: number; // 0..100
  teethWhiten: number; // 0..100
  // Detail
  sharpen: number; // 0..100
  vignette: number; // -100..100 (negative darkens corners)
  colorMatch: ColorMatch | null;
  // Boosted retouching (v0.2)
  /** 1x = classic retouch; 2x / 3x add a recursive low-frequency clean-up pass on skin. */
  skinMultiplier: SkinMultiplier;
  strayHairRemoval: StrayHairSettings;
  depthBokeh: DepthBokehSettings;
  // Spot healing, colour mixer and grading (v0.5)
  /** Spot healing (the blemishRemoval slider) leaves moles / beauty marks alone. */
  protectBeautyMarks: boolean;
  /** CIELAB a* and b* lock: retouched skin may not turn ashen or drift off its own undertone. */
  skinToneGuard: boolean;
  hsl: HslMixer;
  colorGrading: ColorGrading;
  // Pro studio (v0.6)
  /** Removes flash / window reflections on eyeglass lenses (needs eye landmarks). */
  glareReduction: number; // 0..100
  /** Evens yellow or dark patches on teeth toward their own enamel colour. */
  teethStainRemoval: number; // 0..100
  clothing: ClothingSettings;
  /** Reference look as a 3D LUT, applied after colour grading. */
  lut: LookLut | null;
  // Per-effect controls (v0.7, effects.ts): enable / 1x-2x-3x / opacity per retouch effect,
  // and a master strength over all of them. Slider values above are never rewritten.
  effects: EffectControls;
  /** 0..100: scales every retouch effect's contribution (not light / colour). */
  globalStrength: number;
  /** v0.8 skin mask refinement: AI face parsing and the user's add / erase brush strokes. */
  skinMask: SkinMaskSettings;
}

/** Keys of the plain numeric sliders (structured settings have their own controls). */
export type AdjustmentKey = Exclude<
  keyof Adjustments,
  | 'colorMatch'
  | 'skinMultiplier'
  | 'strayHairRemoval'
  | 'depthBokeh'
  | 'protectBeautyMarks'
  | 'skinToneGuard'
  | 'keepFreckles'
  | 'hsl'
  | 'colorGrading'
  | 'clothing'
  | 'lut'
  | 'effects'
  | 'globalStrength'
  | 'skinMask'
>;

export const DEFAULT_STRAY_HAIR: Readonly<StrayHairSettings> = Object.freeze({ enabled: false, strength: 60 });
export const DEFAULT_CLOTHING: Readonly<ClothingSettings> = Object.freeze({
  enabled: false,
  wrinkleStrength: 60,
  texturePreservation: 80,
  hueShift: 0,
  saturation: 0,
});
export const DEFAULT_DEPTH_BOKEH: Readonly<DepthBokehSettings> = Object.freeze({
  enabled: false,
  blurRadius: 50,
  bokehMultiplier: 100,
  apertureShape: 'circle',
  highlightBoost: 40,
});

export const DEFAULT_ADJUSTMENTS: Readonly<Adjustments> = Object.freeze({
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  temperature: 0,
  tint: 0,
  saturation: 0,
  vibrance: 0,
  skinSmoothing: 0,
  skinTexture: 50,
  blemishRemoval: 0,
  skinToneUnify: 0,
  acneRemoval: 0,
  darkSpotRemoval: 0,
  keepFreckles: false,
  eyeBrighten: 0,
  darkCircles: 0,
  teethWhiten: 0,
  sharpen: 0,
  vignette: 0,
  colorMatch: null,
  skinMultiplier: 1,
  strayHairRemoval: DEFAULT_STRAY_HAIR,
  depthBokeh: DEFAULT_DEPTH_BOKEH,
  protectBeautyMarks: true,
  skinToneGuard: true,
  hsl: DEFAULT_HSL,
  colorGrading: DEFAULT_COLOR_GRADING,
  glareReduction: 0,
  teethStainRemoval: 0,
  clothing: DEFAULT_CLOTHING,
  lut: null,
  effects: DEFAULT_EFFECTS,
  globalStrength: 100,
  skinMask: DEFAULT_SKIN_MASK,
});

export const SLIDERS: readonly SliderSpec[] = [
  { key: 'exposure', label: 'Exposure', min: -100, max: 100, step: 1, group: 'light' },
  { key: 'contrast', label: 'Contrast', min: -100, max: 100, step: 1, group: 'light' },
  { key: 'highlights', label: 'Highlights', min: -100, max: 100, step: 1, group: 'light' },
  { key: 'shadows', label: 'Shadows', min: -100, max: 100, step: 1, group: 'light' },
  { key: 'temperature', label: 'Temperature', min: -100, max: 100, step: 1, group: 'color' },
  { key: 'tint', label: 'Tint', min: -100, max: 100, step: 1, group: 'color' },
  { key: 'saturation', label: 'Saturation', min: -100, max: 100, step: 1, group: 'color' },
  { key: 'vibrance', label: 'Vibrance', min: -100, max: 100, step: 1, group: 'color' },
  {
    key: 'skinSmoothing', label: 'Skin smoothing', min: 0, max: 100, step: 1, group: 'skin',
    hint: 'Evens blotches and roughness on the tone layer only (frequency separation); pores come back through Texture preservation.',
  },
  {
    key: 'skinTexture', label: 'Texture preservation', min: 0, max: 100, step: 1, group: 'skin', unit: '%',
    hint: 'Share of the pore and fine-detail layer put back after smoothing. 100% keeps every pore.',
  },
  {
    key: 'acneRemoval', label: 'Acne removal', min: 0, max: 100, step: 1, group: 'skin',
    hint: 'Finds red spots, pimples and raised bumps one by one, heals them with real skin texture, then reduces the redness around them. Unaffected skin is untouched.',
  },
  {
    key: 'darkSpotRemoval', label: 'Dark spot removal', min: 0, max: 100, step: 1, group: 'skin',
    hint: 'Heals isolated dark marks and evens brownish pigmentation patches. Facial shadows, beauty marks and (optionally) freckles are kept.',
  },
  {
    key: 'blemishRemoval', label: 'Blemish removal', min: 0, max: 100, step: 1, group: 'skin',
    hint: 'General clean-up: every kind of small spot or mark (acne, dark spots, minor imperfections) in one pass.',
  },
  {
    key: 'skinToneUnify', label: 'Skin tone correction', min: 0, max: 100, step: 1, group: 'skin',
    hint: 'Evens uneven colour and redness toward the person\'s own local skin tone (never a preset complexion).',
  },
  {
    key: 'darkCircles', label: 'Under-eye correction', min: 0, max: 100, step: 1, group: 'skin', needsFace: true,
    hint: 'Lifts dark circles and under-eye discoloration toward the cheek tone, without enlarging or reshaping the eyes.',
  },
  { key: 'eyeBrighten', label: 'Brighten eyes', min: 0, max: 100, step: 1, group: 'face', needsFace: true },
  { key: 'teethWhiten', label: 'Whiten teeth', min: 0, max: 100, step: 1, group: 'face', needsFace: true },
  {
    key: 'teethStainRemoval', label: 'Teeth stains', min: 0, max: 100, step: 1, group: 'face', needsFace: true,
    hint: 'Evens yellow or dark patches toward the teeth\'s own enamel colour, keeping their texture.',
  },
  {
    key: 'glareReduction', label: 'Glasses glare', min: 0, max: 100, step: 1, group: 'face', needsFace: true,
    hint: 'Removes flash and window reflections on eyeglass lenses so the eyes behind them show again.',
  },
  { key: 'sharpen', label: 'Sharpen', min: 0, max: 100, step: 1, group: 'detail' },
  { key: 'vignette', label: 'Vignette', min: -100, max: 100, step: 1, group: 'detail' },
];

export const GROUP_LABELS: Record<AdjustmentGroup, string> = {
  light: 'Light',
  color: 'Color',
  skin: 'Skin retouch',
  face: 'Face features',
  detail: 'Detail',
};

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Coerce untrusted JSON (a preset file, an API body) into a valid Adjustments object.
 * Unknown keys are dropped, missing keys fall back to defaults, numbers are clamped.
 */
export function sanitizeAdjustments(input: unknown): Adjustments {
  const src = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>;
  const out: Adjustments = { ...DEFAULT_ADJUSTMENTS };
  for (const s of SLIDERS) {
    const v = src[s.key];
    if (typeof v === 'number' && Number.isFinite(v)) out[s.key] = clamp(v, s.min, s.max);
  }
  const cm = src.colorMatch as Record<string, unknown> | null | undefined;
  if (cm && typeof cm === 'object') {
    const ref = cm.reference as Record<string, unknown> | undefined;
    const triple = (t: unknown): [number, number, number] | null =>
      Array.isArray(t) && t.length === 3 && t.every((n) => typeof n === 'number' && Number.isFinite(n))
        ? (t as [number, number, number])
        : null;
    const reference = sanitizeColorStats(ref);
    const amount = typeof cm.amount === 'number' && Number.isFinite(cm.amount) ? clamp(cm.amount, 0, 100) : 100;
    if (reference) out.colorMatch = { reference, amount };
  }
  const m = src.skinMultiplier;
  if (m === 1 || m === 2 || m === 3) out.skinMultiplier = m;
  const sh = asRecord(src.strayHairRemoval);
  out.strayHairRemoval = {
    enabled: typeof sh.enabled === 'boolean' ? sh.enabled : DEFAULT_STRAY_HAIR.enabled,
    strength: num(sh.strength, 0, 100, DEFAULT_STRAY_HAIR.strength),
  };
  const db = asRecord(src.depthBokeh);
  out.depthBokeh = {
    enabled: typeof db.enabled === 'boolean' ? db.enabled : DEFAULT_DEPTH_BOKEH.enabled,
    blurRadius: num(db.blurRadius, 0, 100, DEFAULT_DEPTH_BOKEH.blurRadius),
    bokehMultiplier: num(db.bokehMultiplier, 100, 300, DEFAULT_DEPTH_BOKEH.bokehMultiplier),
    apertureShape: db.apertureShape === 'hex' ? 'hex' : 'circle',
    highlightBoost: num(db.highlightBoost, 0, 100, DEFAULT_DEPTH_BOKEH.highlightBoost),
  };
  if (typeof src.protectBeautyMarks === 'boolean') out.protectBeautyMarks = src.protectBeautyMarks;
  if (typeof src.skinToneGuard === 'boolean') out.skinToneGuard = src.skinToneGuard;
  if (typeof src.keepFreckles === 'boolean') out.keepFreckles = src.keepFreckles;
  out.hsl = sanitizeHsl(src.hsl);
  out.colorGrading = sanitizeColorGrading(src.colorGrading);
  const cl = asRecord(src.clothing);
  out.clothing = {
    enabled: typeof cl.enabled === 'boolean' ? cl.enabled : DEFAULT_CLOTHING.enabled,
    wrinkleStrength: num(cl.wrinkleStrength, 0, 100, DEFAULT_CLOTHING.wrinkleStrength),
    texturePreservation: num(cl.texturePreservation, 0, 100, DEFAULT_CLOTHING.texturePreservation),
    hueShift: num(cl.hueShift, -180, 180, DEFAULT_CLOTHING.hueShift),
    saturation: num(cl.saturation, -100, 100, DEFAULT_CLOTHING.saturation),
  };
  out.lut = sanitizeLookLut(src.lut);
  out.effects = sanitizeEffects(src.effects);
  out.globalStrength = num(src.globalStrength, 0, 100, 100);
  out.skinMask = sanitizeSkinMask(src.skinMask);
  return out;
}

/** Base64 length of a size^3 uint16 RGB LUT. */
export const lutDataLength = (size: number) => 4 * Math.ceil((size ** 3 * 6) / 3);

export function sanitizeLookLut(input: unknown): LookLut | null {
  if (!input || typeof input !== 'object') return null;
  const r = input as Record<string, unknown>;
  const size = r.size;
  if (typeof size !== 'number' || !Number.isInteger(size) || size < 2 || size > LUT_MAX_SIZE) return null;
  if (typeof r.data !== 'string' || r.data.length !== lutDataLength(size) || !/^[A-Za-z0-9+/]*={0,2}$/.test(r.data)) return null;
  const name = typeof r.name === 'string' && r.name.trim() ? r.name.slice(0, 120) : 'Look';
  return { name, size, data: r.data, amount: num(r.amount, 0, 100, 100) };
}

export function sanitizeHsl(input: unknown): HslMixer {
  const r = asRecord(input);
  const out = {} as HslMixer;
  for (const c of HSL_CHANNELS) {
    const v = asRecord(r[c]);
    out[c] = { hue: num(v.hue, -100, 100, 0), saturation: num(v.saturation, -100, 100, 0), luminance: num(v.luminance, -100, 100, 0) };
  }
  return out;
}

export function sanitizeColorGrading(input: unknown): ColorGrading {
  const r = asRecord(input);
  const wheel = (w: unknown): GradeWheel => {
    const v = asRecord(w);
    return { hue: num(v.hue, 0, 360, 0), saturation: num(v.saturation, 0, 100, 0), luminance: num(v.luminance, -100, 100, 0) };
  };
  return {
    shadows: wheel(r.shadows),
    midtones: wheel(r.midtones),
    highlights: wheel(r.highlights),
    blending: num(r.blending, 0, 100, DEFAULT_COLOR_GRADING.blending),
    balance: num(r.balance, -100, 100, DEFAULT_COLOR_GRADING.balance),
  };
}

const asRecord = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' ? (v as Record<string, unknown>) : {});
const num = (v: unknown, lo: number, hi: number, fallback: number) =>
  typeof v === 'number' && Number.isFinite(v) ? clamp(v, lo, hi) : fallback;
const finiteArray = (v: unknown, len: number): number[] | null =>
  Array.isArray(v) && v.length === len && v.every((n) => typeof n === 'number' && Number.isFinite(n)) ? (v as number[]) : null;

/** Validate untrusted colour statistics (mean/std required; cov and quantiles optional). */
export function sanitizeColorStats(input: unknown): ColorStats | null {
  const r = asRecord(input);
  const mean = finiteArray(r.mean, 3) as [number, number, number] | null;
  const std = finiteArray(r.std, 3) as [number, number, number] | null;
  if (!mean || !std) return null;
  const out: ColorStats = { mean: [...mean], std: [...std] };
  const cov = finiteArray(r.cov, 9);
  if (cov) out.cov = [...cov];
  const q = finiteArray(r.lQuantiles, TONE_QUANTILE_COUNT);
  if (q) out.lQuantiles = q.map((v) => clamp(v, 0, 100));
  return out;
}

export function isDefault(adj: Adjustments): boolean {
  return (
    SLIDERS.every((s) => adj[s.key] === DEFAULT_ADJUSTMENTS[s.key]) &&
    adj.colorMatch === null &&
    adj.skinMultiplier === 1 &&
    !adj.strayHairRemoval.enabled &&
    !adj.depthBokeh.enabled &&
    hslIsDefault(adj.hsl) &&
    gradingIsDefault(adj.colorGrading) &&
    !adj.clothing.enabled &&
    adj.lut === null &&
    adj.globalStrength === 100 &&
    effectsAreDefault(adj.effects) &&
    skinMaskIsDefault(adj.skinMask)
  );
}

export interface Preset {
  id: string;
  name: string;
  description: string;
  /** Partial: keys not listed keep the photo's current value when applied with `merge`. */
  adjustments: Partial<Adjustments>;
  builtIn?: boolean;
}

export const BUILT_IN_PRESETS: readonly Preset[] = [
  {
    id: 'natural-portrait',
    name: 'Natural Portrait',
    description: 'Light skin cleanup that keeps pores visible.',
    builtIn: true,
    adjustments: { skinSmoothing: 35, skinTexture: 70, blemishRemoval: 50, skinToneUnify: 20, eyeBrighten: 20, sharpen: 15 },
  },
  {
    id: 'studio-beauty',
    name: 'Studio Beauty',
    description: 'Polished beauty retouch for headshots and studio work.',
    builtIn: true,
    adjustments: {
      skinSmoothing: 65, skinTexture: 45, blemishRemoval: 80, skinToneUnify: 45,
      eyeBrighten: 35, darkCircles: 50, teethWhiten: 35, contrast: 8, sharpen: 20,
    },
  },
  {
    id: 'bright-airy',
    name: 'Bright & Airy',
    description: 'Lifted exposure and soft contrast for weddings and lifestyle.',
    builtIn: true,
    adjustments: { exposure: 18, contrast: -12, highlights: -25, shadows: 30, temperature: 6, vibrance: 15, skinSmoothing: 30, skinTexture: 65 },
  },
  {
    id: 'warm-film',
    name: 'Warm Film',
    description: 'Warm, slightly faded look with gentle vignette.',
    builtIn: true,
    adjustments: { temperature: 22, tint: 6, contrast: -8, saturation: -10, vibrance: 12, shadows: 18, vignette: -25 },
  },
  {
    id: 'moody-editorial',
    name: 'Moody Editorial',
    description: 'Deeper shadows and cool tones for editorial portraits.',
    builtIn: true,
    adjustments: { exposure: -10, contrast: 25, highlights: -30, shadows: -10, temperature: -12, saturation: -15, sharpen: 25, vignette: -35 },
  },
  {
    id: 'teal-orange',
    name: 'Teal & Orange',
    description: 'Cinematic split tone: teal shadows, warm skin and highlights.',
    builtIn: true,
    adjustments: {
      contrast: 12,
      vibrance: 8,
      hsl: {
        ...DEFAULT_HSL,
        orange: { hue: -4, saturation: 8, luminance: 6 },
        blue: { hue: -12, saturation: 10, luminance: -10 },
        aqua: { hue: 8, saturation: -10, luminance: 0 },
        green: { hue: 25, saturation: -35, luminance: 0 },
      },
      colorGrading: {
        ...DEFAULT_COLOR_GRADING,
        shadows: { hue: 195, saturation: 35, luminance: -4 },
        highlights: { hue: 38, saturation: 25, luminance: 0 },
        balance: -10,
      },
    },
  },
  {
    id: 'mono-classic',
    name: 'Classic Mono',
    description: 'Black and white with punchy contrast.',
    builtIn: true,
    adjustments: { saturation: -100, vibrance: 0, contrast: 22, sharpen: 20, vignette: -20 },
  },
];

/** Apply a preset on top of a photo's adjustments. `replace` starts from defaults first. */
export function applyPreset(current: Adjustments, preset: Preset, mode: 'merge' | 'replace' = 'replace'): Adjustments {
  const base = mode === 'replace' ? { ...DEFAULT_ADJUSTMENTS } : { ...current };
  // Brush strokes belong to the photo, never to the preset.
  return withOwnStrokes(sanitizeAdjustments({ ...base, ...preset.adjustments }), current);
}

// ---------------------------------------------------------------------------
// AI analysis contract (services/ai -> apps/web)
// ---------------------------------------------------------------------------

export type Point = [number, number];

/** Face regions in normalised [0,1] image coordinates, as closed polygons. */
export interface FaceRegions {
  faceOval: Point[];
  leftEye: Point[];
  rightEye: Point[];
  leftUnderEye: Point[];
  rightUnderEye: Point[];
  innerMouth: Point[];
  lips: Point[];
  leftBrow: Point[];
  rightBrow: Point[];
  /** Lower nose (nostrils, alae): kept out of spot healing. Absent from older AI services. */
  nose?: Point[];
}

export interface FaceAnalysis {
  box: [number, number, number, number]; // x, y, w, h normalised
  landmarkCount: number;
  regions: FaceRegions;
  /** 0 = closed, 1 = wide open, from eye aspect ratio. */
  eyeOpenness: number;
}

export interface ImageAnalysis {
  width: number;
  height: number;
  faces: FaceAnalysis[];
  /** Variance of the Laplacian on a normalised grey image; higher = sharper. */
  sharpness: number;
  colorStats: ColorStats;
  cull: CullFlags;
}

export interface CullFlags {
  blurry: boolean;
  eyesClosed: boolean;
  noFace: boolean;
}

// ---------------------------------------------------------------------------
// Batch jobs (apps/web -> Redis/BullMQ -> services/worker -> services/ai)
// ---------------------------------------------------------------------------

export const BATCH_QUEUE = 'fineretouch-batch';
export const RENDER_QUEUE = 'fineretouch-render';

export interface BatchJobItem {
  imageId: string;
  /** Object storage key of the original upload. */
  sourceKey: string;
  /** Per-image overrides; falls back to the batch preset. */
  adjustments?: Partial<Adjustments>;
}

export interface BatchJobData {
  batchId: string;
  userId: string;
  preset: Partial<Adjustments>;
  items: BatchJobItem[];
  output: { format: 'jpeg' | 'png' | 'webp'; quality: number; maxLongEdge?: number; prefix: string };
}

export interface RenderJobData {
  batchId: string;
  imageId: string;
  sourceKey: string;
  outputKey: string;
  adjustments: Adjustments;
  output: BatchJobData['output'];
}

export interface RenderJobResult {
  imageId: string;
  outputKey: string;
  bytes: number;
  ms: number;
  cull?: CullFlags;
}

export interface BatchProgress {
  batchId: string;
  total: number;
  done: number;
  failed: number;
  state: 'queued' | 'running' | 'completed' | 'completed_with_errors';
}

export const MAX_BATCH_ITEMS = 5000;

export * from './style.js';
export * from './lut.js';
export * from './effects.js';
export * from './skinMask.js';
