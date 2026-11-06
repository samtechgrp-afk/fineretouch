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

export interface SliderSpec {
  key: AdjustmentKey;
  label: string;
  min: number;
  max: number;
  step: number;
  group: AdjustmentGroup;
  /** Needs face landmarks from the AI service to have any effect. */
  needsFace?: boolean;
}

export type AdjustmentGroup = 'light' | 'color' | 'skin' | 'face' | 'detail';

export interface ColorStats {
  /** Mean of CIE L*, a*, b* (D65, L 0-100, a/b signed). */
  mean: [number, number, number];
  /** Standard deviation of L, a, b. */
  std: [number, number, number];
}

export interface ColorMatch {
  /** Statistics of the reference image the look is copied from. */
  reference: ColorStats;
  /** 0-100 blend between the original and the fully transferred look. */
  amount: number;
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
  skinTexture: number; // 0..100 how much high-frequency pore detail is preserved
  blemishRemoval: number; // 0..100
  skinToneUnify: number; // 0..100 pulls skin chroma toward the face average
  // Face (landmark driven)
  eyeBrighten: number; // 0..100
  darkCircles: number; // 0..100
  teethWhiten: number; // 0..100
  // Detail
  sharpen: number; // 0..100
  vignette: number; // -100..100 (negative darkens corners)
  colorMatch: ColorMatch | null;
}

export type AdjustmentKey = Exclude<keyof Adjustments, 'colorMatch'>;

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
  eyeBrighten: 0,
  darkCircles: 0,
  teethWhiten: 0,
  sharpen: 0,
  vignette: 0,
  colorMatch: null,
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
  { key: 'skinSmoothing', label: 'Smoothing', min: 0, max: 100, step: 1, group: 'skin' },
  { key: 'skinTexture', label: 'Keep texture', min: 0, max: 100, step: 1, group: 'skin' },
  { key: 'blemishRemoval', label: 'Blemish removal', min: 0, max: 100, step: 1, group: 'skin' },
  { key: 'skinToneUnify', label: 'Even skin tone', min: 0, max: 100, step: 1, group: 'skin' },
  { key: 'eyeBrighten', label: 'Brighten eyes', min: 0, max: 100, step: 1, group: 'face', needsFace: true },
  { key: 'darkCircles', label: 'Reduce dark circles', min: 0, max: 100, step: 1, group: 'face', needsFace: true },
  { key: 'teethWhiten', label: 'Whiten teeth', min: 0, max: 100, step: 1, group: 'face', needsFace: true },
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

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

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
    const mean = triple(ref?.mean);
    const std = triple(ref?.std);
    const amount = typeof cm.amount === 'number' && Number.isFinite(cm.amount) ? clamp(cm.amount, 0, 100) : 100;
    if (mean && std) out.colorMatch = { reference: { mean, std }, amount };
  }
  return out;
}

export function isDefault(adj: Adjustments): boolean {
  return SLIDERS.every((s) => adj[s.key] === DEFAULT_ADJUSTMENTS[s.key]) && adj.colorMatch === null;
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
  return sanitizeAdjustments({ ...base, ...preset.adjustments });
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
