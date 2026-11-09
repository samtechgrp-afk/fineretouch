/**
 * Professional retouching tools (v0.10, Phase 5): healing and clone brushes, and local retouch
 * layers (frequency separation, dodge & burn, correction brushes) with their own masks.
 *
 * Everything here is non-destructive: strokes and layer settings are stored in the edit and
 * replayed on the original pixels at render time.
 *
 * Mirrored by LocalRetouch / HealStroke / RetouchLayer in services/ai/app/schemas.py and applied
 * by services/ai/app/engines/local.py (export) and apps/web/src/engine/heal.ts + the LOCAL_FRAG
 * pass in apps/web/src/engine/shaders.ts (preview and browser export).
 */

export const MAX_HEAL_STROKES = 300;
export const MAX_LAYERS = 8;
export const MAX_LAYER_STROKES = 200;
export const MAX_TOOL_POINTS = 4000;
export const MIN_TOOL_SIZE = 0.001;
export const MAX_TOOL_SIZE = 0.2;

export interface HealStroke {
  /** heal: texture from the source, tone and colour from around the destination. clone: copy. */
  mode: 'heal' | 'clone';
  /** Brush radius as a share of the image's long edge. */
  size: number;
  /** 0..100: share of the radius at full strength (100 = hard edge, 0 = soft cone). */
  hardness: number;
  /** 0..100 */
  opacity: number;
  /** Source minus destination, normalised (x share of the width, y share of the height). */
  offset: [number, number];
  /** Normalised [x0, y0, x1, y1, ...] in image coordinates (0..1). */
  points: number[];
}

export type LayerKind = 'frequency' | 'dodgeBurn' | 'correction';
export type LayerStrokeMode = 'add' | 'erase' | 'dodge' | 'burn';
export type MaskBase = 'none' | 'skin' | 'all';
export type ToneRange = 'all' | 'shadows' | 'midtones' | 'highlights';

export interface LayerStroke {
  /** add / erase paint the layer's mask; dodge / burn paint a dodge & burn layer. */
  mode: LayerStrokeMode;
  size: number;
  /** 0..100: how much one pass adds (passes build up, as with a low-flow brush). */
  flow: number;
  points: number[];
}

export interface LayerMask {
  /** Starting coverage before strokes: nothing (paint it in), the AI skin mask, or everything. */
  base: MaskBase;
  /** 0..100: softness of stroke edges (up to the stroke's radius). */
  feather: number;
  invert: boolean;
  /** Luminance range 0..100 the layer applies to, with a soft roll-off. */
  lumaMin: number;
  lumaMax: number;
  lumaSoftness: number;
  /** 0..100: overall mask density. */
  density: number;
  strokes: LayerStroke[];
}

export interface RetouchLayer {
  id: string;
  name: string;
  kind: LayerKind;
  enabled: boolean;
  /** 0..100 */
  opacity: number;
  mask: LayerMask;
  /** Frequency separation: 0..100 maps to a split radius of 0.08%..2.4% of the long edge. */
  splitRadius: number;
  /** 0..100: edge-aware smoothing of the tone (low-frequency) layer. */
  toneSmoothing: number;
  /** 0..200: detail (high-frequency) layer strength; 100 keeps texture as it is. */
  texture: number;
  /** Dodge & burn: maximum exposure change at full flow, 0..100 = 0..1.5 EV. */
  dodgeBurnStrength: number;
  dodgeBurnRange: ToneRange;
  /** Correction brush (-100..100 each). */
  exposure: number;
  contrast: number;
  saturation: number;
  temperature: number;
  tint: number;
}

export interface LocalRetouch {
  heal: HealStroke[];
  layers: RetouchLayer[];
}

export const DEFAULT_LOCAL_RETOUCH: Readonly<LocalRetouch> = Object.freeze({ heal: [], layers: [] });

export const DEFAULT_LAYER_MASK: Readonly<LayerMask> = Object.freeze({
  base: 'none',
  feather: 50,
  invert: false,
  lumaMin: 0,
  lumaMax: 100,
  lumaSoftness: 20,
  density: 100,
  strokes: [],
});

const LAYER_DEFAULTS: Omit<RetouchLayer, 'id' | 'name' | 'kind' | 'mask'> = {
  enabled: true,
  opacity: 100,
  splitRadius: 40,
  toneSmoothing: 0,
  texture: 100,
  dodgeBurnStrength: 50,
  dodgeBurnRange: 'all',
  exposure: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  tint: 0,
};

export const LAYER_NAMES: Record<LayerKind, string> = {
  frequency: 'Frequency separation',
  dodgeBurn: 'Dodge & burn',
  correction: 'Correction brush',
};

/** A new layer with sensible starting values for its kind. */
export function createLayer(kind: LayerKind, id: string, name = LAYER_NAMES[kind]): RetouchLayer {
  const base: MaskBase = kind === 'frequency' ? 'skin' : kind === 'dodgeBurn' ? 'all' : 'none';
  return {
    ...LAYER_DEFAULTS,
    id,
    name,
    kind,
    toneSmoothing: kind === 'frequency' ? 50 : 0,
    mask: { ...DEFAULT_LAYER_MASK, base, strokes: [] },
  };
}

/** Split radius in pixels for a long edge of L pixels (same formula in engines/local.py). */
export function splitSigma(splitRadius: number, L: number): number {
  return Math.max(0.6, L * 0.0008 * Math.pow(30, clampN(splitRadius, 0, 100) / 100));
}

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clampN = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const num = (v: unknown, lo: number, hi: number, d: number) => (finite(v) ? clampN(v, lo, hi) : d);
const rec = (v: unknown) => (v && typeof v === 'object' ? (v as Record<string, unknown>) : {});
const pick = <T extends string>(v: unknown, allowed: readonly T[], d: T): T => (allowed.includes(v as T) ? (v as T) : d);

function points(v: unknown): number[] | null {
  if (!Array.isArray(v)) return null;
  const pts = v.slice(0, 2 * MAX_TOOL_POINTS);
  if (pts.length < 2 || !pts.every(finite)) return null;
  return (pts as number[]).slice(0, Math.floor(pts.length / 2) * 2).map((x) => clampN(x, -0.5, 1.5));
}

export function sanitizeHealStroke(input: unknown): HealStroke | null {
  const r = rec(input);
  const pts = points(r.points);
  const off = r.offset;
  if (!pts || !Array.isArray(off) || off.length !== 2 || !off.every(finite)) return null;
  return {
    mode: r.mode === 'clone' ? 'clone' : 'heal',
    size: num(r.size, MIN_TOOL_SIZE, MAX_TOOL_SIZE, 0.02),
    hardness: num(r.hardness, 0, 100, 50),
    opacity: num(r.opacity, 0, 100, 100),
    offset: [clampN(off[0] as number, -1, 1), clampN(off[1] as number, -1, 1)],
    points: pts,
  };
}

export function sanitizeLayerStroke(input: unknown): LayerStroke | null {
  const r = rec(input);
  const pts = points(r.points);
  if (!pts) return null;
  return {
    mode: pick(r.mode, ['add', 'erase', 'dodge', 'burn'] as const, 'add'),
    size: num(r.size, MIN_TOOL_SIZE, MAX_TOOL_SIZE, 0.02),
    flow: num(r.flow, 1, 100, 100),
    points: pts,
  };
}

export function sanitizeLayer(input: unknown, index = 0): RetouchLayer | null {
  const r = rec(input);
  const kind = pick(r.kind, ['frequency', 'dodgeBurn', 'correction'] as const, 'correction');
  const d = createLayer(kind, `layer-${index + 1}`);
  const m = rec(r.mask);
  const strokes = Array.isArray(m.strokes)
    ? m.strokes.map(sanitizeLayerStroke).filter((s): s is LayerStroke => s !== null).slice(-MAX_LAYER_STROKES)
    : [];
  const lumaMin = num(m.lumaMin, 0, 100, 0);
  return {
    id: typeof r.id === 'string' && r.id ? r.id.slice(0, 64) : d.id,
    name: typeof r.name === 'string' && r.name.trim() ? r.name.slice(0, 60) : d.name,
    kind,
    enabled: typeof r.enabled === 'boolean' ? r.enabled : true,
    opacity: num(r.opacity, 0, 100, 100),
    mask: {
      base: pick(m.base, ['none', 'skin', 'all'] as const, d.mask.base),
      feather: num(m.feather, 0, 100, DEFAULT_LAYER_MASK.feather),
      invert: typeof m.invert === 'boolean' ? m.invert : false,
      lumaMin,
      lumaMax: Math.max(lumaMin, num(m.lumaMax, 0, 100, 100)),
      lumaSoftness: num(m.lumaSoftness, 0, 100, DEFAULT_LAYER_MASK.lumaSoftness),
      density: num(m.density, 0, 100, 100),
      strokes,
    },
    splitRadius: num(r.splitRadius, 0, 100, d.splitRadius),
    toneSmoothing: num(r.toneSmoothing, 0, 100, d.toneSmoothing),
    texture: num(r.texture, 0, 200, d.texture),
    dodgeBurnStrength: num(r.dodgeBurnStrength, 0, 100, d.dodgeBurnStrength),
    dodgeBurnRange: pick(r.dodgeBurnRange, ['all', 'shadows', 'midtones', 'highlights'] as const, 'all'),
    exposure: num(r.exposure, -100, 100, 0),
    contrast: num(r.contrast, -100, 100, 0),
    saturation: num(r.saturation, -100, 100, 0),
    temperature: num(r.temperature, -100, 100, 0),
    tint: num(r.tint, -100, 100, 0),
  };
}

export function sanitizeLocalRetouch(input: unknown): LocalRetouch {
  const r = rec(input);
  const heal = Array.isArray(r.heal)
    ? r.heal.map(sanitizeHealStroke).filter((s): s is HealStroke => s !== null).slice(-MAX_HEAL_STROKES)
    : [];
  const layers: RetouchLayer[] = [];
  const seen = new Set<string>();
  if (Array.isArray(r.layers)) {
    for (const [i, l] of r.layers.slice(0, MAX_LAYERS).entries()) {
      const s = sanitizeLayer(l, i);
      if (!s) continue;
      while (seen.has(s.id)) s.id = `${s.id}-${i}`;
      seen.add(s.id);
      layers.push(s);
    }
  }
  return { heal, layers };
}

export function localRetouchIsDefault(l: LocalRetouch): boolean {
  return l.heal.length === 0 && l.layers.length === 0;
}

/** Whether a layer changes any pixel with its current settings (ignoring its mask). */
export function layerIsActive(l: RetouchLayer): boolean {
  if (!l.enabled || l.opacity <= 0 || l.mask.density <= 0) return false;
  if (l.kind === 'dodgeBurn') return l.dodgeBurnStrength > 0 && l.mask.strokes.some((s) => s.mode === 'dodge' || s.mode === 'burn');
  const fs = l.toneSmoothing > 0 || l.texture !== 100;
  const corr = l.exposure !== 0 || l.contrast !== 0 || l.saturation !== 0 || l.temperature !== 0 || l.tint !== 0;
  if (!fs && !corr) return false;
  return l.mask.base !== 'none' || l.mask.invert || l.mask.strokes.some((s) => s.mode === 'add');
}
