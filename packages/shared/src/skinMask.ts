/**
 * Skin mask refinement (v0.8, Phase 3): AI face parsing and the user's brush strokes.
 * Mirrored by SkinMaskSettings / MaskStroke in services/ai/app/schemas.py and applied by
 * services/ai/app/masking.py (export) and apps/web/src/engine/masks.ts (preview).
 */

export const MAX_MASK_STROKES = 200;
export const MAX_STROKE_POINTS = 4000;
export const MIN_STROKE_SIZE = 0.001;
export const MAX_STROKE_SIZE = 0.2;

export interface MaskStroke {
  /** add: paint skin the AI missed. erase: protect an area from every skin tool. */
  mode: 'add' | 'erase';
  /** Brush radius as a share of the image's long edge. */
  size: number;
  /** Normalised [x0, y0, x1, y1, ...] in image coordinates (0..1). */
  points: number[];
}

export interface SkinMaskSettings {
  /** Use the AI face parsing (face skin, hair) when the engine has the segmentation model. */
  useSegmentation: boolean;
  /** Also retouch neck, chest and arm skin found by the face parsing. */
  includeBodySkin: boolean;
  /** 0..100: softness of brush stroke edges (up to the stroke's radius). */
  feather: number;
  strokes: MaskStroke[];
}

export const DEFAULT_SKIN_MASK: Readonly<SkinMaskSettings> = Object.freeze({
  useSegmentation: true,
  includeBodySkin: false,
  feather: 50,
  strokes: [],
});

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clampN = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function sanitizeStroke(input: unknown): MaskStroke | null {
  if (!input || typeof input !== 'object') return null;
  const r = input as Record<string, unknown>;
  if (!Array.isArray(r.points)) return null;
  const pts = r.points.slice(0, 2 * MAX_STROKE_POINTS);
  if (pts.length < 2 || !pts.every(finite)) return null;
  return {
    mode: r.mode === 'erase' ? 'erase' : 'add',
    size: finite(r.size) ? clampN(r.size, MIN_STROKE_SIZE, MAX_STROKE_SIZE) : 0.02,
    points: (pts as number[]).slice(0, Math.floor(pts.length / 2) * 2).map((v) => clampN(v, -0.5, 1.5)),
  };
}

export function sanitizeSkinMask(input: unknown): SkinMaskSettings {
  const r = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>;
  const strokes = Array.isArray(r.strokes)
    ? r.strokes.map(sanitizeStroke).filter((s): s is MaskStroke => s !== null).slice(-MAX_MASK_STROKES)
    : [];
  return {
    useSegmentation: typeof r.useSegmentation === 'boolean' ? r.useSegmentation : DEFAULT_SKIN_MASK.useSegmentation,
    includeBodySkin: typeof r.includeBodySkin === 'boolean' ? r.includeBodySkin : DEFAULT_SKIN_MASK.includeBodySkin,
    feather: finite(r.feather) ? clampN(r.feather, 0, 100) : DEFAULT_SKIN_MASK.feather,
    strokes,
  };
}

export function skinMaskIsDefault(m: SkinMaskSettings): boolean {
  return (
    m.strokes.length === 0 &&
    m.useSegmentation === DEFAULT_SKIN_MASK.useSegmentation &&
    m.includeBodySkin === DEFAULT_SKIN_MASK.includeBodySkin &&
    m.feather === DEFAULT_SKIN_MASK.feather
  );
}

/**
 * Brush strokes belong to one photo's pixels. When settings are copied, synced or applied as a
 * preset, the target keeps its own strokes and only the mask options travel. The pro tools
 * (healing, clone and local retouch layers, localRetouch.ts) are all pixel-specific, so the
 * target keeps its own localRetouch whole.
 */
export function withOwnStrokes<
  T extends { skinMask: SkinMaskSettings; localRetouch?: unknown; skinPerfection?: { overrides: unknown[] } },
>(next: T, target: { skinMask: SkinMaskSettings; localRetouch?: unknown; skinPerfection?: { overrides: unknown[] } }): T {
  const out = { ...next, skinMask: { ...next.skinMask, strokes: target.skinMask.strokes } };
  if (target.localRetouch !== undefined) out.localRetouch = target.localRetouch as T['localRetouch'];
  // Skin Perfection "ignore" / "correct" decisions point at marks on this photo's pixels.
  if (next.skinPerfection && target.skinPerfection) {
    out.skinPerfection = { ...next.skinPerfection, overrides: target.skinPerfection.overrides } as T['skinPerfection'];
  }
  return out;
}
