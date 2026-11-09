/**
 * Skin Perfection Studio (v0.13). Mirrors SkinPerfectionSettings / PerfectionCategory /
 * PerfectionOverride in services/ai/app/schemas.py; the engine is services/ai/app/engines/
 * perfection.py (tiled multi-scale detection, classification, targeted correction and
 * verified residual passes). It runs in the local AI engine only: the browser shows the
 * engine's perfected photo as its source and its report as an overlay.
 */

export const PERFECTION_CATEGORIES = [
  'acne',
  'pimple',
  'blackhead',
  'whitehead',
  'darkSpot',
  'postAcne',
  'smallDot',
  'blemish',
  'discoloration',
  'scar',
  'texture',
  'redness',
  'underEye',
] as const;
export type PerfectionCategoryName = (typeof PERFECTION_CATEGORIES)[number];

export const PERFECTION_CATEGORY_LABELS: Record<PerfectionCategoryName, string> = {
  acne: 'Acne',
  pimple: 'Pimples',
  blackhead: 'Blackheads',
  whitehead: 'Whiteheads',
  darkSpot: 'Dark spots',
  postAcne: 'Post-acne marks',
  smallDot: 'Small black dots',
  blemish: 'Blemishes',
  discoloration: 'Skin discoloration',
  scar: 'Acne scars',
  texture: 'Uneven texture',
  redness: 'Redness',
  underEye: 'Under-eye darkness',
};

/** Default [sensitivity, strength] per category; same as PERFECTION_DEFAULTS in schemas.py. */
export const PERFECTION_DEFAULTS: Readonly<Record<PerfectionCategoryName, readonly [number, number]>> = Object.freeze({
  acne: [75, 100],
  pimple: [75, 100],
  blackhead: [70, 100],
  whitehead: [70, 100],
  darkSpot: [70, 100],
  postAcne: [70, 90],
  smallDot: [70, 100],
  blemish: [75, 100],
  discoloration: [60, 50],
  scar: [50, 25],
  texture: [50, 20],
  redness: [60, 50],
  underEye: [50, 35],
});

/**
 * The main panel's detection sliders each drive a group of categories (moving one sets the
 * sensitivity of every category in it; the advanced list shows them one by one).
 */
export const PERFECTION_GROUPS = {
  acne: { label: 'Acne removal', categories: ['acne', 'pimple', 'whitehead'] },
  blackhead: { label: 'Blackhead removal', categories: ['blackhead', 'smallDot'] },
  darkSpot: { label: 'Dark spot correction', categories: ['darkSpot', 'postAcne', 'discoloration'] },
  smallBlemish: { label: 'Small blemish detection', categories: ['blemish'] },
} as const satisfies Record<string, { label: string; categories: readonly PerfectionCategoryName[] }>;
export type PerfectionGroup = keyof typeof PERFECTION_GROUPS;

export interface PerfectionCategory {
  enabled: boolean;
  /** 0..100 detection sensitivity. */
  sensitivity: number;
  /** 0..100 correction strength. */
  strength: number;
}

export interface PerfectionOverride {
  /** Normalised centre. */
  x: number;
  y: number;
  /** Radius as a share of the long edge. */
  radius: number;
  /** ignore: leave this mark alone. correct: correct this spot even if it was not detected. */
  action: 'ignore' | 'correct';
}

export interface SkinPerfectionSettings {
  enabled: boolean;
  /** 0..100 overall Skin Perfection strength. */
  strength: number;
  /** 1x natural, 2x strong, 3x maximum (sensitivity, radius and area corrections, safely capped). */
  multiplier: 1 | 2 | 3;
  categories: Record<PerfectionCategoryName, PerfectionCategory>;
  /** Neck, chest, shoulders, arms, hands, back and legs (semantic body skin). */
  bodyCoverage: boolean;
  bodySensitivity: number;
  texturePreservation: number;
  porePreservation: number;
  correctionRadius: number;
  edgeProtection: number;
  confidenceThreshold: number;
  residualIntensity: number;
  /** 1..4 correction passes (each followed by a residual scan). */
  passes: number;
  preserveFreckles: boolean;
  preserveMoles: boolean;
  preserveBirthmarks: boolean;
  overrides: PerfectionOverride[];
}

export const MAX_PERFECTION_OVERRIDES = 2000;

function defaultCategories(): Record<PerfectionCategoryName, PerfectionCategory> {
  const out = {} as Record<PerfectionCategoryName, PerfectionCategory>;
  for (const k of PERFECTION_CATEGORIES) {
    const [sensitivity, strength] = PERFECTION_DEFAULTS[k];
    out[k] = Object.freeze({ enabled: true, sensitivity, strength });
  }
  return out;
}

export const DEFAULT_SKIN_PERFECTION: Readonly<SkinPerfectionSettings> = Object.freeze({
  enabled: false,
  strength: 100,
  multiplier: 1,
  categories: Object.freeze(defaultCategories()),
  bodyCoverage: true,
  bodySensitivity: 70,
  texturePreservation: 90,
  porePreservation: 90,
  correctionRadius: 50,
  edgeProtection: 60,
  confidenceThreshold: 50,
  residualIntensity: 70,
  passes: 2,
  preserveFreckles: true,
  preserveMoles: true,
  preserveBirthmarks: true,
  overrides: [],
});

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const num = (v: unknown, lo: number, hi: number, d: number) => (finite(v) ? clamp(v, lo, hi) : d);
const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d);

export function sanitizeSkinPerfection(input: unknown): SkinPerfectionSettings {
  const r = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>;
  const d = DEFAULT_SKIN_PERFECTION;
  const cats = (r.categories && typeof r.categories === 'object' ? r.categories : {}) as Record<string, unknown>;
  const categories = {} as Record<PerfectionCategoryName, PerfectionCategory>;
  for (const k of PERFECTION_CATEGORIES) {
    const c = (cats[k] && typeof cats[k] === 'object' ? cats[k] : {}) as Record<string, unknown>;
    categories[k] = {
      enabled: bool(c.enabled, true),
      sensitivity: num(c.sensitivity, 0, 100, d.categories[k].sensitivity),
      strength: num(c.strength, 0, 100, d.categories[k].strength),
    };
  }
  const overrides = Array.isArray(r.overrides)
    ? r.overrides
        .map((o): PerfectionOverride | null => {
          if (!o || typeof o !== 'object') return null;
          const v = o as Record<string, unknown>;
          if (!finite(v.x) || !finite(v.y)) return null;
          return {
            x: clamp(v.x, 0, 1),
            y: clamp(v.y, 0, 1),
            radius: num(v.radius, 0.0003, 0.05, 0.004),
            action: v.action === 'correct' ? 'correct' : 'ignore',
          };
        })
        .filter((o): o is PerfectionOverride => o !== null)
        .slice(-MAX_PERFECTION_OVERRIDES)
    : [];
  const m = Math.round(num(r.multiplier, 1, 3, 1));
  return {
    enabled: bool(r.enabled, d.enabled),
    strength: num(r.strength, 0, 100, d.strength),
    multiplier: (m === 2 ? 2 : m === 3 ? 3 : 1) as 1 | 2 | 3,
    categories,
    bodyCoverage: bool(r.bodyCoverage, d.bodyCoverage),
    bodySensitivity: num(r.bodySensitivity, 0, 100, d.bodySensitivity),
    texturePreservation: num(r.texturePreservation, 0, 100, d.texturePreservation),
    porePreservation: num(r.porePreservation, 0, 100, d.porePreservation),
    correctionRadius: num(r.correctionRadius, 0, 100, d.correctionRadius),
    edgeProtection: num(r.edgeProtection, 0, 100, d.edgeProtection),
    confidenceThreshold: num(r.confidenceThreshold, 0, 100, d.confidenceThreshold),
    residualIntensity: num(r.residualIntensity, 0, 100, d.residualIntensity),
    passes: Math.round(num(r.passes, 1, 4, d.passes)),
    preserveFreckles: bool(r.preserveFreckles, d.preserveFreckles),
    preserveMoles: bool(r.preserveMoles, d.preserveMoles),
    preserveBirthmarks: bool(r.preserveBirthmarks, d.preserveBirthmarks),
    overrides,
  };
}

export function skinPerfectionIsDefault(s: SkinPerfectionSettings): boolean {
  return !s.enabled && s.overrides.length === 0;
}

/** Value shown on a group slider: the highest sensitivity among its categories. */
export function groupSensitivity(s: SkinPerfectionSettings, group: PerfectionGroup): number {
  return Math.max(...PERFECTION_GROUPS[group].categories.map((k) => s.categories[k].sensitivity));
}

/** Set every category of a group to one sensitivity. */
export function setGroupSensitivity(s: SkinPerfectionSettings, group: PerfectionGroup, value: number): SkinPerfectionSettings {
  const categories = { ...s.categories };
  for (const k of PERFECTION_GROUPS[group].categories) categories[k] = { ...categories[k], sensitivity: clamp(value, 0, 100) };
  return { ...s, categories };
}

/** Everything that changes the engine's result (overrides included); the preview cache key. */
export function perfectionKey(s: SkinPerfectionSettings): string {
  return JSON.stringify(s);
}

// --- report (POST /v1/perfection) -----------------------------------------------------------

export type MarkStatus = 'corrected' | 'remaining' | 'review' | 'protected' | 'ignored';

export interface PerfectionMark {
  id: number;
  /** Normalised centre; r is a share of the long edge. */
  x: number;
  y: number;
  r: number;
  kind: string;
  confidence: number;
  region: 'face' | 'body';
  status: MarkStatus;
  attempts: number;
  residual: boolean;
  contrast: number;
  after: number | null;
}

export interface PerfectionReport {
  width: number;
  height: number;
  passes: number;
  tiles: number;
  workScale: number;
  faceWidth: number;
  ms: number;
  /** remaining includes the low-confidence review marks; detected excludes protected ones. */
  counts: { detected: number; corrected: number; remaining: number; review: number; protected: number; ignored: number };
  byKind: Record<string, number>;
  meanConfidence: number | null;
  textureRetention: number | null;
  areas: Record<string, number>;
  marks: PerfectionMark[];
}

/** Overrides that correct every remaining mark at or above `minConfidence` (one undo step). */
export function correctRemaining(s: SkinPerfectionSettings, report: PerfectionReport, minConfidence: number): SkinPerfectionSettings {
  const add = report.marks
    .filter((m) => (m.status === 'remaining' || m.status === 'review') && m.confidence >= minConfidence)
    .map((m): PerfectionOverride => ({ x: m.x, y: m.y, radius: Math.max(0.0005, m.r * 1.3), action: 'correct' }));
  return { ...s, overrides: [...s.overrides, ...add].slice(-MAX_PERFECTION_OVERRIDES) };
}

/** Add (or flip) an override at one mark. */
export function overrideMark(s: SkinPerfectionSettings, mark: Pick<PerfectionMark, 'x' | 'y' | 'r'>, action: PerfectionOverride['action']): SkinPerfectionSettings {
  const rest = s.overrides.filter((o) => Math.hypot(o.x - mark.x, o.y - mark.y) > Math.max(o.radius, mark.r));
  const o: PerfectionOverride = { x: mark.x, y: mark.y, radius: Math.max(0.0005, mark.r * 1.3), action };
  return { ...s, overrides: [...rest, o].slice(-MAX_PERFECTION_OVERRIDES) };
}
