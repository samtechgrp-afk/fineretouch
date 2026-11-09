/**
 * QA label files for the Skin Perfection real-photo validation set (v0.13 phase 2A).
 *
 * The app's label mode writes `<photo>.labels.json` beside the photo (never into it); the
 * benchmark (services/ai/qa/dataset.py) reads it. Kinds are pinned to the Python list by a test.
 * Labels pre-filled from the engine's report start unconfirmed: the benchmark ignores them until
 * a person confirms each one, so the engine is never graded against its own output.
 */

import type { PerfectionReport } from './perfection.js';

export const LABEL_FORMAT = 'fineretouch-labels';
export const LABEL_REMOVE_KINDS = ['acne', 'pimple', 'blackhead', 'whitehead', 'darkSpot', 'postAcne', 'smallDot', 'blemish', 'scar'] as const;
export const LABEL_KEEP_KINDS = ['mole', 'freckle', 'birthmark', 'follicle', 'hair'] as const;
export type LabelKind = (typeof LABEL_REMOVE_KINDS)[number] | (typeof LABEL_KEEP_KINDS)[number];
export const LABEL_KINDS: readonly LabelKind[] = [...LABEL_REMOVE_KINDS, ...LABEL_KEEP_KINDS];

export const LABEL_KIND_NAMES: Record<LabelKind, string> = {
  acne: 'Acne',
  pimple: 'Pimple',
  blackhead: 'Blackhead',
  whitehead: 'Whitehead',
  darkSpot: 'Dark spot',
  postAcne: 'Post-acne mark',
  smallDot: 'Small dot',
  blemish: 'Blemish',
  scar: 'Acne scar',
  mole: 'Mole (keep)',
  freckle: 'Freckle (keep)',
  birthmark: 'Birthmark (keep)',
  follicle: 'Follicle / stubble (keep)',
  hair: 'Hair (keep)',
};

export const isKeepKind = (k: LabelKind) => (LABEL_KEEP_KINDS as readonly string[]).includes(k);

export interface QaLabel {
  id: string;
  /** Shares of width / height. */
  x: number;
  y: number;
  /** Share of the long edge. */
  r: number;
  kind: LabelKind;
  source: 'manual' | 'engine';
  confirmed: boolean;
  note?: string;
}

export interface LabelFile {
  format: typeof LABEL_FORMAT;
  version: 1;
  image: { name: string; width: number; height: number };
  annotator: string | null;
  reviewer: string | null;
  labels: QaLabel[];
}

export function makeLabelFile(image: LabelFile['image'], labels: QaLabel[], annotator: string | null = null, reviewer: string | null = null): LabelFile {
  return { format: LABEL_FORMAT, version: 1, image, annotator, reviewer, labels };
}

/** Parse and check a label file (throws with the reason). */
export function parseLabelFile(text: string): LabelFile {
  const d = JSON.parse(text) as Partial<LabelFile>;
  if (d.format !== LABEL_FORMAT || d.version !== 1) throw new Error('Not a FineRetouch label file');
  const im = d.image;
  if (!im || !(im.width > 0) || !(im.height > 0)) throw new Error('The label file has no image size');
  const labels = (d.labels ?? []).map((l, i) => {
    if (!LABEL_KINDS.includes(l.kind)) throw new Error(`Label ${i + 1} has an unknown kind: ${String(l.kind)}`);
    if (!(l.x >= 0 && l.x <= 1 && l.y >= 0 && l.y <= 1 && l.r > 0 && l.r <= 0.1)) throw new Error(`Label ${i + 1} is out of range`);
    return { id: String(l.id ?? `l${i + 1}`), x: l.x, y: l.y, r: l.r, kind: l.kind, source: l.source === 'engine' ? 'engine' : 'manual', confirmed: l.confirmed !== false, note: l.note } satisfies QaLabel;
  });
  return makeLabelFile({ name: String(im.name ?? ''), width: im.width, height: im.height }, labels, d.annotator ?? null, d.reviewer ?? null);
}

const ENGINE_KIND: Record<string, LabelKind | undefined> = {
  acne: 'acne', pimple: 'pimple', blackhead: 'blackhead', whitehead: 'whitehead', darkSpot: 'darkSpot', postAcne: 'postAcne',
  smallDot: 'smallDot', blemish: 'blemish', mole: 'mole', freckle: 'freckle', birthmark: 'birthmark', follicle: 'follicle',
};

/** Draft labels from the engine's report (unconfirmed; ignored by the benchmark until confirmed). */
export function labelsFromReport(report: PerfectionReport, existing: readonly QaLabel[] = []): QaLabel[] {
  const out: QaLabel[] = [];
  let n = 0;
  for (const m of report.marks) {
    const kind = ENGINE_KIND[m.kind];
    if (!kind) continue;
    if (existing.some((l) => Math.hypot(l.x - m.x, l.y - m.y) < Math.max(l.r, m.r))) continue;
    out.push({ id: `e${m.id}-${n++}`, x: m.x, y: m.y, r: Math.max(0.0005, m.r), kind, source: 'engine', confirmed: false });
  }
  return out;
}

/** The label under a point (normalised x / y; aspect = width / height), smallest first. */
export function labelAt(labels: readonly QaLabel[], x: number, y: number, aspect: number, slack = 0.002): QaLabel | null {
  // r is a share of the long edge; convert offsets to long-edge units.
  const sx = aspect >= 1 ? 1 : aspect;
  const sy = aspect >= 1 ? 1 / aspect : 1;
  let best: QaLabel | null = null;
  for (const l of labels) {
    const d = Math.hypot((l.x - x) * sx, (l.y - y) * sy);
    if (d <= l.r * 1.6 + slack && (!best || l.r < best.r)) best = l;
  }
  return best;
}

export function labelCounts(labels: readonly QaLabel[]): { remove: number; keep: number; unconfirmed: number } {
  let remove = 0;
  let keep = 0;
  let unconfirmed = 0;
  for (const l of labels) {
    if (isKeepKind(l.kind)) keep++;
    else remove++;
    if (!l.confirmed) unconfirmed++;
  }
  return { remove, keep, unconfirmed };
}
