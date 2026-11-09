'use client';

/**
 * Label mode (Skin Perfection QA, phase 2A): mark each imperfection and each natural feature on
 * a real photo, then save `<photo>.labels.json` for the benchmark
 * (services/ai/qa/dataset.py). Labels are not part of the edit: they never touch the photo, its
 * adjustments, undo history or exports. They are kept in this browser per photo fingerprint so a
 * reload does not lose work, and saved to a file only when asked.
 */

import { create } from 'zustand';
import {
  labelsFromReport,
  makeLabelFile,
  parseLabelFile,
  type LabelFile,
  type LabelKind,
  type PerfectionReport,
  type QaLabel,
} from '@fineretouch/shared';
import { isDesktop } from './desktop';
import { downloadBlob } from './export';
import { useEditor } from './store';

export interface PhotoLabels {
  labels: QaLabel[];
  annotator: string | null;
  reviewer: string | null;
  /** Changed since the last save or load. */
  dirty: boolean;
}

interface LabelState {
  /** Label mode: clicks on the photo place labels instead of selecting marks. */
  on: boolean;
  kind: LabelKind;
  /** New label radius as a share of the long edge. */
  radius: number;
  selected: string | null;
  byPhoto: Record<string, PhotoLabels>;
  setOn(on: boolean): void;
  setKind(kind: LabelKind): void;
  setRadius(r: number): void;
  select(id: string | null): void;
  add(photoId: string, x: number, y: number): QaLabel;
  update(photoId: string, labelId: string, patch: Partial<QaLabel>): void;
  remove(photoId: string, labelId: string): void;
  confirmAll(photoId: string): void;
  prefill(photoId: string, report: PerfectionReport): number;
  clear(photoId: string): void;
  setPeople(photoId: string, patch: Partial<Pick<PhotoLabels, 'annotator' | 'reviewer'>>): void;
  replace(photoId: string, file: LabelFile): void;
}

const EMPTY: PhotoLabels = { labels: [], annotator: null, reviewer: null, dirty: false };
const STORE_KEY = 'fineretouch.qaLabels.v1';
let seq = 0;
const newId = () => `l${Date.now().toString(36)}${(seq++).toString(36)}`;

function keyOf(photoId: string): string | null {
  const p = useEditor.getState().photos[photoId];
  return p?.fingerprint ?? null;
}

function readSaved(): Record<string, PhotoLabels> {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) ?? '{}') as Record<string, PhotoLabels>;
  } catch {
    return {};
  }
}

function persist(photoId: string, v: PhotoLabels) {
  const k = keyOf(photoId);
  if (!k) return;
  try {
    const all = readSaved();
    if (v.labels.length) all[k] = v;
    else delete all[k];
    localStorage.setItem(STORE_KEY, JSON.stringify(all));
  } catch {
    // Storage full or blocked: labels stay in memory until saved to a file.
  }
}

/** Labels kept in this browser for a photo (by fingerprint), if any. */
export function restoreLabels(photoId: string) {
  const k = keyOf(photoId);
  if (!k || useLabels.getState().byPhoto[photoId]) return;
  const saved = readSaved()[k];
  if (saved) useLabels.setState((s) => ({ byPhoto: { ...s.byPhoto, [photoId]: saved } }));
}

export const useLabels = create<LabelState>((set, get) => {
  const change = (photoId: string, f: (p: PhotoLabels) => PhotoLabels) => {
    const next = f(get().byPhoto[photoId] ?? EMPTY);
    set((s) => ({ byPhoto: { ...s.byPhoto, [photoId]: next } }));
    persist(photoId, next);
  };
  return {
    on: false,
    kind: 'acne',
    radius: 0.004,
    selected: null,
    byPhoto: {},
    setOn: (on) => set({ on, selected: null }),
    setKind: (kind) => set({ kind }),
    setRadius: (radius) => set({ radius: Math.min(0.05, Math.max(0.0005, radius)) }),
    select: (selected) => set({ selected }),
    add(photoId, x, y) {
      const l: QaLabel = { id: newId(), x, y, r: get().radius, kind: get().kind, source: 'manual', confirmed: true };
      change(photoId, (p) => ({ ...p, labels: [...p.labels, l], dirty: true }));
      set({ selected: l.id });
      return l;
    },
    update: (photoId, labelId, patch) =>
      change(photoId, (p) => ({ ...p, labels: p.labels.map((l) => (l.id === labelId ? { ...l, ...patch } : l)), dirty: true })),
    remove(photoId, labelId) {
      change(photoId, (p) => ({ ...p, labels: p.labels.filter((l) => l.id !== labelId), dirty: true }));
      if (get().selected === labelId) set({ selected: null });
    },
    confirmAll: (photoId) => change(photoId, (p) => ({ ...p, labels: p.labels.map((l) => ({ ...l, confirmed: true })), dirty: true })),
    prefill(photoId, report) {
      const add = labelsFromReport(report, get().byPhoto[photoId]?.labels ?? []);
      change(photoId, (p) => ({ ...p, labels: [...p.labels, ...add], dirty: true }));
      return add.length;
    },
    clear: (photoId) => change(photoId, (p) => ({ ...p, labels: [], dirty: true })),
    setPeople: (photoId, patch) => change(photoId, (p) => ({ ...p, ...patch, dirty: true })),
    replace: (photoId, file) =>
      change(photoId, () => ({ labels: file.labels, annotator: file.annotator, reviewer: file.reviewer, dirty: false })),
  };
});

export function labelFileFor(photoId: string): LabelFile | null {
  const p = useEditor.getState().photos[photoId];
  if (!p) return null;
  const v = useLabels.getState().byPhoto[photoId] ?? EMPTY;
  return makeLabelFile({ name: p.name, width: p.width, height: p.height }, v.labels, v.annotator, v.reviewer);
}

/** Save `<photo name>.labels.json` (a Save As dialog in the desktop app, a download on the web). */
export async function saveLabels(photoId: string): Promise<string | null> {
  const file = labelFileFor(photoId);
  if (!file) return null;
  const name = `${file.image.name}.labels.json`;
  const text = JSON.stringify(file, null, 1);
  if (isDesktop()) {
    const { save } = await import('@tauri-apps/plugin-dialog');
    const { writeFile } = await import('@tauri-apps/plugin-fs');
    const path = await save({ title: 'Save labels', defaultPath: name, filters: [{ name: 'FineRetouch labels', extensions: ['json'] }] });
    if (!path) return null;
    // writeFile: the permission exports already use; the Save As pick adds the path to the scope.
    await writeFile(path, new TextEncoder().encode(text));
  } else {
    downloadBlob(new Blob([text], { type: 'application/json' }), name);
  }
  useLabels.setState((s) => ({ byPhoto: { ...s.byPhoto, [photoId]: { ...(s.byPhoto[photoId] ?? EMPTY), dirty: false } } }));
  return name;
}

/** Load a label file for this photo (checks that it was made for a photo of the same shape). */
export async function loadLabels(photoId: string, f: File): Promise<number> {
  const file = parseLabelFile(await f.text());
  const p = useEditor.getState().photos[photoId];
  if (!p) return 0;
  const a = file.image.width / file.image.height;
  const b = p.width / p.height;
  if (Math.abs(a - b) > 0.01 * b) throw new Error(`These labels are for a ${file.image.width}×${file.image.height} photo, not this one.`);
  useLabels.getState().replace(photoId, file);
  return file.labels.length;
}
