/**
 * Edit autosave and crash recovery (v0.7).
 *
 * Each photo's adjustments are kept in localStorage under a fingerprint of its original file
 * (name, byte size, last-modified time and pixel size), so re-importing the same file after a
 * reload, crash or app restart brings its edits back. Only edit instructions are stored, never
 * pixels, and the original file is never written to. Entries are evicted oldest-first.
 */
import { isDefault, sanitizeAdjustments, type Adjustments } from '@fineretouch/shared';

export const EDITS_STORAGE_KEY = 'fineretouch.edits.v1';
export const MAX_SAVED_EDITS = 400;

interface Entry {
  a: Adjustments;
  t: number;
}

export function fileFingerprint(file: Pick<File, 'name' | 'size' | 'lastModified'>, width: number, height: number): string {
  return `${file.name}|${file.size}|${file.lastModified}|${width}x${height}`;
}

function storage(): Storage | null {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null;
  } catch {
    return null;
  }
}

function readAll(): Record<string, Entry> {
  try {
    const raw = storage()?.getItem(EDITS_STORAGE_KEY);
    const v = raw ? JSON.parse(raw) : {};
    return v && typeof v === 'object' ? (v as Record<string, Entry>) : {};
  } catch {
    return {};
  }
}

function writeAll(all: Record<string, Entry>): void {
  const st = storage();
  if (!st) return;
  let entries = Object.entries(all).sort((x, y) => y[1].t - x[1].t).slice(0, MAX_SAVED_EDITS);
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      if (entries.length) st.setItem(EDITS_STORAGE_KEY, JSON.stringify(Object.fromEntries(entries)));
      else st.removeItem(EDITS_STORAGE_KEY);
      return;
    } catch {
      entries = entries.slice(0, Math.floor(entries.length / 2)); // quota: keep the newest half
    }
  }
}

/** Saved edits for a fingerprint, validated, or null. */
export function loadSavedEdits(fingerprint: string): Adjustments | null {
  const e = readAll()[fingerprint];
  return e && e.a ? sanitizeAdjustments(e.a) : null;
}

/** Upsert the given photos' edits; photos back at defaults are removed. */
export function saveEdits(items: Array<[string, Adjustments]>, now = Date.now()): void {
  if (!items.length) return;
  const all = readAll();
  for (const [fp, a] of items) {
    if (isDefault(a)) delete all[fp];
    else all[fp] = { a, t: now };
  }
  writeAll(all);
}

export function clearSavedEdits(): void {
  try {
    storage()?.removeItem(EDITS_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
