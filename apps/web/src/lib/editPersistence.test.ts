import { DEFAULT_ADJUSTMENTS, DEFAULT_EFFECTS, sanitizeAdjustments } from '@fineretouch/shared';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { EDITS_STORAGE_KEY, MAX_SAVED_EDITS, fileFingerprint, loadSavedEdits, saveEdits } from './editPersistence';

const mem = new Map<string, string>();
vi.stubGlobal('window', {
  localStorage: {
    getItem: (k: string) => mem.get(k) ?? null,
    setItem: (k: string, v: string) => void mem.set(k, v),
    removeItem: (k: string) => void mem.delete(k),
  },
});

beforeEach(() => mem.clear());

describe('edit autosave', () => {
  const edited = sanitizeAdjustments({
    ...DEFAULT_ADJUSTMENTS, skinSmoothing: 45, globalStrength: 80,
    effects: { ...DEFAULT_EFFECTS, skinSmoothing: { enabled: true, multiplier: 2, opacity: 90 } },
  });

  it('fingerprints by name, size, mtime and pixel size', () => {
    expect(fileFingerprint({ name: 'IMG_1.jpg', size: 123, lastModified: 9 }, 6000, 4000)).toBe('IMG_1.jpg|123|9|6000x4000');
  });

  it('round-trips edits including effect controls', () => {
    saveEdits([['fp1', edited]]);
    expect(loadSavedEdits('fp1')).toEqual(edited);
    expect(loadSavedEdits('other')).toBeNull();
  });

  it('drops photos that are back at defaults', () => {
    saveEdits([['fp1', edited]]);
    saveEdits([['fp1', DEFAULT_ADJUSTMENTS]]);
    expect(loadSavedEdits('fp1')).toBeNull();
  });

  it('evicts the oldest entries beyond the limit', () => {
    for (let i = 0; i < MAX_SAVED_EDITS + 5; i++) saveEdits([[`fp${i}`, edited]], i);
    const all = JSON.parse(mem.get(EDITS_STORAGE_KEY)!);
    expect(Object.keys(all)).toHaveLength(MAX_SAVED_EDITS);
    expect(all.fp0).toBeUndefined();
    expect(all[`fp${MAX_SAVED_EDITS + 4}`]).toBeDefined();
  });

  it('survives corrupt storage', () => {
    mem.set(EDITS_STORAGE_KEY, '{not json');
    expect(loadSavedEdits('fp1')).toBeNull();
    saveEdits([['fp1', edited]]);
    expect(loadSavedEdits('fp1')).toEqual(edited);
  });
});
