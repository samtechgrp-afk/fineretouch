'use client';

import {
  DEFAULT_ADJUSTMENTS,
  DEFAULT_EFFECT_CONTROL,
  EFFECT_SOURCE,
  applyPreset,
  autoRetouch,
  mapStyleToAdjustments,
  sanitizeAdjustments,
  smartSyncAdjustments,
  withOwnStrokes,
  type AdjustmentKey,
  type Adjustments,
  type ColorMatch,
  type ColorStats,
  type DepthBokehSettings,
  type EffectControl,
  type EffectId,
  type ImageAnalysis,
  type MaskStroke,
  type SkinMaskSettings,
  type Preset,
  type SkinMultiplier,
  type StrayHairSettings,
  type StyleProfile,
  type StyleSyncMode,
} from '@fineretouch/shared';
import { create } from 'zustand';
import type { StudioOverlay } from '@/components/viewport/shaders';
import { loadSavedEdits, saveEdits } from './editPersistence';
import type { RawInfo } from './images';

export const HISTORY_LIMIT = 100;

export type AnalysisState = 'idle' | 'pending' | 'done' | 'error' | 'offline';
export type CompareMode = 'edited' | 'original' | 'split' | 'side' | 'diff';
/** Where a photo's depth map came from: the AI service's model, or the client-side portrait prior. */
export type DepthState = 'none' | 'pending' | 'model' | 'prior' | 'error';
/** Engine segmentation (clothes / hair / person) for the clothing tools' live preview. */
export type SegmentState = 'none' | 'pending' | 'ready' | 'unavailable';

export interface PaletteSwatch {
  rgb: [number, number, number];
  share: number;
}

/** "Sync style from reference": the reference look and how to apply it. */
export interface StyleSyncState {
  referenceImageId: string | null;
  referenceName: string | null;
  /** Object URL of a small preview of the reference. */
  referenceUrl: string | null;
  profile: StyleProfile | null;
  matchedPalette: PaletteSwatch[];
  mode: StyleSyncMode;
  /** 0..100 strength of the transfer. */
  amount: number;
  /** true: apply to every photo in the shoot ("Sync to All"); false: selection or active photo. */
  syncTargetBatch: boolean;
}

export interface Photo {
  id: string;
  name: string;
  /** Original pixel size (after EXIF rotation). */
  width: number;
  height: number;
  /** Object URL of a small JPEG thumbnail of the original. */
  thumbUrl: string;
  adjustments: Adjustments;
  past: Adjustments[];
  future: Adjustments[];
  analysis: ImageAnalysis | null;
  analysisState: AnalysisState;
  /** Client-side L*a*b* stats of the proxy, used when the AI service is offline. */
  localStats: ColorStats | null;
  /** Style metrics of the original (for style sync, smart sync and auto retouch). */
  profile: StyleProfile | null;
  depthState: DepthState;
  segmentState: SegmentState;
  /** AI face parsing for the semantic skin mask (same states as segmentation). */
  parseState: SegmentState;
  rejected: boolean;
  /** Camera RAW original: format, badge label and EXIF (null for JPEG/PNG/...). */
  raw?: RawInfo | null;
  /** name|bytes|lastModified|WxH of the original: edits are autosaved under it and restored on re-import. */
  fingerprint?: string;
  /** True when this photo's edits were restored from a previous session. */
  restored?: boolean;
}

export interface ViewState {
  mode: CompareMode;
  split: number; // 0..1
  /** 1 = fit to viewport. Absolute zoom is fitScale * zoom. */
  zoom: number;
  panX: number; // CSS px
  panY: number;
  showMask: boolean;
  /** Skin mask brush in the viewport (null = pan / compare as usual). */
  maskBrush: 'add' | 'erase' | null;
  /** Brush radius as a share of the photo's long edge. */
  brushSize: number;
  /** Pro studio QA overlay in the viewport (never exported). */
  overlay: StudioOverlay;
}

export interface EditorState {
  photos: Record<string, Photo>;
  order: string[];
  activeId: string | null;
  selected: string[];
  view: ViewState;
  clipboard: Adjustments | null;
  customPresets: Preset[];
  aiOnline: boolean | null;
  styleSync: StyleSyncState;

  addPhotos(photos: Array<Pick<Photo, 'id' | 'name' | 'width' | 'height' | 'thumbUrl'> & Partial<Pick<Photo, 'raw' | 'fingerprint'>>>): void;
  removePhotos(ids: string[]): void;
  /** Reorder a subset of photos (e.g. back to file order after a parallel import). */
  sortPhotos(ids: string[]): void;
  setActive(id: string): void;
  toggleSelected(id: string, additive: boolean, range: boolean): void;
  selectAll(): void;
  /** Start of a slider drag or any discrete edit: snapshot for undo. */
  beginEdit(id?: string): void;
  /** Live change during a drag; does not create history entries. */
  setAdjustment(key: AdjustmentKey, value: number, id?: string): void;
  /** Discrete change that records its own undo step. */
  updateAdjustments(patch: Partial<Adjustments>, ids?: string[]): void;
  setColorMatch(match: ColorMatch | null, ids?: string[]): void;
  /** Live change (e.g. during a drag) without an undo step. */
  patchAdjustments(patch: Partial<Adjustments>, id?: string): void;
  setSkinMultiplier(multiplier: SkinMultiplier, ids?: string[]): void;
  /** Per-effect enable / multiplier / opacity. `live` skips history (opacity drags). */
  setEffectControl(id: EffectId, patch: Partial<EffectControl>, opts?: { live?: boolean; ids?: string[] }): void;
  /** Effect slider back to its default and its controls to enabled, 1x, 100%. */
  resetEffect(id: EffectId, ids?: string[]): void;
  /** Master retouch strength (0..100). `live` skips history. */
  setGlobalStrength(value: number, opts?: { live?: boolean; ids?: string[] }): void;
  /** `live` skips history (slider drags); call beginEdit() at drag start. */
  /** Skin mask options (parsing, body skin, feather); strokes go through addMaskStroke. */
  setSkinMask(patch: Partial<Omit<SkinMaskSettings, 'strokes'>>, opts?: { live?: boolean; ids?: string[] }): void;
  /** One brush stroke on the active photo's skin mask (one undo step). */
  addMaskStroke(stroke: MaskStroke): void;
  /** Remove the active photo's brush strokes (one undo step). */
  clearMaskStrokes(): void;
  setStrayHairRemoval(patch: Partial<StrayHairSettings>, opts?: { live?: boolean; ids?: string[] }): void;
  setDepthBokeh(patch: Partial<DepthBokehSettings>, opts?: { live?: boolean; ids?: string[] }): void;
  setStyleReference(ref: Pick<StyleSyncState, 'referenceImageId' | 'referenceName' | 'referenceUrl' | 'profile' | 'matchedPalette'> | null): void;
  setStyleSyncOptions(patch: Partial<Pick<StyleSyncState, 'mode' | 'amount' | 'syncTargetBatch'>>): void;
  /** Apply the reference look; returns how many photos changed. */
  applyStyleSync(): number;
  /** Copy the active photo's edits to the selection (or all), compensating exposure/WB per photo. */
  smartSync(): number;
  /** One-click AI starting point from each photo's own measurements. */
  autoRetouch(ids?: string[]): number;
  setProfile(id: string, profile: StyleProfile): void;
  setDepthState(id: string, state: DepthState): void;
  setSegmentState(id: string, state: SegmentState): void;
  setParseState(id: string, state: SegmentState): void;
  undo(id?: string): void;
  redo(id?: string): void;
  resetPhotos(ids?: string[]): void;
  copySettings(): void;
  pasteSettings(ids?: string[]): void;
  syncToSelected(): void;
  applyPresetTo(preset: Preset, ids?: string[]): void;
  saveCustomPreset(name: string): Preset | null;
  deleteCustomPreset(id: string): void;
  setAnalysis(id: string, state: AnalysisState, analysis?: ImageAnalysis | null): void;
  setLocalStats(id: string, stats: ColorStats): void;
  toggleRejected(id: string): void;
  setView(patch: Partial<ViewState>): void;
  setAiOnline(online: boolean): void;
  navigate(delta: number): void;
}

const PRESET_STORAGE_KEY = 'fineretouch.customPresets.v1';

function loadCustomPresets(): Preset[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(PRESET_STORAGE_KEY);
    const list = raw ? (JSON.parse(raw) as Preset[]) : [];
    return Array.isArray(list)
      ? list.map((p) => ({ ...p, builtIn: false, adjustments: sanitizeAdjustments(p.adjustments) }))
      : [];
  } catch {
    return [];
  }
}

function persistCustomPresets(presets: Preset[]) {
  try {
    window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(presets));
  } catch {
    /* storage may be unavailable (private mode); presets stay in memory */
  }
}

const withHistory = (p: Photo, next: Adjustments): Photo => ({
  ...p,
  past: [...p.past, p.adjustments].slice(-HISTORY_LIMIT),
  future: [],
  adjustments: next,
});

/** Ids an action should target: explicit ids, else the selection, else the active photo. */
function targets(s: EditorState, ids?: string[]): string[] {
  if (ids?.length) return ids.filter((id) => s.photos[id]);
  if (s.selected.length) return s.selected;
  return s.activeId ? [s.activeId] : [];
}

function mapPhotos(s: EditorState, ids: string[], fn: (p: Photo) => Photo): Record<string, Photo> {
  const photos = { ...s.photos };
  for (const id of ids) if (photos[id]) photos[id] = fn(photos[id]);
  return photos;
}

export const useEditor = create<EditorState>()((set, get) => ({
  photos: {},
  order: [],
  activeId: null,
  selected: [],
  view: { mode: 'edited', split: 0.5, zoom: 1, panX: 0, panY: 0, showMask: false, maskBrush: null, brushSize: 0.03, overlay: 'none' },
  clipboard: null,
  customPresets: loadCustomPresets(),
  aiOnline: null,
  styleSync: {
    referenceImageId: null,
    referenceName: null,
    referenceUrl: null,
    profile: null,
    matchedPalette: [],
    mode: 'palette',
    amount: 100,
    syncTargetBatch: false,
  },

  addPhotos(list) {
    set((s) => {
      const photos = { ...s.photos };
      const order = [...s.order];
      for (const p of list) {
        if (photos[p.id]) continue;
        const saved = p.fingerprint ? loadSavedEdits(p.fingerprint) : null;
        photos[p.id] = {
          ...p,
          adjustments: saved ?? { ...DEFAULT_ADJUSTMENTS },
          restored: Boolean(saved),
          past: [],
          future: [],
          analysis: null,
          analysisState: 'idle',
          localStats: null,
          profile: null,
          depthState: 'none',
          segmentState: 'none',
          parseState: 'none',
          rejected: false,
        };
        order.push(p.id);
      }
      return { photos, order, activeId: s.activeId ?? list[0]?.id ?? null };
    });
  },

  removePhotos(ids) {
    set((s) => {
      const photos = { ...s.photos };
      for (const id of ids) {
        if (photos[id]) URL.revokeObjectURL(photos[id].thumbUrl);
        delete photos[id];
      }
      const order = s.order.filter((id) => photos[id]);
      const activeId = s.activeId && photos[s.activeId] ? s.activeId : (order[0] ?? null);
      return { photos, order, activeId, selected: s.selected.filter((id) => photos[id]) };
    });
  },

  sortPhotos(ids) {
    set((s) => {
      const wanted = ids.filter((id) => s.photos[id]);
      const slots = s.order.map((id, i) => (wanted.includes(id) ? i : -1)).filter((i) => i >= 0);
      const order = [...s.order];
      slots.forEach((slot, k) => (order[slot] = wanted[k]));
      return { order };
    });
  },

  setActive(id) {
    if (get().photos[id]) set({ activeId: id, view: { ...get().view, zoom: 1, panX: 0, panY: 0 } });
  },

  toggleSelected(id, additive, range) {
    set((s) => {
      if (range && s.activeId) {
        const a = s.order.indexOf(s.activeId);
        const b = s.order.indexOf(id);
        const [lo, hi] = a < b ? [a, b] : [b, a];
        return { selected: Array.from(new Set([...s.selected, ...s.order.slice(lo, hi + 1)])) };
      }
      if (additive) {
        const has = s.selected.includes(id);
        const base = s.selected.length || !s.activeId ? s.selected : [s.activeId];
        return { selected: has ? base.filter((x) => x !== id) : [...base, id] };
      }
      return { selected: [], activeId: id };
    });
  },

  selectAll() {
    set((s) => ({ selected: [...s.order] }));
  },

  beginEdit(id) {
    const s = get();
    const pid = id ?? s.activeId;
    if (!pid || !s.photos[pid]) return;
    const p = s.photos[pid];
    // Skip duplicate snapshots (e.g. a click on a slider that did not move it).
    if (p.past.length && JSON.stringify(p.past[p.past.length - 1]) === JSON.stringify(p.adjustments)) return;
    set({ photos: mapPhotos(s, [pid], (p) => withHistory(p, p.adjustments)) });
  },

  setAdjustment(key, value, id) {
    const s = get();
    const pid = id ?? s.activeId;
    if (!pid || !s.photos[pid]) return;
    set({
      photos: mapPhotos(s, [pid], (p) => ({ ...p, adjustments: sanitizeAdjustments({ ...p.adjustments, [key]: value }) })),
    });
  },

  updateAdjustments(patch, ids) {
    const s = get();
    set({
      photos: mapPhotos(s, targets(s, ids), (p) => withHistory(p, sanitizeAdjustments({ ...p.adjustments, ...patch }))),
    });
  },

  setColorMatch(match, ids) {
    const s = get();
    set({ photos: mapPhotos(s, targets(s, ids), (p) => withHistory(p, { ...p.adjustments, colorMatch: match })) });
  },

  patchAdjustments(patch, id) {
    const s = get();
    const pid = id ?? s.activeId;
    if (!pid || !s.photos[pid]) return;
    set({ photos: mapPhotos(s, [pid], (p) => ({ ...p, adjustments: sanitizeAdjustments({ ...p.adjustments, ...patch }) })) });
  },

  setSkinMultiplier(multiplier, ids) {
    get().updateAdjustments({ skinMultiplier: multiplier }, ids);
  },

  setEffectControl(id, patch, opts) {
    const s = get();
    const apply = (a: Adjustments) => sanitizeAdjustments({ ...a, effects: { ...a.effects, [id]: { ...a.effects[id], ...patch } } });
    if (opts?.live) {
      const p = selectActivePhoto(s);
      if (p) set({ photos: mapPhotos(s, [p.id], (q) => ({ ...q, adjustments: apply(q.adjustments) })) });
      return;
    }
    set({ photos: mapPhotos(s, targets(s, opts?.ids), (p) => withHistory(p, apply(p.adjustments))) });
  },

  resetEffect(id, ids) {
    const s = get();
    const src = EFFECT_SOURCE[id];
    const apply = (a: Adjustments): Adjustments => {
      const next: Record<string, unknown> = { ...a, effects: { ...a.effects, [id]: { ...DEFAULT_EFFECT_CONTROL } } };
      if (src.length === 1) next[src[0]] = DEFAULT_ADJUSTMENTS[src[0]];
      else {
        const [key, field] = src as [keyof Adjustments, string];
        next[key] = { ...(a[key] as object), [field]: (DEFAULT_ADJUSTMENTS[key] as unknown as Record<string, number>)[field] };
      }
      return sanitizeAdjustments(next);
    };
    set({ photos: mapPhotos(s, targets(s, ids), (p) => withHistory(p, apply(p.adjustments))) });
  },

  setGlobalStrength(value, opts) {
    if (opts?.live) get().patchAdjustments({ globalStrength: value });
    else get().updateAdjustments({ globalStrength: value }, opts?.ids);
  },

  setSkinMask(patch, opts) {
    const s = get();
    const apply = (a: Adjustments) => sanitizeAdjustments({ ...a, skinMask: { ...a.skinMask, ...patch } });
    if (opts?.live) {
      const p = selectActivePhoto(s);
      if (p) set({ photos: mapPhotos(s, [p.id], (q) => ({ ...q, adjustments: apply(q.adjustments) })) });
      return;
    }
    set({ photos: mapPhotos(s, targets(s, opts?.ids), (p) => withHistory(p, apply(p.adjustments))) });
  },

  addMaskStroke(stroke) {
    const s = get();
    const p = selectActivePhoto(s);
    if (!p || stroke.points.length < 2) return;
    const next = sanitizeAdjustments({ ...p.adjustments, skinMask: { ...p.adjustments.skinMask, strokes: [...p.adjustments.skinMask.strokes, stroke] } });
    set({ photos: mapPhotos(s, [p.id], (q) => withHistory(q, next)) });
  },

  clearMaskStrokes() {
    const s = get();
    const p = selectActivePhoto(s);
    if (!p || !p.adjustments.skinMask.strokes.length) return;
    set({ photos: mapPhotos(s, [p.id], (q) => withHistory(q, { ...q.adjustments, skinMask: { ...q.adjustments.skinMask, strokes: [] } })) });
  },

  setStrayHairRemoval(patch, opts) {
    const s = get();
    if (opts?.live) {
      const p = selectActivePhoto(s);
      if (p) s.patchAdjustments({ strayHairRemoval: { ...p.adjustments.strayHairRemoval, ...patch } });
      return;
    }
    set({
      photos: mapPhotos(s, targets(s, opts?.ids), (p) =>
        withHistory(p, sanitizeAdjustments({ ...p.adjustments, strayHairRemoval: { ...p.adjustments.strayHairRemoval, ...patch } })),
      ),
    });
  },

  setDepthBokeh(patch, opts) {
    const s = get();
    if (opts?.live) {
      const p = selectActivePhoto(s);
      if (p) s.patchAdjustments({ depthBokeh: { ...p.adjustments.depthBokeh, ...patch } });
      return;
    }
    set({
      photos: mapPhotos(s, targets(s, opts?.ids), (p) =>
        withHistory(p, sanitizeAdjustments({ ...p.adjustments, depthBokeh: { ...p.adjustments.depthBokeh, ...patch } })),
      ),
    });
  },

  setStyleReference(ref) {
    set((s) => {
      if (s.styleSync.referenceUrl && s.styleSync.referenceUrl !== ref?.referenceUrl) URL.revokeObjectURL(s.styleSync.referenceUrl);
      return {
        styleSync: ref
          ? { ...s.styleSync, ...ref }
          : { ...s.styleSync, referenceImageId: null, referenceName: null, referenceUrl: null, profile: null, matchedPalette: [] },
      };
    });
  },

  setStyleSyncOptions(patch) {
    set((s) => ({ styleSync: { ...s.styleSync, ...patch } }));
  },

  applyStyleSync() {
    const s = get();
    const { profile: ref, mode, amount, syncTargetBatch } = s.styleSync;
    if (!ref) return 0;
    const ids = (syncTargetBatch ? s.order : targets(s)).filter((id) => s.photos[id]?.profile);
    set({
      photos: mapPhotos(s, ids, (p) =>
        withHistory(p, sanitizeAdjustments({ ...p.adjustments, ...mapStyleToAdjustments(ref, p.profile!, mode, amount) })),
      ),
    });
    return ids.length;
  },

  smartSync() {
    const s = get();
    if (!s.activeId) return 0;
    const hero = s.photos[s.activeId];
    const heroStats = hero.analysis?.colorStats ?? hero.localStats;
    if (!heroStats) return 0;
    const ids = (s.selected.length > 1 ? s.selected : s.order).filter((id) => id !== s.activeId);
    let n = 0;
    set({
      photos: mapPhotos(s, ids, (p) => {
        const st = p.analysis?.colorStats ?? p.localStats;
        if (!st) return p;
        n++;
        return withHistory(p, withOwnStrokes(smartSyncAdjustments(hero.adjustments, heroStats, st), p.adjustments));
      }),
    });
    return n;
  },

  autoRetouch(ids) {
    const s = get();
    const list = targets(s, ids).filter((id) => s.photos[id].profile);
    set({
      photos: mapPhotos(s, list, (p) =>
        withHistory(p, sanitizeAdjustments({ ...p.adjustments, ...autoRetouch(p.profile!, Boolean(p.analysis?.faces.length)) })),
      ),
    });
    return list.length;
  },

  setProfile(id, profile) {
    const s = get();
    if (s.photos[id]) set({ photos: { ...s.photos, [id]: { ...s.photos[id], profile } } });
  },

  setParseState(id, state) {
    const s = get();
    if (s.photos[id]) set({ photos: { ...s.photos, [id]: { ...s.photos[id], parseState: state } } });
  },

  setSegmentState(id, state) {
    const s = get();
    if (s.photos[id]) set({ photos: { ...s.photos, [id]: { ...s.photos[id], segmentState: state } } });
  },

  setDepthState(id, state) {
    const s = get();
    if (s.photos[id]) set({ photos: { ...s.photos, [id]: { ...s.photos[id], depthState: state } } });
  },

  undo(id) {
    const s = get();
    const pid = id ?? s.activeId;
    const p = pid ? s.photos[pid] : null;
    if (!p || !p.past.length) return;
    const prev = p.past[p.past.length - 1];
    set({
      photos: { ...s.photos, [p.id]: { ...p, adjustments: prev, past: p.past.slice(0, -1), future: [p.adjustments, ...p.future] } },
    });
  },

  redo(id) {
    const s = get();
    const pid = id ?? s.activeId;
    const p = pid ? s.photos[pid] : null;
    if (!p || !p.future.length) return;
    const [next, ...rest] = p.future;
    set({
      photos: { ...s.photos, [p.id]: { ...p, adjustments: next, past: [...p.past, p.adjustments].slice(-HISTORY_LIMIT), future: rest } },
    });
  },

  resetPhotos(ids) {
    const s = get();
    set({ photos: mapPhotos(s, targets(s, ids), (p) => withHistory(p, { ...DEFAULT_ADJUSTMENTS })) });
  },

  copySettings() {
    const s = get();
    if (s.activeId) set({ clipboard: { ...s.photos[s.activeId].adjustments } });
  },

  pasteSettings(ids) {
    const s = get();
    if (!s.clipboard) return;
    const clip = s.clipboard;
    set({ photos: mapPhotos(s, targets(s, ids), (p) => withHistory(p, withOwnStrokes({ ...clip }, p.adjustments))) });
  },

  syncToSelected() {
    const s = get();
    if (!s.activeId) return;
    const src = s.photos[s.activeId].adjustments;
    const ids = (s.selected.length > 1 ? s.selected : s.order).filter((id) => id !== s.activeId);
    set({ photos: mapPhotos(s, ids, (p) => withHistory(p, withOwnStrokes({ ...src }, p.adjustments))) });
  },

  applyPresetTo(preset, ids) {
    const s = get();
    set({ photos: mapPhotos(s, targets(s, ids), (p) => withHistory(p, applyPreset(p.adjustments, preset, 'replace'))) });
  },

  saveCustomPreset(name) {
    const s = get();
    if (!s.activeId || !name.trim()) return null;
    const preset: Preset = {
      id: `custom-${Date.now().toString(36)}`,
      name: name.trim().slice(0, 60),
      description: 'Saved from your edits',
      // Brush strokes belong to this photo's pixels, not to a reusable preset.
      adjustments: { ...s.photos[s.activeId].adjustments, skinMask: { ...s.photos[s.activeId].adjustments.skinMask, strokes: [] } },
    };
    const customPresets = [...s.customPresets, preset];
    persistCustomPresets(customPresets);
    set({ customPresets });
    return preset;
  },

  deleteCustomPreset(id) {
    const customPresets = get().customPresets.filter((p) => p.id !== id);
    persistCustomPresets(customPresets);
    set({ customPresets });
  },

  setAnalysis(id, state, analysis) {
    const s = get();
    if (!s.photos[id]) return;
    set({ photos: { ...s.photos, [id]: { ...s.photos[id], analysisState: state, analysis: analysis ?? s.photos[id].analysis } } });
  },

  setLocalStats(id, stats) {
    const s = get();
    if (s.photos[id]) set({ photos: { ...s.photos, [id]: { ...s.photos[id], localStats: stats } } });
  },

  toggleRejected(id) {
    const s = get();
    if (s.photos[id]) set({ photos: { ...s.photos, [id]: { ...s.photos[id], rejected: !s.photos[id].rejected } } });
  },

  setView(patch) {
    set((s) => ({ view: { ...s.view, ...patch } }));
  },

  setAiOnline(online) {
    set({ aiOnline: online });
  },

  navigate(delta) {
    const s = get();
    if (!s.order.length) return;
    const i = s.activeId ? s.order.indexOf(s.activeId) : -1;
    const next = s.order[Math.min(s.order.length - 1, Math.max(0, i + delta))];
    get().setActive(next);
  },
}));

export const selectActivePhoto = (s: EditorState): Photo | null => (s.activeId ? s.photos[s.activeId] ?? null : null);

// Autosave: every photo's current edits are written (debounced) under its file fingerprint, so
// re-importing the same original after a reload or crash restores them. The original is never touched.
if (typeof window !== 'undefined') {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let last: Record<string, Photo> | null = null;
  useEditor.subscribe((state) => {
    if (state.photos === last) return;
    last = state.photos;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      const entries: Array<[string, Adjustments]> = [];
      for (const p of Object.values(useEditor.getState().photos)) if (p.fingerprint) entries.push([p.fingerprint, p.adjustments]);
      saveEdits(entries);
    }, 600);
  });
}
