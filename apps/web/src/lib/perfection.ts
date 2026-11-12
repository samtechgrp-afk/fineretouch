'use client';

/**
 * Skin Perfection in the app (v0.13). The engine (services/ai/app/engines/perfection.py) runs at
 * full resolution in the local AI engine; the browser keeps:
 *
 *  - the latest report per photo (counts, marks with class / confidence / status) for the panel
 *    and the viewport overlay;
 *  - the perfected photo at preview size, used by the viewport as its source pixels so every
 *    other slider stays live on top of it (the export renders everything in the engine).
 *
 * The Professional Skin Finish (v0.14) also runs only in the engine, after Skin Perfection: with
 * either on, the engine returns the photo with both applied and that is the viewport's source.
 *
 * Results are cached per photo by a key of everything that changes them (the Skin Perfection
 * settings with the user's overrides, the Skin Finish settings and the skin mask), so undo / redo back to an earlier
 * state is instant. Settings changes re-run the engine after a short pause (one run at a time
 * per photo; a change during a run queues exactly one more).
 */

import { create } from 'zustand';
import type { Adjustments, PerfectionReport } from '@fineretouch/shared';

/** The Skin Finish part of the engine's answer: what each correction measured before / after. */
export interface FinishSummary {
  ms?: number;
  edgeKept?: number | null;
  textureKept?: number | null;
  local?: Record<string, Record<string, number | null> | number>;
  [k: string]: unknown;
}
import { perfectionScan } from './ai';
import { ensureProxy, getFile } from './images';
import { useEditor } from './store';

export type PerfectionStatus = 'idle' | 'running' | 'ready' | 'error';

export interface PhotoPerfection {
  status: PerfectionStatus;
  /** Key of the settings the shown result was made with. */
  key: string | null;
  report: PerfectionReport | null;
  /** Skin Finish measures of the shown result (null when the finish is off). */
  finish: FinishSummary | null;
  error: string | null;
  ms: number | null;
}

interface PerfectionState {
  byPhoto: Record<string, PhotoPerfection>;
  /** Show detected marks over the photo. */
  overlay: boolean;
  /** Re-run automatically when the settings change. */
  auto: boolean;
  /** Mark ids selected in the overlay (active photo). */
  selected: number[];
  set(id: string, patch: Partial<PhotoPerfection>): void;
  setOverlay(on: boolean): void;
  setAuto(on: boolean): void;
  select(ids: number[]): void;
}

const EMPTY: PhotoPerfection = { status: 'idle', key: null, report: null, finish: null, error: null, ms: null };

export const usePerfection = create<PerfectionState>((set) => ({
  byPhoto: {},
  overlay: true,
  auto: true,
  selected: [],
  set: (id, patch) => set((s) => ({ byPhoto: { ...s.byPhoto, [id]: { ...(s.byPhoto[id] ?? EMPTY), ...patch } } })),
  setOverlay: (overlay) => set({ overlay }),
  setAuto: (auto) => set({ auto }),
  select: (selected) => set({ selected }),
}));

export const perfectionOf = (id: string | null | undefined): PhotoPerfection => (id && usePerfection.getState().byPhoto[id]) || EMPTY;

/** Whether the viewport's source is the engine's result: Skin Perfection or Skin Finish is on. */
export function engineSourceOn(adj: Pick<Adjustments, 'skinPerfection'> & Partial<Pick<Adjustments, 'skinFinish'>>): boolean {
  return adj.skinPerfection.enabled || Boolean(adj.skinFinish?.enabled);
}

/** Everything in an edit that changes the engine's Skin Perfection / Skin Finish result. */
export function perfectionKeyOf(adj: Pick<Adjustments, 'skinPerfection' | 'skinMask'> & Partial<Pick<Adjustments, 'skinFinish'>>): string {
  const sm = adj.skinMask;
  const key: unknown[] = [adj.skinPerfection, sm.useSegmentation, sm.feather, sm.strokes];
  if (adj.skinFinish?.enabled) key.push(adj.skinFinish);
  return JSON.stringify(key);
}

interface CachedResult {
  key: string;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  report: PerfectionReport | null;
  finish: FinishSummary | null;
}

const RESULTS_PER_PHOTO = 4;
const results = new Map<string, CachedResult[]>();

/** Perfected preview pixels for this photo: the exact key if cached, else the latest result. */
export function perfectedPixels(id: string, key: string, w: number, h: number): { pixels: Uint8ClampedArray; current: boolean; key: string } | null {
  const list = results.get(id);
  if (!list?.length) return null;
  const hit = list.find((r) => r.key === key) ?? list[0];
  if (hit.width !== w || hit.height !== h) return null;
  return { pixels: hit.pixels, current: hit.key === key, key: hit.key };
}

export function forgetPerfection(id: string) {
  results.delete(id);
  timers.delete(id);
}

const inflight = new Map<string, { key: string; ctrl: AbortController }>();
const queued = new Set<string>();
const timers = new Map<string, ReturnType<typeof setTimeout>>();

async function decodeToPixels(dataUrl: string, w: number, h: number): Promise<Uint8ClampedArray> {
  const blob = await (await fetch(dataUrl)).blob();
  const bmp = await createImageBitmap(blob, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' });
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return ctx.getImageData(0, 0, w, h).data;
}

/** Run the engine for this photo's current settings (no-op when that result is cached). */
export async function runPerfection(id: string, opts: { force?: boolean } = {}): Promise<void> {
  const st = useEditor.getState();
  const p = st.photos[id];
  const file = getFile(id);
  if (!p || !file) return;
  const adj = p.adjustments;
  if (!engineSourceOn(adj)) return;
  const key = perfectionKeyOf(adj);
  const store = usePerfection.getState();
  const cached = results.get(id)?.find((r) => r.key === key);
  if (cached && !opts.force) {
    store.set(id, { status: 'ready', key, report: cached.report, finish: cached.finish, error: null });
    // Most recent first, so the stale fallback is the last one looked at.
    results.set(id, [cached, ...results.get(id)!.filter((r) => r !== cached)]);
    return;
  }
  if (!st.aiOnline) {
    store.set(id, { status: 'error', error: 'Skin Perfection and Skin Finish run in the local AI engine, which is not running.' });
    return;
  }
  const running = inflight.get(id);
  if (running) {
    if (running.key === key) return;
    queued.add(id); // one more run with the newest settings when this one ends
    return;
  }
  const proxy = await ensureProxy(id);
  if (!proxy) return;
  const ctrl = new AbortController();
  inflight.set(id, { key, ctrl });
  store.set(id, { status: 'running', error: null });
  try {
    const res = await perfectionScan(file, { skinPerfection: adj.skinPerfection, skinFinish: adj.skinFinish, skinMask: adj.skinMask }, Math.max(proxy.width, proxy.height), ctrl.signal);
    const pixels = await decodeToPixels(res.image, proxy.width, proxy.height);
    const entry: CachedResult = { key, width: proxy.width, height: proxy.height, pixels, report: res.report, finish: res.finish ?? null };
    results.set(id, [entry, ...(results.get(id) ?? []).filter((r) => r.key !== key)].slice(0, RESULTS_PER_PHOTO));
    usePerfection.getState().set(id, { status: 'ready', key, report: res.report, finish: res.finish ?? null, error: null, ms: res.ms });
    usePerfection.getState().select([]);
  } catch (e) {
    if (!ctrl.signal.aborted) usePerfection.getState().set(id, { status: 'error', error: e instanceof Error ? e.message : String(e) });
  } finally {
    inflight.delete(id);
    if (queued.delete(id)) void runPerfection(id);
  }
}

/** Debounced re-run after a settings change (sliders fire many changes per second). */
export function schedulePerfection(id: string, delayMs = 700) {
  clearTimeout(timers.get(id));
  timers.set(id, setTimeout(() => {
    timers.delete(id);
    void runPerfection(id);
  }, delayMs));
}
