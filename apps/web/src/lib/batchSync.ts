'use client';

/**
 * Batch style sync helpers: capture the reference photo's finished look, and talk to the
 * engine's local SQLite batch queue (services/ai/app/batch_engine.py, desktop only).
 */
import type { Adjustments, ColorStats } from '@fineretouch/shared';
import { colorStatsFromRgba } from '@/engine/color';
import { engineJson } from './ai';
import { renderFullRes } from './export';
import { bitmapPixels, ensureProxy, getProxy } from './images';
import type { Photo } from './store';

const LOOK_LONG_EDGE = 1024;

export interface ReferenceLook {
  /** The reference's original, as JPEG (for LUT fitting). */
  before: Blob;
  /** The reference rendered with its edits, as JPEG. */
  after: Blob;
  /** L*a*b* statistics of the finished reference (for palette sync). */
  gradedStats: ColorStats;
}

/** Render the reference at a small size with its edits, on the GPU, and measure it. */
export async function captureReferenceLook(photo: Photo): Promise<ReferenceLook> {
  const proxy = getProxy(photo.id) ?? (await ensureProxy(photo.id));
  if (!proxy) throw new Error('The reference photo is not loaded yet');
  const load = () => createImageBitmap(proxy);
  const before = await renderFullRes(load, { ...photo.adjustments, ...NEUTRAL }, ctxFor(photo), { format: 'jpeg', quality: 95, maxLongEdge: LOOK_LONG_EDGE });
  const after = await renderFullRes(load, photo.adjustments, ctxFor(photo), { format: 'jpeg', quality: 95, maxLongEdge: LOOK_LONG_EDGE });
  const bmp = await createImageBitmap(after);
  const gradedStats = colorStatsFromRgba(bitmapPixels(bmp, 512));
  bmp.close();
  return { before, after, gradedStats };
}

/** "Before" for LUT fitting is the untouched original: every colour / tone control neutral. */
const NEUTRAL: Partial<Adjustments> = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0, temperature: 0, tint: 0, saturation: 0, vibrance: 0,
  skinSmoothing: 0, blemishRemoval: 0, skinToneUnify: 0, eyeBrighten: 0, darkCircles: 0, teethWhiten: 0, teethStainRemoval: 0,
  glareReduction: 0, skinBrighten: 0, sharpen: 0, vignette: 0, colorMatch: null, skinMultiplier: 1, lut: null,
};

function ctxFor(photo: Photo) {
  return { id: photo.id, analysis: photo.analysis, sourceStats: photo.analysis?.colorStats ?? photo.localStats };
}

// --- engine batch queue --------------------------------------------------------------

export type ThermalProfile = 'quiet' | 'balanced' | 'performance';
export type BatchSyncMode = 'copy' | 'smart' | 'palette';

export interface BatchJob {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'paused' | 'done' | 'cancelled';
  outputDir: string;
  total: number;
  done: number;
  failed: number;
  pending: number;
  running: number;
  cancelled: number;
  percent: number;
  avgMs: number | null;
  etaSeconds: number | null;
  current: string[];
  errors: Array<{ source: string; error: string }>;
}

export interface BatchJobRequest {
  name: string;
  sources: string[];
  recursive: boolean;
  outputDir: string;
  profile?: ThermalProfile;
  settings: {
    adjustments: Adjustments;
    sync: { mode: BatchSyncMode; referenceStats?: ColorStats | null; referenceLookStats?: ColorStats | null; amount?: number };
    output: { format: 'jpeg' | 'png' | 'webp'; quality: number; maxLongEdge?: number | null; prefix?: string };
  };
}

export const createBatchJob = (req: BatchJobRequest) => engineJson<BatchJob>('/v1/batch/jobs', { body: req });
export const listBatchJobs = () => engineJson<{ profile: ThermalProfile; workers: number; jobs: BatchJob[] }>('/v1/batch/jobs');
export const batchAction = (id: string, action: 'pause' | 'resume' | 'cancel' | 'retry') =>
  engineJson<BatchJob>(`/v1/batch/jobs/${encodeURIComponent(id)}/${action}`, { method: 'POST' });
export const removeBatchJob = (id: string) => engineJson<{ deleted: string }>(`/v1/batch/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' });
export function setThermalProfile(profile: ThermalProfile) {
  const form = new FormData();
  form.append('profile', profile);
  return engineJson<{ profile: ThermalProfile; workers: number }>('/v1/batch/profile', { body: form });
}

/** Native multi-folder picker (desktop). */
export async function pickSourceFolders(): Promise<string[]> {
  const { open } = await import('@tauri-apps/plugin-dialog');
  const picked = await open({ directory: true, multiple: true, title: 'Choose folders of photos to process' });
  if (!picked) return [];
  return Array.isArray(picked) ? picked : [picked];
}

export function formatEta(seconds: number | null): string {
  if (seconds === null) return '';
  if (seconds < 60) return `${Math.round(seconds)} s left`;
  const m = Math.round(seconds / 60);
  return m < 60 ? `${m} min left` : `${Math.floor(m / 60)} h ${m % 60} min left`;
}
