'use client';

import type { Adjustments, ColorStats, ImageAnalysis } from '@fineretouch/shared';

/** Base URL of services/ai. Configure with NEXT_PUBLIC_AI_URL. */
export const AI_URL = (process.env.NEXT_PUBLIC_AI_URL ?? 'http://localhost:8000').replace(/\/$/, '');

export async function aiHealth(signal?: AbortSignal): Promise<{ status: string; faceBackend: string } | null> {
  try {
    const res = await fetch(`${AI_URL}/health`, { signal, cache: 'no-store' });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

async function postFile<T>(path: string, file: Blob, extra?: Record<string, string>, signal?: AbortSignal): Promise<Response> {
  const form = new FormData();
  form.append('file', file, (file as File).name ?? 'image.jpg');
  for (const [k, v] of Object.entries(extra ?? {})) form.append(k, v);
  const res = await fetch(`${AI_URL}${path}`, { method: 'POST', body: form, signal });
  if (!res.ok) throw new Error(`AI service ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res;
}

/**
 * Landmarks are resolution independent, so we upload a downscaled JPEG instead
 * of a 40 MB original; the server then works on the same pixels the proxy shows.
 */
export async function analyze(proxyJpeg: Blob, signal?: AbortSignal): Promise<ImageAnalysis> {
  return (await postFile('/v1/analyze', proxyJpeg, undefined, signal)).json();
}

export async function referenceStats(file: Blob): Promise<ColorStats> {
  return (await postFile('/v1/color-stats', file)).json();
}

export async function renderOnServer(
  file: Blob,
  adjustments: Adjustments,
  opts: { format?: 'jpeg' | 'png' | 'webp'; quality?: number; maxLongEdge?: number } = {},
): Promise<Blob> {
  const extra: Record<string, string> = {
    adjustments: JSON.stringify(adjustments),
    format: opts.format ?? 'jpeg',
    quality: String(opts.quality ?? 92),
  };
  if (opts.maxLongEdge) extra.maxLongEdge = String(opts.maxLongEdge);
  return (await postFile('/v1/render', file, extra)).blob();
}

/** Run async tasks with bounded concurrency. */
export async function pool<T>(items: T[], limit: number, fn: (item: T, index: number) => Promise<void>): Promise<void> {
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      await fn(items[i], i);
    }
  });
  await Promise.all(workers);
}
