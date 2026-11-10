'use client';

import type { Adjustments, ColorStats, ImageAnalysis } from '@fineretouch/shared';

import { desktopInfo } from './desktop';

/**
 * Base URL of services/ai. In the desktop app this is the bundled engine sidecar on a
 * loopback port chosen at launch; on the web it comes from NEXT_PUBLIC_AI_URL.
 */
export function aiUrl(): string {
  return (desktopInfo()?.engineUrl ?? process.env.NEXT_PUBLIC_AI_URL ?? 'http://localhost:8000').replace(/\/$/, '');
}

/** The desktop engine rejects requests without its per-launch token. */
function aiHeaders(): HeadersInit | undefined {
  const token = desktopInfo()?.engineToken;
  return token ? { 'x-fineretouch-token': token } : undefined;
}

export async function aiHealth(signal?: AbortSignal): Promise<{ status: string; faceBackend: string; depthBackend?: string } | null> {
  try {
    const res = await fetch(`${aiUrl()}/health`, { signal, cache: 'no-store', headers: aiHeaders() });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

async function postFile<T>(path: string, file: Blob, extra?: Record<string, string>, signal?: AbortSignal): Promise<Response> {
  const form = new FormData();
  form.append('file', file, (file as File).name ?? 'image.jpg');
  for (const [k, v] of Object.entries(extra ?? {})) form.append(k, v);
  const res = await fetch(`${aiUrl()}${path}`, { method: 'POST', body: form, signal, headers: aiHeaders() });
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

/** Relative depth (8-bit PNG, 0 = subject, 255 = far) computed by the AI service's depth model. */
export async function depthMap(proxyJpeg: Blob, signal?: AbortSignal): Promise<Blob> {
  return (await postFile('/v1/depth', proxyJpeg, undefined, signal)).blob();
}

/** Face parsing PNG (r face skin, g body skin, b hair) of the proxy, for the preview's skin mask. */
export async function parseMap(proxyJpeg: Blob, signal?: AbortSignal): Promise<Blob> {
  return (await postFile('/v1/parse', proxyJpeg, undefined, signal)).blob();
}

/** Segmentation PNG (r clothes, g hair, b person) of the proxy, for the clothing tools' preview. */
export async function segmentMap(proxyJpeg: Blob, signal?: AbortSignal): Promise<Blob> {
  return (await postFile('/v1/segment', proxyJpeg, undefined, signal)).blob();
}

export interface FittedLook {
  name: string;
  size: number;
  data: string;
  amount: number;
  /** The same LUT as .cube text, for saving to disk. */
  cube: string;
}

/** Fit the reference's grade as a 3D LUT from its original and its edited render. */
export async function fitLook(before: Blob, after: Blob, name: string, size = 17): Promise<FittedLook> {
  const form = new FormData();
  form.append('before', before, 'before.jpg');
  form.append('after', after, 'after.jpg');
  form.append('size', String(size));
  form.append('name', name);
  const res = await fetch(`${aiUrl()}/v1/lut/fit`, { method: 'POST', body: form, headers: aiHeaders() });
  if (!res.ok) throw new Error(`AI service ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

/** JSON call to the engine (the local batch queue). */
export async function engineJson<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { ...(aiHeaders() as Record<string, string> | undefined) };
  let body: BodyInit | undefined;
  if (init.body instanceof FormData) body = init.body;
  else if (init.body !== undefined) {
    headers['content-type'] = 'application/json';
    body = JSON.stringify(init.body);
  }
  const res = await fetch(`${aiUrl()}${path}`, { method: init.method ?? (body ? 'POST' : 'GET'), body, headers, cache: 'no-store' });
  if (!res.ok) {
    let detail = (await res.text()).slice(0, 300);
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      /* plain text */
    }
    throw new Error(String(detail));
  }
  return res.json();
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

/**
 * Skin Perfection at full resolution in the local engine: the verification report and the
 * perfected photo at preview size (a JPEG data URL), which the viewport uses as its source.
 */
export async function perfectionScan(
  file: Blob,
  adjustments: Pick<Adjustments, 'skinPerfection' | 'skinMask'>,
  previewLongEdge: number,
  signal?: AbortSignal,
): Promise<{ report: import('@fineretouch/shared').PerfectionReport | null; image: string; faces: number; ms: number }> {
  const extra = { adjustments: JSON.stringify(adjustments), previewLongEdge: String(Math.round(previewLongEdge)) };
  return (await postFile('/v1/perfection', file, extra, signal)).json();
}
