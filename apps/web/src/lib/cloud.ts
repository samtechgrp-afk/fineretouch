'use client';

import type { Adjustments, BatchProgress } from '@fineretouch/shared';
import { pool } from './ai';
import { uploadContentType } from './rawParser';

export async function cloudAvailable(): Promise<boolean> {
  try {
    const res = await fetch('/api/health', { cache: 'no-store' });
    return res.ok && Boolean((await res.json()).cloud);
  } catch {
    return false;
  }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.json().catch(() => ({ error: res.statusText }))).error ?? res.statusText);
  return res.json() as Promise<T>;
}

export interface CloudItem {
  id: string;
  file: File;
  adjustments: Adjustments;
}

/**
 * Upload originals straight to object storage, queue one batch job, then poll
 * progress until the worker pool has rendered everything.
 */
export async function runCloudBatch(
  items: CloudItem[],
  output: { format: 'jpeg' | 'png' | 'webp'; quality: number; maxLongEdge?: number },
  onProgress: (phase: 'upload' | 'render', done: number, total: number) => void,
  signal: AbortSignal,
): Promise<{ progress: BatchProgress; files: Array<{ imageId: string; url: string }> }> {
  const { uploads } = await json<{ uploads: Array<{ id: string; key: string; url: string }> }>(
    await fetch('/api/uploads', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ files: items.map((i) => ({ id: i.id, type: uploadContentType(i.file), size: i.file.size })) }),
      signal,
    }),
  );
  const byId = new Map(items.map((i) => [i.id, i]));
  let uploaded = 0;
  await pool(uploads, 4, async (u) => {
    const item = byId.get(u.id)!;
    const res = await fetch(u.url, { method: 'PUT', body: item.file, headers: { 'content-type': uploadContentType(item.file) }, signal });
    if (!res.ok) throw new Error(`Upload failed for ${item.file.name} (${res.status})`);
    onProgress('upload', ++uploaded, uploads.length);
  });

  const { batchId } = await json<{ batchId: string }>(
    await fetch('/api/batches', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        items: uploads.map((u) => ({ imageId: u.id, sourceKey: u.key, adjustments: byId.get(u.id)!.adjustments })),
        output,
      }),
      signal,
    }),
  );

  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    const p = await json<BatchProgress & { files?: Array<{ imageId: string; url: string }> }>(
      await fetch(`/api/batches/${batchId}`, { cache: 'no-store', signal }),
    );
    onProgress('render', p.done + p.failed, p.total);
    if (p.state === 'completed' || p.state === 'completed_with_errors') return { progress: p, files: p.files ?? [] };
  }
}
