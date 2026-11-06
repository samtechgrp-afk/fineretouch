import type { RenderJobResult } from '@fineretouch/shared';
import { NextResponse } from 'next/server';
import { cloudEnabled } from '@/lib/server/config';
import { batchQueue, readProgress } from '@/lib/server/queue';
import { presignDownload, readJson } from '@/lib/server/s3';
import { userId } from '@/lib/server/session';

export const dynamic = 'force-dynamic';

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  if (!cloudEnabled()) return NextResponse.json({ error: 'Cloud processing is not configured' }, { status: 503 });
  const { id } = await ctx.params;
  if (!/^[a-f0-9]{32}$/.test(id)) return NextResponse.json({ error: 'Not found' }, { status: 404 });
  const uid = await userId();
  const job = await batchQueue().getJob(id);
  if (!job || job.data.userId !== uid) return NextResponse.json({ error: 'Not found' }, { status: 404 });

  if ((await job.getState()) === 'failed') {
    return NextResponse.json({ error: `Batch failed: ${job.failedReason ?? 'unknown error'}` }, { status: 500 });
  }
  const progress = (await readProgress(id)) ?? { batchId: id, total: job.data.items.length, done: 0, failed: 0, state: 'queued' as const };
  if (progress.state === 'completed' || progress.state === 'completed_with_errors') {
    const manifest = await readJson<{ exports: RenderJobResult[] }>(`exports/${uid}/${id}/manifest.json`);
    const files = await Promise.all(
      (manifest?.exports ?? []).map(async (e) => ({ imageId: e.imageId, url: await presignDownload(e.outputKey, e.outputKey.split('/').pop()) })),
    );
    return NextResponse.json({ ...progress, files });
  }
  return NextResponse.json(progress);
}
