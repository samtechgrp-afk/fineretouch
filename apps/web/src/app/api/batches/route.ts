import { MAX_BATCH_ITEMS, sanitizeAdjustments, type BatchJobData } from '@fineretouch/shared';
import { NextResponse } from 'next/server';
import { cloudEnabled } from '@/lib/server/config';
import { recordBatch } from '@/lib/server/db';
import { batchQueue } from '@/lib/server/queue';
import { userId } from '@/lib/server/session';

export const dynamic = 'force-dynamic';

interface CreateBatch {
  preset?: unknown;
  items: Array<{ imageId: string; sourceKey: string; adjustments?: unknown }>;
  output?: Partial<BatchJobData['output']>;
}

/** Queue a batch render. The worker fans it out to one job per image. */
export async function POST(req: Request) {
  if (!cloudEnabled()) return NextResponse.json({ error: 'Cloud processing is not configured' }, { status: 503 });
  const body = (await req.json().catch(() => null)) as CreateBatch | null;
  if (!body || !Array.isArray(body.items) || !body.items.length || body.items.length > MAX_BATCH_ITEMS) {
    return NextResponse.json({ error: `A batch needs 1-${MAX_BATCH_ITEMS} items` }, { status: 400 });
  }
  const uid = await userId();
  for (const item of body.items) {
    if (typeof item.sourceKey !== 'string' || !item.sourceKey.startsWith(`uploads/${uid}/`) || item.sourceKey.includes('..')) {
      return NextResponse.json({ error: 'Items must reference your own uploads' }, { status: 403 });
    }
  }
  const batchId = crypto.randomUUID().replace(/-/g, '');
  const format = body.output?.format && ['jpeg', 'png', 'webp'].includes(body.output.format) ? body.output.format : 'jpeg';
  const data: BatchJobData = {
    batchId,
    userId: uid,
    preset: sanitizeAdjustments(body.preset ?? {}),
    items: body.items.map((i) => ({
      imageId: String(i.imageId),
      sourceKey: i.sourceKey,
      adjustments: i.adjustments ? sanitizeAdjustments(i.adjustments) : undefined,
    })),
    output: {
      format,
      quality: Math.min(100, Math.max(1, Number(body.output?.quality) || 92)),
      maxLongEdge: Number(body.output?.maxLongEdge) || undefined,
      prefix: String(body.output?.prefix ?? '').replace(/[^\w.-]/g, '').slice(0, 40),
    },
  };
  await batchQueue().add('batch', data, { jobId: batchId });
  await recordBatch(batchId, uid, data.items.length, data.preset).catch((e) => console.error('recordBatch', e));
  return NextResponse.json({ batchId, total: data.items.length }, { status: 202 });
}
