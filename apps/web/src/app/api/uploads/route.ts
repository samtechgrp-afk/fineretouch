import { MAX_BATCH_ITEMS } from '@fineretouch/shared';
import { NextResponse } from 'next/server';
import { cloudEnabled } from '@/lib/server/config';
import { presignUpload } from '@/lib/server/s3';
import { userId } from '@/lib/server/session';

export const dynamic = 'force-dynamic';

const MAX_BYTES = 150 * 1024 * 1024;
const EXT: Record<string, string> = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/tiff': 'tif', 'image/avif': 'avif' };

interface UploadRequest {
  files: Array<{ id: string; type: string; size: number }>;
}

/** Returns presigned PUT URLs so originals go straight from the browser to object storage. */
export async function POST(req: Request) {
  if (!cloudEnabled()) return NextResponse.json({ error: 'Cloud processing is not configured' }, { status: 503 });
  const body = (await req.json().catch(() => null)) as UploadRequest | null;
  const files = body?.files;
  if (!Array.isArray(files) || !files.length || files.length > MAX_BATCH_ITEMS) {
    return NextResponse.json({ error: `Send 1-${MAX_BATCH_ITEMS} files` }, { status: 400 });
  }
  const uid = await userId();
  const out = [];
  for (const f of files) {
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(f.id) || !EXT[f.type] || !(f.size > 0 && f.size <= MAX_BYTES)) {
      return NextResponse.json({ error: `Invalid file ${String(f.id)}` }, { status: 400 });
    }
    const key = `uploads/${uid}/${f.id}.${EXT[f.type]}`;
    out.push({ id: f.id, key, url: await presignUpload(key, f.type) });
  }
  return NextResponse.json({ uploads: out });
}
