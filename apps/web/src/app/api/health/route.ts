import { NextResponse } from 'next/server';
import { cloudEnabled } from '@/lib/server/config';

export const dynamic = 'force-dynamic';

export function GET() {
  return NextResponse.json({ status: 'ok', cloud: cloudEnabled() });
}
