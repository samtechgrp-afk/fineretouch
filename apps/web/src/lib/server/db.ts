import 'server-only';
import { Pool } from 'pg';
import { serverConfig } from './config';

let pool: Pool | null = null;

/** Optional: when DATABASE_URL is unset the app runs without persistence. */
export function db(): Pool | null {
  if (!serverConfig.databaseUrl) return null;
  return (pool ??= new Pool({ connectionString: serverConfig.databaseUrl, max: 10 }));
}

export async function recordBatch(batchId: string, userId: string, total: number, preset: unknown): Promise<void> {
  const p = db();
  if (!p) return;
  await p.query(
    `INSERT INTO batches (id, user_id, total, preset, state) VALUES ($1, $2, $3, $4, 'queued')
     ON CONFLICT (id) DO NOTHING`,
    [batchId, userId, total, JSON.stringify(preset)],
  );
}
