import type { BatchProgress } from '@fineretouch/shared';
import type { Redis } from 'ioredis';

/** Batch progress lives in a Redis hash so the web app can poll it cheaply. */
export const progressKey = (batchId: string) => `fineretouch:batch:${batchId}`;
export const progressChannel = (batchId: string) => `fineretouch:batch:${batchId}:events`;
const TTL_SECONDS = 60 * 60 * 24 * 7;

export async function initProgress(redis: Redis, batchId: string, total: number): Promise<void> {
  await redis
    .multi()
    .hset(progressKey(batchId), { total, done: 0, failed: 0, state: 'running' })
    .expire(progressKey(batchId), TTL_SECONDS)
    .exec();
}

export async function bump(redis: Redis, batchId: string, field: 'done' | 'failed'): Promise<void> {
  await redis.hincrby(progressKey(batchId), field, 1);
  const p = await readProgress(redis, batchId);
  if (p) await redis.publish(progressChannel(batchId), JSON.stringify(p));
}

export async function finish(redis: Redis, batchId: string): Promise<BatchProgress | null> {
  const p = await readProgress(redis, batchId);
  if (!p) return null;
  const state = p.failed > 0 ? 'completed_with_errors' : 'completed';
  await redis.hset(progressKey(batchId), 'state', state);
  const final = { ...p, state } as BatchProgress;
  await redis.publish(progressChannel(batchId), JSON.stringify(final));
  return final;
}

export async function readProgress(redis: Redis, batchId: string): Promise<BatchProgress | null> {
  const h = await redis.hgetall(progressKey(batchId));
  if (!h || !h.total) return null;
  return {
    batchId,
    total: Number(h.total),
    done: Number(h.done ?? 0),
    failed: Number(h.failed ?? 0),
    state: (h.state as BatchProgress['state']) ?? 'queued',
  };
}
