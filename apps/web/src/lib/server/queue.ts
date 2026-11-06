import 'server-only';
import { BATCH_QUEUE, type BatchJobData, type BatchProgress } from '@fineretouch/shared';
import { Queue } from 'bullmq';
import { Redis } from 'ioredis';
import { serverConfig } from './config';

let redis: Redis | null = null;
let queue: Queue<BatchJobData> | null = null;

function connection(): Redis {
  if (!serverConfig.redisUrl) throw new Error('REDIS_URL is not configured');
  return (redis ??= new Redis(serverConfig.redisUrl, { maxRetriesPerRequest: null }));
}

export function batchQueue(): Queue<BatchJobData> {
  return (queue ??= new Queue<BatchJobData>(BATCH_QUEUE, {
    connection: connection(),
    defaultJobOptions: { attempts: 3, backoff: { type: 'exponential', delay: 1000 }, removeOnComplete: { age: 604_800 } },
  }));
}

/** Same key layout as services/worker/src/progress.ts. */
export async function readProgress(batchId: string): Promise<BatchProgress | null> {
  const h = await connection().hgetall(`fineretouch:batch:${batchId}`);
  if (!h?.total) return null;
  return {
    batchId,
    total: Number(h.total),
    done: Number(h.done ?? 0),
    failed: Number(h.failed ?? 0),
    state: (h.state as BatchProgress['state']) ?? 'queued',
  };
}
