import { PutObjectCommand, type S3Client } from '@aws-sdk/client-s3';
import {
  BATCH_QUEUE,
  RENDER_QUEUE,
  type BatchJobData,
  type RenderJobData,
  type RenderJobResult,
} from '@fineretouch/shared';
import { UnrecoverableError, type FlowProducer, type Job } from 'bullmq';
import type { Redis } from 'ioredis';
import { config } from './config.js';
import { BatchValidationError, planRenderJobs } from './plan.js';
import { bump, finish, initProgress } from './progress.js';

export interface Deps {
  redis: Redis;
  flow: FlowProducer;
  s3: S3Client;
  fetch: typeof fetch;
}

/**
 * `batch` job: validate, then fan out as a BullMQ flow. The parent `finalize`
 * job runs once every child render has either succeeded or exhausted its retries.
 */
export async function processBatch(job: Job<BatchJobData>, deps: Deps): Promise<{ total: number }> {
  let renders: RenderJobData[];
  try {
    renders = planRenderJobs(job.data);
  } catch (err) {
    if (err instanceof BatchValidationError) throw new UnrecoverableError(err.message);
    throw err;
  }
  const { batchId, userId } = job.data;
  await initProgress(deps.redis, batchId, renders.length);
  await deps.flow.add({
    name: 'finalize',
    queueName: BATCH_QUEUE,
    data: { batchId, userId },
    opts: { jobId: `${batchId}-finalize` }, // BullMQ ids may not contain ":"
    children: renders.map((r) => ({
      name: 'render',
      queueName: RENDER_QUEUE,
      data: r,
      opts: {
        jobId: `${batchId}-${r.imageId}`, // idempotent if the batch job is retried
        attempts: config.renderAttempts,
        backoff: { type: 'exponential', delay: 2000 },
        ignoreDependencyOnFailure: true,
        removeOnComplete: { age: 86_400 },
        removeOnFail: { age: 604_800 },
      },
    })),
  });
  await job.updateProgress({ total: renders.length, queued: renders.length });
  return { total: renders.length };
}

/** `render` job: ask the AI service to render source -> output in object storage. */
export async function processRender(job: Job<RenderJobData>, deps: Deps): Promise<RenderJobResult> {
  const d = job.data;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), config.renderTimeoutMs);
  try {
    const res = await deps.fetch(`${config.aiServiceUrl}/v1/render-object`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        sourceKey: d.sourceKey,
        outputKey: d.outputKey,
        adjustments: d.adjustments,
        output: d.output,
      }),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      // 4xx means the input is bad: retrying will not help.
      if (res.status >= 400 && res.status < 500 && res.status !== 429) {
        throw new UnrecoverableError(`AI service ${res.status}: ${text.slice(0, 300)}`);
      }
      throw new Error(`AI service ${res.status}: ${text.slice(0, 300)}`);
    }
    const body = (await res.json()) as { outputKey: string; bytes: number; ms: number; cull: RenderJobResult['cull'] };
    await bump(deps.redis, d.batchId, 'done');
    return { imageId: d.imageId, outputKey: body.outputKey, bytes: body.bytes, ms: body.ms, cull: body.cull };
  } finally {
    clearTimeout(timer);
  }
}

/** Called from the worker's `failed` event once a render has no attempts left. */
export async function onRenderFailedFinally(
  job: Job<RenderJobData> | undefined,
  err: Error,
  deps: Deps,
): Promise<void> {
  if (!job) return;
  const exhausted = err instanceof UnrecoverableError || job.attemptsMade >= (job.opts.attempts ?? 1);
  if (exhausted) await bump(deps.redis, job.data.batchId, 'failed');
}

/** `finalize` job: write a manifest of every export next to the files. */
export async function processFinalize(job: Job<{ batchId: string; userId: string }>, deps: Deps) {
  const { batchId, userId } = job.data;
  const results = (await job.getChildrenValues()) as Record<string, RenderJobResult>;
  const failures = await job.getIgnoredChildrenFailures();
  const manifest = {
    batchId,
    createdAt: new Date().toISOString(),
    exports: Object.values(results),
    failed: Object.entries(failures).map(([key, reason]) => ({ job: key, reason })),
  };
  await deps.s3.send(
    new PutObjectCommand({
      Bucket: config.s3.bucket,
      Key: `exports/${userId}/${batchId}/manifest.json`,
      Body: JSON.stringify(manifest, null, 2),
      ContentType: 'application/json',
    }),
  );
  return finish(deps.redis, batchId);
}
