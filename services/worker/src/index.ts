import { CreateBucketCommand, HeadBucketCommand, S3Client } from '@aws-sdk/client-s3';
import { BATCH_QUEUE, RENDER_QUEUE, type BatchJobData, type RenderJobData } from '@fineretouch/shared';
import { FlowProducer, Worker, type Job } from 'bullmq';
import { Redis } from 'ioredis';
import { config } from './config.js';
import { onRenderFailedFinally, processBatch, processFinalize, processRender, type Deps } from './processors.js';

const connection = new Redis(config.redisUrl, { maxRetriesPerRequest: null });
const deps: Deps = {
  redis: connection,
  flow: new FlowProducer({ connection }),
  s3: new S3Client({
    endpoint: config.s3.endpoint,
    region: config.s3.region,
    forcePathStyle: Boolean(config.s3.endpoint),
    // Newer SDKs add CRC checksums to every request and presigned URL; several
    // S3-compatible stores (R2, SeaweedFS) reject or mis-handle them.
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
    credentials: { accessKeyId: config.s3.accessKeyId, secretAccessKey: config.s3.secretAccessKey },
  }),
  fetch: globalThis.fetch,
};

/** Create the bucket on first boot (local dev); production buckets are provisioned by IaC. */
async function ensureBucket() {
  try {
    await deps.s3.send(new HeadBucketCommand({ Bucket: config.s3.bucket }));
  } catch {
    try {
      await deps.s3.send(new CreateBucketCommand({ Bucket: config.s3.bucket }));
      console.log(`Created bucket ${config.s3.bucket}`);
    } catch (err) {
      console.warn(`Could not verify or create bucket ${config.s3.bucket}: ${(err as Error).message}`);
    }
  }
}
await ensureBucket();

const batchWorker = new Worker(
  BATCH_QUEUE,
  async (job: Job) => {
    if (job.name === 'finalize') return processFinalize(job as Job<{ batchId: string; userId: string }>, deps);
    return processBatch(job as Job<BatchJobData>, deps);
  },
  { connection, concurrency: 2 },
);

const renderWorker = new Worker<RenderJobData>(RENDER_QUEUE, (job) => processRender(job, deps), {
  connection,
  concurrency: config.renderConcurrency,
});

renderWorker.on('failed', (job, err) => {
  console.warn(`[render] ${job?.id} failed (attempt ${job?.attemptsMade}): ${err.message}`);
  void onRenderFailedFinally(job, err, deps);
});
batchWorker.on('failed', (job, err) => console.error(`[batch] ${job?.id} failed: ${err.message}`));
batchWorker.on('completed', (job, result) => console.log(`[batch] ${job.id} ${job.name} done`, result ?? ''));

console.log(
  `FineRetouch worker up: render concurrency ${config.renderConcurrency}, AI service ${config.aiServiceUrl}`,
);

async function shutdown() {
  await Promise.allSettled([batchWorker.close(), renderWorker.close(), deps.flow.close()]);
  connection.disconnect();
  process.exit(0);
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
