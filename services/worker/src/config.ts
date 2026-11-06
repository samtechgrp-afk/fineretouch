const num = (v: string | undefined, d: number) => (v && Number.isFinite(Number(v)) ? Number(v) : d);

export const config = {
  redisUrl: process.env.REDIS_URL ?? 'redis://localhost:6379',
  aiServiceUrl: (process.env.AI_SERVICE_URL ?? 'http://localhost:8000').replace(/\/$/, ''),
  /** Parallel renders per worker process. Scale horizontally with more replicas. */
  renderConcurrency: num(process.env.RENDER_CONCURRENCY, 4),
  renderAttempts: num(process.env.RENDER_ATTEMPTS, 3),
  renderTimeoutMs: num(process.env.RENDER_TIMEOUT_MS, 120_000),
  s3: {
    endpoint: process.env.S3_ENDPOINT || undefined,
    region: process.env.S3_REGION ?? 'us-east-1',
    bucket: process.env.S3_BUCKET ?? 'fineretouch',
    accessKeyId: process.env.S3_ACCESS_KEY ?? '',
    secretAccessKey: process.env.S3_SECRET_KEY ?? '',
  },
};
