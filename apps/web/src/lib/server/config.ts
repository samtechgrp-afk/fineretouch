import 'server-only';

export const serverConfig = {
  redisUrl: process.env.REDIS_URL,
  databaseUrl: process.env.DATABASE_URL,
  s3: {
    endpoint: process.env.S3_ENDPOINT || undefined,
    /** Endpoint the browser uses for presigned URLs (MinIO is "minio:9000" inside Docker). */
    publicEndpoint: process.env.S3_PUBLIC_ENDPOINT || process.env.S3_ENDPOINT || undefined,
    region: process.env.S3_REGION ?? 'us-east-1',
    bucket: process.env.S3_BUCKET ?? 'fineretouch',
    accessKeyId: process.env.S3_ACCESS_KEY ?? '',
    secretAccessKey: process.env.S3_SECRET_KEY ?? '',
  },
};

/** Cloud batch export needs Redis (queue) and object storage. */
export const cloudEnabled = () => Boolean(serverConfig.redisUrl && serverConfig.s3.accessKeyId);
