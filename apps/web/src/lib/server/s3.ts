import 'server-only';
import { GetObjectCommand, PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { serverConfig } from './config';

const make = (endpoint: string | undefined) =>
  new S3Client({
    endpoint,
    region: serverConfig.s3.region,
    forcePathStyle: Boolean(endpoint),
    // Newer SDKs add CRC checksums to every request and presigned URL; several
    // S3-compatible stores (R2, SeaweedFS) reject or mis-handle them.
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED',
    credentials: { accessKeyId: serverConfig.s3.accessKeyId, secretAccessKey: serverConfig.s3.secretAccessKey },
  });

let internal: S3Client | null = null;
let browser: S3Client | null = null;
const s3 = () => (internal ??= make(serverConfig.s3.endpoint));
const s3Public = () => (browser ??= make(serverConfig.s3.publicEndpoint));

export async function presignUpload(key: string, contentType: string): Promise<string> {
  return getSignedUrl(s3Public(), new PutObjectCommand({ Bucket: serverConfig.s3.bucket, Key: key, ContentType: contentType }), {
    expiresIn: 900,
  });
}

export async function presignDownload(key: string, filename?: string): Promise<string> {
  return getSignedUrl(
    s3Public(),
    new GetObjectCommand({
      Bucket: serverConfig.s3.bucket,
      Key: key,
      ResponseContentDisposition: filename ? `attachment; filename="${filename.replace(/"/g, '')}"` : undefined,
    }),
    { expiresIn: 3600 },
  );
}

export async function readJson<T>(key: string): Promise<T | null> {
  try {
    const res = await s3().send(new GetObjectCommand({ Bucket: serverConfig.s3.bucket, Key: key }));
    return JSON.parse(await res.Body!.transformToString()) as T;
  } catch {
    return null;
  }
}
