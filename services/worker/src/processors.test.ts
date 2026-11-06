import { UnrecoverableError, type Job } from 'bullmq';
import { describe, expect, it, vi } from 'vitest';
import type { RenderJobData } from '@fineretouch/shared';
import { sanitizeAdjustments } from '@fineretouch/shared';
import { onRenderFailedFinally, processBatch, processRender, type Deps } from './processors.js';

function fakeRedis() {
  const hashes = new Map<string, Record<string, string>>();
  return {
    hashes,
    hincrby: vi.fn(async (k: string, f: string, by: number) => {
      const h = hashes.get(k) ?? {};
      h[f] = String(Number(h[f] ?? 0) + by);
      hashes.set(k, h);
    }),
    hgetall: vi.fn(async (k: string) => hashes.get(k) ?? {}),
    hset: vi.fn(),
    publish: vi.fn(),
  };
}

const data: RenderJobData = {
  batchId: 'b1',
  imageId: 'i1',
  sourceKey: 'uploads/u1/i1.jpg',
  outputKey: 'exports/u1/b1/i1.jpg',
  adjustments: sanitizeAdjustments({ skinSmoothing: 30 }),
  output: { format: 'jpeg', quality: 90, prefix: '' },
};
const job = (attemptsMade = 0) => ({ data, attemptsMade, opts: { attempts: 3 } }) as unknown as Job<RenderJobData>;

function deps(fetchImpl: typeof fetch) {
  const redis = fakeRedis();
  redis.hashes.set('fineretouch:batch:b1', { total: '1', done: '0', failed: '0', state: 'running' });
  return { redis, deps: { redis, fetch: fetchImpl } as unknown as Deps };
}

describe('processRender', () => {
  it('posts the job to the AI service and counts it as done', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ outputKey: data.outputKey, bytes: 1234, ms: 50, cull: { blurry: false, eyesClosed: false, noFace: false } })),
    );
    const { redis, deps: d } = deps(fetchMock as unknown as typeof fetch);
    const result = await processRender(job(), d);
    expect(result).toMatchObject({ imageId: 'i1', bytes: 1234 });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/v1\/render-object$/);
    expect(JSON.parse(init.body as string).adjustments.skinSmoothing).toBe(30);
    expect(redis.hashes.get('fineretouch:batch:b1')!.done).toBe('1');
  });

  it('does not retry client errors', async () => {
    const { deps: d } = deps((async () => new Response('bad', { status: 422 })) as unknown as typeof fetch);
    await expect(processRender(job(), d)).rejects.toBeInstanceOf(UnrecoverableError);
  });

  it('retries server errors', async () => {
    const { deps: d } = deps((async () => new Response('boom', { status: 503 })) as unknown as typeof fetch);
    const err = await processRender(job(), d).catch((e) => e);
    expect(err).not.toBeInstanceOf(UnrecoverableError);
  });
});

describe('onRenderFailedFinally', () => {
  it('counts a failure only once the retries are exhausted', async () => {
    const { redis, deps: d } = deps(fetch);
    await onRenderFailedFinally(job(1), new Error('x'), d);
    expect(redis.hashes.get('fineretouch:batch:b1')!.failed).toBe('0');
    await onRenderFailedFinally(job(3), new Error('x'), d);
    expect(redis.hashes.get('fineretouch:batch:b1')!.failed).toBe('1');
    await onRenderFailedFinally(job(1), new UnrecoverableError('4xx'), d);
    expect(redis.hashes.get('fineretouch:batch:b1')!.failed).toBe('2');
  });
});

describe('processBatch', () => {
  it('fans out one render child per image with BullMQ-safe, idempotent ids', async () => {
    const flow = { add: vi.fn(async () => ({})) };
    const multi = { hset: vi.fn().mockReturnThis(), expire: vi.fn().mockReturnThis(), exec: vi.fn(async () => []) };
    const d = { redis: { multi: () => multi }, flow } as unknown as Deps;
    const job = {
      data: {
        batchId: 'abc123',
        userId: 'u1',
        preset: { skinSmoothing: 20 },
        items: Array.from({ length: 500 }, (_, i) => ({ imageId: `img${i}`, sourceKey: `uploads/u1/img${i}.jpg` })),
        output: { format: 'jpeg', quality: 90, prefix: '' },
      },
      updateProgress: vi.fn(),
    } as unknown as Job;
    await expect(processBatch(job as never, d)).resolves.toEqual({ total: 500 });
    const flowArg = (flow.add.mock.calls[0] as unknown as [{ opts: { jobId: string }; children: Array<{ opts: { jobId: string } }> }])[0];
    expect(flowArg.children).toHaveLength(500);
    const ids = [flowArg.opts.jobId, ...flowArg.children.map((c) => c.opts.jobId)];
    expect(ids.every((id) => !id.includes(':'))).toBe(true);
    expect(new Set(ids).size).toBe(501);
    expect(multi.hset).toHaveBeenCalledWith('fineretouch:batch:abc123', expect.objectContaining({ total: 500 }));
  });

  it('turns validation errors into unrecoverable failures', async () => {
    const d = { redis: {}, flow: { add: vi.fn() } } as unknown as Deps;
    const job = { data: { batchId: 'b', userId: 'u1', preset: {}, items: [], output: { format: 'jpeg', quality: 90, prefix: '' } } } as unknown as Job;
    await expect(processBatch(job as never, d)).rejects.toBeInstanceOf(UnrecoverableError);
  });
});
