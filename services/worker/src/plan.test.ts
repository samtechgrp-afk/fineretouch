import { describe, expect, it } from 'vitest';
import type { BatchJobData } from '@fineretouch/shared';
import { BatchValidationError, planRenderJobs } from './plan.js';

const batch = (n: number, extra: Partial<BatchJobData> = {}): BatchJobData => ({
  batchId: 'b1',
  userId: 'u1',
  preset: { skinSmoothing: 40, exposure: 10 },
  items: Array.from({ length: n }, (_, i) => ({ imageId: `img${i}`, sourceKey: `uploads/u1/img${i}.jpg` })),
  output: { format: 'jpeg', quality: 90, prefix: 'wedding_' },
  ...extra,
});

describe('planRenderJobs', () => {
  it('expands a 500-image batch with the preset applied to every image', () => {
    const jobs = planRenderJobs(batch(500));
    expect(jobs).toHaveLength(500);
    expect(jobs[0].adjustments.skinSmoothing).toBe(40);
    expect(jobs[0].adjustments.skinTexture).toBe(50); // default filled in
    expect(jobs[499].outputKey).toBe('exports/u1/b1/wedding_img499.jpg');
  });

  it('lets per-image overrides win over the preset and clamps values', () => {
    const b = batch(2);
    b.items[1].adjustments = { exposure: 999, skinSmoothing: 5 };
    const jobs = planRenderJobs(b);
    expect(jobs[1].adjustments.exposure).toBe(100);
    expect(jobs[1].adjustments.skinSmoothing).toBe(5);
    expect(jobs[0].adjustments.exposure).toBe(10);
  });

  it('rejects sources outside the user prefix and path tricks', () => {
    const b = batch(1);
    b.items[0].sourceKey = 'uploads/someone-else/a.jpg';
    expect(() => planRenderJobs(b)).toThrow(BatchValidationError);
    b.items[0].sourceKey = 'uploads/u1/../someone-else/a.jpg';
    expect(() => planRenderJobs(b)).toThrow(BatchValidationError);
  });

  it('rejects empty, duplicate and unsafe ids', () => {
    expect(() => planRenderJobs(batch(0))).toThrow(/no items/);
    const dup = batch(2);
    dup.items[1].imageId = 'img0';
    expect(() => planRenderJobs(dup)).toThrow(/Duplicate/);
    expect(() => planRenderJobs(batch(1, { batchId: '../x' }))).toThrow(/batchId/);
  });

  it('uses the right extension per format and strips unsafe prefix characters', () => {
    const jobs = planRenderJobs(batch(1, { output: { format: 'webp', quality: 500, prefix: '../a b' } }));
    expect(jobs[0].outputKey).toBe('exports/u1/b1/..abimg0.webp'); // no slashes survive
    expect(jobs[0].output.quality).toBe(100);
  });
});
