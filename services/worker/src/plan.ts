import {
  MAX_BATCH_ITEMS,
  sanitizeAdjustments,
  type BatchJobData,
  type RenderJobData,
} from '@fineretouch/shared';

export class BatchValidationError extends Error {}

const SAFE_SEGMENT = /^[A-Za-z0-9._-]{1,128}$/;
const EXT = { jpeg: 'jpg', png: 'png', webp: 'webp' } as const;

/**
 * Expand a batch into one render job per image. Pure, so it is unit tested
 * without Redis. The batch preset is applied first, then per-image overrides
 * (the "copy settings to 1,000 photos, then tweak a few" workflow).
 */
export function planRenderJobs(batch: BatchJobData): RenderJobData[] {
  if (!SAFE_SEGMENT.test(batch.batchId)) throw new BatchValidationError('Invalid batchId');
  if (!SAFE_SEGMENT.test(batch.userId)) throw new BatchValidationError('Invalid userId');
  if (!Array.isArray(batch.items) || batch.items.length === 0) throw new BatchValidationError('Batch has no items');
  if (batch.items.length > MAX_BATCH_ITEMS) {
    throw new BatchValidationError(`Batch has ${batch.items.length} items; limit is ${MAX_BATCH_ITEMS}`);
  }
  const format = batch.output?.format ?? 'jpeg';
  if (!(format in EXT)) throw new BatchValidationError(`Unsupported format ${format}`);
  const quality = Math.min(100, Math.max(1, Math.round(batch.output?.quality ?? 92)));
  const prefix = (batch.output?.prefix ?? '').replace(/[^A-Za-z0-9._-]/g, '');

  const seen = new Set<string>();
  return batch.items.map((item) => {
    if (!SAFE_SEGMENT.test(item.imageId)) throw new BatchValidationError(`Invalid imageId ${item.imageId}`);
    if (seen.has(item.imageId)) throw new BatchValidationError(`Duplicate imageId ${item.imageId}`);
    seen.add(item.imageId);
    // Users may only render their own uploads.
    if (!item.sourceKey.startsWith(`uploads/${batch.userId}/`) || item.sourceKey.includes('..')) {
      throw new BatchValidationError(`Source ${item.sourceKey} is outside the user's uploads`);
    }
    return {
      batchId: batch.batchId,
      imageId: item.imageId,
      sourceKey: item.sourceKey,
      outputKey: `exports/${batch.userId}/${batch.batchId}/${prefix}${item.imageId}.${EXT[format]}`,
      adjustments: sanitizeAdjustments({ ...batch.preset, ...item.adjustments }),
      output: { format, quality, maxLongEdge: batch.output?.maxLongEdge, prefix },
    };
  });
}
