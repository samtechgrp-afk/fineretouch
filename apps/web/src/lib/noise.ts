'use client';

import { estimateNoiseLevel } from '@/engine/noise';

const CROP = 384;
const levels = new Map<string, number>();

/**
 * Noise level of an image at its own resolution (a centre crop, so the measurement is cheap and
 * mostly on the subject). Cached per key: the viewport's proxy and an export's full-size source
 * are measured separately, since downscaling removes noise.
 */
export function noiseLevelOf(key: string, bmp: ImageBitmap): number {
  const hit = levels.get(key);
  if (hit !== undefined) return hit;
  const w = Math.min(CROP, bmp.width);
  const h = Math.min(CROP, bmp.height);
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, (bmp.width - w) / 2, (bmp.height - h) / 2, w, h, 0, 0, w, h);
  const level = estimateNoiseLevel(ctx.getImageData(0, 0, w, h).data, w, h);
  levels.set(key, level);
  return level;
}

export const forgetNoiseLevel = (id: string) => {
  for (const k of levels.keys()) if (k.startsWith(`${id}:`)) levels.delete(k);
};
