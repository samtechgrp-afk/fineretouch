'use client';

/**
 * Pixel data lives outside the Zustand store: originals (File), proxy bitmaps
 * for the viewport, and lazily built face masks. The store only keeps ids.
 */

export const PROXY_LONG_EDGE = 2560;
const THUMB_LONG_EDGE = 320;

export interface LoadedImage {
  file: File;
  width: number;
  height: number;
  proxy: ImageBitmap;
  thumbUrl: string;
}

const files = new Map<string, File>();
const proxies = new Map<string, ImageBitmap>();
const masks = new Map<string, Uint8Array>();
const hairRois = new Map<string, Uint8Array>();

/** A depth map ready for Renderer.setDepth: a decoded image or single-channel bytes. */
export interface DepthData {
  source: ImageBitmap | Uint8Array;
  width: number;
  height: number;
  kind: 'model' | 'prior';
}
const depths = new Map<string, DepthData>();

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/avif', 'image/bmp', 'image/tiff'];

export function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID().replace(/-/g, '').slice(0, 16)
    : Math.random().toString(36).slice(2, 18);
}

export function fitSize(w: number, h: number, longEdge: number): [number, number] {
  const s = Math.min(1, longEdge / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

export async function loadImage(id: string, file: File): Promise<LoadedImage> {
  // createImageBitmap applies EXIF orientation by default in current browsers.
  const full = await createImageBitmap(file, { imageOrientation: 'from-image' });
  try {
    const [pw, ph] = fitSize(full.width, full.height, PROXY_LONG_EDGE);
    const proxy = await createImageBitmap(full, { resizeWidth: pw, resizeHeight: ph, resizeQuality: 'high' });
    const [tw, th] = fitSize(full.width, full.height, THUMB_LONG_EDGE);
    const thumbBmp = await createImageBitmap(full, { resizeWidth: tw, resizeHeight: th, resizeQuality: 'medium' });
    const canvas = document.createElement('canvas');
    canvas.width = tw;
    canvas.height = th;
    canvas.getContext('2d')!.drawImage(thumbBmp, 0, 0);
    thumbBmp.close();
    const blob = await new Promise<Blob>((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error('thumb'))), 'image/jpeg', 0.8));
    files.set(id, file);
    proxies.set(id, proxy);
    return { file, width: full.width, height: full.height, proxy, thumbUrl: URL.createObjectURL(blob) };
  } finally {
    full.close();
  }
}

export const getFile = (id: string) => files.get(id);
export const getProxy = (id: string) => proxies.get(id);
export const getCachedMask = (id: string) => masks.get(id);
export const setCachedMask = (id: string, m: Uint8Array) => masks.set(id, m);
export const clearCachedMask = (id: string) => {
  masks.delete(id);
  hairRois.delete(id);
};
export const getCachedHairRoi = (id: string) => hairRois.get(id);
export const setCachedHairRoi = (id: string, m: Uint8Array) => hairRois.set(id, m);
export const getDepth = (id: string) => depths.get(id);
export const setDepth = (id: string, d: DepthData) => {
  const old = depths.get(id);
  if (old && old.source !== d.source && 'close' in old.source) old.source.close();
  depths.set(id, d);
};

export function forgetImage(id: string) {
  proxies.get(id)?.close();
  proxies.delete(id);
  files.delete(id);
  masks.delete(id);
  hairRois.delete(id);
  const d = depths.get(id);
  if (d && 'close' in d.source) d.source.close();
  depths.delete(id);
}

/** Read RGBA pixels of a bitmap (used for client-side colour statistics). */
export function bitmapPixels(bmp: ImageBitmap, maxLongEdge = 512): Uint8ClampedArray {
  const [w, h] = fitSize(bmp.width, bmp.height, maxLongEdge);
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}
