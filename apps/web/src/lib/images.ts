'use client';

/**
 * Pixel data lives outside the Zustand store: originals (File), proxy bitmaps
 * for the viewport, and lazily built face masks. The store only keeps ids.
 */

import { isRawFileName, parseRawFile, type RawExif, type RawFormat } from './rawParser';

export const PROXY_LONG_EDGE = 2560;
const THUMB_LONG_EDGE = 320;

/** What the editor knows about a camera RAW original. */
export interface RawInfo {
  format: RawFormat;
  /** Badge text, e.g. "CR3". */
  label: string;
  exif: RawExif;
}

export interface LoadedImage {
  file: File;
  width: number;
  height: number;
  proxy: ImageBitmap;
  thumbUrl: string;
  raw: RawInfo | null;
}

const files = new Map<string, File>();
/** For RAW originals: the embedded camera JPEG (a lazy slice of the file) and its orientation. */
const rawPreviews = new Map<string, { blob: Blob; orientation: number }>();
const proxies = new Map<string, ImageBitmap>();
const masks = new Map<string, { key: string; data: Uint8Array }>();
const edits = new Map<string, { key: unknown[]; data: Uint8Array | null }>();
const parses = new Map<string, ImageBitmap>();
const hairRois = new Map<string, Uint8Array>();

/** A depth map ready for Renderer.setDepth: a decoded image or single-channel bytes. */
export interface DepthData {
  source: ImageBitmap | Uint8Array;
  width: number;
  height: number;
  kind: 'model' | 'prior';
}
const depths = new Map<string, DepthData>();
const segmentations = new Map<string, ImageBitmap>();
const lensRois = new Map<string, Uint8Array>();

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/avif', 'image/bmp', 'image/tiff'];

/** True for files the importer takes: web formats, TIFF and camera RAW. */
export function isImportable(f: File): boolean {
  return ACCEPTED_TYPES.includes(f.type) || /\.(jpe?g|png|webp|avif|bmp|tiff?)$/i.test(f.name) || isRawFileName(f.name);
}

/** Apply an EXIF orientation (1..8) that the browser did not apply itself. Closes `bmp`. */
export async function orientBitmap(bmp: ImageBitmap, orientation: number): Promise<ImageBitmap> {
  if (orientation <= 1 || orientation > 8) return bmp;
  const { width: w, height: h } = bmp;
  const swap = orientation >= 5;
  const c = new OffscreenCanvas(swap ? h : w, swap ? w : h);
  const ctx = c.getContext('2d')!;
  const t: Record<number, [number, number, number, number, number, number]> = {
    2: [-1, 0, 0, 1, w, 0],
    3: [-1, 0, 0, -1, w, h],
    4: [1, 0, 0, -1, 0, h],
    5: [0, 1, 1, 0, 0, 0],
    6: [0, 1, -1, 0, h, 0],
    7: [0, -1, -1, 0, h, w],
    8: [0, -1, 1, 0, 0, w],
  };
  ctx.setTransform(...t[orientation]);
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return c.transferToImageBitmap();
}

/**
 * Decode any importable file to an upright bitmap. Camera RAW files decode their embedded
 * full-size camera JPEG (see rawParser); full 16-bit development happens on the AI service.
 */
export async function decodeUpright(file: File): Promise<{ bitmap: ImageBitmap; raw: RawInfo | null; preview: { blob: Blob; orientation: number } | null }> {
  if (!isRawFileName(file.name)) return { bitmap: await createImageBitmap(file, { imageOrientation: 'from-image' }), raw: null, preview: null };
  const r = await parseRawFile(file);
  const bmp = await createImageBitmap(r.preview, { imageOrientation: 'from-image' });
  return {
    bitmap: await orientBitmap(bmp, r.orientation),
    raw: { format: r.format, label: r.label, exif: r.exif },
    preview: { blob: r.preview, orientation: r.orientation },
  };
}

/** Full-size upright bitmap of an imported photo (for GPU export). */
export async function decodeOriginal(id: string): Promise<ImageBitmap> {
  const p = rawPreviews.get(id);
  if (p) return orientBitmap(await createImageBitmap(p.blob, { imageOrientation: 'from-image' }), p.orientation);
  const f = files.get(id);
  if (!f) throw new Error('Photo is no longer loaded');
  return createImageBitmap(f, { imageOrientation: 'from-image' });
}

export const isRawPhoto = (id: string) => rawPreviews.has(id);

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
  const { bitmap: full, raw, preview } = await decodeUpright(file);
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
    if (preview) rawPreviews.set(id, preview);
    return { file, width: full.width, height: full.height, proxy, thumbUrl: URL.createObjectURL(blob), raw };
  } finally {
    full.close();
  }
}

export const getFile = (id: string) => files.get(id);
export const getProxy = (id: string) => proxies.get(id);
/** Face mask cache; `key` names what it was built from (size, parsing, body skin). */
export const getCachedMask = (id: string, key: string) => (masks.get(id)?.key === key ? masks.get(id)!.data : undefined);
export const setCachedMask = (id: string, key: string, data: Uint8Array) => masks.set(id, { key, data });
/** Brush-stroke texture cache, keyed by [strokes array, feather, width, height] identity. */
export const getCachedEdit = (id: string, key: unknown[]) => {
  const e = edits.get(id);
  return e && e.key.length === key.length && e.key.every((k, i) => k === key[i]) ? e : undefined;
};
export const setCachedEdit = (id: string, key: unknown[], data: Uint8Array | null) => edits.set(id, { key, data });
export const getParse = (id: string) => parses.get(id);
export const setParse = (id: string, bmp: ImageBitmap) => {
  parses.get(id)?.close();
  parses.set(id, bmp);
  masks.delete(id);
};
export const clearCachedMask = (id: string) => {
  masks.delete(id);
  edits.delete(id);
  hairRois.delete(id);
  lensRois.delete(id);
};
export const getCachedHairRoi = (id: string) => hairRois.get(id);
export const setCachedHairRoi = (id: string, m: Uint8Array) => hairRois.set(id, m);
export const getDepth = (id: string) => depths.get(id);
export const setDepth = (id: string, d: DepthData) => {
  const old = depths.get(id);
  if (old && old.source !== d.source && 'close' in old.source) old.source.close();
  depths.set(id, d);
};

export const getSegmentation = (id: string) => segmentations.get(id);
export const setSegmentation = (id: string, bmp: ImageBitmap) => {
  segmentations.get(id)?.close();
  segmentations.set(id, bmp);
};
export const getCachedLensRoi = (id: string) => lensRois.get(id);
export const setCachedLensRoi = (id: string, m: Uint8Array) => lensRois.set(id, m);

export function forgetImage(id: string) {
  segmentations.get(id)?.close();
  segmentations.delete(id);
  lensRois.delete(id);
  proxies.get(id)?.close();
  proxies.delete(id);
  files.delete(id);
  rawPreviews.delete(id);
  masks.delete(id);
  edits.delete(id);
  parses.get(id)?.close();
  parses.delete(id);
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
