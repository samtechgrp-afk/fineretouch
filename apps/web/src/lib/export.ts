'use client';

import type { Adjustments, ColorStats, ImageAnalysis, SkinMaskSettings } from '@fineretouch/shared';
import JSZip from 'jszip';
import { Renderer } from '@/engine/renderer';
import { buildEditMask, buildFaceMask } from '@/engine/masks';
import { ensureDepth, hairRoiFor } from './depth';
import { ensureParse, ensureSegmentation, lensRoiFor } from './studio';
import { fitSize, getCachedEdit, getCachedMask, getParse, setCachedEdit, setCachedMask } from './images';
import { noiseLevelOf } from './noise';
import { drawWatermark, watermarkActive, type WatermarkSettings } from './watermark';

export const MASK_LONG_EDGE = 1536;

export interface MaskData {
  data: Uint8Array;
  width: number;
  height: number;
}

/** Brush strokes are drawn at this size (they are soft, and sampled by UV like the face mask). */
export const EDIT_LONG_EDGE = 1536;

function parsePixels(bmp: ImageBitmap, w: number, h: number): Uint8ClampedArray {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

/**
 * Face mask for a photo at a fixed working size (cached; masks are soft and sampled by UV).
 * With the engine's face parsing (and skinMask.useSegmentation) the skin channel is the
 * semantic mask of pipeline.build_masks; otherwise landmarks only.
 */
export function faceMaskFor(
  id: string,
  analysis: ImageAnalysis | null,
  aspectW: number,
  aspectH: number,
  skinMask?: SkinMaskSettings,
): MaskData | null {
  if (!analysis?.faces.length) return null;
  const [w, h] = fitSize(aspectW, aspectH, MASK_LONG_EDGE);
  const parse = skinMask?.useSegmentation ? getParse(id) : undefined;
  const key = `${w}x${h}:${parse ? (skinMask!.includeBodySkin ? 'parse+body' : 'parse') : 'landmarks'}:${analysis.faces.length}`;
  let data = getCachedMask(id, key);
  if (!data) {
    data = buildFaceMask(analysis.faces, w, h, parse ? { data: parsePixels(parse, w, h), includeBody: skinMask!.includeBodySkin } : null);
    setCachedMask(id, key, data);
  }
  return { data, width: w, height: h };
}

/** The user's brush strokes as a texture (r add, g erase), or null without strokes. Cached. */
export function editMaskFor(id: string, skinMask: SkinMaskSettings, aspectW: number, aspectH: number): MaskData | null {
  if (!skinMask.strokes.length) return null;
  const [w, h] = fitSize(aspectW, aspectH, EDIT_LONG_EDGE);
  const key = [skinMask.strokes, skinMask.feather, w, h];
  let entry = getCachedEdit(id, key);
  if (!entry) {
    setCachedEdit(id, key, buildEditMask(skinMask.strokes, w, h, skinMask.feather));
    entry = getCachedEdit(id, key)!;
  }
  return entry.data ? { data: entry.data, width: w, height: h } : null;
}

export interface ExportOptions {
  format: 'jpeg' | 'png' | 'webp';
  quality: number; // 1..100
  maxLongEdge?: number;
  /** Burned into the pixels right before encoding (sized relative to the image). */
  watermark?: WatermarkSettings;
}

export const MIME = { jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' } as const;
export const EXT = { jpeg: 'jpg', png: 'png', webp: 'webp' } as const;

/**
 * Full-resolution export on the GPU: the same shader chain as the viewport,
 * run once at the original size (or the largest texture the GPU allows).
 */
export async function renderFullRes(
  /** Upright full-size bitmap of the original (see decodeOriginal; RAW uses its embedded JPEG). */
  loadSource: () => Promise<ImageBitmap>,
  adjustments: Adjustments,
  ctx: { id: string; analysis: ImageAnalysis | null; sourceStats: ColorStats | null },
  opts: ExportOptions,
): Promise<Blob> {
  const bmp = await loadSource();
  const canvas = new OffscreenCanvas(1, 1);
  const renderer = new Renderer(canvas);
  try {
    const limit = Math.min(renderer.maxTextureSize, opts.maxLongEdge ?? Infinity);
    const [w, h] = fitSize(bmp.width, bmp.height, limit);
    const source = w === bmp.width && h === bmp.height ? bmp : await createImageBitmap(bmp, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' });
    renderer.setSource(source, w, h);
    const noiseLevel = adjustments.skinNoise > 0 ? noiseLevelOf(`${ctx.id}:full:${w}x${h}`, source) : undefined;
    if (source !== bmp) source.close();
    if (adjustments.skinMask.useSegmentation && ctx.analysis?.faces.length) await ensureParse(ctx.id);
    const mask = faceMaskFor(ctx.id, ctx.analysis, w, h, adjustments.skinMask);
    if (mask) renderer.setMask(mask.data, mask.width, mask.height);
    const edit = editMaskFor(ctx.id, adjustments.skinMask, w, h);
    if (edit) renderer.setEdit(edit.data, edit.width, edit.height);
    if (adjustments.strayHairRemoval.enabled) {
      const roi = hairRoiFor(ctx.id, ctx.analysis, w, h);
      if (roi) renderer.setHairRoi(roi.data, roi.width, roi.height);
    }
    if (adjustments.glareReduction > 0) {
      const roi = lensRoiFor(ctx.id, ctx.analysis, w, h);
      if (roi) renderer.setLensRoi(roi.data, roi.width, roi.height);
    }
    if (adjustments.clothing.enabled) {
      const seg = await ensureSegmentation(ctx.id);
      if (seg) renderer.setSegmentation(seg, seg.width, seg.height);
    }
    if (adjustments.depthBokeh.enabled) {
      const d = await ensureDepth(ctx.id);
      if (d) renderer.setDepth(d.source, d.width, d.height);
    }
    renderer.render(adjustments, { hasFace: Boolean(mask), sourceStats: ctx.sourceStats, noiseLevel });
    const pixels = renderer.readPixels();
    const out = new OffscreenCanvas(w, h);
    const ctx2d = out.getContext('2d')!;
    ctx2d.putImageData(new ImageData(pixels, w, h), 0, 0);
    if (opts.watermark && watermarkActive(opts.watermark)) await drawWatermark(ctx2d, opts.watermark, w, h);
    return await out.convertToBlob({ type: MIME[opts.format], quality: opts.quality / 100 });
  } finally {
    bmp.close();
    renderer.dispose(true);
  }
}

export function exportName(original: string, format: ExportOptions['format'], used: Set<string>): string {
  const base = original.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_') || 'photo';
  let name = `${base}_fineretouch.${EXT[format]}`;
  for (let i = 2; used.has(name); i++) name = `${base}_fineretouch_${i}.${EXT[format]}`;
  used.add(name);
  return name;
}

export async function zipBlobs(entries: Array<{ name: string; blob: Blob }>): Promise<Blob> {
  const zip = new JSZip();
  for (const e of entries) zip.file(e.name, e.blob);
  // Images are already compressed; STORE avoids burning CPU for ~0% gain.
  return zip.generateAsync({ type: 'blob', compression: 'STORE' });
}

export function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
