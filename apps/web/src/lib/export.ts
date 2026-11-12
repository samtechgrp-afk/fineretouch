'use client';

import { layerIsActive, splitSigma, type Adjustments, type ColorStats, type ImageAnalysis, type SkinMaskSettings } from '@fineretouch/shared';
import JSZip from 'jszip';
import { Renderer } from '@/engine/renderer';
import { buildEditMask, buildFaceMask } from '@/engine/masks';
import { ensureDepth, hairRoiFor } from './depth';
import { ensureParse, ensureSegmentation, lensRoiFor } from './studio';
import { fitSize, getCachedEdit, getCachedMask, getParse, setCachedEdit, setCachedMask } from './images';
import { noiseLevelOf } from './noise';
import { LAYER_LONG_EDGE, applyHealForExport, syncLayerMasks } from './localRetouch';
import { BLUR_SIGMAS, BOKEH_MAX_WORKING_COC, BOKEH_RADIUS_SHARE, FS_SMOOTH_RATIO } from '@/engine/shaders';
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
/** Widest detected face as a share of the photo's width (for face-scaled preview stages). */
export function faceShareOf(analysis: ImageAnalysis | null | undefined): number | undefined {
  const faces = analysis?.faces ?? [];
  return faces.length ? Math.max(...faces.map((f) => f.box[2])) : undefined;
}

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

/** GPU memory per pixel of a render: about 13 RGBA8 targets (source, bands, passes, layers). */
export const GPU_BYTES_PER_PIXEL = 56;
/** GPU memory budgets for export (Settings > Processing). */
export const GPU_BUDGETS = { low: 512, balanced: 1536, high: 3072 } as const;
export type GpuBudget = keyof typeof GPU_BUDGETS;
/** Below this, tiles would be mostly overlap: the export is scaled down instead (and says so). */
const MIN_TILE_INNER = 256;

/**
 * Pixels of context each tile needs on every side so its inner part renders exactly as in a
 * whole-photo pass: the widest blur band (3 sigma), each frequency-separation layer's blur, and
 * the bokeh gather (whose radii add up because the stages run one after another).
 */
export function exportApron(adj: Adjustments, longEdge: number): number {
  let r = 3 * BLUR_SIGMAS.s3 * longEdge + 4;
  for (const l of adj.localRetouch.layers) {
    if (!layerIsActive(l) || l.kind === 'dodgeBurn') continue;
    if (l.toneSmoothing > 0 || l.texture !== 100) r += 3 * FS_SMOOTH_RATIO * splitSigma(l.splitRadius, longEdge) + 2;
  }
  const b = adj.depthBokeh;
  if (b.enabled && b.blurRadius > 0) {
    const maxCoc = (b.blurRadius / 100) * BOKEH_RADIUS_SHARE * longEdge * (b.bokehMultiplier / 100);
    r += 1.5 * maxCoc + 2 * Math.max(1, maxCoc / BOKEH_MAX_WORKING_COC) + 2;
  }
  return Math.ceil(r);
}

export interface Tile {
  /** Rendered rect (inner part plus apron, clipped to the photo). */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Rect this tile writes to the output. */
  ix: number;
  iy: number;
  iw: number;
  ih: number;
}

/** Split a w x h photo into tiles of at most `maxTile` px a side with `apron` px of context. */
export function planTiles(w: number, h: number, maxTile: number, apron: number): Tile[] | null {
  const inner = maxTile - 2 * apron;
  if (w <= maxTile && h <= maxTile) return [{ x: 0, y: 0, w, h, ix: 0, iy: 0, iw: w, ih: h }];
  if (inner < MIN_TILE_INNER) return null;
  // Even split, so no tile is a thin sliver.
  const nx = Math.ceil(w / Math.min(inner, w)), ny = Math.ceil(h / Math.min(inner, h));
  const tiles: Tile[] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const ix = Math.round((i * w) / nx), iy = Math.round((j * h) / ny);
      const iw = Math.round(((i + 1) * w) / nx) - ix, ih = Math.round(((j + 1) * h) / ny) - iy;
      const x = Math.max(0, ix - apron), y = Math.max(0, iy - apron);
      tiles.push({ x, y, w: Math.min(w, ix + iw + apron) - x, h: Math.min(h, iy + ih + apron) - y, ix, iy, iw, ih });
    }
  }
  return tiles;
}

export interface ExportPlan {
  width: number;
  height: number;
  tiles: Tile[];
  /** True when the GPU could not render this size even in tiles, so the output was scaled down. */
  scaled: boolean;
}

/**
 * How a w x h photo is rendered on this GPU: in one pass when it fits the texture limit and the
 * memory budget, otherwise in tiles; scaled down only when even tiles cannot fit.
 */
export function planExport(srcW: number, srcH: number, adj: Adjustments, maxTexture: number, budgetMb: number, maxLongEdge?: number): ExportPlan {
  let [w, h] = fitSize(srcW, srcH, maxLongEdge ?? Infinity);
  const budgetSide = Math.floor(Math.sqrt((budgetMb * 2 ** 20) / GPU_BYTES_PER_PIXEL));
  const maxTile = Math.max(512, Math.min(maxTexture, budgetSide));
  if (w <= maxTexture && h <= maxTexture && w * h * GPU_BYTES_PER_PIXEL <= budgetMb * 2 ** 20) {
    return { width: w, height: h, tiles: [{ x: 0, y: 0, w, h, ix: 0, iy: 0, iw: w, ih: h }], scaled: false };
  }
  let scaled = false;
  for (;;) {
    const tiles = planTiles(w, h, maxTile, exportApron(adj, Math.max(w, h)));
    if (tiles) return { width: w, height: h, tiles, scaled };
    // The apron grows with the photo, so a smaller output needs less of it.
    [w, h] = fitSize(w, h, Math.floor(Math.max(w, h) * 0.8));
    scaled = true;
  }
}

/** Rows y..y+h, columns x..x+w of an RGBA buffer that is `width` px wide. */
function cropRgba(px: Uint8ClampedArray, width: number, t: { x: number; y: number; w: number; h: number }): ImageData {
  const out = new Uint8ClampedArray(t.w * t.h * 4);
  for (let r = 0; r < t.h; r++) {
    const s = ((t.y + r) * width + t.x) * 4;
    out.set(px.subarray(s, s + t.w * 4), r * t.w * 4);
  }
  return new ImageData(out, t.w, t.h);
}

export interface RenderOptions {
  /** Reuse this renderer (batch export) instead of creating a WebGL context per photo. */
  renderer?: Renderer;
  /** GPU memory budget in MB (tiles above it). */
  gpuBudgetMb?: number;
  /** Told how the photo is rendered (tiles, scaling) before it starts. */
  onPlan?(plan: ExportPlan): void;
}

/**
 * Full-resolution export on the GPU: the same shader chain as the viewport, run at the
 * original size. Photos larger than the GPU texture limit or the memory budget are rendered in
 * overlapping tiles (v0.12), each with enough context that the result matches a single pass.
 */
export async function renderFullRes(
  /** Upright full-size bitmap of the original (see decodeOriginal; RAW uses its embedded JPEG). */
  loadSource: () => Promise<ImageBitmap>,
  adjustments: Adjustments,
  ctx: { id: string; analysis: ImageAnalysis | null; sourceStats: ColorStats | null },
  opts: ExportOptions,
  ropts: RenderOptions = {},
): Promise<Blob> {
  const bmp = await loadSource();
  const own = !ropts.renderer;
  const renderer = ropts.renderer ?? new Renderer(new OffscreenCanvas(1, 1));
  try {
    const plan = planExport(bmp.width, bmp.height, adjustments, renderer.maxTextureSize, ropts.gpuBudgetMb ?? GPU_BUDGETS.balanced, opts.maxLongEdge);
    ropts.onPlan?.(plan);
    const { width: w, height: h } = plan;
    const source = w === bmp.width && h === bmp.height ? bmp : await createImageBitmap(bmp, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' });
    // Heal / clone strokes at the export size (same code as the preview, stage 0b).
    const healed = adjustments.localRetouch.heal.length ? applyHealForExport(source, adjustments.localRetouch.heal) : null;
    const noiseLevel = adjustments.skinNoise > 0 ? noiseLevelOf(`${ctx.id}:full:${w}x${h}`, source) : undefined;
    if (adjustments.skinMask.useSegmentation && ctx.analysis?.faces.length) await ensureParse(ctx.id);
    const seg = adjustments.clothing.enabled ? await ensureSegmentation(ctx.id) : null;
    const depth = adjustments.depthBokeh.enabled ? await ensureDepth(ctx.id) : null;
    // Whole-photo masks (small, sampled by UV), built once and given to every tile.
    const mask = faceMaskFor(ctx.id, ctx.analysis, w, h, adjustments.skinMask);
    const edit = editMaskFor(ctx.id, adjustments.skinMask, w, h);
    const hair = adjustments.strayHairRemoval.enabled ? hairRoiFor(ctx.id, ctx.analysis, w, h) : null;
    const lens = adjustments.glareReduction > 0 ? lensRoiFor(ctx.id, ctx.analysis, w, h) : null;
    const sm = adjustments.skinMask;
    const layerKey = `export:${mask ? 'face' : 'none'}:${sm.strokes.length}:${sm.feather}:${sm.useSegmentation}:${sm.includeBodySkin}`;

    const out = new OffscreenCanvas(w, h);
    const ctx2d = out.getContext('2d')!;
    for (const t of plan.tiles) {
      const whole = t.w === w && t.h === h;
      const tileSrc = whole ? source : await createImageBitmap(source, t.x, t.y, t.w, t.h);
      renderer.setSource(tileSrc, t.w, t.h, { x: t.x, y: t.y, fullW: w, fullH: h });
      if (tileSrc !== source) tileSrc.close();
      if (healed) renderer.setHealedSource(whole ? healed : cropRgba(healed, w, t).data);
      if (mask) renderer.setMask(mask.data, mask.width, mask.height);
      if (edit) renderer.setEdit(edit.data, edit.width, edit.height);
      syncLayerMasks(renderer, `${ctx.id}:export`, adjustments.localRetouch, fitSize(w, h, LAYER_LONG_EDGE), mask, edit, layerKey);
      if (hair) renderer.setHairRoi(hair.data, hair.width, hair.height);
      if (lens) renderer.setLensRoi(lens.data, lens.width, lens.height);
      if (seg) renderer.setSegmentation(seg, seg.width, seg.height);
      if (depth) renderer.setDepth(depth.source, depth.width, depth.height);
      renderer.render(adjustments, { hasFace: Boolean(mask), sourceStats: ctx.sourceStats, noiseLevel, faceShare: faceShareOf(ctx.analysis) });
      const px = renderer.readPixels({ x: t.ix - t.x, y: t.iy - t.y, width: t.iw, height: t.ih });
      ctx2d.putImageData(new ImageData(px, t.iw, t.ih), t.ix, t.iy);
      // Let the browser breathe between tiles (UI stays responsive on a big export).
      if (plan.tiles.length > 1) await new Promise((r) => setTimeout(r, 0));
    }
    if (source !== bmp) source.close();
    if (opts.watermark && watermarkActive(opts.watermark)) await drawWatermark(ctx2d, opts.watermark, w, h);
    return await out.convertToBlob({ type: MIME[opts.format], quality: opts.quality / 100 });
  } finally {
    bmp.close();
    if (own) renderer.dispose(true);
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
