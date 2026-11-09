'use client';

import type { HealStroke, LocalRetouch, RetouchLayer } from '@fineretouch/shared';
import { layerIsActive } from '@fineretouch/shared';
import { applyHealStroke, buildLayerMask } from '@/engine/heal';
import type { Renderer } from '@/engine/renderer';
import type { MaskData } from './export';

/** Layer masks are drawn at this size (soft, sampled by UV like the face mask). */
export const LAYER_LONG_EDGE = 1536;

/** Cheap structural signature of a stroke list (strokes are immutable once committed). */
function strokeSig(s: { mode: string; size: number; points: number[] } & Record<string, unknown>): string {
  const p = s.points;
  return `${s.mode}:${s.size}:${s.flow ?? ''}:${s.hardness ?? ''}:${s.opacity ?? ''}:${(s.offset as number[] | undefined)?.join(',') ?? ''}:${p.length}:${p[0]},${p[1]}:${p[p.length - 2]},${p[p.length - 1]}`;
}

interface HealEntry {
  sigs: string[];
  pixels: Uint8ClampedArray;
}
const healed = new Map<string, HealEntry>();
const basePixels = new Map<string, Uint8ClampedArray>();

/** Upright RGBA pixels of a bitmap at its own size (cached per key). */
export function pixelsOf(key: string, bmp: ImageBitmap): Uint8ClampedArray {
  let px = basePixels.get(key);
  if (!px) {
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = c.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bmp, 0, 0);
    px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
    basePixels.set(key, px);
  }
  return px;
}

/**
 * The photo with its heal / clone strokes applied (null without strokes). Incremental: when
 * strokes were only added since the last call, only the new ones are replayed.
 */
export function healedPixels(key: string, base: Uint8ClampedArray, w: number, h: number, strokes: readonly HealStroke[]): Uint8ClampedArray | null {
  if (!strokes.length) {
    healed.delete(key);
    return null;
  }
  const sigs = strokes.map((s) => strokeSig(s as never));
  const hit = healed.get(key);
  if (hit && hit.sigs.length <= sigs.length && hit.sigs.every((v, i) => v === sigs[i])) {
    if (hit.sigs.length === sigs.length) return hit.pixels;
    const px = new Uint8ClampedArray(hit.pixels);
    for (const s of strokes.slice(hit.sigs.length)) applyHealStroke(px, w, h, s);
    healed.set(key, { sigs, pixels: px });
    return px;
  }
  const px = new Uint8ClampedArray(base);
  for (const s of strokes) applyHealStroke(px, w, h, s);
  healed.set(key, { sigs, pixels: px });
  return px;
}

/** Heal / clone strokes on a full-size bitmap for a browser export (not cached). */
export function applyHealForExport(bmp: ImageBitmap, strokes: readonly HealStroke[]): Uint8ClampedArray {
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  for (const s of strokes) applyHealStroke(px, bmp.width, bmp.height, s);
  return px;
}

/** Skin weight for the layers' 'skin' base, from the face mask (r) and the skin-mask strokes. */
export function skinWeight(face: MaskData | null, edit: MaskData | null): Float32Array | null {
  if (!face) return null;
  const n = face.width * face.height;
  const out = new Float32Array(n);
  const sameEdit = edit && edit.width === face.width && edit.height === face.height ? edit : null;
  for (let i = 0; i < n; i++) {
    let v = face.data[i * 4] / 255;
    if (sameEdit) v = Math.max(v, sameEdit.data[i * 4] / 255) * (1 - sameEdit.data[i * 4 + 1] / 255);
    out[i] = v;
  }
  return out;
}

interface LayerEntry {
  sig: string;
  data: Uint8Array;
}
const layerMasks = new Map<string, LayerEntry>();

function layerSig(l: RetouchLayer, w: number, h: number, skinKey: string): string {
  const m = l.mask;
  return [w, h, l.kind, m.base, m.base === 'skin' ? skinKey : '', m.feather, m.invert, ...m.strokes.map((s) => strokeSig(s as never))].join('|');
}

/** RGBA mask texture of a layer (r coverage, g dodge, b burn), cached per photo and layer. */
export function layerMaskFor(photoId: string, l: RetouchLayer, w: number, h: number, skin: Float32Array | null, skinKey: string): Uint8Array {
  const key = `${photoId}:${l.id}`;
  const sig = layerSig(l, w, h, skin ? skinKey : 'none');
  const hit = layerMasks.get(key);
  if (hit && hit.sig === sig) return hit.data;
  const data = buildLayerMask(l, w, h, skin);
  layerMasks.set(key, { sig, data });
  return data;
}

/**
 * Upload the masks of every active layer to the renderer (only changed ones are re-uploaded)
 * and drop the rest. `size` is the layer mask size, the same aspect as the photo.
 */
export function syncLayerMasks(
  r: Renderer,
  photoId: string,
  local: LocalRetouch,
  size: [number, number],
  face: MaskData | null,
  edit: MaskData | null,
  skinKey: string,
): void {
  const active = local.layers.filter(layerIsActive);
  const needSkin = active.some((l) => l.mask.base === 'skin');
  const fits = face !== null && face.width === size[0] && face.height === size[1];
  const skin = needSkin && fits ? skinWeightCached(photoId, face, edit, skinKey) : null;
  for (const l of active) r.setLayerMask(l.id, layerMaskFor(photoId, l, size[0], size[1], skin, skinKey), size[0], size[1]);
  r.pruneLayerMasks(new Set(active.map((l) => l.id)));
}

const skins = new Map<string, { key: string; data: Float32Array | null }>();
function skinWeightCached(id: string, face: MaskData | null, edit: MaskData | null, key: string): Float32Array | null {
  const hit = skins.get(id);
  if (hit && hit.key === key) return hit.data;
  const data = skinWeight(face, edit);
  skins.set(id, { key, data });
  return data;
}

export function forgetLocalRetouch(id: string): void {
  for (const m of [healed, basePixels, layerMasks, skins] as Map<string, unknown>[]) {
    for (const k of m.keys()) if (k === id || k.startsWith(`${id}:`)) m.delete(k);
  }
}
