'use client';

import { buildLensRoiMask } from '@/engine/masks';
import { glareFaces, type ImageAnalysis } from '@fineretouch/shared';
import { parseMap, segmentMap } from './ai';
import { proxyJpeg } from './depth';
import { fitSize, getCachedLensRoi, getParse, getSegmentation, hasImage, setCachedLensRoi, setParse, setSegmentation } from './images';
import { useEditor } from './store';

const LENS_ROI_LONG_EDGE = 1024;
const inflight = new Map<string, Promise<ImageBitmap | null>>();
const parseInflight = new Map<string, Promise<ImageBitmap | null>>();

/**
 * Eyeglass lens region for the glare preview (cached per photo; sampled by UV). Only faces whose
 * glasses the engine confidently detected get lenses, or every face with `force` (the manual
 * override), as in the export (pipeline.py 5d); null when no face qualifies.
 */
export function lensRoiFor(id: string, analysis: ImageAnalysis | null, aspectW: number, aspectH: number, force = false) {
  const faces = glareFaces(analysis?.faces ?? [], force);
  if (!faces.length) return null;
  const [w, h] = fitSize(aspectW, aspectH, LENS_ROI_LONG_EDGE);
  const key = `${id}:${force ? 'forced' : 'detected'}`;
  let data = getCachedLensRoi(key);
  if (!data || data.length !== w * h) {
    data = buildLensRoiMask(faces, w, h);
    setCachedLensRoi(key, data);
  }
  return { data, width: w, height: h };
}

/**
 * Garment / hair / person masks from the engine's segmenter, for the clothing iron and
 * recolour preview. Without the engine (or its model) the preview shows no clothing
 * effect; exports from the engine still run it with their own fallback mask.
 */
export function ensureSegmentation(id: string): Promise<ImageBitmap | null> {
  const cached = getSegmentation(id);
  if (cached) return Promise.resolve(cached);
  const st = useEditor.getState();
  if (!st.aiOnline || !hasImage(id)) {
    st.setSegmentState(id, 'unavailable');
    return Promise.resolve(null);
  }
  const running = inflight.get(id);
  if (running) return running;
  st.setSegmentState(id, 'pending');
  const job = (async () => {
    try {
      const bmp = await createImageBitmap(await segmentMap(await proxyJpeg(id)));
      setSegmentation(id, bmp);
      useEditor.getState().setSegmentState(id, 'ready');
      return bmp;
    } catch (e) {
      console.warn('segmentation unavailable', e);
      useEditor.getState().setSegmentState(id, 'unavailable');
      return null;
    }
  })().finally(() => inflight.delete(id));
  inflight.set(id, job);
  return job;
}

/**
 * AI face parsing (face skin, body skin, hair) from the engine, for the semantic skin mask in
 * the preview and the in-browser export. Without the engine or its model the landmark-only
 * skin mask is used; exports rendered by the engine parse the full image themselves.
 */
export function ensureParse(id: string): Promise<ImageBitmap | null> {
  const cached = getParse(id);
  if (cached) return Promise.resolve(cached);
  const st = useEditor.getState();
  if (!st.aiOnline || !hasImage(id)) {
    st.setParseState(id, 'unavailable');
    return Promise.resolve(null);
  }
  const running = parseInflight.get(id);
  if (running) return running;
  st.setParseState(id, 'pending');
  const job = (async () => {
    try {
      const bmp = await createImageBitmap(await parseMap(await proxyJpeg(id)));
      setParse(id, bmp);
      useEditor.getState().setParseState(id, 'ready');
      return bmp;
    } catch (e) {
      console.warn('face parsing unavailable', e);
      useEditor.getState().setParseState(id, 'unavailable');
      return null;
    }
  })().finally(() => parseInflight.delete(id));
  parseInflight.set(id, job);
  return job;
}
