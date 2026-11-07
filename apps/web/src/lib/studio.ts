'use client';

import { buildLensRoiMask } from '@/engine/masks';
import type { ImageAnalysis } from '@fineretouch/shared';
import { segmentMap } from './ai';
import { proxyJpeg } from './depth';
import { fitSize, getCachedLensRoi, getProxy, getSegmentation, setCachedLensRoi, setSegmentation } from './images';
import { useEditor } from './store';

const LENS_ROI_LONG_EDGE = 1024;
const inflight = new Map<string, Promise<ImageBitmap | null>>();

/** Eyeglass lens region for the glare preview (cached per photo; sampled by UV). */
export function lensRoiFor(id: string, analysis: ImageAnalysis | null, aspectW: number, aspectH: number) {
  if (!analysis?.faces.length) return null;
  const [w, h] = fitSize(aspectW, aspectH, LENS_ROI_LONG_EDGE);
  let data = getCachedLensRoi(id);
  if (!data || data.length !== w * h) {
    data = buildLensRoiMask(analysis.faces, w, h);
    setCachedLensRoi(id, data);
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
  if (!st.aiOnline || !getProxy(id)) {
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
