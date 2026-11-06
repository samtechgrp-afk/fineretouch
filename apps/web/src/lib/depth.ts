'use client';

import { buildDepthPrior, buildHairRoiMask } from '@/engine/masks';
import type { ImageAnalysis } from '@fineretouch/shared';
import { depthMap } from './ai';
import { fitSize, getCachedHairRoi, getDepth, getProxy, setCachedHairRoi, setDepth, type DepthData } from './images';
import { useEditor } from './store';

const PRIOR_LONG_EDGE = 256;
const HAIR_ROI_LONG_EDGE = 1024;
const inflight = new Map<string, Promise<DepthData | null>>();

/** Downscaled JPEG of the proxy: what the AI service analyses (landmarks/depth don't need full res). */
export async function proxyJpeg(id: string, longEdge = 1280): Promise<Blob> {
  const proxy = getProxy(id)!;
  const [w, h] = fitSize(proxy.width, proxy.height, longEdge);
  const c = new OffscreenCanvas(w, h);
  c.getContext('2d')!.drawImage(proxy, 0, 0, w, h);
  return c.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
}

function priorDepth(id: string): DepthData | null {
  const st = useEditor.getState();
  const p = st.photos[id];
  const proxy = getProxy(id);
  if (!p || !proxy) return null;
  const [w, h] = fitSize(proxy.width, proxy.height, PRIOR_LONG_EDGE);
  return { source: buildDepthPrior(p.analysis?.faces ?? [], w, h), width: w, height: h, kind: 'prior' };
}

/**
 * Depth for the bokeh preview and GPU export. Uses the AI service's depth model when it is
 * online, otherwise a portrait prior built from the detected faces. Cached per photo; a
 * prior is upgraded to the model's depth the next time it is requested with the service up.
 */
export function ensureDepth(id: string): Promise<DepthData | null> {
  const cached = getDepth(id);
  const online = useEditor.getState().aiOnline;
  if (cached && (cached.kind === 'model' || !online)) return Promise.resolve(cached);
  const running = inflight.get(id);
  if (running) return running;
  const st = useEditor.getState();
  st.setDepthState(id, 'pending');
  const job = (async () => {
    let d: DepthData | null = null;
    if (online && getProxy(id)) {
      try {
        const bmp = await createImageBitmap(await depthMap(await proxyJpeg(id)));
        d = { source: bmp, width: bmp.width, height: bmp.height, kind: 'model' };
      } catch (e) {
        console.warn('depth service failed, using portrait prior', e);
      }
    }
    d ??= priorDepth(id);
    if (d) setDepth(id, d);
    useEditor.getState().setDepthState(id, d ? d.kind : 'error');
    return d;
  })().finally(() => inflight.delete(id));
  inflight.set(id, job);
  return job;
}

/** Stray-hair region for a photo (cached; sampled by UV so its size is independent of the target). */
export function hairRoiFor(id: string, analysis: ImageAnalysis | null, aspectW: number, aspectH: number) {
  if (!analysis?.faces.length) return null;
  const [w, h] = fitSize(aspectW, aspectH, HAIR_ROI_LONG_EDGE);
  let data = getCachedHairRoi(id);
  if (!data || data.length !== w * h) {
    data = buildHairRoiMask(analysis.faces, w, h);
    setCachedHairRoi(id, data);
  }
  return { data, width: w, height: h };
}
