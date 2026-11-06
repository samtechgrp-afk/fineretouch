import type { StyleProfile } from '@fineretouch/shared';
import { colorStatsFromRgba, rgb8ToLab } from './color';

/**
 * Style metrics of an image, measured on RGBA pixels at <= 512 px long edge.
 * Port of style_profile in services/ai/app/retouch_engine.py; keep the two in step.
 */
export function profileFromRgba(px: Uint8ClampedArray | Uint8Array, width: number, height: number): StyleProfile {
  const n = width * height;
  const stats = colorStatsFromRgba(px);
  const grey = new Float32Array(n);
  let chroma = 0;
  const skin = new Uint8Array(n);
  let skinCount = 0;
  let skinL = 0;
  for (let i = 0; i < n; i++) {
    const r = px[i * 4];
    const g = px[i * 4 + 1];
    const b = px[i * 4 + 2];
    grey[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    const lab = rgb8ToLab(r, g, b);
    chroma += Math.hypot(lab[1], lab[2]);
    if (skinProbability(r / 255, g / 255, b / 255) > 0.5) {
      skin[i] = 1;
      skinCount++;
      skinL += lab[0];
    }
  }

  // Variance of the 4-neighbour Laplacian (replicated borders), as cv2.Laplacian(ksize=1).
  const at = (x: number, y: number) => grey[Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))];
  let lapSum = 0;
  let lapSq = 0;
  // 5x5 box blur (replicated borders) for the skin high-pass, as cv2.blur((5, 5)).
  let hpSkin = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const c = grey[y * width + x];
      const lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * c;
      lapSum += lap;
      lapSq += lap * lap;
      if (skin[y * width + x]) {
        let box = 0;
        for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) box += at(x + dx, y + dy);
        hpSkin += Math.abs(c - box / 25);
      }
    }
  }
  const lapMean = lapSum / n;
  const enoughSkin = skinCount >= Math.max(64, 0.01 * n);
  return {
    stats,
    meanL: stats.mean[0],
    contrast: stats.std[0],
    warmth: stats.mean[2],
    tintAxis: stats.mean[1],
    chroma: chroma / n,
    sharpness: lapSq / n - lapMean * lapMean,
    skinLuma: enoughSkin ? skinL / skinCount : null,
    skinTexture: enoughSkin ? hpSkin / skinCount : null,
  };
}

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** YCbCr skin box, identical to skinProb in shaders.ts and skin_probability in pipeline.py. */
export function skinProbability(r: number, g: number, b: number): number {
  const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5;
  const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5;
  return (
    smoothstep(0.5, 0.55, cr) * (1 - smoothstep(0.66, 0.71, cr)) * smoothstep(0.27, 0.32, cb) * (1 - smoothstep(0.48, 0.53, cb))
  );
}
