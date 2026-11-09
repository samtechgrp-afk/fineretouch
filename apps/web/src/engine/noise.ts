/**
 * Noise level of a photo for the skin noise reduction preview (port of estimate_noise in
 * services/ai/app/engines/noise.py). Sensor noise is independent per colour channel, so it is
 * measured on the chroma fine bands, where pores and wrinkles (almost pure luma) cannot inflate
 * it, then converted to a luma level and capped by the luma band's own MAD.
 */

export const NOISE_FLOOR = 0.002;
export const CHROMA_TO_LUMA = 0.6686 / 0.64;

/** rgba: 8-bit RGBA pixels (w x h, native resolution, e.g. a centre crop). Returns 0..1 luma. */
export function estimateNoiseLevel(rgba: Uint8Array | Uint8ClampedArray, w: number, h: number): number {
  if (w < 3 || h < 3) return NOISE_FLOOR;
  const n = w * h;
  const y = new Float32Array(n);
  const cb = new Float32Array(n);
  const cr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] / 255;
    const g = rgba[i * 4 + 1] / 255;
    const b = rgba[i * 4 + 2] / 255;
    y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
    cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
  }
  const fine = (ch: Float32Array) => {
    const out: number[] = [];
    for (let yy = 1; yy < h - 1; yy++)
      for (let xx = 1; xx < w - 1; xx++) {
        let s = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += ch[(yy + dy) * w + xx + dx];
        out.push(Math.abs(ch[yy * w + xx] - s / 9));
      }
    return out;
  };
  const mad = (v: number[]) => {
    v.sort((a, b) => a - b);
    return 1.4826 * v[v.length >> 1];
  };
  // The 3x3 box removes 8/9 of white noise variance less than a sigma-0.9 Gaussian does; the
  // ratio is the same for luma and chroma, so it cancels in the chroma-to-luma conversion.
  const fromChroma = CHROMA_TO_LUMA * 0.5 * (mad(fine(cb)) + mad(fine(cr)));
  return Math.max(NOISE_FLOOR, Math.min(fromChroma, mad(fine(y))));
}
