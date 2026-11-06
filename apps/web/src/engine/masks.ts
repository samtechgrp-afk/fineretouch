import type { FaceAnalysis, Point } from '@fineretouch/shared';

/**
 * Rasterise the AI service's normalised face regions into one RGBA mask
 * (r = face skin, g = eyes, b = under-eye, a = mouth), feathered with the same
 * relative radii as services/ai/app/pipeline.py:build_masks.
 *
 * Pure CPU (scanline fill + 3x box blur) so it runs in workers, tests and
 * browsers without Canvas `filter` support.
 */
export function buildFaceMask(faces: FaceAnalysis[], width: number, height: number): Uint8Array {
  const n = width * height;
  const skin = new Float32Array(n);
  const eyes = new Float32Array(n);
  const under = new Float32Array(n);
  const mouth = new Float32Array(n);
  let faceW = 1;
  const px = (pts: Point[]) => pts.map(([x, y]) => [x * width, y * height] as [number, number]);

  for (const f of faces) {
    faceW = Math.max(faceW, f.box[2] * width);
    const r = f.regions;
    fillPolygon(skin, width, height, px(r.faceOval), 1);
    const cutouts = [r.leftEye, r.rightEye, r.lips, r.leftBrow, r.rightBrow].filter((p) => p.length >= 3);
    const thickness = Math.max(1, Math.floor(faceW * 0.012));
    for (const c of cutouts) {
      const poly = px(c);
      fillPolygon(skin, width, height, poly, 0);
      strokePolygon(skin, width, height, poly, thickness, 0);
    }
    for (const [region, canvas] of [
      [r.leftEye, eyes],
      [r.rightEye, eyes],
      [r.leftUnderEye, under],
      [r.rightUnderEye, under],
      [r.innerMouth, mouth],
    ] as const) {
      if (region.length >= 3) fillPolygon(canvas, width, height, px(region), 1);
    }
  }

  if (faces.length) {
    gaussianApprox(skin, width, height, faceW * 0.025);
    gaussianApprox(eyes, width, height, faceW * 0.006);
    gaussianApprox(under, width, height, faceW * 0.02);
    gaussianApprox(mouth, width, height, faceW * 0.004);
  }

  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.round(clamp01(skin[i]) * 255);
    out[i * 4 + 1] = Math.round(clamp01(eyes[i]) * 255);
    out[i * 4 + 2] = Math.round(clamp01(under[i]) * 255);
    out[i * 4 + 3] = Math.round(clamp01(mouth[i]) * 255);
  }
  return out;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Even-odd scanline fill, sampling at pixel centres. */
export function fillPolygon(buf: Float32Array, w: number, h: number, pts: [number, number][], value: number): void {
  if (pts.length < 3) return;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [, y] of pts) {
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const y0 = Math.max(0, Math.floor(minY));
  const y1 = Math.min(h - 1, Math.ceil(maxY));
  const xs: number[] = [];
  for (let y = y0; y <= y1; y++) {
    const cy = y + 0.5;
    xs.length = 0;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i];
      const [xj, yj] = pts[j];
      if (yi > cy !== yj > cy) xs.push(xi + ((cy - yi) * (xj - xi)) / (yj - yi));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xa = Math.max(0, Math.ceil(xs[k] - 0.5));
      const xb = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5));
      const row = y * w;
      for (let x = xa; x <= xb; x++) buf[row + x] = value;
    }
  }
}

function strokePolygon(buf: Float32Array, w: number, h: number, pts: [number, number][], thickness: number, value: number) {
  const r = thickness / 2;
  for (let i = 0; i < pts.length; i++) {
    const [ax, ay] = pts[i];
    const [bx, by] = pts[(i + 1) % pts.length];
    const len = Math.hypot(bx - ax, by - ay);
    const steps = Math.max(1, Math.ceil(len));
    for (let s = 0; s <= steps; s++) {
      const cx = ax + ((bx - ax) * s) / steps;
      const cy = ay + ((by - ay) * s) / steps;
      for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(h - 1, Math.ceil(cy + r)); y++) {
        for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(w - 1, Math.ceil(cx + r)); x++) {
          if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) buf[y * w + x] = value;
        }
      }
    }
  }
}

/** Three box-blur passes per axis approximate a Gaussian of the given sigma. */
export function gaussianApprox(buf: Float32Array, w: number, h: number, sigma: number): void {
  if (sigma < 0.5) return;
  const boxW = Math.sqrt((12 * sigma * sigma) / 3 + 1);
  const radius = Math.max(1, Math.round((boxW - 1) / 2));
  const tmp = new Float32Array(Math.max(w, h));
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < h; y++) boxLine(buf, y * w, 1, w, radius, tmp);
    for (let x = 0; x < w; x++) boxLine(buf, x, w, h, radius, tmp);
  }
}

function boxLine(buf: Float32Array, start: number, stride: number, len: number, r: number, tmp: Float32Array) {
  // Running sum with clamped (replicated) edges.
  const at = (i: number) => buf[start + Math.min(len - 1, Math.max(0, i)) * stride];
  let acc = 0;
  for (let i = -r; i <= r; i++) acc += at(i);
  const norm = 1 / (2 * r + 1);
  for (let i = 0; i < len; i++) {
    tmp[i] = acc * norm;
    acc += at(i + r + 1) - at(i - r);
  }
  for (let i = 0; i < len; i++) buf[start + i * stride] = tmp[i];
}
