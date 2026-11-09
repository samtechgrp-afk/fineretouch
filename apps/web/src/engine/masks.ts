import type { FaceAnalysis, MaskStroke, Point } from '@fineretouch/shared';

/**
 * Rasterise the AI service's normalised face regions into one RGBA mask
 * (r = face skin, g = eyes, b = under-eye, a = mouth), feathered with the same
 * relative radii as services/ai/app/pipeline.py:build_masks.
 *
 * Pure CPU (scanline fill + 3x box blur) so it runs in workers, tests and
 * browsers without Canvas `filter` support.
 */
/** Face parsing from the engine (POST /v1/parse) at the mask's size: RGBA, r face skin, g body skin, b hair. */
export interface FaceParsePixels {
  data: Uint8Array | Uint8ClampedArray;
  /** Add neck / chest / arm skin (SkinMaskSettings.includeBodySkin). */
  includeBody: boolean;
}

/** Semantic face skin only counts within this many face widths of a face (masking.FACE_REACH). */
const FACE_REACH = 1.0;
/** Hair probability ramp that removes skin (masking.HAIR_CUT). */
const HAIR_CUT: [number, number] = [0.35, 0.75];

export function buildFaceMask(faces: FaceAnalysis[], width: number, height: number, parse?: FaceParsePixels | null): Uint8Array {
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
    if (parse && parse.data.length === n * 4) applyParse(skin, faces, width, height, faceW, parse);
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
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};

/**
 * Port of the parse branch of pipeline.build_masks: semantic face skin near each face (and body
 * skin when asked) extends the landmark mask, eyes / brows / lips stay cut out, and hair lying
 * across the face removes skin.
 */
function applyParse(skin: Float32Array, faces: FaceAnalysis[], w: number, h: number, faceW: number, parse: FaceParsePixels) {
  const n = w * h;
  const px = (pts: Point[]) => pts.map(([x, y]) => [x * w, y * h] as [number, number]);
  const features = new Float32Array(n);
  const zone = new Float32Array(n);
  for (const f of faces) {
    const r = f.regions;
    for (const region of [r.leftEye, r.rightEye, r.leftBrow, r.rightBrow, r.lips]) {
      if (region.length < 3) continue;
      const poly = px(region);
      fillPolygon(features, w, h, poly, 1);
      strokePolygon(features, w, h, poly, Math.max(1, faceW * 0.03), 1); // dilate like cv2.dilate
    }
    const [bx, by, bw, bh] = f.box;
    const cx = (bx + bw / 2) * w;
    const cy = (by + bh / 2) * h;
    const ax = Math.max(1, FACE_REACH * bw * w);
    const ay = Math.max(1, FACE_REACH * bh * h * 1.1);
    const ell: [number, number][] = Array.from({ length: 64 }, (_, i) => {
      const t = (i / 64) * Math.PI * 2;
      return [cx + Math.cos(t) * ax, cy + Math.sin(t) * ay];
    });
    fillPolygon(zone, w, h, ell, 1);
  }
  gaussianApprox(features, w, h, faceW * 0.006);
  const face = new Float32Array(n);
  const body = new Float32Array(n);
  const hair = new Float32Array(n);
  const d = parse.data;
  for (let i = 0; i < n; i++) {
    face[i] = (d[i * 4] / 255) * zone[i];
    body[i] = parse.includeBody ? d[i * 4 + 1] / 255 : 0;
    hair[i] = d[i * 4 + 2] / 255;
  }
  const s = faceW * 0.004;
  gaussianApprox(face, w, h, s);
  if (parse.includeBody) gaussianApprox(body, w, h, s);
  gaussianApprox(hair, w, h, s);
  for (let i = 0; i < n; i++) {
    const cut = 1 - clamp01(features[i]);
    const v = Math.max(skin[i], face[i] * cut, body[i] * cut);
    skin[i] = v * (1 - smoothstep(HAIR_CUT[0], HAIR_CUT[1], hair[i]));
  }
}

/** Stroke feather at feather = 100, as a share of the stroke radius (masking.STROKE_FEATHER_MAX). */
const STROKE_FEATHER_MAX = 1.0;

/**
 * User brush strokes as (add, erase) masks, port of masking.rasterize_strokes: size is a radius as
 * a share of the long edge, points are normalised; each kind is feathered by up to its radius.
 */
export function rasterizeMaskStrokes(strokes: readonly MaskStroke[], w: number, h: number, feather: number): { add: Float32Array; erase: Float32Array } {
  const add = new Float32Array(w * h);
  const erase = new Float32Array(w * h);
  const L = Math.max(w, h);
  const f = clamp01(feather / 100) * STROKE_FEATHER_MAX;
  for (const [kind, canvas] of [['add', add], ['erase', erase]] as const) {
    let sigma = 0;
    for (const s of strokes) {
      if (s.mode !== kind) continue;
      const r = Math.max(0.5, s.size * L);
      const pts: [number, number][] = [];
      for (let i = 0; i + 1 < s.points.length; i += 2) pts.push([s.points[i] * w, s.points[i + 1] * h]);
      if (!pts.length) continue;
      if (pts.length === 1) stampDisc(canvas, w, h, pts[0][0], pts[0][1], r);
      for (let i = 1; i < pts.length; i++) stampCapsule(canvas, w, h, pts[i - 1], pts[i], r);
      sigma = Math.max(sigma, f * r * 0.5);
    }
    if (sigma >= 0.5) gaussianApprox(canvas, w, h, sigma);
  }
  for (let i = 0; i < add.length; i++) {
    add[i] = clamp01(add[i]);
    erase[i] = clamp01(erase[i]);
  }
  return { add, erase };
}

/** RGBA texture of the strokes for Renderer.setEdit (r = add, g = erase), or null without strokes. */
export function buildEditMask(strokes: readonly MaskStroke[], w: number, h: number, feather: number): Uint8Array | null {
  if (!strokes.length) return null;
  const { add, erase } = rasterizeMaskStrokes(strokes, w, h, feather);
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < add.length; i++) {
    out[i * 4] = Math.round(add[i] * 255);
    out[i * 4 + 1] = Math.round(erase[i] * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

function stampDisc(buf: Float32Array, w: number, h: number, cx: number, cy: number, r: number) {
  stampCapsule(buf, w, h, [cx, cy], [cx, cy], r);
}

/** Anti-aliased capsule (segment of radius r), max-composited at pixel centres. */
function stampCapsule(buf: Float32Array, w: number, h: number, a: [number, number], b: [number, number], r: number) {
  const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - r - 1));
  const x1 = Math.min(w - 1, Math.ceil(Math.max(a[0], b[0]) + r + 1));
  const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - r - 1));
  const y1 = Math.min(h - 1, Math.ceil(Math.max(a[1], b[1]) + r + 1));
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  for (let y = y0; y <= y1; y++) {
    const py = y + 0.5;
    for (let x = x0; x <= x1; x++) {
      const pxx = x + 0.5;
      const t = len2 > 0 ? clamp01(((pxx - a[0]) * dx + (py - a[1]) * dy) / len2) : 0;
      const ex = pxx - (a[0] + t * dx);
      const ey = py - (a[1] + t * dy);
      const cov = clamp01(r + 0.5 - Math.sqrt(ex * ex + ey * ey));
      const i = y * w + x;
      if (cov > buf[i]) buf[i] = cov;
    }
  }
}

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

/** Scale a pixel polygon about its centroid; points above the centroid use `syUp`, below `syDown`. */
function scalePolygon(pts: [number, number][], sx: number, syUp: number, syDown: number): [number, number][] {
  const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
  const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  return pts.map(([x, y]) => [cx + (x - cx) * sx, cy + (y - cy) * (y < cy ? syUp : syDown)]);
}

/**
 * Where stray-hair cleanup may act: a band around the face oval and forehead, minus eyes,
 * brows and lips, and nothing below the mouth. Port of stray_hair_roi in
 * services/ai/app/retouch_engine.py (feature exclusion is approximated by enlarged polygons).
 * Returns a single-channel 0..255 mask.
 */
export function buildHairRoiMask(faces: FaceAnalysis[], width: number, height: number): Uint8Array {
  const roi = new Float32Array(width * height);
  const px = (pts: Point[]) => pts.map(([x, y]) => [x * width, y * height] as [number, number]);
  let faceW = 1;
  for (const f of faces) {
    const r = f.regions;
    if (r.faceOval.length < 3) continue;
    faceW = Math.max(faceW, f.box[2] * width);
    const oval = px(r.faceOval);
    const band = new Float32Array(width * height);
    fillPolygon(band, width, height, scalePolygon(oval, 1.28, 1.45, 1.05), 1);
    fillPolygon(band, width, height, scalePolygon(oval, 0.78, 0.55, 0.8), 0);
    for (const feat of [r.leftEye, r.rightEye, r.leftBrow, r.rightBrow, r.lips]) {
      if (feat.length >= 3) fillPolygon(band, width, height, scalePolygon(px(feat), 1.6, 2.2, 2.2), 0);
    }
    const mouthY = (r.lips.length ? r.lips.reduce((a, p) => a + p[1], 0) / r.lips.length : f.box[1] + f.box[3] * 0.75) * height;
    for (let y = Math.max(0, Math.floor(mouthY)); y < height; y++) band.fill(0, y * width, (y + 1) * width);
    for (let i = 0; i < roi.length; i++) roi[i] = Math.max(roi[i], band[i]);
  }
  gaussianApprox(roi, width, height, faceW * 0.01);
  return Uint8Array.from(roi, (v) => Math.round(clamp01(v) * 255));
}

/**
 * Client-side depth fallback (0 = subject, 255 = far) when the AI service cannot provide
 * a depth map: a head ellipse plus shoulders from the face boxes, as portrait_prior in
 * services/ai/app/depth.py, or a centre-weighted falloff when there is no face.
 */
export function buildDepthPrior(faces: FaceAnalysis[], width: number, height: number): Uint8Array {
  const person = new Float32Array(width * height);
  for (const f of faces) {
    const [x, y, bw, bh] = f.box;
    const cx = (x + bw / 2) * width;
    const cy = (y + bh / 2) * height;
    const fw = bw * width;
    const fh = bh * height;
    const ell: [number, number][] = Array.from({ length: 48 }, (_, i) => {
      const t = (i / 48) * Math.PI * 2;
      return [cx + Math.cos(t) * fw * 0.75, cy - fh * 0.1 + Math.sin(t) * fh * 0.85];
    });
    fillPolygon(person, width, height, ell, 1);
    const neckY = cy + fh * 0.55;
    fillPolygon(person, width, height, [[cx - fw * 0.45, neckY], [cx + fw * 0.45, neckY], [cx + fw * 1.9, height], [cx - fw * 1.9, height]], 1);
  }
  const out = new Uint8Array(width * height);
  if (!faces.length) {
    for (let yy = 0; yy < height; yy++)
      for (let xx = 0; xx < width; xx++) {
        const d = Math.hypot((xx + 0.5) / width - 0.5, (yy + 0.5) / height - 0.45) / 0.6;
        out[yy * width + xx] = Math.round(clamp01(d) * 255);
      }
    return out;
  }
  gaussianApprox(person, width, height, Math.max(width, height) * 0.01);
  for (let yy = 0; yy < height; yy++) {
    const t = clamp01(((yy + 0.5) / height - 0.6) / 0.4);
    const bg = 1 - 0.35 * t;
    for (let xx = 0; xx < width; xx++) {
      const i = yy * width + xx;
      out[i] = Math.round(clamp01(bg * (1 - person[i])) * 255);
    }
  }
  return out;
}

/**
 * Where eyeglass lenses sit: an ellipse around each eye (0.2 x 0.15 face widths, centred
 * slightly below the eye). Port of glasses_lens_roi in services/ai/app/retouch_engine.py,
 * lightly feathered. Single-channel 0..255.
 */
export function buildLensRoiMask(faces: FaceAnalysis[], width: number, height: number): Uint8Array {
  const roi = new Float32Array(width * height);
  let faceW = 1;
  for (const f of faces) {
    const fw = f.box[2] * width;
    faceW = Math.max(faceW, fw);
    for (const eye of [f.regions.leftEye, f.regions.rightEye]) {
      if (eye.length < 3) continue;
      const cx = (eye.reduce((a, p) => a + p[0], 0) / eye.length) * width;
      const cy = (eye.reduce((a, p) => a + p[1], 0) / eye.length) * height + 0.02 * fw;
      const ellipse: [number, number][] = Array.from({ length: 48 }, (_, i) => {
        const t = (i / 48) * Math.PI * 2;
        return [cx + 0.2 * fw * Math.cos(t), cy + 0.15 * fw * Math.sin(t)];
      });
      fillPolygon(roi, width, height, ellipse, 1);
    }
  }
  gaussianApprox(roi, width, height, faceW * 0.01);
  return Uint8Array.from(roi, (v) => Math.round(clamp01(v) * 255));
}
