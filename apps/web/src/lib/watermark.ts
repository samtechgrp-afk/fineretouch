'use client';

/**
 * Watermark stamping for exports: a text line (e.g. "© 2026 Your Name") and/or a logo,
 * burned into the full-resolution pixels just before encoding.
 *
 * Every size is relative to the image, never in pixels: the text height is a share of the
 * short edge and the margin a share of the short edge too, so a 1080p proof and a 45 MP
 * print carry a watermark that looks identical. The export modal's preview runs the same
 * function on a small canvas, so what you see is what lands in the file.
 */

export type WatermarkAnchor =
  | 'top-left' | 'top-center' | 'top-right'
  | 'middle-left' | 'center' | 'middle-right'
  | 'bottom-left' | 'bottom-center' | 'bottom-right';

export const WATERMARK_ANCHORS: readonly WatermarkAnchor[] = [
  'top-left', 'top-center', 'top-right',
  'middle-left', 'center', 'middle-right',
  'bottom-left', 'bottom-center', 'bottom-right',
];

/** Fonts every desktop OS ships (the app is offline, so no web fonts). First match wins. */
export const WATERMARK_FONTS: ReadonlyArray<{ id: string; label: string; stack: string }> = [
  { id: 'sans', label: 'Sans', stack: '"Segoe UI", "Helvetica Neue", Helvetica, Arial, "DejaVu Sans", sans-serif' },
  { id: 'serif', label: 'Serif', stack: 'Georgia, "Times New Roman", Times, "DejaVu Serif", serif' },
  { id: 'mono', label: 'Mono', stack: '"SF Mono", Consolas, Menlo, "DejaVu Sans Mono", monospace' },
  { id: 'script', label: 'Script', stack: '"Segoe Script", "Brush Script MT", "Apple Chancery", cursive' },
  { id: 'condensed', label: 'Condensed', stack: '"Arial Narrow", "Helvetica Neue Condensed", "DejaVu Sans Condensed", sans-serif-condensed, sans-serif' },
];

export interface WatermarkSettings {
  enabled: boolean;
  text: string;
  /** Id from WATERMARK_FONTS. */
  font: string;
  bold: boolean;
  /** Text cap height as a percentage of the image's short edge (0.5..15). */
  size: number;
  /** 0..100 */
  opacity: number;
  /** #rrggbb */
  color: string;
  anchor: WatermarkAnchor;
  /** Distance from the edges, percentage of the short edge (0..20). */
  margin: number;
  /** Soft shadow so light text stays readable on bright areas. */
  shadow: boolean;
  /** Optional logo as a PNG data: URL (transparency kept), drawn before the text. */
  logo: string | null;
  /** Logo height as a percentage of the short edge (1..40). */
  logoSize: number;
}

export const DEFAULT_WATERMARK: Readonly<WatermarkSettings> = Object.freeze({
  enabled: false,
  text: `© ${new Date().getFullYear()} Your Name`,
  font: 'sans',
  bold: false,
  size: 3,
  opacity: 70,
  color: '#ffffff',
  anchor: 'bottom-right',
  margin: 3,
  shadow: true,
  logo: null,
  logoSize: 8,
});

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const num = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? clamp(v, lo, hi) : d);

/** Coerce stored or imported JSON into valid settings. */
export function sanitizeWatermark(input: unknown): WatermarkSettings {
  const r = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>;
  const d = DEFAULT_WATERMARK;
  return {
    enabled: typeof r.enabled === 'boolean' ? r.enabled : d.enabled,
    text: typeof r.text === 'string' ? r.text.slice(0, 200) : d.text,
    font: WATERMARK_FONTS.some((f) => f.id === r.font) ? (r.font as string) : d.font,
    bold: typeof r.bold === 'boolean' ? r.bold : d.bold,
    size: num(r.size, 0.5, 15, d.size),
    opacity: num(r.opacity, 0, 100, d.opacity),
    color: typeof r.color === 'string' && /^#[0-9a-f]{6}$/i.test(r.color) ? r.color.toLowerCase() : d.color,
    anchor: WATERMARK_ANCHORS.includes(r.anchor as WatermarkAnchor) ? (r.anchor as WatermarkAnchor) : d.anchor,
    margin: num(r.margin, 0, 20, d.margin),
    shadow: typeof r.shadow === 'boolean' ? r.shadow : d.shadow,
    logo: typeof r.logo === 'string' && /^data:image\/(png|jpeg|webp);base64,/.test(r.logo) ? r.logo : null,
    logoSize: num(r.logoSize, 1, 40, d.logoSize),
  };
}

export function watermarkActive(w: WatermarkSettings): boolean {
  return w.enabled && w.opacity > 0 && (w.text.trim().length > 0 || w.logo !== null);
}

/** Pixel geometry of the watermark for an image of width x height. */
export interface WatermarkLayout {
  fontPx: number;
  marginPx: number;
  logoPx: number;
  /** Gap between logo and text. */
  gapPx: number;
}

export function watermarkLayout(w: WatermarkSettings, width: number, height: number): WatermarkLayout {
  const short = Math.min(width, height);
  const fontPx = Math.max(6, (w.size / 100) * short);
  return {
    fontPx,
    marginPx: (w.margin / 100) * short,
    logoPx: Math.max(4, (w.logoSize / 100) * short),
    gapPx: fontPx * 0.5,
  };
}

const logoCache = new Map<string, Promise<ImageBitmap>>();

async function logoBitmap(dataUrl: string): Promise<ImageBitmap> {
  let p = logoCache.get(dataUrl);
  if (!p) {
    p = fetch(dataUrl)
      .then((r) => r.blob())
      .then((b) => createImageBitmap(b));
    logoCache.set(dataUrl, p);
    p.catch(() => logoCache.delete(dataUrl));
  }
  return p;
}

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/**
 * Draw the watermark onto a 2D context whose canvas is exactly the image (width x height).
 * The logo (if any) sits before the text on the same baseline row; the group is aligned
 * to the anchor on the 3x3 grid.
 */
export async function drawWatermark(ctx: Ctx2D, w: WatermarkSettings, width: number, height: number): Promise<void> {
  if (!watermarkActive(w)) return;
  const L = watermarkLayout(w, width, height);
  const stack = WATERMARK_FONTS.find((f) => f.id === w.font)?.stack ?? WATERMARK_FONTS[0].stack;
  const text = w.text.trim();
  const logo = w.logo ? await logoBitmap(w.logo).catch(() => null) : null;

  ctx.save();
  ctx.font = `${w.bold ? 700 : 400} ${L.fontPx}px ${stack}`;
  ctx.textBaseline = 'alphabetic';
  const m = text ? ctx.measureText(text) : null;
  const ascent = m ? (m.actualBoundingBoxAscent || L.fontPx * 0.75) : 0;
  const descent = m ? (m.actualBoundingBoxDescent || L.fontPx * 0.2) : 0;
  const textW = m ? m.width : 0;
  const textH = ascent + descent;
  const logoH = logo ? L.logoPx : 0;
  const logoW = logo ? (logo.width / Math.max(1, logo.height)) * logoH : 0;
  const groupW = logoW + (logo && text ? L.gapPx : 0) + textW;
  const groupH = Math.max(logoH, textH);

  const [v, hz] = anchorParts(w.anchor);
  const x = hz === 'left' ? L.marginPx : hz === 'right' ? width - L.marginPx - groupW : (width - groupW) / 2;
  const y = v === 'top' ? L.marginPx : v === 'bottom' ? height - L.marginPx - groupH : (height - groupH) / 2;

  ctx.globalAlpha = clamp(w.opacity, 0, 100) / 100;
  if (w.shadow) {
    ctx.shadowColor = isLight(w.color) ? 'rgba(0,0,0,0.45)' : 'rgba(255,255,255,0.35)';
    ctx.shadowBlur = L.fontPx * 0.18;
    ctx.shadowOffsetY = L.fontPx * 0.04;
  }
  if (logo) ctx.drawImage(logo, x, y + (groupH - logoH) / 2, logoW, logoH);
  if (text) {
    ctx.fillStyle = w.color;
    ctx.fillText(text, x + logoW + (logo ? L.gapPx : 0), y + (groupH - textH) / 2 + ascent);
  }
  ctx.restore();
}

function anchorParts(a: WatermarkAnchor): ['top' | 'middle' | 'bottom', 'left' | 'center' | 'right'] {
  if (a === 'center') return ['middle', 'center'];
  const [v, h] = a.split('-') as ['top' | 'middle' | 'bottom', 'left' | 'center' | 'right'];
  return [v, h];
}

function isLight(hex: string): boolean {
  const n = parseInt(hex.slice(1), 16);
  return ((n >> 16) & 255) * 0.299 + ((n >> 8) & 255) * 0.587 + (n & 255) * 0.114 > 140;
}

/**
 * Burn the watermark into an already encoded image (e.g. a render from the AI engine) and
 * re-encode it. Callers that still hold pixels should use drawWatermark before encoding instead.
 */
export async function stampBlob(blob: Blob, w: WatermarkSettings, type: string, quality: number): Promise<Blob> {
  if (!watermarkActive(w)) return blob;
  const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  try {
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(bmp, 0, 0);
    await drawWatermark(ctx, w, bmp.width, bmp.height);
    return await canvas.convertToBlob({ type, quality: quality / 100 });
  } finally {
    bmp.close();
  }
}

/** Read a picked logo file as a data URL (kept small: logos are stored with the export settings). */
export async function logoFromFile(file: File, maxEdge = 1024): Promise<string> {
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) throw new Error('Use a PNG, JPEG or WebP logo (PNG keeps transparency)');
  const bmp = await createImageBitmap(file);
  try {
    const s = Math.min(1, maxEdge / Math.max(bmp.width, bmp.height));
    const c = new OffscreenCanvas(Math.max(1, Math.round(bmp.width * s)), Math.max(1, Math.round(bmp.height * s)));
    c.getContext('2d')!.drawImage(bmp, 0, 0, c.width, c.height);
    const png = await c.convertToBlob({ type: 'image/png' });
    return await new Promise<string>((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result as string);
      r.onerror = () => reject(r.error);
      r.readAsDataURL(png);
    });
  } finally {
    bmp.close();
  }
}
