/**
 * 3D look LUTs: the wire format shared with services/ai (color_engine.decode_lut /
 * encode_lut, batch_engine.parse_cube / lut_to_cube), .cube import and export, and batch
 * sync helpers.
 *
 * Layout everywhere: size^3 RGB triples, red index fastest, then green, then blue (the
 * .cube order). That is also the memory layout a WebGL2 3D texture expects with
 * x = red, y = green, z = blue, so decoded tables upload without reordering.
 */

import { DEFAULT_COLOR_GRADING, DEFAULT_HSL, LUT_MAX_SIZE, clamp, type Adjustments, type ColorStats, type LookLut } from './index.js';

export interface DecodedLut {
  size: number;
  /** size^3 * 3 floats in [0, 1]. */
  table: Float32Array;
  title: string;
}

function toBase64(bytes: Uint8Array): string {
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(bin);
}

function fromBase64(data: string): Uint8Array {
  const bin = atob(data);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Float table (0..1) -> base64 little-endian uint16, as LookLut.data. */
export function encodeLut(table: ArrayLike<number>): string {
  const bytes = new Uint8Array(table.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < table.length; i++) view.setUint16(i * 2, Math.round(clamp(table[i], 0, 1) * 65535), true);
  return toBase64(bytes);
}

export function decodeLut(lut: Pick<LookLut, 'data' | 'size'>): Float32Array {
  const bytes = fromBase64(lut.data);
  const n = lut.size ** 3 * 3;
  if (bytes.length !== n * 2) throw new Error(`LUT data holds ${bytes.length} bytes, expected ${n * 2}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getUint16(i * 2, true) / 65535;
  return out;
}

export function identityTable(size: number): Float32Array {
  const t = new Float32Array(size ** 3 * 3);
  let i = 0;
  for (let b = 0; b < size; b++)
    for (let g = 0; g < size; g++)
      for (let r = 0; r < size; r++) {
        t[i++] = r / (size - 1);
        t[i++] = g / (size - 1);
        t[i++] = b / (size - 1);
      }
  return t;
}

/** Trilinear lookup of one colour (0..1), the same maths as apply_lut and the LUT shader. */
export function sampleLut(table: ArrayLike<number>, size: number, rgb: [number, number, number]): [number, number, number] {
  const n = size;
  const idx = (r: number, g: number, b: number) => ((b * n + g) * n + r) * 3;
  const c = rgb.map((v) => clamp(v, 0, 1) * (n - 1));
  const i0 = c.map((v) => Math.min(Math.floor(v), n - 2));
  const f = c.map((v, k) => v - i0[k]);
  const out: [number, number, number] = [0, 0, 0];
  for (let db = 0; db < 2; db++)
    for (let dg = 0; dg < 2; dg++)
      for (let dr = 0; dr < 2; dr++) {
        const w = (dr ? f[0] : 1 - f[0]) * (dg ? f[1] : 1 - f[1]) * (db ? f[2] : 1 - f[2]);
        const j = idx(i0[0] + dr, i0[1] + dg, i0[2] + db);
        out[0] += table[j] * w;
        out[1] += table[j + 1] * w;
        out[2] += table[j + 2] * w;
      }
  return out;
}

/** Parse a .cube 3D LUT (TITLE, LUT_3D_SIZE, DOMAIN_MIN/MAX; 1D LUTs rejected). */
export function parseCube(text: string): DecodedLut {
  let size = 0;
  let title = 'Imported LUT';
  let lo = [0, 0, 0];
  let hi = [1, 1, 1];
  const values: number[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.split('#', 1)[0].trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    const key = parts[0].toUpperCase();
    if (key === 'TITLE') title = line.slice(5).trim().replace(/^"|"$/g, '') || title;
    else if (key === 'LUT_3D_SIZE') size = Number(parts[1]);
    else if (key === 'LUT_1D_SIZE') throw new Error('1D LUTs are not supported; export a 3D .cube');
    else if (key === 'DOMAIN_MIN') lo = parts.slice(1, 4).map(Number);
    else if (key === 'DOMAIN_MAX') hi = parts.slice(1, 4).map(Number);
    else if (/^[-+.\d]/.test(key)) {
      for (let k = 0; k < 3; k++) values.push(Number(parts[k]));
    }
  }
  if (!Number.isInteger(size) || size < 2 || size > LUT_MAX_SIZE) throw new Error(`LUT_3D_SIZE missing or outside 2..${LUT_MAX_SIZE}`);
  if (values.length !== size ** 3 * 3) throw new Error(`Expected ${size ** 3} LUT entries, found ${values.length / 3}`);
  if (values.some((v) => !Number.isFinite(v))) throw new Error('LUT contains a value that is not a number');
  const table = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const ch = i % 3;
    table[i] = clamp((values[i] - lo[ch]) / Math.max(hi[ch] - lo[ch], 1e-6), 0, 1);
  }
  return { size, table, title };
}

export function lutToCube(table: ArrayLike<number>, size: number, title = 'FineRetouch look'): string {
  const safe = title.replace(/"/g, "'").replace(/\n/g, ' ').slice(0, 120);
  const lines = [`TITLE "${safe}"`, `LUT_3D_SIZE ${size}`, 'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0'];
  for (let i = 0; i < size ** 3; i++) {
    lines.push(`${table[i * 3].toFixed(6)} ${table[i * 3 + 1].toFixed(6)} ${table[i * 3 + 2].toFixed(6)}`);
  }
  return lines.join('\n') + '\n';
}

/** A .cube file as an adjustments look. Tables above LUT_MAX_SIZE are rejected by parseCube. */
export function lookFromCube(text: string, amount = 100): LookLut {
  const { size, table, title } = parseCube(text);
  return { name: title.slice(0, 120), size, data: encodeLut(table), amount: clamp(amount, 0, 100) };
}

const TONE_AND_COLOUR = ['exposure', 'contrast', 'highlights', 'shadows', 'temperature', 'tint', 'saturation', 'vibrance'] as const;

/**
 * Batch "palette" sync: the reference's retouch (skin, face, clothing, detail) with its
 * colour and tone replaced by a statistical transfer to the graded reference, so each
 * photo is matched to how the finished reference looks from wherever its own original
 * starts. Mirror of palette_adjustments in services/ai/app/batch_engine.py.
 */
export function paletteSyncAdjustments(reference: Adjustments, gradedReference: ColorStats, amount = 100): Adjustments {
  const out: Adjustments = { ...reference, hsl: DEFAULT_HSL, colorGrading: DEFAULT_COLOR_GRADING, lut: null };
  for (const k of TONE_AND_COLOUR) out[k] = 0;
  out.colorMatch = { reference: gradedReference, amount: clamp(amount, 0, 100) };
  return out;
}
