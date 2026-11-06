/**
 * Camera RAW import for the browser.
 *
 * Demosaicing a 24-100 MP sensor dump in JavaScript is far too slow for a filmstrip, but
 * every camera RAW also carries the camera's own rendering as an embedded JPEG (usually
 * full size). This module finds that JPEG by walking the container structure, reading only
 * the few kilobytes of headers it needs through `Blob.slice`, and returns it as a lazy slice
 * of the original file: no copy, no decode, typically a few milliseconds per file. The
 * viewport and WebGL preview then work on that JPEG; full 16-bit development happens on the
 * AI service (services/ai/app/raw_engine.py) when exporting through Server or Cloud.
 *
 * Containers: TIFF family (CR2, NEF, ARW, DNG, ORF, RW2, PEF), Canon CR3 (ISOBMFF) and
 * Fujifilm RAF. Canon has not published a ".cr4" format; the extension is accepted and the
 * file is handled if its content is one of the containers above.
 *
 * No DOM APIs at module level: this file is shared with server routes and unit tests.
 */

export type RawFormat = 'cr2' | 'cr3' | 'nef' | 'arw' | 'raf' | 'dng' | 'orf' | 'rw2' | 'pef';

/** Extension -> the container we expect inside (null: accepted, decided by content). */
export const RAW_EXTENSIONS: Readonly<Record<string, RawFormat | null>> = {
  cr2: 'cr2',
  cr3: 'cr3',
  cr4: null,
  nef: 'nef',
  nrw: 'nef',
  arw: 'arw',
  srf: 'arw',
  sr2: 'arw',
  raf: 'raf',
  dng: 'dng',
  orf: 'orf',
  rw2: 'rw2',
  pef: 'pef',
};

/** Content types used when uploading RAW originals (browsers usually report ""). */
export const RAW_MIME: Readonly<Record<string, string>> = {
  cr2: 'image/x-canon-cr2',
  cr3: 'image/x-canon-cr3',
  nef: 'image/x-nikon-nef',
  nrw: 'image/x-nikon-nrw',
  arw: 'image/x-sony-arw',
  srf: 'image/x-sony-srf',
  sr2: 'image/x-sony-sr2',
  raf: 'image/x-fuji-raf',
  dng: 'image/x-adobe-dng',
  orf: 'image/x-olympus-orf',
  rw2: 'image/x-panasonic-rw2',
  pef: 'image/x-pentax-pef',
};

/** Value for `<input type="file" accept>`: web formats, TIFF and every RAW extension. */
export const IMPORT_ACCEPT = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/avif',
  'image/bmp',
  'image/tiff',
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.avif',
  '.tif',
  '.tiff',
  ...Object.keys(RAW_EXTENSIONS).map((e) => `.${e}`),
].join(',');

export function fileExtension(name: string): string {
  const m = /\.([A-Za-z0-9]+)$/.exec(name);
  return m ? m[1].toLowerCase() : '';
}

export function isRawFileName(name: string): boolean {
  return fileExtension(name) in RAW_EXTENSIONS;
}

/** Content type for uploading `file` to object storage (RAW files get a stable x- type). */
export function uploadContentType(file: { name: string; type: string }): string {
  const ext = fileExtension(file.name);
  if (ext in RAW_EXTENSIONS) return RAW_MIME[ext] ?? 'application/octet-stream';
  return file.type || 'application/octet-stream';
}

export interface RawExif {
  make?: string;
  model?: string;
  lens?: string;
  iso?: number;
  /** Seconds. */
  exposureTime?: number;
  /** "1/250" or "2s". */
  shutter?: string;
  fNumber?: number;
  /** Millimetres. */
  focalLength?: number;
  dateTimeOriginal?: string;
  /** EXIF orientation of the RAW (1..8). */
  orientation: number;
  colorSpace?: string;
}

export interface RawPreview {
  format: RawFormat;
  /** Short badge text: the file's extension in capitals ("CR3", "NEF"). */
  label: string;
  /** The embedded JPEG: a lazy slice of the original file. */
  preview: Blob;
  /** Pixel size of the embedded JPEG as stored (before orientation). */
  width: number;
  height: number;
  /**
   * EXIF orientation still to apply after the browser decodes `preview` with
   * imageOrientation 'from-image' (1 when the JPEG carries its own orientation).
   */
  orientation: number;
  exif: RawExif;
}

export class RawParseError extends Error {
  override name = 'RawParseError';
}

// ------------------------------------------------------------------------------------------
// Byte access
// ------------------------------------------------------------------------------------------

const HEAD_BYTES = 256 * 1024;

/** Random access over a Blob with the first HEAD_BYTES cached (where headers live). */
class ByteSource {
  private constructor(
    readonly blob: Blob,
    private readonly head: Uint8Array,
  ) {}

  static async open(blob: Blob): Promise<ByteSource> {
    return new ByteSource(blob, new Uint8Array(await blob.slice(0, HEAD_BYTES).arrayBuffer()));
  }

  get size(): number {
    return this.blob.size;
  }

  get headBytes(): Uint8Array {
    return this.head;
  }

  async bytes(offset: number, length: number): Promise<Uint8Array> {
    if (offset < 0 || length < 0 || offset + length > this.blob.size) throw new RawParseError('offset outside file');
    if (offset + length <= this.head.length) return this.head.subarray(offset, offset + length);
    return new Uint8Array(await this.blob.slice(offset, offset + length).arrayBuffer());
  }
}

const ascii = (b: Uint8Array, start: number, len: number) => String.fromCharCode(...b.subarray(start, start + len));

// ------------------------------------------------------------------------------------------
// Format detection
// ------------------------------------------------------------------------------------------

const MAKER_FORMATS: Array<[string, RawFormat]> = [
  ['NIKON', 'nef'],
  ['SONY', 'arw'],
  ['PENTAX', 'pef'],
  ['RICOH', 'pef'],
  ['CANON', 'cr2'],
  ['OLYMPUS', 'orf'],
  ['OM DIGITAL', 'orf'],
  ['PANASONIC', 'rw2'],
];

function magicFormat(head: Uint8Array): RawFormat | 'tiff' | null {
  if (head.length >= 15 && ascii(head, 0, 15) === 'FUJIFILMCCD-RAW') return 'raf';
  if (head.length >= 12 && ascii(head, 4, 4) === 'ftyp' && ascii(head, 8, 4) === 'crx ') return 'cr3';
  const m4 = head.length >= 4 ? ascii(head, 0, 4) : '';
  if (m4 === 'IIRO' || m4 === 'IIRS' || m4 === 'MMOR') return 'orf';
  if (m4 === 'IIU\0') return 'rw2';
  if (m4 === 'II*\0' || m4 === 'MM\0*') return head.length >= 10 && ascii(head, 8, 2) === 'CR' ? 'cr2' : 'tiff';
  return null;
}

/**
 * Identify a RAW container from the first bytes of a file (the first 256 KB is plenty).
 * Returns null for anything else, including ordinary TIFF scans.
 */
export function detectRawFormat(head: Uint8Array, name?: string): RawFormat | null {
  const magic = magicFormat(head);
  if (magic !== 'tiff') return magic;
  try {
    const t = new Tiff(head, 0);
    const ifd0 = t.ifdSync(t.firstIfd);
    if (ifd0.has(0xc612)) return 'dng';
    const make = (t.textSync(ifd0, 0x010f) ?? '').trim().toUpperCase();
    for (const [prefix, fmt] of MAKER_FORMATS) if (make.startsWith(prefix)) return fmt;
  } catch {
    /* IFD0 outside the head or malformed: decide by extension */
  }
  const byExt = name ? RAW_EXTENSIONS[fileExtension(name)] : undefined;
  return byExt ?? null;
}

// ------------------------------------------------------------------------------------------
// TIFF / EXIF
// ------------------------------------------------------------------------------------------

const TYPE_SIZE: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4 };

interface IfdEntry {
  type: number;
  count: number;
  /** Absolute offset of the value bytes in the source. */
  at: number;
  /** The 4 inline bytes (values that fit are read from here). */
  inline: Uint8Array;
}
type Ifd = Map<number, IfdEntry>;

/** TIFF structure at `base` within either a byte array (sync) or a ByteSource (async). */
class Tiff {
  readonly le: boolean;
  readonly firstIfd: number;
  constructor(
    private readonly buf: Uint8Array,
    readonly base: number,
    private readonly src?: ByteSource,
  ) {
    const order = ascii(buf, base, 2);
    if (order !== 'II' && order !== 'MM') throw new RawParseError('not a TIFF structure');
    this.le = order === 'II';
    this.firstIfd = this.u32(buf, base + 4);
  }

  u16(b: Uint8Array, off: number): number {
    if (off < 0 || off + 2 > b.length) throw new RawParseError('short read');
    return this.le ? b[off] | (b[off + 1] << 8) : (b[off] << 8) | b[off + 1];
  }

  u32(b: Uint8Array, off: number): number {
    if (off < 0 || off + 4 > b.length) throw new RawParseError('short read');
    return new DataView(b.buffer, b.byteOffset + off, 4).getUint32(0, this.le);
  }

  private parseEntries(b: Uint8Array, at: number, n: number, entriesStart: number): Ifd {
    const out: Ifd = new Map();
    for (let i = 0; i < n; i++) {
      const e = at + i * 12;
      const tag = this.u16(b, e);
      const type = this.u16(b, e + 2);
      const count = this.u32(b, e + 4);
      const size = (TYPE_SIZE[type] ?? 1) * count;
      const valueAt = size <= 4 ? entriesStart + i * 12 + 8 : this.base + this.u32(b, e + 8);
      out.set(tag, { type, count, at: valueAt, inline: b.subarray(e + 8, e + 12) });
    }
    return out;
  }

  ifdSync(rel: number): Ifd {
    const off = this.base + rel;
    const n = this.u16(this.buf, off);
    if (n > 1024) throw new RawParseError('implausible IFD');
    return this.parseEntries(this.buf, off + 2, n, off + 2);
  }

  async ifd(rel: number): Promise<{ ifd: Ifd; next: number }> {
    if (!this.src) {
      const ifd = this.ifdSync(rel);
      const off = this.base + rel;
      return { ifd, next: this.u32(this.buf, off + 2 + this.u16(this.buf, off) * 12) };
    }
    const off = this.base + rel;
    const n = this.u16(await this.src.bytes(off, 2), 0);
    if (n > 1024) throw new RawParseError('implausible IFD');
    const body = await this.src.bytes(off + 2, n * 12 + 4);
    return { ifd: this.parseEntries(body, 0, n, off + 2), next: this.u32(body, n * 12) };
  }

  private async valueBytes(e: IfdEntry, len: number): Promise<Uint8Array> {
    if (len <= 4) return e.inline.subarray(0, len);
    if (this.src) return this.src.bytes(e.at, len);
    if (e.at + len > this.buf.length) throw new RawParseError('short read');
    return this.buf.subarray(e.at, e.at + len);
  }

  textSync(ifd: Ifd, tag: number): string | undefined {
    const e = ifd.get(tag);
    if (!e || ![1, 2, 7].includes(e.type)) return undefined;
    const len = Math.min(e.count, 256);
    const b = len <= 4 ? e.inline.subarray(0, len) : this.buf.subarray(e.at, e.at + len);
    return cleanText(b);
  }

  async text(ifd: Ifd, tag: number): Promise<string | undefined> {
    const e = ifd.get(tag);
    if (!e || ![1, 2, 7].includes(e.type)) return undefined;
    return cleanText(await this.valueBytes(e, Math.min(e.count, 256)));
  }

  async numbers(ifd: Ifd, tag: number, max = 64): Promise<number[]> {
    const e = ifd.get(tag);
    if (!e) return [];
    const count = Math.min(e.count, max);
    const size = TYPE_SIZE[e.type] ?? 1;
    const b = await this.valueBytes(e, count * size);
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
      const o = i * size;
      switch (e.type) {
        case 1:
        case 7:
          out.push(dv.getUint8(o));
          break;
        case 3:
          out.push(dv.getUint16(o, this.le));
          break;
        case 4:
        case 13:
          out.push(dv.getUint32(o, this.le));
          break;
        case 8:
          out.push(dv.getInt16(o, this.le));
          break;
        case 9:
          out.push(dv.getInt32(o, this.le));
          break;
        case 5: {
          const d = dv.getUint32(o + 4, this.le);
          out.push(d ? dv.getUint32(o, this.le) / d : 0);
          break;
        }
        case 10: {
          const d = dv.getInt32(o + 4, this.le);
          out.push(d ? dv.getInt32(o, this.le) / d : 0);
          break;
        }
        default:
          return out;
      }
    }
    return out;
  }

  async first(ifd: Ifd, tag: number): Promise<number | undefined> {
    return (await this.numbers(ifd, tag, 1))[0];
  }
}

function cleanText(b: Uint8Array): string | undefined {
  const end = b.indexOf(0);
  const s = new TextDecoder().decode(end >= 0 ? b.subarray(0, end) : b).trim();
  return s || undefined;
}

function shutterText(t: number | undefined): string | undefined {
  if (!t) return undefined;
  return t < 0.5 ? `1/${Math.round(1 / t)}` : `${+t.toFixed(1)}s`;
}

const COLOR_SPACES: Record<number, string> = { 1: 'sRGB', 2: 'Adobe RGB', 0xffff: 'Uncalibrated' };

async function readExifInto(exif: RawExif, t: Tiff, ifd0Rel: number | null, exifRel: number | null = null): Promise<void> {
  if (ifd0Rel !== null) {
    const { ifd } = await t.ifd(ifd0Rel);
    exif.make ??= await t.text(ifd, 0x010f);
    exif.model ??= await t.text(ifd, 0x0110);
    const o = await t.first(ifd, 0x0112);
    if (o && o >= 1 && o <= 8 && exif.orientation === 1) exif.orientation = o;
    if (exifRel === null && ifd.has(0x8769)) exifRel = (await t.first(ifd, 0x8769)) ?? null;
  }
  if (!exifRel) return;
  const { ifd: ex } = await t.ifd(exifRel);
  exif.exposureTime ??= await t.first(ex, 0x829a);
  exif.fNumber ??= await t.first(ex, 0x829d);
  exif.iso ??= (await t.first(ex, 0x8827)) ?? (await t.first(ex, 0x8833));
  exif.dateTimeOriginal ??= await t.text(ex, 0x9003);
  exif.focalLength ??= await t.first(ex, 0x920a);
  exif.lens ??= await t.text(ex, 0xa434);
  const cs = await t.first(ex, 0xa001);
  if (cs !== undefined) exif.colorSpace ??= COLOR_SPACES[cs];
}

// ------------------------------------------------------------------------------------------
// JPEG candidates
// ------------------------------------------------------------------------------------------

interface JpegInfo {
  width: number;
  height: number;
  /** EXIF orientation inside the JPEG's own APP1, if any. */
  ownOrientation: number | null;
}

/**
 * Parse JPEG markers up to the frame header. Returns null for lossless/hierarchical JPEG
 * (CR2 and DNG store their sensor data that way, and browsers cannot decode it).
 */
export function readJpegInfo(b: Uint8Array): JpegInfo | null {
  if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return null;
  let off = 2;
  let ownOrientation: number | null = null;
  for (let guard = 0; guard < 64 && off + 4 <= b.length; guard++) {
    if (b[off] !== 0xff) return null;
    const marker = b[off + 1];
    if (marker === 0xff) {
      off++;
      continue;
    }
    const len = (b[off + 2] << 8) | b[off + 3];
    if (marker === 0xe1 && ascii(b, off + 4, 6) === 'Exif\0\0') {
      try {
        const t = new Tiff(b, off + 10);
        const o = t.ifdSync(t.firstIfd).get(0x0112);
        if (o) ownOrientation = t.u16(o.inline, 0);
      } catch {
        /* unreadable APP1: ignore */
      }
    }
    if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2) {
      if (off + 9 > b.length) return null;
      const height = (b[off + 5] << 8) | b[off + 6];
      const width = (b[off + 7] << 8) | b[off + 8];
      return width && height ? { width, height, ownOrientation } : null;
    }
    if ((marker >= 0xc3 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) || marker === 0xda || marker === 0xd9) return null;
    off += 2 + len;
  }
  return null;
}

interface Candidate {
  offset: number;
  length: number;
}

/** Bytes read from the start of each candidate to find its frame header. */
const PROBE_BYTES = 128 * 1024;

async function probe(src: ByteSource, c: Candidate): Promise<(Candidate & JpegInfo) | null> {
  if (c.offset <= 0 || c.length < 64 || c.offset + c.length > src.size) return null;
  try {
    const info = readJpegInfo(await src.bytes(c.offset, Math.min(c.length, PROBE_BYTES)));
    return info ? { ...c, ...info } : null;
  } catch {
    return null;
  }
}

async function tiffCandidates(src: ByteSource, t: Tiff): Promise<Candidate[]> {
  const out: Candidate[] = [];
  const seen = new Set<number>();
  const queue: number[] = [t.firstIfd];
  while (queue.length && seen.size < 32) {
    const rel = queue.shift()!;
    if (!rel || seen.has(rel)) continue;
    seen.add(rel);
    let parsed: { ifd: Ifd; next: number };
    try {
      parsed = await t.ifd(rel);
    } catch {
      continue;
    }
    const { ifd, next } = parsed;
    const jOff = await t.first(ifd, 0x0201);
    const jLen = await t.first(ifd, 0x0202);
    if (jOff && jLen) out.push({ offset: t.base + jOff, length: jLen });
    const compression = await t.first(ifd, 0x0103);
    if (compression === 6 || compression === 7) {
      const offs = await t.numbers(ifd, 0x0111, 2);
      const lens = await t.numbers(ifd, 0x0117, 2);
      if (offs.length === 1 && lens.length === 1) out.push({ offset: t.base + offs[0], length: lens[0] });
    }
    const jpgFromRaw = ifd.get(0x002e); // Panasonic RW2
    if (jpgFromRaw && jpgFromRaw.type === 7) out.push({ offset: jpgFromRaw.at, length: jpgFromRaw.count });
    for (const sub of await t.numbers(ifd, 0x014a, 16)) queue.push(sub);
    if (next) queue.push(next);
  }
  return out;
}

interface Box {
  type: string;
  start: number; // payload start
  end: number;
}

async function boxes(src: ByteSource, start: number, end: number): Promise<Box[]> {
  const out: Box[] = [];
  let off = start;
  while (off + 8 <= end && out.length < 256) {
    const h = await src.bytes(off, Math.min(16, end - off));
    const dv = new DataView(h.buffer, h.byteOffset, h.byteLength);
    let size = dv.getUint32(0);
    let hdr = 8;
    if (size === 1) {
      if (h.length < 16) break;
      size = Number(dv.getBigUint64(8));
      hdr = 16;
    } else if (size === 0) size = end - off;
    if (size < hdr || off + size > end) break;
    out.push({ type: ascii(h, 4, 4), start: off + hdr, end: off + size });
    off += size;
  }
  return out;
}

const CANON_UUID = '85c0b687820f11e08111f4ce462b6a48';
const CANON_PREVIEW_UUID = 'eaf42b5e1c984b88b9fbb7dc406e4d16';
const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');

async function cr3(src: ByteSource, exif: RawExif): Promise<Candidate[]> {
  const out: Candidate[] = [];
  for (const top of await boxes(src, 0, src.size)) {
    if (top.type === 'uuid' && hex(await src.bytes(top.start, 16)) === CANON_PREVIEW_UUID) {
      // PRVW: small (~1620 px) JPEG; locate its SOI inside the box.
      const body = await src.bytes(top.start, Math.min(top.end - top.start, 256));
      const soi = findSoi(body);
      if (soi >= 0) out.push({ offset: top.start + soi, length: top.end - top.start - soi });
    }
    if (top.type !== 'moov') continue;
    let firstTrak = true;
    for (const b of await boxes(src, top.start, top.end)) {
      if (b.type === 'uuid' && hex(await src.bytes(b.start, 16)) === CANON_UUID) {
        for (const c of await boxes(src, b.start + 16, b.end)) {
          if (c.type !== 'CMT1' && c.type !== 'CMT2') continue;
          const head = await src.bytes(c.start, Math.min(c.end - c.start, 64 * 1024));
          const t = new Tiff(head, 0);
          if (c.type === 'CMT1') await readExifInto(exif, t, t.firstIfd);
          else await readExifInto(exif, t, null, t.firstIfd);
        }
      } else if (b.type === 'trak' && firstTrak) {
        // Track 1 of a CR3 is the full-size JPEG: one sample, located via stco/co64 + stsz.
        firstTrak = false;
        const c = await cr3TrackJpeg(src, b);
        if (c) out.push(c);
      }
    }
  }
  return out;
}

async function cr3TrackJpeg(src: ByteSource, trak: Box): Promise<Candidate | null> {
  const find = async (parent: Box, type: string) => (await boxes(src, parent.start, parent.end)).find((b) => b.type === type);
  const mdia = await find(trak, 'mdia');
  const minf = mdia && (await find(mdia, 'minf'));
  const stbl = minf && (await find(minf, 'stbl'));
  if (!stbl) return null;
  const kids = await boxes(src, stbl.start, stbl.end);
  const stsz = kids.find((b) => b.type === 'stsz');
  const co = kids.find((b) => b.type === 'co64' || b.type === 'stco');
  if (!stsz || !co) return null;
  const sz = await src.bytes(stsz.start, 16);
  const szv = new DataView(sz.buffer, sz.byteOffset, 16);
  const length = szv.getUint32(4) || szv.getUint32(12); // fixed size, else first entry
  const cb = await src.bytes(co.start, 16);
  const cv = new DataView(cb.buffer, cb.byteOffset, 16);
  const offset = co.type === 'co64' ? Number(cv.getBigUint64(8)) : cv.getUint32(8);
  return { offset, length };
}

function findSoi(b: Uint8Array, from = 0): number {
  for (let i = from; i + 2 < b.length; i++) if (b[i] === 0xff && b[i + 1] === 0xd8 && b[i + 2] === 0xff) return i;
  return -1;
}

/** Last resort: scan the whole file for embedded JPEGs and keep the largest. */
async function scanCandidates(src: ByteSource): Promise<Candidate[]> {
  const all = await src.bytes(0, src.size);
  const out: Candidate[] = [];
  for (let i = findSoi(all); i >= 0 && out.length < 256; i = findSoi(all, i + 3)) out.push({ offset: i, length: src.size - i });
  return out;
}

// ------------------------------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------------------------------

/**
 * Find the embedded preview and EXIF of a camera RAW. Throws RawParseError for files that
 * are not a supported RAW or carry no browser-decodable preview.
 */
export async function parseRawFile(file: Blob & { name?: string }): Promise<RawPreview> {
  const src = await ByteSource.open(file);
  const name = file.name ?? '';
  const format = detectRawFormat(src.headBytes, name);
  if (!format) {
    const ext = fileExtension(name);
    throw new RawParseError(
      ext === 'cr4' ? 'This .cr4 file is not a RAW format FineRetouch can read (Canon has no published CR4 format).' : `${name || 'File'} is not a supported camera RAW file.`,
    );
  }
  const exif: RawExif = { orientation: 1 };
  let candidates: Candidate[] = [];
  try {
    if (format === 'cr3') {
      candidates = await cr3(src, exif);
    } else if (format === 'raf') {
      const h = src.headBytes;
      const dv = new DataView(h.buffer, h.byteOffset, h.byteLength);
      if (h.length >= 92) candidates.push({ offset: dv.getUint32(84), length: dv.getUint32(88) });
    } else {
      const t = new Tiff(src.headBytes, 0, src);
      await readExifInto(exif, t, t.firstIfd).catch(() => undefined);
      candidates = await tiffCandidates(src, t);
    }
  } catch {
    /* damaged structure: fall through to the scan */
  }

  let best: (Candidate & JpegInfo) | null = null;
  const consider = async (list: Candidate[]) => {
    for (const c of list) {
      const p = await probe(src, c);
      if (p && (!best || p.width * p.height > best.width * best.height)) best = p;
    }
  };
  await consider(candidates);
  if (!best) await consider(await scanCandidates(src));
  const chosen = best as (Candidate & JpegInfo) | null;
  if (!chosen) throw new RawParseError(`No embedded preview found in ${name || 'this RAW file'}.`);

  const preview = file.slice(chosen.offset, chosen.offset + chosen.length, 'image/jpeg');
  // Fill any EXIF gaps (RAF keeps everything in the preview's own APP1).
  if (!exif.make || !exif.iso) {
    try {
      const head = await src.bytes(chosen.offset, Math.min(chosen.length, PROBE_BYTES));
      const app1 = findApp1(head);
      if (app1 >= 0) {
        const t = new Tiff(head, app1);
        await readExifInto(exif, t, t.firstIfd);
      }
    } catch {
      /* keep what we have */
    }
  }
  exif.shutter = shutterText(exif.exposureTime);
  const own = chosen.ownOrientation;
  return {
    format,
    label: (fileExtension(name) || format).toUpperCase(),
    preview,
    width: chosen.width,
    height: chosen.height,
    // Browsers apply a JPEG's own EXIF orientation; otherwise the RAW's orientation applies.
    orientation: own && own !== 1 ? 1 : exif.orientation,
    exif,
  };
}

function findApp1(b: Uint8Array): number {
  let off = 2;
  for (let guard = 0; guard < 32 && off + 10 <= b.length && b[off] === 0xff; guard++) {
    const len = (b[off + 2] << 8) | b[off + 3];
    if (b[off + 1] === 0xe1 && ascii(b, off + 4, 6) === 'Exif\0\0') return off + 10;
    if (b[off + 1] === 0xda) break;
    off += 2 + len;
  }
  return -1;
}

/** `parseRawFile` plus an object URL of the preview (revoke it with URL.revokeObjectURL). */
export async function createRawPreviewUrl(file: Blob & { name?: string }): Promise<RawPreview & { url: string }> {
  const raw = await parseRawFile(file);
  return { ...raw, url: URL.createObjectURL(raw.preview) };
}

/** One line for tooltips: "Canon EOS R5 · RF85mm F1.2 · ISO 800 · 1/125 · f/5.6". */
export function describeExif(e: RawExif): string {
  const parts = [
    e.model ?? e.make,
    e.lens,
    e.iso ? `ISO ${e.iso}` : undefined,
    e.shutter,
    e.fNumber ? `f/${+e.fNumber.toFixed(1)}` : undefined,
    e.focalLength ? `${Math.round(e.focalLength)} mm` : undefined,
  ];
  return parts.filter(Boolean).join(' · ');
}
