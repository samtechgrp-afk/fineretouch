import { describe, expect, it } from 'vitest';
import {
  IMPORT_ACCEPT,
  RawParseError,
  describeExif,
  detectRawFormat,
  isRawFileName,
  parseRawFile,
  readJpegInfo,
  uploadContentType,
} from './rawParser';

// ---------------------------------------------------------------- synthetic files

const enc = (s: string) => new TextEncoder().encode(s);

function concat(...parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

const u16be = (v: number) => new Uint8Array([v >> 8, v & 255]);
const u32be = (v: number) => new Uint8Array([(v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255]);

/** A JPEG skeleton: SOI, optional Exif APP1 with an orientation, a frame header and EOI. */
function fakeJpeg(width: number, height: number, opts: { sof?: number; orientation?: number; make?: string; pad?: number } = {}): Uint8Array {
  const parts: Uint8Array[] = [new Uint8Array([0xff, 0xd8])];
  if (opts.orientation || opts.make) {
    const entries: Entry[] = [];
    if (opts.make) entries.push([0x010f, 2, opts.make]);
    if (opts.orientation) entries.push([0x0112, 3, [opts.orientation]]);
    const tiff = buildTiff({ ifd0: entries }, {});
    parts.push(new Uint8Array([0xff, 0xe1]), u16be(2 + 6 + tiff.length), enc('Exif\0\0'), tiff);
  }
  const sof = opts.sof ?? 0xc0;
  parts.push(new Uint8Array([0xff, sof]), u16be(17), new Uint8Array([8]), u16be(height), u16be(width), new Uint8Array([3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]));
  parts.push(new Uint8Array(opts.pad ?? 32).fill(0x55), new Uint8Array([0xff, 0xd9]));
  return concat(...parts);
}

type Value = number[] | string;
type Entry = [tag: number, type: number, value: Value | `@${string}`];
const SIZE: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1 };

/** Little-endian TIFF; values "@name" resolve to the offset of IFD/blob `name`. */
function buildTiff(ifds: Record<string, Entry[]>, blobs: Record<string, Uint8Array>, magic = 'II*\0'): Uint8Array {
  const count = (type: number, v: Value | string) => (type === 2 ? (v as string).length + 1 : typeof v === 'string' ? 1 : v.length);
  const names = Object.keys(ifds);
  const offsets: Record<string, number> = {};
  let pos = 8;
  for (const n of names) {
    offsets[n] = pos;
    let extra = 0;
    for (const [, type, v] of ifds[n]) {
      const len = count(type, v) * SIZE[type];
      if (len > 4) extra += len + (len & 1);
    }
    pos += 2 + 12 * ifds[n].length + 4 + extra;
  }
  for (const [n, b] of Object.entries(blobs)) {
    offsets[n] = pos;
    pos += b.length + (b.length & 1);
  }
  const out = new Uint8Array(pos);
  const dv = new DataView(out.buffer);
  out.set(enc(magic), 0);
  dv.setUint32(4, offsets[names[0]], true);
  for (const n of names) {
    const entries = [...ifds[n]].sort((a, b) => a[0] - b[0]);
    let p = offsets[n];
    let data = p + 2 + 12 * entries.length + 4;
    dv.setUint16(p, entries.length, true);
    p += 2;
    for (const [tag, type, raw] of entries) {
      const v: Value = typeof raw === 'string' && raw.startsWith('@') ? [offsets[raw.slice(1)]] : raw;
      const c = count(type, v);
      const bytes = new Uint8Array(c * SIZE[type]);
      const bv = new DataView(bytes.buffer);
      if (type === 2) bytes.set(enc(v as string));
      else
        (v as number[]).forEach((x, i) => {
          if (type === 1 || type === 7) bv.setUint8(i, x);
          else if (type === 3) bv.setUint16(i * 2, x, true);
          else if (type === 4) bv.setUint32(i * 4, x, true);
          else if (type === 5) {
            bv.setUint32(i * 8, Math.round(x * 10000), true);
            bv.setUint32(i * 8 + 4, 10000, true);
          }
        });
      dv.setUint16(p, tag, true);
      dv.setUint16(p + 2, type, true);
      dv.setUint32(p + 4, c, true);
      if (bytes.length <= 4) out.set(bytes, p + 8);
      else {
        dv.setUint32(p + 8, data, true);
        out.set(bytes, data);
        data += bytes.length + (bytes.length & 1);
      }
      p += 12;
    }
  }
  for (const [n, b] of Object.entries(blobs)) out.set(b, offsets[n]);
  return out;
}

const file = (bytes: Uint8Array, name: string) => new File([bytes as BlobPart], name);
const blobBytes = async (b: Blob) => new Uint8Array(await b.arrayBuffer());

const EXIF: Entry[] = [
  [0x829a, 5, [1 / 250]],
  [0x829d, 5, [2.8]],
  [0x8827, 3, [1600]],
  [0x920a, 5, [85]],
  [0xa001, 3, [1]],
  [0xa434, 2, 'NIKKOR Z 85mm f/1.8 S'],
  [0x9003, 2, '2026:11:06 10:30:00'],
];

function nef(orientation = 6) {
  const small = fakeJpeg(160, 120);
  const big = fakeJpeg(6048, 4024, { pad: 400 });
  const lossless = fakeJpeg(6064, 4040, { sof: 0xc3 });
  const bytes = buildTiff(
    {
      ifd0: [
        [0x010f, 2, 'NIKON CORPORATION'],
        [0x0110, 2, 'NIKON Z 8'],
        [0x0112, 3, [orientation]],
        [0x0201, 4, '@small'],
        [0x0202, 4, [small.length]],
        [0x014a, 4, '@sub'],
        [0x8769, 4, '@exif'],
      ],
      sub: [
        [0x0201, 4, '@big'],
        [0x0202, 4, [big.length]],
      ],
      exif: EXIF,
      raw: [
        [0x0103, 3, [7]],
        [0x0111, 4, '@lossless'],
        [0x0117, 4, [lossless.length]],
      ],
    },
    { small, big, lossless },
  );
  return { bytes, big };
}

function box(type: string, ...payload: Uint8Array[]): Uint8Array {
  const body = concat(...payload);
  return concat(u32be(8 + body.length), enc(type), body);
}

const hexBytes = (h: string) => new Uint8Array(h.match(/../g)!.map((x) => parseInt(x, 16)));

function cr3() {
  const cmt1 = buildTiff({ ifd0: [[0x010f, 2, 'Canon'], [0x0110, 2, 'Canon EOS R5'], [0x0112, 3, [8]]] }, {});
  const cmt2 = buildTiff({ exif: [[0x8827, 3, [800]], [0x829a, 5, [1 / 125]], [0x829d, 5, [5.6]], [0xa434, 2, 'RF85mm F1.2 L USM']] }, {});
  const prvw = fakeJpeg(1620, 1080);
  const full = fakeJpeg(8192, 5464, { pad: 300 });
  const ftyp = box('ftyp', enc('crx '), new Uint8Array(4), enc('crx isom'));
  const prvwBox = box('uuid', hexBytes('eaf42b5e1c984b88b9fbb7dc406e4d16'), new Uint8Array(8), box('PRVW', new Uint8Array(16), prvw));
  const canon = box('uuid', hexBytes('85c0b687820f11e08111f4ce462b6a48'), box('CMT1', cmt1), box('CMT2', cmt2));
  // stco needs the absolute offset of the JPEG inside mdat, which depends on moov's size:
  // build once with a placeholder, then patch.
  const build = (offset: number) => {
    const stsz = box('stsz', new Uint8Array(4), u32be(0), u32be(1), u32be(full.length));
    const stco = box('stco', new Uint8Array(4), u32be(1), u32be(offset));
    const trak = box('trak', box('mdia', box('minf', box('stbl', stsz, stco))));
    const moov = box('moov', canon, trak);
    return concat(ftyp, moov, prvwBox, box('mdat', full));
  };
  const probe = build(0);
  const jpegAt = probe.length - full.length;
  return { bytes: build(jpegAt), full, prvw };
}

function raf(orientation = 6) {
  const jpeg = fakeJpeg(7728, 5152, { orientation, make: 'FUJIFILM' });
  const header = new Uint8Array(100);
  header.set(enc('FUJIFILMCCD-RAW 0201FF383501'));
  new DataView(header.buffer).setUint32(84, 100);
  new DataView(header.buffer).setUint32(88, jpeg.length);
  return { bytes: concat(header, jpeg), jpeg };
}

// ---------------------------------------------------------------- tests

describe('detectRawFormat', () => {
  it.each([
    ['dng', buildTiff({ ifd0: [[0xc612, 1, [1, 4, 0, 0]]] }, {})],
    ['nef', nef().bytes],
    ['arw', buildTiff({ ifd0: [[0x010f, 2, 'SONY']] }, {})],
    ['pef', buildTiff({ ifd0: [[0x010f, 2, 'PENTAX Corporation']] }, {})],
    ['orf', buildTiff({ ifd0: [[0x010f, 2, 'OLYMPUS']] }, {}, 'IIRO')],
    ['rw2', buildTiff({ ifd0: [[0x010f, 2, 'Panasonic']] }, {}, 'IIU\0')],
    ['cr3', cr3().bytes],
    ['raf', raf().bytes],
  ])('identifies %s', (fmt, bytes) => {
    expect(detectRawFormat(bytes)).toBe(fmt);
  });

  it('identifies CR2 by its header marker', () => {
    const b = buildTiff({ ifd0: [[0x010f, 2, 'Canon']] }, {});
    b.set(enc('CR'), 8);
    expect(detectRawFormat(b)).toBe('cr2');
  });

  it('does not mistake ordinary TIFFs or junk for RAW', () => {
    expect(detectRawFormat(buildTiff({ ifd0: [[0x0100, 3, [8]]] }, {}), 'scan.tif')).toBeNull();
    expect(detectRawFormat(fakeJpeg(10, 10))).toBeNull();
    expect(detectRawFormat(new Uint8Array(64), 'IMG_0001.CR4')).toBeNull();
    expect(detectRawFormat(buildTiff({ ifd0: [[0x0100, 3, [8]]] }, {}), 'DSC_1.NEF')).toBe('nef');
  });
});

describe('parseRawFile', () => {
  it('extracts the largest browser-decodable preview and the EXIF of a NEF', async () => {
    const { bytes, big } = nef(6);
    const r = await parseRawFile(file(bytes, 'DSC_0042.NEF'));
    expect(r.format).toBe('nef');
    expect(r.label).toBe('NEF');
    expect([r.width, r.height]).toEqual([6048, 4024]); // not the 160 px thumb, not the lossless raw
    expect(await blobBytes(r.preview)).toEqual(big);
    expect(r.preview.type).toBe('image/jpeg');
    expect(r.orientation).toBe(6);
    expect(r.exif).toMatchObject({ make: 'NIKON CORPORATION', model: 'NIKON Z 8', iso: 1600, shutter: '1/250', lens: 'NIKKOR Z 85mm f/1.8 S', colorSpace: 'sRGB', dateTimeOriginal: '2026:11:06 10:30:00' });
    expect(r.exif.fNumber).toBeCloseTo(2.8);
    expect(r.exif.focalLength).toBeCloseTo(85);
  });

  it('reads a CR3: Canon CMT boxes and the full-size JPEG of track 1', async () => {
    const { bytes, full } = cr3();
    const r = await parseRawFile(file(bytes, 'IMG_0001.CR3'));
    expect(r.format).toBe('cr3');
    expect([r.width, r.height]).toEqual([8192, 5464]);
    expect(await blobBytes(r.preview)).toEqual(full);
    expect(r.orientation).toBe(8);
    expect(r.exif).toMatchObject({ make: 'Canon', model: 'Canon EOS R5', iso: 800, shutter: '1/125', lens: 'RF85mm F1.2 L USM' });
  });

  it('labels a .cr4 file by its name but reads it by content', async () => {
    const r = await parseRawFile(file(cr3().bytes, 'IMG_0001.CR4'));
    expect(r.format).toBe('cr3');
    expect(r.label).toBe('CR4');
  });

  it('lets the browser apply orientation when the preview carries its own (RAF)', async () => {
    const { bytes, jpeg } = raf(6);
    const r = await parseRawFile(file(bytes, 'DSCF0001.RAF'));
    expect(r.format).toBe('raf');
    expect(await blobBytes(r.preview)).toEqual(jpeg);
    expect(r.exif.orientation).toBe(6);
    expect(r.orientation).toBe(1);
    expect(r.exif.make).toBe('FUJIFILM');
  });

  it('falls back to scanning when the structure is damaged', async () => {
    const { bytes, big } = nef();
    const broken = bytes.slice();
    new DataView(broken.buffer).setUint32(4, 0x7ffffff0, true); // IFD0 offset past EOF
    const r = await parseRawFile(file(broken, 'DSC_0001.NEF'));
    expect([r.width, r.height]).toEqual([6048, 4024]);
    expect((await blobBytes(r.preview)).subarray(0, big.length)).toEqual(big);
    expect(r.orientation).toBe(1); // orientation tag unreadable
  });

  it('rejects files without a usable preview, and explains .cr4', async () => {
    const noPreview = buildTiff({ ifd0: [[0x010f, 2, 'SONY']] }, {});
    await expect(parseRawFile(file(noPreview, 'DSC.ARW'))).rejects.toBeInstanceOf(RawParseError);
    await expect(parseRawFile(file(new Uint8Array(512), 'IMG.CR4'))).rejects.toThrow(/CR4/);
  });
});

describe('helpers', () => {
  it('reads JPEG frame sizes and skips lossless JPEG', () => {
    expect(readJpegInfo(fakeJpeg(640, 480, { orientation: 3 }))).toEqual({ width: 640, height: 480, ownOrientation: 3 });
    expect(readJpegInfo(fakeJpeg(640, 480, { sof: 0xc3 }))).toBeNull();
  });

  it('names, accept list and upload types', () => {
    expect(isRawFileName('a.CR3')).toBe(true);
    expect(isRawFileName('a.jpg')).toBe(false);
    for (const ext of ['.cr2', '.cr3', '.cr4', '.arw', '.nef', '.raf', '.dng', '.orf', '.rw2', '.pef', '.tif', '.tiff']) expect(IMPORT_ACCEPT.split(',')).toContain(ext);
    expect(uploadContentType({ name: 'x.NEF', type: '' })).toBe('image/x-nikon-nef');
    expect(uploadContentType({ name: 'x.cr4', type: '' })).toBe('application/octet-stream');
    expect(uploadContentType({ name: 'x.jpg', type: 'image/jpeg' })).toBe('image/jpeg');
    expect(describeExif({ orientation: 1, model: 'X-T5', iso: 200, shutter: '1/500', fNumber: 2, focalLength: 33 })).toBe('X-T5 · ISO 200 · 1/500 · f/2 · 33 mm');
  });
});
