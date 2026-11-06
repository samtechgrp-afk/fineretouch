import { describe, expect, it } from 'vitest';
import { DEFAULT_EXPORT_PREFS, freeName, isScopeError, sanitizeExportPrefs } from './exportManager';
import { DEFAULT_WATERMARK, sanitizeWatermark, watermarkActive, watermarkLayout } from './watermark';

describe('export preferences', () => {
  it('fall back to defaults for garbage', () => {
    expect(sanitizeExportPrefs(null)).toEqual({ ...DEFAULT_EXPORT_PREFS, watermark: sanitizeWatermark(null) });
    const p = sanitizeExportPrefs({ format: 'tiff', quality: 500, destination: 'cloud', folder: '', maxLongEdge: 10 });
    expect(p.format).toBe('jpeg');
    expect(p.quality).toBe(100);
    expect(p.destination).toBe('ask');
    expect(p.folder).toBeNull();
    expect(p.maxLongEdge).toBeNull();
  });

  it('keep a chosen default folder and auto-save mode', () => {
    const p = sanitizeExportPrefs({ destination: 'folder', folder: 'D:\\Shoots\\Exports', revealAfterSave: true });
    expect(p).toMatchObject({ destination: 'folder', folder: 'D:\\Shoots\\Exports', revealAfterSave: true });
  });
});

describe('freeName', () => {
  const join = async (...p: string[]) => p.join('/');
  it('never overwrites files on disk or earlier files in the batch', async () => {
    const onDisk = new Set(['/out/a.jpg', '/out/a_2.jpg']);
    const taken = new Set<string>();
    const exists = async (p: string) => onDisk.has(p);
    expect(await freeName('/out', 'a.jpg', taken, exists, join)).toBe('a_3.jpg');
    expect(await freeName('/out', 'a.jpg', taken, exists, join)).toBe('a_4.jpg');
    expect(await freeName('/out', 'b.png', taken, exists, join)).toBe('b.png');
    expect(await freeName('/out', 'B.PNG', taken, exists, join)).toBe('B_2.PNG');
  });
});

describe('isScopeError', () => {
  it('recognises fs scope refusals', () => {
    expect(isScopeError(new Error('forbidden path: C:\\x'))).toBe(true);
    expect(isScopeError('fs.write_file not allowed')).toBe(true);
    expect(isScopeError(new Error('disk full'))).toBe(false);
  });
});

describe('watermark', () => {
  it('sanitises settings', () => {
    const w = sanitizeWatermark({ size: 99, opacity: -1, color: 'red', anchor: 'nowhere', logo: 'javascript:alert(1)', font: 'comic' });
    expect(w.size).toBe(15);
    expect(w.opacity).toBe(0);
    expect(w.color).toBe(DEFAULT_WATERMARK.color);
    expect(w.anchor).toBe('bottom-right');
    expect(w.logo).toBeNull();
    expect(w.font).toBe('sans');
  });

  it('is only active with something to draw', () => {
    expect(watermarkActive({ ...DEFAULT_WATERMARK, enabled: true })).toBe(true);
    expect(watermarkActive({ ...DEFAULT_WATERMARK, enabled: true, text: '  ' })).toBe(false);
    expect(watermarkActive({ ...DEFAULT_WATERMARK, enabled: true, opacity: 0 })).toBe(false);
    expect(watermarkActive({ ...DEFAULT_WATERMARK, enabled: false })).toBe(false);
  });

  it('scales with the image, so 1080p and 4K look identical', () => {
    const w = { ...DEFAULT_WATERMARK, size: 3, margin: 2 };
    const hd = watermarkLayout(w, 1920, 1080);
    const uhd = watermarkLayout(w, 3840, 2160);
    expect(uhd.fontPx / hd.fontPx).toBeCloseTo(2);
    expect(uhd.marginPx / hd.marginPx).toBeCloseTo(2);
    expect(hd.fontPx / 1080).toBeCloseTo(0.03);
    // Portrait and landscape of the same photo get the same mark (short edge drives it).
    expect(watermarkLayout(w, 4000, 6000).fontPx).toBeCloseTo(watermarkLayout(w, 6000, 4000).fontPx);
  });
});
