import { beforeEach, describe, expect, it, vi } from 'vitest';

// Minimal browser stand-ins: decoding gives a bitmap-like object; thumbnails go through a canvas.
const closed: string[] = [];
let decodes = 0;
vi.stubGlobal('createImageBitmap', async (src: { name?: string; tag?: string }, opts?: { resizeWidth?: number }) => {
  decodes++;
  const tag = src.name ?? src.tag ?? 'x';
  return { width: opts?.resizeWidth ?? 4000, height: 3000, tag, close: () => closed.push(tag) };
});
vi.stubGlobal('document', {
  createElement: () => ({ width: 0, height: 0, getContext: () => ({ drawImage() {} }), toBlob: (cb: (b: Blob) => void) => cb(new Blob(['j'])) }),
});
URL.createObjectURL = () => 'blob:t';

const images = await import('./images');

describe('preview cache (memory management)', () => {
  beforeEach(() => {
    closed.length = 0;
  });

  it('keeps the most recent previews and decodes an evicted one again on demand', async () => {
    const n = images.PROXY_CACHE_SIZE + 3;
    for (let i = 0; i < n; i++) await images.loadImage(`id${i}`, new File(['x'], `p${i}.jpg`, { type: 'image/jpeg' }));
    expect(images.proxyCount()).toBe(n);
    images.getProxy('id0'); // recently used: survives the trim
    expect(images.trimProxies()).toBe(3);
    expect(images.proxyCount()).toBe(images.PROXY_CACHE_SIZE);
    expect(images.getProxy('id0')).toBeDefined();
    expect(images.getProxy('id1')).toBeUndefined();
    expect(images.hasImage('id1')).toBe(true);
    const before = decodes;
    const again = await images.ensureProxy('id1');
    expect(again?.width).toBe(images.PROXY_LONG_EDGE);
    expect(decodes).toBeGreaterThan(before);
    expect(images.proxyCount()).toBe(images.PROXY_CACHE_SIZE);
  });
});
