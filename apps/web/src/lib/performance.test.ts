import { describe, expect, it } from 'vitest';
import { DEFAULT_ADJUSTMENTS, createLayer, sanitizeAdjustments, type Adjustments } from '@fineretouch/shared';
import { GPU_BUDGETS, GPU_BYTES_PER_PIXEL, exportApron, planExport, planTiles } from './export';
import { isSoftwareRenderer, resolveEngine, type GpuInfo } from './processing';
import { stageKeys } from '@/engine/renderer';
import { resolveAdjustments } from '@fineretouch/shared';

const base = (over: Partial<Adjustments> = {}): Adjustments => sanitizeAdjustments({ ...DEFAULT_ADJUSTMENTS, ...over });

describe('tiled export plan', () => {
  it('covers every pixel exactly once with inner rects', () => {
    const tiles = planTiles(9000, 6000, 4096, 600)!;
    const cover = new Uint8Array(9000 * 6000 / 100);
    let area = 0;
    for (const t of tiles) {
      area += t.iw * t.ih;
      expect(t.ix - t.x).toBeLessThanOrEqual(600);
      expect(t.w).toBeLessThanOrEqual(4096);
      expect(t.h).toBeLessThanOrEqual(4096);
      // Apron on every side that is not the photo border.
      if (t.ix > 0) expect(t.ix - t.x).toBe(600);
      if (t.ix + t.iw < 9000) expect(t.x + t.w - (t.ix + t.iw)).toBe(600);
      for (let y = t.iy; y < t.iy + t.ih; y += 10) for (let x = t.ix; x < t.ix + t.iw; x += 10) cover[(y / 10) * 900 + x / 10]++;
    }
    expect(area).toBe(9000 * 6000);
    expect(cover.every((c) => c === 1)).toBe(true);
  });

  it('renders in one pass when the photo fits the texture limit and the budget', () => {
    const p = planExport(6000, 4000, base(), 16384, GPU_BUDGETS.balanced);
    expect(p.tiles).toHaveLength(1);
    expect(p.scaled).toBe(false);
  });

  it('tiles past the texture limit instead of scaling down', () => {
    const p = planExport(12000, 8000, base(), 8192, GPU_BUDGETS.high);
    expect(p.width).toBe(12000);
    expect(p.scaled).toBe(false);
    expect(p.tiles.length).toBeGreaterThan(1);
  });

  it('tiles when the memory budget is smaller than the photo', () => {
    const p = planExport(6000, 4000, base(), 16384, GPU_BUDGETS.low);
    expect(p.tiles.length).toBeGreaterThan(1);
    for (const t of p.tiles) expect(t.w * t.h * GPU_BYTES_PER_PIXEL).toBeLessThanOrEqual(GPU_BUDGETS.low * 2 ** 20);
  });

  it('grows the apron with frequency separation and bokeh', () => {
    const L = 6000;
    const plain = exportApron(base(), L);
    expect(plain).toBeGreaterThanOrEqual(3 * 0.03 * L);
    const fs = createLayer('frequency', 'fs');
    expect(exportApron(base({ localRetouch: { heal: [], layers: [fs] } }), L)).toBeGreaterThan(plain);
    expect(exportApron(base({ depthBokeh: { ...DEFAULT_ADJUSTMENTS.depthBokeh, enabled: true, blurRadius: 80 } }), L)).toBeGreaterThan(plain);
  });

  it('scales down only when even tiles cannot fit, and says so', () => {
    const p = planExport(20000, 15000, base(), 2048, GPU_BUDGETS.low);
    expect(p.scaled).toBe(true);
    expect(Math.max(p.width, p.height)).toBeLessThan(20000);
  });
});

describe('preview stage cache keys', () => {
  const keys = (a: Adjustments, v = 1, local = 0) => stageKeys(resolveAdjustments(a), { hasFace: true, sourceStats: null }, { retouch: v, studio: 0, local, bokeh: 0 });

  it('a bokeh change re-runs only the bokeh stage', () => {
    const a = keys(base());
    const b = keys(base({ depthBokeh: { ...DEFAULT_ADJUSTMENTS.depthBokeh, blurRadius: 77 } }));
    expect(b.retouch).toBe(a.retouch);
    expect(b.studio).toBe(a.studio);
    expect(b.local).toBe(a.local);
    expect(b.bokeh).not.toBe(a.bokeh);
  });

  it('a glare change keeps the retouch stage', () => {
    const a = keys(base());
    const b = keys(base({ glareReduction: 40 }));
    expect(b.retouch).toBe(a.retouch);
    expect(b.studio).not.toBe(a.studio);
    expect(b.bokeh).not.toBe(a.bokeh);
  });

  it('a slider or a new texture re-runs everything', () => {
    const a = keys(base());
    expect(keys(base({ exposure: 10 })).retouch).not.toBe(a.retouch);
    expect(keys(base(), 2).retouch).not.toBe(a.retouch);
  });

  it('a new layer mask re-runs the layers and bokeh only', () => {
    const a = keys(base());
    const b = keys(base(), 1, 1);
    expect(b.studio).toBe(a.studio);
    expect(b.local).not.toBe(a.local);
  });

  it('skin mask strokes do not enter the keys (they arrive as textures)', () => {
    const a = keys(base());
    const b = keys(base({ skinMask: { ...DEFAULT_ADJUSTMENTS.skinMask, strokes: [{ mode: 'add', size: 0.02, points: [0.1, 0.1] }] } }));
    expect(b.bokeh).toBe(a.bokeh);
  });
});

describe('CPU / GPU selection', () => {
  const hw: GpuInfo = { available: true, renderer: 'AMD Radeon(TM) Graphics', maxTexture: 16384, software: false };
  const sw: GpuInfo = { available: true, renderer: 'Google SwiftShader', maxTexture: 8192, software: true };

  it('recognises software rasterisers', () => {
    expect(isSoftwareRenderer('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)')).toBe(true);
    expect(isSoftwareRenderer('llvmpipe (LLVM 15.0.7, 256 bits)')).toBe(true);
    expect(isSoftwareRenderer('Microsoft Basic Render Driver')).toBe(true);
    expect(isSoftwareRenderer('ANGLE (AMD, AMD Radeon(TM) Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)')).toBe(false);
  });

  it('auto uses the GPU on real hardware and the CPU engine on a software GPU', () => {
    expect(resolveEngine('auto', hw, true).engine).toBe('browser');
    expect(resolveEngine('auto', sw, true).engine).toBe('server');
    expect(resolveEngine('auto', sw, false).engine).toBe('browser');
    expect(resolveEngine('auto', { ...hw, available: false }, true).engine).toBe('server');
  });

  it('respects an explicit choice, falling back when the CPU engine is down', () => {
    expect(resolveEngine('gpu', sw, true).engine).toBe('browser');
    expect(resolveEngine('cpu', hw, true).engine).toBe('server');
    expect(resolveEngine('cpu', hw, false).engine).toBe('browser');
  });
});
