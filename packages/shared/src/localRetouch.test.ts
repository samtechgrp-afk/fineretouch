import { describe, expect, it } from 'vitest';
import { DEFAULT_ADJUSTMENTS, isDefault, sanitizeAdjustments } from './index.js';
import { createLayer, layerIsActive, MAX_LAYERS, sanitizeLocalRetouch, splitSigma } from './localRetouch.js';

describe('local retouch model', () => {
  it('defaults to nothing and keeps old edits default', () => {
    expect(DEFAULT_ADJUSTMENTS.localRetouch).toEqual({ heal: [], layers: [] });
    expect(sanitizeAdjustments({}).localRetouch).toEqual({ heal: [], layers: [] });
    expect(isDefault(sanitizeAdjustments({ localRetouch: { heal: [], layers: [] } }))).toBe(true);
  });

  it('sanitises strokes and layers from untrusted JSON', () => {
    const l = sanitizeLocalRetouch({
      heal: [
        { mode: 'clone', size: 9, hardness: -5, opacity: 300, offset: [2, -0.1], points: [0.1, 0.2, 0.3] },
        { mode: 'heal', points: [0.5, 0.5] }, // no offset: dropped
        'junk',
      ],
      layers: [
        { id: 'a', kind: 'dodgeBurn', dodgeBurnRange: 'nope', mask: { base: 'skin', lumaMin: 80, lumaMax: 20, strokes: [{ mode: 'burn', flow: 0, points: [0.5, 0.5] }] } },
        { id: 'a', kind: 'frequency', texture: 500 },
        ...Array.from({ length: 20 }, () => ({ kind: 'correction' })),
      ],
    });
    expect(l.heal).toEqual([{ mode: 'clone', size: 0.2, hardness: 0, opacity: 100, offset: [1, -0.1], points: [0.1, 0.2] }]);
    expect(l.layers.length).toBe(MAX_LAYERS);
    expect(l.layers[0]).toMatchObject({ kind: 'dodgeBurn', dodgeBurnRange: 'all', mask: { base: 'skin', lumaMin: 80, lumaMax: 80 } });
    expect(l.layers[0].mask.strokes[0].flow).toBe(1);
    expect(l.layers[1].id).not.toBe('a');
    expect(l.layers[1].texture).toBe(200);
  });

  it('knows when a layer changes pixels', () => {
    const fs = createLayer('frequency', 'f');
    expect(layerIsActive(fs)).toBe(true); // AI skin base, smoothing 50
    expect(layerIsActive({ ...fs, enabled: false })).toBe(false);
    const corr = createLayer('correction', 'c');
    expect(layerIsActive({ ...corr, exposure: 20 })).toBe(false); // nothing painted yet
    expect(layerIsActive({ ...corr, exposure: 20, mask: { ...corr.mask, invert: true } })).toBe(true);
    const db = createLayer('dodgeBurn', 'd');
    expect(layerIsActive(db)).toBe(false);
    expect(layerIsActive({ ...db, mask: { ...db.mask, strokes: [{ mode: 'dodge', size: 0.02, flow: 30, points: [0.5, 0.5] }] } })).toBe(true);
  });

  it('maps the split radius like engines/local.split_sigma', () => {
    expect(splitSigma(0, 2000)).toBeCloseTo(1.6, 6);
    expect(splitSigma(100, 2000)).toBeCloseTo(48, 6);
    expect(splitSigma(0, 100)).toBe(0.6);
  });
});
