import { describe, expect, it } from 'vitest';
import { DEFAULT_ADJUSTMENTS, type PerfectionReport } from '@fineretouch/shared';
import { hitMark } from '@/components/viewport/perfectionOverlay';
import { engineSourceOn, perfectedPixels, perfectionKeyOf } from './perfection';

const report = (marks: Array<[number, number, number]>): PerfectionReport => ({
  width: 1000, height: 1000, passes: 1, tiles: 1, workScale: 1, faceWidth: 500, ms: 1,
  counts: { detected: marks.length, corrected: 0, remaining: marks.length, review: 0, protected: 0, ignored: 0 },
  byKind: {}, meanConfidence: null, textureRetention: null, areas: {},
  marks: marks.map(([x, y, r], id) => ({ id, x, y, r, kind: 'acne', confidence: 0.9, region: 'face', status: 'remaining', attempts: 1, residual: false, contrast: 0.1, after: null })),
});
const frame = { display: { mode: 'edited', scale: 1, offsetX: 0, offsetY: 0, split: 0.5, showMask: false, dpr: 1, background: [0, 0, 0] }, width: 1000, height: 1000 } as never;

describe('Skin Perfection in the app', () => {
  it('keys a result by its settings and skin mask, not by unrelated sliders', () => {
    const a = perfectionKeyOf(DEFAULT_ADJUSTMENTS);
    expect(perfectionKeyOf({ ...DEFAULT_ADJUSTMENTS, exposure: 40 } as never)).toBe(a);
    expect(perfectionKeyOf({ ...DEFAULT_ADJUSTMENTS, skinPerfection: { ...DEFAULT_ADJUSTMENTS.skinPerfection, strength: 50 } })).not.toBe(a);
    expect(perfectionKeyOf({ ...DEFAULT_ADJUSTMENTS, skinMask: { ...DEFAULT_ADJUSTMENTS.skinMask, feather: 0.9 } })).not.toBe(a);
  });

  it('uses the engine as the source when Skin Perfection or the Skin Finish is on', () => {
    expect(engineSourceOn(DEFAULT_ADJUSTMENTS)).toBe(false);
    expect(engineSourceOn({ ...DEFAULT_ADJUSTMENTS, skinFinish: { ...DEFAULT_ADJUSTMENTS.skinFinish, enabled: true } })).toBe(true);
    expect(engineSourceOn({ ...DEFAULT_ADJUSTMENTS, skinPerfection: { ...DEFAULT_ADJUSTMENTS.skinPerfection, enabled: true } })).toBe(true);
  });

  it('keys by the Skin Finish only while it is on (a Perfection-only key is unchanged)', () => {
    const a = perfectionKeyOf(DEFAULT_ADJUSTMENTS);
    const off = { ...DEFAULT_ADJUSTMENTS, skinFinish: { ...DEFAULT_ADJUSTMENTS.skinFinish, strength: 40 } };
    expect(perfectionKeyOf(off)).toBe(a);
    const on = { ...DEFAULT_ADJUSTMENTS, skinFinish: { ...DEFAULT_ADJUSTMENTS.skinFinish, enabled: true } };
    expect(perfectionKeyOf(on)).not.toBe(a);
    const moved = { ...on, skinFinish: { ...on.skinFinish, shine: { enabled: true, amount: 90 } } };
    expect(perfectionKeyOf(moved)).not.toBe(perfectionKeyOf(on));
  });

  it('has no perfected pixels before the engine ran', () => {
    expect(perfectedPixels('nope', 'k', 10, 10)).toBeNull();
  });

  it('hits the smallest mark under the pointer and nothing elsewhere', () => {
    const rep = report([[0.5, 0.5, 0.02], [0.505, 0.5, 0.004]]);
    expect(hitMark(505, 500, frame, rep)?.id).toBe(1);
    expect(hitMark(480, 500, frame, rep)?.id).toBe(0);
    expect(hitMark(100, 100, frame, rep)).toBeNull();
    expect(hitMark(505, 500, frame, null)).toBeNull();
  });
});
