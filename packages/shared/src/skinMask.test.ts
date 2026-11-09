import { describe, expect, it } from 'vitest';
import { applyPreset, DEFAULT_ADJUSTMENTS, isDefault, sanitizeAdjustments } from './index.js';
import { DEFAULT_SKIN_MASK, MAX_MASK_STROKES, MAX_STROKE_POINTS, sanitizeSkinMask, sanitizeStroke, withOwnStrokes } from './skinMask.js';

describe('skin mask settings', () => {
  it('defaults for old edits and junk', () => {
    expect(sanitizeAdjustments({}).skinMask).toEqual(DEFAULT_SKIN_MASK);
    expect(sanitizeSkinMask('x')).toEqual(DEFAULT_SKIN_MASK);
    expect(sanitizeSkinMask({ feather: 400, useSegmentation: 'no' })).toMatchObject({ feather: 100, useSegmentation: true });
  });

  it('cleans strokes: modes, sizes, odd and non-finite points, limits', () => {
    expect(sanitizeStroke({ mode: 'paint', size: 5, points: [0.1, 0.2, 0.3] })).toEqual({ mode: 'add', size: 0.2, points: [0.1, 0.2] });
    expect(sanitizeStroke({ mode: 'erase', points: [0.1, Number.NaN] })).toBeNull();
    expect(sanitizeStroke({ points: [0.5] })).toBeNull();
    const long = sanitizeStroke({ points: Array.from({ length: 3 * MAX_STROKE_POINTS }, () => 0.5) })!;
    expect(long.points.length).toBe(2 * MAX_STROKE_POINTS);
    const many = sanitizeSkinMask({ strokes: Array.from({ length: MAX_MASK_STROKES + 5 }, (_, i) => ({ points: [i / 1000, 0.5] })) });
    expect(many.strokes.length).toBe(MAX_MASK_STROKES);
    expect(many.strokes[MAX_MASK_STROKES - 1].points[0]).toBeCloseTo((MAX_MASK_STROKES + 4) / 1000); // newest kept
  });

  it('counts toward isDefault', () => {
    expect(isDefault({ ...DEFAULT_ADJUSTMENTS })).toBe(true);
    expect(isDefault(sanitizeAdjustments({ skinMask: { includeBodySkin: true } }))).toBe(false);
    expect(isDefault(sanitizeAdjustments({ skinMask: { strokes: [{ points: [0.5, 0.5] }] } }))).toBe(false);
  });

  it('presets and copies never move one photo\'s strokes onto another', () => {
    const mine = sanitizeAdjustments({ skinMask: { strokes: [{ mode: 'erase', points: [0.2, 0.2] }] } });
    const other = sanitizeAdjustments({ skinSmoothing: 40, skinMask: { includeBodySkin: true, strokes: [{ points: [0.8, 0.8] }] } });
    const out = applyPreset(mine, { id: 'p', name: 'p', description: '', adjustments: other });
    expect(out.skinSmoothing).toBe(40);
    expect(out.skinMask.includeBodySkin).toBe(true);
    expect(out.skinMask.strokes).toEqual(mine.skinMask.strokes);
    expect(withOwnStrokes(other, mine).skinMask.strokes).toBe(mine.skinMask.strokes);
  });
});
