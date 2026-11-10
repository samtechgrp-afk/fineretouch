import { describe, expect, it } from 'vitest';

import { DEFAULT_ADJUSTMENTS, applyPreset, isDefault, sanitizeAdjustments } from './index.js';
import {
  DEFAULT_SKIN_PERFECTION,
  bulkPlan,
  correctRemaining,
  groupSensitivity,
  overrideMark,
  qualitySummary,
  sanitizeSkinPerfection,
  setGroupSensitivity,
  type PerfectionReport,
} from './perfection.js';

const report = (marks: Partial<PerfectionReport['marks'][number]>[]): PerfectionReport => ({
  width: 100, height: 100, passes: 1, tiles: 1, workScale: 1, faceWidth: 50, ms: 1,
  counts: { detected: 0, corrected: 0, remaining: 0, review: 0, protected: 0, ignored: 0 },
  byKind: {}, meanConfidence: null, textureRetention: null, areas: {},
  marks: marks.map((m, i) => ({
    id: i, x: 0.5, y: 0.5, r: 0.01, kind: 'acne', confidence: 0.9, region: 'face', status: 'remaining',
    attempts: 1, residual: false, contrast: 0.1, after: 0.08, ...m,
  })),
});

describe('skin perfection settings', () => {
  it('is off and default in new adjustments', () => {
    expect(DEFAULT_ADJUSTMENTS.skinPerfection.enabled).toBe(false);
    expect(isDefault(sanitizeAdjustments({}))).toBe(true);
    expect(isDefault(sanitizeAdjustments({ skinPerfection: { enabled: true } }))).toBe(false);
  });

  it('sanitizes ranges, categories and overrides', () => {
    const s = sanitizeSkinPerfection({
      multiplier: 7, passes: 9, strength: -3,
      categories: { acne: { sensitivity: 400, strength: 'x' } },
      overrides: [{ x: 2, y: 0.5, action: 'correct' }, { x: 'bad' }],
    });
    expect(s.multiplier).toBe(3);
    expect(s.passes).toBe(4);
    expect(s.strength).toBe(0);
    expect(s.categories.acne).toEqual({ enabled: true, sensitivity: 100, strength: DEFAULT_SKIN_PERFECTION.categories.acne.strength });
    expect(s.categories.blackhead).toEqual(DEFAULT_SKIN_PERFECTION.categories.blackhead);
    expect(s.overrides).toEqual([{ x: 1, y: 0.5, radius: 0.004, action: 'correct' }]);
  });

  it('group sliders drive every category in the group', () => {
    const s = setGroupSensitivity(DEFAULT_SKIN_PERFECTION, 'acne', 20);
    expect([s.categories.acne.sensitivity, s.categories.pimple.sensitivity, s.categories.whitehead.sensitivity]).toEqual([20, 20, 20]);
    expect(groupSensitivity(s, 'acne')).toBe(20);
    expect(s.categories.blackhead.sensitivity).toBe(DEFAULT_SKIN_PERFECTION.categories.blackhead.sensitivity);
  });

  it('correct-remaining adds overrides for confident remaining marks only', () => {
    const r = report([{ confidence: 0.9 }, { confidence: 0.3, x: 0.2 }, { status: 'corrected', x: 0.8 }]);
    const s = correctRemaining(DEFAULT_SKIN_PERFECTION, r, 0.5);
    expect(s.overrides).toHaveLength(1);
    expect(s.overrides[0]).toMatchObject({ x: 0.5, y: 0.5, action: 'correct' });
  });

  it('marks bulk corrections so the engine holds them to the quality gate (B1)', () => {
    const r = report([{ confidence: 0.9 }, { confidence: 0.8, x: 0.2 }]);
    const s = correctRemaining(DEFAULT_SKIN_PERFECTION, r, 0.5);
    expect(s.overrides.every((o) => o.bulk === true && o.action === 'correct')).toBe(true);
    // The flag survives a save / load round trip; a hand-picked choice carries none.
    expect(sanitizeSkinPerfection(JSON.parse(JSON.stringify(s))).overrides.map((o) => o.bulk)).toEqual([true, true]);
    const own = overrideMark(s, r.marks[0], 'correct');
    expect(own.overrides).toHaveLength(2);
    expect(own.overrides[1]).toMatchObject({ x: 0.5, action: 'correct' });
    expect(own.overrides[1].bulk).toBeUndefined();
    expect(sanitizeSkinPerfection({ ...s, overrides: [{ x: 0.1, y: 0.1, radius: 0.01, action: 'ignore', bulk: true }] }).overrides[0].bulk).toBeUndefined();
  });

  it('correct-remaining never corrects a mark that may be a natural feature', () => {
    const r = report([
      { status: 'review', reason: 'lowConfidence', confidence: 0.6 },
      { status: 'review', reason: 'mayBeFreckle', confidence: 0.95, x: 0.2 },
      { status: 'review', reason: 'mayBeFollicle', confidence: 0.95, x: 0.8 },
    ]);
    const s = correctRemaining(DEFAULT_SKIN_PERFECTION, r, 0.5);
    expect(s.overrides).toHaveLength(1);
    expect(s.overrides[0]).toMatchObject({ x: 0.5 });
    // A per-mark decision still can.
    expect(overrideMark(DEFAULT_SKIN_PERFECTION, r.marks[1], 'correct').overrides).toHaveLength(1);
  });

  it('correct-remaining never re-corrects a repair that failed the quality checks', () => {
    const r = report([
      { status: 'remaining', confidence: 0.9 },
      { status: 'review', reason: 'repairFailed', confidence: 0.95, x: 0.2 },
    ]);
    const s = correctRemaining(DEFAULT_SKIN_PERFECTION, r, 0.5);
    expect(s.overrides).toHaveLength(1);
    expect(s.overrides[0]).toMatchObject({ x: 0.5 });
  });

  it('counts only what the bulk action will attempt, and none of it as a sure correction (B2)', () => {
    const failed = { failed: ['colour' as const], values: {}, incomplete: false };
    const r = report([
      { attempts: 2, quality: failed }, // tried twice, still failing a check
      { attempts: 1, quality: null, x: 0.1 }, // tried, still visible
      { status: 'review', reason: 'lowConfidence', confidence: 0.6, attempts: 0, x: 0.2 }, // never tried
      { residual: true, attempts: 1, quality: failed, x: 0.3 }, // found by the last rescan
      { residual: true, status: 'review', reason: 'lowConfidence', confidence: 0.6, attempts: 0, x: 0.4 },
      { status: 'review', reason: 'repairFailed', attempts: 2, quality: failed, x: 0.5 }, // already undone
      { status: 'review', reason: 'mayBeFreckle', attempts: 0, x: 0.6 },
      { confidence: 0.3, x: 0.7 }, // below the threshold
      { status: 'corrected', x: 0.8 },
    ]);
    expect(bulkPlan(r, 0.5)).toEqual({ attempt: 3, failedBefore: 2, oneByOne: 2 });
    // The count is exactly what the button sends to the engine.
    expect(correctRemaining(DEFAULT_SKIN_PERFECTION, r, 0.5).overrides.map((o) => o.x)).toEqual([0.5, 0.1, 0.2]);
    expect(bulkPlan(null, 0.5)).toEqual({ attempt: 0, failedBefore: 0, oneByOne: 0 });
  });

  it('after one click nothing is left to try in bulk (the count converges)', () => {
    // What the engine reports after the click: the retried marks either passed (corrected) or
    // were undone (review / repairFailed); the rescan marks are still one-by-one.
    const failed = { failed: ['residual' as const], values: {}, incomplete: false };
    const after = report([
      { status: 'corrected', attempts: 3 },
      { status: 'review', reason: 'repairFailed', attempts: 3, quality: failed, x: 0.1 },
      { residual: true, attempts: 1, quality: failed, x: 0.3 },
    ]);
    expect(bulkPlan(after, 0.5)).toEqual({ attempt: 0, failedBefore: 0, oneByOne: 1 });
  });

  it('quality summary names the failed checks and what happened to the repair', () => {
    expect(qualitySummary(null)).toBeNull();
    expect(qualitySummary({ failed: [], incomplete: false, values: { residual: 0.1, colour: 0.4, texture: 1 } })).toBe(
      'Passed all 3 quality checks that apply to it.',
    );
    const kept = qualitySummary({ failed: ['texture', 'edges'], incomplete: false, kept: true, values: { texture: 2.1, edges: 2.2 } })!;
    expect(kept).toContain('Failed: texture, edges.');
    expect(kept).toContain('Kept because you chose Correct');
    expect(qualitySummary({ failed: ['colour'], incomplete: false, kept: false, values: { colour: 5 } })).toContain('original skin was restored');
  });

  it('override at a mark replaces an earlier decision there', () => {
    const a = overrideMark(DEFAULT_SKIN_PERFECTION, { x: 0.5, y: 0.5, r: 0.01 }, 'ignore');
    const b = overrideMark(a, { x: 0.5, y: 0.5, r: 0.01 }, 'correct');
    expect(b.overrides).toHaveLength(1);
    expect(b.overrides[0].action).toBe('correct');
  });

  it('overrides belong to the photo: presets keep the target photo\'s decisions', () => {
    const current = sanitizeAdjustments({ skinPerfection: { enabled: true, overrides: [{ x: 0.1, y: 0.1, action: 'ignore' }] } });
    const out = applyPreset(current, { id: 'p', name: 'p', builtIn: false, adjustments: { skinPerfection: { ...DEFAULT_SKIN_PERFECTION, enabled: true, overrides: [] } } });
    expect(out.skinPerfection.overrides).toHaveLength(1);
  });
});
