import { describe, expect, it } from 'vitest';

import { DEFAULT_ADJUSTMENTS, applyPreset, isDefault, sanitizeAdjustments } from './index.js';
import {
  DEFAULT_SKIN_PERFECTION,
  correctRemaining,
  groupSensitivity,
  overrideMark,
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
