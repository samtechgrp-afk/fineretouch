import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ADJUSTMENTS,
  DEFAULT_SKIN_FINISH,
  FINISH_CONTROLS,
  FINISH_STYLES,
  applyFinishStyle,
  isDefault,
  finishControlActive,
  resetFinishControl,
  sanitizeAdjustments,
  sanitizeSkinFinish,
  setFinishControl,
} from './index';

describe('Professional Skin Finish settings', () => {
  it('is off by default and does not make an edit non-default', () => {
    expect(DEFAULT_SKIN_FINISH.enabled).toBe(false);
    expect(DEFAULT_ADJUSTMENTS.skinFinish).toEqual(DEFAULT_SKIN_FINISH);
    expect(isDefault(DEFAULT_ADJUSTMENTS)).toBe(true);
    expect(isDefault({ ...DEFAULT_ADJUSTMENTS, skinFinish: { ...DEFAULT_SKIN_FINISH, enabled: true } })).toBe(false);
  });

  it('starts as the Natural style with every effect switched on', () => {
    expect(DEFAULT_SKIN_FINISH.style).toBe('natural');
    for (const k of FINISH_CONTROLS) expect(DEFAULT_SKIN_FINISH[k]).toEqual({ enabled: true, amount: FINISH_STYLES.natural[k] });
  });

  it('a style sets every amount and switches every effect on; Overall strength stays', () => {
    let s = { ...DEFAULT_SKIN_FINISH, enabled: true, strength: 70 };
    s = setFinishControl(s, 'shine', { enabled: false, amount: 3 });
    const pro = applyFinishStyle(s, 'professional');
    expect(pro.style).toBe('professional');
    expect(pro.strength).toBe(70);
    for (const k of FINISH_CONTROLS) expect(pro[k]).toEqual({ enabled: true, amount: FINISH_STYLES.professional[k] });
  });

  it('styles grow from Natural to Advanced on every correction', () => {
    for (const k of ['smoothing', 'shine', 'darkSpots', 'toneEvenness', 'underEye', 'dodgeBurn'] as const) {
      expect(FINISH_STYLES.natural[k]).toBeLessThan(FINISH_STYLES.professional[k]);
      expect(FINISH_STYLES.professional[k]).toBeLessThan(FINISH_STYLES.advanced[k]);
    }
  });

  it('any change to one effect makes the style custom and leaves the others alone', () => {
    const s = applyFinishStyle({ ...DEFAULT_SKIN_FINISH, enabled: true }, 'advanced');
    const t = setFinishControl(s, 'darkSpots', { amount: 20 });
    expect(t.style).toBe('custom');
    expect(t.darkSpots).toEqual({ enabled: true, amount: 20 });
    for (const k of FINISH_CONTROLS) if (k !== 'darkSpots') expect(t[k]).toEqual(s[k]);
  });

  it('reset switches an effect back on at its style amount', () => {
    const s = applyFinishStyle({ ...DEFAULT_SKIN_FINISH, enabled: true }, 'professional');
    const off = { ...setFinishControl(s, 'underEye', { enabled: false, amount: 5 }), style: 'professional' as const };
    expect(resetFinishControl(off, 'underEye').underEye).toEqual({ enabled: true, amount: FINISH_STYLES.professional.underEye });
    const custom = setFinishControl(s, 'underEye', { amount: 5 });
    expect(resetFinishControl(custom, 'underEye').underEye.amount).toBe(FINISH_STYLES.natural.underEye);
  });

  it('an effect is active only when the finish, Overall and its own switch and amount are on', () => {
    const s = { ...DEFAULT_SKIN_FINISH, enabled: true };
    expect(finishControlActive(s, 'shine')).toBe(true);
    expect(finishControlActive({ ...s, strength: 0 }, 'shine')).toBe(false);
    expect(finishControlActive(setFinishControl(s, 'shine', { enabled: false }), 'shine')).toBe(false);
    expect(finishControlActive(setFinishControl(s, 'shine', { amount: 0 }), 'shine')).toBe(false);
    expect(finishControlActive(DEFAULT_SKIN_FINISH, 'shine')).toBe(false);
  });

  it('sanitises stored edits: clamps, fills missing effects, unknown style is custom', () => {
    const s = sanitizeSkinFinish({ enabled: true, strength: 140, style: 'glam', shine: { amount: -5 }, darkSpots: { enabled: false, amount: 60 } });
    expect(s.strength).toBe(100);
    expect(s.style).toBe('custom');
    expect(s.shine).toEqual({ enabled: true, amount: 0 });
    expect(s.darkSpots).toEqual({ enabled: false, amount: 60 });
    expect(s.dodgeBurn).toEqual(DEFAULT_SKIN_FINISH.dodgeBurn);
    expect(sanitizeSkinFinish(undefined)).toEqual(DEFAULT_SKIN_FINISH);
    // An edit saved before the finish existed loads with it off.
    const old = sanitizeAdjustments({ exposure: 10 } as never);
    expect(old.skinFinish).toEqual(DEFAULT_SKIN_FINISH);
  });
});
