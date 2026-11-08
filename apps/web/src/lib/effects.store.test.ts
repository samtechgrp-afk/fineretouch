import { DEFAULT_ADJUSTMENTS, DEFAULT_EFFECTS, resolveAdjustments } from '@fineretouch/shared';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useEditor } from './store';

vi.stubGlobal('URL', { ...URL, revokeObjectURL: vi.fn() });

beforeEach(() => {
  useEditor.setState({ photos: {}, order: [], activeId: null, selected: [], clipboard: null });
  useEditor.getState().addPhotos([{ id: 'a', name: 'a.jpg', width: 100, height: 100, thumbUrl: 'blob:a' }]);
});

const adj = () => useEditor.getState().photos.a.adjustments;

describe('per-effect controls in the store', () => {
  it('sets a multiplier with one undo step and changes the effective value only', () => {
    const s = useEditor.getState();
    s.updateAdjustments({ skinSmoothing: 30 });
    s.setEffectControl('skinSmoothing', { multiplier: 2 });
    expect(adj().skinSmoothing).toBe(30);
    expect(resolveAdjustments(adj()).skinSmoothing).toBe(60);
    s.undo();
    expect(adj().effects.skinSmoothing.multiplier).toBe(1);
  });

  it('disables an effect without losing its slider, and resets it fully', () => {
    const s = useEditor.getState();
    s.updateAdjustments({ darkCircles: 55 });
    s.setEffectControl('darkCircles', { enabled: false, multiplier: 3, opacity: 40 });
    expect(adj().darkCircles).toBe(55);
    expect(resolveAdjustments(adj()).darkCircles).toBe(0);
    s.resetEffect('darkCircles');
    expect(adj().darkCircles).toBe(DEFAULT_ADJUSTMENTS.darkCircles);
    expect(adj().effects.darkCircles).toEqual(DEFAULT_EFFECTS.darkCircles);
  });

  it('resets structured effects (stray hair strength) to their own default', () => {
    const s = useEditor.getState();
    s.setStrayHairRemoval({ enabled: true, strength: 90 });
    s.setEffectControl('strayHair', { multiplier: 2 });
    s.resetEffect('strayHair');
    expect(adj().strayHairRemoval).toEqual({ enabled: true, strength: DEFAULT_ADJUSTMENTS.strayHairRemoval.strength });
  });

  it('global strength scales effects but keeps every individual setting', () => {
    const s = useEditor.getState();
    s.updateAdjustments({ skinSmoothing: 40, teethWhiten: 20, exposure: 30 });
    s.setGlobalStrength(25);
    const r = resolveAdjustments(adj());
    expect([adj().skinSmoothing, adj().teethWhiten]).toEqual([40, 20]);
    expect([r.skinSmoothing, r.teethWhiten, r.exposure]).toEqual([10, 5, 30]);
  });

  it('presets and copy/paste carry effect controls', () => {
    const s = useEditor.getState();
    s.updateAdjustments({ blemishRemoval: 50, globalStrength: 70 });
    s.setEffectControl('blemishRemoval', { multiplier: 3 });
    const p = s.saveCustomPreset('Mine');
    expect(p?.adjustments.effects?.blemishRemoval.multiplier).toBe(3);
    expect(p?.adjustments.globalStrength).toBe(70);
  });
});
