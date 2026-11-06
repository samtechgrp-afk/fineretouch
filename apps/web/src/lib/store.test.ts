import { BUILT_IN_PRESETS, DEFAULT_ADJUSTMENTS } from '@fineretouch/shared';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { HISTORY_LIMIT, useEditor } from './store';

vi.stubGlobal('URL', { ...URL, revokeObjectURL: vi.fn() });

const add = (n: number) =>
  useEditor.getState().addPhotos(Array.from({ length: n }, (_, i) => ({ id: `p${i}`, name: `IMG_${i}.jpg`, width: 6000, height: 4000, thumbUrl: `blob:${i}` })));

beforeEach(() => {
  useEditor.setState({ photos: {}, order: [], activeId: null, selected: [], clipboard: null });
});

describe('editor store', () => {
  it('adds photos with defaults and activates the first', () => {
    add(3);
    const s = useEditor.getState();
    expect(s.order).toEqual(['p0', 'p1', 'p2']);
    expect(s.activeId).toBe('p0');
    expect(s.photos.p1.adjustments).toEqual(DEFAULT_ADJUSTMENTS);
  });

  it('records one undo step per slider drag, not per tick', () => {
    add(1);
    const s = useEditor.getState();
    s.beginEdit();
    for (let v = 1; v <= 40; v++) s.setAdjustment('skinSmoothing', v);
    expect(useEditor.getState().photos.p0.past).toHaveLength(1);
    s.undo();
    expect(useEditor.getState().photos.p0.adjustments.skinSmoothing).toBe(0);
    s.redo();
    expect(useEditor.getState().photos.p0.adjustments.skinSmoothing).toBe(40);
  });

  it('does not stack duplicate snapshots for no-op clicks', () => {
    add(1);
    const s = useEditor.getState();
    s.beginEdit();
    s.beginEdit();
    expect(useEditor.getState().photos.p0.past).toHaveLength(1);
  });

  it('caps history length', () => {
    add(1);
    const s = useEditor.getState();
    for (let i = 0; i < HISTORY_LIMIT + 20; i++) s.updateAdjustments({ exposure: (i % 100) - 50 });
    expect(useEditor.getState().photos.p0.past).toHaveLength(HISTORY_LIMIT);
  });

  it('clamps out-of-range values', () => {
    add(1);
    useEditor.getState().setAdjustment('exposure', 900);
    expect(useEditor.getState().photos.p0.adjustments.exposure).toBe(100);
  });

  it('syncs the active photo to all others, each undoable on its own', () => {
    add(4);
    const s = useEditor.getState();
    s.updateAdjustments({ skinSmoothing: 55, temperature: 12 });
    s.syncToSelected();
    const st = useEditor.getState();
    for (const id of ['p1', 'p2', 'p3']) expect(st.photos[id].adjustments.skinSmoothing).toBe(55);
    st.undo('p2');
    expect(useEditor.getState().photos.p2.adjustments.skinSmoothing).toBe(0);
    expect(useEditor.getState().photos.p3.adjustments.skinSmoothing).toBe(55);
  });

  it('syncs only to the selection when several photos are selected', () => {
    add(4);
    const s = useEditor.getState();
    s.updateAdjustments({ vignette: -30 });
    s.toggleSelected('p2', true, false);
    useEditor.getState().syncToSelected();
    const st = useEditor.getState();
    expect(st.selected).toEqual(['p0', 'p2']);
    expect(st.photos.p2.adjustments.vignette).toBe(-30);
    expect(st.photos.p1.adjustments.vignette).toBe(0);
  });

  it('copy/paste and presets target the selection', () => {
    add(3);
    const s = useEditor.getState();
    s.updateAdjustments({ contrast: 20 });
    s.copySettings();
    s.selectAll();
    useEditor.getState().pasteSettings();
    expect(useEditor.getState().photos.p2.adjustments.contrast).toBe(20);
    const mono = BUILT_IN_PRESETS.find((p) => p.id === 'mono-classic')!;
    useEditor.getState().applyPresetTo(mono);
    const st = useEditor.getState();
    expect(st.photos.p1.adjustments.saturation).toBe(-100);
    expect(st.photos.p1.adjustments.contrast).toBe(22); // replace mode starts from defaults
  });

  it('range-selects with shift and restores file order after import', () => {
    add(5);
    useEditor.getState().toggleSelected('p3', false, true);
    expect(useEditor.getState().selected).toEqual(['p0', 'p1', 'p2', 'p3']);
    useEditor.getState().sortPhotos(['p4', 'p3', 'p2', 'p1', 'p0']);
    expect(useEditor.getState().order).toEqual(['p4', 'p3', 'p2', 'p1', 'p0']);
  });

  it('navigates and removes photos', () => {
    add(3);
    const s = useEditor.getState();
    s.navigate(1);
    expect(useEditor.getState().activeId).toBe('p1');
    useEditor.getState().removePhotos(['p1']);
    expect(useEditor.getState().order).toEqual(['p0', 'p2']);
    expect(useEditor.getState().activeId).toBe('p0');
  });

  it('v0.2: bokeh drags are one undo step; toggles apply to the selection', () => {
    add(3);
    const s = useEditor.getState();
    s.setDepthBokeh({ enabled: true });
    expect(useEditor.getState().photos.p0.adjustments.depthBokeh.enabled).toBe(true);
    s.beginEdit();
    for (let v = 100; v <= 300; v += 20) s.setDepthBokeh({ bokehMultiplier: v }, { live: true });
    const p0 = useEditor.getState().photos.p0;
    expect(p0.adjustments.depthBokeh.bokehMultiplier).toBe(300);
    expect(p0.past).toHaveLength(2);
    useEditor.setState({ selected: ['p1', 'p2'] });
    s.setSkinMultiplier(3);
    s.setStrayHairRemoval({ enabled: true });
    const st = useEditor.getState();
    expect(st.photos.p1.adjustments.skinMultiplier).toBe(3);
    expect(st.photos.p2.adjustments.strayHairRemoval.enabled).toBe(true);
    expect(st.photos.p0.adjustments.skinMultiplier).toBe(1);
  });

  it('v0.2: style sync to all, smart sync and auto retouch', () => {
    add(3);
    const prof = (meanL: number) => ({
      stats: { mean: [meanL, 5, 10] as [number, number, number], std: [20, 5, 5] as [number, number, number] },
      meanL, contrast: 20, warmth: 10, tintAxis: 5, chroma: 15, sharpness: 150, skinLuma: 60, skinTexture: 3,
    });
    const s = useEditor.getState();
    ['p0', 'p1', 'p2'].forEach((id, i) => {
      s.setProfile(id, prof(45 + i * 10));
      s.setLocalStats(id, prof(45 + i * 10).stats);
    });
    s.setStyleReference({ referenceImageId: 'r', referenceName: 'ref.jpg', referenceUrl: null, profile: prof(70), matchedPalette: [] });
    s.setStyleSyncOptions({ syncTargetBatch: true, mode: 'sliders' });
    expect(useEditor.getState().applyStyleSync()).toBe(3);
    const ex = ['p0', 'p1', 'p2'].map((id) => useEditor.getState().photos[id].adjustments.exposure);
    expect(ex[0]).toBeGreaterThan(ex[2]); // darker originals get more exposure
    expect(ex[2]).toBeGreaterThan(0);

    useEditor.getState().resetPhotos(['p0', 'p1', 'p2']);
    useEditor.getState().updateAdjustments({ skinSmoothing: 50 }, ['p1']);
    useEditor.getState().setActive('p1');
    expect(useEditor.getState().smartSync()).toBe(2);
    const after = useEditor.getState().photos;
    expect(after.p0.adjustments.skinSmoothing).toBe(50);
    expect(after.p0.adjustments.exposure).toBeGreaterThan(0); // p0 is darker than the hero
    expect(after.p2.adjustments.exposure).toBeLessThan(0);

    expect(useEditor.getState().autoRetouch(['p0'])).toBe(1);
    expect(useEditor.getState().photos.p0.adjustments.skinSmoothing).toBeGreaterThan(0);
  });
});
