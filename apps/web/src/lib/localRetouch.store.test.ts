import { beforeEach, describe, expect, it, vi } from 'vitest';
import { applyPreset, BUILT_IN_PRESETS } from '@fineretouch/shared';
import { useEditor } from './store';

vi.stubGlobal('URL', { ...URL, revokeObjectURL: vi.fn() });

beforeEach(() => {
  useEditor.setState({ photos: {}, order: [], activeId: null, selected: [], clipboard: null, customPresets: [] });
  useEditor.getState().addPhotos([
    { id: 'a', name: 'a.jpg', width: 100, height: 100, thumbUrl: 'blob:a' },
    { id: 'b', name: 'b.jpg', width: 100, height: 100, thumbUrl: 'blob:b' },
  ]);
  useEditor.getState().setActive('a');
});

const local = (id: string) => useEditor.getState().photos[id].adjustments.localRetouch;

describe('pro tools in the store', () => {
  it('records heal strokes and layer edits as undo steps', () => {
    const s = useEditor.getState();
    s.addHealStroke({ mode: 'heal', size: 0.02, hardness: 50, opacity: 100, offset: [0.1, 0], points: [0.3, 0.3] });
    const id = s.addLayer('dodgeBurn')!;
    expect(useEditor.getState().view.activeLayerId).toBe(id);
    s.addLayerStroke(id, { mode: 'burn', size: 0.02, flow: 20, points: [0.5, 0.5, 0.6, 0.6] });
    s.updateLayer(id, { dodgeBurnStrength: 70, mask: { base: 'skin' } });
    expect(local('a').heal.length).toBe(1);
    expect(local('a').layers[0]).toMatchObject({ kind: 'dodgeBurn', dodgeBurnStrength: 70, mask: { base: 'skin' } });
    expect(local('a').layers[0].mask.strokes[0].flow).toBe(20);
    s.undo();
    expect(local('a').layers[0].dodgeBurnStrength).toBe(50);
    s.undo();
    expect(local('a').layers[0].mask.strokes.length).toBe(0);
    s.undo();
    s.undo();
    expect(local('a')).toEqual({ heal: [], layers: [] });
  });

  it('reorders and removes layers', () => {
    const s = useEditor.getState();
    const a = s.addLayer('frequency')!;
    const b = s.addLayer('correction')!;
    s.moveLayer(b, -1);
    expect(local('a').layers.map((l) => l.id)).toEqual([b, a]);
    s.removeLayer(b);
    expect(local('a').layers.map((l) => l.id)).toEqual([a]);
  });

  it('never copies pro tools by paste, sync, presets or saved presets', () => {
    const s = useEditor.getState();
    s.addHealStroke({ mode: 'clone', size: 0.02, hardness: 50, opacity: 100, offset: [0.1, 0], points: [0.3, 0.3] });
    s.addLayer('frequency');
    s.copySettings();
    s.pasteSettings(['b']);
    expect(local('b')).toEqual({ heal: [], layers: [] });
    s.syncToSelected();
    expect(local('b')).toEqual({ heal: [], layers: [] });
    const p = s.saveCustomPreset('mine')!;
    expect(p.adjustments.localRetouch).toEqual({ heal: [], layers: [] });
    const a = useEditor.getState().photos.a.adjustments;
    expect(applyPreset(a, BUILT_IN_PRESETS[0]).localRetouch).toEqual(a.localRetouch);
  });
});
