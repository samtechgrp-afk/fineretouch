import { beforeEach, describe, expect, it, vi } from 'vitest';
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

const mask = (id: string) => useEditor.getState().photos[id].adjustments.skinMask;

describe('skin mask in the store', () => {
  it('adds each stroke as one undo step and clears them', () => {
    const s = useEditor.getState();
    s.addMaskStroke({ mode: 'add', size: 0.03, points: [0.1, 0.1, 0.2, 0.2] });
    s.addMaskStroke({ mode: 'erase', size: 0.03, points: [0.5, 0.5] });
    expect(mask('a').strokes.map((x) => x.mode)).toEqual(['add', 'erase']);
    s.undo();
    expect(mask('a').strokes.length).toBe(1);
    s.clearMaskStrokes();
    expect(mask('a').strokes.length).toBe(0);
    s.undo();
    expect(mask('a').strokes.length).toBe(1);
  });

  it('copy / paste and sync carry mask options but not strokes', () => {
    const s = useEditor.getState();
    s.setSkinMask({ includeBodySkin: true }, { ids: ['a'] });
    s.addMaskStroke({ mode: 'erase', size: 0.03, points: [0.5, 0.5] });
    s.copySettings();
    s.pasteSettings(['b']);
    expect(mask('b').includeBodySkin).toBe(true);
    expect(mask('b').strokes).toEqual([]);
    s.syncToSelected();
    expect(mask('b').strokes).toEqual([]);
    expect(mask('a').strokes.length).toBe(1);
  });

  it('saved presets leave strokes behind', () => {
    const s = useEditor.getState();
    s.addMaskStroke({ mode: 'add', size: 0.03, points: [0.5, 0.5] });
    const preset = s.saveCustomPreset('mine')!;
    expect(preset.adjustments.skinMask?.strokes).toEqual([]);
    s.applyPresetTo(preset, ['a']);
    expect(mask('a').strokes.length).toBe(1);
  });
});
