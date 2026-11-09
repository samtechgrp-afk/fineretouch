import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { StyleProfile } from '@fineretouch/shared';
import { displayAdjustments, useEditor } from './store';

vi.stubGlobal('URL', { ...URL, revokeObjectURL: vi.fn() });

const profile = (meanL: number): StyleProfile => ({
  stats: { mean: [meanL, 8, 12], std: [20, 6, 8] },
  meanL,
  contrast: 20,
  warmth: 12,
  tintAxis: 8,
  chroma: 20,
  sharpness: 200,
  skinLuma: 62,
  skinTexture: 4,
});

beforeEach(() => {
  useEditor.setState({ photos: {}, order: [], activeId: null, selected: [], clipboard: null, customPresets: [], assistant: { pending: null, batch: false } });
  useEditor.getState().addPhotos([
    { id: 'a', name: 'a.jpg', width: 100, height: 100, thumbUrl: 'blob:a' },
    { id: 'b', name: 'b.jpg', width: 100, height: 100, thumbUrl: 'blob:b' },
  ]);
  useEditor.getState().setActive('a');
});

const photo = (id: string) => useEditor.getState().photos[id];

describe('retouch assistant in the store', () => {
  it('previews a plan without changing the edit, then approves it as one undo step', () => {
    const s = useEditor.getState();
    const before = photo('a').adjustments;
    const plan = s.proposeInstruction('remove acne and reduce redness')!;
    expect(plan.steps.length).toBeGreaterThan(1);
    expect(photo('a').adjustments).toBe(before);
    expect(photo('a').past.length).toBe(0);
    const shown = displayAdjustments(useEditor.getState(), photo('a'));
    expect(shown.acneRemoval).toBeGreaterThan(0);
    // Other photos never show the preview.
    expect(displayAdjustments(useEditor.getState(), photo('b'))).toBe(photo('b').adjustments);

    expect(s.approvePlan()).toBe(1);
    expect(useEditor.getState().assistant.pending).toBeNull();
    expect(photo('a').adjustments.acneRemoval).toBe(shown.acneRemoval);
    expect(photo('a').past.length).toBe(1);
    s.undo();
    expect(photo('a').adjustments).toEqual(before);
  });

  it('applies the user\'s step edits, unticks and strength', () => {
    const s = useEditor.getState();
    s.proposeInstruction('remove acne and reduce redness');
    s.updatePlanStep('slider:acneRemoval', { value: 90 });
    s.updatePlanStep('slider:rednessReduction', { enabled: false });
    s.setPlanStrength(50);
    s.approvePlan();
    expect(photo('a').adjustments.acneRemoval).toBe(45);
    expect(photo('a').adjustments.rednessReduction).toBe(0);
  });

  it('reject discards the plan and leaves no history', () => {
    const s = useEditor.getState();
    s.proposeStyle('beauty');
    s.rejectPlan();
    expect(useEditor.getState().assistant.pending).toBeNull();
    expect(photo('a').past.length).toBe(0);
  });

  it('switching photos discards a pending plan', () => {
    const s = useEditor.getState();
    s.proposeInstruction('remove acne');
    s.setActive('b');
    expect(useEditor.getState().assistant.pending).toBeNull();
  });

  it('rebuilds a style for each selected photo on approval', () => {
    const s = useEditor.getState();
    s.setProfile('a', profile(40));
    s.setProfile('b', profile(70));
    useEditor.setState({ selected: ['a', 'b'] });
    s.proposeStyle('headshot');
    s.setPlanBatch(true);
    expect(s.approvePlan()).toBe(2);
    expect(photo('a').adjustments.exposure).toBeGreaterThan(0);
    expect(photo('b').adjustments.exposure).toBeLessThan(0);
    expect(photo('a').adjustments.clothing.wrinkleStrength).toBe(50);
    // No face was detected in these test photos, so the face-only stray hair step stays off.
    expect(photo('a').adjustments.strayHairRemoval.enabled).toBe(false);
    expect(photo('b').past.length).toBe(1);
  });

  it('needs measurements for recommendations', () => {
    const s = useEditor.getState();
    expect(s.proposeRecommendation()).toBeNull();
    s.setProfile('a', profile(40));
    const plan = s.proposeRecommendation()!;
    expect(plan.steps.find((st) => st.id === 'slider:exposure')?.reason).toMatch(/dark/);
  });
});
