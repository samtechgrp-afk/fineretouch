import { describe, expect, it } from 'vitest';
import { DEFAULT_ADJUSTMENTS, sanitizeAdjustments, type Adjustments } from './index.js';
import {
  applyPlan,
  effectiveSteps,
  planForPhoto,
  planFromInstruction,
  recommendPlan,
  RETOUCH_STYLES,
  stylePlan,
  type EditPlan,
} from './assistant.js';
import type { StyleProfile } from './style.js';

const base = (): Adjustments => sanitizeAdjustments(DEFAULT_ADJUSTMENTS);
const face = { hasFace: true };
const profile = (over: Partial<StyleProfile> = {}): StyleProfile => ({
  stats: { mean: [55, 8, 12], std: [20, 6, 8] },
  meanL: 55,
  contrast: 20,
  warmth: 12,
  tintAxis: 8,
  chroma: 20,
  sharpness: 200,
  skinLuma: 62,
  skinTexture: 3,
  ...over,
});
const step = (p: EditPlan, id: string) => p.steps.find((s) => s.id === id);

describe('planFromInstruction', () => {
  it('maps retouching vocabulary to the matching tools', () => {
    const p = planFromInstruction('remove acne, reduce redness and fix the dark circles', base(), face);
    expect(step(p, 'slider:acneRemoval')?.value).toBeGreaterThan(0);
    expect(step(p, 'slider:rednessReduction')?.value).toBeGreaterThan(0);
    expect(step(p, 'slider:darkCircles')?.value).toBeGreaterThan(0);
    expect(p.unmatched).toEqual([]);
    for (const s of p.steps) expect(s.reason).toMatch(/you said/);
  });

  it('keeps specific phrases from being matched twice', () => {
    const p = planFromInstruction('soften under-eye lines', base(), face);
    expect(step(p, 'slider:underEyeTexture')).toBeDefined();
    expect(step(p, 'slider:darkCircles')).toBeUndefined();
    expect(step(p, 'slider:eyeBrighten')).toBeUndefined();
  });

  it('scales amounts with intensity words', () => {
    const soft = planFromInstruction('slightly smooth the skin', base(), face);
    const strong = planFromInstruction('smooth the skin a lot', base(), face);
    const plain = planFromInstruction('smooth the skin', base(), face);
    const v = (p: EditPlan) => step(p, 'slider:skinSmoothing')!.value as number;
    expect(v(soft)).toBeLessThan(v(plain));
    expect(v(strong)).toBeGreaterThan(v(plain));
  });

  it('uses an explicit number and a 2x/3x multiplier', () => {
    const p = planFromInstruction('blemish removal to 80, acne 3x', base(), face);
    expect(step(p, 'slider:blemishRemoval')?.value).toBe(80);
    expect(step(p, 'multiplier:acneRemoval')?.value).toBe(3);
  });

  it('adds texture preservation whenever it smooths', () => {
    const p = planFromInstruction('smooth the skin', base(), face);
    expect((step(p, 'slider:skinTexture')?.value as number) ?? base().skinTexture).toBeGreaterThanOrEqual(60);
  });

  it('understands keep and remove for freckles and moles', () => {
    const keep = planFromInstruction("remove blemishes but keep the freckles and don't touch moles", { ...base(), keepFreckles: false, protectBeautyMarks: false }, face);
    expect(step(keep, 'toggle:keepFreckles')?.value).toBe(true);
    expect(step(keep, 'toggle:protectBeautyMarks')?.value).toBe(true);
    expect(step(keep, 'slider:blemishRemoval')?.value).toBeGreaterThan(0);
    const remove = planFromInstruction('remove freckles', { ...base(), keepFreckles: true }, face);
    expect(step(remove, 'toggle:keepFreckles')?.value).toBe(false);
  });

  it('tells weakening a tool apart from removing a problem', () => {
    const adj = { ...base(), skinSmoothing: 40, rednessReduction: 40 };
    expect(step(planFromInstruction('less smoothing', adj, face), 'slider:skinSmoothing')?.value).toBe(20);
    expect(step(planFromInstruction('too smooth', adj, face), 'slider:skinSmoothing')?.value).toBe(20);
    expect(step(planFromInstruction('tone down the redness', adj, face), 'slider:rednessReduction')?.value).toBeGreaterThan(40);
    expect(step(planFromInstruction('tone down the redness reduction', adj, face), 'slider:rednessReduction')?.value).toBe(20);
    expect(step(planFromInstruction('smoother skin but natural', adj, face), 'slider:skinSmoothing')?.value).toBeGreaterThan(40);
  });

  it('keeps multi-word phrases together and ignores pure modifiers', () => {
    const p = planFromInstruction('make it black and white, keep it natural please', base(), face);
    expect(step(p, 'slider:saturation')?.value).toBe(-100);
    expect(p.unmatched).toEqual([]);
  });

  it('does not read a verb on another subject as a light change', () => {
    const p = planFromInstruction('brighten the eyes slightly', base(), face);
    expect(step(p, 'slider:eyeBrighten')).toBeDefined();
    expect(step(p, 'slider:exposure')).toBeUndefined();
    const q = planFromInstruction('brighten the eyes, then make the whole photo brighter', base(), face);
    expect(step(q, 'slider:exposure')).toBeDefined();
  });

  it('turns a tool off on a negation', () => {
    const p = planFromInstruction('no sharpening', { ...base(), sharpen: 40 }, face);
    expect(step(p, 'slider:sharpen')?.value).toBe(0);
  });

  it('reads light and colour direction from the words', () => {
    const p = planFromInstruction('make it a bit brighter and warmer', base(), face);
    expect(step(p, 'slider:exposure')?.value).toBeGreaterThan(0);
    expect(step(p, 'slider:temperature')?.value).toBeGreaterThan(0);
    const q = planFromInstruction('too bright, cooler please', base(), face);
    expect(step(q, 'slider:exposure')?.value).toBeLessThan(0);
    expect(step(q, 'slider:temperature')?.value).toBeLessThan(0);
  });

  it('declines identity changes and reports what it did not understand', () => {
    const p = planFromInstruction('slim the face, add a unicorn and remove acne', base(), face);
    expect(p.declined).toHaveLength(1);
    expect(p.unmatched).toEqual(['add a unicorn']);
    expect(step(p, 'slider:acneRemoval')).toBeDefined();
    expect(p.steps.some((s) => s.label.toLowerCase().includes('slim'))).toBe(false);
  });

  it('switches off face-only steps when no face was found', () => {
    const p = planFromInstruction('whiten teeth and remove acne', base(), { hasFace: false });
    expect(step(p, 'slider:teethWhiten')?.enabled).toBe(false);
    expect(step(p, 'slider:acneRemoval')?.enabled).toBe(true);
  });

  it('turns a general request into a natural full pass', () => {
    const p = planFromInstruction('retouch this portrait', base(), face);
    expect(step(p, 'slider:blemishRemoval')).toBeDefined();
    expect(step(p, 'slider:skinTexture')?.value).toBeGreaterThanOrEqual(70);
  });

  it('drives the stray hair and clothing tools', () => {
    const p = planFromInstruction('clean up flyaways and iron the shirt', base(), face);
    const out = applyPlan(base(), p);
    expect(out.strayHairRemoval.enabled).toBe(true);
    expect(out.clothing.enabled).toBe(true);
    expect(out.clothing.wrinkleStrength).toBeGreaterThan(0);
  });
});

describe('applyPlan', () => {
  it('applies ticked steps only and scales by strength', () => {
    const p = planFromInstruction('remove acne and reduce redness', base(), face);
    const acne = step(p, 'slider:acneRemoval')!.value as number;
    const full = applyPlan(base(), p);
    expect(full.acneRemoval).toBe(acne);
    const half = applyPlan(base(), { ...p, strength: 50 });
    expect(half.acneRemoval).toBe(Math.round(acne / 2));
    const off = applyPlan(base(), { ...p, steps: p.steps.map((s) => (s.id === 'slider:acneRemoval' ? { ...s, enabled: false } : s)) });
    expect(off.acneRemoval).toBe(base().acneRemoval);
    expect(off.rednessReduction).toBe(full.rednessReduction);
    expect(applyPlan(base(), { ...p, strength: 0 })).toEqual(base());
  });

  it('never touches local strokes or the skin mask', () => {
    const adj = base();
    const out = applyPlan(adj, stylePlan('beauty', profile(), adj, face));
    expect(out.localRetouch).toBe(adj.localRetouch);
    expect(out.skinMask).toBe(adj.skinMask);
    expect(sanitizeAdjustments(out)).toEqual(out);
  });

  it('effectiveSteps drops what is already set', () => {
    const p = planFromInstruction('remove acne', base(), face);
    expect(effectiveSteps(applyPlan(base(), p), p)).toEqual([]);
  });
});

describe('recommendPlan', () => {
  it('explains each change with the measurement behind it', () => {
    const p = recommendPlan(profile({ meanL: 40, contrast: 12 }), base(), face);
    const exp = step(p, 'slider:exposure')!;
    expect(exp.value).toBeGreaterThan(0);
    expect(exp.reason).toMatch(/L\* 40/);
    expect(step(p, 'slider:contrast')!.reason).toMatch(/flat/);
  });

  it('caps brightening so skin keeps its real tone', () => {
    const dark = profile({ meanL: 33, skinLuma: 40 });
    const free = recommendPlan({ ...dark, skinLuma: null }, base(), face);
    const p = recommendPlan(dark, base(), face);
    const e = step(p, 'slider:exposure')!;
    expect(e.value as number).toBeLessThan(step(free, 'slider:exposure')!.value as number);
    expect(e.reason).toMatch(/real skin tone/);
    expect(step(stylePlan('headshot', dark, base(), face), 'slider:exposure')!.value).toBe(e.value);
  });

  it('asks for nothing when the photo already matches', () => {
    const p0 = recommendPlan(profile(), base(), face);
    const p1 = recommendPlan(profile(), applyPlan(base(), p0), face);
    expect(p1.steps).toEqual([]);
  });
});

describe('stylePlan', () => {
  it('builds every style without hard-coding the skin amounts', () => {
    for (const s of RETOUCH_STYLES) {
      const smooth = stylePlan(s.id, profile({ skinTexture: 2.2 }), base(), face);
      const rough = stylePlan(s.id, profile({ skinTexture: 6 }), base(), face);
      const v = (p: EditPlan) => step(p, 'slider:skinSmoothing')!.value as number;
      expect(v(rough)).toBeGreaterThan(v(smooth));
      expect(smooth.title).toContain(s.name);
    }
  });

  it('keeps natural looks gentler than beauty', () => {
    const n = applyPlan(base(), stylePlan('natural', profile(), base(), face));
    const b = applyPlan(base(), stylePlan('beauty', profile(), base(), face));
    expect(n.skinSmoothing).toBeLessThan(b.skinSmoothing);
    expect(n.skinTexture).toBeGreaterThan(b.skinTexture);
    expect(n.keepFreckles).toBe(true);
  });

  it('is rebuilt per photo and keeps unticked steps and strength', () => {
    const p = stylePlan('headshot', profile({ meanL: 40 }), base(), face);
    const edited = { ...p, strength: 70, steps: p.steps.map((s) => (s.id === 'feature:clothingIron' ? { ...s, enabled: false } : s)) };
    const other = planForPhoto(edited, profile({ meanL: 70 }), base(), face);
    expect(other.strength).toBe(70);
    expect(step(other, 'feature:clothingIron')?.enabled).toBe(false);
    expect(step(other, 'slider:exposure')!.value as number).toBeLessThan(step(p, 'slider:exposure')!.value as number);
  });
});
