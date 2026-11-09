/**
 * Retouch assistant (v0.11, Phase 6): plain-English instructions, recommendations from a photo's
 * own measurements and per-photo retouching styles, all expressed as one structured EditPlan the
 * user reviews before anything changes.
 *
 * Everything here runs offline in the app. The instruction parser is a deterministic phrase
 * matcher with a fixed vocabulary of retouching terms (no language model and no network): it
 * reports the parts it did not understand instead of guessing, and it declines requests that
 * would change a person's face shape or identity.
 */

import type { Adjustments, AdjustmentKey } from './index.js';
import { autoRetouch, exposureForLightness, type StyleProfile } from './style.js';
import type { EffectId, EffectMultiplier } from './effects.js';

export type PlanTarget =
  | { kind: 'slider'; key: AdjustmentKey }
  | { kind: 'toggle'; key: 'keepFreckles' | 'protectBeautyMarks' | 'skinToneGuard' }
  | { kind: 'multiplier'; effect: EffectId }
  /** Stray hair, clothing iron and background blur: strength 0..100, 0 switches the tool off. */
  | { kind: 'feature'; feature: 'strayHair' | 'clothingIron' | 'bokeh' };

export interface PlanStep {
  /** Stable id of the target (one step per target), e.g. 'slider:skinSmoothing'. */
  id: string;
  target: PlanTarget;
  label: string;
  /** Recommended value; the user can change it before approving. */
  value: number | boolean;
  reason: string;
  /** Unticked steps are not applied. */
  enabled: boolean;
  /** Has no effect without a detected face. */
  needsFace?: boolean;
}

export type PlanSource = 'instruction' | 'recommendation' | 'style';

export interface EditPlan {
  source: PlanSource;
  title: string;
  /** The user's words (instruction plans). */
  instruction?: string;
  /** Retouching style id (style plans), so the plan can be rebuilt for other photos. */
  style?: RetouchStyleId;
  steps: PlanStep[];
  /** 0..100: share of every numeric change applied (100 = the values shown). */
  strength: number;
  /** Parts of the instruction that matched nothing the app can do. */
  unmatched: string[];
  /** Requests the assistant will not carry out, with the reason. */
  declined: string[];
  /** Other remarks (e.g. a tool that only covers part of the request). */
  notes: string[];
}

// ---------------------------------------------------------------------------------------------
// Targets
// ---------------------------------------------------------------------------------------------

export function targetId(t: PlanTarget): string {
  if (t.kind === 'slider' || t.kind === 'toggle') return `${t.kind}:${t.key}`;
  if (t.kind === 'multiplier') return `multiplier:${t.effect}`;
  return `feature:${t.feature}`;
}

export function currentValue(adj: Adjustments, t: PlanTarget): number | boolean {
  switch (t.kind) {
    case 'slider':
      return adj[t.key];
    case 'toggle':
      return adj[t.key];
    case 'multiplier':
      return adj.effects[t.effect].multiplier;
    case 'feature':
      if (t.feature === 'strayHair') return adj.strayHairRemoval.enabled ? adj.strayHairRemoval.strength : 0;
      if (t.feature === 'clothingIron') return adj.clothing.enabled ? adj.clothing.wrinkleStrength : 0;
      return adj.depthBokeh.enabled ? adj.depthBokeh.blurRadius : 0;
  }
}

/** Range of a numeric target (for the review UI and clamping). */
export function targetRange(t: PlanTarget): [number, number] {
  if (t.kind === 'multiplier') return [1, 3];
  if (t.kind === 'slider' && BIPOLAR.has(t.key)) return [-100, 100];
  return [0, 100];
}

const BIPOLAR = new Set<string>(['exposure', 'contrast', 'highlights', 'shadows', 'temperature', 'tint', 'saturation', 'vibrance', 'vignette']);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function setValue(adj: Adjustments, t: PlanTarget, v: number | boolean): Adjustments {
  switch (t.kind) {
    case 'slider':
      return { ...adj, [t.key]: v as number };
    case 'toggle':
      return { ...adj, [t.key]: Boolean(v) };
    case 'multiplier':
      return { ...adj, effects: { ...adj.effects, [t.effect]: { ...adj.effects[t.effect], multiplier: clamp(Math.round(v as number), 1, 3) as EffectMultiplier } } };
    case 'feature': {
      const n = v as number;
      if (t.feature === 'strayHair') return { ...adj, strayHairRemoval: { enabled: n > 0, strength: n > 0 ? n : adj.strayHairRemoval.strength } };
      if (t.feature === 'clothingIron') return { ...adj, clothing: { ...adj.clothing, enabled: n > 0 || adj.clothing.hueShift !== 0 || adj.clothing.saturation !== 0, wrinkleStrength: n } };
      return { ...adj, depthBokeh: { ...adj.depthBokeh, enabled: n > 0, blurRadius: n > 0 ? n : adj.depthBokeh.blurRadius } };
    }
  }
}

/**
 * The edit with the plan applied: each ticked step moves its target from the current value
 * toward the step's value by `strength` percent (switches and multipliers apply when strength
 * is above zero). The caller sanitises the result.
 */
export function applyPlan(adj: Adjustments, plan: EditPlan): Adjustments {
  const k = clamp(plan.strength, 0, 100) / 100;
  let out = adj;
  for (const s of plan.steps) {
    if (!s.enabled) continue;
    const cur = currentValue(out, s.target);
    if (typeof s.value === 'boolean' || s.target.kind === 'multiplier') {
      if (k > 0) out = setValue(out, s.target, s.value);
      continue;
    }
    const [lo, hi] = targetRange(s.target);
    const next = Math.round(clamp((cur as number) + ((s.value as number) - (cur as number)) * k, lo, hi));
    out = setValue(out, s.target, next);
  }
  return out;
}

/** Steps that would actually change something on this edit. */
export function effectiveSteps(adj: Adjustments, plan: EditPlan): PlanStep[] {
  return plan.steps.filter((s) => s.enabled && currentValue(adj, s.target) !== s.value);
}

// ---------------------------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------------------------

type Topic = {
  name: string;
  re: RegExp;
  target: PlanTarget;
  label: string;
  /** Value for a plain "remove / fix / reduce X" at normal intensity. */
  base: number;
  /** "keep / protect X" means: raise this target (texture) or switch it on (toggles). */
  protect?: number | boolean;
  /** What a "remove X" means for a protect-type topic (freckles, moles: switch protection off). */
  remove?: number | boolean;
  needsFace?: boolean;
  /** Effect whose 1x/2x/3x a "2x" in the same clause sets. */
  effect?: EffectId;
  note?: string;
  /** Its words double as verbs ("brighten the eyes"): skipped when an earlier topic in the same clause took the subject. */
  verb?: boolean;
};

const slider = (key: AdjustmentKey): PlanTarget => ({ kind: 'slider', key });

/** Most specific first: a phrase consumed by one topic is not matched again by a later one. */
const TOPICS: Topic[] = [
  { name: 'clothing', re: /\b(wrinkl\w*|creas\w*) (on |in )?(the |his |her |their )?(shirt|clothes|clothing|suit|dress|jacket|blouse|fabric)\b|\b(shirt|clothes|clothing|suit|dress|jacket|blouse|fabric) (wrinkl\w*|creas\w*)\b|\biron\w*\b/, target: { kind: 'feature', feature: 'clothingIron' }, label: 'Clothing wrinkles', base: 60, effect: 'clothingIron' },
  { name: 'underEyeLines', re: /\bunder[- ]?eye (lines?|wrinkles?|texture|creases?)\b|\bcrepey\b|\bfine lines? under\b/, target: slider('underEyeTexture'), label: 'Under-eye texture', base: 45, needsFace: true, effect: 'underEyeTexture' },
  { name: 'darkCircles', re: /\bdark circles?\b|\b(eye ?bags?|bags under (the |his |her |their )?eyes)\b|\bunder[- ]?eye(s)? ?(circles?|darkness|shadows?|area)?\b|\btired eyes\b/, target: slider('darkCircles'), label: 'Under-eye correction', base: 50, needsFace: true, effect: 'darkCircles' },
  { name: 'wrinkles', re: /\b(wrinkles?|fine lines?|crow'?s feet|forehead lines?|laugh lines?)\b/, target: slider('underEyeTexture'), label: 'Under-eye texture', base: 40, needsFace: true, effect: 'underEyeTexture', note: 'Line softening covers the under-eye area only for now; crow\'s feet and forehead lines are not supported yet.' },
  { name: 'teethStains', re: /\b(teeth|tooth) (stains?|discolou?r\w*)\b|\b(stained|yellow) teeth\b|\bstains? on (the |his |her |their )?teeth\b/, target: slider('teethStainRemoval'), label: 'Teeth stains', base: 50, needsFace: true, effect: 'teethStainRemoval' },
  { name: 'teeth', re: /\b(teeth|tooth|smile)\b/, target: slider('teethWhiten'), label: 'Whiten teeth', base: 35, needsFace: true, effect: 'teethWhiten' },
  { name: 'glare', re: /\b(glare|reflections?|glasses|spectacles|lens(es)?)\b/, target: slider('glareReduction'), label: 'Glasses glare', base: 60, needsFace: true, effect: 'glareReduction' },
  { name: 'darkSpots', re: /\b(dark (spots?|marks?|patches?)|sun ?spots?|age spots?|liver spots?|hyper ?pigment\w*|pigmentation|melasma|discolou?ration)\b/, target: slider('darkSpotRemoval'), label: 'Dark spot removal', base: 60, effect: 'darkSpotRemoval' },
  { name: 'acne', re: /\b(acne|pimples?|zits?|breakouts?|spots? from acne)\b/, target: slider('acneRemoval'), label: 'Acne removal', base: 70, effect: 'acneRemoval' },
  { name: 'freckles', re: /\bfreckl\w*\b/, target: { kind: 'toggle', key: 'keepFreckles' }, label: 'Keep freckles', base: 0, protect: true, remove: false },
  { name: 'moles', re: /\b(moles?|beauty marks?|birth ?marks?)\b/, target: { kind: 'toggle', key: 'protectBeautyMarks' }, label: 'Protect beauty marks', base: 0, protect: true, remove: false },
  { name: 'redness', re: /\b(redness|red(dish)? (cheeks?|nose|skin|patches?|areas?)|rosacea|flush\w*|blotch\w*|irritat\w*)\b/, target: slider('rednessReduction'), label: 'Redness reduction', base: 55, effect: 'rednessReduction' },
  { name: 'noise', re: /\b(noise|noisy|grain\w*|high iso)\b/, target: slider('skinNoise'), label: 'Skin noise reduction', base: 50, effect: 'skinNoise' },
  { name: 'tone', re: /\b(even(ing)? (out )?(the )?(skin ?)?tone|uneven (skin ?)?(tone|colou?r)|skin ?tone|blotchy|patchy skin)\b/, target: slider('skinToneUnify'), label: 'Skin tone correction', base: 40, effect: 'skinToneUnify' },
  { name: 'blemish', re: /\b(blemish\w*|spots?|imperfections?|marks?|scars?)\b/, target: slider('blemishRemoval'), label: 'Blemish removal', base: 60, effect: 'blemishRemoval' },
  { name: 'texture', re: /\b(texture|pores?|natural skin|real skin|plastic|waxy|fake skin)\b/, target: slider('skinTexture'), label: 'Texture preservation', base: 70, protect: 85 },
  { name: 'smoothing', re: /\b(smooth\w*|soft(en|er)? (the )?skin|skin soft\w*|airbrush\w*|flawless|porcelain|retouch (the )?skin|clean (up )?(the )?skin|skin clean ?up|clear(er)? skin)\b/, target: slider('skinSmoothing'), label: 'Skin smoothing', base: 45, effect: 'skinSmoothing' },
  { name: 'strayHair', re: /\b(stray|fly-?aways?|loose hairs?|frizz\w*|flyaway hairs?|baby hairs?)\b/, target: { kind: 'feature', feature: 'strayHair' }, label: 'Stray hair cleanup', base: 60, needsFace: true, effect: 'strayHair' },
  { name: 'bokeh', re: /\b(blur (the |out the )?background|background blur\w*|blurr?(y|ed)? background|bokeh|depth of field|shallow focus)\b/, target: { kind: 'feature', feature: 'bokeh' }, label: 'Background blur', base: 50 },
  { name: 'eyes', re: /\b(eyes?|iris(es)?|catchlights?)\b/, target: slider('eyeBrighten'), label: 'Brighten eyes', base: 35, needsFace: true, effect: 'eyeBrighten' },
  { name: 'sharpen', re: /\b(sharp\w*|crisp\w*|blurry|soft focus|details?)\b/, target: slider('sharpen'), label: 'Sharpen', base: 30, effect: 'sharpen' },
  { name: 'vignette', re: /\bvignett\w*\b/, target: slider('vignette'), label: 'Vignette', base: -20 },
  { name: 'shadows', re: /\b(shadows|shadow detail)\b/, target: slider('shadows'), label: 'Shadows', base: 20 },
  { name: 'highlights', re: /\b(highlights?|blown|bright spots)\b/, target: slider('highlights'), label: 'Highlights', base: -20 },
  { name: 'contrast', re: /\b(contrast\w*|punch\w*|flat|washed[- ]out|muddy)\b/, target: slider('contrast'), label: 'Contrast', base: 20 },
  { name: 'warmth', re: /\b(warm\w*|cool\w*|cold|too (yellow|orange|blue)|white balance|golden)\b/, target: slider('temperature'), label: 'Temperature', base: 20 },
  { name: 'tint', re: /\b(too (green|magenta|pink)|green(ish)? (cast|tint)|magenta (cast|tint)|tint)\b/, target: slider('tint'), label: 'Tint', base: 15 },
  { name: 'mono', re: /\b(black (and|&) white|b ?& ?w|monochrome|greyscale|grayscale)\b/, target: slider('saturation'), label: 'Saturation', base: -100 },
  { name: 'saturation', re: /\b(saturat\w*|desaturat\w*|colou?rful|vivid|vibrant|muted|dull colou?rs?|colou?rs? pop)\b/, target: slider('vibrance'), label: 'Vibrance', base: 20, verb: true },
  { name: 'exposure', re: /\b(brighter|brighten|lighter|lighten|darker|darken|exposure|under-?exposed|over-?exposed|too (dark|bright|light)|dim)\b/, target: slider('exposure'), label: 'Exposure', base: 20, verb: true },
];

/** Requests that would change a person's identity: declined on purpose. */
const IDENTITY = /\b(slim\w*|thinn?er|skinn(y|ier)|lose weight|weight loss|nose job|smaller nose|bigger (eyes|lips)|fuller lips|reshape|liquify|change (the |his |her |their )?(face|nose|jaw|chin|eye colou?r|skin colou?r|race|ethnicity)|lighten (the |his |her |their )?skin(?! tone)|skin lightening|whiten (the |his |her |their )?skin|make (him|her|them|me) (look )?(younger|older|thinner|prettier)|face ?swap|jawline|v-?line|cheekbones?)\b/;

const NEGATE = /\b(don'?t|do not|never|no|not|without|avoid|leave|keep|preserve|protect|retain|maintain)\b/;
const REMOVE = /\b(remove|removing|get rid|eliminate|erase|clear|clean|fix|hide|cover|delete|take (out|away)|reduce|reducing|soften|tone down|minimi[sz]e|less)\b/;
const DECREASE = /\b(less|lower|reduce|weaker|decrease|flatten|tone (it )?down|too (much|strong|smooth|soft|bright|warm|saturated|sharp)|back off|dial (it )?(back|down)|cooler|cool (it )?down|darker|darken|muted|desaturat\w*|flatter|softer contrast)\b/;
const INCREASE = /\b(more|increase|stronger|boost|raise|higher|extra|add|brighter|brighten|lighter|lighten|warmer|warm (it )?up|punchier|vivid|vibrant|pop|crisper|sharper)\b/;
/** Topics that name a problem (acne, redness): "less acne" means more of the tool. */
const PROBLEMS = new Set(['clothing', 'underEyeLines', 'darkCircles', 'wrinkles', 'teethStains', 'glare', 'darkSpots', 'acne', 'redness', 'noise', 'tone', 'blemish', 'strayHair']);
/** The tool itself is named ("acne removal", "the redness slider"). */
const TOOL_WORD = /\b(removal|reduction|correction|clean ?up|effect|tool|slider|setting)\b/;
const LESS_TOOL = /\b(less|lower|weaker|decrease|tone (it )?down|too (much|strong|smooth|soft|white|bright|sharp|blurr?y)|back off|dial (it )?(back|down)|over-?done|overcooked)\b/;
const MORE_TOOL = /\b(more|increase|stronger|boost|raise|higher|extra|smoother|sharper|crisper|whiter|brighter|softer)\b/;
/** Words that only set the feel of the request; a clause made of them alone is not "unmatched". */
const MODIFIER_ONLY = /^(?:(?:but|keep (?:it|things)|make (?:it|them)|look|looking|very|really|please|thanks?(?: you)?|so|it|still|stays?|just|overall|natural|subtle|realistic|soft|light|minimal|glam\w*|dramatic|strong|heavy|professional(?:ly)?|clean)\s*)+$/;

const SOFT =/\b(slight(ly)?|a (little )?bit|a touch|a little|subtle|subtly|gentl[ey]|light(ly)?|mild(ly)?|barely|just a)\b/;
const STRONG = /\b(very|strong(ly)?|lots?|heav(y|ily)|a lot|really|completely|totally|fully|much|all|every)\b/;
const MULT = /\b(2x|2 ?times|double|twice|3x|3 ?times|triple|max(imum)?)\b/;

/** Overall feel of the whole instruction (applies to every topic). */
function overallScale(text: string): number {
  if (/\b(natural|subtle|light|soft touch|realistic|minimal|barely)\b/.test(text)) return 0.75;
  if (/\b(glam\w*|flawless|magazine|beauty retouch|heavy|strong|dramatic|porcelain)\b/.test(text)) return 1.3;
  return 1;
}

function clausesOf(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[“”"]/g, '')
    .replace(/[‘’]/g, "'")
    .replace(/\bblack (and|&) white\b/g, 'monochrome')
    .replace(/\bno more\b/g, 'remove')
    .split(/[.;!?\n]+|,|\b(?:and then|then|also|plus|but|and)\b/)
    .map((c) => c.trim())
    .filter((c) => c.length > 1);
}

function explicitNumber(clause: string): number | null {
  const m = clause.match(/\b(?:to|at|by|=)\s*(-?\d{1,3})\s*%?|\b(-?\d{1,3})\s*(?:%|percent)/);
  if (!m) return null;
  return Number(m[1] ?? m[2]);
}

function multiplierOf(clause: string): EffectMultiplier | null {
  const m = clause.match(MULT);
  if (!m) return null;
  return /3|triple|max/.test(m[1]) ? 3 : 2;
}

// ---------------------------------------------------------------------------------------------
// Instruction -> plan
// ---------------------------------------------------------------------------------------------

export interface PlanContext {
  hasFace: boolean;
}

/** Turn a plain-English request into an edit plan for this edit. Nothing is applied yet. */
export function planFromInstruction(instruction: string, adj: Adjustments, ctx: PlanContext): EditPlan {
  const text = instruction.trim().slice(0, 1000);
  const plan: EditPlan = { source: 'instruction', title: 'Your request', instruction: text, steps: [], strength: 100, unmatched: [], declined: [], notes: [] };
  const scale = overallScale(text.toLowerCase());
  const steps = new Map<string, PlanStep>();
  const add = (s: Omit<PlanStep, 'id' | 'enabled'>) => steps.set(targetId(s.target), { ...s, id: targetId(s.target), enabled: true });

  for (const clause of clausesOf(text)) {
    const idMatch = clause.match(IDENTITY);
    if (idMatch) {
      plan.declined.push(`"${idMatch[0]}": FineRetouch never changes face shape, body shape or skin colour, so a person still looks like themselves.`);
      continue;
    }
    let rest = ` ${clause} `;
    let matched = false;
    for (const t of TOPICS) {
      const m = rest.match(t.re);
      if (!m) continue;
      if (t.verb && matched && !/\b(photo|image|picture|overall|everything|whole)\b/.test(clause)) continue;
      matched = true;
      rest = rest.replace(m[0], ' ');
      const r = buildStep(t, clause, adj, scale);
      if (!r) continue;
      if (r.step) add(r.step);
      if (r.mult) add(r.mult);
      if (t.note && !plan.notes.includes(t.note)) plan.notes.push(t.note);
    }
    if (!matched) {
      if (/\b(retouch|edit|fix|improve|enhance|clean|touch up|make (it|this) (look )?(better|good|nice|professional))\b/.test(clause)) {
        // A general "retouch this" asks for a full but natural skin and light pass.
        for (const s of genericPass(adj, ctx, scale)) add(s);
      } else if (!MODIFIER_ONLY.test(clause)) {
        plan.unmatched.push(clause);
      }
    }
  }

  // Natural texture rule: smoothing never goes out without texture preservation.
  const smooth = steps.get('slider:skinSmoothing');
  if (smooth && (smooth.value as number) > 0 && !steps.has('slider:skinTexture') && adj.skinTexture < 60) {
    add({ target: slider('skinTexture'), label: 'Texture preservation', value: 65, reason: 'Keeps pores and fine skin texture under the smoothing.' });
  }

  plan.steps = [...steps.values()].filter((s) => currentValue(adj, s.target) !== s.value);
  for (const s of plan.steps) {
    if (s.needsFace && !ctx.hasFace) {
      s.enabled = false;
      s.reason += ' No face was found in this photo, so this has no effect here.';
    }
  }
  if (!plan.steps.length && !plan.declined.length && !plan.unmatched.length) plan.notes.push('Everything you asked for is already set on this photo.');
  plan.title = plan.steps.length ? `Your request: ${plan.steps.length} change${plan.steps.length === 1 ? '' : 's'}` : 'Your request';
  return plan;
}

function buildStep(t: Topic, clause: string, adj: Adjustments, scale: number): { step?: Omit<PlanStep, 'id' | 'enabled'>; mult?: Omit<PlanStep, 'id' | 'enabled'> } | null {
  const cur = currentValue(adj, t.target);
  const negated = NEGATE.test(clause);
  const decrease = DECREASE.test(clause);
  const increase = INCREASE.test(clause);
  const removes = REMOVE.test(clause);
  const intensity = (SOFT.test(clause) ? 0.6 : STRONG.test(clause) ? 1.35 : 1) * scale;
  const n = explicitNumber(clause);
  const [lo, hi] = targetRange(t.target);
  const quote = `"${clause}"`;

  // Protect-type topics: freckles, moles (toggles) and texture.
  if (t.protect !== undefined) {
    const keep = /\b(keep|preserve|protect|retain|maintain|leave|natural|real)\b/.test(clause) || (negated && !removes) || /\b(plastic|waxy|fake)\b/.test(clause);
    if (t.target.kind === 'toggle') {
      const value = keep && !/\b(remove|get rid|erase|delete)\b/.test(clause) ? (t.protect as boolean) : (t.remove as boolean);
      const step: Omit<PlanStep, 'id' | 'enabled'> = { target: t.target, label: t.label, value, reason: `${value ? 'Left untouched' : 'Treated as spots'}: you said ${quote}.` };
      if (t.name === 'freckles' && value === false) return { step: { ...step } };
      return { step };
    }
    // Texture: keep / more texture raises preservation, "less texture" lowers it.
    const value = n !== null ? clamp(n, lo, hi) : keep || increase ? Math.max(cur as number, t.protect as number) : decrease || removes ? Math.round((cur as number) * 0.6) : (t.protect as number);
    return { step: { target: t.target, label: t.label, value, reason: `${value >= (cur as number) ? 'More pores and fine texture kept' : 'Less fine texture kept'}: you said ${quote}.` } };
  }

  const bipolar = lo < 0;
  let value: number;
  let why: string;
  if (n !== null) {
    value = clamp(n, lo, hi);
    why = `set to ${value} as you asked`;
  } else if (bipolar) {
    // Light and colour: direction comes from the words; amount from the intensity.
    let dir = t.base >= 0 ? 1 : -1;
    if (t.name === 'warmth') dir = /\b(cool\w*|cold|too (yellow|orange))\b/.test(clause) ? -1 : 1;
    else if (t.name === 'exposure') dir = /\b(darker|darken|over-?exposed|too (bright|light))\b/.test(clause) ? -1 : 1;
    else if (t.name === 'tint') dir = /\btoo (magenta|pink)|magenta\b/.test(clause) ? -1 : 1;
    else if (t.name === 'mono') {
      value = negated ? 0 : -100;
      return { step: { target: t.target, label: t.label, value, reason: `${value === -100 ? 'Black and white' : 'Colour kept'}: you said ${quote}.` } };
    } else if (t.name === 'highlights') dir = /\b(brighter|boost|more)\b/.test(clause) ? 1 : -1;
    else if (t.name === 'vignette') dir = /\b(white|light|remove|no|less)\b/.test(clause) && !/\bdark\b/.test(clause) ? 0 : -1;
    else if (decrease && !increase) dir = -1;
    if (negated && t.name !== 'exposure' && t.name !== 'warmth') return { step: { target: t.target, label: t.label, value: 0, reason: `Left neutral: you said ${quote}.` } };
    if (dir === 0) {
      value = 0;
      why = 'removed';
    } else {
      value = Math.round(clamp((cur as number) + dir * Math.abs(t.base) * intensity, lo, hi));
      why = `${dir > 0 ? 'raised' : 'lowered'} by ${Math.abs(value - (cur as number))}`;
    }
  } else if (negated && !removes) {
    // "don't smooth the skin", "no sharpening": switch the tool off.
    value = 0;
    why = 'turned off';
  } else if (LESS_TOOL.test(clause) && (!PROBLEMS.has(t.name) || TOOL_WORD.test(clause))) {
    // "less smoothing", "too sharp", "tone down the acne removal": weaken the tool.
    // ("less acne" and "tone down the redness" name the problem, so they strengthen the tool.)
    value = Math.round((cur as number) * 0.5);
    why = (cur as number) > 0 ? `lowered from ${cur}` : 'already off';
  } else {
    const c = cur as number;
    const target = Math.round(clamp(t.base * intensity, 5, hi));
    value = MORE_TOOL.test(clause) && c > 0 ? Math.round(clamp(Math.max(target, c + 20 * intensity), lo, hi)) : Math.max(target, c);
    why = c > 0 && value > c ? `raised from ${c}` : c >= value ? 'already set' : `set to ${value}`;
  }
  const out: { step?: Omit<PlanStep, 'id' | 'enabled'>; mult?: Omit<PlanStep, 'id' | 'enabled'> } = {
    step: { target: t.target, label: t.label, value, reason: `${capital(why)}: you said ${quote}.`, needsFace: t.needsFace },
  };
  const mult = multiplierOf(clause);
  if (mult && t.effect && value > 0) {
    out.mult = { target: { kind: 'multiplier', effect: t.effect }, label: `${t.label} multiplier`, value: mult, reason: `${mult}x strength: you said ${quote}.`, needsFace: t.needsFace };
  }
  return out;
}

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function genericPass(adj: Adjustments, ctx: PlanContext, scale: number): Omit<PlanStep, 'id' | 'enabled'>[] {
  const v = (n: number) => Math.round(clamp(n * scale, 0, 100));
  const out: Omit<PlanStep, 'id' | 'enabled'>[] = [
    { target: slider('blemishRemoval'), label: 'Blemish removal', value: Math.max(adj.blemishRemoval, v(50)), reason: 'General retouch: heals spots and blemishes.' },
    { target: slider('skinSmoothing'), label: 'Skin smoothing', value: Math.max(adj.skinSmoothing, v(35)), reason: 'General retouch: evens blotches on the tone layer.' },
    { target: slider('skinTexture'), label: 'Texture preservation', value: Math.max(adj.skinTexture, 70), reason: 'General retouch: keeps pores.' },
    { target: slider('skinToneUnify'), label: 'Skin tone correction', value: Math.max(adj.skinToneUnify, v(25)), reason: 'General retouch: evens skin colour.' },
  ];
  if (ctx.hasFace) {
    out.push({ target: slider('darkCircles'), label: 'Under-eye correction', value: Math.max(adj.darkCircles, v(30)), reason: 'General retouch: lifts under-eye shadows.', needsFace: true });
    out.push({ target: slider('eyeBrighten'), label: 'Brighten eyes', value: Math.max(adj.eyeBrighten, v(20)), reason: 'General retouch: a little life in the eyes.', needsFace: true });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Recommendations from the photo's measurements
// ---------------------------------------------------------------------------------------------

const AUTO_LABELS: Partial<Record<keyof Adjustments, string>> = {
  exposure: 'Exposure', contrast: 'Contrast', temperature: 'Temperature', tint: 'Tint', vibrance: 'Vibrance', sharpen: 'Sharpen',
  skinSmoothing: 'Skin smoothing', skinTexture: 'Texture preservation', blemishRemoval: 'Blemish removal', skinToneUnify: 'Skin tone correction',
  eyeBrighten: 'Brighten eyes', darkCircles: 'Under-eye correction', teethWhiten: 'Whiten teeth',
};

/** Most a brightening may lift measured skin (L*): a dark-toned subject in a dark scene keeps their skin tone. */
export const SKIN_LIFT_MAX = 10;
const SKIN_CAP_NOTE = ` Limited so the skin is lifted no more than ${SKIN_LIFT_MAX} L*, keeping the person's real skin tone.`;

function recommendReason(key: string, p: StyleProfile, value: number): string {
  const L = p.meanL.toFixed(0);
  switch (key) {
    case 'exposure':
      return value > 0 ? `The photo is on the dark side (mean lightness L* ${L}; a well-exposed portrait sits near 56).` : `The photo is on the bright side (mean lightness L* ${L}; target near 56).`;
    case 'contrast':
      return value > 0 ? `Tones are flat (spread ${p.contrast.toFixed(0)} L*; target near 22).` : `Tones are harsh (spread ${p.contrast.toFixed(0)} L*; target near 22).`;
    case 'temperature':
      return value > 0 ? `Colour leans cool (b* ${p.warmth.toFixed(1)}; skin-friendly warmth is near 9).` : `Colour leans warm (b* ${p.warmth.toFixed(1)}; target near 9).`;
    case 'tint':
      return `Green-magenta balance is off (a* ${p.tintAxis.toFixed(1)}; target near 6).`;
    case 'vibrance':
      return `Colours are muted (mean chroma ${p.chroma.toFixed(0)}).`;
    case 'sharpen':
      return p.sharpness < 120 ? `The photo is a little soft (sharpness ${p.sharpness.toFixed(0)}).` : 'A light output sharpen.';
    case 'skinSmoothing':
    case 'blemishRemoval':
      return p.skinTexture !== null ? `Skin texture measures ${(p.skinTexture / 2.2).toFixed(1)}x a smooth portrait's, so the skin tools are set to match.` : 'Skin found by colour.';
    case 'skinTexture':
      return 'Keeps pores and fine texture under the smoothing.';
    case 'skinToneUnify':
      return 'Evens out small colour differences in the skin.';
    case 'eyeBrighten':
      return 'A face was found: a little life in the eyes.';
    case 'darkCircles':
      return 'A face was found: lifts under-eye shadows.';
    case 'teethWhiten':
      return 'A face was found: a gentle teeth clean-up (no effect if teeth are not visible).';
    default:
      return 'From this photo\'s measurements.';
  }
}

/**
 * Auto Retouch brightens toward a mean-L* target for the whole frame, which would lift the skin
 * of a darker-skinned subject well past their real tone. Brightening is capped so measured skin
 * rises at most SKIN_LIFT_MAX L*.
 */
function skinSafe(auto: Partial<Adjustments>, p: StyleProfile): { auto: Partial<Adjustments>; capped: boolean } {
  const e = auto.exposure;
  if (e === undefined || e <= 0 || p.skinLuma == null) return { auto, capped: false };
  const cap = Math.round(exposureForLightness(p.skinLuma, p.skinLuma + SKIN_LIFT_MAX));
  return e > cap ? { auto: { ...auto, exposure: cap }, capped: true } : { auto, capped: false };
}

/** Recommendations from the photo's own measurements, each with the measurement behind it. */
export function recommendPlan(p: StyleProfile, adj: Adjustments, ctx: PlanContext): EditPlan {
  const { auto, capped } = skinSafe(autoRetouch(p, ctx.hasFace), p);
  const steps: PlanStep[] = [];
  for (const [key, value] of Object.entries(auto) as Array<[keyof Adjustments, number]>) {
    if (key === 'skinMultiplier') continue;
    const target: PlanTarget = slider(key as AdjustmentKey);
    if (currentValue(adj, target) === value || !AUTO_LABELS[key]) continue;
    const reason = recommendReason(key, p, value) + (key === 'exposure' && capped ? SKIN_CAP_NOTE : '');
    steps.push({ id: targetId(target), target, label: AUTO_LABELS[key]!, value, reason, enabled: true, needsFace: ['eyeBrighten', 'darkCircles', 'teethWhiten'].includes(key) });
  }
  if (auto.skinMultiplier === 2 && adj.effects.blemishRemoval.multiplier < 2) {
    const target: PlanTarget = { kind: 'multiplier', effect: 'blemishRemoval' };
    steps.push({ id: targetId(target), target, label: 'Blemish removal multiplier', value: 2, reason: 'Skin texture is more than twice the target, so spot healing reaches further.', enabled: true });
  }
  return { source: 'recommendation', title: 'Recommended for this photo', steps, strength: 100, unmatched: [], declined: [], notes: steps.length ? [] : ['This photo already matches the recommendations.'] };
}

// ---------------------------------------------------------------------------------------------
// Retouching styles (automatic presets built per photo)
// ---------------------------------------------------------------------------------------------

export type RetouchStyleId = 'natural' | 'headshot' | 'beauty' | 'wedding' | 'mature' | 'grooming';

interface StyleSpec {
  id: RetouchStyleId;
  name: string;
  description: string;
  /** Scales the measured skin work (smoothing, blemish). */
  skin: number;
  values: Partial<Record<AdjustmentKey, number>>;
  toggles?: Partial<Record<'keepFreckles' | 'protectBeautyMarks', boolean>>;
  features?: Partial<Record<'strayHair' | 'clothingIron', number>>;
  multipliers?: Partial<Record<EffectId, EffectMultiplier>>;
}

export const RETOUCH_STYLES: readonly StyleSpec[] = [
  { id: 'natural', name: 'Natural', description: 'Light clean-up that keeps freckles, moles and full texture.', skin: 0.7, values: { skinTexture: 80, skinToneUnify: 20, darkCircles: 25, eyeBrighten: 15 }, toggles: { keepFreckles: true, protectBeautyMarks: true } },
  { id: 'headshot', name: 'Corporate headshot', description: 'Clean, even and sharp: blemishes, glare, stray hairs and shirt creases.', skin: 1, values: { skinTexture: 70, skinToneUnify: 30, rednessReduction: 30, darkCircles: 40, eyeBrighten: 25, teethWhiten: 20, glareReduction: 50, sharpen: 25 }, features: { strayHair: 50, clothingIron: 50 } },
  { id: 'beauty', name: 'Beauty', description: 'Polished skin with pores kept, brighter eyes and teeth.', skin: 1.4, values: { skinTexture: 60, skinToneUnify: 40, rednessReduction: 45, acneRemoval: 60, darkSpotRemoval: 45, darkCircles: 55, underEyeTexture: 40, eyeBrighten: 35, teethWhiten: 30, vibrance: 10 }, multipliers: { blemishRemoval: 2 }, features: { strayHair: 60 } },
  { id: 'wedding', name: 'Wedding', description: 'Soft, warm and forgiving, with redness and tired eyes calmed.', skin: 1.1, values: { skinTexture: 70, skinToneUnify: 30, rednessReduction: 40, darkCircles: 45, eyeBrighten: 25, teethWhiten: 25, temperature: 8, contrast: -5 } },
  { id: 'mature', name: 'Mature skin', description: 'Gentle: evens tone and lifts shadows but keeps character lines.', skin: 0.6, values: { skinTexture: 85, skinToneUnify: 30, rednessReduction: 35, darkSpotRemoval: 40, darkCircles: 40, underEyeTexture: 30, eyeBrighten: 20 }, toggles: { protectBeautyMarks: true } },
  { id: 'grooming', name: 'Men\'s grooming', description: 'Clears breakouts and redness, keeps strong texture.', skin: 0.6, values: { skinTexture: 85, acneRemoval: 50, rednessReduction: 40, darkCircles: 35, eyeBrighten: 15, sharpen: 30 }, features: { strayHair: 40 } },
];

const STYLE_LABELS: Record<string, string> = {
  ...(AUTO_LABELS as Record<string, string>),
  rednessReduction: 'Redness reduction', acneRemoval: 'Acne removal', darkSpotRemoval: 'Dark spot removal', underEyeTexture: 'Under-eye texture',
  glareReduction: 'Glasses glare', keepFreckles: 'Keep freckles', protectBeautyMarks: 'Protect beauty marks',
};
const FACE_KEYS = new Set(['eyeBrighten', 'darkCircles', 'teethWhiten', 'underEyeTexture', 'glareReduction', 'teethStainRemoval']);

/** A retouching style built from this photo's measurements (light and skin from Auto Retouch). */
export function stylePlan(styleId: RetouchStyleId, p: StyleProfile | null, adj: Adjustments, ctx: PlanContext): EditPlan {
  const spec = RETOUCH_STYLES.find((s) => s.id === styleId) ?? RETOUCH_STYLES[0];
  const steps: PlanStep[] = [];
  const push = (target: PlanTarget, label: string, value: number | boolean, reason: string, needsFace = false) => {
    if (currentValue(adj, target) === value) return;
    steps.push({ id: targetId(target), target, label, value, reason, enabled: !(needsFace && !ctx.hasFace), needsFace });
  };
  const { auto, capped } = p ? skinSafe(autoRetouch(p, ctx.hasFace), p) : { auto: {} as Partial<Adjustments>, capped: false };
  for (const key of ['exposure', 'contrast', 'temperature', 'tint'] as const) {
    const measured = (auto[key] as number | undefined) ?? 0;
    const value = Math.round(clamp(measured + (spec.values[key] ?? 0), -100, 100));
    if (value !== 0 || adj[key] !== 0) push(slider(key), STYLE_LABELS[key] ?? key, value, p && measured ? `${spec.name}: ${recommendReason(key, p, measured)}${key === 'exposure' && capped ? SKIN_CAP_NOTE : ''}` : `${spec.name} look.`);
  }
  if (p?.skinTexture != null) {
    const rough = clamp(p.skinTexture / 2.2, 0.5, 3);
    push(slider('skinSmoothing'), 'Skin smoothing', Math.round(clamp((20 + (rough - 1) * 25) * spec.skin, 10, 70)), `${spec.name}: matched to the measured skin texture (${rough.toFixed(1)}x a smooth portrait's).`);
    push(slider('blemishRemoval'), 'Blemish removal', Math.round(clamp((30 + (rough - 1) * 30) * spec.skin, 15, 90)), `${spec.name}: matched to the measured skin texture.`);
  } else {
    push(slider('skinSmoothing'), 'Skin smoothing', Math.round(30 * spec.skin), `${spec.name}: no skin was measured, so a moderate setting.`);
    push(slider('blemishRemoval'), 'Blemish removal', Math.round(40 * spec.skin), `${spec.name}: no skin was measured, so a moderate setting.`);
  }
  for (const [key, value] of Object.entries(spec.values) as Array<[AdjustmentKey, number]>) {
    if (key === 'temperature' || key === 'contrast' || key === 'exposure' || key === 'tint') continue;
    push(slider(key), STYLE_LABELS[key] ?? key, value, `${spec.name}: ${spec.description}`, FACE_KEYS.has(key));
  }
  for (const [key, value] of Object.entries(spec.toggles ?? {}) as Array<['keepFreckles' | 'protectBeautyMarks', boolean]>) {
    push({ kind: 'toggle', key }, STYLE_LABELS[key], value, `${spec.name}: ${value ? 'left untouched' : 'treated as spots'}.`);
  }
  for (const [f, value] of Object.entries(spec.features ?? {}) as Array<['strayHair' | 'clothingIron', number]>) {
    push({ kind: 'feature', feature: f }, f === 'strayHair' ? 'Stray hair cleanup' : 'Clothing wrinkles', value, `${spec.name}: ${spec.description}`, f === 'strayHair');
  }
  for (const [effect, value] of Object.entries(spec.multipliers ?? {}) as Array<[EffectId, EffectMultiplier]>) {
    push({ kind: 'multiplier', effect }, `${STYLE_LABELS[effect] ?? effect} multiplier`, value, `${spec.name}: spot healing reaches fainter and larger spots.`);
  }
  return { source: 'style', style: spec.id, title: `${spec.name} style`, steps, strength: 100, unmatched: [], declined: [], notes: p ? [] : ['This photo has not been measured yet, so light and colour are left alone.'] };
}

/**
 * The same plan for another photo: a style or recommendation is rebuilt from that photo's own
 * measurements (keeping which steps the user unticked and the overall strength); an instruction
 * plan carries the same values.
 */
export function planForPhoto(plan: EditPlan, p: StyleProfile | null, adj: Adjustments, ctx: PlanContext): EditPlan {
  let next: EditPlan;
  if (plan.source === 'style' && plan.style) next = stylePlan(plan.style, p, adj, ctx);
  else if (plan.source === 'recommendation' && p) next = recommendPlan(p, adj, ctx);
  else return { ...plan, steps: plan.steps.map((s) => ({ ...s, enabled: s.enabled && !(s.needsFace && !ctx.hasFace) })) };
  const off = new Set(plan.steps.filter((s) => !s.enabled).map((s) => s.id));
  next.steps = next.steps.map((s) => ({ ...s, enabled: s.enabled && !off.has(s.id) }));
  next.strength = plan.strength;
  return next;
}
