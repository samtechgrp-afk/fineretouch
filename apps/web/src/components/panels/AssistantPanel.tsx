'use client';

import { RETOUCH_STYLES, currentValue, targetRange, type EditPlan, type PlanStep } from '@fineretouch/shared';
import { useState } from 'react';
import { useEditor, type Photo } from '@/lib/store';
import { Heading, RangeRow, Segmented, Toggle } from './controls';

const EXAMPLES = ['Remove acne but keep freckles', 'Brighten the eyes and whiten teeth slightly', 'Natural skin, reduce redness and dark circles', 'Fix the glasses glare and iron the shirt'];

/**
 * Retouch assistant (v0.11): instructions, recommendations and retouching styles all produce an
 * edit plan that is previewed in the viewport and only applied when the user approves it.
 */
export function AssistantPanel({ photo }: { photo: Photo }) {
  const pending = useEditor((s) => (s.assistant.pending?.photoId === photo.id ? s.assistant.pending.plan : null));
  const proposeInstruction = useEditor((s) => s.proposeInstruction);
  const proposeRecommendation = useEditor((s) => s.proposeRecommendation);
  const proposeStyle = useEditor((s) => s.proposeStyle);
  const [text, setText] = useState('');
  const [note, setNote] = useState<string | null>(null);

  const submit = () => {
    if (!text.trim()) return;
    setNote(null);
    proposeInstruction(text);
  };

  return (
    <div className="space-y-4">
      <section>
        <Heading
          title="Ask for an edit"
          hint="Describe the retouch in plain English. The assistant runs on this computer and never uploads the photo: it matches retouching terms (acne, redness, dark circles, teeth, glare, flyaways, brighter, warmer…) to FineRetouch's tools and shows you the plan before anything changes. It is rule-based, not a language model."
        />
        <textarea
          className="h-16 w-full resize-none rounded-md border border-ink-700 bg-ink-900 px-2 py-1.5 text-xs text-ink-100 placeholder:text-ink-500 focus:border-apricot-400 focus:outline-none"
          placeholder="e.g. remove acne and dark spots, keep the freckles, slightly brighter"
          value={text}
          maxLength={1000}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          aria-label="Describe the edit"
        />
        <div className="mt-1.5 flex items-center justify-between gap-2">
          <div className="flex flex-wrap gap-1">
            {EXAMPLES.slice(0, 2).map((ex) => (
              <button key={ex} className="rounded bg-ink-800 px-1.5 py-0.5 text-[10px] text-ink-300 hover:text-ink-100" onClick={() => setText(ex)}>
                {ex}
              </button>
            ))}
          </div>
          <button className="btn-primary shrink-0" disabled={!text.trim()} onClick={submit}>
            Make a plan
          </button>
        </div>
      </section>

      <section>
        <Heading title="Recommended for this photo" hint="Measures this photo's exposure, contrast, white balance, colour and skin texture and suggests corrections, each with the measurement behind it." />
        <button
          className="btn w-full"
          disabled={!photo.profile}
          onClick={() => {
            const plan = proposeRecommendation();
            setNote(plan && !plan.steps.length ? 'This photo already matches the recommendations.' : null);
          }}
        >
          {photo.profile ? 'Show recommendations' : 'Measuring the photo…'}
        </button>
      </section>

      <section>
        <Heading title="Retouching styles" hint="Each style is built for this photo: light and colour from its own measurements and skin work matched to its measured skin texture, so the same style suits different photos." />
        <div className="grid grid-cols-2 gap-1.5">
          {RETOUCH_STYLES.map((st) => (
            <button key={st.id} className="btn text-left" title={st.description} onClick={() => { setNote(null); proposeStyle(st.id); }}>
              {st.name}
            </button>
          ))}
        </div>
      </section>

      {note && <p className="text-[11px] text-mint-400">{note}</p>}
      {pending && <PlanReview photo={photo} plan={pending} />}
    </div>
  );
}

function PlanReview({ photo, plan }: { photo: Photo; plan: EditPlan }) {
  const updatePlanStep = useEditor((s) => s.updatePlanStep);
  const setPlanStrength = useEditor((s) => s.setPlanStrength);
  const setPlanBatch = useEditor((s) => s.setPlanBatch);
  const approvePlan = useEditor((s) => s.approvePlan);
  const rejectPlan = useEditor((s) => s.rejectPlan);
  const batch = useEditor((s) => s.assistant.batch);
  const others = useEditor((s) => s.selected.filter((id) => id !== photo.id).length);
  const [done, setDone] = useState<string | null>(null);
  const ticked = plan.steps.filter((s) => s.enabled).length;

  return (
    <section className="rounded-md border border-apricot-400/40 bg-ink-900/60 p-2.5" aria-label="Edit plan">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <h3 className="text-xs font-semibold text-ink-100">{plan.title}</h3>
        <span className="text-[10px] uppercase tracking-wider text-apricot-400">Previewing</span>
      </div>
      {plan.instruction && <p className="mb-2 text-[11px] italic text-ink-400">“{plan.instruction}”</p>}

      {plan.declined.map((d) => (
        <p key={d} className="mb-1 text-[11px] text-rose-300">Not done: {d}</p>
      ))}
      {plan.unmatched.length > 0 && (
        <p className="mb-1 text-[11px] text-amber-300">Not understood: {plan.unmatched.map((u) => `"${u}"`).join(', ')}. Try naming the problem (acne, redness, glare, teeth…).</p>
      )}
      {plan.notes.map((n) => (
        <p key={n} className="mb-1 text-[11px] text-ink-400">{n}</p>
      ))}

      {plan.steps.length > 0 && (
        <>
          <ul className="mt-2 space-y-2.5">
            {plan.steps.map((st) => (
              <PlanStepRow key={st.id} step={st} from={currentValue(photo.adjustments, st.target)} onChange={(patch) => updatePlanStep(st.id, patch)} />
            ))}
          </ul>
          <div className="mt-3 border-t border-ink-800 pt-2.5">
            <RangeRow label="Plan strength" hint="Share of every change that is applied: 100 = the values above, 50 = halfway from the current edit." value={plan.strength} min={0} max={100} format={(v) => `${Math.round(v)}%`} onStart={() => {}} onChange={setPlanStrength} />
          </div>
          {others > 0 && (
            <label className="mt-2.5 flex items-center justify-between gap-2 text-xs text-ink-300">
              <span title={plan.source === 'instruction' ? 'The same values go to each selected photo.' : "Rebuilt from each photo's own measurements; your unticked steps and plan strength carry over."}>
                Also apply to {others} other selected photo{others === 1 ? '' : 's'}
              </span>
              <Toggle checked={batch} onChange={setPlanBatch} label="Apply to selected photos" />
            </label>
          )}
        </>
      )}

      <div className="mt-3 grid grid-cols-2 gap-2">
        <button
          className="btn-primary"
          disabled={!ticked || plan.strength === 0}
          onClick={() => {
            const n = approvePlan();
            setDone(n ? `Applied to ${n} photo${n === 1 ? '' : 's'}. Undo reverts it in one step.` : 'Nothing changed.');
          }}
        >
          Approve{ticked ? ` (${ticked})` : ''}
        </button>
        <button className="btn" onClick={rejectPlan}>
          {plan.steps.length ? 'Reject' : 'Close'}
        </button>
      </div>
      {done && <p className="mt-1.5 text-[11px] text-mint-400">{done}</p>}
    </section>
  );
}

function PlanStepRow({ step, from, onChange }: { step: PlanStep; from: number | boolean; onChange(patch: Partial<Pick<PlanStep, 'value' | 'enabled'>>): void }) {
  const [lo, hi] = targetRange(step.target);
  return (
    <li className={step.enabled ? '' : 'opacity-50'}>
      <div className="flex items-start gap-2">
        <input
          type="checkbox"
          className="mt-0.5 accent-apricot-400"
          checked={step.enabled}
          onChange={(e) => onChange({ enabled: e.target.checked })}
          aria-label={`Apply ${step.label}`}
        />
        <div className="min-w-0 flex-1">
          {typeof step.value === 'boolean' ? (
            <div className="flex items-center justify-between gap-2 text-xs text-ink-300">
              <span>{step.label}</span>
              <Toggle checked={step.value} onChange={(v) => onChange({ value: v })} label={step.label} disabled={!step.enabled} />
            </div>
          ) : step.target.kind === 'multiplier' ? (
            <Segmented<'1' | '2' | '3'>
              label={step.label}
              value={`${step.value}` as '1' | '2' | '3'}
              options={[['1', '1x'], ['2', '2x'], ['3', '3x']]}
              onChange={(v) => onChange({ value: Number(v) })}
              disabled={!step.enabled}
            />
          ) : (
            <RangeRow
              label={`${step.label} (now ${from})`}
              value={step.value}
              min={lo}
              max={hi}
              disabled={!step.enabled}
              onStart={() => {}}
              onChange={(v) => onChange({ value: v })}
            />
          )}
          <p className="mt-0.5 text-[10px] leading-snug text-ink-500">{step.reason}</p>
        </div>
      </div>
    </li>
  );
}
