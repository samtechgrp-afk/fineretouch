'use client';

import {
  FINISH_CONTROLS,
  FINISH_CONTROL_HINTS,
  FINISH_CONTROL_LABELS,
  FINISH_STYLE_LABELS,
  applyFinishStyle,
  finishResetValue,
  resetFinishControl,
  setFinishControl,
  DEFAULT_SKIN_FINISH,
  type FinishControlName,
  type FinishStyle,
  type SkinFinishSettings,
} from '@fineretouch/shared';
import { runPerfection, usePerfection, type FinishSummary } from '@/lib/perfection';
import { useEditor, type Photo } from '@/lib/store';
import { Heading, RangeRow, Segmented, Toggle } from './controls';

const STYLE_OPTIONS: Array<[FinishStyle, string]> = [
  ['natural', FINISH_STYLE_LABELS.natural],
  ['professional', FINISH_STYLE_LABELS.professional],
  ['advanced', FINISH_STYLE_LABELS.advanced],
  ['custom', 'Custom'],
];

/** The texture finish (phase 1) and the local corrections (phase 2), in the order they show. */
const TEXTURE: FinishControlName[] = ['smoothing', 'texturePreservation', 'porePreservation'];
const LOCAL: FinishControlName[] = ['darkSpots', 'toneEvenness', 'shine', 'underEye', 'dodgeBurn'];

/**
 * Professional Skin Finish (v0.14): every control is a field of adjustments.skinFinish, read by
 * the engine (services/ai/app/engines/finish.py and finish_local.py). It runs in the local AI
 * engine at full resolution after Skin Perfection; the viewport shows the engine's result, so
 * what you see is what the export renders.
 */
export function SkinFinishPanel({ photo }: { photo: Photo }) {
  const beginEdit = useEditor((s) => s.beginEdit);
  const patchAdjustments = useEditor((s) => s.patchAdjustments);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const aiOnline = useEditor((s) => s.aiOnline);
  const view = useEditor((s) => s.view);
  const setView = useEditor((s) => s.setView);
  const perf = usePerfection((s) => s.byPhoto[photo.id]);
  const fs = photo.adjustments.skinFinish;
  const on = fs.enabled;

  /** Live (slider drag): no undo step; the drag start recorded one. */
  const live = (next: SkinFinishSettings) => patchAdjustments({ skinFinish: next }, photo.id);
  /** Discrete change: one undo step. */
  const commit = (next: SkinFinishSettings) => updateAdjustments({ skinFinish: next }, [photo.id]);
  const start = () => beginEdit(photo.id);

  const status = !on
    ? 'Off'
    : !aiOnline
      ? 'Needs the local AI engine'
      : perf?.status === 'running'
        ? 'Finishing at full resolution…'
        : perf?.status === 'error'
          ? perf.error ?? 'Failed'
          : perf?.status === 'ready'
            ? `Done in ${((perf.ms ?? 0) / 1000).toFixed(1)} s`
            : 'Waiting';

  const row = (k: FinishControlName) => {
    const c = fs[k];
    const changed = c.amount !== finishResetValue(fs, k) || !c.enabled;
    return (
      <div key={k} className="flex items-end gap-2" data-testid={`finish-${k}`}>
        <div className="pb-0.5">
          <Toggle checked={c.enabled} disabled={!on} onChange={(v) => commit(setFinishControl(fs, k, { enabled: v }))} label={`${FINISH_CONTROL_LABELS[k]} on`} />
        </div>
        <div className={`min-w-0 flex-1 ${c.enabled ? '' : 'line-through decoration-ink-500'}`}>
          <RangeRow
            label={FINISH_CONTROL_LABELS[k]}
            value={c.amount}
            min={0}
            max={100}
            hint={FINISH_CONTROL_HINTS[k]}
            disabled={!on || !c.enabled}
            onStart={start}
            onChange={(v) => live(setFinishControl(fs, k, { amount: v }))}
          />
        </div>
        <button
          className="pb-0.5 text-[11px] text-ink-400 hover:text-ink-100 disabled:opacity-30"
          disabled={!on || !changed}
          onClick={() => commit(resetFinishControl(fs, k))}
          title={`Reset to ${finishResetValue(fs, k)} and switch on`}
          aria-label={`Reset ${FINISH_CONTROL_LABELS[k]}`}
        >
          Reset
        </button>
      </div>
    );
  };

  return (
    <div className="space-y-4" data-testid="skin-finish">
      <section>
        <Heading
          title="Professional Skin Finish"
          hint="A retoucher’s finish after Skin Perfection: evens dark marks, blotchy tone, shine and under-eye shadows and refines texture, keeping pores, the face’s shape and your skin tone. Runs at full resolution in the local AI engine."
          right={<Toggle checked={on} onChange={(v) => commit({ ...fs, enabled: v })} label="Professional Skin Finish" />}
        />
        <p className={`text-[11px] ${perf?.status === 'error' || (on && !aiOnline) ? 'text-rose-400' : 'text-ink-400'}`} data-testid="finish-status">
          {status}
        </p>
      </section>

      <section className={on ? 'space-y-2.5' : 'space-y-2.5 opacity-60'}>
        <Segmented<FinishStyle>
          label="Finishing style"
          value={fs.style}
          options={STYLE_OPTIONS}
          disabled={!on}
          onChange={(v) => v !== 'custom' && commit(applyFinishStyle(fs, v))}
        />
        <div className="flex items-end gap-2">
          <div className="min-w-0 flex-1">
            <RangeRow
              label="Overall finish strength"
              value={fs.strength}
              min={0}
              max={100}
              hint="Scales every effect below together (0 leaves the photo as Skin Perfection left it)."
              disabled={!on}
              onStart={start}
              onChange={(v) => live({ ...fs, strength: v })}
            />
          </div>
          <button
            className="pb-0.5 text-[11px] text-ink-400 hover:text-ink-100 disabled:opacity-30"
            disabled={!on || fs.strength === DEFAULT_SKIN_FINISH.strength}
            onClick={() => commit({ ...fs, strength: DEFAULT_SKIN_FINISH.strength })}
            aria-label="Reset overall finish strength"
          >
            Reset
          </button>
        </div>
      </section>

      <section className={on ? 'space-y-2.5' : 'space-y-2.5 opacity-60'}>
        <h4 className="text-[11px] font-medium uppercase tracking-wider text-ink-400">Corrections</h4>
        {LOCAL.map(row)}
      </section>

      <section className={on ? 'space-y-2.5' : 'space-y-2.5 opacity-60'}>
        <h4 className="text-[11px] font-medium uppercase tracking-wider text-ink-400">Texture</h4>
        {TEXTURE.map(row)}
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-ink-300" title="Neck, shoulders, arms, hands and legs found by face parsing. Clothing, hair and background are never changed.">
            Body skin
          </span>
          <Toggle checked={fs.bodyCoverage} disabled={!on} onChange={(v) => commit({ ...fs, bodyCoverage: v })} label="Finish body skin" />
        </div>
      </section>

      <section className={on ? 'space-y-2' : 'space-y-2 opacity-60'}>
        <h4 className="text-[11px] font-medium uppercase tracking-wider text-ink-400">View</h4>
        <Segmented<'edited' | 'original' | 'split'>
          label="Before and after"
          value={view.mode === 'original' || view.mode === 'split' ? view.mode : 'edited'}
          options={[
            ['original', 'Before'],
            ['edited', 'After'],
            ['split', 'Split'],
          ]}
          onChange={(v) => setView({ mode: v })}
        />
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-ink-300" title="Shows the skin area the finish works on (the skin mask). Shortcut: M">Show finish area</span>
          <Toggle checked={view.showMask} onChange={(v) => setView({ showMask: v })} label="Show finish area" />
        </div>
        <Segmented<'off' | 'add' | 'erase'>
          label="Finish area brush"
          value={view.maskBrush ?? 'off'}
          options={[
            ['off', 'No brush'],
            ['add', 'Add skin'],
            ['erase', 'Protect'],
          ]}
          disabled={!on}
          onChange={(v) => setView({ maskBrush: v === 'off' ? null : v, proTool: v === 'off' ? view.proTool : null, showMask: v === 'off' ? view.showMask : true })}
        />
        <p className="text-[11px] text-ink-400">The brush edits the skin mask, shared with the other skin tools: Protect keeps an area out of the finish.</p>
      </section>

      {on && perf?.finish && <FinishMeasures finish={perf.finish} />}

      <div className="flex text-[11px]">
        <button
          className="ml-auto text-ink-400 hover:text-ink-200 disabled:opacity-40"
          disabled={!on || !aiOnline || perf?.status === 'running'}
          onClick={() => void runPerfection(photo.id, { force: true })}
        >
          Render again
        </button>
      </div>
    </div>
  );
}

const MEASURE_ROWS: Array<[string, string, string, string]> = [
  ['darkSpots', 'Dark marks (mean depth, L*)', 'meanDepthBefore', 'meanDepthAfter'],
  ['toneEvenness', 'Colour variation', 'colourVariationBefore', 'colourVariationAfter'],
  ['shine', 'Brightest 1% (L*)', 'p99Before', 'p99After'],
  ['underEye', 'Under-eye shadow (L*)', 'deficitBefore', 'deficitAfter'],
  ['dodgeBurn', 'Light irregularity', 'microBefore', 'microAfter'],
];

/** What the engine measured on this render (not what the sliders promise). */
function FinishMeasures({ finish }: { finish: FinishSummary }) {
  const local = (finish.local ?? {}) as Record<string, Record<string, number | null>>;
  const rows = MEASURE_ROWS.filter(([k, , b, a]) => typeof local[k]?.[b] === 'number' && typeof local[k]?.[a] === 'number');
  return (
    <section data-testid="finish-measures">
      <Heading title="Measured on this photo" hint="Before and after each correction, on the skin the finish covers." />
      <dl className="space-y-0.5 text-[11px]">
        {rows.map(([k, label, b, a]) => (
          <div key={k} className="flex justify-between gap-2">
            <dt className="text-ink-400">{label}</dt>
            <dd className="font-mono tabular-nums text-ink-200">
              {local[k][b]!.toFixed(2)} → {local[k][a]!.toFixed(2)}
            </dd>
          </div>
        ))}
        {typeof finish.edgeKept === 'number' && (
          <div className="flex justify-between gap-2">
            <dt className="text-ink-400">Contours kept</dt>
            <dd className="font-mono tabular-nums text-ink-200">{Math.round(finish.edgeKept * 100)}%</dd>
          </div>
        )}
        {typeof finish.textureKept === 'number' && (
          <div className="flex justify-between gap-2">
            <dt className="text-ink-400">Texture kept</dt>
            <dd className="font-mono tabular-nums text-ink-200">{Math.round(finish.textureKept * 100)}%</dd>
          </div>
        )}
      </dl>
    </section>
  );
}
