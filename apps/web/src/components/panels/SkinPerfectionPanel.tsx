'use client';

import {
  PERFECTION_CATEGORIES,
  PERFECTION_CATEGORY_LABELS,
  PERFECTION_GROUPS,
  correctRemaining,
  groupSensitivity,
  overrideMark,
  setGroupSensitivity,
  type PerfectionCategoryName,
  type PerfectionGroup,
  type SkinPerfectionSettings,
} from '@fineretouch/shared';
import { useState } from 'react';
import { runPerfection, usePerfection } from '@/lib/perfection';
import { useEditor, type Photo } from '@/lib/store';
import { MARK_COLOURS } from '../viewport/perfectionOverlay';
import { Heading, RangeRow, Segmented, Toggle } from './controls';

const MULTIPLIER_OPTIONS: Array<['1' | '2' | '3', string]> = [
  ['1', '1× Natural'],
  ['2', '2× Strong'],
  ['3', '3× Maximum'],
];

/**
 * Skin Perfection Studio (v0.13): every control here is a field of adjustments.skinPerfection,
 * read by the engine (services/ai/app/engines/perfection.py). It runs in the local AI engine at
 * full resolution; the preview shows the engine's result and its report (Final quality check).
 */
export function SkinPerfectionPanel({ photo }: { photo: Photo }) {
  const beginEdit = useEditor((s) => s.beginEdit);
  const patchAdjustments = useEditor((s) => s.patchAdjustments);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const aiOnline = useEditor((s) => s.aiOnline);
  const perf = usePerfection((s) => s.byPhoto[photo.id]);
  const overlay = usePerfection((s) => s.overlay);
  const setOverlay = usePerfection((s) => s.setOverlay);
  const selected = usePerfection((s) => s.selected);
  const select = usePerfection((s) => s.select);
  const [advanced, setAdvanced] = useState(false);
  const sp = photo.adjustments.skinPerfection;
  const on = sp.enabled;

  /** Live (slider drag): no undo step; the drag start recorded one. */
  const live = (next: SkinPerfectionSettings) => patchAdjustments({ skinPerfection: next }, photo.id);
  /** Discrete change: one undo step. */
  const commit = (next: SkinPerfectionSettings) => updateAdjustments({ skinPerfection: next }, [photo.id]);
  const start = () => beginEdit(photo.id);
  const slider = (label: string, value: number, set: (v: number) => SkinPerfectionSettings, hint: string, opts: { min?: number; max?: number; format?: (v: number) => string } = {}) => (
    <RangeRow label={label} value={value} min={opts.min ?? 0} max={opts.max ?? 100} format={opts.format} hint={hint} disabled={!on} onStart={start} onChange={(v) => live(set(v))} />
  );
  const setCat = (k: PerfectionCategoryName, patch: Partial<SkinPerfectionSettings['categories'][PerfectionCategoryName]>) => ({
    ...sp,
    categories: { ...sp.categories, [k]: { ...sp.categories[k], ...patch } },
  });

  const report = on && perf?.report ? perf.report : null;
  const counts = report?.counts;
  const marksById = new Map(report?.marks.map((m) => [m.id, m]) ?? []);
  const chosen = selected.map((id) => marksById.get(id)).filter((m): m is NonNullable<typeof m> => Boolean(m));
  const minConf = sp.confidenceThreshold / 100;
  const fixable = report?.marks.filter((m) => (m.status === 'remaining' || m.status === 'review') && m.confidence >= minConf).length ?? 0;
  const status = !on
    ? 'Off'
    : !aiOnline
      ? 'Needs the local AI engine'
      : perf?.status === 'running'
        ? 'Analysing at full resolution…'
        : perf?.status === 'error'
          ? perf.error ?? 'Failed'
          : perf?.status === 'ready'
            ? `Done in ${((perf.ms ?? 0) / 1000).toFixed(1)} s`
            : 'Waiting';

  return (
    <div className="space-y-4" data-testid="skin-perfection">
      <section>
        <Heading
          title="Skin Perfection Studio"
          hint="Finds acne, blackheads, dark spots and small blemishes on face and body skin at full resolution, corrects each one with real skin texture, then rescans and reports what is left. Freckles, moles, pores and hair are kept."
          right={<Toggle checked={on} onChange={(v) => commit({ ...sp, enabled: v })} label="Skin Perfection" />}
        />
        <p className={`text-[11px] ${perf?.status === 'error' || (on && !aiOnline) ? 'text-rose-400' : 'text-ink-400'}`} data-testid="perfection-status">
          {status}
        </p>
      </section>

      <section className={on ? '' : 'opacity-60'}>
        <Segmented<'1' | '2' | '3'>
          label="Skin Perfection multiplier"
          value={`${sp.multiplier}` as '1' | '2' | '3'}
          options={MULTIPLIER_OPTIONS}
          disabled={!on}
          onChange={(v) => commit({ ...sp, multiplier: Number(v) as 1 | 2 | 3 })}
        />
        <div className="mt-3 space-y-2.5">
          {slider('Overall strength', sp.strength, (v) => ({ ...sp, strength: v }), 'Scales every correction below (0 leaves the photo as it is).')}
          {(Object.keys(PERFECTION_GROUPS) as PerfectionGroup[]).map((g) =>
            slider(PERFECTION_GROUPS[g].label, groupSensitivity(sp, g), (v) => setGroupSensitivity(sp, g, v),
              `Detection sensitivity for ${PERFECTION_GROUPS[g].categories.map((k) => PERFECTION_CATEGORY_LABELS[k].toLowerCase()).join(', ')}.`),
          )}
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-ink-300" title="Neck, chest, shoulders, arms, hands, back and legs (clothing, hair and background are excluded).">Body blemish correction</span>
            <Toggle checked={sp.bodyCoverage} disabled={!on} onChange={(v) => commit({ ...sp, bodyCoverage: v })} label="Body blemish correction" />
          </div>
          {sp.bodyCoverage && slider('Body sensitivity', sp.bodySensitivity, (v) => ({ ...sp, bodySensitivity: v }), 'How faint a mark on body skin may be and still be corrected.')}
          {slider('Skin texture preservation', sp.texturePreservation, (v) => ({ ...sp, texturePreservation: v }), 'How much real skin texture is rebuilt inside every repair (high keeps it photographic).')}
          {slider('Pore preservation', sp.porePreservation, (v) => ({ ...sp, porePreservation: v }), 'Keeps pore-sized dots out of the correction and restores pore detail in repairs.')}
          {slider('Residual imperfection scan', sp.residualIntensity, (v) => ({ ...sp, residualIntensity: v }), 'How hard the rescan after each pass looks for marks that are still visible.')}
        </div>
      </section>

      <section className={on ? '' : 'opacity-60'}>
        <button className="text-[11px] text-apricot-400 hover:underline" onClick={() => setAdvanced((a) => !a)} aria-expanded={advanced}>
          {advanced ? 'Hide advanced settings' : 'Advanced settings (radius, edges, confidence, passes, each category)'}
        </button>
        {advanced && (
          <div className="mt-2.5 space-y-2.5">
            {slider('Correction radius', sp.correctionRadius, (v) => ({ ...sp, correctionRadius: v }), 'How far each repair reaches past the mark it covers.')}
            {slider('Edge protection', sp.edgeProtection, (v) => ({ ...sp, edgeProtection: v }), 'Keeps repairs off lips, eyes, nostrils, hairlines and strong edges.')}
            {slider('Confidence threshold', sp.confidenceThreshold, (v) => ({ ...sp, confidenceThreshold: v }), 'Marks below this confidence are not corrected automatically; they are flagged for review.')}
            {slider('Passes', sp.passes, (v) => ({ ...sp, passes: Math.round(v) }), 'Correction passes, each followed by a residual scan (stops early when nothing is left).', { min: 1, max: 4 })}
            <div className="space-y-2">
              {(
                [
                  ['preserveFreckles', 'Preserve freckles'],
                  ['preserveMoles', 'Preserve moles'],
                  ['preserveBirthmarks', 'Preserve birthmarks'],
                ] as const
              ).map(([k, label]) => (
                <div key={k} className="flex items-center justify-between gap-2">
                  <span className="text-xs text-ink-300" title="Off: these are corrected like dark spots.">{label}</span>
                  <Toggle checked={sp[k]} disabled={!on} onChange={(v) => commit({ ...sp, [k]: v })} label={label} />
                </div>
              ))}
            </div>
            <div className="space-y-2 border-t border-ink-800 pt-2.5">
              {PERFECTION_CATEGORIES.map((k) => {
                const c = sp.categories[k];
                const label = PERFECTION_CATEGORY_LABELS[k];
                return (
                  <div key={k} className="space-y-1.5 rounded bg-ink-900/60 p-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs text-ink-200">{label}</span>
                      <Toggle checked={c.enabled} disabled={!on} onChange={(v) => commit(setCat(k, { enabled: v }))} label={label} />
                    </div>
                    {c.enabled && (
                      <>
                        {slider(`${label} sensitivity`, c.sensitivity, (v) => setCat(k, { sensitivity: v }), `How faint a ${label.toLowerCase()} mark may be and still be found.`)}
                        {slider(`${label} strength`, c.strength, (v) => setCat(k, { strength: v }), `How fully ${label.toLowerCase()} is corrected.`)}
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      <section className={on ? '' : 'opacity-60'} data-testid="perfection-quality">
        <Heading
          title="Final quality check"
          hint="Measured after correction, not assumed: a mark counts as corrected only when its contrast fell enough on the rescan."
          right={
            <label className="flex items-center gap-1.5 text-[11px] text-ink-300">
              Overlay
              <Toggle checked={overlay} onChange={setOverlay} label="Show marks overlay" />
            </label>
          }
        />
        <dl className="grid grid-cols-3 gap-1.5 text-center text-[11px]">
          <Count label="Detected" value={counts?.detected} colour="text-ink-100" />
          <Count label="Corrected" value={counts?.corrected} colour="text-mint-400" />
          <Count label="Needs review" value={counts?.remaining} colour="text-rose-400" />
        </dl>
        {report && (
          <p className="mt-1.5 text-[11px] text-ink-400">
            {counts!.protected} protected (freckles, moles, follicles) · {counts!.ignored} ignored
            {report.meanConfidence !== null && ` · confidence ${Math.round(report.meanConfidence * 100)}%`}
            {report.textureRetention !== null && ` · texture kept ${Math.round(report.textureRetention * 100)}%`}
          </p>
        )}
        {report && (
          <ul className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-ink-400" aria-label="Overlay legend">
            {(['corrected', 'remaining', 'review', 'protected', 'ignored'] as const).map((k) => (
              <li key={k} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full border-2" style={{ borderColor: MARK_COLOURS[k] }} />
                {k === 'review' ? 'low confidence' : k}
              </li>
            ))}
          </ul>
        )}
        <div className="mt-2.5 grid grid-cols-2 gap-2">
          <button
            className="btn-primary"
            disabled={!on || !report || fixable === 0}
            onClick={() => commit(correctRemaining(sp, report!, minConf))}
            title="Correct every remaining mark at or above the confidence threshold (one undo step)"
          >
            Correct all remaining {fixable ? `(${fixable})` : ''}
          </button>
          <button
            className="btn"
            disabled={!on || !aiOnline || perf?.status === 'running' || sp.passes >= 4}
            onClick={() => commit({ ...sp, passes: Math.min(4, sp.passes + 1) })}
            title="One more correction pass with a residual scan (at most 4)"
          >
            Run another pass ({sp.passes}/4)
          </button>
        </div>
        {chosen.length > 0 ? (
          <div className="mt-2 rounded bg-ink-850 p-2 text-[11px]" data-testid="perfection-selection">
            <p className="text-ink-200">
              {chosen.length === 1
                ? `${kindLabel(chosen[0].kind)} · ${chosen[0].region} · ${Math.round(chosen[0].confidence * 100)}% · ${chosen[0].status}`
                : `${chosen.length} marks selected`}
            </p>
            <div className="mt-1.5 flex gap-2">
              <button className="btn flex-1" onClick={() => { commit(chosen.reduce((s, m) => overrideMark(s, m, 'correct'), sp)); select([]); }}>Correct</button>
              <button className="btn flex-1" onClick={() => { commit(chosen.reduce((s, m) => overrideMark(s, m, 'ignore'), sp)); select([]); }}>Ignore</button>
              <button className="btn" onClick={() => select([])} aria-label="Clear selection">✕</button>
            </div>
          </div>
        ) : (
          report && <p className="mt-2 text-[11px] text-ink-400">Click a mark on the photo to select it (Shift-click for more), then correct or ignore it. Undo reverts any of these.</p>
        )}
        <div className="mt-2 flex gap-2 text-[11px]">
          {sp.overrides.length > 0 && (
            <button className="text-ink-400 hover:text-ink-200" onClick={() => commit({ ...sp, overrides: [] })}>
              Clear {sp.overrides.length} manual choice{sp.overrides.length > 1 ? 's' : ''}
            </button>
          )}
          <button className="ml-auto text-ink-400 hover:text-ink-200 disabled:opacity-40" disabled={!on || !aiOnline || perf?.status === 'running'} onClick={() => void runPerfection(photo.id, { force: true })}>
            Rescan
          </button>
        </div>
      </section>
    </div>
  );
}

function kindLabel(kind: string): string {
  return (PERFECTION_CATEGORY_LABELS as Record<string, string>)[kind] ?? kind.charAt(0).toUpperCase() + kind.slice(1);
}

function Count({ label, value, colour }: { label: string; value: number | undefined; colour: string }) {
  return (
    <div className="rounded bg-ink-850 px-1.5 py-1.5">
      <dt className="text-ink-400">{label}</dt>
      <dd className={`font-mono text-base ${colour}`}>{value ?? '–'}</dd>
    </div>
  );
}
