'use client';

import { LABEL_KEEP_KINDS, LABEL_KIND_NAMES, LABEL_REMOVE_KINDS, labelCounts, type LabelKind } from '@fineretouch/shared';
import { useRef, useState } from 'react';
import { usePerfection } from '@/lib/perfection';
import { loadLabels, saveLabels, useLabels } from '@/lib/qaLabels';
import type { Photo } from '@/lib/store';
import { Heading, RangeRow, Toggle } from './controls';

/**
 * Label mode for the Skin Perfection test set: mark imperfections (to remove) and natural
 * features (to keep) on a real photo and save `<photo>.labels.json` for the QA benchmark.
 * Labels never change the photo, the edit or exports.
 */
export function QaLabelsPanel({ photo }: { photo: Photo }) {
  const on = useLabels((s) => s.on);
  const kind = useLabels((s) => s.kind);
  const radius = useLabels((s) => s.radius);
  const selectedId = useLabels((s) => s.selected);
  const mine = useLabels((s) => s.byPhoto[photo.id]);
  const st = useLabels.getState();
  const report = usePerfection((s) => s.byPhoto[photo.id]?.report ?? null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [note, setNote] = useState<string | null>(null);
  const labels = mine?.labels ?? [];
  const sel = labels.find((l) => l.id === selectedId) ?? null;
  const c = labelCounts(labels);

  const kindSelect = (value: LabelKind, onChange: (k: LabelKind) => void, label: string) => (
    <select
      aria-label={label}
      className="w-full rounded bg-ink-850 px-2 py-1 text-xs text-ink-100"
      value={value}
      onChange={(e) => onChange(e.target.value as LabelKind)}
    >
      <optgroup label="Imperfection (should be removed)">
        {LABEL_REMOVE_KINDS.map((k) => <option key={k} value={k}>{LABEL_KIND_NAMES[k]}</option>)}
      </optgroup>
      <optgroup label="Natural feature (must be kept)">
        {LABEL_KEEP_KINDS.map((k) => <option key={k} value={k}>{LABEL_KIND_NAMES[k]}</option>)}
      </optgroup>
    </select>
  );

  return (
    <div className="space-y-3" data-testid="qa-labels">
      <Heading
        title="Label mode"
        hint="For building the real-photo test set. Click the photo to place a label, click a label to edit it, Alt-click to delete it. Labels are saved to a separate file, never into the photo or your edit."
        right={<Toggle checked={on} onChange={(v) => st.setOn(v)} label="Label mode" />}
      />
      <div className="grid grid-cols-3 gap-1.5 text-center text-[11px]">
        <Stat label="To remove" value={c.remove} />
        <Stat label="To keep" value={c.keep} />
        <Stat label="Unconfirmed" value={c.unconfirmed} />
      </div>
      {on && (
        <div className="space-y-2">
          {kindSelect(kind, st.setKind, 'Kind for new labels')}
          <RangeRow
            label="Label size"
            value={radius * 1000}
            min={0.5}
            max={30}
            step={0.5}
            format={(v) => `${v.toFixed(1)}‰`}
            hint="Radius of new labels as a share of the photo's long edge (match the mark)."
            onStart={() => undefined}
            onChange={(v) => st.setRadius(v / 1000)}
          />
        </div>
      )}
      {sel && (
        <div className="space-y-1.5 rounded bg-ink-850 p-2 text-[11px]" data-testid="qa-label-selected">
          <p className="text-ink-300">{sel.source === 'engine' ? 'Suggested by the engine' : 'Placed by hand'} · {sel.confirmed ? 'confirmed' : 'not confirmed yet'}</p>
          {kindSelect(sel.kind, (k) => st.update(photo.id, sel.id, { kind: k, confirmed: true }), 'Label kind')}
          <div className="flex gap-2">
            {!sel.confirmed && <button className="btn flex-1" onClick={() => st.update(photo.id, sel.id, { confirmed: true })}>Confirm</button>}
            <button className="btn flex-1" onClick={() => st.remove(photo.id, sel.id)}>Delete</button>
            <button className="btn" onClick={() => st.select(null)} aria-label="Deselect label">✕</button>
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        <button
          className="btn"
          disabled={!report}
          title="Adds the engine's marks as unconfirmed suggestions. The benchmark ignores them until you confirm each one."
          onClick={() => setNote(`Added ${st.prefill(photo.id, report!)} suggestions to check.`)}
        >
          Suggest from engine
        </button>
        <button className="btn" disabled={!c.unconfirmed} onClick={() => st.confirmAll(photo.id)} title="Only after checking every suggestion">
          Confirm all
        </button>
        <button
          className="btn-primary"
          disabled={!labels.length}
          onClick={() => void saveLabels(photo.id).then((n) => n && setNote(`Saved ${n}`)).catch((e) => setNote(String(e)))}
        >
          Save labels{mine?.dirty ? ' •' : ''}
        </button>
        <button className="btn" onClick={() => fileRef.current?.click()}>Load labels</button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        className="hidden"
        aria-label="Load labels file"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (f) void loadLabels(photo.id, f).then((n) => setNote(`Loaded ${n} labels.`)).catch((err) => setNote(err instanceof Error ? err.message : String(err)));
        }}
      />
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <input className="rounded bg-ink-850 px-2 py-1 text-ink-100" placeholder="Annotator" aria-label="Annotator" value={mine?.annotator ?? ''} onChange={(e) => st.setPeople(photo.id, { annotator: e.target.value || null })} />
        <input className="rounded bg-ink-850 px-2 py-1 text-ink-100" placeholder="Reviewer" aria-label="Reviewer" value={mine?.reviewer ?? ''} onChange={(e) => st.setPeople(photo.id, { reviewer: e.target.value || null })} />
      </div>
      {note && <p className="text-[11px] text-ink-300" data-testid="qa-label-note">{note}</p>}
      {labels.length > 0 && (
        <button className="text-[11px] text-ink-400 hover:text-ink-200" onClick={() => window.confirm('Delete every label on this photo?') && st.clear(photo.id)}>
          Clear all labels
        </button>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded bg-ink-850 px-1.5 py-1">
      <div className="text-ink-400">{label}</div>
      <div className="font-mono text-ink-100">{value}</div>
    </div>
  );
}
