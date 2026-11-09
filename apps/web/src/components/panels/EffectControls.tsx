'use client';

/**
 * Per-effect controls under a retouch slider: on/off, 1x / 2x / 3x, reset, and an expandable
 * advanced row with opacity and the effective strength the engines will actually render.
 * Every control here changes the rendered result (see packages/shared/src/effects.ts).
 */
import {
  EFFECT_CURVES,
  EFFECT_MULTIPLIERS,
  EFFECT_SOURCE,
  effectIsDefault,
  effectiveStrength,
  type Adjustments,
  type EffectId,
  type EffectMultiplier,
} from '@fineretouch/shared';
import { useState } from 'react';
import { useEditor } from '@/lib/store';
import { RangeRow, Toggle } from './controls';

export function sliderValue(adj: Adjustments, id: EffectId): number {
  const src = EFFECT_SOURCE[id];
  if (src.length === 1) return adj[src[0]] as number;
  return (adj[src[0]] as unknown as Record<string, number>)[src[1]];
}

export function EffectSwitch({ id, adj, disabled }: { id: EffectId; adj: Adjustments; disabled?: boolean }) {
  const setEffectControl = useEditor((s) => s.setEffectControl);
  const c = adj.effects[id];
  return (
    <Toggle
      label={`${EFFECT_CURVES[id].label} on`}
      checked={c.enabled}
      disabled={disabled}
      onChange={(v) => setEffectControl(id, { enabled: v })}
    />
  );
}

export function EffectControlsRow({ id, adj, disabled }: { id: EffectId; adj: Adjustments; disabled?: boolean }) {
  const setEffectControl = useEditor((s) => s.setEffectControl);
  const resetEffect = useEditor((s) => s.resetEffect);
  const beginEdit = useEditor((s) => s.beginEdit);
  const [open, setOpen] = useState(false);
  const c = adj.effects[id];
  const curve = EFFECT_CURVES[id];
  const value = sliderValue(adj, id);
  const eff = effectiveStrength(value, c, curve, adj.globalStrength);
  const changed = !effectIsDefault(c) || adj.globalStrength !== 100;
  const off = disabled || !c.enabled;

  return (
    <div className="mt-1.5">
      <div className="flex items-center gap-1.5">
        <div role="group" aria-label={`${curve.label} multiplier`} className="flex gap-0.5 rounded-md bg-ink-850 p-0.5" title={curve.note}>
          {EFFECT_MULTIPLIERS.map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={c.multiplier === m}
              disabled={off}
              onClick={() => setEffectControl(id, { multiplier: m as EffectMultiplier })}
              className="seg px-1.5 text-[11px] disabled:opacity-40"
            >
              {m}×
            </button>
          ))}
        </div>
        <span
          className={`ml-auto font-mono text-[11px] tabular-nums ${changed ? 'text-apricot-300' : 'text-ink-500'}`}
          title="Effective strength the engine renders: slider × multiplier curve × opacity × global strength"
        >
          {off ? 'off' : `= ${Math.round(eff)}%`}
        </span>
        <button
          type="button"
          className="btn-ghost px-1 text-[11px]"
          aria-expanded={open}
          aria-label={`${curve.label} advanced`}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? '▾' : '▸'}
        </button>
        <button
          type="button"
          className="btn-ghost px-1 text-[11px]"
          disabled={disabled}
          aria-label={`Reset ${curve.label}`}
          title="Reset this effect (slider, multiplier, opacity)"
          onClick={() => resetEffect(id)}
        >
          ↺
        </button>
      </div>
      {open && (
        <div className="mt-2 space-y-2 rounded-md bg-ink-850/60 p-2">
          <RangeRow
            label="Opacity"
            value={c.opacity}
            min={0}
            max={100}
            disabled={off}
            format={(v) => `${Math.round(v)}%`}
            hint="Share of this effect's contribution that is applied, without changing its slider."
            onStart={() => beginEdit()}
            onChange={(v) => setEffectControl(id, { opacity: v }, { live: true })}
          />
          <p className="text-[11px] leading-snug text-ink-400">{curve.note}</p>
        </div>
      )}
    </div>
  );
}

export function GlobalStrength({ adj }: { adj: Adjustments }) {
  const setGlobalStrength = useEditor((s) => s.setGlobalStrength);
  const beginEdit = useEditor((s) => s.beginEdit);
  return (
    <RangeRow
      label="Global retouch strength"
      value={adj.globalStrength}
      min={0}
      max={100}
      format={(v) => `${Math.round(v)}%`}
      hint="Scales every retouch effect's final contribution together. Your individual sliders and multipliers are kept."
      onStart={() => beginEdit()}
      onChange={(v) => setGlobalStrength(v, { live: true })}
    />
  );
}
