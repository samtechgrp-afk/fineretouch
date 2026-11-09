'use client';

import * as Slider from '@radix-ui/react-slider';
import { DEFAULT_ADJUSTMENTS, GROUP_LABELS, SLIDERS, effectForSlider, type AdjustmentGroup, type Adjustments, type SliderSpec } from '@fineretouch/shared';
import { Fragment, useState } from 'react';
import { selectActivePhoto, useEditor } from '@/lib/store';
import { ColorMixerPanel } from './ColorMixerPanel';
import { RetouchControlPanel } from './RetouchControlPanel';
import { ProToolsPanel } from './panels/ProToolsPanel';
import { BatchSyncPanel } from './panels/BatchSyncPanel';
import { EffectControlsRow, EffectSwitch, GlobalStrength } from './panels/EffectControls';
import { RetouchStudioPanel } from './panels/RetouchStudioPanel';
import { AssistantPanel } from './panels/AssistantPanel';
import { SkinPerfectionPanel } from './panels/SkinPerfectionPanel';
import { QaLabelsPanel } from './panels/QaLabelsPanel';

const GROUP_ORDER: AdjustmentGroup[] = ['skin', 'face', 'light', 'color', 'detail'];

export function AdjustmentPanel() {
  const photo = useEditor(selectActivePhoto);
  const [open, setOpen] = useState<Record<string, boolean>>({ studio: true, batch: false, pro: true, skin: true, face: true, light: true, color: true, mixer: false, detail: false });
  if (!photo) return <aside className="p-4 text-sm text-ink-400">Select a photo to edit.</aside>;

  const faceState =
    photo.analysisState === 'pending'
      ? 'Detecting faces…'
      : photo.analysisState === 'offline'
        ? 'AI service offline: face tools need it'
        : photo.analysisState === 'error'
          ? 'Face detection failed'
          : photo.analysis && !photo.analysis.faces.length
            ? 'No face found in this photo'
            : photo.analysis
              ? `${photo.analysis.faces.length} face${photo.analysis.faces.length > 1 ? 's' : ''} detected`
              : '';
  const hasFace = Boolean(photo.analysis?.faces.length);

  return (
    <aside className="flex h-full min-w-0 flex-col overflow-y-auto overflow-x-hidden" aria-label="Adjustments">
      <Section title="Retouch strength" open={open.strength ?? true} onToggle={() => setOpen((o) => ({ ...o, strength: !(o.strength ?? true) }))}
        note={photo.restored ? 'Edits restored from your last session.' : undefined}>
        <GlobalStrength adj={photo.adjustments} />
      </Section>
      <Section title="Skin Perfection Studio" open={open.perfection ?? true} onToggle={() => setOpen((o) => ({ ...o, perfection: !(o.perfection ?? true) }))}
        note={photo.adjustments.skinPerfection.enabled ? 'Full-resolution blemish engine (AI engine)' : undefined}>
        <SkinPerfectionPanel photo={photo} />
      </Section>
      <Section title="Test labels (QA)" open={open.qa ?? false} onToggle={() => setOpen((o) => ({ ...o, qa: !(o.qa ?? false) }))}
        note="Mark blemishes and natural features for the quality benchmark">
        <QaLabelsPanel photo={photo} />
      </Section>
      <Section title="Retouch assistant" open={open.assistant ?? true} onToggle={() => setOpen((o) => ({ ...o, assistant: !(o.assistant ?? true) }))}
        note="Runs on this computer: nothing is uploaded">
        <AssistantPanel photo={photo} />
      </Section>
      <Section title="Pro studio retouch" open={open.studio} onToggle={() => setOpen((o) => ({ ...o, studio: !o.studio }))} note={faceState}>
        <RetouchStudioPanel photo={photo} />
      </Section>
      <Section title="Batch style sync" open={open.batch} onToggle={() => setOpen((o) => ({ ...o, batch: !o.batch }))}>
        <BatchSyncPanel photo={photo} />
      </Section>
      <Section title="Healing, clone & layers" open={open.tools} onToggle={() => setOpen((o) => ({ ...o, tools: !o.tools }))}
        note={photo.adjustments.localRetouch.heal.length || photo.adjustments.localRetouch.layers.length ? 'Pro tools in use on this photo' : undefined}>
        <ProToolsPanel photo={photo} />
      </Section>
      <Section title="AI pro tools" open={open.pro} onToggle={() => setOpen((o) => ({ ...o, pro: !o.pro }))}>
        <RetouchControlPanel photo={photo} />
      </Section>
      {GROUP_ORDER.map((g) => (
        <Fragment key={g}>
          <Section title={GROUP_LABELS[g]} open={open[g]} onToggle={() => setOpen((o) => ({ ...o, [g]: !o[g] }))}
            note={g === 'face' ? faceState : g === 'skin' && !hasFace ? 'Using colour-based skin detection' : undefined}>
            {SLIDERS.filter((s) => s.group === g).map((s) => (
              <AdjustmentSlider key={s.key} spec={s} adj={photo.adjustments} disabled={Boolean(s.needsFace && !hasFace)} />
            ))}
          </Section>
          {g === 'color' && (
            <Section title="Colour mixer & grading" open={open.mixer} onToggle={() => setOpen((o) => ({ ...o, mixer: !o.mixer }))}>
              <ColorMixerPanel photo={photo} />
            </Section>
          )}
        </Fragment>
      ))}
    </aside>
  );
}

function Section(props: { title: string; open: boolean; onToggle(): void; note?: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-ink-800">
      <button
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-ink-300 hover:text-ink-100"
        onClick={props.onToggle}
        aria-expanded={props.open}
      >
        {props.title}
        <span className={`transition-transform ${props.open ? 'rotate-90' : ''}`} aria-hidden>
          ›
        </span>
      </button>
      {props.open && (
        <div className="space-y-3 px-4 pb-4">
          {props.note && <p className="text-[11px] text-ink-400">{props.note}</p>}
          {props.children}
        </div>
      )}
    </section>
  );
}

function AdjustmentSlider({ spec, adj, disabled }: { spec: SliderSpec; adj: Adjustments; disabled: boolean }) {
  const value = adj[spec.key];
  const effect = effectForSlider(spec.key);
  const effectOff = effect !== null && !adj.effects[effect].enabled;
  const beginEdit = useEditor((s) => s.beginEdit);
  const setAdjustment = useEditor((s) => s.setAdjustment);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const def = DEFAULT_ADJUSTMENTS[spec.key];
  const id = `slider-${spec.key}`;
  const bipolar = spec.min < 0;

  return (
    <div className={disabled ? 'opacity-40' : ''}>
      <div className="mb-1 flex items-center justify-between gap-2 text-xs">
        {effect && <EffectSwitch id={effect} adj={adj} disabled={disabled} />}
        <label
          htmlFor={id}
          className={`mr-auto cursor-pointer ${effectOff ? 'text-ink-500 line-through' : 'text-ink-300'}`}
          title={spec.hint ? `${spec.hint} Double-click to reset.` : 'Double-click to reset'}
          onDoubleClick={() => !disabled && updateAdjustments({ [spec.key]: def }, [useEditor.getState().activeId!])}
        >
          {spec.label}
        </label>
        <span className={`font-mono tabular-nums ${value !== def ? 'text-apricot-300' : 'text-ink-400'}`}>
          {value > 0 && bipolar ? '+' : ''}
          {Math.round(value)}
          {spec.unit}
        </span>
      </div>
      <Slider.Root
        id={id}
        className="relative flex h-4 w-full touch-none select-none items-center"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={[value]}
        disabled={disabled}
        onPointerDown={() => beginEdit()}
        onKeyDown={(e) => {
          if (!e.repeat && e.key.startsWith('Arrow')) beginEdit();
        }}
        onValueChange={([v]) => setAdjustment(spec.key, v)}
        aria-label={spec.label}
      >
        <Slider.Track className="relative h-[3px] grow rounded-full bg-ink-700">
          <Slider.Range className={`absolute h-full rounded-full ${bipolar ? 'bg-transparent' : 'bg-apricot-400'}`} />
          {bipolar && <CentreFill value={value} min={spec.min} max={spec.max} />}
        </Slider.Track>
        <Slider.Thumb
          aria-label={spec.label}
          className="block h-3.5 w-3.5 rounded-full border-2 border-ink-900 bg-ink-100 shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-apricot-400"
          onDoubleClick={() => updateAdjustments({ [spec.key]: def }, [useEditor.getState().activeId!])}
        />
      </Slider.Root>
      {effect && <EffectControlsRow id={effect} adj={adj} disabled={disabled} />}
    </div>
  );
}

/** For -100..100 sliders, fill from the centre toward the value. */
function CentreFill({ value, min, max }: { value: number; min: number; max: number }) {
  const zero = ((0 - min) / (max - min)) * 100;
  const pos = ((value - min) / (max - min)) * 100;
  return (
    <span
      className="absolute h-full rounded-full bg-apricot-400"
      style={{ left: `${Math.min(zero, pos)}%`, width: `${Math.abs(pos - zero)}%` }}
    />
  );
}
