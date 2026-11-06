'use client';

import * as Slider from '@radix-ui/react-slider';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import {
  DEFAULT_COLOR_GRADING,
  DEFAULT_HSL,
  GRADE_ZONES,
  HSL_CENTERS,
  HSL_CHANNELS,
  HSL_HUE_RANGE,
  HSL_LABELS,
  gradingIsDefault,
  hslIsDefault,
  type ColorGrading,
  type GradeWheel,
  type GradeZone,
  type HslChannel,
  type HslValue,
} from '@fineretouch/shared';
import { useRef, useState } from 'react';
import { hsvToRgb } from '@/engine/colorMixer';
import { useEditor, type Photo } from '@/lib/store';

type MixerProp = keyof HslValue;
const PROP_LABEL: Record<MixerProp, string> = { hue: 'Hue', saturation: 'Saturation', luminance: 'Luminance' };
const ZONE_LABEL: Record<GradeZone, string> = { shadows: 'Shadows', midtones: 'Midtones', highlights: 'Highlights' };

const css = (h: number, s = 1, v = 1) => {
  const [r, g, b] = hsvToRgb(h, s, v).map((x) => Math.round(x * 255));
  return `rgb(${r} ${g} ${b})`;
};

/**
 * 8-channel HSL colour mixer and 3-way colour grading wheels. Edits preview live on the
 * GPU (RETOUCH_FRAG) and export through the same maths (services/ai/app/color_engine.py).
 * Drags are live; the undo step is taken when a drag starts.
 */
export function ColorMixerPanel({ photo }: { photo: Photo }) {
  return (
    <div className="space-y-5">
      <HslMixerControls photo={photo} />
      <ColorGradingControls photo={photo} />
    </div>
  );
}

// --- HSL mixer -----------------------------------------------------------------

function HslMixerControls({ photo }: { photo: Photo }) {
  const [prop, setProp] = useState<MixerProp>('hue');
  const beginEdit = useEditor((s) => s.beginEdit);
  const patchAdjustments = useEditor((s) => s.patchAdjustments);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const hsl = photo.adjustments.hsl;

  const set = (ch: HslChannel, v: number) => {
    const cur = useEditor.getState().photos[photo.id]?.adjustments.hsl ?? hsl;
    patchAdjustments({ hsl: { ...cur, [ch]: { ...cur[ch], [prop]: v } } }, photo.id);
  };

  return (
    <section aria-label="Colour mixer">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-medium text-ink-100">Colour mixer</h3>
        <button className="btn-ghost px-1.5 py-0.5 text-[11px]" disabled={hslIsDefault(hsl)} onClick={() => updateAdjustments({ hsl: DEFAULT_HSL }, [photo.id])}>
          Reset
        </button>
      </div>
      <ToggleGroup.Root
        type="single"
        value={prop}
        onValueChange={(v) => v && setProp(v as MixerProp)}
        aria-label="Mixer property"
        className="mb-3 flex gap-0.5 rounded-md bg-ink-850 p-0.5"
      >
        {(Object.keys(PROP_LABEL) as MixerProp[]).map((p) => (
          <ToggleGroup.Item key={p} value={p} className="seg flex-1 data-[state=on]:bg-ink-700 data-[state=on]:text-ink-100">
            {PROP_LABEL[p]}
          </ToggleGroup.Item>
        ))}
      </ToggleGroup.Root>
      <div className="space-y-2.5">
        {HSL_CHANNELS.map((ch) => (
          <ChannelSlider
            key={ch}
            channel={ch}
            prop={prop}
            value={hsl[ch][prop]}
            onStart={() => beginEdit(photo.id)}
            onChange={(v) => set(ch, v)}
            onReset={() => updateAdjustments({ hsl: { ...hsl, [ch]: { ...hsl[ch], [prop]: 0 } } }, [photo.id])}
          />
        ))}
      </div>
    </section>
  );
}

function trackGradient(ch: HslChannel, prop: MixerProp): string {
  const h = HSL_CENTERS[ch];
  if (prop === 'hue') return `linear-gradient(90deg, ${css(h - HSL_HUE_RANGE, 0.85, 0.9)}, ${css(h, 0.85, 0.9)}, ${css(h + HSL_HUE_RANGE, 0.85, 0.9)})`;
  if (prop === 'saturation') return `linear-gradient(90deg, ${css(h, 0, 0.75)}, ${css(h, 1, 0.9)})`;
  return `linear-gradient(90deg, ${css(h, 0.9, 0.25)}, ${css(h, 0.9, 0.7)}, ${css(h, 0.35, 1)})`;
}

function ChannelSlider(props: { channel: HslChannel; prop: MixerProp; value: number; onStart(): void; onChange(v: number): void; onReset(): void }) {
  const { channel, prop, value } = props;
  const label = `${HSL_LABELS[channel]} ${PROP_LABEL[prop].toLowerCase()}`;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-ink-300" onDoubleClick={props.onReset} title="Double-click to reset">
          <span className="h-2 w-2 rounded-full" style={{ background: css(HSL_CENTERS[channel], 0.85, 0.95) }} aria-hidden />
          {HSL_LABELS[channel]}
        </span>
        <span className={`font-mono tabular-nums ${value !== 0 ? 'text-apricot-300' : 'text-ink-400'}`}>
          {value > 0 ? '+' : ''}
          {Math.round(value)}
        </span>
      </div>
      <Slider.Root
        className="relative flex h-4 w-full touch-none select-none items-center"
        min={-100}
        max={100}
        step={1}
        value={[value]}
        onPointerDown={props.onStart}
        onKeyDown={(e) => !e.repeat && e.key.startsWith('Arrow') && props.onStart()}
        onValueChange={([v]) => props.onChange(v)}
        aria-label={label}
      >
        <Slider.Track className="relative h-[5px] grow rounded-full" style={{ background: trackGradient(channel, prop) }}>
          <Slider.Range className="absolute h-full rounded-full bg-transparent" />
        </Slider.Track>
        <Slider.Thumb
          aria-label={label}
          className="block h-3.5 w-3.5 rounded-full border-2 border-ink-900 bg-ink-100 shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-apricot-400"
          onDoubleClick={props.onReset}
        />
      </Slider.Root>
    </div>
  );
}

// --- colour grading ------------------------------------------------------------

function ColorGradingControls({ photo }: { photo: Photo }) {
  const beginEdit = useEditor((s) => s.beginEdit);
  const patchAdjustments = useEditor((s) => s.patchAdjustments);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const g = photo.adjustments.colorGrading;

  const live = (patch: Partial<ColorGrading>) => {
    const cur = useEditor.getState().photos[photo.id]?.adjustments.colorGrading ?? g;
    patchAdjustments({ colorGrading: { ...cur, ...patch } }, photo.id);
  };
  const setWheel = (z: GradeZone, w: Partial<GradeWheel>) => {
    const cur = useEditor.getState().photos[photo.id]?.adjustments.colorGrading ?? g;
    live({ [z]: { ...cur[z], ...w } });
  };

  return (
    <section aria-label="Colour grading">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <h3 className="text-xs font-medium text-ink-100">Colour grading</h3>
          <p className="mt-0.5 text-[11px] leading-snug text-ink-400">Drag a puck to tint shadows, midtones or highlights. Double-click a wheel to clear it.</p>
        </div>
        <button className="btn-ghost shrink-0 px-1.5 py-0.5 text-[11px]" disabled={gradingIsDefault(g) && g.blending === 50 && g.balance === 0} onClick={() => updateAdjustments({ colorGrading: DEFAULT_COLOR_GRADING }, [photo.id])}>
          Reset
        </button>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {GRADE_ZONES.map((z) => (
          <div key={z} className="flex min-w-0 flex-col items-center gap-1.5">
            <ColorWheel
              label={ZONE_LABEL[z]}
              wheel={g[z]}
              onStart={() => beginEdit(photo.id)}
              onChange={(w) => setWheel(z, w)}
              onReset={() => updateAdjustments({ colorGrading: { ...g, [z]: { ...g[z], hue: 0, saturation: 0 } } }, [photo.id])}
            />
            <span className="text-[11px] text-ink-300">{ZONE_LABEL[z]}</span>
            <span className="font-mono text-[10px] tabular-nums text-ink-400">
              {g[z].saturation > 0 ? `${Math.round(g[z].hue)}° · ${Math.round(g[z].saturation)}` : '—'}
            </span>
            <CompactSlider
              label={`${ZONE_LABEL[z]} luminance`}
              value={g[z].luminance}
              min={-100}
              max={100}
              onStart={() => beginEdit(photo.id)}
              onChange={(luminance) => setWheel(z, { luminance })}
            />
          </div>
        ))}
      </div>
      <div className="mt-3 space-y-2.5">
        <LabeledSlider label="Blending" value={g.blending} min={0} max={100} onStart={() => beginEdit(photo.id)} onChange={(blending) => live({ blending })} />
        <LabeledSlider label="Balance" value={g.balance} min={-100} max={100} onStart={() => beginEdit(photo.id)} onChange={(balance) => live({ balance })} />
      </div>
    </section>
  );
}

/**
 * Hue / saturation wheel: angle = hue (red at 3 o'clock, counter-clockwise like a colour
 * circle), distance from the centre = saturation. Arrow keys: left/right hue, up/down saturation.
 */
function ColorWheel(props: { label: string; wheel: GradeWheel; onStart(): void; onChange(w: Partial<GradeWheel>): void; onReset(): void }) {
  const ref = useRef<HTMLDivElement>(null);
  const { hue, saturation } = props.wheel;
  const r = saturation / 100;
  const px = 50 + Math.cos((hue * Math.PI) / 180) * r * 50;
  const py = 50 - Math.sin((hue * Math.PI) / 180) * r * 50;

  const fromPointer = (e: React.PointerEvent) => {
    const box = ref.current!.getBoundingClientRect();
    const x = (e.clientX - box.left) / box.width - 0.5;
    const y = 0.5 - (e.clientY - box.top) / box.height;
    const dist = Math.min(1, Math.hypot(x, y) / 0.5);
    let h = (Math.atan2(y, x) * 180) / Math.PI;
    if (h < 0) h += 360;
    props.onChange({ hue: Math.round(h), saturation: Math.round(dist * 100) });
  };

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label={`${props.label} colour`}
      aria-valuetext={saturation > 0 ? `hue ${Math.round(hue)} degrees, saturation ${Math.round(saturation)}` : 'neutral'}
      aria-valuenow={Math.round(saturation)}
      aria-valuemin={0}
      aria-valuemax={100}
      className="relative aspect-square w-full max-w-[84px] cursor-crosshair touch-none rounded-full shadow-inner outline-offset-2"
      style={{
        background: `radial-gradient(circle closest-side, rgb(128 128 128) 0%, rgb(128 128 128 / 0) 100%), conic-gradient(from 90deg, ${[0, 300, 240, 180, 120, 60, 0].map((h) => css(h, 0.85, 0.85)).join(', ')})`,
      }}
      onPointerDown={(e) => {
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        props.onStart();
        fromPointer(e);
      }}
      onPointerMove={(e) => e.buttons & 1 && fromPointer(e)}
      onDoubleClick={props.onReset}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 10 : 2;
        const keys: Record<string, Partial<GradeWheel>> = {
          ArrowLeft: { hue: (hue + 360 - step) % 360 },
          ArrowRight: { hue: (hue + step) % 360 },
          ArrowUp: { saturation: Math.min(100, saturation + step) },
          ArrowDown: { saturation: Math.max(0, saturation - step) },
        };
        if (keys[e.key]) {
          e.preventDefault();
          if (!e.repeat) props.onStart();
          props.onChange(keys[e.key]);
        } else if (e.key === 'Backspace' || e.key === 'Delete') props.onReset();
      }}
    >
      <span
        className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.6)]"
        style={{ left: `${px}%`, top: `${py}%`, background: saturation > 0 ? css(hue, r, 1) : 'transparent' }}
      />
    </div>
  );
}

function CompactSlider(props: { label: string; value: number; min: number; max: number; onStart(): void; onChange(v: number): void }) {
  return (
    <Slider.Root
      className="relative flex h-4 w-full touch-none select-none items-center"
      min={props.min}
      max={props.max}
      step={1}
      value={[props.value]}
      onPointerDown={props.onStart}
      onKeyDown={(e) => !e.repeat && e.key.startsWith('Arrow') && props.onStart()}
      onValueChange={([v]) => props.onChange(v)}
      aria-label={props.label}
      title={`${props.label}: ${Math.round(props.value)}`}
    >
      <Slider.Track className="relative h-[3px] grow rounded-full" style={{ background: 'linear-gradient(90deg, #0d0e10, #e4e4e7)' }} />
      <Slider.Thumb aria-label={props.label} className="block h-3 w-3 rounded-full border-2 border-ink-900 bg-ink-100 shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-apricot-400" />
    </Slider.Root>
  );
}

function LabeledSlider(props: { label: string; value: number; min: number; max: number; onStart(): void; onChange(v: number): void }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-ink-300">{props.label}</span>
        <span className="font-mono tabular-nums text-ink-400">
          {props.min < 0 && props.value > 0 ? '+' : ''}
          {Math.round(props.value)}
        </span>
      </div>
      <CompactSlider {...props} />
    </div>
  );
}
