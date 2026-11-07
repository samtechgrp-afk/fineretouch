'use client';

/** Small Radix-based controls shared by the editor's side panels. */
import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import * as ToggleGroup from '@radix-ui/react-toggle-group';

export function Heading({ title, hint, right }: { title: string; hint?: string; right?: React.ReactNode }) {
  return (
    <div className="mb-2 flex items-start justify-between gap-2">
      <div>
        <h3 className="text-xs font-medium text-ink-100">{title}</h3>
        {hint && <p className="mt-0.5 text-[11px] leading-snug text-ink-400">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

export function Toggle({ checked, onChange, label, disabled }: { checked: boolean; onChange(v: boolean): void; label: string; disabled?: boolean }) {
  return (
    <Switch.Root
      checked={checked}
      onCheckedChange={onChange}
      disabled={disabled}
      aria-label={label}
      className="relative h-[18px] w-8 shrink-0 rounded-full bg-ink-700 transition-colors data-[state=checked]:bg-apricot-400 disabled:opacity-40"
    >
      <Switch.Thumb className="block h-3.5 w-3.5 translate-x-0.5 rounded-full bg-ink-100 shadow transition-transform data-[state=checked]:translate-x-[15px] data-[state=checked]:bg-ink-950" />
    </Switch.Root>
  );
}

export function Segmented<T extends string>(props: { value: T; options: Array<[T, string]>; onChange(v: T): void; label: string; disabled?: boolean }) {
  return (
    <ToggleGroup.Root
      type="single"
      value={props.value}
      onValueChange={(v) => v && props.onChange(v as T)}
      aria-label={props.label}
      disabled={props.disabled}
      className="flex gap-0.5 rounded-md bg-ink-850 p-0.5"
    >
      {props.options.map(([v, label]) => (
        <ToggleGroup.Item key={v} value={v} className="seg flex-1 data-[state=on]:bg-ink-700 data-[state=on]:text-ink-100 disabled:opacity-40">
          {label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

export function RangeRow(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  format?: (v: number) => string;
  disabled?: boolean;
  /** Tooltip on the label. */
  hint?: string;
  onStart(): void;
  onChange(v: number): void;
}) {
  return (
    <div className={props.disabled ? 'opacity-40' : ''}>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-ink-300" title={props.hint}>{props.label}</span>
        <span className="font-mono tabular-nums text-ink-400">{props.format ? props.format(props.value) : Math.round(props.value)}</span>
      </div>
      <Slider.Root
        className="relative flex h-4 w-full touch-none select-none items-center"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={[props.value]}
        disabled={props.disabled}
        onPointerDown={props.onStart}
        onKeyDown={(e) => !e.repeat && e.key.startsWith('Arrow') && props.onStart()}
        onValueChange={([v]) => props.onChange(v)}
        aria-label={props.label}
      >
        <Slider.Track className="relative h-[3px] grow rounded-full bg-ink-700">
          <Slider.Range className="absolute h-full rounded-full bg-apricot-400" />
        </Slider.Track>
        <Slider.Thumb aria-label={props.label} className="block h-3.5 w-3.5 rounded-full border-2 border-ink-900 bg-ink-100 shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-apricot-400" />
      </Slider.Root>
    </div>
  );
}


export function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-ink-850 px-1.5 py-1">
      <dt className="text-ink-400">{label}</dt>
      <dd className="font-mono text-ink-100">{value}</dd>
    </div>
  );
}
