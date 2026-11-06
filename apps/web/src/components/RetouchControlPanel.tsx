'use client';

import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { SKIN_MULTIPLIERS, mapStyleToAdjustments, type ApertureShape, type SkinMultiplier, type StyleSyncMode } from '@fineretouch/shared';
import { useRef, useState } from 'react';
import { extractPalette } from '@/engine/color';
import { profileFromRgba } from '@/engine/style';
import { bitmapPixels, decodeUpright, fitSize, newId } from '@/lib/images';
import { IMPORT_ACCEPT, isRawFileName } from '@/lib/rawParser';
import { useEditor, type DepthState, type Photo } from '@/lib/store';

/**
 * Controls for the v0.2 retouch features: skin boost multiplier, stray-hair cleanup,
 * depth bokeh, style sync from a reference photo, and the AI assist actions.
 */
export function RetouchControlPanel({ photo }: { photo: Photo }) {
  const hasFace = Boolean(photo.analysis?.faces.length);
  return (
    <div className="space-y-5">
      <AiAssist photo={photo} />
      <SkinBoost photo={photo} />
      <StrayHair photo={photo} hasFace={hasFace} />
      <Bokeh photo={photo} />
      <StyleSync />
    </div>
  );
}

// --- building blocks ---------------------------------------------------------

function Heading({ title, hint, right }: { title: string; hint?: string; right?: React.ReactNode }) {
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

function Toggle({ checked, onChange, label, disabled }: { checked: boolean; onChange(v: boolean): void; label: string; disabled?: boolean }) {
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

function Segmented<T extends string>(props: { value: T; options: Array<[T, string]>; onChange(v: T): void; label: string; disabled?: boolean }) {
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

function RangeRow(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  format?: (v: number) => string;
  disabled?: boolean;
  onStart(): void;
  onChange(v: number): void;
}) {
  return (
    <div className={props.disabled ? 'opacity-40' : ''}>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-ink-300">{props.label}</span>
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
        <Slider.Thumb className="block h-3.5 w-3.5 rounded-full border-2 border-ink-900 bg-ink-100 shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-apricot-400" />
      </Slider.Root>
    </div>
  );
}

// --- sections ----------------------------------------------------------------

function AiAssist({ photo }: { photo: Photo }) {
  const autoRetouch = useEditor((s) => s.autoRetouch);
  const smartSync = useEditor((s) => s.smartSync);
  const selected = useEditor((s) => s.selected);
  const total = useEditor((s) => s.order.length);
  const [note, setNote] = useState<string | null>(null);
  const others = selected.length > 1 ? selected.filter((id) => id !== photo.id).length : total - 1;
  return (
    <section>
      <Heading title="AI assist" hint="Auto Retouch measures this photo and sets a natural starting point. Smart Sync copies your edits but evens out exposure and white balance per photo." />
      <div className="grid grid-cols-2 gap-2">
        <button
          className="btn-primary"
          disabled={!photo.profile}
          onClick={() => setNote(`Auto retouched ${autoRetouch()} photo(s).`)}
          title="Exposure, contrast, white balance, skin and eyes from this photo's own measurements"
        >
          Auto retouch
        </button>
        <button className="btn" disabled={others < 1} onClick={() => setNote(`Smart-synced ${smartSync()} photo(s).`)}>
          Smart sync {others > 0 ? `(${others})` : ''}
        </button>
      </div>
      {note && <p className="mt-1.5 text-[11px] text-mint-400">{note}</p>}
    </section>
  );
}

function SkinBoost({ photo }: { photo: Photo }) {
  const setSkinMultiplier = useEditor((s) => s.setSkinMultiplier);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const value = photo.adjustments.skinMultiplier;
  const keepMarks = photo.adjustments.protectBeautyMarks;
  return (
    <section>
      <Heading
        title="Skin & blemish boost"
        hint="Sets how far Spot healing reaches: 2x and 3x catch fainter and larger acne and dark spots, and add recursive clean-up of stubborn redness while keeping pore texture. Applies to selected photos."
      />
      <Segmented<`${SkinMultiplier}`>
        label="Skin retouch multiplier"
        value={`${value}`}
        options={SKIN_MULTIPLIERS.map((m) => [`${m}`, `${m}x`])}
        onChange={(v) => setSkinMultiplier(Number(v) as SkinMultiplier)}
      />
      <div className="mt-2.5 flex items-center justify-between gap-2">
        <span className="text-xs text-ink-300" title="Spot healing leaves moles and beauty marks in place. Eyes, brows, lips and nostrils are always protected.">
          Keep beauty marks
        </span>
        <Toggle checked={keepMarks} onChange={(v) => updateAdjustments({ protectBeautyMarks: v })} label="Keep beauty marks" />
      </div>
    </section>
  );
}

function StrayHair({ photo, hasFace }: { photo: Photo; hasFace: boolean }) {
  const setStray = useEditor((s) => s.setStrayHairRemoval);
  const beginEdit = useEditor((s) => s.beginEdit);
  const st = photo.adjustments.strayHairRemoval;
  return (
    <section>
      <Heading
        title="Stray hair cleanup"
        hint={hasFace ? 'Removes flyaways around the face outline and forehead. Export runs the full strand detector and inpainting.' : 'Needs a detected face.'}
        right={<Toggle label="Stray hair cleanup" checked={st.enabled} disabled={!hasFace} onChange={(v) => setStray({ enabled: v })} />}
      />
      <RangeRow
        label="Strength"
        value={st.strength}
        min={0}
        max={100}
        disabled={!st.enabled || !hasFace}
        onStart={() => beginEdit()}
        onChange={(v) => setStray({ strength: v }, { live: true })}
      />
    </section>
  );
}

const DEPTH_LABEL: Record<DepthState, string> = {
  none: '',
  pending: 'Estimating depth…',
  model: 'Depth from AI model',
  prior: 'Depth estimated from the face (start the AI service for a full depth map)',
  error: 'Depth unavailable',
};

function Bokeh({ photo }: { photo: Photo }) {
  const setBokeh = useEditor((s) => s.setDepthBokeh);
  const beginEdit = useEditor((s) => s.beginEdit);
  const b = photo.adjustments.depthBokeh;
  const off = !b.enabled;
  return (
    <section>
      <Heading
        title="Background bokeh"
        hint={b.enabled ? DEPTH_LABEL[photo.depthState] : 'Depth-aware lens blur that keeps the subject sharp.'}
        right={<Toggle label="Background bokeh" checked={b.enabled} onChange={(v) => setBokeh({ enabled: v })} />}
      />
      <div className="space-y-3">
        <RangeRow
          label="Blur radius"
          value={b.blurRadius}
          min={0}
          max={100}
          disabled={off}
          onStart={() => beginEdit()}
          onChange={(v) => setBokeh({ blurRadius: v }, { live: true })}
        />
        <RangeRow
          label="Bokeh multiplier"
          value={b.bokehMultiplier}
          min={100}
          max={300}
          step={5}
          format={(v) => `${Math.round(v)}%`}
          disabled={off}
          onStart={() => beginEdit()}
          onChange={(v) => setBokeh({ bokehMultiplier: v }, { live: true })}
        />
        <div className="flex gap-1">
          {[100, 200, 300].map((m) => (
            <button key={m} className="seg flex-1 border border-ink-800" aria-pressed={b.bokehMultiplier === m} disabled={off} onClick={() => setBokeh({ bokehMultiplier: m })}>
              {m / 100}x
            </button>
          ))}
        </div>
        <RangeRow
          label="Highlight bloom"
          value={b.highlightBoost}
          min={0}
          max={100}
          disabled={off}
          onStart={() => beginEdit()}
          onChange={(v) => setBokeh({ highlightBoost: v }, { live: true })}
        />
        <div>
          <p className="mb-1 text-xs text-ink-300">Aperture</p>
          <Segmented<ApertureShape>
            label="Aperture shape"
            value={b.apertureShape}
            options={[
              ['circle', 'Circular'],
              ['hex', 'Hexagonal'],
            ]}
            disabled={off}
            onChange={(v) => setBokeh({ apertureShape: v })}
          />
        </div>
      </div>
    </section>
  );
}

function StyleSync() {
  const style = useEditor((s) => s.styleSync);
  const setStyleReference = useEditor((s) => s.setStyleReference);
  const setOptions = useEditor((s) => s.setStyleSyncOptions);
  const applyStyleSync = useEditor((s) => s.applyStyleSync);
  const selected = useEditor((s) => s.selected);
  const total = useEditor((s) => s.order.length);
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const read = async (file: File) => {
    if (!file.type.startsWith('image/') && !isRawFileName(file.name)) return;
    setBusy(true);
    setNote(null);
    try {
      const { bitmap: bmp, preview } = await decodeUpright(file);
      const [w, h] = fitSize(bmp.width, bmp.height, 512);
      const px = bitmapPixels(bmp, 512);
      bmp.close();
      setStyleReference({
        referenceImageId: newId(),
        referenceName: file.name,
        referenceUrl: URL.createObjectURL(preview?.blob ?? file),
        profile: profileFromRgba(px, w, h),
        matchedPalette: extractPalette(px, 5),
      });
      // Instant preview on the photo being edited; "Sync to all" pushes it to the shoot.
      const s = useEditor.getState();
      if (s.activeId) {
        s.setStyleSyncOptions({ syncTargetBatch: false });
        useEditor.setState({ selected: [] });
        s.applyStyleSync();
        useEditor.setState({ selected: s.selected });
      }
    } catch {
      setNote('Could not read that image.');
    } finally {
      setBusy(false);
    }
  };

  const apply = (batch: boolean) => {
    setOptions({ syncTargetBatch: batch });
    // Options are applied synchronously by Zustand, so the sync sees the new target scope.
    const n = applyStyleSync();
    setNote(`Synced the reference look to ${n} photo${n === 1 ? '' : 's'}.`);
  };

  const p = style.profile;
  return (
    <section>
      <Heading title="Sync style from reference" hint="Drop any photo whose look you like. Colour, tone and retouch strength are measured and mapped onto your shots." />
      <input ref={inputRef} type="file" accept={IMPORT_ACCEPT} hidden onChange={(e) => e.target.files?.[0] && read(e.target.files[0])} />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation(); // keep the editor's import drop zone out of it
          setOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) void read(f);
        }}
        className={`flex w-full items-center gap-3 rounded-lg border border-dashed p-2 text-left text-xs transition-colors ${
          over ? 'border-apricot-400 bg-ink-800' : 'border-ink-700 hover:border-ink-600'
        }`}
      >
        {style.referenceUrl ? (
          <img src={style.referenceUrl} alt="Reference" className="h-12 w-12 shrink-0 rounded object-cover" />
        ) : (
          <span className="grid h-12 w-12 shrink-0 place-items-center rounded bg-ink-800 text-lg text-ink-400" aria-hidden>
            +
          </span>
        )}
        <span className="min-w-0">
          <span className="block truncate text-ink-100">{busy ? 'Reading…' : (style.referenceName ?? 'Drop a reference photo')}</span>
          <span className="text-ink-400">{style.referenceName ? 'Click or drop to replace' : 'or click to choose a file'}</span>
        </span>
      </button>

      {p && (
        <div className="mt-3 space-y-3">
          {style.matchedPalette.length > 0 && (
            <div className="flex h-5 overflow-hidden rounded" title="Dominant palette of the reference">
              {style.matchedPalette.map((c, i) => (
                <span key={i} style={{ background: `rgb(${c.rgb.join(',')})`, flexGrow: Math.max(c.share, 0.04) }} />
              ))}
            </div>
          )}
          <dl className="grid grid-cols-3 gap-1 text-[11px]">
            <Metric label="Brightness" value={p.meanL.toFixed(0)} />
            <Metric label="Contrast" value={p.contrast.toFixed(1)} />
            <Metric label="Warmth" value={(p.warmth > 0 ? '+' : '') + p.warmth.toFixed(1)} />
            <Metric label="Colourfulness" value={p.chroma.toFixed(1)} />
            <Metric label="Sharpness" value={p.sharpness.toFixed(0)} />
            <Metric label="Skin texture" value={p.skinTexture === null ? 'n/a' : p.skinTexture.toFixed(1)} />
          </dl>
          <Segmented<StyleSyncMode>
            label="How to apply"
            value={style.mode}
            options={[
              ['palette', 'Exact palette'],
              ['sliders', 'Editable sliders'],
            ]}
            onChange={(v) => setOptions({ mode: v })}
          />
          <p className="text-[11px] text-ink-400">
            {style.mode === 'palette'
              ? 'Matches the full colour distribution and tone curve, plus skin smoothing and sharpening.'
              : 'Sets exposure, contrast, temperature, tint, saturation, skin and sharpen sliders you can fine-tune.'}
          </p>
          <RangeRow
            label="Strength"
            value={style.amount}
            min={0}
            max={100}
            onStart={() => useEditor.getState().beginEdit()}
            onChange={(v) => {
              setOptions({ amount: v });
              // Live preview: re-map the active photo while dragging (one undo step per drag).
              const s = useEditor.getState();
              const photo = s.activeId ? s.photos[s.activeId] : null;
              if (photo?.profile) s.patchAdjustments(mapStyleToAdjustments(p, photo.profile, style.mode, v));
            }}
          />
          <div className="grid grid-cols-2 gap-2">
            <button className="btn" onClick={() => apply(false)}>
              Apply {selected.length > 1 ? `to ${selected.length}` : 'to photo'}
            </button>
            <button className="btn-primary" onClick={() => apply(true)} disabled={!total}>
              Sync style to all {total}
            </button>
          </div>
          <button className="btn-ghost w-full" onClick={() => setStyleReference(null)}>
            Remove reference
          </button>
        </div>
      )}
      {note && <p className="mt-1.5 text-[11px] text-mint-400">{note}</p>}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-ink-850 px-1.5 py-1">
      <dt className="text-ink-400">{label}</dt>
      <dd className="font-mono text-ink-100">{value}</dd>
    </div>
  );
}
