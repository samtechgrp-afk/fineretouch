'use client';

import { MAX_LAYERS, type LayerKind, type LayerStroke, type RetouchLayer, type ToneRange } from '@fineretouch/shared';
import { useEditor, type Photo } from '@/lib/store';
import { Heading, RangeRow, Segmented, Toggle } from './controls';

/**
 * Professional retouching tools (v0.10, Phase 5): healing and clone brushes, and local retouch
 * layers (frequency separation, dodge & burn, correction brush) with their own masks. Every
 * control here drives engines/local.py on export and the same math in the preview.
 */
export function ProToolsPanel({ photo }: { photo: Photo }) {
  return (
    <div className="space-y-5">
      <HealClone photo={photo} />
      <Layers photo={photo} />
    </div>
  );
}

function BrushSize() {
  const view = useEditor((s) => s.view);
  const setView = useEditor((s) => s.setView);
  return (
    <RangeRow
      label="Brush size"
      value={view.brushSize * 100}
      min={0.3}
      max={15}
      step={0.1}
      format={(v) => `${v.toFixed(1)}%`}
      hint="Brush radius as a share of the photo's long edge"
      onStart={() => undefined}
      onChange={(v) => setView({ brushSize: v / 100 })}
    />
  );
}

function HealClone({ photo }: { photo: Photo }) {
  const view = useEditor((s) => s.view);
  const setView = useEditor((s) => s.setView);
  const clear = useEditor((s) => s.clearHealStrokes);
  const heal = photo.adjustments.localRetouch.heal;
  const tool = view.proTool === 'heal' || view.proTool === 'clone' ? view.proTool : 'off';
  const source = view.cloneOffset ? 'aligned to the first stroke' : view.cloneSource ? 'set' : 'automatic (spot healing)';
  return (
    <section>
      <Heading title="Healing & clone" hint="Alt-click sets the source. Healing takes texture from the source and tone from around the stroke; clone copies the source exactly." />
      <div className="space-y-2">
        <Segmented<'off' | 'heal' | 'clone'>
          label="Healing and clone brush"
          value={tool}
          options={[
            ['off', 'No brush'],
            ['heal', 'Healing'],
            ['clone', 'Clone'],
          ]}
          onChange={(v) => setView({ proTool: v === 'off' ? null : v, maskBrush: null })}
        />
        <BrushSize />
        <RangeRow label="Hardness" value={view.brushHardness} min={0} max={100} hint="Share of the brush radius at full strength" onStart={() => undefined} onChange={(v) => setView({ brushHardness: v })} />
        <RangeRow label="Opacity" value={view.brushOpacity} min={1} max={100} format={(v) => `${Math.round(v)}%`} onStart={() => undefined} onChange={(v) => setView({ brushOpacity: v })} />
        <div className="flex items-center justify-between gap-2 text-[11px] text-ink-400">
          <span data-testid="heal-source">Source: {source}</span>
          <button className="btn" disabled={!view.cloneSource && !view.cloneOffset} onClick={() => setView({ cloneSource: null, cloneOffset: null })}>
            Reset source
          </button>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] text-ink-400" data-testid="heal-count">
            {heal.length ? `${heal.length} heal / clone stroke${heal.length === 1 ? '' : 's'}` : 'No heal or clone strokes'}
          </span>
          <button className="btn" disabled={!heal.length} onClick={() => clear()}>
            Clear
          </button>
        </div>
        {tool === 'clone' && !view.cloneSource && !view.cloneOffset && (
          <p className="text-[11px] text-amber-300">Alt-click the area to copy from first.</p>
        )}
      </div>
    </section>
  );
}

const KIND_HINT: Record<LayerKind, string> = {
  frequency: 'Splits the skin into a tone layer and a detail layer: smooth the tone, keep or adjust the pores.',
  dodgeBurn: 'Paint light (dodge) and shadow (burn) in linear light; colour is kept.',
  correction: 'Paint local exposure, contrast, saturation and white balance.',
};

function Layers({ photo }: { photo: Photo }) {
  const layers = photo.adjustments.localRetouch.layers;
  const view = useEditor((s) => s.view);
  const setView = useEditor((s) => s.setView);
  const addLayer = useEditor((s) => s.addLayer);
  const active = layers.find((l) => l.id === view.activeLayerId) ?? null;
  const full = layers.length >= MAX_LAYERS;
  return (
    <section>
      <Heading title="Retouch layers" hint="Applied bottom to top, after the global adjustments. Each layer has its own mask." />
      <div className="mb-2 flex flex-wrap gap-1">
        {(['frequency', 'dodgeBurn', 'correction'] as const).map((k) => (
          <button key={k} className="btn" disabled={full} title={KIND_HINT[k]} onClick={() => addLayer(k)}>
            + {k === 'frequency' ? 'Frequency sep.' : k === 'dodgeBurn' ? 'Dodge & burn' : 'Correction'}
          </button>
        ))}
      </div>
      {layers.length === 0 ? (
        <p className="text-[11px] text-ink-400">No layers yet.</p>
      ) : (
        <ul className="mb-3 space-y-1" data-testid="layer-list">
          {[...layers].reverse().map((l) => (
            <LayerRow key={l.id} layer={l} active={l.id === active?.id} onSelect={() => setView({ activeLayerId: l.id })} />
          ))}
        </ul>
      )}
      {active && <LayerEditor layer={active} />}
    </section>
  );
}

function LayerRow({ layer, active, onSelect }: { layer: RetouchLayer; active: boolean; onSelect(): void }) {
  const update = useEditor((s) => s.updateLayer);
  const move = useEditor((s) => s.moveLayer);
  const remove = useEditor((s) => s.removeLayer);
  return (
    <li className={`flex items-center gap-1 rounded px-1.5 py-1 text-xs ${active ? 'bg-ink-800 text-ink-100' : 'text-ink-300 hover:bg-ink-850'}`}>
      <Toggle checked={layer.enabled} onChange={(v) => update(layer.id, { enabled: v })} label={`Show ${layer.name}`} />
      <button className="min-w-0 flex-1 truncate text-left" onClick={onSelect} title={KIND_HINT[layer.kind]}>
        {layer.name}
      </button>
      <button className="px-1 text-ink-400 hover:text-ink-100" aria-label={`Move ${layer.name} up`} onClick={() => move(layer.id, 1)}>
        ▲
      </button>
      <button className="px-1 text-ink-400 hover:text-ink-100" aria-label={`Move ${layer.name} down`} onClick={() => move(layer.id, -1)}>
        ▼
      </button>
      <button className="px-1 text-ink-400 hover:text-rose-300" aria-label={`Delete ${layer.name}`} onClick={() => remove(layer.id)}>
        ✕
      </button>
    </li>
  );
}

function LayerEditor({ layer }: { layer: RetouchLayer }) {
  const view = useEditor((s) => s.view);
  const setView = useEditor((s) => s.setView);
  const update = useEditor((s) => s.updateLayer);
  const clear = useEditor((s) => s.clearLayerStrokes);
  const beginEdit = useEditor((s) => s.beginEdit);
  const db = layer.kind === 'dodgeBurn';
  const painting = view.proTool === 'layer' && view.activeLayerId === layer.id;
  type Brush = 'off' | LayerStroke['mode'];
  const brushValue: Brush = painting ? (db ? (view.layerBrush === 'add' ? 'dodge' : view.layerBrush) : view.layerBrush === 'dodge' || view.layerBrush === 'burn' ? 'add' : view.layerBrush) : 'off';
  const brushOptions: Array<[Brush, string]> = db
    ? [['off', 'No brush'], ['dodge', 'Dodge'], ['burn', 'Burn'], ['erase', 'Erase']]
    : [['off', 'No brush'], ['add', 'Paint'], ['erase', 'Erase']];
  const live = (patch: Parameters<typeof update>[1]) => update(layer.id, patch, { live: true });
  const r = (label: string, key: 'opacity' | 'splitRadius' | 'toneSmoothing' | 'texture' | 'dodgeBurnStrength' | 'exposure' | 'contrast' | 'saturation' | 'temperature' | 'tint', min: number, max: number, hint?: string, unit = '') => (
    <RangeRow
      key={key}
      label={label}
      value={layer[key]}
      min={min}
      max={max}
      hint={hint}
      format={(v) => `${Math.round(v)}${unit}`}
      onStart={() => beginEdit()}
      onChange={(v) => live({ [key]: v })}
    />
  );
  const m = layer.mask;
  return (
    <div className="space-y-2 rounded-md border border-ink-800 p-2" data-testid="layer-editor">
      <input
        className="w-full rounded bg-ink-850 px-2 py-1 text-xs text-ink-100"
        value={layer.name}
        aria-label="Layer name"
        onChange={(e) => update(layer.id, { name: e.target.value }, { live: true })}
      />
      <Segmented<Brush>
        label="Layer brush"
        value={brushValue}
        options={brushOptions}
        onChange={(v) =>
          setView(v === 'off' ? { proTool: null } : { proTool: 'layer', layerBrush: v, maskBrush: null, activeLayerId: layer.id, showMask: v === 'erase' || v === 'add' ? true : view.showMask })
        }
      />
      <BrushSize />
      <RangeRow label="Flow" value={view.brushFlow} min={1} max={100} format={(v) => `${Math.round(v)}%`} hint="How much one stroke adds; strokes build up" onStart={() => undefined} onChange={(v) => setView({ brushFlow: v })} />
      {r('Layer opacity', 'opacity', 0, 100, undefined, '%')}
      {layer.kind === 'frequency' && (
        <>
          {r('Split radius', 'splitRadius', 0, 100, 'Detail smaller than this stays in the texture layer (0.08% to 2.4% of the long edge)')}
          {r('Tone smoothing', 'toneSmoothing', 0, 100, 'Edge-aware smoothing of the tone layer only')}
          {r('Texture', 'texture', 0, 200, '100% keeps the pores as they are; lower softens, higher brings out detail', '%')}
        </>
      )}
      {db && (
        <>
          {r('Strength', 'dodgeBurnStrength', 0, 100, 'Full flow = up to 1.5 EV')}
          <Segmented<ToneRange>
            label="Tonal range"
            value={layer.dodgeBurnRange}
            options={[
              ['all', 'All'],
              ['shadows', 'Shadows'],
              ['midtones', 'Mids'],
              ['highlights', 'Highs'],
            ]}
            onChange={(v) => update(layer.id, { dodgeBurnRange: v })}
          />
        </>
      )}
      {layer.kind === 'correction' && (
        <>
          {r('Exposure', 'exposure', -100, 100)}
          {r('Contrast', 'contrast', -100, 100)}
          {r('Saturation', 'saturation', -100, 100)}
          {r('Temperature', 'temperature', -100, 100)}
          {r('Tint', 'tint', -100, 100)}
          {r('Smoothing', 'toneSmoothing', 0, 100, 'Frequency-separation smoothing inside the mask')}
          {r('Texture', 'texture', 0, 200, undefined, '%')}
        </>
      )}
      <div className="pt-1 text-[11px] font-medium uppercase tracking-wide text-ink-400">Mask</div>
      <Segmented<'none' | 'skin' | 'all'>
        label="Mask starts from"
        value={m.base}
        options={db ? [['all', 'Anywhere'], ['skin', 'Skin only']] : [['none', 'Painted'], ['skin', 'AI skin'], ['all', 'Everything']]}
        onChange={(v) => update(layer.id, { mask: { base: v } })}
      />
      {!db && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-ink-300">Invert mask</span>
          <Toggle checked={m.invert} onChange={(v) => update(layer.id, { mask: { invert: v } })} label="Invert mask" />
        </div>
      )}
      <RangeRow label="Feather" value={m.feather} min={0} max={100} onStart={() => beginEdit()} onChange={(v) => live({ mask: { feather: v } })} />
      <RangeRow label="Density" value={m.density} min={0} max={100} format={(v) => `${Math.round(v)}%`} onStart={() => beginEdit()} onChange={(v) => live({ mask: { density: v } })} />
      <RangeRow
        label="Luminance from"
        value={m.lumaMin}
        min={0}
        max={100}
        hint="Only pixels at least this bright"
        onStart={() => beginEdit()}
        onChange={(v) => live({ mask: { lumaMin: Math.min(v, m.lumaMax) } })}
      />
      <RangeRow
        label="Luminance to"
        value={m.lumaMax}
        min={0}
        max={100}
        hint="Only pixels at most this bright"
        onStart={() => beginEdit()}
        onChange={(v) => live({ mask: { lumaMax: Math.max(v, m.lumaMin) } })}
      />
      <RangeRow label="Range softness" value={m.lumaSoftness} min={0} max={100} onStart={() => beginEdit()} onChange={(v) => live({ mask: { lumaSoftness: v } })} />
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-ink-400">{m.strokes.length ? `${m.strokes.length} stroke${m.strokes.length === 1 ? '' : 's'}` : 'No strokes'}</span>
        <button className="btn" disabled={!m.strokes.length} onClick={() => clear(layer.id)}>
          Clear strokes
        </button>
      </div>
      <p className="text-[11px] leading-snug text-ink-400">
        Press M to see this layer&apos;s mask while its brush is active. Layers stay with this photo; presets and sync never copy them.
      </p>
    </div>
  );
}
