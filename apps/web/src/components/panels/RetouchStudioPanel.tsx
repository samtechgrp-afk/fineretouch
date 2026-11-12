'use client';

import { EffectControlsRow } from './EffectControls';
import { DEFAULT_ADJUSTMENTS, effectForSlider, GLASSES_CONFIDENT, type AdjustmentKey, type ClothingSettings } from '@fineretouch/shared';
import type { StudioOverlay } from '@/components/viewport/shaders';
import { useEditor, type Photo, type SegmentState } from '@/lib/store';
import { Heading, RangeRow, Segmented, Toggle } from './controls';

/**
 * Pro studio retouch: the commercial-portrait controls in one place.
 *
 *  Skin          frequency-separation smoothing with the texture-retention (detail re-injection) control
 *  Glasses       lens glare reduction
 *  Stray hair    flyaway cleanup around the face outline and forehead
 *  Teeth         whitening and stain removal on an enamel mask
 *  Clothing      "virtual iron" for creases and folds, and garment recolour
 *  QA overlay    glare veil, texture retention and garment mask on the viewport (never exported)
 *
 * The viewport previews each tool on the GPU; exports from the engine run the full
 * algorithms in services/ai/app/retouch_engine.py. All controls apply to the selected photos.
 */
export function RetouchStudioPanel({ photo }: { photo: Photo }) {
  const faces = photo.analysis?.faces ?? [];
  const hasFace = faces.length > 0;
  const hasEyes = faces.some((f) => f.regions.leftEye.length >= 3 || f.regions.rightEye.length >= 3);
  const hasMouth = faces.some((f) => f.regions.innerMouth.length >= 3);
  const withGlasses = faces.filter((f) => (f.glasses ?? 0) >= GLASSES_CONFIDENT).length;
  const force = photo.adjustments.glareForce;
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  return (
    <div className="space-y-5">
      <SkinSection photo={photo} />
      <section>
        <Heading
          title="Glasses glare"
          hint={hasEyes ? 'Lifts flash and window reflections off eyeglass lenses so the eyes behind them show again. Catchlights and frame highlights are kept.' : 'Needs a detected face with eyes.'}
        />
        <SliderFor photo={photo} k="glareReduction" label="Glare reduction" disabled={!hasEyes} />
        {hasEyes && (
          <div className="mt-2 space-y-1">
            <p className="text-[11px] text-ink-400" data-testid="glasses-status">
              {withGlasses > 0
                ? `Glasses found on ${withGlasses === 1 ? 'this face' : `${withGlasses} faces`}: glare reduction works on their lenses only.`
                : force
                  ? 'No glasses detected: applied anyway (override on).'
                  : 'No glasses detected, so glare reduction is skipped and skin around the eyes is left as it is.'}
            </p>
            <label className="flex items-center justify-between gap-3 text-xs text-ink-200">
              <span>Apply even if no glasses are detected</span>
              <Toggle label="Apply glare reduction without detected glasses" checked={force} onChange={(v) => updateAdjustments({ glareForce: v })} />
            </label>
          </div>
        )}
      </section>
      <StrayHairSection photo={photo} hasFace={hasFace} />
      <section>
        <Heading title="Teeth" hint={hasMouth ? 'Whitens toward a natural, slightly warm white and evens out stains. Enamel texture and the gaps between teeth are kept.' : 'Needs a detected face with an open smile.'} />
        <div className="space-y-3">
          <SliderFor photo={photo} k="teethWhiten" label="Teeth whitening" disabled={!hasMouth} />
          <SliderFor photo={photo} k="teethStainRemoval" label="Stain removal" disabled={!hasMouth} />
        </div>
      </section>
      <ClothingSection photo={photo} />
      <OverlaySection />
    </div>
  );
}

function SliderFor({ photo, k, label, disabled, unit, hint }: { photo: Photo; k: AdjustmentKey; label: string; disabled?: boolean; unit?: string; hint?: string }) {
  const beginEdit = useEditor((s) => s.beginEdit);
  const setAdjustment = useEditor((s) => s.setAdjustment);
  const value = photo.adjustments[k];
  const effect = effectForSlider(k);
  return (
    <div>
      <RangeRow
        label={label}
        value={value}
        min={0}
        max={100}
        hint={hint}
        disabled={disabled}
        format={(v) => `${Math.round(v)}${unit ?? ''}`}
        onStart={() => beginEdit()}
        onChange={(v) => setAdjustment(k, v)}
      />
      {effect && <EffectControlsRow id={effect} adj={photo.adjustments} disabled={disabled} />}
    </div>
  );
}

function SkinSection({ photo }: { photo: Photo }) {
  const a = photo.adjustments;
  const plastic = a.skinSmoothing >= 60 && a.skinTexture < 40;
  return (
    <section>
      <Heading
        title="Natural skin"
        hint="Smoothing evens blotches and redness on the tone layer only; Texture retention puts the pore and fine-hair layer back on top. Spot healing repairs acne and dark spots one by one."
      />
      <div className="space-y-3">
        <SliderFor photo={photo} k="skinSmoothing" label="Skin smoothing" />
        <SliderFor
          photo={photo}
          k="skinTexture"
          label="Texture preservation"
          unit="%"
          hint="Share of the pore and fine-detail layer re-injected after smoothing. 100% keeps every pore."
        />
        <SliderFor photo={photo} k="acneRemoval" label="Acne removal" hint="Red spots, pimples and raised bumps, then the redness around them." />
        <SliderFor photo={photo} k="darkSpotRemoval" label="Dark spot removal" hint="Isolated dark marks and brownish pigmentation; shading, beauty marks and (optionally) freckles stay." />
        <SliderFor photo={photo} k="blemishRemoval" label="Blemish removal" hint="General clean-up of every kind of small spot in one pass." />
      </div>
      {plastic && (
        <p className="mt-1.5 text-[11px] text-apricot-300">
          Heavy smoothing with low texture retention can look airbrushed. Try retention above 60%, or check the Texture overlay below.
        </p>
      )}
    </section>
  );
}

function StrayHairSection({ photo, hasFace }: { photo: Photo; hasFace: boolean }) {
  const setStray = useEditor((s) => s.setStrayHairRemoval);
  const beginEdit = useEditor((s) => s.beginEdit);
  const st = photo.adjustments.strayHairRemoval;
  return (
    <section>
      <Heading
        title="Stray hair"
        hint={hasFace ? 'Erases thin flyaways crossing the face outline and forehead; the hairstyle itself is protected.' : 'Needs a detected face.'}
        right={<Toggle label="Stray hair removal" checked={st.enabled} disabled={!hasFace} onChange={(v) => setStray({ enabled: v })} />}
      />
      <RangeRow
        label="Hair cleanup strength"
        value={st.strength}
        min={0}
        max={100}
        disabled={!st.enabled || !hasFace}
        onStart={() => beginEdit()}
        onChange={(v) => setStray({ strength: v }, { live: true })}
      />
      <EffectControlsRow id="strayHair" adj={photo.adjustments} disabled={!st.enabled || !hasFace} />
    </section>
  );
}

const SEGMENT_NOTE: Record<SegmentState, string> = {
  none: '',
  pending: 'Finding the clothes…',
  ready: 'Garments found by the AI segmenter.',
  unavailable: 'Live preview needs the AI engine and its segmentation model; exports still iron clothing below the face.',
};

function ClothingSection({ photo }: { photo: Photo }) {
  const beginEdit = useEditor((s) => s.beginEdit);
  const updateAdjustments = useEditor((s) => s.updateAdjustments);
  const patchAdjustments = useEditor((s) => s.patchAdjustments);
  const cl = photo.adjustments.clothing;
  const set = (patch: Partial<ClothingSettings>, live = false) => {
    const next = { ...cl, ...patch };
    if (live) patchAdjustments({ clothing: next });
    else updateAdjustments({ clothing: next });
  };
  const off = !cl.enabled;
  return (
    <section>
      <Heading
        title="Clothing (virtual iron)"
        hint={cl.enabled ? SEGMENT_NOTE[photo.segmentState] : 'Smooths creases and folds on shirts, suits and dresses while keeping the weave, seams and drape shadows. Can also recolour the garment.'}
        right={<Toggle label="Clothing retouch" checked={cl.enabled} onChange={(v) => set({ enabled: v })} />}
      />
      <div className="space-y-3">
        <RangeRow label="Wrinkle removal" value={cl.wrinkleStrength} min={0} max={100} disabled={off} onStart={() => beginEdit()} onChange={(v) => set({ wrinkleStrength: v }, true)} />
        <EffectControlsRow id="clothingIron" adj={photo.adjustments} disabled={off} />
        <RangeRow
          label="Fabric texture kept"
          value={cl.texturePreservation}
          min={0}
          max={100}
          format={(v) => `${Math.round(v)}%`}
          hint="Thread and weave detail kept at full ironing strength. Lower only for very noisy fabric."
          disabled={off}
          onStart={() => beginEdit()}
          onChange={(v) => set({ texturePreservation: v }, true)}
        />
        <RangeRow
          label="Garment hue"
          value={cl.hueShift}
          min={-180}
          max={180}
          format={(v) => `${v > 0 ? '+' : ''}${Math.round(v)}°`}
          disabled={off}
          onStart={() => beginEdit()}
          onChange={(v) => set({ hueShift: v }, true)}
        />
        <RangeRow
          label="Garment saturation"
          value={cl.saturation}
          min={-100}
          max={100}
          format={(v) => `${v > 0 ? '+' : ''}${Math.round(v)}`}
          disabled={off}
          onStart={() => beginEdit()}
          onChange={(v) => set({ saturation: v }, true)}
        />
        {!off && (cl.hueShift !== DEFAULT_ADJUSTMENTS.clothing.hueShift || cl.saturation !== 0) && (
          <button className="btn-ghost w-full" onClick={() => set({ hueShift: 0, saturation: 0 })}>
            Reset garment colour
          </button>
        )}
      </div>
    </section>
  );
}

function OverlaySection() {
  const overlay = useEditor((s) => s.view.overlay);
  const setView = useEditor((s) => s.setView);
  return (
    <section>
      <Heading title="QA overlay" hint="Check the retouch before export: reflections found on the lenses, how much skin texture survived (green kept, amber softened, red flat) and the garment mask. Never exported." />
      <Segmented<StudioOverlay>
        label="Studio overlay"
        value={overlay}
        options={[
          ['none', 'Off'],
          ['glare', 'Glare'],
          ['texture', 'Texture'],
          ['clothing', 'Clothes'],
        ]}
        onChange={(v) => setView({ overlay: v })}
      />
    </section>
  );
}
