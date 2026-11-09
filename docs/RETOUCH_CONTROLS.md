# Retouch control system (v0.7), skin masks (v0.8), skin engine (v0.9), pro tools (v0.10), retouch assistant (v0.11) and upgrade audit

This document has two parts: the Phase 1 audit of FineRetouch against the professional
portrait-retouching specification, and the Phase 2 deliverable (per-effect strength,
1x / 2x / 3x multipliers, global strength, enable, reset, opacity and persistence).

## Part 1: audit (2026-11-08)

### Architecture

| Layer | Where | Notes |
| :--- | :--- | :--- |
| UI and app state | `apps/web` (Next.js 16, React 19, Zustand, Radix, Tailwind) | Editor, filmstrip, panels, per-photo undo/redo |
| Live preview | `apps/web/src/engine` (WebGL2, one retouch pass on a ≤2560 px proxy) | Mirrors the Python stages; spot healing, 2x/3x boost and stray hair are approximations |
| Edit model | `packages/shared` (TypeScript), mirrored by `services/ai/app/schemas.py` (Pydantic) | One JSON `Adjustments` object per photo: editing is non-destructive and the original is never written |
| Full-resolution engine | `services/ai/app/pipeline.py`, `retouch_engine.py`, `color_engine.py`, `raw_engine.py` | NumPy / OpenCV on CPU, 8- and 16-bit input |
| Analysis | `services/ai/app/landmarks.py` (MediaPipe Face Landmarker, Apache-2.0), `depth.py` (selfie multiclass segmenter, optional MiDaS small, MIT) | CPU only |
| Batch | `services/worker` (BullMQ → S3) and `services/ai/app/batch_engine.py` (desktop SQLite queue) | |
| Desktop | `src-tauri` (Tauri v2) with the Python engine frozen as a sidecar | Offline; Windows/macOS/Linux installers built in CI |

### Feature status before this upgrade

| Area | Status |
| :--- | :--- |
| Face detection and 478 landmarks, eye/brow/lip/under-eye/nose polygons | Implemented |
| Skin mask | Partial: landmark oval × YCbCr box × the person's own skin colour. No semantic face parsing (hair, neck, facial hair, clothing as classes) |
| Skin smoothing (frequency separation, tone-layer smoothing, texture re-injection) | Implemented |
| Spot / acne / dark-spot healing (multi-scale LoG, inpaint + donor texture, beauty-mark protection) | Implemented as one tool; no separate dark-spot or redness-only controls |
| Even skin tone, CIELAB skin-tone guard | Implemented |
| Under-eye, eye brightening, teeth whitening and stains, glasses glare, stray hair, clothing iron | Implemented |
| 1x / 2x / 3x | Partial: one skin-only multiplier |
| Per-effect enable / opacity / reset, global strength | Not implemented |
| Noise reduction, dodge and burn, healing / clone brushes, mask painting, layers | Not implemented |
| Project files, autosave, crash recovery | Not implemented (edits in memory; presets in localStorage) |
| NL assistant | Not implemented |
| GPU inference | CPU only; no DirectML or ROCm path |

### Weaknesses found

- `retouch_engine.py` is 1,600 lines covering eight unrelated tools; it should be split by tool as new engines land.
- The preview approximates spot healing and the 2x/3x boost, so the preview and export can differ on blemishes.
- Edits are lost on reload.
- Skin masks rely on colour; dark hair on dark skin and warm backgrounds can leak in or out.

### Plan

1. **Phase 2 (this change):** retouch control system.
2. **Phase 3 (done, see Part 3):** face parsing for semantic masks, editable masks. The licence check ruled out the public BiSeNet / CelebAMask-HQ weights (non-commercial), so the bundled MediaPipe selfie multiclass model (Apache-2.0) is used.
3. **Phase 4 (done, see Part 4):** split the skin engine; dedicated dark-spot (pigmentation, done in v0.7) and redness tools, noise reduction, under-eye texture.
4. **Later phases:** brushes (heal, clone, dodge and burn), project files, presets per style, the NL plan engine, tiled rendering, optional DirectML after testing on the target hardware.

## Part 1b: separate skin tools

| Tool | Slider | Engine | What it touches |
| :--- | :--- | :--- | :--- |
| Skin smoothing | `skinSmoothing` | `natural_skin_retouch` (tone layer) | Blotches and roughness on the tone layer |
| Texture preservation | `skinTexture` | same | Share of the pore layer put back |
| Acne removal | `acneRemoval` | `remove_acne`: `detect_spots(kind="acne")` + `heal_spots` + `reduce_acne_redness` | Spots whose redness carries ≥35% of their contrast, raised bumps, then excess Cr in a halo around them |
| Dark spot removal | `darkSpotRemoval` | `remove_dark_spots`: `detect_spots(kind="dark")` + `heal_spots` + `correct_pigmentation` | Dark, non-red marks; then patches darker **and** yellower (lower Cb) than the skin 2.5× wider; strong dark edges excluded so shading stays |
| Blemish removal | `blemishRemoval` | `heal_spots` (all kinds) | Every spot type in one pass (the former Spot healing) |
| Skin tone correction | `skinToneUnify` | chroma toward local skin average + tone guard | Uneven colour and redness |
| Under-eye correction | `darkCircles` | under-eye landmark mask | Dark circles, lifted toward the cheek |

Beauty marks stay protected in every spot tool. **Keep freckles** (`keepFreckles`) leaves dark spots smaller than 2% of the face width and fainter than 0.09 luma, and softens pigmentation correction to 60%. Each tool's multiplier also sets its own detector's reach. The WebGL preview approximates acne, dark spots and pigmentation with the same signals on its fixed bands. Export runs the full detectors.

Tests: `services/ai/tests/test_skin_tools.py` checks that acne removal changes red spots more than 4× as much as dark ones (and the reverse for dark spot removal), that blemish removal heals both, that clear skin is unchanged, that freckles are kept, that a brown patch is evened while an equal neutral shadow is mostly kept, and the multiplier and switch.

## Part 2: the control system

### Data model

`Adjustments` gains two fields (TypeScript `packages/shared/src/effects.ts`, Python `services/ai/app/schemas.py`):

```ts
effects: Record<EffectId, { enabled: boolean; multiplier: 1 | 2 | 3; opacity: number /* 0..100 */ }>;
globalStrength: number; // 0..100, default 100
```

Effects: `skinSmoothing`, `blemishRemoval`, `skinToneUnify`, `darkCircles`, `eyeBrighten`,
`teethWhiten`, `teethStainRemoval`, `glareReduction`, `sharpen`, `strayHair` (stray-hair
strength) and `clothingIron` (clothing wrinkle strength).

Older edits, presets and API requests don't have these fields. They get the defaults (on, 1x,
100%, global 100) and render exactly as before.

### How a slider becomes an engine value

The user's slider is never rewritten. Each renderer calls `resolveAdjustments` (TS) or
`resolve_adjustments` (Python) first:

```
x        = slider × multiplier
soft     = x                                              if x ≤ knee·cap
         = k + (cap − k)·(1 − e^(−(x − k)/(cap − k)))     otherwise, k = knee·cap
boosted  = max(slider, soft)              (1x is exactly the slider)
effective = enabled ? boosted × opacity/100 × global/100 : 0
```

| Effect | Knee | Cap | Effect-specific behaviour |
| :--- | :--- | :--- | :--- |
| Smoothing | 0.8 | 100 | Acts on the tone layer only, so pores return through Texture retention at any multiplier |
| Spot healing | 0.85 | 100 | The multiplier also goes to the spot detector, which then finds fainter and larger spots |
| Even skin tone | 0.75 | 100 | The skin-tone guard still locks the undertone |
| Dark circles | 0.75 | 100 | The lift is bounded by the cheek's own brightness |
| Brighten eyes | 0.6 | 85 | Lower cap so eyes don't glow |
| Whiten teeth | 0.6 | 85 | Lower cap so teeth keep their enamel colour |
| Teeth stains | 0.8 | 100 | |
| Glasses glare | 0.85 | 100 | |
| Sharpen | 0.6 | 90 | Bends early and caps lower to limit halos |
| Stray hair, clothing iron | 0.8 | 100 | |

Examples (smoothing): 30% gives 30 / 60 / 87.9 at 1x / 2x / 3x; 40% at 3x gives 97.3.

### Where it runs

- WebGL preview and in-browser export: `apps/web/src/engine/renderer.ts` resolves before setting uniforms; the shader receives `uSpotMultiplier` for detection.
- Python export, batch and cloud worker: `pipeline.render` resolves first; `_skin_stages` passes the spot multiplier to `heal_spots`.

### UI

Every effect slider shows an on/off switch, 1× / 2× / 3× buttons, the effective strength
(`= 60%`), a reset button (slider, multiplier and opacity) and an expandable advanced row with
opacity. The **Retouch strength** section at the top of the right panel holds the global
strength. Every change is one undo step; opacity and global strength drags are one step each.

### Persistence

- Presets, copy/paste and Sync to all carry effect controls and global strength, because they are part of `Adjustments`.
- Autosave (`apps/web/src/lib/editPersistence.ts`) keeps each photo's edits in localStorage under `name|bytes|lastModified|WxH` of the original, 600 ms after the last change. Re-importing the same file restores them and shows "Edits restored from your last session". Up to 400 photos are kept; the oldest are evicted first, and photos reset to defaults are removed.

### Status

| Requirement | Status |
| :--- | :--- |
| Independent 0–100% slider per effect | Implemented (existing sliders, now with controls) |
| 0% disables the operation | Implemented (engines skip stages at 0) |
| 1x / 2x / 3x changing real output, effect-specific curves and caps | Implemented; tested on rendered pixels |
| Global strength independent of individual settings | Implemented |
| Enable / disable, reset, opacity per effect | Implemented. Opacity scales the effect's contribution in parameter space. For effects that blend linearly with strength (most of them) this equals layer opacity; for spot healing it reduces the healing blend rather than cross-fading a separate layer. True per-layer pixel blending comes with the operation-node layer system |
| Parameter persistence | Implemented: presets, copy/paste, autosave and restore on re-import. Project files (`.frproj`) with masks and history are planned |
| Advanced controls (detection sensitivity, radius, edge preservation, mask feathering, thresholds, quality) | Planned: needs the engine split in Phase 4 so each can be validated per tool |
| Texture retention multiplier | Not applicable: it's a preservation amount, not an effect |

### Tests

- `packages/shared/src/effects.test.ts`: curve vectors, monotonicity and caps, resolve, sanitising, `isDefault`.
- `services/ai/tests/test_effects.py`: the same vectors and curve table (TS/Python parity), resolve, and render tests: 2x/3x increase the change monotonically, 30%×2 renders identically to 60%×1, disabling equals the slider at 0, 50% opacity equals 50% global, global 0 equals no retouch, the spot multiplier heals more spots.
- `apps/web/src/lib/effects.store.test.ts` and `editPersistence.test.ts`: store actions, undo, reset of structured effects, presets carrying controls, autosave round-trip, eviction and corrupt storage.
- Browser check (Chromium, SwiftShader WebGL2, real MediaPipe face analysis): 2× and 3× change the rendered preview, the readout shows the effective value, and edits come back after a reload and re-import.

## Part 3: face parsing and editable skin masks (v0.8)

### Model

MediaPipe selfie multiclass segmenter (`selfie_multiclass_256x256.tflite`, Apache-2.0), already bundled for bokeh and stray hair. Classes: background, hair, body skin, face skin, clothes, accessories. It runs on CPU on a ≤512 px proxy: 0.57 s on the test portrait in the cloud container (not yet timed on the Ryzen 3 7330U). Face-parsing models with separate brow / eye / lip / beard classes (BiSeNet on CelebAMask-HQ) were not used because their weights are licensed for non-commercial use only.

### How the skin mask is built

1. Landmark mask (unchanged): face oval minus eyes, brows and lips, feathered.
2. With parsing (`skinMask.useSegmentation`, default on): `skin = max(landmark, blur(faceSkin × faceZone) × featureCut)`. `faceZone` is an ellipse one face width around each detected face, so a face on a poster in the background is not retouched. `featureCut` keeps eyes, brows and lips out.
3. `includeBodySkin`: adds the body-skin class (neck, chest, arms) the same way.
4. Hair: `skin × (1 − smoothstep(0.35, 0.75, hair))`.
5. The skin-colour gate runs as before, then the user's strokes: `skin = max(skin, add) × (1 − erase)`. Strokes are absolute: Protect wins over everything, Add works on any colour.

Engine: `services/ai/app/masking.py`, `pipeline.build_masks` and `render`. Preview: `apps/web/src/engine/masks.ts` (`buildFaceMask` with parse pixels, `rasterizeMaskStrokes`, `buildEditMask`), a stroke texture (`uEdit`, r = add, g = erase) in the retouch shader, and in the mask overlay.

### Data model

```ts
skinMask: {
  useSegmentation: boolean;  // default true
  includeBodySkin: boolean;  // default false
  feather: number;           // 0..100, stroke edge softness up to the stroke radius
  strokes: { mode: 'add' | 'erase'; size: number /* radius, share of long edge */; points: number[] /* normalised x,y */ }[];
}
```

Limits: 200 strokes, 4,000 points per stroke. Old edits get the defaults.

### Status

| Requirement | Status |
| :--- | :--- |
| Face detection and landmarks | Implemented (unchanged, MediaPipe Face Landmarker) |
| Semantic skin: face skin, neck / body skin, hair excluded | Implemented (engine and preview) |
| Clothing class | Implemented earlier for the clothing tools; not part of the skin mask |
| Eyes, brows, lips, teeth, nostrils protected | Implemented through landmarks (the segmenter has no such classes) |
| Beard / facial hair as its own class | Partial: dark facial hair usually reads as hair and is excluded; light stubble may stay in the skin mask. Use Protect |
| Mask brush: add, erase (protect), size, feather, clear, show | Implemented |
| Strokes saved with the edit, one undo step each | Implemented (autosave, undo / redo) |
| Invert mask, per-tool masks, lasso / gradient masks | Planned (comes with the operation-node layer system) |
| Strokes on the cloud batch worker | Implemented: the worker posts the same `Adjustments` to the engine |

### Tests

- `services/ai/tests/test_masking.py`: stroke rasterising and feathering, erase wins over add, an erase stroke keeps an area pixel-identical under five skin tools, an add stroke retouches a grey (non-skin) area, parsing extends skin and removes hair, body skin only when asked, parsing ignored when switched off.
- `packages/shared/src/skinMask.test.ts`: sanitising and limits, `isDefault`, presets keep the photo's own strokes.
- `apps/web/src/engine/skinMask.test.ts`: the TypeScript port of the parse branch (extension, poster ignored, hair cut, body skin) and of the stroke rasteriser.
- `apps/web/src/lib/skinMask.store.test.ts`: one undo step per stroke, clear and undo, copy / paste / sync / presets never move strokes.
- Browser check (Chromium, SwiftShader WebGL2, the real engine on the MediaPipe sample portrait): the preview fetches `/v1/parse`, the panel reports "Skin from AI face parsing", a Protect stroke across the cheeks clears the mask overlay there, an Add stroke adds the flag area, Ctrl+Z removes the last stroke.

## Part 4: skin engine (v0.9)

### Engine layout

`services/ai/app/retouch_engine.py` is now a thin facade over `app/engines/`:

| Module | Contents |
| :--- | :--- |
| `common.py` | colour conversions, blurs, masked blur, inpainting helpers |
| `boost.py` | 2x/3x boosted retouch |
| `spots.py` | spot detection (LoG, bumps, strands, beauty marks) and healing |
| `skin.py` | natural retouch, acne and dark-spot tools, pigmentation, skin-tone guard |
| `redness.py`, `noise.py`, `under_eye.py` | the new v0.9 tools |
| `hair.py`, `bokeh.py`, `style.py`, `studio.py`, `clothing.py` | stray hair, bokeh, style sync, glare and teeth, clothing |

### New tools

| Tool | Slider | What it does | What it never touches |
| :--- | :--- | :--- | :--- |
| Redness reduction | `rednessReduction` | Tone layer (sigma S1) compared with a skin-only average 0.2 face widths wide, computed twice so the red areas don't raise their own reference. Excess Cr above 0.004 (full at 0.025) is pulled back 90%, and Cb follows the reference by 40% | Luma (shading, contour), pores, non-skin |
| Skin noise reduction | `skinNoise` | Noise level = 1.045 x MAD of the Cb/Cr fine bands on skin (capped by the luma band's MAD). The finest luma band (0.9 px) is soft-thresholded at 2.5x that level; chroma is averaged over 2 px on skin, except at strong colour edges | Pores and fine hair stronger than the noise; clean files (low measured level) |
| Under-eye texture | `underEyeTexture` | Inside the under-eye region, the luma band between S0 and S1 (capped at 0.012 face widths) is reduced: dark creases fully at 100%, light ridges by 35% | The finest band (pores) and the tone layer (shape, shading); everything outside the region |

Order in the export: noise reduction first (so grain doesn't read as spots or texture), then acne, dark spots, natural retouch, redness, tone correction, boost, tone guard. Under-eye texture runs with the face-feature tools, after under-eye correction.

Preview (`shaders.ts`):
- Noise reduction uses 3x3 and 5x5 neighbourhoods and a noise level measured from a centre crop of the proxy (`engine/noise.ts`, the same estimator). The export measures its own full-size source.
- Redness compares with the redder of the two broad bands, because those bands also average background and clothes. On the test portrait the preview's a* change is within 0.1 of the export's.

### Status

| Requirement | Status |
| :--- | :--- |
| Engine split by tool | Implemented (no behaviour change) |
| Redness reduction, separate from acne | Implemented |
| Skin noise reduction with texture kept | Implemented |
| Under-eye fine lines | Implemented for the landmark under-eye region |
| Crow's feet and forehead lines | Planned: outside the under-eye region; needs its own region and an oriented line detector |
| Per-tool advanced controls (sensitivity, radius, thresholds) | Planned: the constants live at the top of each engine module now, so each can become a validated control |

### Tests

`services/ai/tests/test_skin_engine.py`:
- Redness: a red patch loses at least 60% of its excess redness while an equal neutral shadow barely changes, luma is unchanged (< 0.001), and pores correlate > 0.99.
- Noise: error to the clean image falls below 55% and chroma noise below 50%. Pores keep > 70% of their contrast. A clean image changes less than a third as much as a noisy one, and strong luma-only pores do not read as noise.
- Under-eye: crease depth more than halves, nothing changes outside the region, pores keep > 75% of their contrast, and 40 < 100 in effect.
- Render: 1x < 3x, off equals the slider at 0 for all three tools, and a Protect stroke leaves the under-eye area untouched.

`apps/web/src/engine/noise.test.ts` pins the browser estimator to the same behaviour.

Real-photo check (MediaPipe sample portrait, 820 x 1024): each tool takes 0.2 to 0.35 s for the whole render on CPU. Noise reduction removes added sensor noise and leaves the clean file nearly unchanged (mean change 0.05/255).

## Part 5: professional retouching tools (v0.10)

All of these are non-destructive: strokes and layer settings are stored in the edit
(`Adjustments.localRetouch`, `packages/shared/src/localRetouch.ts`) and replayed on the original
pixels at render time. The original file is never written.

### Where each tool runs

| Tool | Export (engine) | Preview and browser export |
| :--- | :--- | :--- |
| Healing, clone | `engines/local.apply_heal_strokes`, stage 0b: on the original pixels, before every other stage | `engine/heal.ts` on the proxy pixels (full-size pixels for a browser export), uploaded as the renderer's working source |
| Frequency separation, dodge & burn, correction brush | `engines/local.apply_local_layers`, stage 10: after global tone, before bokeh | `LOCAL_FRAG`, one WebGL pass per layer after the studio pass |
| Layer masks | `engines/local.layer_masks` at full size | `engine/heal.ts buildLayerMask` at 1536 px, sampled by UV |

### How the tools work

| Tool | Algorithm |
| :--- | :--- |
| Clone | Copies the source (destination + offset, bilinear) through a brush with hardness and opacity |
| Healing | `healed = source + fill(destination - source)`: the difference is known around the stroke and filled across it by a push-pull membrane, so texture comes from the source and tone and colour meet the surroundings with no seam |
| Spot healing | With no source set, the brush picks one when the stroke ends: 48 candidates around the stroke, scored by how well their border matches the destination's border and how close their inside is to that tone (so a blemish is never copied in). The chosen offset is saved with the stroke, so the export uses the same source |
| Frequency separation | Split at a user radius (0.08% to 2.4% of the long edge). The tone layer is smoothed toward a 5x wider blur, stopping at edges (tone step above 0.04 to 0.12 in luma); the detail layer is kept at 0 to 200% |
| Dodge & burn | Painted exposure of up to 1.5 EV, in linear light so colour is kept, optionally limited to shadows, midtones or highlights. Low-flow strokes build up |
| Correction brush | Local exposure (same scale as the global slider), contrast, saturation, temperature and tint, plus optional frequency-separation smoothing and texture |

### Masks

Each layer has its own mask: a base (painted only, the AI skin mask, or everything), add and
erase strokes with flow, feather, invert, a luminance range with soft roll-off, density and
layer opacity. M shows the active layer's mask while its brush is on (red for coverage, yellow
for dodge, blue for burn). Layers stack bottom to top, can be reordered, switched off, renamed
and deleted (up to 8 per photo).

Healing, clone and layers belong to one photo: copy/paste, sync, smart sync, presets and saved
presets never carry them to another photo.

### Status

| Requirement | Status |
| :--- | :--- |
| Frequency separation | Implemented as a layer (adjustable split radius, tone smoothing, texture 0 to 200%) |
| Healing brush | Implemented, with Alt-click source, aligned offset and automatic spot-healing source |
| Clone stamp | Implemented, with hardness and opacity |
| Dodge and burn | Implemented as a layer with dodge, burn and erase brushes, flow, strength and tonal range |
| Manual correction brushes | Implemented: exposure, contrast, saturation, temperature, tint, smoothing and texture through a painted mask |
| Advanced masks | Implemented per layer: base, add / erase with flow, feather, invert, luminance range, density, opacity, overlay |
| 1x / 2x / 3x on the pro tools | Not applicable: these are manual tools with their own strength, flow and opacity |
| Global retouch strength on the pro tools | Not applied, by design: manual strokes do exactly what was painted |
| Per-layer blend modes, layer groups | Planned |
| Skin base without a detected face | Partial: the export uses the colour-based skin weight; the preview has no skin mask without a face, so an "AI skin" layer shows no effect there until a face is found |

### Parity

- Healing and clone: the engine and the browser run the same algorithm and are pinned to the same
  test vectors (`HEAL_VECTORS` in `services/ai/tests/test_local_tools.py` and
  `apps/web/src/engine/heal.test.ts`), equal to uint8 rounding.
- In the real app on the sample portrait with every tool used at once, the GPU export and the
  engine export differ by 0.05/255 on average over the whole image; inside each tool's area the
  mean difference is 0.0 (clone), 2.2 (the healing area, which also sits under the frequency-separation layer on skin),
  0.4 to 0.6 (dodge, burn), 0.7 (frequency separation) and 0.01 (correction brush).

### Tests

- `services/ai/tests/test_local_tools.py` (13): push-pull fill, healing removes a blemish to the
  surrounding tone (not the brighter source's) and keeps > 60% of the texture, nothing outside the
  brush changes, clone and opacity, stroke order, frequency separation evens blotches by > 25%
  with pores kept > 90% and texture 0 removing them, no smearing across a strong edge, mask
  strokes / invert / luminance range, flow build-up and erase, dodge & burn keep channel ratios
  and respect the tonal range, skin base, full render, schema round trip, the parity vectors.
- `apps/web/src/engine/heal.test.ts` (7): the same properties for the browser port, spot healing
  avoiding a source with its own blemish, and the parity vectors.
- `apps/web/src/lib/localRetouch.store.test.ts` (3): undo steps, layer order, and that paste,
  sync and presets never copy the pro tools.
- `packages/shared/src/localRetouch.test.ts` (4): sanitising untrusted JSON, defaults, when a
  layer is active, split radius mapping.

## Part 6: retouch assistant (v0.11)

The **Retouch assistant** section (top of the right panel) suggests edits in three ways. All
three produce the same thing: an **edit plan** (`EditPlan` in `packages/shared/src/assistant.ts`)
that lists each change with its value and the reason for it. Nothing changes until the plan is
approved.

### Where it runs

Everything runs in the app, on this computer. The photo is never uploaded for the assistant, and
the instruction text is never sent anywhere. There is no language model: the instruction reader
is a deterministic phrase matcher with a fixed retouching vocabulary. It says which parts of a
request it did not understand instead of guessing.

### 1. Ask for an edit (plain-English instructions)

Type a request such as *"remove acne and dark spots, keep the freckles, brighten the eyes
slightly"*. The request is split into clauses, and each clause is matched against the vocabulary,
most specific phrase first, so "under-eye lines" never also triggers "eyes".

| You say | Tool |
| --- | --- |
| acne, pimples, breakouts | Acne removal |
| dark spots, sun spots, pigmentation, melasma | Dark spot removal |
| blemishes, spots, marks, scars | Blemish removal |
| redness, rosacea, flushed, blotchy | Redness reduction |
| noise, grain | Skin noise reduction |
| uneven skin tone | Skin tone correction |
| smooth, airbrush, clear skin | Skin smoothing (plus texture preservation) |
| pores, texture, natural skin, "not plastic" | Texture preservation |
| dark circles, eye bags, tired eyes | Under-eye correction |
| under-eye lines, crepey skin (also wrinkles, see below) | Under-eye texture |
| eyes | Brighten eyes |
| teeth, smile / stained teeth | Whiten teeth / Teeth stains |
| glasses, glare, reflections | Glasses glare |
| flyaways, stray hairs, frizz | Stray hair cleanup |
| shirt wrinkles, iron | Clothing wrinkles |
| blur the background, bokeh | Background blur |
| brighter / darker, warmer / cooler, contrast, shadows, highlights, vivid / muted, black and white, vignette, sharpen, tint | The light, colour and detail sliders |
| freckles, moles, beauty marks | Keep freckles / Protect beauty marks switches |

- **Amounts.** Each tool has a normal amount. "Slightly", "a touch" give 60% of it; "a lot",
  "very" give 135%. An overall "natural" or "subtle" scales the whole request to 75%; "glam" or
  "flawless" to 130%. A number ("blemish removal to 80") is used as given, and "2x" / "3x" /
  "double" sets the tool's multiplier.
- **Direction.** "Less smoothing", "too sharp", "tone down the acne removal" weaken the named
  tool. Naming the problem instead ("less acne", "tone down the redness") strengthens the tool
  that removes it. "Don't", "no", "without" switch a tool off. "Keep the freckles" switches
  protection on; "remove the freckles" switches it off.
- **A verb on another subject is not a light change.** "Brighten the eyes" sets Brighten eyes,
  not Exposure (unless the clause also says "the photo", "overall", "everything").
- **Natural texture rule.** Smoothing never goes out on its own: if the plan smooths and texture
  preservation is below 60, a texture preservation step is added.
- **General requests.** "Retouch this portrait" gives a natural full pass: blemishes, light
  smoothing with texture kept, tone, under-eyes and eyes (face tools only with a face).
- **Declined on purpose.** Requests to slim, reshape, change the nose, jaw, eye or skin colour,
  or make someone look younger or thinner are refused with the reason ("never changes face
  shape, body shape or skin colour"). This is the identity rule from the master spec.
- **Not understood.** Anything left over is listed as "Not understood", with a hint to name the
  problem.
- **No face.** Face-only steps (eyes, teeth, under-eyes, glare, stray hair) are unticked and say
  so when no face was detected.

### 2. Recommended for this photo

Uses the photo's own measurements (the `StyleProfile` already used by Auto Retouch: mean L*,
tonal spread, b*, a*, chroma, sharpness, skin texture) and gives each step the measurement
behind it, for example *"The photo is on the dark side (mean lightness L* 33; a well-exposed
portrait sits near 56)"*.

**Skin-tone safeguard (new).** Auto Retouch brightens toward a whole-frame lightness target. On a
darker-skinned subject in a dark scene that would lift the skin far beyond its real tone. The
assistant caps brightening so the measured skin lightness rises by at most 10 L*
(`SKIN_LIFT_MAX`), and says so in the step. The older **Auto retouch** button is unchanged (it
applies directly, without the cap); the assistant's recommendations are the reviewed path.

### 3. Retouching styles (automatic presets)

Natural, Corporate headshot, Beauty, Wedding, Mature skin and Men's grooming. A style is not a
fixed preset: it is built for each photo.

- Light and colour come from the photo's measurements (as in Recommendations, with the
  skin-tone cap), plus the style's own small offset (Wedding: a little warmer, softer contrast).
- Skin smoothing and blemish removal are matched to the measured skin texture and scaled by the
  style (Natural 0.7, Beauty 1.4, …), so a smooth-skinned and a rough-skinned photo get
  different amounts from the same style.
- The style's other settings (texture preservation, under-eyes, eyes, teeth, glare, stray hair,
  clothing, freckles and beauty marks, multipliers) are its starting values, all editable in the
  plan.

### Approval workflow

- **Preview.** A pending plan is shown live in the viewport on its photo only. It is a preview:
  history, autosave and export keep using the saved edit until approval.
- **Adjust.** Each step has a checkbox and its own control (slider, switch or 1x/2x/3x), with
  "now X" showing the current value. **Plan strength** (0 to 100%) scales every numeric change
  between the current edit and the plan.
- **Approve** commits the ticked steps as **one undo step** per photo. **Reject** discards the
  plan. Switching photos discards a pending plan.
- **Batch.** With several photos selected, *Also apply to N other selected photos* applies on
  approval. A style or recommendation is rebuilt from each photo's own measurements, keeping
  the steps you unticked and the plan strength. An instruction plan carries the same values.
- The assistant never touches healing, clone, layers or skin mask strokes.

### Status

| Feature | Status |
| --- | --- |
| Edit plan with reasons, adjustable steps and strength | Implemented |
| Plain-English instructions (offline, rule-based) | Implemented, with a fixed vocabulary. Free-form phrasing outside it is reported as not understood |
| Identity-changing requests declined | Implemented |
| Recommendations from measurements, with reasons | Implemented |
| Skin-tone safeguard on brightening | Implemented in the assistant; the old Auto retouch button is unchanged |
| Retouching styles built per photo | Implemented (6 styles) |
| Preview, approve as one undo step, reject, batch | Implemented |
| Crow's feet and forehead lines | Partial: the request maps to under-eye texture and the plan says only the under-eye area is covered |
| A local language model for free-form requests | Planned / blocked: a small on-device model would add hundreds of MB to the installer and seconds per request on the Ryzen 3 CPU. Not added; no cloud model is used |
| Learning a user's own style from their past edits | Planned |

### Tests

- `packages/shared/src/assistant.test.ts` (24): vocabulary mapping, phrase priority, intensity
  words, numbers and multipliers, texture rule, keep/remove freckles and moles, negation,
  weakening a tool versus removing a problem, multi-word phrases, verbs on another subject, light
  and colour direction, declined and unmatched parts, face-only steps without a face, general
  requests, stray hair and clothing, plan application (ticks, strength, never touching strokes),
  recommendations with reasons, the skin-tone cap, styles scaling with skin texture, natural
  versus beauty, rebuilding a style per photo.
- `apps/web/src/lib/assistant.store.test.ts` (6): preview without saving, approve as one undo
  step, step edits and strength, reject, switching photos, batch rebuild per photo,
  recommendations needing measurements.
- Browser check on the sample portrait: a mixed request produced 5 steps, declined "slim the
  face", reported "add a hat" as not understood, previewed live and was applied on approval; the
  Corporate headshot style and recommendations showed their measured reasons, with the exposure
  limited by the skin-tone cap. No console errors.
