# Retouch control system (v0.7) and upgrade audit

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
2. **Phase 3:** face parsing for semantic masks (a BiSeNet-class ONNX model, run on CPU through `cv2.dnn` or ONNX Runtime, after a licence check of the weights), editable masks.
3. **Phase 4:** split the skin engine; dedicated dark-spot (pigmentation) and redness tools, noise reduction, under-eye texture.
4. **Later phases:** brushes (heal, clone, dodge and burn), project files, presets per style, the NL plan engine, tiled rendering, optional DirectML after testing on the target hardware.

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
