# Pro studio retouch and batch style sync (v0.6)

Everything here runs on the user's machine. The engine (`services/ai`) does the full-resolution
work; the WebGL viewport previews each tool in real time.

## Retouch tools

| Tool | Engine (`services/ai/app`) | Preview (`apps/web/src/components/viewport/shaders`) |
| --- | --- | --- |
| Natural skin: smoothing, texture retention 0–100 %, spot healing | `retouch_engine.natural_skin_retouch`, `heal_spots` (v0.5) | `engine/shaders.ts` RETOUCH pass |
| Stray hair | `retouch_engine.remove_stray_hairs` (v0.2) | RETOUCH pass |
| Glasses glare | `reduce_glasses_glare` | `glareReduction.ts` |
| Teeth whitening and stain removal | `correct_teeth` | whitening in RETOUCH; stains export only |
| Clothing virtual iron and recolour | `remove_clothing_wrinkles`, `shift_clothing_color` | `clothingIron.ts` |
| Look LUT | `color_engine.apply_lut` | `lut3d.ts` (RGBA16F 3D texture) |
| QA overlay: glare veil, texture retention, garment mask | n/a | `textureRetention.ts`, `studio.ts` OVERLAY pass (display only) |

**Glasses glare.** A lens is modelled as `observed = scene * (1 - a) + a * G`. The lens
region comes from the eye landmarks. Reflections are found as bright structures above a
morphological opening of the lens. Their opacity `a` is measured as the smooth brightness lift
over the scene around the reflection, with dark detail behind the veil (iris, lashes) closed
first, so it does not read as "no reflection". Where `a <= 0.75`, the scene is recovered exactly.
Brighter cores are inpainted. Catchlights inside the eye opening and thin highlights on the
frame rims are skipped.

**Teeth.** Enamel is the bright, non-red part of the inner-mouth outline, so gums, tongue,
lips and the gaps between teeth are excluded. Whitening lowers b\* toward a slightly warm
target and lifts L\*. Stain removal pulls stain-sized patches toward the teeth's own median
colour. Every change is an offset computed from smoothed values, so enamel texture is carried
over. Moving or straightening teeth is not implemented.

**Virtual iron.** Creases are shading. They change luminance, not chromaticity, so the filter
works on log-luminance only, inside the garment mask. It uses normalised convolution, so skin
and background never bleed in. Bands are measured in face widths: weave (< 0.6 %) is kept,
creases (< 6 %) are cut by up to 90 %, folds (< 30 %) by up to 45 %, and the drape is never
touched. Seams, garment edges and bold prints are protected by an amplitude and gradient gate.
The garment mask comes from the selfie segmenter's clothes class. Without the model, the
fallback is the person region below the chin, minus skin.

## Batch style sync

`services/ai/app/batch_engine.py`; UI in `apps/web/src/components/panels/BatchSyncPanel.tsx`.

* **Smart**: copies the reference's edits and compensates exposure and white balance per photo
  (`smart_sync_adjustments`, pinned to the TypeScript `smartSyncAdjustments` by tests on both sides).
* **Exact copy**: the same adjustments on every photo.
* **Palette**: keeps the reference's retouch and replaces its colour and tone with an L\*a\*b\*
  covariance and tone-curve transfer to the *finished* reference (`palette_adjustments`).
* **Look LUT**: `fit_lut` fits a 17³ LUT from the reference's before and after. Trilinear
  splatting is followed by offset diffusion into colours the reference never contained.
  LUTs load and save as `.cube` (up to 33³).

### Local queue (desktop)

`BatchQueue` keeps jobs and items in SQLite (WAL) at `<data dir>/batch.sqlite3`. Items are
claimed atomically. A failed item is retried once and then marked failed without stopping
the batch. Items that were rendering when the app quit go back to pending on the next launch.
Outputs are written atomically and mirror the source sub-folders.

`BatchRunner` thermal profiles:

| Profile | Photos at once | OpenCV threads each | Pause between photos |
| --- | --- | --- | --- |
| quiet | 1 | half the cores | 0.75 s |
| balanced (default) | 1, or 2 on 8+ cores | the rest, minus one core | 0.1 s |
| performance | cores / 4 (max 4) | the rest | none |

HTTP: `POST /v1/batch/jobs`, `GET /v1/batch/jobs[/{id}]`, `POST /v1/batch/jobs/{id}/{pause|resume|cancel|retry}`,
`DELETE /v1/batch/jobs/{id}`, `POST /v1/batch/profile`, `POST /v1/lut/fit`, `POST /v1/lut/parse`,
`POST /v1/segment`. The queue reads and writes local folders, so it is on only in the desktop
engine (where `FINERETOUCH_DATA_DIR` is set) or with `FINERETOUCH_LOCAL_BATCH=1`.
