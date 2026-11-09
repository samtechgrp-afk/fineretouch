# Changelog

## v0.12.0 (unreleased): rendering and performance (Phase 7)

- **High-resolution GPU export in tiles.** Photos larger than the GPU's texture limit or the chosen GPU memory budget (Low 512 MB / Balanced 1.5 GB / High 3 GB) render in overlapping tiles instead of being scaled down.
  - Each tile's margin covers the widest blur, frequency-separation and bokeh reach.
  - Measured against a single pass: within 1/255. With background bokeh, mean 0.09/255. No seams.
- **Preview caching.** The renderer's four stages (retouch, studio, layers, bokeh) re-run only when their own inputs change.
  - Pan and zoom no longer re-render. A bokeh change re-runs only bokeh.
  - The viewport shows "cached" or "n/4 stages".
- **CPU/GPU selection.** A saved default engine (Auto / GPU / CPU engine) with GPU detection. Auto picks the CPU engine when only a software GPU is available, and the dialog says why.
- **Memory management.**
  - The editor keeps decoded previews for the 12 most recent photos and decodes others again when opened.
  - The engine admits full-resolution renders against a RAM budget (60% of memory or `FINERETOUCH_RENDER_MEMORY_MB`), so parallel exports can't push a 16 GB laptop into swap. `/health` reports the budget.
- **Faster, leaner CPU engine.** The per-pixel stages run in 256-row strips, and coarse blur bands are built only when used.
  - 20 MP plain render: 15 s / 3.0 GB → 7.5 s / 1.8 GB.
  - 20 MP with skin tools: peak 3.6 GB → 2.7 GB.
- **Export optimisation.** One WebGL context for a whole GPU export, and the next photo decodes while the current one renders.
- Details, measurements and status: `docs/PERFORMANCE.md`.

## v0.11.0 (unreleased): retouch assistant (Phase 6)

- **Retouch assistant** panel: every suggestion is an **edit plan**. The plan lists each change with its value and the reason for it. You can untick a step, change any value, and scale everything with plan strength. The plan is previewed live and applied only when you **approve**, as one undo step. **Reject** discards it.
- **Ask for an edit**: plain-English requests ("remove acne but keep the freckles, brighten the eyes slightly") are mapped to FineRetouch's tools.
  - It understands intensity words, numbers and 2x/3x, and "keep / don't / less".
  - It tells weakening a tool ("less smoothing") apart from removing a problem ("less acne").
  - It runs offline with a fixed vocabulary and no language model, so nothing is uploaded. It lists what it did not understand.
  - It **declines** requests that would change face shape, body shape or skin colour.
- **Recommended for this photo**: corrections from the photo's own measurements, each with the measurement behind it.
  - New **skin-tone safeguard**: brightening is capped so measured skin rises at most 10 L*, so a darker-skinned subject keeps their real tone.
- **Retouching styles**: Natural, Corporate headshot, Beauty, Wedding, Mature skin and Men's grooming.
  - Each style is built for each photo: light and colour come from that photo's measurements, and skin work is matched to its measured skin texture.
  - With several photos selected, a style is rebuilt for each photo on approval.
- The assistant never touches healing, clone, layers or mask strokes. Export and autosave ignore an unapproved plan.

## v0.10.0 (unreleased): professional retouching tools (Phase 5)

- **Healing brush**: Alt-click sets the source; healing takes the source's texture and the tone and colour around the stroke (push-pull membrane fill of the difference), so there is no visible patch. Without a source, **spot healing** picks one automatically and avoids sources with their own blemish. The source is aligned after the first stroke, as in Photoshop.
- **Clone stamp** with hardness and opacity.
- **Retouch layers** (up to 8 per photo, reorderable, switchable, renamable):
  - **Frequency separation**: split radius, edge-aware tone smoothing, texture 0 to 200%. Starts on the AI skin mask.
  - **Dodge & burn**: dodge, burn and erase brushes with flow (strokes build up), up to 1.5 EV in linear light (colour kept), limited to shadows, midtones or highlights if wanted.
  - **Correction brush**: local exposure, contrast, saturation, temperature, tint, smoothing and texture.
- **Advanced masks** per layer: start from nothing, the AI skin or everything; add / erase strokes; feather; invert; luminance range; density; opacity; mask overlay (M).
- All non-destructive and stored in the edit (`localRetouch`); works in the live preview, the browser export and the engine (`services/ai/app/engines/local.py`). Healing and clone match between browser and engine to uint8 rounding (shared test vectors).
- Pro tools belong to one photo: paste, sync, presets and saved presets never copy them.

## v0.9.0 (unreleased): skin engine (Phase 4)

- The 1,700-line `retouch_engine.py` is split into per-tool modules under `services/ai/app/engines/` (common, boost, spots, skin, hair, bokeh, style, studio, clothing). It still re-exports every name, so nothing that imports it changes. All 203 engine tests passed unchanged after the split.
- **Redness reduction** (`rednessReduction`): diffuse redness, flushed cheeks and rosacea, separate from acne removal. The tone layer's excess Cr over the person's own surrounding skin colour (a skin-only average that leaves the red areas out) is pulled back. Brightness, shading and pores are never changed.
- **Skin noise reduction** (`skinNoise`): camera grain and colour noise on skin. The noise level is measured on the chroma fine bands, where pores can't inflate it, then the finest luma band is soft-thresholded at 2.5x that level and skin chroma is averaged. Clean files are barely touched; pores stronger than the noise stay.
- **Under-eye texture** (`underEyeTexture`): fine lines and crepey skin in the landmark under-eye region. The crease band is faded (dark lines fully at 100%, lighter ridges by 35%); pores and the tone layer are kept. Capped at 90 for 2x/3x.
- Each new tool has its own slider, switch, 1x/2x/3x, opacity and reset, a live preview, and works on export, batch and the cloud worker. Protect brush strokes now also cover the under-eye tools.


## v0.8.0 (unreleased): face parsing and editable skin masks (Phase 3)

- **Semantic skin mask.** The MediaPipe selfie multiclass segmenter (Apache-2.0, CPU, already bundled) now parses each photo into face skin, body skin and hair. Face skin near a detected face extends the landmark mask to the hairline, ears and jaw; hair lying across the face is taken out; eyes, brows and lips stay cut out. Used by the engine export (`services/ai/app/masking.py`, `pipeline.build_masks`) and by the live preview through the new `POST /v1/parse` endpoint. Parsing runs on the local engine; nothing is uploaded anywhere else.
- **Include neck & body** switch: also retouches neck, chest and arm skin found by the parsing.
- **Mask brush** in the viewport: *Add skin* paints in areas the AI missed, *Protect* keeps an area untouched by every skin tool. Brush size, stroke feather, clear strokes, and the mask overlay (M) shows strokes. Each stroke is one undo step and is saved with the photo's edit (autosave). Strokes are applied after colour gating, so they are absolute.
- Strokes belong to one photo: copy/paste, sync, smart sync and presets carry the mask options but never another photo's strokes.
- Without the engine or its segmentation model, the landmark-only mask is used and the panel says so.


## v0.7.0 (unreleased): retouch control system

- Per-effect controls for 11 retouch effects: enable switch, 1x / 2x / 3x multiplier, opacity and reset (`packages/shared/src/effects.ts`, `services/ai/app/effects.py`).
- Global retouch strength that scales all retouch effects without changing any individual setting. Light and colour adjustments are not affected.
- Multipliers use per-effect soft-knee curves with safety caps (eyes, teeth 85; sharpening 90). The spot healing multiplier also widens the spot detector.
- The WebGL preview, the in-browser export and the Python engine all resolve the same effective values. Both resolvers are pinned to the same test vectors.
- Edit autosave and recovery: edits are saved in local storage under a fingerprint of each original (name, size, modified time, pixel size) and restored when that file is re-imported.
- Separate skin tools, each with its own slider, switch, 1x/2x/3x, opacity and reset: **Skin smoothing**, **Texture preservation**, **Acne removal** (red spots, pimples, raised bumps, then post-acne redness around them; clear skin untouched), **Dark spot removal** (isolated dark marks healed with real texture, then brownish pigmentation evened one band coarser; neutral facial shading excluded), **Blemish removal** (general clean-up of every spot type), **Skin tone correction** and **Under-eye correction**. A **Keep freckles** switch leaves small, faint dark spots alone.
- Backward compatible: edits, presets and API requests without the new fields render exactly as before.

## v0.6 and earlier

See the merged pull requests #1 to #6: skin boost, stray hair, depth bokeh, style sync, RAW, the desktop app, spot healing, the colour mixer, natural retouch, glasses glare, teeth, clothing, LUTs and batch sync.
