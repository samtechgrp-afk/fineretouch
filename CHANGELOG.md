# Changelog

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
