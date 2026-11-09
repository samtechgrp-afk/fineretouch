# Skin Perfection Studio (v0.13)

Skin Perfection is a full-resolution blemish engine for face and body skin. It finds acne,
pimples, blackheads, whiteheads, dark spots, post-acne marks, small black dots and other
blemishes. It corrects each one with real skin texture, rescans to check what is still
visible, and reports detected, corrected and remaining marks with an overlay. Freckles,
moles, birthmarks, hair follicles, pores and identity are protected.

- Engine: `services/ai/app/engines/perfection.py`
- Settings: `adjustments.skinPerfection` (`packages/shared/src/perfection.ts` ↔
  `services/ai/app/schemas.py`, pinned by a test)
- Panel: `apps/web/src/components/panels/SkinPerfectionPanel.tsx`
- Preview, cache and overlay: `apps/web/src/lib/perfection.ts`,
  `apps/web/src/components/viewport/perfectionOverlay.ts`
- Endpoint: `POST /v1/perfection` (returns the report plus a preview of the result)
- Tests: `services/ai/tests/test_perfection.py`, `packages/shared/src/perfection.test.ts`,
  `apps/web/src/lib/perfection.test.ts`
- Benchmark: `python -m scripts.perfection_benchmark [--quick] [--old] [--json out.json]`
  (run from `services/ai`)

## 1. Why small marks were left behind (audit)

| # | Root cause | Where | Effect |
|---|---|---|---|
| 1 | Export "Auto" chose the GPU path on any hardware WebGL2, including integrated Radeon. That path only has shader approximations of spot healing. | `ExportModal.tsx` engine choice | Real detection never ran on most exports. |
| 2 | The detector worked on a copy shrunk to a 420 px face width. | `SPOT_WORK_FACE` in the old spot healer | Blackheads (2–6 px on a 700 px face) fell under its minimum size. |
| 3 | A 5 px median blur on the detection signal. | old spot detector | Removed pore-sized blackheads and dots before detection. |
| 4 | Connected components on a threshold map. | old spot detector | Dense acne merged into blobs that then failed the size and roundness tests. A 0.75 ring-isotropy test rejected clustered marks. |
| 5 | Heal blend `max(soft, heal×0.85) × 0.9` at 1×. | `heal_spots` | About 23% of each mark stayed visible. |
| 6 | No residual check. | pipeline | A weak repair was counted as done. |
| 7 | Face-only detection (body only through a parse option). | pipeline | Neck, chest, arms and legs kept their marks. |
| 8 | A local baseline that included the marks themselves. | old spot detector | Dense acne raised the baseline and hid the marks. |

Blurring more would not fix any of these. The engine below replaces the detection and correction steps.

## 2. Pipeline (stages A–I)

All stages run in the local AI engine on the full-resolution image (CPU, OpenCV + NumPy; no
new dependencies or models).

- **A. High-resolution analysis.** The skin bounding box is cropped and scaled so the face is
  1100 px wide (`WORK_FACE`). It is never shrunk below that, and it is upscaled only when the
  face is smaller. Body-only photos use the skin area. The crop is processed in 1024 px tiles
  with an apron, so a mark on a seam is found once (a test covers this).
- **B. Skin segmentation.** Face skin (MediaPipe landmarks + face parsing) and body skin (selfie
  multiclass segmenter: body skin × not-hair × skin probability), with clothing, hair and
  background excluded. Manual add/erase strokes from the skin-mask brush apply to both. Edges
  of the region and a closed-region guard keep repairs off lips, eyes, nostrils and hairlines.
- **C. Multi-scale detection.** Scale-space normalised Laplacian-of-Gaussian at 8 scales
  (0.0012–0.026 × face width), on lightness for dark marks and the first 5 scales for bright
  heads. A robust baseline excludes outlier marks. Each scale gets its own threshold, set
  from the median absolute deviation (MAD) of its responses. A tile's spread is clamped to
  0.8–1.25× the photo's skin-weighted median, so one acne-dense tile can't raise its own
  threshold.
- **D. Features.** Each candidate is measured for:
  - darkness, brightness, redness (a*) and brownness (b*)
  - ring isotropy (12 samples) and in-skin ring share
  - Hessian anisotropy, which rejects hairs and creases
  - a local shine map, which rejects specular highlights
  - an edge map
  - a pore reference (75th percentile of fine dark peaks), so pores stay pores
  - neighbour density (clusters mean freckles or stubble)
- **E. Classification.** Each candidate is sorted into one of these:
  - repaired: acne, pimple, blackhead, whitehead, blemish, dark spot, post-acne, small dot
  - protected: mole (large, very dark, sharp), freckle (small, faint, brown, in clusters),
    birthmark (large, flat), follicle (dense dark field in the beard area or on body skin)
  - rejected: shine
  Each category has its own sensitivity, and confidence comes from the margin over the threshold.
  Marks below the confidence threshold become "review" and are not corrected automatically.
- **F. Targeted correction.** Heal-type marks get a push-pull membrane fill from the
  surrounding ring. It is capped so it is never lighter than the ring's median, which prevents
  pale blobs. Pigment-type marks get a pigment-only correction above the pore scale. Acne also
  gets halo redness reduction.
- **G. Texture reconstruction.** Repairs take fine texture from the best donor patch nearby and
  mix the pore band back in (texture and pore preservation sliders), so a repair is not a flat
  disc.
- **H. Residual scan.** After every pass, each mark's contrast is measured again. A mark counts as
  corrected only if its contrast fell below 40% of the original (`VERIFY_RATIO`). The scan
  then runs detection again with extra sensitivity (residual slider) to find marks that are
  still visible. Each mark is attempted at most twice and there are at most 4 passes, so the
  loop always ends.
- **I. Quality verification and report.** The report gives:
  - detected, corrected, remaining, review, protected and ignored counts
  - counts per kind and mean confidence
  - texture retention (fine-band energy after vs before)
  - every mark (position, size, kind, confidence, region, status, contrast before and after)
  Area tools follow: redness, discoloration, texture/scar smoothing and under-eye. Finally,
  every pixel outside the skin region is copied back unchanged.

## 3. Controls (every one is a real engine input)

| Control | Field | What it does in the engine |
|---|---|---|
| Skin Perfection on/off | `enabled` | Runs stage 0c of the render pipeline. |
| 1× Natural / 2× Strong / 3× Maximum | `multiplier` | Sensitivity boost (+0/+10/+18), radius (×1/1.12/1.25), area tools (×1/1.35/1.6). Each is capped. |
| Overall strength | `strength` | Scales every correction. |
| Acne removal | acne, pimple, whitehead sensitivity | Detection threshold k for those classes. |
| Blackhead removal | blackhead, smallDot | Same. |
| Dark spot correction | darkSpot, postAcne, discoloration | Same, plus the discoloration area tool. |
| Small blemish detection | blemish | Same. |
| Body blemish correction | `bodyCoverage`, `bodySensitivity` | Body skin on/off, and the body threshold (category sensitivity × body/70). |
| Skin texture preservation | `texturePreservation` | Donor-texture mix in repairs, and how far area tools reach. |
| Pore preservation | `porePreservation` | Pore-test strictness, and the pore band put back after a repair. |
| Residual imperfection scan | `residualIntensity` | Sensitivity boost of the rescan (0–20). |
| Correction radius | `correctionRadius` | Repair footprint around each mark. |
| Edge protection | `edgeProtection` | Edge-map rejection and the edge guard around features. |
| Confidence threshold | `confidenceThreshold` | Below this a mark is "review" (not auto-corrected). |
| Passes | `passes` (1–4) | Correction + residual-scan rounds. |
| Preserve freckles / moles / birthmarks | `preserve*` | Off: those are corrected as dark spots. |
| Advanced, per category (13) | `categories[k]` | Enabled, sensitivity and strength for acne, pimples, blackheads, whiteheads, dark spots, post-acne, small dots, blemishes, discoloration, scars, texture, redness and under-eye. |

`test_perfection.py::test_every_control_changes_the_result` checks every control: each one
must change the output on a control scene.

### Per-mark actions (all undoable)

- **Click** a mark in the overlay to select it. **Shift-click** adds to the selection.
- **Correct** / **Ignore** the selected marks. Each becomes an override (`overrides`: x, y,
  radius, action). Correct also works where nothing was detected and adds a manual mark.
- **Correct all remaining** adds a correct override for every remaining or review mark at or
  above the confidence threshold, as one undo step.
- **Run another pass** adds one pass (up to 4).
- **Rescan** runs the engine again with the same settings.
- **Undo / redo**: overrides and settings are ordinary edit history. Earlier results are cached
  per photo, so undo shows them instantly.

The original file is never changed. Everything is stored as settings in the edit.

## 4. Preview and export

- **Preview.** The browser sends the original to `POST /v1/perfection` with the settings and
  gets back a preview-size render and the report. That render becomes the viewport's source
  pixels, so heal strokes, layers and every other slider stay live on top of it. Settings
  changes re-run the engine after 0.7 s. A change made during a run queues exactly one more
  run.
- **Export.** A photo with Skin Perfection on always renders in the AI engine at full
  resolution, even when the browser/GPU engine is selected. The export dialog says so. This
  removes root cause 1.

## 5. Measured results

Synthetic dataset: `services/ai/tests/perfection_fixtures.py` builds Lab skin at Fitzpatrick-
like tones I–VI. It includes lighting, multi-octave relief, a pore field, sensor noise,
labelled imperfections, and natural features (moles, a freckle cluster and a stubble field)
that must survive. The scenes are 10 in total: 1200 px normal density, and 1600 px dense acne.

| Metric (mean of 10 scenes) | Old tools (2× boost) | Skin Perfection (defaults) |
|---|---|---|
| Precision | – | **0.97** |
| Recall | – | **0.95** |
| F1 | – | **0.96** |
| Removal rate (contrast < 40% of before) | 0.52 | **0.94** |
| Natural features damaged (sum over 10 scenes) | 7 | **4** (all in the dense type VI scene) |
| Texture retention (fine-band energy) | 1.0 | **1.0** |

The old tools leave about half of all marks visible at 2×. Skin Perfection removes 94% and verifies each one.

Real photo (4480×6720 full-body portrait, face 712 px wide), at defaults:

- 189 marks detected, 154 corrected and verified, 35 left for review, and 254 natural
  features protected (freckles, moles, follicles on hands and legs).
- After "Correct all remaining": 172 corrected and 22 still visible. These are reported as
  remaining, not counted as removed.
- Texture retention is about 0.78, and pores stay visible at 100% zoom.
- Time: full-resolution render about 35–45 s on 4 CPU cores. Preview scan (2560 px long edge)
  about 10 s.

### Known limits

- Whiteheads on very deep skin (type VI) are found about half the time, because their bright
  heads have very low contrast. They show up as "remaining" rather than being missed silently.
- The 30 MP full-resolution render takes about 40 s on a 4-core CPU. The preview uses the
  2560 px proxy.
- Synthetic metrics are an upper bound. Real skin needs the review overlay for edge cases.

## 6. Performance and hardware

- **Target:** Windows, Ryzen with integrated Radeon, 16 GB RAM.
- The engine is CPU-only (OpenCV + NumPy) and needs no GPU. Peak memory at 30 MP stays within
  the engine's render admission budget (`FINERETOUCH_RENDER_MEMORY_MB`).
- Large blurs run at low resolution (`_lowres`), and detection runs tile by tile.
- New dependencies: **none**. Models: **none new**. It uses the existing MediaPipe face
  landmarker and selfie multiclass segmenter.

## 7. Phases and acceptance criteria

| Phase | Scope | Acceptance |
|---|---|---|
| 1 | Engine (A–I), renderer stage 0c, export always via engine, `/v1/perfection` | Removal ≥ 0.9 and F1 ≥ 0.9 on the benchmark. No protected features damaged on normal scenes. Pixels outside skin unchanged. Every control changes the result. |
| 2 | Skin Perfection Studio panel | Every control in the reference image is present and wired. 1×/2×/3× chips. Final quality check counts come from the report. |
| 3 | Overlay + per-mark actions | Select, ignore, correct, correct all remaining, another pass, rescan. All undoable. Counts update from a fresh engine run, never assumed. |
| Next | Real-photo test set with hand labels | Precision/recall on real skin, with 100% zoom crops per tone. |

Regression: `pytest services/ai/tests/test_perfection.py` (28 tests, about 150 s) and the
benchmark (`--quick` about 20 s, full about 1 min). Run with `--json` to keep a record.
