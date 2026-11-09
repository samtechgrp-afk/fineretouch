# Skin Perfection quality assurance (phases 2A–2B)

This is how Skin Perfection is measured. Each phase of the v0.13 upgrade (2B–2E) has to beat
the baseline below without a regression.

**Synthetic and real results are never pooled.** Synthetic scenes have exact ground truth but
are generated skin. They show whether a change helps or hurts. They are **not** evidence of
accuracy on real photographs. Any general claim waits for the real-photo test split (section 4).

## 1. Running it

```
cd services/ai
python -m scripts.perfection_benchmark --html report.html --json report.json --baseline qa/baselines/synthetic-2B.json
python -m scripts.perfection_benchmark --quick                       # 4 scenes, ~25 s (CI runs this as a test)
python -m scripts.perfection_benchmark --dev                         # tuning scenes (other seeds)
python -m scripts.perfection_benchmark --suite real --dataset D:\qa-photos --split validation --html real.html
python -m scripts.perfection_benchmark --suite real --dataset D:\qa-photos --coverage
python -m scripts.perfection_benchmark --perf D:\photos\IMG_0012.jpg --json perf.json
```

- `--baseline` prints every regression and exits with code 1 when there is one.
- `tests/test_perfection_qa.py::test_quick_suite_matches_its_baseline` is the same gate in the test suite.
  Its baseline (`synthetic-quick.json`) is the latest accepted phase, so a later phase cannot
  give back what an earlier one gained. `synthetic.json` stays the PR #13 reference.
- **Tuning vs scoring.** `--dev` runs the same 12 configurations with other seeds. Thresholds
  are tuned on those scenes only; the benchmark scenes confirm the result.
- `--perf` times one full-resolution render through the export path. It does one warm-up run,
  then two timed runs, and needs no labels. Run it on the target computer (Ryzen 3 7330U) for
  the phase 2E before/after comparison.

## 2. Metrics

The definitions are in `services/ai/qa/scoring.py`. They are identical for synthetic and real photos.

| Metric | Meaning |
|---|---|
| Precision | Flagged marks (corrected, remaining or review) that match a labelled imperfection, divided by all flagged marks. |
| Recall | Labelled imperfections with a flagged match, divided by all imperfections. |
| F1 | Harmonic mean of precision and recall. |
| Removal rate | Imperfections whose contrast against their ring fell to ≤ 40% of before. This is **measured on the output pixels**, not taken from the engine's report. |
| Residual rate | 1 − removal: imperfections still visible after processing. |
| Reported corrected but still visible | The engine said "corrected" but the mark is still visible. Shows whether the report can be trusted. |
| False positives | Flagged marks that match no imperfection. Split into those on a natural feature, false corrections (the engine changed the skin), and false positives per megapixel. |
| Natural features preserved | Labelled freckles, moles, birthmarks, follicles and hair whose contrast stayed ≥ 50%. Also counts how many were wrongly flagged. |
| Texture retention | Pore-band energy on clean skin, output ÷ input (1.0 = untouched). |
| Clean-skin change | Mean \|ΔL*\| on skin away from every label and mark. |
| Time / peak memory | Wall time of the engine (synthetic) or the full render (real, perf). Peak memory is the process's resident memory, sampled every 10 ms. |

Results are reported overall, by skin tone, by category, and by tone × category. Counts are
pooled across photos (micro-averaged), so a photo with many marks weighs more than one with few.

## 3. Baseline: the PR #13 engine (synthetic)

Built from 12 generated scenes: Fitzpatrick-like tones I–VI at normal density, dense acne on
II, IV and VI, hard light on III and VI, and one 2400 px scene. Machine: the 4-core cloud
container. Saved as `qa/baselines/synthetic.json`, with `synthetic-quick.json` for the 4-scene
gate.

| Tone | Scenes | Precision | Recall | F1 | Removal | Residual | False pos. | Natural kept | Texture | Time | Peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **All** | 12 | 0.961 | 0.943 | 0.952 | 0.937 | 0.063 | 30 | 0.995 | 1.000 | 5.4 s | 1,241 MB |
| I | 1 | 1.000 | 0.978 | 0.989 | 0.957 | 0.043 | 0 | 1.000 | 1.000 | 4.7 s | 487 MB |
| II | 2 | 0.952 | 0.969 | 0.960 | 0.963 | 0.037 | 8 | 0.968 | 1.000 | 5.7 s | 628 MB |
| III | 2 | 0.989 | 0.978 | 0.984 | 0.978 | 0.022 | 1 | 1.000 | 1.000 | 4.0 s | 568 MB |
| IV | 2 | 0.988 | 0.963 | 0.975 | 0.956 | 0.044 | 2 | 1.000 | 1.000 | 5.9 s | 658 MB |
| V | 2 | 0.989 | 0.967 | 0.978 | 0.946 | 0.054 | 1 | 1.000 | 1.000 | 6.3 s | 1,241 MB |
| **VI** | 3 | **0.910** | **0.874** | **0.891** | **0.874** | **0.126** | **18** | 1.000 | 1.000 | 5.5 s | 661 MB |

Imperfections found, by tone and category:

| Tone | Acne | Pimple | Blackhead | Whitehead | Dark spot | Post-acne |
|---|---|---|---|---|---|---|
| I | 10/10 | 5/5 | 11/12 | 6/6 | 8/8 | 5/5 |
| II | 35/35 | 17/17 | 39/42 | 21/21 | 27/28 | 16/17 |
| III | 20/20 | 10/10 | 22/24 | 12/12 | 16/16 | 10/10 |
| IV | 35/35 | 14/17 | 39/42 | 21/21 | 28/28 | 17/17 |
| V | 20/20 | 9/10 | 22/24 | 12/12 | 16/16 | 10/10 |
| **VI** | 44/45 | **18/22** | **48/54** | **15/27** | **33/36** | 22/22 |

- **Deepest tone is the weakest.** Type VI is weakest on every count: whiteheads 0.56, pimples
  0.82, and 18 of the 30 false positives. Phase 2B targets this.
- **Natural features:** 646 of 650 follicles were kept, and all moles and freckles were kept.
  The 4 damaged follicles are in the dense type II scene.
- **The report is mostly honest:** 3 marks were reported corrected but are still visible.
- **The old tools were weaker.** The pre-v0.13 tools removed 0.542 of the same marks
  (`--old`), compared with 0.937 for this engine.

### 30 MP performance baseline (export path)

`qa/baselines/perf-30mp-container.json`: a 4480×6720 JPEG (30.1 MP) on the 4-core container.

- Median **47.4 s** per photo, with the engine itself taking 33–36 s.
- Peak memory **6.07 GB**.
- The phase 2E target is ≤ 25 s and ≤ 4 GB on the Ryzen 3 7330U. That machine needs its own
  baseline, from `--perf` run on it.

## 4. Real-photo validation set

### Consent and handling

- Only photos with a signed consent for **internal quality testing** are used (`"consent": "internal-qa"`
  in the manifest; others are refused by the loader). No external processing, no model training and
  no sharing without a separate consent.
- The set stays on the user's own computer. The benchmark runs locally, and label files hold
  only coordinates and categories.
- Use unretouched originals only (camera RAW or straight-out-of-camera JPEG). Retouched exports
  would hide the marks being measured.

### Folder layout and manifest

```
qa-photos/
  manifest.json
  S01/IMG_0012.CR3
  S01/IMG_0012.CR3.labels.json      <- saved from the app's label mode
  S02/...
```

```json
{"photos": [
  {"file": "S01/IMG_0012.CR3", "subject": "S01", "tone": "monk-8", "lighting": "hard",
   "region": "face", "conditions": ["whitehead", "stubble"], "consent": "internal-qa"}
]}
```

- **Splits are by subject.** All photos of one person go to the same split. The split is
  assigned from a hash of the subject id: 40% validation (used for tuning) and 60% test
  (scored only for release reports). A manifest may also set `"split"` explicitly. The loader
  refuses a subject that appears in both splits.
- **Model training.** If training is ever introduced, it gets its own split of new subjects.

### Labelling in the app

The **Test labels (QA)** section in the editor:

1. Turn on **Label mode**. Pick a kind (imperfection to remove, or natural feature to keep) and a
   label size, then click each mark. Click a label to change it, and Alt-click to delete it.
2. **Suggest from engine** (optional) adds the engine's marks as dashed, **unconfirmed** labels.
   The benchmark ignores them until each one is confirmed, so the engine is never graded
   against its own output.
3. Enter the annotator and the reviewer (ideally a dermatologist or senior retoucher), then
   **Save labels** to write `<photo>.labels.json`. Labels never change the photo, the edit or
   exports. They are also kept in the browser, per photo, until saved.

### How many photos

The coverage matrix is:

- **Tone groups (4):** Monk 1–3, 4–6, 7–8 and 9–10.
- **Conditions (8):** dense acne, whiteheads, blackheads, hyperpigmentation, post-acne marks,
  freckles/moles, stubble, body blemishes.
- **Lighting (3):** soft, hard, mixed.

| Split | Per-cell minimum | Lighting bound (4 × 3 × min) | Condition bound (4 × 8 × min ÷ 2.5 conditions per photo) | Photos needed |
|---|---|---|---|---|
| Test | 5 | 60 | 64 | **64** |
| Validation | 3 | 36 | 39 | **39** |
| **Total** | | | | **≈ 103 photos** |

- **Subjects:** at least 30 distinct subjects (about 3–4 photos each), with half of them at Monk 7–10.
- **Body blemishes:** need photos that show neck, chest, back or arms.
- **What 10–20 photos give.** They fill at most a few cells, which is enough to start tuning
  (validation) and to see failure modes. It is not enough for a release claim. `--coverage`
  lists the empty cells after each batch.

## 5. What each later phase must show

- **Before / after.** Each phase reports this table again, with the baseline comparison and no
  regression beyond the tolerances in `qa/report.py`. A real-photo validation run is added
  once labels exist.
- **Phase 2B (tone fairness).** On type VI, whitehead recall must reach ≥ 0.85 (from 0.56) and
  pimple recall ≥ 0.95 (from 0.82). The worst tone's recall must be ≥ 0.9 × the best tone's.
  No tone may lose more than 0.02 precision.

## 6. Phase 2B results: tone-fair detection

### What changed

These are engine changes in `services/ai/app/engines/perfection.py`; the details are in `SKIN_PERFECTION.md`, stages A, D, E and H.

- **Independent texture references.** There is a bright reference for light marks and an a* reference for red marks. Sensor noise is taken out of the luma references.
- **Tone-fair texture test.** Clearly significant marks outside lit texture can pass against their own kind of texture.
- **Native-resolution finest scales.** These run when the face is wider than 1100 px.
- **Pimple head + dome fusion.**
- **Rim rejection.**
- **Rescan changes.** Rescan residue now joins its mark. Colourless marks are not boosted in the rescan.

### Root cause, re-measured on the tuning scenes

The dark pore reference grows from 0.013 (tone I) to 0.032 (tone VI) relative luma, while the marks have the same relative contrast. Deep-skin whiteheads (pore ratio 1.8–2.2) and red acne (1.4–2.3) failed the ≥ 2.45 pore test even at 10–25 spreads above the detection threshold.

### Synthetic benchmark (held-out scenes) against the PR #13 baseline

There are no regressions (`--baseline qa/baselines/synthetic.json` exits 0). The full report is saved as `qa/baselines/synthetic-2B.json`.

| Tone | Precision | Recall | Removal | False pos. | Natural kept |
|---|---|---|---|---|---|
| **All** | 0.961 → **0.984** | 0.943 → **0.967** | 0.937 → **0.958** | 30 → **12** | 0.995 → 0.995 |
| I | 1.000 → 1.000 | 0.978 → 0.978 | 0.957 → 0.957 | 0 → 0 | 1.000 |
| II | 0.952 → 0.975 | 0.969 → 0.969 | 0.963 → 0.963 | 8 → 4 | 0.968 |
| III | 0.989 → 1.000 | 0.978 → 0.978 | 0.978 → 0.978 | 1 → 0 | 1.000 |
| IV | 0.988 → 0.988 | 0.963 → 0.981 | 0.956 → 0.975 | 2 → 2 | 1.000 |
| V | 0.989 → 1.000 | 0.967 → 0.978 | 0.946 → 0.957 | 1 → 0 | 1.000 |
| **VI** | 0.910 → **0.970** | 0.874 → **0.942** | 0.874 → **0.932** | 18 → **6** | 1.000 |

Type VI by category: whitehead 15 → **24** of 27, pimple 18 → **22** of 22, acne 44 → 45 of 45, dark spot 33 → 33 of 36, blackhead 48 → 48 of 54.

The phase 2B targets are all met on the synthetic benchmark:

| Target | Result | Met |
|---|---|---|
| Type VI whitehead recall ≥ 0.85 | 0.89 | ✓ |
| Type VI pimple recall ≥ 0.95 | 1.00 | ✓ |
| Worst tone recall ≥ 0.9 × best | 0.942 / 0.981 = 0.96 | ✓ |
| No tone loses more than 0.02 precision | none loses any | ✓ |

### Real-photo check (no labels yet)

Five already-retouched 30 MP portraits (deep skin, faces 350–740 px wide) were run through first-pass detection. These photos have few real marks left, so extra detections there are mostly false.

| Engine | Marks detected (5 photos) | Change |
|---|---|---|
| PR #13 | 976 | |
| First 2B version, without the lit-texture guard | 1,999 | +105%, mostly sheen and shimmer read as whiteheads and acne |
| Phase 2B as committed | 989 | +1.3% |

This check is why the guard and the significance condition exist. It does not measure accuracy; the labelled real test set will.

### Time and memory

- **Synthetic:** the mean engine time is 5.4 s → 6.8 s per scene. The native fine-scale pass costs most on the 1600 and 2400 px scenes.
- **30 MP export path** (CRPixels-1, face 712 px, so no native pass): the median is 47.4 s → **51.1 s** (+8%), with the engine at 33–36 s → 37 s. Peak memory is 6.07 → 6.12 GB.

Phase 2E owns the ≤ 25 s / ≤ 4 GB target.

### Remaining limitations

- **Type VI dark spots.** 3 of 36 are classified as freckles: they are brown, faint and clustered. These go to phase 2C, through cluster separation and Needs Review.
- **Type VI blackheads.** 6 of 54 are inside the stubble-field logic. These go to phase 2C.
- **Colourless blemishes on real photos.** The retouched photos still show about 540 "blemish" detections in both PR #13 and 2B. These are likely skin relief. Phase 2C (review state) and 2D (quality checks) address them, and the real validation labels will settle it.
- **Real-photo evidence.** The native-resolution pass has not run on a real photo yet: none of the available photos has a face wider than 1100 px.

