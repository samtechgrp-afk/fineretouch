# Skin Perfection quality assurance (phases 2A–2C)

This is how Skin Perfection is measured. Each phase of the v0.13 upgrade (2B–2E) has to beat
the baseline below without a regression.

**Synthetic and real results are never pooled.** Synthetic scenes have exact ground truth but
are generated skin. They show whether a change helps or hurts. They are **not** evidence of
accuracy on real photographs. Any general claim waits for the real-photo test split (section 4).

## 1. Running it

```
cd services/ai
python -m scripts.perfection_benchmark --html report.html --json report.json --baseline qa/baselines/synthetic-2C.json
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


## 7. Phase 2C results: dense clusters, stubble and Needs Review

### What changed

- Review marks carry a reason (`lowConfidence`, `mayBeFollicle`, `mayBeFreckle`). A mark that
  may be a natural feature is never corrected unless the user chooses it.
- Stubble-field and freckle-group outlier tests, a lone-dot and a hair-direction (coherence)
  follicle test, and the cluster split. Rules and thresholds: `SKIN_PERFECTION.md`, stage E.
- Thresholds were tuned on the `--dev` scenes only. The held-out scenes were run once for
  confirmation.
- The report gains **False flags left for review** (`falseReviews`): false positives that were
  only asked about, pixels untouched. False corrections are the ones that changed pixels.

### Synthetic benchmark (held-out scenes) against phase 2B

| Metric | 2B | 2C |
|---|---|---|
| Precision | 0.984 | 0.979 |
| Recall | 0.967 | **0.985** |
| F1 | 0.975 | 0.982 |
| Removal rate | 0.958 | 0.960 |
| False positives | 12 | 16 |
| … of which corrected (false corrections) | 11 | **7** |
| … of which left for review, untouched | 1 (12 − 11; not split out in 2B) | 9 |
| Natural features damaged | 4 | **0** |
| Type VI dark spots found | 33 / 36 | **36 / 36** |
| Type VI blackheads found | 48 / 54 | **51 / 54** |

**The gate flags one regression: false positives 12 → 16** (tolerance 2). All four extra are
freckles flagged `mayBeFreckle`. They were left untouched, which is the intended trade: the
engine asks instead of correcting. Over the same scenes the marks that changed natural skin
fell from 11 to 7, and natural features damaged fell from 4 to 0 (stubble follicles on type II
are no longer corrected). The newly found type VI dark spots and blackheads are review marks,
so removal stays the same until the user decides. `qa/baselines/synthetic-2C.json` is the new
reference. The quick CI baseline (4 scenes) is ratcheted to 2C: recall 0.948 → 0.976,
false corrections 5 → 5, and false positives 6 → 8, the two extra being review flags only.

### Real-photo check (no labels yet)

On the five retouched portraits, first-pass detections are 989 → 1,001. Of these, 23 marks now
carry a feature reason (15 `mayBeFollicle`, 8 `mayBeFreckle`) and are left untouched.

### In the app

On a real 30 MP portrait (CRPixels-5), the panel reports 13 marks that may be a freckle or a
follicle. Selecting one shows its reason. **Correct all remaining** left all 13 untouched and
still waiting for review. This check found a bug: an override placed for one blemish also
forced a "may be a follicle" mark or a protected feature inside its circle. After the fix,
the bulk action corrected 346 marks instead of 358; the 12 it no longer forces were those
neighbours. An engine test covers this case.

### Remaining limitations

- Each extra freckle asked about costs the user a click. The real validation photos will show
  whether 1.4× / 1.8× asks too often.
- Coherence (hair direction) changed no count on the synthetic scenes, which have no hair
  shafts. Its effect is unmeasured until real stubble photos are labelled.
- Colourless "blemish" detections on the retouched photos (about 540) are unchanged. Phase 2D's
  correction checks and the real labels address them.

## 8. Phase 2D results: correction quality

### What changed

- Seven quality checks per repair (residual, colour ΔE2000, texture, pores, edges, repetition,
  surround). Definitions and limits: `SKIN_PERFECTION.md`, stage H2. A mark counts as corrected
  only when its repair passes all checks that apply to it.
- A failed repair is retried from the original skin with another donor, wider. If it still
  fails, the original skin is restored and the mark is asked about (`repairFailed`). The user
  can still choose Correct, which keeps the repair and shows the failed checks.
- Repair fixes the checks found: donors no longer carry neighbouring marks into a repair, the
  donor search window is wide enough on dense skin, pore reinjection no longer leaves a ghost
  ring of the mark's edge, and the fill tone is taken from skin around the mark, not the mark.
- Marks waiting for review are untouched by the area tools too (the tone tool used to lighten a
  restored dark spot).
- Later residual scans only add marks where the pass changed pixels.
- **Repair quality review page** for calibrating the thresholds on real photos:

      python -m scripts.perfection_benchmark --suite real --dataset D:/qa-photos --split validation \
          --quality-gallery quality-review.html

  It shows every failed repair and 4 passed ones per photo, before | after, each check's value
  against its limit, and boxes to tick "false alarm" (a failed repair that looks natural) or
  "looks wrong" (a passed repair that does not). Thresholds are set on the validation split only.

### Synthetic benchmark (held-out scenes) against phase 2C

| Metric | 2C | 2D |
|---|---|---|
| Precision | 0.979 | 0.978 |
| Recall | 0.985 | 0.985 |
| Removal rate | 0.960 | 0.958 |
| Corrected but still visible | 4 | **1** |
| False positives | 16 | 17 |
| … corrected (false corrections) | 7 | **5** |
| … left for review, untouched | 9 | 12 |
| Natural features damaged | 0 | 0 |
| Texture retention (clean skin) | 1.000 | 1.000 |
| Mean time per scene | 7.6 s | 10.2 s |
| Peak memory | 1,167 MB | 1,307 MB |

The gate reports no regression. `qa/baselines/synthetic-2D.json` is the new reference; the
quick CI baseline is unchanged (it passes). On the dev scenes, false positives were 14 → 17 and
false corrections 8 → 10: marks on untouched skin found by the first residual scan, whose tile
statistics shift with the repairs around them. Limiting that scan to changed skin removed them
but lowered type VI removal (0.917 → 0.893), because it also finds real marks the first
detection missed on deep skin, so it was kept. Only later scans are limited to changed skin;
that rule changed no count on these scenes.

The synthetic scenes have clean, uniform skin texture, so nearly every repair passes there; the
checks matter on real skin.

### Real-photo check (retouched portraits, no labels)

These photos are already retouched, so most detections are faint and a good engine changes
little. Fine texture lost over the whole skin: 3.4% → **1.6%** (photo 1) and 5.8% → **2.7%**
(photo 12). Of 1,274 repairs tried on the five photos, 366 passed, 742 were restored and asked
about, and 166 stayed remaining (partly corrected colour). Failures per check: texture 663,
colour 399, residual 337, pores 266, edges 15, repetition 2, surround 1. Most texture failures
are in sheen and highlight areas, where the repair is smoother than the lit skin around it.

### In the app

On a 30 MP portrait (CRPixels-5) the panel reports 225 repairs that did not pass, and selecting
one shows, for example, "Failed: colour, texture. The original skin was restored; choose
Correct to repair it anyway." This check found a bug: **Correct all remaining** also corrected
12 of the failed repairs, because they touched the circle of a remaining blemish beside them.
A correct override now forces a neighbour only when its centre is inside the circle. After the
fix, 4 of the 225 passed the checks when measured again and 2 were still forced: one sat inside
a remaining blemish's circle, and one was the nearest new detection to an override whose own
mark was not found again. An engine test covers the circle rule.

### Remaining limitations

- **The thresholds are provisional.** On the retouched photos 58% of attempted repairs fail,
  mostly on texture in lit areas. Whether those are real problems or false alarms needs the
  review page on the validation photos. Until then a strict gate means more marks to review.
- Time is up 34% on the synthetic scenes (retries and wider donor search). Phase 2E works on
  this.

## 9. Phase 2E results: performance

### What changed (the output is unchanged)

Every change keeps the result pixel-identical to phase 2D. That was checked on six benchmark
scenes, including the 2400 px scene that uses the native-resolution pass, on two real photos
through the engine, and on one real photo through the whole render. The held-out benchmark gives
the same scores as `synthetic-2D.json` on every quality metric.

- **Parallel detection tiles.** Up to 4 threads, no more than OpenCV's thread count, and at most
  about 900 MB of tiles in flight. Memory that worker threads free is given back to the system
  afterwards: glibc otherwise keeps about 1 GB of it.
- **Parallel finishing strips and skin probability** in the render pipeline.
- **Residual scans reuse the first scan's masks.** Skin, body, protected features and zones are
  built once per render.
- **Faster classification loop.** Per-kind settings and thresholds are cached, the edge term is
  computed once for all candidates, and a small-array median replaces `np.median`.
- **Less memory.** The blurred photo copies are freed before the engine runs. No copy is made
  of an image that is already float32. Clipping and blending are done in place. Undoing a failed
  repair only touches the area around it. The redness, pigmentation and relief tools free their
  temporaries early.

### Before and after (this container: 4-vCPU Xeon 2.3 GHz, 16 GB, Linux)

`python -m scripts.perfection_benchmark --perf <photo> --repeats 2`, Skin Perfection on, full render:

| Photo | 2D time | 2E time | 2D peak | 2E peak |
|---|---|---|---|---|
| CRPixels-1 (30.1 MP) | 63.0 s | **36.6 s** | 6,500 MB | **4,094 MB** |
| CRPixels-5 (25.2 MP) | 55.0 s | **39.8 s** | 6,264 MB | **4,452 MB** |
| Synthetic held-out, mean per scene | 10.2 s | **7.4 s** | 1,307 MB | 1,246 MB |

(The 2D figure for CRPixels-1 is one timed run; the others are the median of two.)

### Against the target (30 MP in 25 s, 4 GB, on the Ryzen 3 7330U)

Not met on this container. Time is 37 to 40 s against 25 s, and the 25 MP photo peaks at
4.45 GB. The Ryzen has 4 cores and 8 threads at up to 4.3 GHz, against 4 virtual cores at
2.3 GHz here, so it should be faster, but that is untested. On Windows, run:

    cd services\ai
    python -m scripts.perfection_benchmark --perf D:\photos\<a 30 MP photo>.jpg --repeats 2 --json perf.json

### Where the time goes now (30 MP, this container)

- Detection: about 15 s over 3 scans. Each tile runs 8 dark and 5 bright scales.
- Area tools: redness and pigmentation 6 s, relief 1.5 s.
- Repairs: 3.7 s.
- Rest of the render: about 7 s.

### Decision (2026-11-09)

Sam accepted about 37 s for a 30 MP photo. Quality comes first: full-resolution detection and
every verification stage stay. Later performance work must leave the output unchanged, so the
two options below are declined unless they are shown to cause no quality loss.

### Options that would change the output (declined)

- **Body-skin weight floor.** On CRPixels-1, weights between 0.1% and 5% (segmentation noise)
  cover 90% of the frame, so the area tools work on the whole photo, at about 1% strength in the
  background. Treating weights under 2% as no skin would limit the work to the real skin (5% of
  the frame here) and keep the tools off clothes and background. The pixels that change are
  ones the tools touched at no more than 2% strength.
- **Area tools on a smaller proxy.** For faces under 320 px they run at nearly full size
  (CRPixels-5: 91%).
