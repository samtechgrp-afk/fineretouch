# Professional Skin Finish: phase 3b fixes and validation (2026-11-12)

A separate, reviewable phase on the PR #14 branch. Nothing is merged, tagged or released; PR #13,
PR #14 and the release branch are untouched. Comparison images are in the project files
(`fineretouch/skin-finish/phase3b/`), not in the repository.

| Commit | Fix |
|---|---|
| `9d9074e` | 1. Any-shape dark-patch correction only on facial skin; body keeps round-mark correction |
| `5a492a1` | 2. Auto Retouch never changes the photo's exposure or contrast; new skin-only "Skin brightening" |
| `4ad0cdf`, `cf6127a` | 3. Glasses glare only on confidently detected glasses, manual override, skin hue kept |
| `b2a35b8` | 4. Dark-spot stage memory cut, output identical |

## 1. Dark spots: facial skin only for any-shape patches

**What changed.** The landmark face oval, kept off its edge (no jaw line, neck or body), is the
only place where a larger dark patch may be any shape. Everywhere else (jaw line, neck, arms,
legs, back, chest, or a photo without a face) a patch is lifted only if it is a round mark: strict
roundness at every scale, checked again in a wider window (the tip of a long shadow is only round
close up), and no body "marks" above the 0.024 face-width scale (a patch roughly 1 cm across). Individual
blemishes, post-acne marks and dark spots on the body are still corrected.

**Results, Professional style, ΔL\* = finished minus original (positive = lighter).**
Full resolution, the three photos from the phase 3 report:

| Photo | Region | Before fix mean / p95 | After fix mean / p95 |
|---|---|---|---|
| CRPixels-5 (full length, 25 MP) | body (legs, arms) | +6.1 / +15.6 | +0.5 / +2.6 |
| | jaw and neck band | +1.3 / +8.4 | −0.1 / +2.4 |
| | face (inside the oval) | +1.0 / +8.3 | +0.9 / +8.3 |
| CRPixels-15 | body | +2.7 / +9.9 | +0.2 / +2.2 |
| CRPixels-26 night | body | +3.0 / +8.6 | +0.3 / +1.9 |

Advanced on CRPixels-5: body +8.0 / +19.7 → +0.7 / +3.7. The face results are unchanged (within
0.1 L\*), so marks on the face are still corrected as before.

All 31 CRPixels photos and the 9 licensed photos (2400 px, Professional): body p95 lift fell from
+7 to +23 L\* to +2 to +5 L\* on 27 of 31 CRPixels photos. The other four keep +5 to +10 at
p95: CRPixels-23, -25 and -28 (on the CRPixels-23 lift map these are small round spots on the
legs, not the shading) and CRPixels-6 (see
"new finding" below). The share of
skin treated as dark spots fell from 24–66 % to 4–30 %. On the light-skin acne photos (back,
chest) round marks are still lifted while the broad shading is kept.

Images: `1-body-before-after-CRPixels5.jpg` (legs, hands and forearm, jaw and neck at 75 %),
`2-where-skin-is-lifted-CRPixels5.jpg` (lift map, Advanced).

## 2. Auto Retouch: the photo's exposure is never changed

**What changed.** Auto Retouch (and the assistant's recommendations and styles, which use it) no
longer sets Exposure or Contrast at all, so the background, clothing, hair, highlights and
shadows stay as shot, and your own Light sliders are not reset. When the photo looks
underexposed and a face is found, it sets the new **Skin brightening** slider instead:

- skin only, by the skin mask (needs a face); background, hair and clothing are untouched;
- a gentle gain (×1.12 at 100) that fades out on the skin's own highlights, so shading and
  highlights keep their shape;
- Auto Retouch uses at most 35 of 100; low-key portraits, bright photos and photos without a
  face get none; the slider is in Skin retouch and can be moved freely.

**Results on the 31 CRPixels photos and the night photo (1600 px).** Exposure: before, −13 to
+18; now never set. Background ΔL\* from the tonal part of Auto Retouch: before −5.8 to +5.6;
now 0.00 on every photo (largest ±0.02). Skin brightening chosen: 0 to 27 (night photo: 0).

**Still global: white balance.** Auto Retouch still applies a small temperature / tint and
vibrance correction to the whole frame. That moves the background by −0.3 to −0.7 L\* and about
2–3 ΔE in colour. This is a colour correction, not exposure, so I left it; see the question at
the end.

## 3. Glasses glare: only when glasses are found

**What changed.**

- A glasses detector (`engines/glasses.py`) looks for the frame: a thin, level bridge between the
  eyes plus a lens rim beside or below each eye. Each face now carries a glasses confidence.
- Glasses glare runs only on faces with confidence ≥ 0.6, in the export and in the preview.
  Uncertain or no glasses: skipped, nothing changes.
- Manual override: "Apply even if no glasses are detected" under the Glasses glare slider, and
  the panel says whether glasses were found.
- Skin colour around the eyes is protected even when the tool runs: outside the eye openings the
  correction may restore saturation a reflection took away (limited per pixel by that
  reflection's strength), but it can never turn the hue. Pixels it does not touch come back bit
  for bit.

**Results.**

- Snapshot 13 (your settings, Glasses glare 100) on the night photo: the face scores 0.19, the
  tool is skipped, and the render is identical to Glasses glare 0. The orange/blue patch is gone
  (`3-outer-eye-patch-gone-snapshot13.jpg`).
- No-glasses photos: the highest confidence on all 31 CRPixels and 9 licensed photos is 0.17
  (gate 0.6), so none of them runs the tool.
- Forced on all 31 bare faces at 100: the 99th-percentile hue turn of skin outside the eyes fell
  from 145° to 37°, and skin pixels moving more than 10 in a\*b\* from 29,620 to 17,812.
- Synthetic frames drawn on 6 photos: dark and medium frames 2–5 % of the eye distance thick
  are found on most faces; thin light-coloured metal frames often are not (then the tool is
  skipped unless you use the override).
- Real photos of people wearing glasses: not yet tested. The cloud cannot reach Wikimedia Commons
  (rate-limited), and the Remote Control session reached another PC instead of yours, so the
  photos will be downloaded and tested on your Ryzen PC.

## 4. Memory: dark-spot stage

The dark-spot stage frees its surround planes early, keeps each pass's weights at the detector's
reduced size, and updates its results in place. **Every pixel and every report value is
identical** to the previous commit (CRPixels-26 night and CRPixels-15, full resolution, Natural,
Professional and Advanced). Traced peak in the cloud: 5,062 → 4,555 MB and 7,427 → 6,718 MB.

## Ryzen 3 7330U benchmark

Not run yet: it needs a Remote Control session on your Ryzen PC, which was not reachable today.
The run is `scripts/finish_benchmark.py` on CRPixels-26 and CRPixels-5 (styles off, Professional,
Advanced; preview and export). It will be compared with phase 3: 66–75 s / 6.3 GB at 21.7 MP,
76–93 s / 7.3 GB at 25.2 MP and 115–123 s / 8.2 GB at 30 MP. Expected from the cloud trace:
about 0.5–0.7 GB lower peak, with time about the same.

## Tests

- Python 378 passed, 2 skipped (new: body shading kept / body marks lifted on tones II and V,
  elongated patch kept off the face, face core from landmarks, glasses detector, glare gate and
  override, skin hue lock which fails at 29.7° without the fix).
- Shared 100 passed, web 124 passed (Auto Retouch: no exposure or contrast at any lightness,
  capped skin brightening, low-key kept; glasses gate; new settings sanitised).
- Golden renders (old commit vs new, same machine): 25 of 27 identical. The two that change are
  the snapshot-13 cases with Glasses glare 100 on faces without glasses, as intended. With the
  finish off and default settings nothing changes.

## New finding (not fixed: needs your decision)

On CRPixels-6 and CRPixels-31 the face detector also reports a small second "face" in the
background (leaves on CRPixels-6, a building on CRPixels-31). Every face tool treats that area as
skin, so the Skin Finish lightens a patch of background there by up to about 15 L\*. This is
older than phase 2 (the old code does the same). A fix would reject a "face" that face parsing
does not see as face skin.
