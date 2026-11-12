# Professional Skin Finish: phase 3 validation (2026-11-12)

Validation only: nothing here is merged, tagged or released. Comparison images are in the project
files (`fineretouch/skin-finish/phase3/`), not in the repository.

## Verdict

**Not ready for wider testing yet.** The controls work and the face results are mostly natural, but
two defects would be visible to users and one existing tool damages photos:

1. **Body skin is flattened and lightened (new, phase 2).** On CRPixels-5 (full-length, 25 MP) the
   dark-spot correction treats the natural shading of the legs, the forearm and the jaw as "large
   dark patches" and lifts them by up to 15–19 L* (95th percentile, Professional / Advanced). The
   legs lose their roundness and look lighter and greyer. The reason: since phase 2, patches of
   0.024 face widths or more may be any shape outside the landmark fold mask, and the fold mask
   only covers the face. Engine-reported mean skin lift on that photo: Natural +3.8, Professional
   +5.7, Advanced +7.4 L*. On face-only photos it is +1 to +3.4 L*.
2. **Auto Retouch still changes daylight exposure.** Low-key photos are kept (night photo:
   background ΔL* 0.0), but on 30 of 31 daylight photos Auto Retouch still sets exposure −13 to
   +18, moving the background −5.8 to +5.6 L*. That breaks "must not automatically brighten the
   whole photograph".
3. **The orange/blue patch at the outer eye is the Glasses glare tool (v0.6), not the finish.**
   `reduce_glasses_glare` runs whenever its slider is above 0, with no check for glasses. On a bare
   face it reads skin highlights inside the lens-shaped area as lens reflections, estimates one
   (warm) glare colour and subtracts it: the bluish rim light turns blue, the skin below turns
   orange. Snapshot 13 has Glasses glare at 100; with it at 0 the patch is gone. At 30 it already
   shifts colour by more than 10 ΔE (chroma) near the eyes on 27 of 31 CRPixels photos.

## What passed

| Check | Result |
|---|---|
| Off by default | `SkinFinishSettings()` and `DEFAULT_SKIN_FINISH` are off |
| Off = unchanged | Overall 0 is bit-identical to off (night and daylight); golden: only approved dark-circle cases differ |
| Every switch | Off changes the render, and equals amount 0 bit for bit (all 8 effects, 2 photos) |
| Every slider | Amount 100 differs from the style value (all 8, 2 photos) |
| Reset | Back to the style value: bit-identical to the style render (all 8, 2 photos) |
| Overall 50 | Halfway between off and the style (night: 0.31 vs 0.58 mean difference) |
| Styles | Natural, Professional and Advanced give different renders (pairwise 0.16–0.40 mean difference) and graded measures (night dark-mark depth 1.49 → 1.04 / 0.82 / 0.62) |

Mean difference = mean absolute 8-bit difference over the whole frame; the finish only touches skin.

## Ryzen 3 7330U benchmark (Sam's PC, Windows 11, 16 GB, CPU only)

Professional Skin Finish alone (no Skin Perfection), full resolution, each run in its own process.
Photoshop and a browser were open (1.9 GB free at the start), so these include paging.

| Photo | Finish off (export) | Skin Perfection scan alone | Finish preview | Finish export | Peak memory (finish) |
|---|---|---|---|---|---|
| CRPixels-26, 21.7 MP | 9.8 s | 68 s | 66–73 s | 69–75 s | 6.3 GB |
| CRPixels-5, 25.2 MP | 12.7 s | 95 s | 76–93 s | 77–93 s | 7.3 GB |
| Canon EOS R CR3, 30.3 MP | 16.1 s | 82 s | 115 s | 123 s | 8.2 GB |

With Skin Perfection on as well, a 21 MP photo takes about two minutes. During the finished
renders Windows reached 90–93% of its commit limit with heavy paging (peaks of 72k–108k pages/s),
but nothing failed.

Slowest stages (21.7 MP, Ryzen): dark spots 19–24 s, the rest of the local corrections ~20 s
(contour weights, measures, shine, under-eye, dodge and burn), texture finish ~15 s. Most of the
time is Gaussian blurs and resizes on full-resolution planes.

## Optimisation done (output identical)

- `d274157`: finish planes are released as soon as they are used, and the texture measure keeps
  two low-passes at a time. Every pixel and report value is identical (2 photos × 3 styles,
  full resolution). Traced allocation peak 5.5 → 4.8 GB and 7.9 → 7.0 GB. **The process peak on
  the Ryzen did not change (6.31 GB)**: the peak now sits in the dark-spot pass, which is next.
- `c56b871`: the QA memory probe returned nothing on Windows (ctypes types missing). Fixed; it now
  agrees with Windows' own reading within 3 MB.

## Visual inspection (100% and normal size)

- **Face, deep skin (CRPixels-26 night, CRPixels-15, CRPixels-5):** pores, stubble, moles and the
  nostril / lip contours are kept. No grey, orange or blue patches, halos or repeated texture
  were found. Dark marks are lightened, not removed. Raised light bumps (whiteheads) stay, as
  designed: they are Skin Perfection's job.
- **Under-eye:** Natural and Professional keep the lower-lid creases. Advanced looks slightly flat
  on the night photo.
- **Jaw and under-chin shadow (CRPixels-5):** lifted at Advanced. This is the same cause as defect 1.
- **Body skin:** defect 1.
- **Skin tone:** on the face it is the same hue, with lightness up by 1–3 L*.

## Not done yet

- **Broader complexions.** No consented photos of light or medium skin are available. Nine openly
  licensed Wikimedia Commons photos were selected (CC BY-SA 3.0/4.0 and US-government public
  domain): light skin with acne, post-acne marks and hyperpigmentation on the face, back and chest,
  plus a deep-skinned woman, a medium-deep man and a medium East Asian face. Wikimedia
  rate-limits the cloud, so they need to be downloaded on Sam's PC, which needs Sam's OK.
  retouchedElle was not used.
- **Shine on lighter skin and strong highlights.** Only measured on deep skin so far (p99 L* down by
  1–3).

## Proposed fixes (need approval before implementation)

1. Body skin: keep the any-shape rule to the face (inside the landmark face oval). On body skin,
   use only the round-patch test, and cap the lift by the local shading scale.
2. Auto Retouch: never change global exposure. Any lift becomes skin-only, capped as now.
3. Glasses glare: run only when glasses are detected. As a smaller step, skip highlights whose
   colour differs from the estimated reflection colour. Default to off when no glasses are found.
4. Memory: free the dark-spot planes per scale. This must stay output-identical.

## Tests at d274157

Python 371 passed, 2 skipped; shared 98 passed; web 124 passed.
