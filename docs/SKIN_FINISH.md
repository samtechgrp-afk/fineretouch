# Professional Skin Finish (v0.14, phases 1 and 2)

The finish is a separate engine (`services/ai/app/engines/finish.py`, render stage 0d). It runs after
Skin Perfection and the healing strokes, on face skin and parsed body skin, and before every older
skin stage. It is off by default. While it is off, or on at Overall strength 0, the renderer never
calls it.

Skin Perfection removes discrete marks. The finish does a retoucher's other job: it calms the relief
of the skin (bumps a few pores wide, rough patches, blotches), keeps believable pores and grain, and
leaves the face's contours alone.

## Phase 1 controls

| Setting (`adjustments.skinFinish`) | Default | What it changes |
|---|---|---|
| `enabled` | false | Turns the finish on |
| `strength` (Overall strength) | 100 | Blends the whole finish |
| `smoothing.amount` (Skin smoothing) | 55 | Cuts the bump band (up to 100%) and blotch band (up to 60%) |
| `porePreservation.amount` | 65 | Keeps the pore band, and moves the bump cut away from the pore scale |
| `texturePreservation.amount` | 75 | Keeps the finest grain, and refills grain where skin ended up flatter than the person's own |
| `bodyCoverage` | true | Includes parsed body skin |

Each control also has `enabled`. Phase 2 (below) adds five local corrections, the three styles and
the panel.

## How it works

1. **Finish mask.** The renderer's skin weight (face skin without eyes, brows and lips, the
   person's own skin colour, Add skin / Protect strokes) plus body skin. Dark cores (nostrils, lash
   shadows, moles) more than 9 L* below the skin around them are protected. Cores are compared at
   the pore scale, so the grain of a noisy night photo does not read as cores.
2. **Bands.** CIELAB L* is split into fine (< 0.0016 F), pore (to 0.0045 F), bump (to 0.014 F) and
   blotch (to 0.045 F) bands, where F is the face width. A close-up and a full-length shot get the
   same treatment, and deep and light skin are treated the same since L* is perceptual.
3. **Edge weight per band.** The gradient at each band's own scale is compared with the person's
   typical skin gradient there. A real contour is 3× steeper than typical relief and at least
   4 L*. Near it the band's smoothing fades out, so nostrils, eyelids, lips, the jaw and shadow
   edges keep their shape.
4. **Pore gap.** Gaussian bands overlap. Cutting the whole bump band also took about a quarter of
   the pores with it, so Pore preservation 100 still lost pores. Pore preservation now also moves
   the lower edge of the bump cut from 1× to 2× the pore scale. At Pore preservation 100 and Skin
   smoothing 80, pores keep over 90% of their relief.
5. **Colour.** a* and b* relief at the pore and bump scales moves with the luminance (a red bump is
   one bump). The base colour is never touched, so complexion holds.
6. **Texture restoration.** Where fine and pore energy falls below 0.85 × Texture preservation × the
   person's typical level, band-limited grain is added back. It is seeded from the image size, so the
   same photo always gets the same grain.
7. **Verification.** Each run reports the bands before and after, feature-edge contrast kept,
   texture kept and complexion shift. Pixels outside the mask are copied bit for bit.

Only NumPy and OpenCV on CPU. No model and no GPU.

## Method choice

Four ways to build the bands were compared with `python -m qa.finish_eval`:

- **edge**: Gaussian bands plus the edge weight. This is the default.
- **gauss**: Gaussian bands only.
- **guided**: self-guided filter bands.
- **domain**: domain-transform bands.

The test set had 6 synthetic tones (I–VI) and 10 real photos, each with a professional retouch of
the same frame:

- Sam's night portrait CRPixels-26, with his Photoshop retouch.
- 9 daylight frames, with Capture One retouches.

The real photos were measured after default Skin Perfection, since the finish runs after it.

**Advanced-like settings** (smoothing 80, texture 60, pores 45). Each cell gives min / mean / max.

| | edge | gauss | guided | domain |
|---|---|---|---|---|
| Contour contrast kept, real | **0.94 / 0.98 / 1.00** | 0.82 / 0.85 / 0.88 | 0.89 / 0.92 / 0.94 | 0.87 / 0.91 / 0.94 |
| Contour contrast kept, synthetic | **0.99 / 0.99 / 0.99** | 0.73 / 0.85 / 0.95 | 0.82 / 0.88 / 0.95 | 0.91 / 0.94 / 0.97 |
| Halo at contours (L*), real | **0.15 / 0.65 / 1.32** | 2.05 / 2.33 / 2.83 | 1.05 / 1.38 / 1.86 | 0.90 / 1.58 / 2.51 |
| Bump band left, real | 0.56 / 0.65 / 0.76 | 0.56 / 0.61 / 0.70 | 0.58 / 0.66 / 0.78 | 0.47 / 0.66 / 0.83 |
| Pore band left, real | 0.76 / 0.80 / 0.83 | 0.74 / 0.77 / 0.80 | 0.70 / 0.77 / 0.80 | 0.55 / 0.68 / 0.74 |
| Complexion shift (L*), real | −0.81 / −0.46 / −0.14 | −1.41 / −1.12 / −0.93 | −0.57 / −0.49 / −0.44 | −0.40 / −0.31 / −0.27 |
| Distance to the professional retouch\* | 0.16 / 0.30 / 0.66 | 0.15 / 0.29 / 0.66 | 0.14 / 0.29 / 0.63 | 0.19 / 0.33 / 0.49 |

\* Mean |log2(after / reference)| over the pore, bump and blotch bands. The original photos score
0.16 / 0.61 / 1.10, so every method about halves the distance.

**Professional-like settings** (55 / 75 / 65), edge vs gauss on real photos:

| | edge | gauss |
|---|---|---|
| Contour kept | 0.966 / 0.987 / 0.997 | 0.893 / 0.913 / 0.928 |
| Halo (L*) | 0.10 / 0.44 / 0.87 | 1.26 / 1.46 / 1.80 |
| Bump left | 0.76 / 0.80 / 0.85 | 0.75 / 0.78 / 0.82 |
| Pore left | 0.89 / 0.91 / 0.94 | 0.89 / 0.90 / 0.94 |

**Result.** The edge method smooths as much as plain Gaussian bands, and keeps the most contour
contrast on every photo. It has the smallest halo and less complexion shift than gauss. Guided and
domain filters keep contours worse and are no faster. Edge is the default; the others stay only for
this comparison.

## Phase 2: local corrections

`services/ai/app/engines/finish_local.py` runs five corrections on the Lab crop *before* the phase 1
texture finish, so the finish sees corrected skin. Each has `enabled` and `amount` and is scaled by
Overall strength. They default to 0 in the engine, so a finish saved with only phase 1 controls
renders exactly as before (tested); the app's styles set all of them.

| Setting | What it changes | What it leaves |
|---|---|---|
| `darkSpots` | Dark spots, post-acne marks, small hyperpigmentation: found at 4 scales (0.006–0.04 F), lifted by their real depth toward the surrounding skin, colour moved with them, two passes | Pores (finer than 0.003 F), creases (shape and gradient-coherence tests), deep cores (moles, nostrils, lash shadows: 14–20 L* below, or 30–45% of the skin's L* on deep skin) and their neighbourhood, the nose and nasolabial folds |
| `toneEvenness` | Blotchy colour: the a*/b* band 0.014–0.10 F reduced; L* only lifted, and only where the colour also deviates | The key light's gradient, skin tone (complexion shift < 0.5 on a*, b*) |
| `shine` | Sheen more than 3 L* above the skin's own base, weighted by how much it washed the colour out (glare); the colour comes back | Skin that is simply better lit gets only 35% of the reduction; highlights keep their place and shape |
| `underEye` | Under-eye shadow lifted toward a reference from the person's own surrounding skin; crepey lines softened | Pores, texture, part of the socket shadow (never lifted past the cheek) |
| `dodgeBurn` | Small light and dark irregularities (0.012–0.05 F) evened, darks more than lights, on edge-preserving low-passes | Steps larger than 20% of the skin's L* (contours), the face's light shape |

**Landmark folds.** A dark, elongated patch on a cheek and the shadow of a nasolabial fold look
alike to a filter (on CRPixels-26 both are 10–16 L* deep on skin of L* 34). The pipeline therefore
draws a fold mask from the face landmarks (`fold_mask`: lower nose, its side walls up to the eyes'
inner corners, and a band from each ala past the mouth corner). Away from it, patches of 0.024 F
and larger may be any shape; inside it, and when there are no landmarks, they must be round. Tone
evenness and dodge and burn also stay out of it.

**Styles** (`FINISH_STYLES` in `schemas.py`, mirrored in `packages/shared/src/finish.ts`):

| | Smooth | Texture | Pores | Shine | Dark spots | Tone | Under-eye | D&B |
|---|---|---|---|---|---|---|---|---|
| Natural | 35 | 85 | 80 | 25 | 45 | 25 | 25 | 15 |
| Professional | 55 | 75 | 65 | 45 | 70 | 45 | 45 | 30 |
| Advanced | 80 | 60 | 45 | 60 | 95 | 65 | 60 | 45 |

**Panel and preview.** The Professional Skin Finish panel has the switch, style, Overall strength,
and for each effect its own switch, slider and Reset (to the style's amount); body skin; Before /
After / Split; the finish-area view and the Add / Protect brush (the shared skin mask); and what the
engine measured on the photo. The preview's source is the engine's result, as for Skin Perfection
(`POST /v1/perfection` now applies the finish too and returns its report), and export renders photos
with the finish on in the engine. With the finish on, heal strokes run after it in the render, the
order the preview uses; with it off nothing changed.

**Measured on CRPixels-26 (night, deep skin), full resolution:**

| Style | Dark-mark depth (mean, L*) | Colour variation | Contours kept | Texture kept |
|---|---|---|---|---|
| Natural | 1.33 → 0.89 | 7.39 → 6.94 | 0.97 | 0.98 |
| Professional | 1.33 → 0.69 | 7.32 → 6.53 | 0.95 | 0.93 |
| Advanced | 1.33 → 0.51 | 7.27 → 6.17 | 0.93 | 0.83 |

## Limitations

- Sam's Photoshop retouch is still much smoother: it removes nearly all relief and every mark, and
  makes the face warmer and darker. Advanced reduces the marks clearly but leaves some, and keeps
  more texture. That is the brief (no plastic skin), but it is not yet Photoshop-level evenness.
- Raised bumps (the forehead) are Skin Perfection's job; the finish lifts dark marks only.
- A mark with a hard, sharp outline (the synthetic test discs) can keep a faint outline after its
  inside is lifted. Real marks have soft edges, where this was not visible.
- The fold mask needs nose and lip landmarks. Without a face (body skin), elongated marks are not
  taken for spots.
- All real test subjects so far have deep skin. Lighter tones are covered by synthetic tones I–VI.
- **retouchedElle** (the Capture One retouches of the daylight set, found on Sam's PC): author and
  usage rights are unverified, so its use is **paused**. Phase 1's method comparison used it before
  the pause; phase 2 uses only Sam's originals, Sam's own Photoshop retouch of CRPixels-26 and
  synthetic skin.

## Cost

Measured in the cloud container on a single run with measures on:

| Photo | Finish stage | Peak extra memory |
|---|---|---|
| Night portrait, 21 MP | 12–14 s | 3.9 GB |
| Daylight full-length, 30 MP | 15–20 s | 5.6 GB |

Phase 2 (Professional style, all five corrections on, measures on), same container:

| Photo | Finish stage | Peak extra memory |
|---|---|---|
| Night portrait, 21 MP | 41 s | 4.3 GB |
| Daylight portrait, 22 MP | 48 s | 4.4 GB |

Most of the added time is the dark spot detector (four scales, two passes, measured before and
after). The before measure reuses the first pass, which gave the identical image. The cloud
container is not Sam's Ryzen 3 7330U; expect a similar order of time there, and the extra memory
fits in 16 GB with the rest of the render.

The work runs on the skin's bounding box. With body coverage, that box is most of the frame. Tiling
the box to the mask would cut both numbers without changing the output.

## Checks

- `tests/test_skin_finish.py` (20 tests) checks:
  - off is off, and outside pixels are bit-identical
  - each control changes its band, and Overall strength scales the change
  - contours and complexion hold on tones I–VI, and gauss is measurably worse on the same fixture
  - dark cores are protected, and darker skin is not lightened
  - restoration refills a flattened patch deterministically
- `python -m qa.golden check <baseline> --manifest <real.json>` re-renders 24 synthetic cases and
  3 real exports and compares SHA-256 hashes with the PR #13 head. All 27 were identical after
  phase 1. After phase 2 only the cases with Dark circles on changed (the six synthetic "heavy"
  cases and the two real "snapshot13" exports, CRPixels-26 and CRPixels-5, all with darkCircles
  88): that is the approved replacement of the stage 6 fill. The other 19 cases are identical.
- `tests/test_skin_finish_local.py` (26 tests) checks each phase 2 correction on the pixels it is
  about: spots lifted on tones I, III, V, VI while a crease, a mole and pores stay; glare lowered
  more than better-lit skin and its colour restored; colour blotches reduced without moving the
  light; the under-eye lifted with texture kept; dodge and burn evening light while contours stay;
  every switch and amount changing the output and amount 0 being identity; phase 1 settings giving
  the identical image; an elongated patch lifted away from the folds and kept inside them; the
  fold mask from landmarks; the styles matching the TypeScript panel; the preview endpoint; and
  heal strokes following the finish.
- `tests/test_dark_circles.py` (10 tests) and Auto Retouch tests in `packages/shared` cover the two
  approved fixes that came with phase 2.
