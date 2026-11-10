# Professional Skin Finish (v0.14, phase 1)

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

Each control also has `enabled`. Shine, tone evenness, dark spot and under-eye refinement, dodge
and burn, the three styles, the panel, mask view and brush come in phase 2.

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

## What phase 1 does not do yet

- Sam's Photoshop retouch of CRPixels-26 removes the bump band almost completely (−81% on the face)
  and the dark post-acne marks with it, and adds fine grain (+59%). Phase 1 at Advanced-like
  settings removes about 44% of the bumps and leaves the dark marks: their edges are steep enough
  that the edge weight protects them. Removing them is the job of phase 2's dark spot refinement and
  tone evenness, not of a stronger blur.
- All 10 real photos show deep skin. Lighter tones are covered only by the synthetic tones I–IV so
  far.

## Cost

Measured in the cloud container on a single run with measures on:

| Photo | Finish stage | Peak extra memory |
|---|---|---|
| Night portrait, 21 MP | 12–14 s | 3.9 GB |
| Daylight full-length, 30 MP | 15–20 s | 5.6 GB |

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
  3 real exports and compares SHA-256 hashes with the PR #13 head. All 27 are identical.
