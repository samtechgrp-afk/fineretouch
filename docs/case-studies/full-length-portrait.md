# Case study: full-length outdoor portrait (v0.5.1)

A real 4480 x 6720 full-length portrait (photographer's watermark at left) run through the
server renderer to remove acne, comedones, blemishes and dark spots while keeping skin texture
and tone. The photo itself is not committed; before/after images live in the project files.

Settings: `blemishRemoval 100, skinMultiplier 2, skinSmoothing 50, skinTexture 85,
protectBeautyMarks false, skinToneGuard true`.

## Where v0.5 fell short on this photo

| Problem | Cause | Fix |
|---|---|---|
| No face found, so nothing was retouched | Detection ran on a 1280 px proxy where the face is ~85 px | Tiled full-resolution second pass, then the mesh is re-run on a crop around every small face (`landmarks.py`) |
| Face looked lighter and flat, a smudge beside the nose | 2x/3x boost wrote a guided-filtered tone layer back over the whole face; a nose-side shading hollow was "healed" | Boost now applies only its repairs as a difference; soft-edged dark patches and one-sided edges are rejected (`_ring_isotropy`, `SOFT_SHADOW_*`) |
| Smoothing radii were wrong for small faces | Skin blur radii scaled from the long edge (s2 was 0.18 x face width here) | Radii capped at 0.012 / 0.036 x face width (`TONE_SIGMA_FACE`, `BASE_SIGMA_FACE`) |
| Forehead comedones untouched | They read lighter than the skin (raised domes), and the landmark oval stops at mid-forehead | Light-bump channel (`_detect_bumps`) and a forehead extension to the hairline gated by the person's own skin colour (`face_skin_likelihood`) |
| Braid knots flagged as spots | The YCbCr skin box accepts copper braids | Own-skin colour gate on the mask and a local hue/chroma "strand" test |
| Small dark papule by the mouth missed | Adaptive LoG threshold is high on glossy, textured skin | Strong-contrast path for clearly dark marks |
| Healed clusters looked smooth | Dense spots left no clean donor patch, so texture fell back to flat grain | Texture cloned in tiles with looser donor fallbacks |
| 129 s per render | Skin stages ran on all 30 MP | Skin stages run on the skin's bounding box: 47 s, almost all of it global stages |

Only face pixels change (rows 1045-1716, columns 2061-2519 of the frame); the watermark and
everything else are bit-identical to the original before JPEG encoding.

## Known limits

- Some of the smallest forehead comedones remain at 2x; 3x clears more but starts to soften the
  glossy highlight, so 2x is the recommended setting on oily, deeper skin.
- The WebGL preview still uses long-edge radii and no bump channel, so on full-length shots the
  export is cleaner than the preview.
- Neck, chest and arms are outside the face mask and are not retouched.
