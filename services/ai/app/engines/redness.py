"""Redness reduction (v0.9): diffuse redness, flushing and rosacea on the skin.

Acne removal treats red *spots*; this tool treats red *areas*. Each skin pixel's tone layer
(Gaussian sigma_tone, so pores are not involved) is compared with the person's own skin colour
around it: a skin-weighted average over REDNESS_REACH face widths that leaves the red areas out. Where the tone layer is
redder (higher Cr) than that reference, the excess Cr is pulled back, with a smaller pull on
Cb so the hue lands on the surrounding skin tone. Luma is never changed, so shading, contour
and pore texture stay; the skin-tone guard runs afterwards and stops any drift to grey.

Mirrored in apps/web/src/engine/shaders.ts (uRedness), which uses the coarsest preview band as
the reference.
"""

from __future__ import annotations

import numpy as np

from .common import _cbcr, _from_ycbcr, _gauss, _luma, _masked_blur, _smoothstep

# Reference skin colour: skin-weighted average within this share of the face width.
REDNESS_REACH = 0.2
# Excess Cr (0..1 YCbCr units) where the correction starts and where it is full.
REDNESS_RAMP = (0.004, 0.025)
# Share of the excess Cr removed at 100%, and how far Cb follows the reference.
REDNESS_CR_PULL = 0.9
REDNESS_CB_PULL = 0.4


def reduce_redness(
    image: np.ndarray, skin: np.ndarray, amount: float, *, sigma_tone: float, face_width: float,
) -> np.ndarray:
    """image: HxWx3 float32 sRGB; skin: HxW weight; amount 0..100 (the effective strength)."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not (skin > 1e-3).any():
        return image
    tone = _gauss(image, sigma_tone)
    core = (skin > 0.5).astype(np.float32)
    if core.sum() < 16:
        core = skin.astype(np.float32)
    reach = max(2.0, REDNESS_REACH * face_width)
    cb_t, cr_t = _cbcr(tone)
    # Two passes: the red areas themselves are left out of the second reference, so a large
    # flushed patch does not raise the "normal" skin colour it is compared with.
    weight = core
    ref = cb_r = cr_r = excess = None
    for _ in range(2):
        ref = cb_r = cr_r = excess = None  # the first pass's arrays, freed before the second is built
        ref = _masked_blur(tone, weight, reach)
        cb_r, cr_r = _cbcr(ref)
        excess = cr_t - cr_r
        weight = core * (1.0 - _smoothstep(REDNESS_RAMP[0], REDNESS_RAMP[1], excess))
    del tone, ref, weight, core, cr_t, cr_r  # full-size temporaries, freed before the output is built
    red = _smoothstep(REDNESS_RAMP[0], REDNESS_RAMP[1], excess)
    k = t * red * skin
    del red
    y = _luma(image)
    cb, cr = _cbcr(image)
    cr = cr - REDNESS_CR_PULL * np.maximum(excess, 0.0) * k
    del excess
    cb = cb + REDNESS_CB_PULL * (cb_r - cb_t) * k
    del cb_r, cb_t, k
    out = _from_ycbcr(y, cb, cr)
    del y, cb, cr
    return np.clip(out, 0.0, 1.0, out=out).astype(np.float32, copy=False)
