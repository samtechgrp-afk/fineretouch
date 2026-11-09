"""Skin noise reduction (v0.9): sensor grain and colour noise on skin, pores kept.

Noise is pixel-scale whatever the photo size, so this tool works in pixels, not shares of the
long edge:

* Luma grain: the finest band (image minus a NOISE_SIGMA_PX blur) is soft-thresholded at
  NOISE_THRESHOLD x the photo's measured noise level, so weak, random grain goes and stronger
  structure (pores, fine hair, the edges of features) keeps all but the threshold's worth of
  contrast.
* The noise level is measured where pores cannot inflate it: sensor noise is independent per
  colour channel, so it shows up in the chroma fine bands, while pores and wrinkles are almost
  pure luma. The luma noise level is CHROMA_TO_LUMA x the robust MAD of the Cb / Cr fine bands
  on skin (capped by the luma band's own MAD). A clean or chroma-denoised file (most JPEGs)
  measures a low level and is barely touched; pure-luma grain is under-estimated, which errs
  toward keeping texture.
* Colour noise: chroma (Cb, Cr) is averaged over CHROMA_SIGMA_PX on skin. Skin colour changes
  slowly, so this removes blotchy colour speckle without softening any detail (luma untouched).

Mirrored in apps/web/src/engine/shaders.ts (uNoise), which uses a 3x3 neighbourhood.
"""

from __future__ import annotations

import numpy as np

from .common import _cbcr, _from_ycbcr, _gauss, _luma, _smoothstep

NOISE_SIGMA_PX = 0.9
CHROMA_SIGMA_PX = 2.0
NOISE_THRESHOLD = 2.5
# Floor of the measured noise level (0..1 luma), so even clean files get a gentle effect.
NOISE_FLOOR = 0.002
# Luma / chroma noise ratio for independent per-channel noise: |Y row| / mean(|Cb row|, |Cr row|).
CHROMA_TO_LUMA = 0.6686 / 0.64


def estimate_noise(fine_y: np.ndarray, fine_cb: np.ndarray, fine_cr: np.ndarray, skin: np.ndarray) -> float:
    """Luma noise level (0..1) from the chroma fine bands on skin, capped by the luma band's MAD."""
    sel = skin > 0.5
    if sel.sum() < 64:
        sel = np.ones_like(skin, bool)
    mad = lambda v: 1.4826 * float(np.median(np.abs(v[sel])))  # noqa: E731
    from_chroma = CHROMA_TO_LUMA * 0.5 * (mad(fine_cb) + mad(fine_cr))
    return max(NOISE_FLOOR, min(from_chroma, mad(fine_y)))


def reduce_skin_noise(image: np.ndarray, skin: np.ndarray, amount: float) -> np.ndarray:
    """image: HxWx3 float32 sRGB; skin: HxW weight; amount 0..100 (the effective strength)."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not (skin > 1e-3).any():
        return image
    y = _luma(image)
    cb, cr = _cbcr(image)
    fine = y - _gauss(y, NOISE_SIGMA_PX)
    cb_s = _gauss(cb, CHROMA_SIGMA_PX)
    cr_s = _gauss(cr, CHROMA_SIGMA_PX)
    level = estimate_noise(fine, cb - _gauss(cb, NOISE_SIGMA_PX), cr - _gauss(cr, NOISE_SIGMA_PX), skin)
    thr = NOISE_THRESHOLD * level
    k = t * skin
    grain = np.clip(fine, -thr, thr)
    y = y - grain * k
    # Chroma: averaged on skin only; strong colour edges (lip line, eye rim) keep theirs.
    edge = _smoothstep(0.02, 0.05, np.abs(cb - cb_s) + np.abs(cr - cr_s))
    kc = k * (1.0 - edge)
    cb = cb + (cb_s - cb) * kc
    cr = cr + (cr_s - cr) * kc
    return np.clip(_from_ycbcr(y, cb, cr), 0.0, 1.0).astype(np.float32)
