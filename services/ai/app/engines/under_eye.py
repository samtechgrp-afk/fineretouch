"""Under-eye texture (v0.9): fine lines and crepey skin below the eyes; Dark circles (v0.14).

Under-eye correction (darkCircles) lifts the colour and brightness of the under-eye area; this
tool works on its texture. Inside the landmark under-eye region, the line band of luma (between
the fine pore scale sigma_fine and the crease scale sigma_line) is reduced: dark creases by up
to LINE_DARK_PULL, lighter ridges by LINE_LIGHT_PULL, so the lines fade rather than flatten
into a smooth patch.
Pores are finer than sigma_fine (in the renderer: the S0 band, capped at 0.3 x sigma_line), so
they sit outside the band and keep their contrast. The finest band (pores) and the tone layer (shape, shading) are not
touched, so the area keeps real skin texture and the eye's own contour.

Mirrored in apps/web/src/engine/shaders.ts (uUnderEyeTex) on the low0 - low1 band.
"""

from __future__ import annotations

import numpy as np

from .common import _cbcr, _from_ycbcr, _gauss, _luma, _smoothstep

LINE_DARK_PULL = 1.0
LINE_LIGHT_PULL = 0.35


def soften_under_eye_texture(
    image: np.ndarray, region: np.ndarray, amount: float, *, sigma_fine: float, sigma_line: float,
) -> np.ndarray:
    """image: HxWx3 float32 sRGB; region: HxW under-eye weight; amount 0..100."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not (region > 1e-3).any():
        return image
    # Work on the region's bounding box (two small patches in a portrait).
    ys, xs = np.nonzero(region > 1e-3)
    pad = int(np.ceil(3.0 * sigma_line)) + 1
    h, w = region.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = image[y0:y1, x0:x1]
    band = _luma(_gauss(crop, sigma_fine)) - _luma(_gauss(crop, sigma_line))
    k = t * region[y0:y1, x0:x1]
    change = k * (LINE_DARK_PULL * np.minimum(band, 0.0) + LINE_LIGHT_PULL * np.maximum(band, 0.0))
    out = image.copy()
    out[y0:y1, x0:x1] = np.clip(crop - change[..., None], 0.0, 1.0)
    return out.astype(np.float32)


# Dark circles (v0.14). The old fill raised every under-eye pixel darker than a wide blur of the
# photo up to it (y += max(0, blur - y)), which flattened pores and lines into a patch, and pulled
# the chroma toward that blur, which takes in the eye, lashes and background: the grey patch.
# Now the shadow is measured on its own scale and lifted as a whole, so texture rides on top
# unchanged, toward the person's own surrounding skin (a blur over face skin only, the under-eye
# area itself left out):
#   lift = max(0, skin_ref - shadow) * DARK_CIRCLE_LIFT * amount * under_eye
# DARK_CIRCLE_LIFT < 1 keeps part of the natural shadow (the eye socket stays a socket).
# Mirrored in apps/web/src/engine/shaders.ts (uDark, uUnderRef).
DARK_CIRCLE_LIFT = 0.85
# Reference blur, as a share of the face width: reaches across the under-eye area to the cheek.
UNDER_REF_SIGMA = 0.06
DARK_CIRCLE_CHROMA = 0.6
# Under-eye weight above which a pixel does not count as surrounding skin.
UNDER_REF_LO, UNDER_REF_HI = 0.05, 0.3


def under_eye_reference_weight(skin: np.ndarray, under_eye: np.ndarray) -> np.ndarray:
    """Face skin minus the under-eye area (its blurred tail widens the cut): where the reference
    colour for the dark-circle lift is taken from."""
    return (skin * (1.0 - _smoothstep(UNDER_REF_LO, UNDER_REF_HI, under_eye))).astype(np.float32)


def correct_dark_circles(
    image: np.ndarray, src: np.ndarray, shadow: np.ndarray, skin: np.ndarray, region: np.ndarray, amount: float,
    *, sigma_ref: float,
) -> np.ndarray:
    """image: HxWx3 float32 (current render); src: the photo the bands come from; shadow: its blur
    at the shadow scale (renderer band S1); skin: face skin weight; region: under-eye weight;
    amount 0..100. Works on the under-eye bounding box."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not (region > 1e-3).any():
        return image
    ys, xs = np.nonzero(region > 1e-3)
    pad = int(np.ceil(3.0 * sigma_ref)) + 1
    h, w = region.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    wgt = under_eye_reference_weight(skin[y0:y1, x0:x1], region[y0:y1, x0:x1])
    s = np.ascontiguousarray(src[y0:y1, x0:x1], np.float32)
    num = _gauss(s * wgt[..., None], sigma_ref)
    den = _gauss(wgt, sigma_ref)
    ok = den > 0.02
    ref = num / np.maximum(den, 0.02)[..., None]
    sh = shadow[y0:y1, x0:x1]
    reg = region[y0:y1, x0:x1]
    d = t * reg * ok * (reg > 1e-3)
    crop = image[y0:y1, x0:x1]
    y = _luma(crop)
    cb, cr = _cbcr(crop)
    cb_r, cr_r = _cbcr(ref)
    cb_s, cr_s = _cbcr(sh)
    y = y + np.maximum(0.0, _luma(ref) - _luma(sh)) * DARK_CIRCLE_LIFT * d
    cb = cb + (cb_r - cb_s) * DARK_CIRCLE_CHROMA * d
    cr = cr + (cr_r - cr_s) * DARK_CIRCLE_CHROMA * d
    out = image.copy()
    out[y0:y1, x0:x1] = np.where((d > 0)[..., None], _from_ycbcr(y, cb, cr), crop)
    return out.astype(np.float32)
