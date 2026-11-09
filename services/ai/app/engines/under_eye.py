"""Under-eye texture (v0.9): fine lines and crepey skin below the eyes.

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

from .common import _gauss, _luma

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
