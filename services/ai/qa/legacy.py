"""The pre-v0.13 acne / dark spot / blemish tools, for before-and-after comparisons."""

from __future__ import annotations

from app.engines.skin import natural_skin_retouch, remove_acne, remove_dark_spots
from tests.perfection_fixtures import oval_face, oval_mask


def run_old(img, F):
    h, w = img.shape[:2]
    m, f = oval_mask(w, h), [oval_face(w, h)]
    c = remove_acne(img, m, f, 100, 1)
    c = remove_dark_spots(c, m, f, 100, 1, sigma_tone=0.012 * F, sigma_base=0.036 * F)
    return natural_skin_retouch(c, m, f, smoothing=0, texture_retention=100, spot_amount=100, sigma_tone=0.012 * F, sigma_base=0.036 * F)
