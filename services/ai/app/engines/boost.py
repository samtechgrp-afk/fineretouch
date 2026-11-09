"""2x/3x boosted skin retouch: recursive guided filtering of the colour layer and patch repair of the detail layer.

Part of the retouch engine split (v0.9); app/retouch_engine.py re-exports everything.
"""


import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..color import color_stats, lab_to_rgb, linear_to_srgb, rgb_to_lab, srgb_to_linear
from ..depth import guided_filter
from ..schemas import (
    Adjustments,
    ColorMatch,
    FaceAnalysis,
    StyleProfile,
    StyleSyncResponse,
    StyleSyncTarget,
)

from .common import *  # noqa: F403

# ---------------------------------------------------------------------------
# 1. Boosted skin retouch
# ---------------------------------------------------------------------------


def apply_boosted_skin_retouch(image: np.ndarray, mask: np.ndarray, multiplier: float = 2.0) -> np.ndarray:
    """Multi-stage frequency separation for stubborn skin problems.

    image: HxWx3 float32 sRGB; mask: HxW float32 skin weight in [0, 1].
    multiplier 1.0 returns the image unchanged; 2.0 and 3.0 run 2 and 3 recursive passes.

    Low-frequency layer (colour and tone): recursive guided filtering with the original
    luma as guide (edges and facial contours survive), then dark spots and redness are
    found with an adaptive z-score against a skin-only base and pulled toward it.
    High-frequency layer (texture): blemishes are outliers relative to the local texture
    energy; they are opened (pores are smaller than the kernel), inpainted, and given
    back real grain from a nearby skin patch, so repaired spots never look airbrushed.
    """
    extra = float(np.clip(multiplier - 1.0, 0.0, 2.0))
    if extra <= 0 or mask.max() <= 1e-3:
        return image
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    # Every radius is relative to the size of the skin area (~face width), not the frame,
    # so a head-and-shoulders crop and a full-length shot get the same treatment.
    F = float(np.clip(np.sqrt(float((mask > 0.5).sum())) * 1.1, 32.0, max(image.shape[:2])))
    guide = _luma(image)

    # --- split ---------------------------------------------------------------
    # Classic Gaussian frequency separation puts spots and pores in the detail layer;
    # the colour layer is then smoothed recursively with an edge-preserving guided filter
    # (guide = original luma), so blotches melt while jawline, nose and lips stay put.
    low = _gauss(image, max(1.0, 0.02 * F))
    high = image - low
    passes = 1 + math.ceil(extra)
    r0 = max(2, round(0.015 * F))
    eps = 0.015**2
    sub = max(1, r0 // 4)
    for i in range(passes):
        low = guided_filter(guide, low, r0 * (i + 1), eps * (1 + i), subsample=sub)

    # --- low frequency: hyperpigmentation, redness, uneven tone ----------------
    base = _masked_blur(low, mask, 0.08 * F)
    y_low, y_base = _luma(low), _luma(base)
    cb_low, cr_low = _cbcr(low)
    cb_base, cr_base = _cbcr(base)
    sigma_ctx = 0.12 * F

    def zscore(d: np.ndarray) -> np.ndarray:
        spread = np.sqrt(_masked_blur(d * d, mask, sigma_ctx))
        return d / (spread + 4e-3)

    d_dark = y_base - y_low
    d_red = cr_low - cr_base
    t0 = 1.4 - 0.35 * extra
    dark = _smoothstep(t0, t0 + 1.2, zscore(d_dark)) * (d_dark > 0.006)
    red = _smoothstep(t0, t0 + 1.2, zscore(d_red)) * (d_red > 0.004)
    # Pigment spots are compact; natural shading (eye sockets, folds, jaw shadow) is large
    # or elongated. Keep only compact blobs so contouring survives and nothing goes flat.
    dark = dark * _compact_components(dark > 0.3, max_side=0.12 * F, min_fill=0.3)
    strength = min(1.0, 0.55 + 0.25 * extra)
    lift = np.minimum(np.maximum(d_dark, 0.0), 0.08)
    y_new = y_low + lift * dark * strength
    spot = np.maximum(dark, red) * strength
    even = 0.12 * extra  # gentle global tone evening on top of spot repair
    t_chroma = np.clip(spot + even, 0.0, 1.0)
    low_fixed = _from_ycbcr(
        y_new,
        cb_low + (cb_base - cb_low) * t_chroma,
        cr_low + (cr_base - cr_low) * t_chroma,
    ).astype(np.float32)

    # --- high frequency: acne / blemish patches --------------------------------
    hy = _luma(high)
    energy = np.sqrt(_gauss(hy * hy, 0.05 * F)) + 1e-4
    k = 2.6 - 0.35 * extra
    # Blemishes are darker than their surroundings in the detail layer; strong bright/dark
    # edges (nostrils, folds, lash lines) are long or sit on contour edges and are rejected below.
    core = mask > 0.85
    cand = ((hy < -k * energy) & core).astype(np.uint8)
    pore = round(0.006 * F) | 1
    if pore >= 3:
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _ellipse(pore))  # drop pore-sized specks
    blem = (_compact_components(cand > 0, max_side=0.06 * F, min_fill=0.35) > 0).astype(np.uint8)
    blem = cv2.dilate(blem, _ellipse(max(3, round(0.008 * F) | 1)))

    high_fixed = high
    if blem.any():
        radius = max(2, round(0.015 * F))
        # Inpaint as 16-bit: OpenCV's float path is unreliable across versions, and the
        # detail layer is centred on 0, so offset it into [0, 1] first.
        filled = np.stack([_inpaint_unit(high[..., ch] + 0.5, blem, radius) - 0.5 for ch in range(3)], axis=-1)
        filled = np.clip(filled, -0.5, 0.5)
        # Re-grain: borrow fine luma texture from skin a little way off so the patch keeps pores.
        shift = max(4, round(0.05 * F))
        donor = np.roll(hy, (shift, shift), axis=(0, 1))
        donor_ok = (np.roll(mask, (shift, shift), axis=(0, 1)) > 0.85) & (np.roll(blem, (shift, shift), axis=(0, 1)) == 0)
        donor_fine = donor - _gauss(donor, max(1.0, 0.006 * F))
        donor_fine = np.clip(donor_fine, -1.5 * energy, 1.5 * energy)
        regrain = np.where(donor_ok, donor_fine * 0.85, 0.0)[..., None]
        soft = _gauss(blem.astype(np.float32), max(0.8, 0.004 * F))[..., None]
        high_fixed = high * (1.0 - soft) + (filled + regrain) * soft

    # Only the repairs are applied, as a difference against the filtered tone layer. Writing the
    # guided-filtered tone layer back in would flatten every highlight and shadow the filter
    # smoothed (on glossy or deeper skin that reads as a grey haze over the whole face);
    # overall evening is the Smoothing slider's job.
    out = image + (low_fixed - low) + (high_fixed - high)
    m = mask[..., None]
    return np.clip(image * (1.0 - m) + out * m, 0.0, 1.0).astype(np.float32)
