"""Depth-aware lens blur.

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
# 3. Depth-aware bokeh
# ---------------------------------------------------------------------------

BOKEH_RADIUS_SHARE = 0.02  # blurRadius 100 => 2% of the long edge, before the multiplier
FOCUS_START, FOCUS_END = 0.04, 0.45  # depth band over which blur ramps up from the subject


def aperture_kernel(radius: float, shape: str = "circle") -> np.ndarray:
    """Normalised, anti-aliased disk or hexagon kernel."""
    r = max(radius, 0.5)
    n = int(math.ceil(r)) + 1
    yy, xx = np.mgrid[-n: n + 1, -n: n + 1].astype(np.float32)
    if shape == "hex":
        ax, ay = np.abs(xx), np.abs(yy)
        dist = np.maximum(ax * math.cos(math.pi / 6) + ay * 0.5, ay) / math.cos(math.pi / 6)
    else:
        dist = np.sqrt(xx * xx + yy * yy)
    k = np.clip(r + 0.5 - dist, 0.0, 1.0)
    return (k / k.sum()).astype(np.float32)


def circle_of_confusion(depth: np.ndarray, max_coc: float) -> np.ndarray:
    return max_coc * _smoothstep(FOCUS_START, FOCUS_END, depth)


def apply_depth_bokeh(
    image: np.ndarray,
    depth_map: np.ndarray,
    blur_factor: float = 2.0,
    blur_radius: float = 50.0,
    aperture: str = "circle",
    highlight_boost: float = 40.0,
    layers: int = 8,
) -> np.ndarray:
    """Layered depth-of-field.

    depth_map: HxW (any size; resized) in [0, 1], 0 = subject plane, 1 = far.
    blur_factor: 1.0-3.0 multiplier on the base radius (the UI's 100-300%).

    Each pixel's circle of confusion picks a pair of radius layers; each layer is
    convolved with the aperture kernel in linear light using normalised convolution,
    so the in-focus subject never bleeds into the background (no halo). Highlights are
    boosted before the blur so bright points bloom into aperture-shaped discs. Large
    radii are processed at a reduced scale (kernels capped at ~32 px) and upsampled,
    which keeps a 24 MP export to a few hundred milliseconds of filtering.
    """
    h, w = image.shape[:2]
    L = float(max(h, w))
    max_coc = (blur_radius / 100.0) * BOKEH_RADIUS_SHARE * L * float(np.clip(blur_factor, 1.0, 3.0))
    if max_coc < 0.75:
        return image
    depth = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coc_full = circle_of_confusion(np.clip(depth, 0, 1), max_coc)

    s = min(1.0, 32.0 / max_coc)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if s < 1 else image
    coc = cv2.resize(coc_full, size, interpolation=cv2.INTER_AREA) * s if s < 1 else coc_full
    max_r = max_coc * s

    lin = srgb_to_linear(small)
    gain = 1.0 + (highlight_boost / 100.0) * 6.0 * _smoothstep(0.55, 0.95, _luma(lin))
    lin = lin * gain[..., None]

    radii = np.linspace(0.0, max_r, layers)
    pos = np.clip(coc / max(max_r, 1e-6) * (layers - 1), 0, layers - 1)
    num = np.zeros_like(lin)
    den = np.zeros(lin.shape[:2], np.float32)
    for i, r in enumerate(radii):
        wgt = np.clip(1.0 - np.abs(pos - i), 0.0, 1.0).astype(np.float32)
        if not wgt.any():
            continue
        if r < 0.5:
            num += lin * wgt[..., None]
            den += wgt
            continue
        k = aperture_kernel(r, aperture)
        num += cv2.filter2D(lin * wgt[..., None], -1, k, borderType=cv2.BORDER_REFLECT)
        den += cv2.filter2D(wgt, -1, k, borderType=cv2.BORDER_REFLECT)
    blurred = linear_to_srgb(np.clip(num / np.maximum(den, 1e-4)[..., None], 0.0, 1.0))
    if s < 1:
        blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = _smoothstep(0.35, 1.5, coc_full)[..., None]
    return np.clip(image * (1.0 - alpha) + blurred * alpha, 0.0, 1.0).astype(np.float32)
