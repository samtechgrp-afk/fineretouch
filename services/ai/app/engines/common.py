"""Shared helpers for the retouch engines (colour conversions, blurs, inpainting)."""


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

__all__ = ['LUMA', '_smoothstep', '_luma', '_cbcr', '_from_ycbcr', '_gauss', '_masked_blur', '_ellipse', '_inpaint_unit', '_compact_components', 'cv2', 'np', 'math', 'Sequence', 'dataclass', 'Path']

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luma(c: np.ndarray) -> np.ndarray:
    return c @ LUMA


def _cbcr(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5, 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5


def _from_ycbcr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb, cr = cb - 0.5, cr - 0.5
    return np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], axis=-1)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sigma, 0.6), borderType=cv2.BORDER_REFLECT)


def _masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: a blur that only averages pixels inside `mask`."""
    m = mask[..., None] if img.ndim == 3 else mask
    num = _gauss(img * m, sigma)
    den = _gauss(mask, sigma)
    den = den[..., None] if img.ndim == 3 else den
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), img)


def _ellipse(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _inpaint_unit(x: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Telea inpainting of a single float channel in [0, 1] via a 16-bit round trip."""
    u16 = (np.clip(x, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    return cv2.inpaint(u16, mask, radius, cv2.INPAINT_TELEA).astype(np.float32) / 65535.0


def _compact_components(binary: np.ndarray, max_side: float, min_fill: float) -> np.ndarray:
    """Float mask of connected components that are small (bbox side <= max_side) and blob-like
    (area / bbox area >= min_fill). Rejects lines, folds and broad shading."""
    b = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return np.zeros(b.shape, np.float32)
    bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    ok = (np.maximum(bw, bh) <= max_side) & (area / (bw * bh) >= min_fill) & (np.minimum(bw, bh) / np.maximum(bw, bh) > 0.3)
    keep = np.concatenate([[False], ok])
    return keep[labels].astype(np.float32)
