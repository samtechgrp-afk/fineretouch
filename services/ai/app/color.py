"""Colour science helpers shared by the renderer and the colour-match endpoint.

The conversions are written out explicitly (instead of cv2.cvtColor) so they
match the GLSL in apps/web/src/engine/shaders.ts bit for bit in spirit:
sRGB (D65) -> linear -> XYZ -> CIE L*a*b*.
"""

from __future__ import annotations

import numpy as np

from .schemas import ColorStats

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) float32 sRGB in [0, 1] -> (..., 3) L*a*b*."""
    xyz = srgb_to_linear(rgb) @ _RGB_TO_XYZ.T
    xyz = xyz / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    f3 = f**3
    xyz = np.where(f3 > _EPS, f3, (116.0 * f - 16.0) / _KAPPA)
    # L itself uses the exact inverse for the Y channel below the threshold.
    xyz[..., 1] = np.where(L > _KAPPA * _EPS, fy**3, L / _KAPPA)
    lin = (xyz * _WHITE) @ _XYZ_TO_RGB.T
    return linear_to_srgb(lin)


def color_stats(rgb: np.ndarray, max_pixels: int = 512 * 512) -> ColorStats:
    """Mean/std of L*a*b* over the image (subsampled for speed)."""
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] > max_pixels:
        step = int(np.ceil(flat.shape[0] / max_pixels))
        flat = flat[::step]
    lab = rgb_to_lab(flat)
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    return ColorStats(mean=tuple(float(x) for x in mean), std=tuple(float(max(x, 1e-3)) for x in std))


def transfer_lab(rgb: np.ndarray, source: ColorStats, reference: ColorStats, amount: float) -> np.ndarray:
    """Reinhard et al. statistics transfer in L*a*b*, blended by `amount` in [0, 1]."""
    if amount <= 0:
        return rgb
    lab = rgb_to_lab(rgb)
    s_mean = np.array(source.mean, dtype=np.float32)
    s_std = np.maximum(np.array(source.std, dtype=np.float32), 1e-3)
    r_mean = np.array(reference.mean, dtype=np.float32)
    r_std = np.array(reference.std, dtype=np.float32)
    matched = (lab - s_mean) / s_std * r_std + r_mean
    out = lab_to_rgb(matched)
    return (rgb * (1.0 - amount) + out * amount).astype(np.float32)
