"""Colour science helpers shared by the renderer and the colour-match endpoint.

The conversions are written out explicitly (instead of cv2.cvtColor) so they
match the GLSL in apps/web/src/engine/shaders.ts bit for bit in spirit:
sRGB (D65) -> linear -> XYZ -> CIE L*a*b*.
"""

from __future__ import annotations

import numpy as np

from .schemas import TONE_QUANTILE_COUNT, ColorStats

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


TONE_QUANTILES = np.linspace(0.0, 1.0, TONE_QUANTILE_COUNT)
# Added to covariance diagonals (L*a*b* units squared) so flat images stay invertible.
COV_EPS = 1.0
# Clipped pixels (blown studio white, crushed black) carry no colour information and would
# dominate the statistics of a backdrop shot; they are ignored unless they are nearly everything.
CLIP_LOW, CLIP_HIGH, MIN_CONTENT_SHARE = 2.0, 98.0, 0.05
# Largest shift (L* units) the tone curve may apply at any knot, so a reference with a very
# different histogram (e.g. mostly white) re-grades the photo instead of blowing it out.
MAX_CURVE_SHIFT = 25.0


def color_stats(rgb: np.ndarray, max_pixels: int = 512 * 512) -> ColorStats:
    """Mean/std/covariance of L*a*b* and L* quantiles over the image (subsampled for speed)."""
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] > max_pixels:
        step = int(np.ceil(flat.shape[0] / max_pixels))
        flat = flat[::step]
    lab = rgb_to_lab(flat).astype(np.float64)
    content = (lab[:, 0] > CLIP_LOW) & (lab[:, 0] < CLIP_HIGH)
    if content.sum() >= MIN_CONTENT_SHARE * lab.shape[0]:
        lab = lab[content]
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    cov = np.cov(lab, rowvar=False, bias=True)
    q = np.quantile(lab[:, 0], TONE_QUANTILES)
    return ColorStats(
        mean=tuple(float(x) for x in mean),
        std=tuple(float(max(x, 1e-3)) for x in std),
        cov=[float(x) for x in cov.reshape(-1)],
        lQuantiles=[float(x) for x in q],
    )


def _sym_pow(m: np.ndarray, p: float) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return (v * np.maximum(w, 1e-9) ** p) @ v.T


def match_matrix(source: ColorStats, reference: ColorStats) -> np.ndarray:
    """3x3 linear map in L*a*b* (applied to lab - source.mean).

    With covariances on both sides it is the Monge-Kantorovich solution, the linear map
    that sends the source covariance exactly onto the reference covariance while moving
    colours as little as possible (Pitie & Kokaram 2007). Otherwise per-channel scaling
    (Reinhard et al. 2001). Mirrored in apps/web/src/engine/color.ts.
    """
    if source.cov is not None and reference.cov is not None:
        cs = np.array(source.cov, np.float64).reshape(3, 3) + np.eye(3) * COV_EPS
        cr = np.array(reference.cov, np.float64).reshape(3, 3) + np.eye(3) * COV_EPS
        cs_h = _sym_pow(cs, 0.5)
        cs_ih = _sym_pow(cs, -0.5)
        return (cs_ih @ _sym_pow(cs_h @ cr @ cs_h, 0.5) @ cs_ih).astype(np.float32)
    s_std = np.maximum(np.array(source.std, np.float32), 1e-3)
    return np.diag(np.array(reference.std, np.float32) / s_std).astype(np.float32)


def monotone_quantiles(q: list[float] | tuple[float, ...]) -> np.ndarray:
    """Quantiles made strictly increasing so they can be used as interpolation knots."""
    out = np.array(q, np.float64)
    for i in range(1, out.size):
        out[i] = max(out[i], out[i - 1] + 1e-3)
    return out


def tone_curve_knots(source: list[float], reference: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-matching knots (x = source L*, y = target L*), with each shift limited to MAX_CURVE_SHIFT."""
    qs = monotone_quantiles(source)
    qr = monotone_quantiles(reference)
    qr = qs + np.clip(qr - qs, -MAX_CURVE_SHIFT, MAX_CURVE_SHIFT)
    return qs, monotone_quantiles(qr)


def transfer_lab(rgb: np.ndarray, source: ColorStats, reference: ColorStats, amount: float) -> np.ndarray:
    """Statistical colour transfer in L*a*b*, blended by `amount` in [0, 1].

    a*/b* (and L* when no tone curve is available) follow the linear `match_matrix`;
    with L* quantiles on both sides, L* is histogram matched through a 17-knot tone curve.
    """
    if amount <= 0:
        return rgb
    lab = rgb_to_lab(rgb)
    m = match_matrix(source, reference)
    matched = (lab - np.array(source.mean, np.float32)) @ m.T + np.array(reference.mean, np.float32)
    if source.lQuantiles is not None and reference.lQuantiles is not None:
        qs, qr = tone_curve_knots(source.lQuantiles, reference.lQuantiles)
        matched[..., 0] = np.interp(lab[..., 0], qs, qr).astype(np.float32)
    out = lab_to_rgb(matched.astype(np.float32))
    return (rgb * (1.0 - amount) + out * amount).astype(np.float32)
