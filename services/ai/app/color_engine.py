"""8-channel HSL colour mixer and 3-way colour grading (FineRetouch v0.5).

Both stages run on float32 sRGB in [0, 1] after the global tone stage of the renderer
(pipeline.render, stage 8b). The WebGL viewport runs the same maths in RETOUCH_FRAG
(apps/web/src/engine/shaders.ts) with constants from packages/shared/src/index.ts; the
three implementations must stay identical, so every constant below has a twin there.

HSL mixer
    Each pixel's HSV hue is split between the two nearest of eight channel centres
    (red 0, orange 30, yellow 60, green 120, aqua 180, blue 240, purple 270, magenta 300)
    with smoothstepped linear weights, so neighbouring bands blend without seams and the
    weights always sum to 1. Per channel:
      hue        -100..100  -> shifts hue by up to +/-30 degrees
      saturation -100..100  -> scales saturation x0 .. x2
      luminance  -100..100  -> darkens / brightens that colour by up to 60%
    Hue and saturation keep each pixel's luma (Rec. 601), so desaturating a navy suit
    gives dark grey rather than the lighter grey of plain HSV.
    Effects fade out on near-neutral pixels (saturation < 0.2), so greys, whites and
    blacks never pick up a cast.

3-way colour grading
    Luma splits the image into shadows (below ~25%), midtones and highlights (above ~75%)
    with soft crossovers. `blending` widens or narrows the crossovers, `balance` moves
    both pivots (positive = the highlight tint reaches further down). Each zone adds a
    zero-luma chroma offset of its wheel's hue (strength = saturation), plus an optional
    luminance offset.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schemas import ColorGrading, HslMixer

HSL_CHANNELS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HSL_CENTERS = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0], np.float32)
HUE_SHIFT_RANGE = 30.0  # degrees at hue = +/-100
LUM_RANGE = 0.6  # multiplicative brightness change at luminance = +/-100 on a fully saturated colour
NEUTRAL_EDGE = 0.2  # saturation where mixer effects reach full strength

GRADE_PIVOTS = (0.25, 0.75)  # shadow / highlight pivots in luma
GRADE_BALANCE_SHIFT = 0.2  # pivot shift at balance = +/-100
GRADE_WIDTH_MIN, GRADE_WIDTH_RANGE = 0.05, 0.4  # crossover width at blending 0 / 100
GRADE_TINT = 0.3  # chroma offset at wheel saturation 100
GRADE_LUM = 0.25  # additive offset at wheel luminance +/-100

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# HSV helpers (degrees, matching rgb2hsv / hsv2rgb in the shader)
# ---------------------------------------------------------------------------


def rgb_to_hsv(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HxWx3 sRGB -> hue (degrees, [0, 360)), saturation, value."""
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(axis=-1)
    mn = c.min(axis=-1)
    d = v - mn
    s = np.where(v > 1e-6, d / np.maximum(v, 1e-6), 0.0)
    dd = np.maximum(d, 1e-6)
    h = np.where(
        v == r,
        np.mod((g - b) / dd, 6.0),
        np.where(v == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0),
    )
    h = np.where(d > 1e-6, h * 60.0, 0.0)
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Branch-free HSV -> RGB (the same formula as the GLSL hsv2rgb)."""
    h = np.mod(h, 360.0)[..., None] / 60.0
    k = np.mod(np.array([5.0, 3.0, 1.0], np.float32) + h, 6.0)
    f = np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)
    return (v[..., None] * (1.0 - s[..., None] * f)).astype(np.float32)


def hue_segments(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees -> (lower channel index, upper channel index, smoothstepped t).

    The pixel's weight is (1 - t) on the lower channel and t on the upper one, so the eight
    weights always sum to 1. Kept as indices (not an HxWx8 array) so 100 MP exports stay cheap.
    """
    centers = np.append(HSL_CENTERS, 360.0)
    h = np.mod(h, 360.0)
    seg = np.clip(np.searchsorted(centers, h, side="right") - 1, 0, 7)
    t = (h - centers[seg]) / (centers[seg + 1] - centers[seg])
    t = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    return seg, (seg + 1) % 8, t


def hue_weights(h: np.ndarray) -> np.ndarray:
    """Dense HxWx8 channel weights (for tests and small arrays)."""
    lo, hi, t = hue_segments(h)
    eye = np.eye(8, dtype=np.float32)
    return eye[lo] * (1.0 - t)[..., None] + eye[hi] * t[..., None]


def mixer_arrays(hsl: HslMixer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel (hue, saturation, luminance) as -1..1 arrays in HSL_CHANNELS order."""
    vals = [getattr(hsl, ch) for ch in HSL_CHANNELS]
    return (
        np.array([v.hue for v in vals], np.float32) / 100.0,
        np.array([v.saturation for v in vals], np.float32) / 100.0,
        np.array([v.luminance for v in vals], np.float32) / 100.0,
    )


def hsl_is_identity(hsl: HslMixer) -> bool:
    h, s, l = mixer_arrays(hsl)
    return not (np.any(h) or np.any(s) or np.any(l))


HUE_LUT_STEPS = 3600  # 0.1 degree bins for the per-hue channel mix


def _hue_luts(hsl: HslMixer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-hue (shift in degrees, saturation scale, luminance) tables at HUE_LUT_STEPS resolution."""
    hue_k, sat_k, lum_k = mixer_arrays(hsl)
    hues = (np.arange(HUE_LUT_STEPS + 1, dtype=np.float32) * (360.0 / HUE_LUT_STEPS)).astype(np.float32)
    lo, hi, t = hue_segments(hues)

    def mix(k: np.ndarray) -> np.ndarray:
        return (k[lo] * (1.0 - t) + k[hi] * t).astype(np.float32)

    return mix(hue_k) * HUE_SHIFT_RANGE, 1.0 + mix(sat_k), mix(lum_k)


def apply_hsl_mixer(rgb: np.ndarray, hsl: HslMixer, rows: int = 512) -> np.ndarray:
    """8-channel Hue / Saturation / Luminance mixer. rgb: HxWx3 float32 sRGB in [0, 1].

    OpenCV's float HSV conversion (hue in degrees) and per-hue lookup tables keep a
    24 MP export to a few hundred milliseconds; it is processed in row bands to bound memory.
    """
    if hsl_is_identity(hsl):
        return rgb
    shift_lut, sat_lut, lum_lut = _hue_luts(hsl)
    src = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    out = np.empty_like(src)
    scale = HUE_LUT_STEPS / 360.0
    for y0 in range(0, src.shape[0], rows):
        band = np.ascontiguousarray(src[y0:y0 + rows])
        hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)  # float input: H in [0, 360), S and V in [0, 1]
        h, s = hsv[..., 0].copy(), hsv[..., 1].copy()
        idx = np.clip((h * scale + 0.5).astype(np.int32), 0, HUE_LUT_STEPS)
        g = _smoothstep(0.0, NEUTRAL_EDGE, s)
        hsv[..., 0] = np.mod(h + shift_lut[idx] * g, 360.0)
        hsv[..., 1] = np.clip(s * sat_lut[idx], 0.0, 1.0)
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # Hue and saturation moves keep the pixel's luma (desaturated navy stays dark grey);
        # only the luminance slider changes brightness.
        luma_in = band @ LUMA
        luma_out = res @ LUMA
        res *= (np.where(luma_out > 1e-4, luma_in / np.maximum(luma_out, 1e-4), 1.0) * (1.0 + LUM_RANGE * lum_lut[idx] * g * s))[..., None]
        out[y0:y0 + rows] = res
    return np.clip(out, 0.0, 1.0, out=out)


# ---------------------------------------------------------------------------
# 3-way colour grading
# ---------------------------------------------------------------------------


def grade_bounds(grading: ColorGrading) -> tuple[float, float, float]:
    """(shadow pivot, highlight pivot, crossover width) in luma."""
    b = grading.balance / 100.0
    lo = GRADE_PIVOTS[0] - GRADE_BALANCE_SHIFT * b
    hi = GRADE_PIVOTS[1] - GRADE_BALANCE_SHIFT * b
    width = GRADE_WIDTH_MIN + GRADE_WIDTH_RANGE * (grading.blending / 100.0)
    return lo, hi, width


def wheel_offset(hue: float, saturation: float, luminance: float) -> np.ndarray:
    """A wheel's RGB offset: zero-luma chroma of `hue` scaled by saturation, plus luminance."""
    pure = hsv_to_rgb(np.array([hue], np.float32), np.ones(1, np.float32), np.ones(1, np.float32))[0]
    tint = (pure - float(pure @ LUMA)) * (saturation / 100.0) * GRADE_TINT
    return (tint + (luminance / 100.0) * GRADE_LUM).astype(np.float32)


def zone_weights(y: np.ndarray, grading: ColorGrading) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi, width = grade_bounds(grading)
    ws = 1.0 - _smoothstep(lo - width / 2, lo + width / 2, y)
    wh = _smoothstep(hi - width / 2, hi + width / 2, y)
    wm = np.clip(1.0 - ws - wh, 0.0, 1.0)
    return ws, wm, wh


def grading_is_identity(grading: ColorGrading) -> bool:
    return all(w.saturation == 0 and w.luminance == 0 for w in (grading.shadows, grading.midtones, grading.highlights))


def apply_color_grading(rgb: np.ndarray, grading: ColorGrading) -> np.ndarray:
    """Shadows / midtones / highlights colour wheels. rgb: HxWx3 float32 sRGB in [0, 1]."""
    if grading_is_identity(grading):
        return rgb
    c = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    y = np.clip(c @ LUMA, 0.0, 1.0)
    ws, wm, wh = zone_weights(y, grading)
    out = c.copy()
    for weight, wheel in ((ws, grading.shadows), (wm, grading.midtones), (wh, grading.highlights)):
        if wheel.saturation == 0 and wheel.luminance == 0:
            continue
        out += weight[..., None] * wheel_offset(wheel.hue, wheel.saturation, wheel.luminance)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_color_engine(rgb: np.ndarray, hsl: HslMixer, grading: ColorGrading) -> np.ndarray:
    """Mixer, then grading: the order the renderer and the shader use."""
    return apply_color_grading(apply_hsl_mixer(rgb, hsl), grading)
