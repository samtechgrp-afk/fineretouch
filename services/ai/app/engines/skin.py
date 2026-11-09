"""Natural frequency-separation retouch, acne / dark-spot tools, pigmentation and the CIELAB skin-tone guard.

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
from .spots import detect_spots, heal_spots

# ---------------------------------------------------------------------------
# 1c. Natural frequency-separation retouch and the skin-tone guard
# ---------------------------------------------------------------------------

# Tone-layer median: the tone layer is shrunk so one pixel is about this share of its blur sigma.
TONE_MEDIAN_STEP = 0.6
# The median only replaces tone that differs from it by less than this (luma): blotches move,
# moles, freckle clusters and feature edges (bigger contrasts) stay where they are.
TONE_MEDIAN_KEEP = (0.025, 0.06)
# Skin-tone guard (CIELAB). Retouching may lower a skin pixel's chroma to at most this share of
# the lower of its own and its surrounding skin's chroma, and may turn its hue at most this many
# degrees beyond the range spanned by those two. Mirrored in shaders.ts (SKIN_TONE_GUARD in shared).
TONE_GUARD_CHROMA_KEEP = 0.9
TONE_GUARD_HUE_DEG = 6.0
# Below this chroma (a*b* units) hue is noise, so the hue lock fades out.
TONE_GUARD_MIN_CHROMA = (2.0, 6.0)


def split_frequencies(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-layer frequency separation: (tone, detail) with tone + detail == image exactly.

    tone is a Gaussian low-pass (colour, shading, redness, blotches); detail is the remainder
    (pores, fine hair, micro texture), signed and centred on zero.
    """
    tone = _gauss(image, sigma)
    return tone, image - tone


def adaptive_tone_median(tone: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Median-filter the tone layer where `weight` allows, leaving real structure alone.

    The tone layer has no detail finer than `sigma`, so the median runs on a copy shrunk to
    about TONE_MEDIAN_STEP * sigma per pixel (cheap at any resolution) and only the
    correction is scaled back up. Where the median would move the tone by more than
    TONE_MEDIAN_KEEP the change fades out, which keeps moles, brows and contours.
    """
    if not np.any(weight > 1e-3):
        return tone
    h, w = tone.shape[:2]
    f = max(1.0, TONE_MEDIAN_STEP * sigma)
    size = (max(1, round(w / f)), max(1, round(h / f)))
    small = cv2.resize(tone, size, interpolation=cv2.INTER_AREA) if f > 1 else tone
    med = cv2.medianBlur(np.ascontiguousarray(small, np.float32), 5)
    delta = med - small
    # Judge each change by the largest one around it, so a mole's rim is kept with its core.
    big = cv2.dilate(np.abs(_luma(delta)).astype(np.float32), _ellipse(5))
    delta *= (1.0 - _smoothstep(TONE_MEDIAN_KEEP[0], TONE_MEDIAN_KEEP[1], big))[..., None]
    if f > 1:
        delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)
    return tone + delta * weight[..., None]


def reduce_acne_redness(image: np.ndarray, heal: np.ndarray, skin: np.ndarray, face_w: float, amount: float) -> np.ndarray:
    """Post-acne redness: around healed acne, pull excess red (Cr above the local skin) back.

    Only a halo of about 3 spot widths around the detected acne is touched, so the cheeks'
    natural flush elsewhere stays.
    """
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not heal.any():
        return image
    F = max(face_w, 32.0)
    halo = cv2.dilate(heal.astype(np.uint8), _ellipse(max(3, int(0.03 * F) | 1))).astype(np.float32)
    halo = _gauss(halo, 0.01 * F) * skin
    y = _luma(image)
    cb, cr = _cbcr(image)
    base_cr = _masked_blur(cr, skin.astype(np.float32), 0.05 * F)
    excess = np.maximum(cr - base_cr, 0.0)
    return np.clip(_from_ycbcr(y, cb, cr - excess * 0.85 * t * halo), 0.0, 1.0).astype(np.float32)


def remove_acne(
    image: np.ndarray, skin: np.ndarray, faces: list[FaceAnalysis], amount: float, multiplier: int = 1,
    protect_beauty_marks: bool = True,
) -> np.ndarray:
    """Acne removal: red spots, pimples and raised bumps healed one by one, then local redness."""
    if amount <= 0:
        return image
    det = detect_spots(image, skin, faces, multiplier, amount, protect_beauty_marks, kind="acne")
    if det.count == 0:
        return image
    c = heal_spots(image, skin, faces, amount, multiplier, protect_beauty_marks, detection=det)
    return reduce_acne_redness(c, det.heal, skin, det.face_width, amount)


# Pigmentation works one band coarser than smoothing: patches (sun spots, melasma, post-acne
# marks) span the S2 band; the wider skin is a blur PIGMENT_WIDE times larger (S3 = 2.5 x S2).
PIGMENT_WIDE = 2.5


def correct_pigmentation(image: np.ndarray, skin: np.ndarray, sigma_base: float, amount: float) -> np.ndarray:
    """Uneven pigmentation: brownish patches on the tone layer pulled toward the skin around them.

    A patch qualifies when its tone layer is darker than the wider skin AND more yellow-brown
    (lower Cb). Shadows darken without that hue shift and strong dark edges (nose side, jaw)
    are excluded, so facial shading stays. Pores are untouched (detail layer is kept).
    Mirrored in shaders.ts (uPigment).
    """
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0:
        return image
    tone = _gauss(image, sigma_base)
    base = _gauss(image, PIGMENT_WIDE * sigma_base)
    d_y = _luma(base) - _luma(tone)
    cb_t, _ = _cbcr(tone)
    cb_b, _ = _cbcr(base)
    pig = _smoothstep(0.004, 0.02, d_y) * _smoothstep(0.002, 0.008, cb_b - cb_t)
    edge = _smoothstep(0.06, 0.14, d_y)
    k = (0.9 * t * pig * (1.0 - edge) * skin)[..., None]
    return np.clip(image + (base - tone) * k, 0.0, 1.0).astype(np.float32)


def remove_dark_spots(
    image: np.ndarray, skin: np.ndarray, faces: list[FaceAnalysis], amount: float, multiplier: int = 1,
    protect_beauty_marks: bool = True, keep_freckles: bool = False, *, sigma_tone: float, sigma_base: float,
) -> np.ndarray:
    """Dark spot removal: isolated dark / pigmented marks healed (texture cloned back), then
    uneven pigmentation evened on the tone layer. Beauty marks and, optionally, freckles stay."""
    if amount <= 0:
        return image
    det = detect_spots(image, skin, faces, multiplier, amount, protect_beauty_marks, kind="dark", keep_freckles=keep_freckles)
    c = heal_spots(image, skin, faces, amount, multiplier, protect_beauty_marks, detection=det) if det.count else image
    pig = amount if not keep_freckles else 0.6 * amount
    return correct_pigmentation(c, skin, sigma_base, pig)


def natural_skin_retouch(
    image: np.ndarray,
    skin: np.ndarray,
    faces: list[FaceAnalysis],
    *,
    smoothing: float,
    texture_retention: float,
    spot_amount: float = 0.0,
    multiplier: int = 1,
    protect_beauty_marks: bool = True,
    sigma_tone: float,
    sigma_base: float,
    bands: tuple[np.ndarray, np.ndarray] | None = None,
    keep_freckles: bool = False,
) -> np.ndarray:
    """Texture-preserving skin retouch on two frequency layers.

    image: HxWx3 float32 sRGB; skin: HxW skin weight. smoothing, texture_retention and
    spot_amount are 0..100 (the Smoothing, Texture retention and Spot healing sliders).

    1. Spot healing (heal_spots): each acne spot, pimple or dark spot is repaired in the
       tone layer (inpainted from the skin around it) and gets healthy pores cloned from a
       nearby patch, because its own outline is detail-scale. Pixels outside spots are
       untouched.
    2. Split into a tone layer (Gaussian sigma_tone) and a detail layer.
    3. Tone layer only: the blotch band between sigma_tone and sigma_base is suppressed and an
       adaptive median evens what is left. Edges (contours, lash lines) are kept.
    4. The detail layer is added back. At 100% texture retention every pore is re-injected
       unchanged; lower values let Smoothing soften fine texture too.

    sigma_tone / sigma_base are the renderer's S1 / S2 blur radii, so the live preview
    (shaders.ts) runs the same split. bands: optional precomputed (tone, base) blurs of
    `image`, reused when no spot was healed.
    """
    c = image
    if spot_amount > 0:
        det = detect_spots(c, skin, faces, multiplier, spot_amount, protect_beauty_marks, keep_freckles=keep_freckles)
        c = heal_spots(c, skin, faces, spot_amount, multiplier, protect_beauty_marks, detection=det)
    smooth = float(np.clip(smoothing / 100.0, 0.0, 1.0))
    if smooth <= 0:
        return c
    retention = float(np.clip(texture_retention / 100.0, 0.0, 1.0))
    if bands is not None and c is image:
        tone, base = bands
        detail = c - tone
    else:
        tone, detail = split_frequencies(c, sigma_tone)
        base = _gauss(c, sigma_base)
    blotch = tone - base
    edge = _smoothstep(0.08, 0.22, np.abs(_luma(detail)) + np.abs(_luma(blotch)))
    k = skin * smooth * (1.0 - edge)
    tone = tone - (0.85 * k)[..., None] * blotch
    tone = adaptive_tone_median(tone, k, sigma_tone)
    detail = detail * (1.0 - k * (1.0 - retention))[..., None]
    return (tone + detail).astype(np.float32)


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def skin_tone_guard(
    image: np.ndarray, original: np.ndarray, reference: np.ndarray, skin: np.ndarray
) -> np.ndarray:
    """Keep retouched skin from going grey or shifting hue (CIELAB a*/b* lock).

    image: the retouched pixels; original: the same pixels before retouching; reference: the
    original's broad local average (the person's own undertone around each pixel); all HxWx3
    float32 sRGB. skin: HxW weight.

    For each skin pixel, chroma may not fall below TONE_GUARD_CHROMA_KEEP of the lower of the
    original and reference chroma, and hue may not leave the arc between the original and
    reference hue by more than TONE_GUARD_HUE_DEG. L* (the retouch's brightness work) is never
    changed. Pixels the retouch did not desaturate or turn are left as they are, so freckles
    and natural variation in melanin stay; only ashen or off-hue results are pulled back.
    """
    sel = skin > 1e-3
    if not sel.any():
        return image
    lab = rgb_to_lab(image[sel])
    lab_o = rgb_to_lab(original[sel])
    lab_r = rgb_to_lab(reference[sel])
    ch = np.hypot(lab[:, 1], lab[:, 2])
    ch_o = np.hypot(lab_o[:, 1], lab_o[:, 2])
    ch_r = np.hypot(lab_r[:, 1], lab_r[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    hue_o = np.degrees(np.arctan2(lab_o[:, 2], lab_o[:, 1]))
    hue_r = np.degrees(np.arctan2(lab_r[:, 2], lab_r[:, 1]))
    hue = np.where(ch < 1e-5, hue_r, hue)
    hue_o = np.where(ch_o < 1e-5, hue_r, hue_o)

    ch_new = np.maximum(ch, TONE_GUARD_CHROMA_KEEP * np.minimum(ch_o, ch_r))
    d = _wrap_deg(hue - hue_r)
    d_o = _wrap_deg(hue_o - hue_r)
    d_new = np.clip(d, np.minimum(d_o, 0.0) - TONE_GUARD_HUE_DEG, np.maximum(d_o, 0.0) + TONE_GUARD_HUE_DEG)
    lock = _smoothstep(TONE_GUARD_MIN_CHROMA[0], TONE_GUARD_MIN_CHROMA[1], np.minimum(ch_r, ch_new))
    hue_new = hue_r + d + (d_new - d) * lock
    # Untouched pixels keep their exact values (no Lab round trip).
    moved = (ch_new > ch + 1e-4) | (np.abs(hue_new - (hue_r + d)) > 1e-3)
    if not moved.any():
        return image
    t = np.radians(hue_new[moved])
    fixed = np.stack([lab[moved, 0], ch_new[moved] * np.cos(t), ch_new[moved] * np.sin(t)], axis=-1)
    rgb = lab_to_rgb(fixed)
    w = skin[sel][moved][:, None]
    vals = image[sel]
    vals[moved] = vals[moved] + (rgb - vals[moved]) * w
    out = image.copy()
    out[sel] = vals
    return out.astype(np.float32)
