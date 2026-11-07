"""Advanced retouch stages for the full-resolution renderer (FineRetouch v0.2).

  natural_skin_retouch        texture-preserving frequency separation (v0.5): spots healed, then
                              blotches smoothed and median-evened on the tone layer only, and
                              the pore/detail layer re-injected by the Texture retention factor.
  skin_tone_guard             CIELAB a*/b* lock so retouched skin never turns ashen or off-hue.
  heal_spots                  local acne / pimple / dark-spot healing (v0.5): scale-normalised
                              Laplacian-of-Gaussian detection on a darkness + redness signal
                              inside the skin mask, beauty-mark and feature protection, then
                              per-spot inpainting of the tone layer and healthy-skin patch
                              cloning of the texture layer. 1x / 2x / 3x set how far it reaches.
  apply_boosted_skin_retouch  2x/3x multi-stage frequency separation: recursive guided
                              filtering of the low-frequency colour layer plus adaptive
                              blemish detection and patch-based repair of the detail layer.
  remove_stray_hairs          flyaway hairs around the face outline and forehead:
                              morphological + Gabor line detection, then inpainting.
  apply_depth_bokeh           layered disk / hexagon depth-of-field from a depth map,
                              with highlight bloom and a 100-300% radius multiplier.
  reduce_glasses_glare        eyeglass lens reflections (v0.6): reflection model inverted per pixel
                              on a landmark lens region, blown-out cores inpainted, catchlights kept.
  correct_teeth               CIELAB whitening and stain removal on an enamel mask (v0.6).
  remove_clothing_wrinkles    "virtual iron" (v0.6): crease / fold bands of log-luminance attenuated
                              inside the garment mask, weave and drape kept, colour untouched.
  shift_clothing_color        garment recolour in CIELAB, skin excluded (v0.6).
  style_profile /             reference-photo metrics and the mapping to editable
  extract_and_sync_style      adjustments for a whole batch (port of packages/shared/src/style.ts).

All functions take and return float32 sRGB in [0, 1] unless stated. The WebGL preview
(apps/web/src/engine/shaders.ts) approximates these stages in real time; exports from
this service run the full versions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .color import color_stats, lab_to_rgb, linear_to_srgb, rgb_to_lab, srgb_to_linear
from .depth import guided_filter
from .schemas import (
    Adjustments,
    ColorMatch,
    FaceAnalysis,
    StyleProfile,
    StyleSyncResponse,
    StyleSyncTarget,
)

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

    out = low_fixed + high_fixed
    m = mask[..., None]
    return np.clip(image * (1.0 - m) + out * m, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1b. Local spot healing (acne, pimples, dark spots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpotProfile:
    """How hard one multiplier step pushes spot healing (all sizes are shares of face width F)."""

    k: float  # detection threshold in robust standard deviations of the LoG response
    max_diameter: float  # largest spot treated
    grow: float  # dilation of each spot before healing (covers its halo)
    strength: float  # blend of the healed patch at amount 100


SPOT_PROFILES: dict[int, SpotProfile] = {
    1: SpotProfile(k=3.0, max_diameter=0.05, grow=0.004, strength=0.9),
    2: SpotProfile(k=2.4, max_diameter=0.07, grow=0.006, strength=1.0),
    3: SpotProfile(k=1.9, max_diameter=0.09, grow=0.008, strength=1.0),
}
# Spots are looked for at these LoG scales (sigma as a share of F; blob radius ~ sigma * sqrt(2)).
SPOT_SIGMAS = (0.0065, 0.01, 0.015, 0.022)
# Beauty marks: much darker than the skin around them, not red, and big enough to be deliberate.
MOLE_MIN_DARKNESS = 0.2
MOLE_MAX_REDNESS = 0.015
MOLE_MIN_DIAMETER = 0.012
# Detection runs with the face about this many pixels wide; pores vanish below the finest scale.
SPOT_WORK_FACE = 420.0


@dataclass
class SpotDetection:
    """Result of detect_spots, in the coordinates of the image passed in."""

    heal: np.ndarray  # uint8 HxW: 1 where a spot (plus its halo) will be healed
    protected: np.ndarray  # uint8 HxW: beauty marks that were found and kept
    count: int
    protected_count: int
    face_width: float


def face_width_px(faces: list[FaceAnalysis], mask: np.ndarray, w: int) -> float:
    if faces:
        return float(max(f.box[2] for f in faces) * w)
    return float(np.clip(np.sqrt(float((mask > 0.5).sum())) * 1.1, 32.0, max(mask.shape)))


def feature_protection_mask(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    """uint8 mask of eyes, brows, lips and nostrils, dilated so their rims are safe too."""
    out = np.zeros((h, w), np.uint8)
    for f in faces:
        fw = max(8.0, f.box[2] * w)
        r = f.regions
        for region, grow in ((r.leftEye, 0.09), (r.rightEye, 0.09), (r.leftBrow, 0.035), (r.rightBrow, 0.035),
                             (r.lips, 0.03), (r.nose, 0.025)):
            if len(region) < 3:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(region, np.float32) * [w, h]).astype(np.int32)], 1)
            k = _ellipse(max(3, int(fw * grow) | 1))
            if region is r.leftEye or region is r.rightEye:
                # Wider than tall: covers the lash line and the crow's-feet creases at the corners.
                kw, kh = max(3, int(fw * grow * 2.2) | 1), max(3, int(fw * grow) | 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
            out |= cv2.dilate(m, k)
    return out


def _log_response(signal: np.ndarray, sigmas_px: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Max over scales of the scale-normalised LoG (-sigma^2 * laplacian) and the winning sigma."""
    best = np.full(signal.shape, -np.inf, np.float32)
    best_sigma = np.zeros(signal.shape, np.float32)
    for sg in sigmas_px:
        g = cv2.GaussianBlur(signal, (0, 0), sigmaX=max(sg, 0.5), borderType=cv2.BORDER_REFLECT)
        r = -(sg * sg) * cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        better = r > best
        best = np.where(better, r, best)
        best_sigma = np.where(better, sg, best_sigma)
    return best, best_sigma


def detect_spots(
    image: np.ndarray,
    mask: np.ndarray,
    faces: list[FaceAnalysis],
    multiplier: int = 1,
    amount: float = 100.0,
    protect_beauty_marks: bool = True,
) -> SpotDetection:
    """Find acne, pimples and dark spots inside the skin mask.

    Signal: darkness and redness relative to the surrounding skin, so a red pimple and a
    brown spot both read as bright blobs. Blobs are found with a multi-scale normalised
    LoG; the threshold adapts to each photo through the response's robust spread on skin
    (a clear complexion and a grainy high-ISO frame get the same false-positive rate).
    Pores sit below the finest scale. Lines (wrinkles, lash lines, hair), large shading
    and anything on or next to the eyes, brows, lips and nostrils are rejected.
    """
    prof = SPOT_PROFILES.get(int(multiplier), SPOT_PROFILES[1])
    h, w = image.shape[:2]
    F = face_width_px(faces, mask, w)
    s = min(1.0, SPOT_WORK_FACE / F)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    work = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if s < 1 else image
    m = cv2.resize(mask, size, interpolation=cv2.INTER_AREA) if s < 1 else mask
    Fw = F * s
    protect = feature_protection_mask(faces, size[1], size[0])

    # The colour-based skin weight drops on the spots themselves (a red pimple fails the
    # skin-tone test), so close the mask over spot-sized holes before looking inside it.
    m = cv2.morphologyEx(m.astype(np.float32), cv2.MORPH_CLOSE, _ellipse(max(3, int(0.08 * Fw) | 1)))
    core = (m > 0.6) & (protect == 0)
    empty = np.zeros((h, w), np.uint8)
    if core.sum() < 64:
        return SpotDetection(empty, empty.copy(), 0, 0, F)

    y = _luma(work)
    _, cr = _cbcr(work)
    ctx = 0.06 * Fw
    base_y = _masked_blur(y, core.astype(np.float32), ctx)
    base_cr = _masked_blur(cr, core.astype(np.float32), ctx)
    darkness = base_y - y
    redness = cr - base_cr
    signal = (darkness + 1.5 * np.maximum(redness, 0.0)).astype(np.float32)
    # A median filter about a pore wide removes pores and pore clusters (sparse, high
    # frequency) but keeps the solid core of real spots, so LoG sees spots, not texture.
    signal = cv2.medianBlur(signal, 5 if Fw >= 300 else 3)

    sigmas = [max(0.7, f * Fw) for f in SPOT_SIGMAS]
    resp, _ = _log_response(signal, sigmas)
    vals = resp[core]
    med = float(np.median(vals))
    spread = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-5
    a = float(np.clip(amount / 100.0, 0.0, 1.0))
    k = prof.k * (1.25 - 0.25 * a)
    # Absolute floor: below ~2.5% contrast a "spot" is invisible and only texture would be lost.
    floor = 0.025 * (1.3 - 0.15 * (int(multiplier) - 1))
    cand = ((resp > med + k * spread) & (resp > floor) & core & (signal > 0.6 * floor)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return SpotDetection(empty, empty.copy(), 0, 0, F)
    bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    side = np.maximum(bw, bh)
    diameter = np.sqrt(4.0 * area / math.pi)
    min_d = max(2.0, 0.008 * Fw)  # pores and single-pixel noise stay
    blob = (
        (diameter <= prof.max_diameter * Fw) & (diameter >= min_d)
        & (area / (bw * bh) >= 0.4) & (np.minimum(bw, bh) / side >= 0.45)
    )

    flat = labels.ravel()
    counts = np.maximum(np.bincount(flat, minlength=n)[1:], 1)
    mean_dark = np.bincount(flat, weights=darkness.ravel(), minlength=n)[1:] / counts
    mean_red = np.bincount(flat, weights=redness.ravel(), minlength=n)[1:] / counts
    peak_dark = np.zeros(n - 1, np.float32)
    np.maximum.at(peak_dark, flat[flat > 0] - 1, darkness.ravel()[flat > 0])
    mole = (
        blob & (peak_dark >= MOLE_MIN_DARKNESS) & (mean_dark >= 0.6 * MOLE_MIN_DARKNESS)
        & (mean_red <= MOLE_MAX_REDNESS) & (diameter >= MOLE_MIN_DIAMETER * Fw)
    )
    if not protect_beauty_marks:
        mole[:] = False
    heal_ids = blob & ~mole

    lut_heal = np.concatenate([[0], heal_ids]).astype(np.uint8)
    lut_mole = np.concatenate([[0], mole]).astype(np.uint8)
    heal = lut_heal[labels]
    moles = lut_mole[labels]
    grow = max(3, round(prof.grow * Fw * 2) | 1)
    heal = cv2.dilate(heal, _ellipse(grow))
    heal[cv2.dilate(moles, _ellipse(grow + 2)) > 0] = 0
    heal[protect > 0] = 0
    if s < 1:
        heal = (cv2.resize(heal.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.35).astype(np.uint8)
        moles = (cv2.resize(moles.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.35).astype(np.uint8)
    return SpotDetection(heal, moles, int(heal_ids.sum()), int(mole.sum()), F)


def _best_donor(
    box: tuple[int, int, int, int], low_y: np.ndarray, ok: np.ndarray, distance: int
) -> tuple[int, int] | None:
    """Offset (dy, dx) of the nearby healthy-skin patch whose tone best matches the spot's rim."""
    x, y, bw, bh = box
    h, w = ok.shape
    pad = max(2, max(bw, bh) // 3)
    y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
    ring = low_y[y0:y1, x0:x1]
    target = float(np.median(ring))
    best, best_cost = None, np.inf
    for i in range(12):
        t = 2 * math.pi * i / 12
        for dist in (distance, int(distance * 1.6)):
            dy, dx = round(math.sin(t) * dist), round(math.cos(t) * dist)
            yy0, yy1, xx0, xx1 = y0 + dy, y1 + dy, x0 + dx, x1 + dx
            if yy0 < 0 or xx0 < 0 or yy1 > h or xx1 > w:
                continue
            if not ok[yy0:yy1, xx0:xx1].all():
                continue
            cost = abs(float(np.median(low_y[yy0:yy1, xx0:xx1])) - target) + 0.002 * dist / max(1, distance)
            if cost < best_cost:
                best, best_cost = (dy, dx), cost
        if best is not None and best_cost < 0.01:
            break
    return best


def heal_spots(
    image: np.ndarray,
    mask: np.ndarray,
    faces: list[FaceAnalysis],
    amount: float = 100.0,
    multiplier: int = 1,
    protect_beauty_marks: bool = True,
    detection: SpotDetection | None = None,
) -> np.ndarray:
    """Heal acne, pimples and dark spots one by one. image: HxWx3 float32 sRGB; mask: HxW skin weight.

    For every detected spot the image is split into a tone layer and a texture layer at
    roughly the spot's size. The tone layer is inpainted (Telea) from the healthy skin
    around the spot, so colour and shading continue across it. The texture layer is
    cloned from the best-matching healthy patch nearby (same lighting, real pores), so
    a healed spot never turns into a smooth hole. Skin outside the spots is untouched:
    this replaces the old global low-band blend with purely local repairs.
    """
    if amount <= 0:
        return image
    det = detection or detect_spots(image, mask, faces, multiplier, amount, protect_beauty_marks)
    if det.count == 0 or not det.heal.any():
        return image
    prof = SPOT_PROFILES.get(int(multiplier), SPOT_PROFILES[1])
    h, w = image.shape[:2]
    F = det.face_width
    ys, xs = np.nonzero(det.heal)
    pad = max(8, int(0.12 * F))
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = image[y0:y1, x0:x1].astype(np.float32)
    heal = det.heal[y0:y1, x0:x1]
    skin = mask[y0:y1, x0:x1]

    sigma = max(1.0, 0.008 * F)
    low = _gauss(crop, sigma)
    high = crop - low
    radius = max(2, round(0.012 * F))
    low_fixed = np.stack([_inpaint_unit(low[..., c], heal, radius) for c in range(3)], axis=-1)

    # Texture: clone from a nearby healthy patch per spot; fall back to inpainted (flat) grain.
    high_fixed = np.stack([_inpaint_unit(high[..., c] + 0.5, heal, radius) - 0.5 for c in range(3)], axis=-1)
    low_y = _luma(low)
    ok = (skin > 0.7) & (cv2.dilate(heal, _ellipse(5)) == 0)  # donors: plain skin, no spot nearby
    n, labels, stats, _ = cv2.connectedComponentsWithStats(heal, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        dist = max(4, int(max(bw, bh) * 1.4))
        off = _best_donor((x, y, bw, bh), low_y, ok, dist)
        if off is None:
            continue
        dy, dx = off
        sel = labels[y:y + bh, x:x + bw] == i
        donor = high[y + dy:y + dy + bh, x + dx:x + dx + bw]
        patch = high_fixed[y:y + bh, x:x + bw]
        patch[sel] = donor[sel]

    soft = _gauss(heal.astype(np.float32), max(0.8, 0.003 * F))
    soft = np.clip(np.maximum(soft, heal * 0.85) * prof.strength * min(1.0, amount / 100.0) * 1.0, 0.0, 1.0)
    skin_closed = cv2.morphologyEx(skin.astype(np.float32), cv2.MORPH_CLOSE, _ellipse(max(3, int(0.08 * F) | 1)))
    soft = (soft * np.clip(skin_closed * 1.4, 0.0, 1.0))[..., None]
    healed = np.clip(low_fixed + high_fixed, 0.0, 1.0)
    out = image.copy()
    out[y0:y1, x0:x1] = crop * (1.0 - soft) + healed * soft
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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
        c = heal_spots(c, skin, faces, spot_amount, multiplier, protect_beauty_marks)
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


# ---------------------------------------------------------------------------
# 2. Stray / flyaway hair cleanup
# ---------------------------------------------------------------------------


def _scaled_polygon(points: Sequence[tuple[float, float]], w: int, h: int, sx: float, sy_up: float, sy_down: float) -> np.ndarray:
    pts = np.array(points, np.float32) * np.array([w, h], np.float32)
    c = pts.mean(axis=0)
    d = pts - c
    d[:, 0] *= sx
    d[:, 1] = np.where(d[:, 1] < 0, d[:, 1] * sy_up, d[:, 1] * sy_down)
    return np.round(c + d).astype(np.int32)


def stray_hair_roi(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    """Band around the face outline and forehead (above the mouth) where flyaways live.

    Interior features (eyes, brows, lips) and everything below the mouth (beards,
    necklines) are excluded so real hair structure is never touched.
    """
    roi = np.zeros((h, w), np.float32)
    for f in faces:
        oval = f.regions.faceOval
        if len(oval) < 3:
            continue
        outer = np.zeros((h, w), np.uint8)
        inner = np.zeros((h, w), np.uint8)
        cv2.fillPoly(outer, [_scaled_polygon(oval, w, h, 1.28, 1.45, 1.05)], 1)
        cv2.fillPoly(inner, [_scaled_polygon(oval, w, h, 0.78, 0.55, 0.8)], 1)
        band = (outer & (1 - inner)).astype(np.float32)
        fw = f.box[2] * w
        r = f.regions
        for feat in (r.leftEye, r.rightEye, r.leftBrow, r.rightBrow, r.lips):
            if len(feat) >= 3:
                fm = np.zeros((h, w), np.uint8)
                cv2.fillPoly(fm, [np.round(np.array(feat, np.float32) * [w, h]).astype(np.int32)], 1)
                band[cv2.dilate(fm, _ellipse(max(3, int(fw * 0.08)))) > 0] = 0
        mouth_y = (np.mean([p[1] for p in r.lips]) if r.lips else f.box[1] + f.box[3] * 0.75) * h
        band[int(max(0, mouth_y)):, :] = 0
        roi = np.maximum(roi, band)
    return roi


def gabor_line_response(grey: np.ndarray, width_px: float, orientations: int = 8) -> np.ndarray:
    """Max response over a bank of elongated Gabor filters tuned to strands ~width_px wide."""
    sigma = max(1.0, width_px * 1.2)
    lambd = max(3.0, width_px * 4.0)
    ksize = int(sigma * 6) | 1
    best = np.zeros_like(grey)
    for i in range(orientations):
        theta = math.pi * i / orientations
        k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma=0.25, psi=0)
        k -= k.mean()
        k /= np.abs(k).sum() + 1e-6
        best = np.maximum(best, cv2.filter2D(grey, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT))
    return best


def remove_stray_hairs(
    image: np.ndarray,
    face_landmarks_mesh: list[FaceAnalysis],
    strength: float = 60.0,
    hair_prob: np.ndarray | None = None,
) -> np.ndarray:
    """Erase thin flyaway strands crossing the face outline and forehead.

    1. ROI from the landmark mesh (band around the oval, forehead weighted, mouth and below excluded).
    2. Thin-structure candidates: black-hat (dark strands) and top-hat (light strands)
       with a kernel a few strands wide, so solid hair masses are ignored.
    3. Line-ness from an oriented Gabor bank; threshold adapts to the ROI's statistics.
    4. Keep elongated components only, then inpaint (Telea) and feather-blend.
    `hair_prob` (optional, from the segmenter) suppresses detections inside the main hair mass.
    """
    if not face_landmarks_mesh or strength <= 0:
        return image
    h, w = image.shape[:2]
    roi_full = stray_hair_roi(face_landmarks_mesh, h, w)
    if roi_full.max() <= 0:
        return image
    ys, xs = np.nonzero(roi_full)
    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = image[y0:y1, x0:x1]
    roi = roi_full[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    # Work on at most ~1600 px; strands stay several pixels wide at that scale.
    s = min(1.0, 1600.0 / max(ch, cw))
    size = (max(1, round(cw * s)), max(1, round(ch * s)))
    work = cv2.resize(crop, size, interpolation=cv2.INTER_AREA) if s < 1 else crop
    roi_w = cv2.resize(roi, size, interpolation=cv2.INTER_NEAREST) if s < 1 else roi

    face_w = max(f.box[2] * w for f in face_landmarks_mesh) * s
    strand = max(1.2, face_w * 0.0035)
    grey = _luma(work).astype(np.float32)
    k = _ellipse(int(strand * 4) | 1)
    dark = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, k)
    light = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, k)
    thin = np.maximum(dark, light * 0.8)
    line = gabor_line_response(thin, strand)

    sel = roi_w > 0.5
    if not sel.any():
        return image
    mu, sd = float(line[sel].mean()), float(line[sel].std()) + 1e-6
    sens = np.clip(strength / 100.0, 0.0, 1.0)
    thr = mu + (3.2 - 1.6 * sens) * sd
    cand = ((line > thr) & (thin > 0.02) & sel).astype(np.uint8)
    if hair_prob is not None:
        hp = hair_prob[y0:y1, x0:x1]
        hp = cv2.resize(hp, size, interpolation=cv2.INTER_LINEAR) if s < 1 else hp
        cand[hp > 0.85] = 0  # inside the hairstyle itself, not a stray

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return image
    bw, bh = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    extent = np.maximum(bw, bh)
    elongated = (extent >= strand * 6) & (area <= extent * strand * 3.5)
    keep = np.concatenate([[False], elongated])
    hair = keep[labels].astype(np.uint8)
    if not hair.any():
        return image
    hair = cv2.dilate(hair, _ellipse(max(3, int(strand * 1.5) | 1)))

    work8 = (np.clip(work, 0, 1) * 255 + 0.5).astype(np.uint8)
    filled = cv2.inpaint(work8, hair, max(3, int(strand * 3)), cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    alpha = _gauss(hair.astype(np.float32), max(0.8, strand * 0.6)) * (0.6 + 0.4 * sens)
    alpha = np.clip(alpha, 0, 1)
    if s < 1:
        filled = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
    out = image.copy()
    a = alpha[..., None]
    out[y0:y1, x0:x1] = crop * (1.0 - a) + filled * a
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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


# ---------------------------------------------------------------------------
# 4. Style profile and sync
# ---------------------------------------------------------------------------

PROFILE_LONG_EDGE = 512


def _skin_probability(c: np.ndarray) -> np.ndarray:
    cb, cr = _cbcr(c)
    return (
        _smoothstep(0.50, 0.55, cr) * (1.0 - _smoothstep(0.66, 0.71, cr))
        * _smoothstep(0.27, 0.32, cb) * (1.0 - _smoothstep(0.48, 0.53, cb))
    )


def style_profile(rgb8: np.ndarray) -> StyleProfile:
    """Metrics of a photo's look. Must match profileFromRgba in apps/web/src/engine/style.ts."""
    h, w = rgb8.shape[:2]
    s = min(1.0, PROFILE_LONG_EDGE / max(h, w))
    small = cv2.resize(rgb8, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb8
    f = small.astype(np.float32) / 255.0
    stats = color_stats(f)
    lab = rgb_to_lab(f.reshape(-1, 3))
    chroma = float(np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2).mean())
    grey = (_luma(f) * 255.0).astype(np.float32)
    sharp = float(cv2.Laplacian(grey, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REPLICATE).var())
    hp = np.abs(grey - cv2.blur(grey, (5, 5), borderType=cv2.BORDER_REPLICATE))
    skin = (_skin_probability(f) > 0.5).reshape(-1)
    if skin.sum() >= max(64, 0.01 * skin.size):
        skin_luma: float | None = float(lab[skin, 0].mean())
        skin_tex: float | None = float(hp.reshape(-1)[skin].mean())
    else:
        skin_luma = skin_tex = None
    return StyleProfile(
        stats=stats,
        meanL=float(stats.mean[0]),
        contrast=float(stats.std[0]),
        warmth=float(stats.mean[2]),
        tintAxis=float(stats.mean[1]),
        chroma=chroma,
        sharpness=sharp,
        skinLuma=skin_luma,
        skinTexture=skin_tex,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def exposure_for_lightness(from_l: float, to_l: float) -> float:
    k = ((to_l + 16) / max(from_l + 16, 1)) ** 3
    return _clamp(50 * math.log2(max(k, 1e-3)), -100, 100)


def temperature_for_warmth(db: float, L: float) -> float:
    return _clamp(100 * db / (0.15 * max(L, 10)), -100, 100)


def tint_for_magenta(da: float, L: float) -> float:
    return _clamp(100 * da / (0.078 * max(L, 10)), -100, 100)


def map_style_to_adjustments(ref: StyleProfile, target: StyleProfile, mode: str = "palette", amount: float = 100) -> dict:
    """Port of mapStyleToAdjustments (packages/shared/src/style.ts). Returns a partial adjustments dict."""
    out: dict = {}
    if ref.skinTexture is not None and target.skinTexture is not None and target.skinTexture > 0:
        ratio = ref.skinTexture / target.skinTexture
        out["skinSmoothing"] = round(_clamp((1 - ratio) * 130, 0, 85))
        out["skinTexture"] = round(_clamp(40 + ratio * 30, 35, 80))
        if ratio < 0.75:
            out["blemishRemoval"] = round(_clamp((1 - ratio) * 120, 0, 80))
        if ratio < 0.45:
            out["skinMultiplier"] = 2
    if target.sharpness > 0:
        out["sharpen"] = round(_clamp((ref.sharpness / target.sharpness - 1) * 35, 0, 60))
    amount = _clamp(amount, 0, 100)
    if mode == "palette":
        out["colorMatch"] = ColorMatch(reference=ref.stats, amount=amount).model_dump()
        return out
    L = (ref.meanL + target.meanL) / 2
    k = amount / 100
    tone = {
        "exposure": exposure_for_lightness(target.meanL, ref.meanL),
        "contrast": _clamp(((ref.contrast / max(target.contrast, 1) - 1) / 0.5) * 100, -60, 60),
        "temperature": temperature_for_warmth(ref.warmth - target.warmth, L),
        "tint": tint_for_magenta(ref.tintAxis - target.tintAxis, L),
        "saturation": _clamp((ref.chroma / max(target.chroma, 1) - 1) * 100, -80, 60),
    }
    out.update({key: round(round(v) * k) for key, v in tone.items()})
    out["colorMatch"] = None
    return out


def _read_rgb8(path: str | Path) -> np.ndarray:
    from . import imageio

    return imageio.decode(Path(path).read_bytes(), 2048)


def extract_and_sync_style(
    reference_image_path: str | Path,
    target_image_paths: Sequence[str | Path],
    mode: str = "palette",
    amount: float = 100,
) -> StyleSyncResponse:
    """Profile a reference photo and compute per-target adjustment patches ("Sync to All").

    Each patch is validated against the Adjustments schema, so it can go straight into a
    batch job's per-image `adjustments` override.
    """
    return sync_style_arrays(
        _read_rgb8(reference_image_path),
        [(Path(p).name, _read_rgb8(p)) for p in target_image_paths],
        mode,
        amount,
    )


def sync_style_arrays(
    reference_rgb8: np.ndarray,
    targets_rgb8: Sequence[tuple[str, np.ndarray]],
    mode: str = "palette",
    amount: float = 100,
) -> StyleSyncResponse:
    """In-memory variant of extract_and_sync_style (used by the HTTP endpoint)."""
    ref = style_profile(reference_rgb8)
    targets: list[StyleSyncTarget] = []
    for name, rgb in targets_rgb8:
        prof = style_profile(rgb)
        patch = map_style_to_adjustments(ref, prof, mode, amount)
        Adjustments.model_validate(patch)  # raises if the mapping ever produced an invalid value
        targets.append(StyleSyncTarget(name=name, profile=prof, adjustments=patch))
    return StyleSyncResponse(reference=ref, targets=targets)


# ---------------------------------------------------------------------------
# 5. Pro studio retouch (v0.6): glasses glare, teeth, clothing "virtual iron"
# ---------------------------------------------------------------------------
#
# All three work on a crop around their region at a face-scaled working resolution, so the
# cost does not grow with the megapixels of the camera; results are resampled and
# feather-blended back into the full-resolution image.

GLARE_WORK_FACE = 480.0  # face width (px) the glare detector works at
GLARE_MAX_VEIL = 0.75  # strongest veil the de-veiling model inverts; brighter cores are inpainted
TEETH_TARGET_B = 3.0  # CIELAB b* that whitening aims for (slightly warm; 0 or below looks blue / fake)
TEETH_MAX_L = 92.0  # whitening never lifts teeth past this L*
WRINKLE_WORK_LONG = 1400.0  # long edge of the clothing crop the wrinkle filter works at


def _face_width(faces: list[FaceAnalysis], w: int) -> float:
    return max((f.box[2] * w for f in faces), default=0.0)


def _poly_px(points: Sequence[tuple[float, float]], w: int, h: int) -> np.ndarray:
    return np.round(np.array(points, np.float32) * np.array([w, h], np.float32)).astype(np.int32)


def _bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 1e-3)
    if ys.size == 0:
        return None
    h, w = mask.shape[:2]
    return max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)


def _resize(img: np.ndarray, size: tuple[int, int], down: bool) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)


def glasses_lens_roi(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    """Where eyeglass lenses sit: an ellipse around each eye, larger than the eye itself.

    Lenses are typically 2-2.5x the width of the eye opening and reach from the brow to the
    upper cheek, so the ellipse is sized from the face width and centred slightly below the
    eye. HxW float32 in [0, 1]; empty when no face has eye landmarks.
    """
    roi = np.zeros((h, w), np.float32)
    for f in faces:
        fw = f.box[2] * w
        for eye in (f.regions.leftEye, f.regions.rightEye):
            if len(eye) < 3:
                continue
            c = np.array(eye, np.float32).mean(axis=0) * np.array([w, h], np.float32)
            centre = (int(round(c[0])), int(round(c[1] + 0.02 * fw)))
            axes = (max(2, int(0.2 * fw)), max(2, int(0.15 * fw)))
            cv2.ellipse(roi, centre, axes, 0, 0, 360, 1.0, -1)
    return roi


def _eye_mask(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for f in faces:
        for eye in (f.regions.leftEye, f.regions.rightEye):
            if len(eye) >= 3:
                cv2.fillPoly(m, [_poly_px(eye, w, h)], 1)
    return m


def detect_glare(crop: np.ndarray, roi: np.ndarray, eyes: np.ndarray, face_w: float, sensitivity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lens reflections in a working-resolution crop.

    crop: HxWx3 float32; roi: lens region; eyes: uint8 eye-opening mask; face_w in crop px.
    Returns (veil, background, glare_colour): veil is the estimated reflection opacity
    (0 = clean, ~1 = blown-out reflection), background the scene without the reflection
    (a grey morphological opening, which removes bright structures smaller than a lens
    while keeping the darker scene behind them), glare_colour the reflection's RGB.

    Catchlights (small highlights entirely inside the eye opening) and thin rim highlights
    on the frames are not glare and are left out, so eyes keep their sparkle and frames
    keep their shape.
    """
    # Pre-smooth so sensor noise neither biases the opening low nor reads as reflection.
    soft = _gauss(crop, max(0.8, 0.005 * face_w))
    y = _luma(soft).astype(np.float32)
    k = _ellipse(int(0.18 * face_w) | 1)
    bg = cv2.morphologyEx(soft, cv2.MORPH_OPEN, k)
    bg = _gauss(bg, max(0.8, 0.005 * face_w))
    bg_y = _luma(bg)
    excess = y - bg_y
    thr = 0.11 - 0.07 * sensitivity
    binary = ((excess > thr) & (roi > 0.5)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool)
    catch_area = (0.035 * face_w) ** 2
    min_side = 0.02 * face_w
    for i in range(1, n):
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if min(bw, bh) < min_side or area < (0.015 * face_w) ** 2:
            continue  # rim highlight on the frame, or noise
        comp = labels == i
        inside_eye = float(eyes[comp].mean()) if area else 0.0
        if inside_eye > 0.6 or (inside_eye > 0.3 and area < catch_area):
            continue  # catchlight, or the white of the eye: not a reflection on the lens
        keep[i] = True
    glare = keep[labels]
    zero = np.zeros(y.shape, np.float32)
    if not glare.any():
        return zero, bg, np.ones(3, np.float32)
    ex = excess[glare]
    top = glare & (excess >= np.percentile(ex, 80))
    # Flash and window reflections are close to clipping: keep the measured tint, at full brightness.
    tint = soft[top].mean(axis=0).astype(np.float32)
    colour = (tint / max(float(tint.max()), 1e-3)).astype(np.float32)
    g_y = float(colour @ LUMA)
    # Opacity from the smooth brightness lift over the scene around the reflection: the scene
    # brightness is measured outside the reflection (normalised convolution), the lift at a
    # scale coarser than eye detail, so an iris behind the veil does not read as "no veil".
    support = cv2.dilate(glare.astype(np.uint8), _ellipse(int(0.15 * face_w) | 1)).astype(np.float32)
    clear = (1.0 - support) * (roi > 0.5) + (1.0 - support) * 0.25
    ref_y = _masked_blur(y, np.clip(clear, 0, 1).astype(np.float32), max(2.0, 0.12 * face_w))
    lift_y = _gauss(cv2.morphologyEx(y, cv2.MORPH_CLOSE, _ellipse(int(0.08 * face_w) | 1)), max(1.0, 0.03 * face_w))
    veil = np.clip((lift_y - ref_y) / np.maximum(g_y - ref_y, 0.05), 0.0, 1.0) * support
    veil = cv2.morphologyEx(veil.astype(np.float32), cv2.MORPH_CLOSE, _ellipse(int(0.08 * face_w) | 1))
    veil = _gauss(veil, max(0.8, 0.02 * face_w))
    support = _gauss(support, 0.03 * face_w)
    return (veil * np.clip(support, 0, 1)).astype(np.float32), bg.astype(np.float32), colour


def reduce_glasses_glare(image: np.ndarray, faces: list[FaceAnalysis], strength: float = 60.0) -> np.ndarray:
    """Remove flash / window reflections on eyeglass lenses while keeping the eyes visible.

    Reflection model: observed = scene * (1 - a) + a * G, where G is the reflection colour
    and a its opacity. a is estimated per pixel from how far the lens is lifted above the
    scene behind it (detect_glare); where a <= GLARE_MAX_VEIL the scene is recovered
    exactly by inverting the model, which keeps the iris, lashes and skin seen through a
    hazy reflection. Blown-out cores (a > GLARE_MAX_VEIL hold too little of the scene) are
    inpainted from the recovered surroundings. strength 0..100 sets detection sensitivity
    and how much of the correction is applied.
    """
    if strength <= 0 or not faces:
        return image
    h, w = image.shape[:2]
    roi_full = glasses_lens_roi(faces, h, w)
    fw_full = _face_width(faces, w)
    box = _bbox(roi_full, int(0.05 * fw_full) + 4)
    if box is None or fw_full <= 0:
        return image
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = min(1.0, GLARE_WORK_FACE / fw_full)
    size = (max(1, round(cw * s)), max(1, round(ch * s)))
    work = _resize(crop, size, True) if s < 1 else crop
    roi = _resize(roi_full[y0:y1, x0:x1], size, True) if s < 1 else roi_full[y0:y1, x0:x1]
    eyes = _eye_mask(faces, h, w)[y0:y1, x0:x1]
    eyes = cv2.resize(eyes, size, interpolation=cv2.INTER_NEAREST) if s < 1 else eyes
    sens = float(np.clip(strength / 100.0, 0.0, 1.0))
    veil, _bg, colour = detect_glare(work, roi, eyes, fw_full * s, sens)
    if veil.max() <= 1e-3:
        return image

    a = np.minimum(veil, GLARE_MAX_VEIL)[..., None]
    scene = np.clip((work - a * colour) / (1.0 - a), 0.0, 1.0)
    core = (veil > GLARE_MAX_VEIL).astype(np.uint8)
    if core.any():
        core = cv2.dilate(core, _ellipse(3))
        radius = max(3, int(0.02 * fw_full * s))
        scene = np.stack([_inpaint_unit(scene[..., i], core, radius) for i in range(3)], axis=-1)
    # Only the correction is resampled, so the crop keeps its full-resolution detail.
    delta = (scene - work) * (0.5 + 0.5 * sens)
    alpha = np.clip(veil * 4.0, 0.0, 1.0) * np.clip(_gauss(roi, 0.01 * fw_full * s) * 1.5, 0.0, 1.0)
    if s < 1:
        delta = cv2.resize(delta, (cw, ch), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
    out = image.copy()
    out[y0:y1, x0:x1] = crop + delta * alpha[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def teeth_mask(image: np.ndarray, faces: list[FaceAnalysis]) -> np.ndarray:
    """Soft mask of visible teeth: inside the inner-mouth outline, bright and not red.

    Gums, tongue and lips are strongly red (CIELAB a* > ~20) and the gaps between teeth are
    dark, so both are excluded; what remains is enamel. HxW float32.
    """
    h, w = image.shape[:2]
    mouth = np.zeros((h, w), np.uint8)
    for f in faces:
        if len(f.regions.innerMouth) >= 3:
            cv2.fillPoly(mouth, [_poly_px(f.regions.innerMouth, w, h)], 1)
    box = _bbox(mouth, 2)
    out = np.zeros((h, w), np.float32)
    if box is None:
        return out
    y0, y1, x0, x1 = box
    lab = rgb_to_lab(image[y0:y1, x0:x1].reshape(-1, 3)).reshape(y1 - y0, x1 - x0, 3)
    sel = mouth[y0:y1, x0:x1] > 0
    l_ref = float(np.percentile(lab[..., 0][sel], 90)) if sel.any() else 70.0
    bright = _smoothstep(max(25.0, 0.55 * l_ref), max(35.0, 0.75 * l_ref), lab[..., 0])
    not_red = 1.0 - _smoothstep(12.0, 22.0, lab[..., 1])
    m = bright * not_red * sel
    fw = _face_width(faces, w)
    out[y0:y1, x0:x1] = _gauss(m.astype(np.float32), max(0.6, 0.002 * fw)) * sel
    return out


def correct_teeth(image: np.ndarray, faces: list[FaceAnalysis], whiten: float = 0.0, stain_removal: float = 0.0) -> np.ndarray:
    """Whiten teeth and even out stains while keeping enamel texture.

    Works in CIELAB on the teeth mask. Every change is computed from a smoothed (masked
    Gaussian) copy of the teeth, then added as an offset, so the fine detail (enamel
    texture, edges between teeth) is carried over untouched:
      stain_removal  pulls local yellow / dark patches toward the teeth's own average.
      whiten         lowers yellow (b*) toward TEETH_TARGET_B, takes out a little red, and
                     lifts L* by a share of the headroom to TEETH_MAX_L.
    Both 0..100.
    """
    if (whiten <= 0 and stain_removal <= 0) or not faces:
        return image
    h, w = image.shape[:2]
    mask_full = teeth_mask(image, faces)
    box = _bbox(mask_full, 4)
    if box is None:
        return image
    y0, y1, x0, x1 = box
    m = mask_full[y0:y1, x0:x1]
    crop = image[y0:y1, x0:x1]
    lab = rgb_to_lab(crop.reshape(-1, 3)).reshape(crop.shape)
    fw = _face_width(faces, w)
    enamel = (m > 0.3).astype(np.float32)
    sm = _masked_blur(lab, enamel, max(1.0, 0.03 * fw))  # broad: whole-tooth colour
    new = lab.copy()
    st = float(np.clip(stain_removal / 100.0, 0, 1))
    if st > 0 and enamel.any():
        local = _masked_blur(lab, enamel, max(0.8, 0.006 * fw))  # stain-sized, finer than a tooth
        sel = enamel > 0
        mean_b = float(np.median(local[..., 2][sel]))
        mean_l = float(np.median(local[..., 0][sel]))
        # Stains: patches yellower or darker than the teeth's own typical enamel.
        new[..., 2] -= st * np.maximum(local[..., 2] - mean_b, 0.0)
        new[..., 0] += st * 0.8 * np.maximum(mean_l - local[..., 0], 0.0)
    wt = float(np.clip(whiten / 100.0, 0, 1))
    if wt > 0:
        b_s = sm[..., 2] - (lab[..., 2] - new[..., 2])
        new[..., 2] -= wt * 0.75 * np.maximum(b_s - TEETH_TARGET_B, 0.0)
        new[..., 1] -= wt * 0.4 * np.maximum(sm[..., 1], 0.0)
        new[..., 0] += wt * 0.35 * np.maximum(TEETH_MAX_L - sm[..., 0], 0.0)
    new[..., 0] = np.minimum(new[..., 0], np.maximum(lab[..., 0], TEETH_MAX_L + 4.0))
    rgb = lab_to_rgb(new.reshape(-1, 3)).reshape(crop.shape)
    out = image.copy()
    out[y0:y1, x0:x1] = crop + (rgb - crop) * m[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def clothing_mask(
    image: np.ndarray,
    faces: list[FaceAnalysis],
    clothes_prob: np.ndarray | None = None,
    person_prob: np.ndarray | None = None,
) -> np.ndarray | None:
    """Where the garments are. HxW float32, or None when nothing can be found.

    Best: the selfie segmenter's clothes class (clothes_prob). Without the model, a portrait
    prior: person (segmenter or a head-and-shoulders shape from the face boxes) below the
    chin, minus anything skin-coloured.
    """
    h, w = image.shape[:2]
    if clothes_prob is not None:
        m = np.clip(clothes_prob.astype(np.float32), 0.0, 1.0)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m if (m > 0.5).any() else None
    if not faces:
        return None
    if person_prob is None:
        from .depth import portrait_prior

        person_prob = portrait_prior(faces, h, w)
    person = cv2.resize(person_prob.astype(np.float32), (w, h)) if person_prob.shape != (h, w) else person_prob
    chin = max((f.box[1] + f.box[3]) * h for f in faces)
    below = np.zeros((h, w), np.float32)
    below[min(h, int(chin + 0.04 * h)):, :] = 1.0
    skin = _skin_probability(_gauss(image, 2.0))
    m = person * below * (1.0 - skin)
    m = _gauss(m.astype(np.float32), 0.004 * max(h, w))
    return m if (m > 0.5).any() else None


def _log_luma(c: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(_luma(c), 1e-3)).astype(np.float32)


def remove_clothing_wrinkles(
    image: np.ndarray,
    mask: np.ndarray,
    strength: float = 60.0,
    texture_preservation: float = 80.0,
    face_width: float | None = None,
) -> np.ndarray:
    """"Virtual iron": smooth creases and folds in clothing, keep the weave and the drape.

    Wrinkles are shading. They change how much light a patch of fabric gets, not its colour,
    and they live at a scale between the fabric's threads and the garment's overall drape.
    So the filter works on log-luminance only (shading is multiplicative; chromaticity is
    kept exactly, which also means prints and dyed patterns are untouched) and splits it,
    using normalised convolution inside the garment mask so skin and background never
    bleed in, into:

      weave     finer than ~0.6% of the face width: threads, knit, twill. Kept; only lowered
                by (1 - texture_preservation) at full strength.
      creases   up to ~6% of the face width: sharp folds and crumpled creases. Attenuated by
                up to 90%.
      folds     up to ~30% of the face width: soft rolls and bunching. Attenuated by up to 45%.
      drape     everything larger: the garment's form and its big shadows. Never touched.

    Garment edges, seams and luminance-only prints (strong steps) are protected by an edge
    gate on the band amplitude, and the effect fades out near the mask border. Colour is
    rebuilt as rgb * (new luminance / old luminance).

    image: HxWx3 float32 sRGB; mask: HxW garment weight; strength and texture_preservation
    0..100; face_width (px) sets the scales, defaulting to 18% of the long edge.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    box = _bbox(mask, 8)
    if box is None:
        return image
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    m_full = np.clip(mask[y0:y1, x0:x1].astype(np.float32), 0, 1)
    ch, cw = crop.shape[:2]
    s = min(1.0, WRINKLE_WORK_LONG / max(ch, cw))
    size = (max(1, round(cw * s)), max(1, round(ch * s)))
    work = _resize(crop, size, True) if s < 1 else crop
    m = _resize(m_full, size, True) if s < 1 else m_full
    fw = (face_width if face_width and face_width > 0 else 0.18 * max(h, w)) * s

    hard = (m > 0.5).astype(np.float32)
    l = _log_luma(work)
    l_w = _masked_blur(l, hard, max(0.6, 0.006 * fw))   # weave removed
    l_c = _masked_blur(l, hard, max(1.0, 0.06 * fw))    # creases removed
    l_d = _masked_blur(l, hard, max(2.0, 0.3 * fw))     # folds removed: the drape
    weave = l - l_w
    creases = l_w - l_c
    folds = l_c - l_d

    k = float(np.clip(strength / 100.0, 0, 1))
    pres = float(np.clip(texture_preservation / 100.0, 0, 1))
    # Edges, seams and bold prints: big, abrupt steps. Wrinkle shading stays well below
    # ~0.35 log units (a 1.4x brightness change) per band.
    edge = _smoothstep(0.35, 0.6, np.abs(creases)) + _smoothstep(0.45, 0.8, np.abs(folds))
    gx = cv2.Sobel(l_w, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_w, cv2.CV_32F, 0, 1, ksize=3)
    step = _smoothstep(0.25, 0.5, np.hypot(gx, gy))
    keep = np.clip(edge + step, 0, 1)
    inner = cv2.erode(hard, _ellipse(max(3, int(0.02 * fw)) | 1))
    inner = _gauss(inner, max(1.0, 0.015 * fw))
    gate = k * inner * m * (1.0 - _gauss(keep, max(1.0, 0.01 * fw)))
    l_new = l_d + folds * (1.0 - 0.45 * gate) + creases * (1.0 - 0.9 * gate) + weave * (1.0 - gate * (1.0 - pres))
    ratio = np.exp(l_new - l)
    if s < 1:
        ratio = cv2.resize(ratio, (cw, ch), interpolation=cv2.INTER_CUBIC)
    out = image.copy()
    out[y0:y1, x0:x1] = crop * ratio[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def shift_clothing_color(image: np.ndarray, mask: np.ndarray, hue_shift: float = 0.0, saturation: float = 0.0) -> np.ndarray:
    """Recolour garments: rotate CIELAB hue by hue_shift degrees and scale chroma by
    (1 + saturation / 100). L* (and with it the fabric's shading and texture) is kept.
    Skin-coloured pixels are excluded from the mask so the edit never bleeds onto skin."""
    if hue_shift == 0 and saturation == 0:
        return image
    box = _bbox(mask, 2)
    if box is None:
        return image
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    m = np.clip(mask[y0:y1, x0:x1], 0, 1) * (1.0 - _skin_probability(_gauss(crop, 1.5)))
    lab = rgb_to_lab(crop.reshape(-1, 3)).reshape(crop.shape)
    t = math.radians(hue_shift)
    cos_t, sin_t = math.cos(t), math.sin(t)
    a, b = lab[..., 1], lab[..., 2]
    k = max(0.0, 1.0 + saturation / 100.0)
    lab[..., 1], lab[..., 2] = (a * cos_t - b * sin_t) * k, (a * sin_t + b * cos_t) * k
    rgb = lab_to_rgb(lab.reshape(-1, 3)).reshape(crop.shape)
    out = image.copy()
    out[y0:y1, x0:x1] = crop + (rgb - crop) * m[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
