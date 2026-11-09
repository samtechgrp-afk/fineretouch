"""Local spot healing: LoG detection of acne, pimples, dark spots and raised bumps, then inpainting and donor-texture cloning.

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
# 1b. Local spot healing (acne, pimples, dark spots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpotProfile:
    """How hard one multiplier step pushes spot healing (all sizes are shares of face width F)."""

    k: float  # detection threshold in robust standard deviations of the LoG response
    max_diameter: float  # largest spot treated
    grow: float  # dilation of each spot before healing (covers its halo)
    strength: float  # blend of the healed patch at amount 100
    bump_k: float = 0.0  # threshold for raised light bumps (0 = not looked for)


SPOT_PROFILES: dict[int, SpotProfile] = {
    1: SpotProfile(k=3.0, max_diameter=0.05, grow=0.004, strength=0.9, bump_k=4.5),
    2: SpotProfile(k=2.4, max_diameter=0.07, grow=0.006, strength=1.0, bump_k=3.4),
    3: SpotProfile(k=1.9, max_diameter=0.09, grow=0.008, strength=1.0, bump_k=3.0),
}
# A mark this many floors darker than the skin is a spot even below the adaptive LoG threshold.
SPOT_STRONG_CONTRAST = 3.5
# Dark patches wider than this (share of F) need an edge gradient of at least SOFT_SHADOW_EDGE
# times their peak darkness per pixel, or they are treated as shading.
SOFT_SHADOW_DIAMETER = 0.04
SOFT_SHADOW_EDGE = 0.15
# Coloured hair on the skin: hue this many degrees more yellow and chroma this much above the face.
STRAND_HUE_DEG = 3.0
STRAND_CHROMA = 1.3
# Texture for a repair is cloned in tiles of at most this share of F.
DONOR_TILE = 0.04
# A detected spot must stand out from at least this share of 8 samples on a ring around it.
SPOT_RING_SHARE = 0.75
# Raised light bumps: LoG scales, size range, local context and growth (shares of F), and how far
# the nose (tip highlight), lips (gloss) and eyes (wet lids) are kept out of the search.
BUMP_SIGMAS = (0.004, 0.0065, 0.01)
BUMP_MIN_DIAMETER, BUMP_MAX_DIAMETER = 0.006, 0.03
BUMP_CONTEXT = 0.02
BUMP_GROW = 0.008
BUMP_NOSE_GUARD, BUMP_LIP_GUARD, BUMP_EYE_GUARD = 0.12, 0.06, 0.12
# Spots are looked for at these LoG scales (sigma as a share of F; blob radius ~ sigma * sqrt(2)).
SPOT_SIGMAS = (0.0065, 0.01, 0.015, 0.022)
# Beauty marks: much darker than the skin around them, not red, and big enough to be deliberate.
MOLE_MIN_DARKNESS = 0.2
MOLE_MAX_REDNESS = 0.015
MOLE_MIN_DIAMETER = 0.012
# Detection runs with the face about this many pixels wide; pores vanish below the finest scale.
SPOT_WORK_FACE = 420.0
# Acne vs dark spot: a spot is acne when its redness (Cr above the skin) passes this and carries
# at least ACNE_RED_SHARE of its darkness; otherwise it is a pigmented / dark mark.
ACNE_MIN_REDNESS = 0.008
ACNE_RED_SHARE = 0.35
# Freckles (kept by Dark spot removal when "Keep freckles" is on): small and low-contrast.
FRECKLE_MAX_DIAMETER = 0.02
FRECKLE_MAX_DARKNESS = 0.09
SPOT_KINDS = ("all", "acne", "dark")


@dataclass
class SpotDetection:
    """Result of detect_spots, in the coordinates of the image passed in."""

    heal: np.ndarray  # uint8 HxW: 1 where a spot (plus its halo) will be healed
    protected: np.ndarray  # uint8 HxW: beauty marks that were found and kept
    count: int
    protected_count: int
    face_width: float


def faces_in_crop(faces: list[FaceAnalysis], x0: int, y0: int, cw: int, ch: int, w: int, h: int) -> list[FaceAnalysis]:
    """Re-normalise faces from a w x h image to its crop at (x0, y0) of size cw x ch."""
    if (x0, y0, cw, ch) == (0, 0, w, h):
        return faces

    def pt(p: tuple[float, float]) -> tuple[float, float]:
        return ((p[0] * w - x0) / cw, (p[1] * h - y0) / ch)

    out = []
    for f in faces:
        bx, by, bw, bh = f.box
        regions = type(f.regions)(**{k: [pt(p) for p in v] for k, v in f.regions.model_dump().items()})
        out.append(f.model_copy(update={
            "box": ((bx * w - x0) / cw, (by * h - y0) / ch, bw * w / cw, bh * h / ch), "regions": regions,
        }))
    return out


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
    kind: str = "all",
    keep_freckles: bool = False,
) -> SpotDetection:
    """Find acne, pimples and dark spots inside the skin mask.

    kind: "all" (general blemish removal), "acne" (red spots and raised bumps only) or "dark"
    (dark / pigmented marks that are not red). keep_freckles leaves small, low-contrast dark
    spots alone for "dark" and "all".

    Signal: darkness and redness relative to the surrounding skin, so a red pimple and a
    brown spot both read as bright blobs. Blobs are found with a multi-scale normalised
    LoG; the threshold adapts to each photo through the response's robust spread on skin
    (a clear complexion and a grainy high-ISO frame get the same false-positive rate).
    Pores sit below the finest scale. Lines (wrinkles, lash lines, hair), large shading
    and anything on or next to the eyes, brows, lips and nostrils are rejected.
    """
    h, w = image.shape[:2]
    F = face_width_px(faces, mask, w)
    ys, xs = np.nonzero(mask > 0.02)
    empty = np.zeros((h, w), np.uint8)
    if ys.size == 0:
        return SpotDetection(empty, empty.copy(), 0, 0, F)
    # Work on the skin's bounding box only (a full-length frame is mostly not face).
    pad = int(0.15 * F) + 8
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    if (y1 - y0) * (x1 - x0) < 0.8 * h * w:
        local = faces_in_crop(faces, x0, y0, x1 - x0, y1 - y0, w, h)
        d = _detect_spots(image[y0:y1, x0:x1], mask[y0:y1, x0:x1], local, multiplier, amount, protect_beauty_marks, F,
                          kind, keep_freckles)
        heal, moles = empty, empty.copy()
        heal[y0:y1, x0:x1] = d.heal
        moles[y0:y1, x0:x1] = d.protected
        return SpotDetection(heal, moles, d.count, d.protected_count, F)
    return _detect_spots(image, mask, faces, multiplier, amount, protect_beauty_marks, F, kind, keep_freckles)


def _ring_isotropy(y: np.ndarray, labels: np.ndarray, stats: np.ndarray, centroids: np.ndarray,
                   ids: np.ndarray, sign: float) -> np.ndarray:
    """Share of 8 ring samples around each component that it stands out from (sign=+1: darker).

    A spot differs from the skin on every side; a shading edge (side of the nose, a crease)
    only on one, so its ring share stays near one half.
    """
    h, w = y.shape
    out = np.zeros(len(ids), np.float32)
    flat = labels.ravel()
    counts = np.maximum(np.bincount(flat, minlength=stats.shape[0]), 1)
    mean_y = np.bincount(flat, weights=y.ravel(), minlength=stats.shape[0]) / counts
    for j, i in enumerate(ids):
        lbl = i + 1
        cx, cy = centroids[lbl]
        r = 0.5 * max(stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT])
        rad = 1.8 * r + 2.0
        diffs = []
        for t in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
            px, py = int(round(cx + rad * math.cos(t))), int(round(cy + rad * math.sin(t)))
            if 0 <= px < w and 0 <= py < h:
                diffs.append(sign * (float(y[py, px]) - float(mean_y[lbl])))
        if not diffs:
            continue
        diffs = np.array(diffs)
        peak = max(float(diffs.max()), 1e-4)
        out[j] = float((diffs > 0.3 * peak).mean())
    return out


def _detect_spots(
    image: np.ndarray, mask: np.ndarray, faces: list[FaceAnalysis], multiplier: int, amount: float,
    protect_beauty_marks: bool, F: float, kind: str = "all", keep_freckles: bool = False,
) -> SpotDetection:
    if kind not in SPOT_KINDS:
        raise ValueError(f"kind must be one of {SPOT_KINDS}")
    prof = SPOT_PROFILES.get(int(multiplier), SPOT_PROFILES[1])
    h, w = image.shape[:2]
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
    core &= ~_coloured_strands(work, core, Fw)

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
    # Floor: below ~2.5% contrast on light skin a "spot" is invisible and only texture would be
    # lost. Contrast scales with skin luma, so deeper complexions get a proportionally lower floor.
    skin_y = float(np.median(base_y[core]))
    floor = 0.025 * (1.3 - 0.15 * (int(multiplier) - 1)) * float(np.clip(skin_y / 0.5, 0.5, 1.0))
    thr = med + k * spread
    # On busy, glossy skin the LoG spread is wide; a small but clearly dark mark still counts.
    strong = (signal > SPOT_STRONG_CONTRAST * floor) & (resp > 0.6 * thr)
    cand = (((resp > thr) | strong) & (resp > floor) & core & (signal > 0.6 * floor)).astype(np.uint8)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    # n may be 1 (no dark spots): the arrays below are then empty and raised bumps still run.
    bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    side = np.maximum(bw, bh)
    diameter = np.sqrt(4.0 * area / math.pi)
    # Pores and single-pixel noise stay; a clearly dark mark may be smaller.
    flat = labels.ravel()
    peak_sig = np.zeros(n - 1, np.float32)
    np.maximum.at(peak_sig, flat[flat > 0] - 1, signal.ravel()[flat > 0])
    min_d = np.where(peak_sig > SPOT_STRONG_CONTRAST * floor, max(2.0, 0.005 * Fw), max(2.0, 0.008 * Fw))
    blob = (
        (diameter <= prof.max_diameter * Fw) & (diameter >= min_d)
        & (area / (bw * bh) >= 0.4) & (np.minimum(bw, bh) / side >= 0.45)
    )

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
    # Shading edges (side of the nose, smile line) pass the blob test but not the ring test.
    cand_ids = np.nonzero(heal_ids)[0]
    if cand_ids.size:
        ys1 = _gauss(y, 1.0)
        iso = _ring_isotropy(ys1, labels, stats, cents, cand_ids, 1.0)
        heal_ids[cand_ids[iso < SPOT_RING_SHARE]] = False
        # Big, soft-edged dark patches are shading hollows (inner eye corner, beside the nose),
        # not marks: a spot's edge is sharp whatever its size.
        grad = np.hypot(cv2.Sobel(ys1, cv2.CV_32F, 1, 0), cv2.Sobel(ys1, cv2.CV_32F, 0, 1)) / 8.0
        for i in cand_ids[diameter[cand_ids] > SOFT_SHADOW_DIAMETER * Fw]:
            sel = labels == i + 1
            if np.percentile(grad[sel], 90) < SOFT_SHADOW_EDGE * max(float(peak_dark[i]), 1e-4):
                heal_ids[i] = False

    # Separate tools: acne (red) vs dark / pigmented marks; freckles optionally kept.
    acne = (mean_red >= ACNE_MIN_REDNESS) & (1.5 * mean_red >= ACNE_RED_SHARE * np.maximum(mean_dark, 1e-4))
    if kind == "acne":
        heal_ids &= acne
    elif kind == "dark":
        heal_ids &= ~acne
    if keep_freckles and kind != "acne":
        heal_ids &= ~((diameter < FRECKLE_MAX_DIAMETER * Fw) & (peak_dark < FRECKLE_MAX_DARKNESS) & ~acne)

    lut_heal = np.concatenate([[0], heal_ids]).astype(np.uint8)
    lut_mole = np.concatenate([[0], mole]).astype(np.uint8)
    heal = lut_heal[labels]
    moles = lut_mole[labels]
    look_bumps = prof.bump_k > 0 and kind != "dark"
    bumps, n_bumps = _detect_bumps(y, core, protect, faces, Fw, prof, a, floor) if look_bumps else (None, 0)
    if bumps is not None:
        heal = np.maximum(heal, bumps)
    grow = max(3, round(prof.grow * Fw * 2) | 1)
    heal = cv2.dilate(heal, _ellipse(grow))
    heal[cv2.dilate(moles, _ellipse(grow + 2)) > 0] = 0
    heal[protect > 0] = 0
    if s < 1:
        heal = (cv2.resize(heal.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.35).astype(np.uint8)
        moles = (cv2.resize(moles.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.35).astype(np.uint8)
    return SpotDetection(heal, moles, int(heal_ids.sum()) + n_bumps, int(mole.sum()), F)


def _coloured_strands(work: np.ndarray, core: np.ndarray, Fw: float) -> np.ndarray:
    """Dyed hair, braids and extensions lying on the skin: more saturated and more yellow than
    the face (hue turned toward b*). Spots go the other way (redder or less saturated), so they
    are never caught. The mask's hole-closing would otherwise pull thin strands back in."""
    lab = cv2.cvtColor(np.ascontiguousarray(_gauss(work, 1.0), np.float32), cv2.COLOR_RGB2Lab)
    ch = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    # Compared with the skin around each pixel: shading changes chroma a lot across a face.
    cm = core.astype(np.float32)
    ch0 = _masked_blur(ch, cm, 0.06 * Fw)
    hue0 = _masked_blur(hue, cm, 0.06 * Fw)
    strand = ((hue > hue0 + STRAND_HUE_DEG) & (ch > ch0 * STRAND_CHROMA)).astype(np.uint8)
    strand = cv2.morphologyEx(strand, cv2.MORPH_OPEN, _ellipse(3))
    return cv2.dilate(strand, _ellipse(max(3, int(0.02 * Fw) | 1))) > 0


def _detect_bumps(
    y: np.ndarray, core: np.ndarray, protect: np.ndarray, faces: list[FaceAnalysis], Fw: float,
    prof: SpotProfile, a: float, floor: float,
) -> tuple[np.ndarray, int]:
    """Raised bumps (closed comedones, small papules) that read lighter than the skin around them.

    On deeper skin tones and under flash these show as tiny bright domes rather than red or
    dark marks, so the darkness/redness signal never sees them. They are much smaller than
    a specular patch and stand out on every side; highlights on the nose, lips and lids are
    excluded by a wider feature guard.
    """
    h, w = y.shape
    local = _masked_blur(y, core.astype(np.float32), BUMP_CONTEXT * Fw)
    bright = cv2.medianBlur((y - local).astype(np.float32), 3)
    resp, _ = _log_response(bright, [max(0.7, f * Fw) for f in BUMP_SIGMAS])
    guard = np.zeros((h, w), np.uint8)
    for f in faces:
        r = f.regions
        for region, grow in ((r.nose, BUMP_NOSE_GUARD), (r.lips, BUMP_LIP_GUARD),
                             (r.leftEye, BUMP_EYE_GUARD), (r.rightEye, BUMP_EYE_GUARD)):
            if len(region) < 3:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(region, np.float32) * [w, h]).astype(np.int32)], 1)
            guard |= cv2.dilate(m, _ellipse(max(3, int(Fw * grow) | 1)))
    zone = core & (guard == 0)
    empty = np.zeros((h, w), np.uint8)
    if zone.sum() < 64:
        return empty, 0
    vals = resp[zone]
    med = float(np.median(vals))
    spread = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-5
    k = prof.bump_k * (1.25 - 0.25 * a)
    cand = ((resp > med + k * spread) & zone & (bright > floor)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return empty, 0
    bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    diameter = np.sqrt(4.0 * area / math.pi)
    ok = ((diameter >= max(2.0, BUMP_MIN_DIAMETER * Fw)) & (diameter <= BUMP_MAX_DIAMETER * Fw)
          & (area / (bw * bh) >= 0.4) & (np.minimum(bw, bh) / np.maximum(bw, bh) >= 0.45))
    ids = np.nonzero(ok)[0]
    if ids.size:
        iso = _ring_isotropy(_gauss(y, 1.0), labels, stats, cents, ids, -1.0)
        ok[ids[iso < SPOT_RING_SHARE]] = False
    out = np.concatenate([[0], ok]).astype(np.uint8)[labels]
    # A bump's dome and its shadowed rim both go: grow by about its own radius.
    out = cv2.dilate(out, _ellipse(max(3, round(BUMP_GROW * Fw) | 1)))
    return out, int(ok.sum())


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
        for dist in (distance, int(distance * 1.6), int(distance * 2.4)):
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
    # Donors: plain skin with no spot nearby. On densely spotted skin (forehead comedones) that
    # can leave no candidate at all, so a second pass only asks the patch itself to be clean.
    ok = (skin > 0.7) & (cv2.dilate(heal, _ellipse(5)) == 0)
    ok_loose = (skin > 0.5) & (heal == 0)
    # Last resort, in dense clusters or where the skin mask feathers out (eyes, hairline): any
    # skin at all, even if it holds another spot, so a repair never ends up a grain-free smear.
    ok_any = skin > 0.15
    n, labels, stats, _ = cv2.connectedComponentsWithStats(heal, connectivity=8)
    # Long or merged repairs (a run of comedones along a crease) are re-textured in tiles: one
    # clean donor the size of the whole run rarely exists, and without one the run turns flat.
    tile = max(6, round(DONOR_TILE * F))
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        for ty in range(y, y + bh, tile):
            for tx in range(x, x + bw, tile):
                th, tw = min(tile, y + bh - ty), min(tile, x + bw - tx)
                sel = labels[ty:ty + th, tx:tx + tw] == i
                if not sel.any():
                    continue
                box = (tx, ty, tw, th)
                dist = max(4, int(max(tw, th) * 1.4))
                off = None
                for allowed in (ok, ok_loose, ok_any):
                    off = _best_donor(box, low_y, allowed, dist)
                    if off is not None:
                        break
                if off is None:
                    continue
                dy, dx = off
                donor = high[ty + dy:ty + dy + th, tx + dx:tx + dx + tw]
                patch = high_fixed[ty:ty + th, tx:tx + tw]
                patch[sel] = donor[sel]

    soft = _gauss(heal.astype(np.float32), max(0.8, 0.003 * F))
    soft = np.clip(np.maximum(soft, heal * 0.85) * prof.strength * min(1.0, amount / 100.0) * 1.0, 0.0, 1.0)
    skin_closed = cv2.morphologyEx(skin.astype(np.float32), cv2.MORPH_CLOSE, _ellipse(max(3, int(0.08 * F) | 1)))
    soft = (soft * np.clip(skin_closed * 1.4, 0.0, 1.0))[..., None]
    healed = np.clip(low_fixed + high_fixed, 0.0, 1.0)
    out = image.copy()
    out[y0:y1, x0:x1] = crop * (1.0 - soft) + healed * soft
    return np.clip(out, 0.0, 1.0).astype(np.float32)
