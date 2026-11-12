"""Professional Skin Finish, phase 2: local corrections before the texture finish (engines/finish.py).

Each correction works on CIELAB of the skin crop, changes only the scales it is about, and leaves
the finer texture on top of it, so pores and grain come through unchanged. Every scale is a share
of the face width F. All of them are 0 at amount 0 (the photo is then untouched by this module).

  shine        Specular sheen: where the skin is brighter than its own wide base by more than
               SHINE_MARGIN L* at the bump scale, the excess is reduced (by up to SHINE_MAX) and
               the colour the glare washed out comes back. Highlights keep their place and shape.
  dark spots   Post-acne marks, dark spots and small hyperpigmentation: roughly round patches
               darker than their surround at three scales (Hessian test: both curvatures
               positive and of similar size, so creases, folds and contour shadows are not
               spots). Their depth is lifted and their colour moved to the surround's. Very deep
               cores (moles, nostrils, lash shadow, fold shadows) and their surroundings are left alone.
  tone         Uneven pigmentation: colour (a*, b*) variation between the bump and patch scales
               is reduced; lightness only where the colour also deviates (a pigment patch, not
               the lighting); lightness is only lifted (bright patches are shine's job).
  under-eye    The under-eye shadow is lifted as a whole toward the person's own surrounding
               skin (see engines/under_eye.correct_dark_circles) and fine crepey lines softened;
               pores and part of the socket's shadow stay.
  dodge/burn   Light and dark irregularities between the blotch and form scales are evened
               (darks lifted more than lights lowered), away from contours. The face's own light
               shape (larger scales) is never changed: amplifying it read as harder shadows
               beside the nose in testing, against the brief to keep contours.

Near real contours (contour_weight) tone and dodge and burn fade out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..schemas import SkinFinishSettings

SHINE_MARGIN, SHINE_MAX, SHINE_COLOUR = 3.0, 0.85, 0.5
SHINE_BASE, SHINE_LIT = 0.08, 0.35  # base scale (x F); weight kept where colour is not washed out
SPOT_SCALES = (0.006, 0.014, 0.024, 0.04)  # x F: small marks to small patches
SPOT_MIN, SPOT_FULL = 1.0, 2.5  # L* depth (at the patch scale) below which nothing is a mark
SPOT_FINE = 0.003  # x F: finer than this is pores and grain, never lifted
SPOT_DEEP0, SPOT_DEEP1 = 14.0, 20.0  # deeper than this is a mole / nostril / lash shadow
SPOT_DEEPR0, SPOT_DEEPR1 = 0.35, 0.45  # or this share of the surround's L* (deep skin)
SPOT_ISO0, SPOT_ISO1 = 0.25, 0.5  # smaller / larger curvature: round enough to be a spot
# Gradient coherence above which a patch is a line, per scale: strict at the crease scales,
# loose for larger patches (pigmentation is often irregular or elongated, a crease is thin).
SPOT_COH = ((0.25, 0.5), (0.25, 0.5), (0.45, 0.7), (0.45, 0.7))
SPOT_LIFT, SPOT_COLOUR = 0.9, 0.8
SPOT_CORE_REACH = 0.012  # x F: the neighbourhood of a deep core left alone
SPOT_SURROUND = 4.0  # surround sigma, in patch sigmas
SPOT_OWN = 0.35  # a patch's own detail sigma, in patch sigmas (finer is another scale's job)
SPOT_PX = 3.0  # pixels per sigma the detector works at
SPOT_FREE_SHAPE = 0.024  # x F: patches this large may be any shape away from the folds
SPOT_PASSES = 2  # a second pass takes what the first one's measure of the surround missed
TONE_LO, TONE_HI = 0.014, 0.10  # x F: the band of uneven colour
TONE_COLOUR, TONE_LIGHT = 0.75, 0.4
UE_SHADOW, UE_REF, UE_LIFT, UE_COLOUR, UE_LINES = 0.02, 0.06, 0.8, 0.6, 0.5
DB_MICRO, DB_FORM = 0.012, 0.05
DB_DARK, DB_LIGHT = 0.6, 0.4
DB_RANGE = 0.2  # share of the skin's L*: steps above it are contours, below it unevenness
CONTOUR_S, CONTOUR_LO, CONTOUR_HI = 0.012, 0.3, 0.45  # x F; step as a share of the skin's L*
CONTOUR_STRICT = (0.45, 0.6)  # with landmark folds known: only nostril / lip-line steps
CONTOUR_LEN, CONTOUR_COH, CONTOUR_REACH = 0.04, (0.3, 0.6), 0.015  # x F; coherence; x F


@dataclass
class LocalReport:
    """Before / after measures of each correction, on the finish mask (L* units)."""

    measures: dict = field(default_factory=dict)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    from .finish import _gauss as g

    return g(img, sigma)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def amount(ctl) -> float:
    return ctl.amount / 100.0 if ctl.enabled else 0.0


def any_local(s: SkinFinishSettings) -> bool:
    return any(amount(c) > 0 for c in (s.shine, s.darkSpots, s.toneEvenness, s.underEye, s.dodgeBurn))


# --- shine ------------------------------------------------------------------------------------

def shine_excess(lab: np.ndarray, F: float) -> np.ndarray:
    """L* by which the skin's sheen (bump scale) rises above its wide base, past SHINE_MARGIN,
    weighted by how much the glare washed the colour out (a specular desaturates; skin that is
    simply better lit keeps its chroma and keeps SHINE_LIT of the weight)."""
    L = np.ascontiguousarray(lab[..., 0])
    s_b, s_w = max(0.8, 0.014 * F), SHINE_BASE * F
    sheen = _gauss(L, s_b) - _gauss(L, s_w)
    c_loc = np.hypot(_gauss(np.ascontiguousarray(lab[..., 1]), s_b), _gauss(np.ascontiguousarray(lab[..., 2]), s_b))
    c_base = np.hypot(_gauss(np.ascontiguousarray(lab[..., 1]), s_w), _gauss(np.ascontiguousarray(lab[..., 2]), s_w))
    washed = 1.0 - _smoothstep(0.75, 0.97, c_loc / np.maximum(c_base, 1e-3))
    return np.clip(sheen - SHINE_MARGIN, 0.0, None) * (SHINE_LIT + (1.0 - SHINE_LIT) * washed)


def reduce_shine(lab: np.ndarray, mask: np.ndarray, F: float, t: float) -> None:
    ex = shine_excess(lab, F)
    w = _smoothstep(0.0, 6.0, ex) * mask * t
    lab[..., 0] -= SHINE_MAX * ex * mask * t
    for c in (1, 2):
        A = lab[..., c]
        A += (_gauss(A, SHINE_BASE * F) - _gauss(A, max(0.8, 0.014 * F))) * SHINE_COLOUR * w


# --- dark spots -------------------------------------------------------------------------------

def spot_map(L: np.ndarray, F: float, prior: list | None = None, folds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list]:
    """(depth to lift in L*, index of the scale it was found at, per-scale weights) for round dark
    patches. `prior`: the weights of an earlier pass, kept, so a mark found once is lifted again by
    its remaining depth (which may be too shallow to be found on its own).

    Each scale is detected on a copy shrunk to ~SPOT_PX pixels per sigma. The depth that is lifted
    is the surround minus the pore-free skin (SPOT_FINE), not the blurred patch, so a small mark
    is lifted by its real depth and the lift follows its own outline, while pores stay."""
    h, w = L.shape
    fine = _gauss(L, max(0.6, SPOT_FINE * F))
    # Deep cores (nostril, mole, lash shadow, a fold's shadow) at full size and their real depth,
    # absolute or relative to the skin around (on deep skin a mole is less L* below it).
    sur0 = _gauss(L, SPOT_SURROUND * max(1.0, SPOT_SCALES[0] * F))
    true0 = sur0 - fine
    core = np.maximum(_smoothstep(SPOT_DEEP0, SPOT_DEEP1, true0),
                      _smoothstep(SPOT_DEEPR0, SPOT_DEEPR1, true0 / np.maximum(sur0, 1.0))).astype(np.float32)
    best = np.zeros_like(L)
    idx = np.zeros(L.shape, np.int8)
    weights = []
    for i, share in enumerate(SPOT_SCALES):
        s_full = max(1.0, share * F)
        f = max(1.0, s_full / SPOT_PX)
        size = (max(1, round(w / f)), max(1, round(h / f)))
        small = L if f == 1.0 else cv2.resize(L, size, interpolation=cv2.INTER_AREA)
        s = s_full / f
        g = cv2.GaussianBlur(small, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        sur = cv2.GaussianBlur(small, (0, 0), SPOT_SURROUND * s, borderType=cv2.BORDER_REFLECT)
        depth = sur - g
        lxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        lyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        lxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        half = 0.5 * (lxx + lyy)
        rad = np.sqrt(0.25 * (lxx - lyy) ** 2 + lxy * lxy)
        lmax, lmin = half + rad, half - rad
        iso = _smoothstep(SPOT_ISO0, SPOT_ISO1, lmin / np.maximum(lmax, 1e-6)) * (lmin > 0)
        # Round, not the end of a line: the gradients around a spot point every way, those
        # around a crease's end mostly across the crease (structure-tensor coherence).
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0 * s)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0 * s)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0 * s)
        coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1e-6)
        iso = iso * (1.0 - _smoothstep(*SPOT_COH[i], coh))
        if folds is not None and share >= SPOT_FREE_SHAPE:
            # Away from the nose and its folds (landmarks), a larger dark patch is pigment
            # whatever its shape: the face's own elongated shadows are in the fold mask.
            fr = folds if f == 1.0 else cv2.resize(folds, size, interpolation=cv2.INTER_AREA)
            iso = np.maximum(iso, (1.0 - np.clip(fr, 0.0, 1.0)) * (depth > 0))
        # The tests hold at the patch's centre; spread them over the patch.
        r = max(1, int(round(1.5 * s)))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
        wgt = (iso * _smoothstep(SPOT_MIN, SPOT_FULL, depth)).astype(np.float32)
        wgt = cv2.GaussianBlur(cv2.dilate(wgt, disk), (0, 0), 0.5 * s)
        # A deep core and its whole neighbourhood stay as they are: lifting the rim around a kept
        # core would only sharpen its outline. (Shrunk by area, so a core never disappears.)
        deep = core if f == 1.0 else np.clip(cv2.resize(core, size, interpolation=cv2.INTER_AREA) * f * f, 0.0, 1.0)
        r2 = max(1, int(round(max(SPOT_CORE_REACH * F, 1.5 * s_full) / f)))
        deep = cv2.GaussianBlur(cv2.dilate(deep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1,) * 2)), (0, 0), 0.5 * r2)
        wgt = (wgt * (1.0 - np.clip(deep, 0.0, 1.0))).astype(np.float32)
        # What is lifted at this scale: the surround minus the skin at SPOT_OWN of the scale, so
        # a larger patch does not also take a thin crease or finer marks running through it
        # (those are each judged at their own scale). The smallest scale uses the pore-free skin.
        own = SPOT_OWN * s
        deficit = None if i == 0 else sur - cv2.GaussianBlur(small, (0, 0), own, borderType=cv2.BORDER_REFLECT)
        if f != 1.0:
            wgt = cv2.resize(wgt, (w, h), interpolation=cv2.INTER_LINEAR)
            sur = cv2.resize(sur, (w, h), interpolation=cv2.INTER_LINEAR)
            if deficit is not None:
                deficit = cv2.resize(deficit, (w, h), interpolation=cv2.INTER_LINEAR)
        if prior is not None:
            wgt = np.maximum(wgt, prior[i])
        weights.append(wgt)
        lift = np.clip(sur - fine if deficit is None else deficit, 0.0, None) * wgt
        better = lift > best
        best = np.where(better, lift, best)
        idx = np.where(better, np.int8(i), idx)
    return _gauss(best, max(0.6, 0.5 * SPOT_FINE * F)), idx, weights


def correct_spots(lab: np.ndarray, mask: np.ndarray, F: float, t: float, folds: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Lifts dark patches; returns the share of the mask that was corrected and the first pass's
    depth map (the marks as found on the input, for the before measure)."""
    k = mask * t
    share, prior, first = 0.0, None, None
    for _ in range(SPOT_PASSES):
        sh, prior, lift = _spot_pass(lab, mask, F, k, prior, folds)
        share = max(share, sh)
        first = lift if first is None else first
    return share, first


def _spot_pass(lab: np.ndarray, mask: np.ndarray, F: float, k: np.ndarray, prior: list | None,
               folds: np.ndarray | None) -> tuple[float, list]:
    L = lab[..., 0]
    lift, idx, weights = spot_map(L, F, prior, folds)
    for c in (1, 2):
        A = lab[..., c]
        dA = np.zeros_like(A)
        for i, share in enumerate(SPOT_SCALES):
            s = max(1.0, share * F)
            sel = idx == i
            if sel.any():
                dA[sel] = (_gauss(A, SPOT_SURROUND * s) - _gauss(A, s))[sel]
        frac = np.clip(lift / 3.0, 0.0, 1.0)  # colour follows only where a real mark was lifted
        A += dA * frac * SPOT_COLOUR * k
    L += lift * SPOT_LIFT * k
    sel = mask > 0.5
    return float(((lift > 1.0) & sel).sum()) / max(1, int(sel.sum())), weights, lift


# --- contours -----------------------------------------------------------------------------------

def contour_weight(L: np.ndarray, F: float, strict: bool = False) -> np.ndarray:
    """1 near a real contour (the side of the nose, a nostril, the jaw's shadow), 0 on skin with
    marks and blotches. A contour is a long, steep, straight-ish step: across ~2.5 sigma at
    CONTOUR_S it falls by a large share of the skin's own L* (CONTOUR_LO..HI), and the gradients
    around it all point the same way (structure-tensor coherence over CONTOUR_LEN, CONTOUR_COH);
    a mark or blotch is a closed outline whose gradients point every way. The weight is widened
    by CONTOUR_REACH, so the shading next to a contour stays as it is."""
    s = max(1.0, CONTOUR_S * F)
    g = _gauss(L, s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    step = np.sqrt(gx * gx + gy * gy) * (2.5 * s) / np.maximum(_gauss(L, DB_FORM * F), 15.0)
    ro = max(1.0, CONTOUR_LEN * F)
    jxx, jyy, jxy = _gauss(gx * gx, ro), _gauss(gy * gy, ro), _gauss(gx * gy, ro)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1e-9)
    lo, hi = (CONTOUR_STRICT if strict else (CONTOUR_LO, CONTOUR_HI))
    e = (_smoothstep(lo, hi, step) * _smoothstep(*CONTOUR_COH, coh)).astype(np.float32)
    r = max(1, int(round(CONTOUR_REACH * F)))
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    return np.clip(_gauss(e, 0.5 * r), 0.0, 1.0)


# --- tone evenness ----------------------------------------------------------------------------

def even_tone(lab: np.ndarray, mask: np.ndarray, F: float, t: float, edge: np.ndarray) -> None:
    lo, hi = max(0.8, TONE_LO * F), TONE_HI * F
    k = mask * t * (1.0 - edge)
    dev = np.zeros(lab.shape[:2], np.float32)
    for c in (1, 2):
        A = lab[..., c]
        band = _gauss(A, lo) - _gauss(A, hi)
        dev += band * band
        A -= band * TONE_COLOUR * k
    L = lab[..., 0]
    w = _smoothstep(1.0, 3.0, np.sqrt(dev))
    # Lightness only lifts darker pigment patches; bright patches are the shine control's job.
    L -= np.minimum(_gauss(L, lo) - _gauss(L, hi), 0.0) * TONE_LIGHT * w * k


# --- under-eye --------------------------------------------------------------------------------

def refine_under_eye(lab: np.ndarray, mask: np.ndarray, under: np.ndarray, F: float, t: float) -> None:
    if not (under > 1e-3).any():
        return
    ref_w = (mask * (1.0 - _smoothstep(0.05, 0.3, under))).astype(np.float32)
    den = _gauss(ref_w, UE_REF * F)
    ok = (den > 0.02).astype(np.float32)
    d = t * under * ok
    L = lab[..., 0]
    shadow = _gauss(L, UE_SHADOW * F)
    refL = _gauss(L * ref_w, UE_REF * F) / np.maximum(den, 0.02)
    # Fine crepey lines (the bump band) softened; pores (finer) untouched.
    lines = _gauss(L, max(0.8, 0.0045 * F)) - _gauss(L, max(1.0, 0.014 * F))
    for c in (1, 2):
        A = lab[..., c]
        refA = _gauss(A * ref_w, UE_REF * F) / np.maximum(den, 0.02)
        A += (refA - _gauss(A, UE_SHADOW * F)) * UE_COLOUR * d
    L += np.maximum(0.0, refL - shadow) * UE_LIFT * d - np.minimum(lines, 0.0) * UE_LINES * d


def under_eye_deficit(L: np.ndarray, mask: np.ndarray, under: np.ndarray, F: float) -> float | None:
    sel = under > 0.6
    if not sel.any():
        return None
    ring = (mask > 0.5) & (under < 0.05) & (_gauss((under > 0.05).astype(np.float32), 0.05 * F) > 0.05)
    if not ring.any():
        return None
    s = _gauss(L, UE_SHADOW * F)
    return float(s[ring].mean() - s[sel].mean())


# --- dodge and burn ---------------------------------------------------------------------------

def _db_bands(L: np.ndarray, mask: np.ndarray, F: float) -> np.ndarray:
    """The micro band (DB_MICRO..DB_FORM) of edge-preserving low-passes: a step larger than
    DB_RANGE of the skin's L* (the side of the nose, a jaw shadow, a nostril) stays in both
    low-passes and so is not in the band; marks, blotches and uneven light are."""
    from .finish import _guided

    sel = mask > 0.5
    ref = float(np.median(L[sel])) if sel.sum() > 64 else float(np.median(L))
    eps = (DB_RANGE * max(ref, 15.0)) ** 2
    return _guided(L, DB_MICRO * F, eps) - _guided(L, DB_FORM * F, eps)


def dodge_burn(lab: np.ndarray, mask: np.ndarray, F: float, t: float, edge: np.ndarray) -> None:
    L = lab[..., 0]
    micro = _db_bands(L, mask, F)
    gain = np.where(micro < 0, DB_DARK, DB_LIGHT)
    L -= micro * gain * (1.0 - edge) * mask * t


def micro_rms(L: np.ndarray, sel: np.ndarray, F: float) -> float:
    m = _gauss(L, DB_MICRO * F) - _gauss(L, DB_FORM * F)
    return float(np.sqrt(np.mean(m[sel] ** 2))) if sel.any() else 0.0


def tone_rms(lab: np.ndarray, sel: np.ndarray, F: float) -> float:
    lo, hi = max(0.8, TONE_LO * F), TONE_HI * F
    acc = 0.0
    for c in (1, 2):
        A = np.ascontiguousarray(lab[..., c])
        b = _gauss(A, lo) - _gauss(A, hi)
        acc += float(np.mean(b[sel] ** 2)) if sel.any() else 0.0
    return math.sqrt(acc)


# --- all of them --------------------------------------------------------------------------------

def apply_local(
    lab: np.ndarray, mask: np.ndarray, spot_mask: np.ndarray, under: np.ndarray | None, F: float,
    s: SkinFinishSettings, k: float, edge_blotch: np.ndarray, measure: bool = True, folds: np.ndarray | None = None,
) -> dict:
    """In place on `lab` (float32 CIELAB of the crop). mask: finish mask (dark cores protected);
    spot_mask: the same without the core protection (marks can be deep); under: under-eye weight;
    k: Overall strength 0..1. Returns the measures (before / after) of each active correction."""
    out: dict = {}
    sel = mask > 0.5
    L0 = lab[..., 0].copy() if measure else None
    t = amount(s.darkSpots) * k
    if t > 0:
        share, first = correct_spots(lab, spot_mask, F, t, folds)
        before = float(np.mean(first[sel])) if measure and sel.any() else None
        out["darkSpots"] = {"correctedShare": round(share, 4)}
        if before is not None:
            out["darkSpots"].update(meanDepthBefore=round(before, 3), meanDepthAfter=round(float(np.mean(spot_map(lab[..., 0], F, folds=folds)[0][sel])), 3))
    t = amount(s.toneEvenness) * k
    if t > 0:
        before = tone_rms(lab, sel, F) if measure else None
        even_tone(lab, mask, F, t, edge_blotch)
        if measure:
            out["toneEvenness"] = {"colourVariationBefore": round(before, 3), "colourVariationAfter": round(tone_rms(lab, sel, F), 3)}
    t = amount(s.shine) * k
    if t > 0:
        before = float(np.percentile(lab[..., 0][sel], 99)) if measure and sel.any() else None
        exb = float(shine_excess(lab, F)[sel].mean()) if measure and sel.any() else None
        reduce_shine(lab, mask, F, t)
        if before is not None:
            out["shine"] = {"p99Before": round(before, 2), "p99After": round(float(np.percentile(lab[..., 0][sel], 99)), 2),
                            "excessBefore": round(exb, 3), "excessAfter": round(float(shine_excess(lab, F)[sel].mean()), 3)}
    t = amount(s.underEye) * k
    if t > 0 and under is not None:
        before = under_eye_deficit(lab[..., 0], mask, under, F) if measure else None
        refine_under_eye(lab, mask, under, F, t)
        if measure:
            out["underEye"] = {"deficitBefore": None if before is None else round(before, 2),
                               "deficitAfter": None if before is None else round(under_eye_deficit(lab[..., 0], mask, under, F), 2)}
    t = amount(s.dodgeBurn) * k
    if t > 0:
        before = micro_rms(lab[..., 0], sel, F) if measure else None
        dodge_burn(lab, mask, F, t, edge_blotch)
        if measure:
            out["dodgeBurn"] = {"microBefore": round(before, 3), "microAfter": round(micro_rms(lab[..., 0], sel, F), 3)}
    if measure and L0 is not None and sel.any():
        out["meanLightnessShift"] = round(float((lab[..., 0] - L0)[sel].mean()), 3)
    return out
