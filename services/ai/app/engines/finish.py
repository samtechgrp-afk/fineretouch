"""Professional Skin Finish engine (v0.14, stage 0d): texture and tone finishing after Skin Perfection.

Skin Perfection (stage 0c) removes discrete marks. A retoucher's finish is a different job: it
calms the relief of the skin (bumps a few pores wide, rough patches, blotches), keeps believable
pores and grain, and leaves the face's contours alone. This engine does that job on face and body
skin, after Skin Perfection and the healing strokes, before every older skin stage.

Phase 1 (this file so far): the finish mask, the band split, edge-aware smoothing, texture
restoration and the first verification measures. Shine, tone evenness, dark-spot refinement,
under-eye refinement and dodge and burn follow in phase 2.

How it works (all at full resolution on the skin's bounding box; every scale is a share of the
face width F, so a close-up and a full-length shot get the same treatment):

  1. Finish mask: the renderer's skin weight (face skin with eyes, brows and lips cut out, the
     person's own skin colour, Add skin / Protect strokes) plus parsed body skin. Pixels much
     darker than the skin around them (nostrils, lash shadows, moles) are protected.
  2. Band split of CIELAB L* (perceptual, so deep and light skin get the same treatment):
         fine   < S_FINE      grain and the finest texture
         pore   S_FINE..S_PORE   pores
         bump   S_PORE..S_BUMP   bumps and rough patches a few pores wide
         blotch S_BUMP..S_BLOTCH uneven tone patches
         base   > S_BLOTCH    contours and lighting (never touched here)
  3. Smoothing: each band is scaled by its own control (Skin smoothing for bumps and blotches,
     Pore preservation for pores, Texture preservation for grain). Near real edges (a nostril,
     an eyelid, the jaw against the neck, a cheek against its shadow) the smoothing fades out,
     band by band, so contours keep their shape (see EDGE_* and the method notes below).
  4. Texture restoration: where fine and pore texture ends up below the floor Texture
     preservation sets (a flattened repair, a very smooth patch), grain matched to the person's
     own skin is added back, seeded so the same photo always gives the same result.
  5. Verification: feature-edge contrast kept, texture retained, complexion shift, and an exact
     copy of every pixel outside the mask.

Methods (chosen by measurement, see docs/SKIN_FINISH.md): "edge" (default) takes its smoothing
power from Gaussian bands and an edge weight per band decides where it may act. "gauss" (no edge
weight), "guided" (self-guided filter bands) and "domain" (domain-transform bands) are kept for
the comparison in qa/finish_eval.py.

Only classical image processing on CPU (NumPy / OpenCV); no model, no GPU requirement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..schemas import SkinFinishSettings
from .finish_local import any_local, apply_local, contour_weight

# Band boundaries as shares of the face width F (Gaussian sigmas). On a face 750 px wide:
# fine < 1.2 px, pores 1.2-3.4 px, bumps 3.4-10.5 px, blotches 10.5-34 px.
S_FINE, S_PORE, S_BUMP, S_BLOTCH = 0.0016, 0.0045, 0.014, 0.045
# Most a band can be reduced at 100%: bumps and blotches by Skin smoothing, pores and grain by
# 100 - their preservation slider (scaled by Skin smoothing, so preservation alone changes nothing).
MAX_BUMP, MAX_BLOTCH, MAX_PORE, MAX_FINE = 1.0, 0.6, 0.75, 0.4
# Edge weight: scale-normalised gradient of L* at each band's scale, against the person's typical
# skin gradient there. Skin relief stays near typical; contours (nostril rims, eyelids, lips, jaw,
# cheek against shadow) are EDGE_RATIO times steeper and at least EDGE_LO L*. EDGE_REACH widens the
# weight over the band's spread (in band sigmas).
EDGE_LO, EDGE_RATIO, EDGE_REACH = 4.0, 3.0, 1.2
# Protected dark cores: this many L* darker than the skin around them (nostrils, lash shadow, moles).
DARK_CORE = 9.0
# Texture restoration: grain is added where fine + pore energy is under this share of the
# person's typical skin energy, times Texture preservation.
RESTORE_FLOOR = 0.85
METHODS = ("edge", "gauss", "guided", "domain")
# Gaussian bands overlap: cutting the whole bump band also takes about a quarter of the pores with
# it. Pore preservation therefore also moves the bump cut's lower edge away from the pore scale,
# from S_PORE (at 0) to (1 + PORE_GAP) * S_PORE (at 100), so pores at 100 keep over 90% of their
# relief at full smoothing. The narrower cut is deepened towards MAX_BUMP_GAPPED to compensate.
PORE_GAP, MAX_BUMP_GAPPED = 1.0, 1.0


@dataclass
class FinishReport:
    ms: float = 0.0
    method: str = "edge"
    face_width: float = 0.0
    mask_share: float = 0.0  # share of the frame the finish works on
    edge_kept: float | None = None  # feature-edge contrast after / before (1 = all kept)
    texture_kept: float | None = None  # fine + pore energy after / before, on smoothed skin
    restored_share: float = 0.0  # share of the mask that got grain back
    complexion_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mean dL*, da*, db* on skin
    bands_before: dict = field(default_factory=dict)
    bands_after: dict = field(default_factory=dict)
    local: dict = field(default_factory=dict)  # phase 2 corrections: before / after measures

    def as_dict(self) -> dict:
        return {
            "ms": round(self.ms, 1), "method": self.method, "faceWidth": round(self.face_width, 1),
            "maskShare": round(self.mask_share, 4),
            "edgeKept": None if self.edge_kept is None else round(self.edge_kept, 3),
            "textureKept": None if self.texture_kept is None else round(self.texture_kept, 3),
            "restoredShare": round(self.restored_share, 4),
            "complexionShift": [round(v, 2) for v in self.complexion_shift],
            "bandsBefore": {k: round(v, 3) for k, v in self.bands_before.items()},
            "bandsAfter": {k: round(v, 3) for k, v in self.bands_after.items()},
            "local": self.local,
        }


# --- small helpers ---------------------------------------------------------------------------

def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur. Wide blurs run on a shrunk copy and are scaled back up: the result is a
    smooth low-pass either way, and a 100 px kernel at full size would cost seconds."""
    if sigma <= 8.0:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sigma, 0.5), borderType=cv2.BORDER_REFLECT)
    h, w = img.shape[:2]
    f = sigma / 4.0
    size = (max(1, round(w / f)), max(1, round(h / f)))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=4.0 * min(size[0] / w, size[1] / h) * f, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def _guided(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al.): edge-preserving low-pass of I with window radius r."""
    r = max(1, int(round(r)))
    if r > 12:  # same filter on a shrunk copy for wide windows
        h, w = I.shape
        f = r / 8.0
        small = cv2.resize(I, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
        out = _guided(small, 8, eps)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    mI = _box(I, r)
    var = _box(I * I, r) - mI * mI
    a = var / (var + eps)
    b = mI - a * mI
    return _box(a, r) * I + _box(b, r)


def _domain(I: np.ndarray, sigma_s: float, sigma_r: float, iters: int = 3) -> np.ndarray:
    """Domain-transform recursive filter (Gastal and Oliveira 2011), self-guided."""
    if sigma_s > 10:  # shrink for wide filters, as above
        h, w = I.shape
        f = sigma_s / 6.0
        small = cv2.resize(I, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(_domain(small, 6.0, sigma_r, iters), (w, h), interpolation=cv2.INTER_LINEAR)
    J = I.astype(np.float32).copy()
    dx = np.abs(np.diff(I, axis=1, prepend=I[:, :1]))
    dy = np.abs(np.diff(I, axis=0, prepend=I[:1, :]))
    ctx = (1.0 + sigma_s / sigma_r * dx).astype(np.float32)
    cty = (1.0 + sigma_s / sigma_r * dy).astype(np.float32)
    for i in range(iters):
        s_i = sigma_s * math.sqrt(3.0) * 2 ** (iters - i - 1) / math.sqrt(4 ** iters - 1)
        a = math.exp(-math.sqrt(2.0) / s_i)
        V = np.power(a, ctx)
        for x in range(1, J.shape[1]):
            J[:, x] += V[:, x] * (J[:, x - 1] - J[:, x])
        for x in range(J.shape[1] - 2, -1, -1):
            J[:, x] += V[:, x + 1] * (J[:, x + 1] - J[:, x])
        V = np.power(a, cty)
        for y in range(1, J.shape[0]):
            J[y] += V[y] * (J[y - 1] - J[y])
        for y in range(J.shape[0] - 2, -1, -1):
            J[y] += V[y + 1] * (J[y + 1] - J[y])
    return J


def lowpasses(L: np.ndarray, F: float, method: str) -> list[np.ndarray]:
    """[L, low(S_FINE), low(S_PORE), low(S_BUMP), low(S_BLOTCH)] for the chosen method."""
    sig = [max(0.6, s * F) for s in (S_FINE, S_PORE, S_BUMP, S_BLOTCH)]
    if method == "guided":
        # eps in L*^2: relief under ~6 L* is smoothed, contours above it are kept.
        return [L] + [_guided(L, s, (6.0 + 2.0 * k) ** 2) for k, s in enumerate(sig)]
    if method == "domain":
        return [L] + [_domain(L, 1.6 * s, 10.0) for s in sig]
    return [L] + [_gauss(L, s) for s in sig]


def edge_weights(L: np.ndarray, F: float, mask: np.ndarray) -> list[np.ndarray]:
    """Per band (pore, bump, blotch; grain uses the pore weight): 1 near a real contour, 0 on
    plain skin. The gradient of L* at the band's own scale (geometric mean of its edges), in L*
    units, is compared with the person's typical skin gradient at that scale: a contour is
    EDGE_RATIO times steeper than typical relief and at least EDGE_LO L*. The weight is then
    widened over the band's reach, so the band's own spread around the edge is kept too."""
    out = []
    sel = mask > 0.5
    for lo_s, hi_s in ((S_FINE, S_PORE), (S_PORE, S_BUMP), (S_BUMP, S_BLOTCH)):
        s = max(0.8, math.sqrt(lo_s * hi_s) * F)
        g = _gauss(L, s)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        mag = np.sqrt(gx * gx + gy * gy) * (2.5 * s)  # L* change across ~2.5 sigma
        typical = float(np.median(mag[sel])) if sel.sum() > 64 else 1.0
        lo = max(EDGE_LO, EDGE_RATIO * typical)
        e = _smoothstep(lo, 2.0 * lo, mag)
        reach = max(1, int(round(EDGE_REACH * hi_s * F)))
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2))
        out.append(np.clip(_gauss(e, 0.5 * reach), 0.0, 1.0))
    return out


def relief_area(L: np.ndarray, mask: np.ndarray, F: float) -> np.ndarray:
    """Skin (mask > 0.5) away from real contours: where band energy measures skin relief and not
    the contour's own step (which the edge check measures instead)."""
    edges = feature_edges(L, mask, F).astype(np.uint8)
    r = max(1, int(round(2.0 * S_BLOTCH * F)))
    near = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    return (mask > 0.5) & ~near


def band_energy(L: np.ndarray, mask: np.ndarray, F: float, area: np.ndarray | None = None) -> dict[str, float]:
    """RMS of each band of L* over `area` (default: the mask > 0.5): the QA's texture measure."""
    lows = lowpasses(L, F, "gauss")
    sel = (mask > 0.5) if area is None else area
    if not sel.any():
        return {}
    names = ("fine", "pore", "bump", "blotch")
    return {n: float(np.sqrt(np.mean((lows[i] - lows[i + 1])[sel] ** 2))) for i, n in enumerate(names)}


def feature_edges(L: np.ndarray, mask: np.ndarray, F: float) -> np.ndarray:
    """Pixels on real contours inside or at the edge of the skin (for the edge-kept check)."""
    s = max(1.0, 0.004 * F)
    g = _gauss(L, s)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0 * (2.5 * s)
    near = cv2.dilate((mask > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return (mag > 10.0) & near


def edge_contrast(L: np.ndarray, edges: np.ndarray, F: float) -> float:
    s = max(1.0, 0.004 * F)
    g = _gauss(L, s)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float(mag[edges].mean()) if edges.any() else 0.0


# --- the engine ------------------------------------------------------------------------------

def finish_mask(skin: np.ndarray, body: np.ndarray | None, L: np.ndarray, F: float) -> np.ndarray:
    """Face skin (already without eyes, brows, lips, hair) plus body skin, minus dark cores."""
    m = skin if body is None else np.maximum(skin, body)
    m = m.astype(np.float32)
    base = _gauss(L, max(2.0, 0.02 * F))
    # Compared at the pore scale, so sensor noise and single dark pixels (a noisy night photo
    # has 3 L* of grain) do not read as cores; nostrils, lash shadows and moles still do.
    Ls = _gauss(L, max(0.8, S_PORE * F))
    core = _smoothstep(DARK_CORE * 0.6, DARK_CORE, base - Ls)
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return m * (1.0 - _gauss(core, max(1.0, 0.003 * F)))


def _amounts(s: SkinFinishSettings) -> tuple[float, float, float, float]:
    """Band reductions (bump, blotch, pore, fine) from the sliders, each 0..1."""
    sm = s.smoothing.amount / 100.0 if s.smoothing.enabled else 0.0
    tp = s.texturePreservation.amount / 100.0 if s.texturePreservation.enabled else 0.0
    pp = s.porePreservation.amount / 100.0 if s.porePreservation.enabled else 0.0
    bump = (MAX_BUMP + (MAX_BUMP_GAPPED - MAX_BUMP) * pp) * sm
    return bump, MAX_BLOTCH * sm, MAX_PORE * sm * (1.0 - pp), MAX_FINE * sm * (1.0 - tp)


def _pore_gap(s: SkinFinishSettings) -> float:
    """Lower edge of the bump cut, in units of S_PORE (1 at Pore preservation 0)."""
    pp = s.porePreservation.amount / 100.0 if s.porePreservation.enabled else 0.0
    return 1.0 + PORE_GAP * pp


def _lowpass(L: np.ndarray, s: float, method: str) -> np.ndarray:
    """One low-pass of L* at Gaussian sigma s (px) with the method's filter."""
    if method == "guided":
        return _guided(L, s, 8.0 ** 2)
    if method == "domain":
        return _domain(L, 1.6 * s, 10.0)
    return _gauss(L, s)


def _restore_grain(L_out: np.ndarray, L_in: np.ndarray, mask: np.ndarray, F: float, tp: float,
                   seed: int) -> tuple[np.ndarray, float]:
    """Add grain where fine + pore texture fell below the floor. Returns (L*, share restored)."""
    if tp <= 0:
        return L_out, 0.0
    s_p = max(0.6, S_PORE * F)
    sel = mask > 0.5
    if sel.sum() < 64:
        return L_out, 0.0
    hi_in = L_in - _gauss(L_in, s_p)
    hi_out = L_out - _gauss(L_out, s_p)
    win = max(2.0, 0.02 * F)
    e_in = np.sqrt(np.maximum(_gauss(hi_in * hi_in, win), 0.0))
    e_out = np.sqrt(np.maximum(_gauss(hi_out * hi_out, win), 0.0))
    typical = float(np.median(e_in[sel]))
    floor = RESTORE_FLOOR * tp * typical
    deficit = np.clip(floor - e_out, 0.0, None) * (mask > 0.05)
    need = deficit > 0.05 * max(typical, 1e-3)
    if not need.any():
        return L_out, 0.0
    # Grain with the person's own spectrum: white noise filtered to the fine + pore band, scaled
    # to unit RMS, then to the deficit. Seeded by the image, so a re-render is identical.
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(L_out.shape).astype(np.float32)
    n = _gauss(n, max(0.5, 0.5 * S_FINE * F)) - _gauss(n, s_p)
    n /= max(float(n.std()), 1e-6)
    out = L_out + n * deficit * mask
    return out, float(need[sel].mean())


def finish_skin(
    image: np.ndarray, skin: np.ndarray, settings: SkinFinishSettings, F: float,
    body: np.ndarray | None = None, method: str = "edge", measure: bool = True,
    under_eye: np.ndarray | None = None, folds: np.ndarray | None = None,
) -> tuple[np.ndarray, FinishReport]:
    """image: HxWx3 float32 sRGB 0..1 (after Skin Perfection). skin: HxW face skin weight.
    body: optional HxW body skin weight. F: face width in px (body uses the same scale).
    under_eye: optional HxW under-eye weight (landmarks), for the under-eye correction.
    folds: optional HxW weight of the nose and nasolabial folds (landmarks): where it is known,
    larger dark patches elsewhere may be any shape; without it they must be round.
    Returns the finished image (pixels outside the mask are copied bit for bit) and a report."""
    t0 = time.perf_counter()
    rep = FinishReport(method=method, face_width=F)
    if method not in METHODS:
        raise ValueError(f"unknown finish method {method!r}")
    h, w = skin.shape
    region = skin if body is None else np.maximum(skin, body)
    ys, xs = np.nonzero(region > 1e-3)
    if ys.size == 0 or not settings.enabled or settings.strength <= 0:
        rep.ms = (time.perf_counter() - t0) * 1000
        return image, rep
    pad = int(math.ceil(3 * S_BLOTCH * F)) + 4
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1], np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    L = np.ascontiguousarray(lab[..., 0])
    sk = skin[y0:y1, x0:x1].astype(np.float32)
    bd = None if body is None else body[y0:y1, x0:x1].astype(np.float32)
    mask = finish_mask(sk, bd, L, F)
    rep.mask_share = float((mask > 0.5).sum()) / float(h * w)
    k = settings.strength / 100.0

    # Phase 2: local corrections (marks, tone, shine, under-eye, dodge and burn) first, so the
    # texture finish below sees the corrected skin (and no longer protects a mark's edge).
    lab0, local = lab, any_local(settings)
    if local:
        raw = sk if bd is None else np.maximum(sk, bd)
        fo = None if folds is None else np.ascontiguousarray(folds[y0:y1, x0:x1], dtype=np.float32)
        e_blotch = contour_weight(L, F) if fo is None else np.maximum(fo, contour_weight(L, F, strict=True))
        ue = None if under_eye is None else under_eye[y0:y1, x0:x1].astype(np.float32)
        lab = lab.copy()
        rep.local = apply_local(lab, mask, raw, ue, F, settings, k, e_blotch, measure, folds=fo)
        L = np.ascontiguousarray(lab[..., 0])

    a_bump, a_blotch, a_pore, a_fine = _amounts(settings)
    lows = lowpasses(L, F, method)
    bands = [lows[i] - lows[i + 1] for i in range(4)]  # fine, pore, bump, blotch
    gap = _pore_gap(settings)
    s_gap = max(0.6, gap * S_PORE * F)
    low_gap = lows[2] if gap <= 1.0 else _lowpass(L, s_gap, method)
    bands[2] = low_gap - lows[3]  # the bump cut starts above the pores (see PORE_GAP)
    if method == "edge":
        e_pore, e_bump, e_blotch = edge_weights(L, F, mask)
        weights = [e_pore, e_pore, e_bump, e_blotch]
    else:
        weights = [0.0, 0.0, 0.0, 0.0]
    cut = (a_fine, a_pore, a_bump, a_blotch)
    dL = np.zeros_like(L)
    for b, a, e in zip(bands, cut, weights):
        if a > 0:
            dL -= b * (a * (1.0 - e))
    dL *= mask * k
    L_new = L + dL

    # Colour relief at the bump and pore scales moves with the luminance (a red bump is one bump).
    lab_new = lab.copy()
    s_f, s_p, s_b = (max(0.6, s * F) for s in (S_FINE, S_PORE, S_BUMP))
    for c in (1, 2):
        A = np.ascontiguousarray(lab[..., c])
        g1, g2, g3 = _gauss(A, s_f), _gauss(A, s_p), _gauss(A, s_b)
        g2b = g2 if gap <= 1.0 else _gauss(A, s_gap)
        e = weights[2] if method == "edge" else 0.0
        dA = -((g1 - g2) * a_pore + (g2b - g3) * a_bump) * (1.0 - e)
        lab_new[..., c] = A + dA * mask * k

    # Texture restoration against the floor Texture preservation sets.
    tp = settings.texturePreservation.amount / 100.0 if settings.texturePreservation.enabled else 0.0
    seed = (h * 73856093) ^ (w * 19349663) ^ 0x5F1A
    L_new, rep.restored_share = _restore_grain(L_new, L, mask, F, tp * k, seed)
    lab_new[..., 0] = L_new
    out_crop = np.clip(cv2.cvtColor(lab_new, cv2.COLOR_Lab2RGB), 0.0, 1.0)

    if measure:
        sel = mask > 0.5
        if sel.any():
            L0 = np.ascontiguousarray(lab0[..., 0])  # the photo as it came in (before phase 2 too)
            area = relief_area(L0, mask, F)
            rep.bands_before = band_energy(L0, mask, F, area)
            rep.bands_after = band_energy(L_new, mask, F, area)
            edges = feature_edges(L0, mask, F)
            before = edge_contrast(L0, edges, F)
            rep.edge_kept = edge_contrast(L_new, edges, F) / before if before > 0 else None
            fb, fa = rep.bands_before, rep.bands_after
            tb = math.hypot(fb.get("fine", 0.0), fb.get("pore", 0.0))
            rep.texture_kept = math.hypot(fa.get("fine", 0.0), fa.get("pore", 0.0)) / tb if tb > 0 else None
            d = (lab_new - lab0)[sel]
            rep.complexion_shift = tuple(float(v) for v in d.mean(axis=0))

    out = image.copy()
    reach = mask if not local else np.maximum(mask, raw)  # dark-spot work also reaches into cores
    keep = (reach > 1e-3)[..., None]
    out[y0:y1, x0:x1] = np.where(keep, out_crop, crop)
    rep.ms = (time.perf_counter() - t0) * 1000
    return out.astype(np.float32, copy=False), rep
