"""Skin Perfection phase 2D: the correction quality gate.

A mark counts as corrected only when its repair passes all seven checks, each measured on the
repaired pixels against a ring of untouched skin around the mark:

  residual    the mark is no longer visible, at its own scale and at a finer one
  colour      CIEDE2000 between the repair and the ring's tone
  texture     fine-band energy inside the repair / in the ring (not flat, not busy)
  pores       pore-scale dark peaks per area, inside vs. ring
  edges       gradient along the repair's boundary vs. the ring's gradients (no seam)
  repetition  correlation of the repair's texture with other repairs (no cloned look)
  surround    change of the ring beyond the repair radius (the skin around is left alone)

Contrast falling is only the first of these: a repair that removed the mark but left a pale
disc, a seam or a stamped texture is not "corrected". A check that cannot be measured on a mark
(too small to hold pores, too little ring skin) is reported as None and does not fail it.

The thresholds are provisional: they were set on synthetic tuning scenes and checked on real
already-retouched photographs. They are to be validated on the labelled real validation split
(`python -m scripts.perfection_benchmark --suite real ... --quality-calibration`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .common import _gauss, _luma

CHECKS = ("residual", "colour", "texture", "pores", "edges", "repetition", "surround")
# The checks that judge how natural a repair looks (residual is "is the mark gone").
NATURALNESS = CHECKS[1:]

MAX_DELTA_E = 2.0  # CIEDE2000 between the repair and its ring (about the visibility threshold)
TEXTURE_RANGE = (0.6, 1.6)  # fine-band energy inside / ring: below is flat, above is busy
MIN_PORE_SHARE = 0.5  # pore density inside the repair, as a share of the ring's
MAX_EDGE_RATIO = 2.0  # mean gradient on the boundary / the ring's 90th percentile
MAX_REPEAT_NCC = 0.9  # texture correlation with another repair
MAX_SURROUND_DL = 1.0  # mean |delta L*| in the ring beyond the repair
MIN_TEXTURE_PX = 3.0  # repair radius (px) below which texture / pores / repetition are not measurable
MIN_EXPECTED_PORES = 3.0  # the ring's pore density must predict at least this many pores inside
INNER, RING_IN, RING_OUT = 0.8, 1.4, 2.2  # core and ring radii, in repair radii
VISIBLE_FINE = 0.003  # fine-band amplitude (relative luma, ~0.3 L*) below which texture differences are invisible


@dataclass
class Quality:
    values: dict[str, float | None] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    # The colour left differs from the ring toward the mark's own colour: the repair is not
    # finished (some of the mark is left), rather than a new, wrong tone.
    incomplete: bool = False

    def to_json(self) -> dict:
        return {"failed": list(self.failed), "incomplete": self.incomplete,
                "values": {k: (None if v is None else round(float(v), 3)) for k, v in self.values.items()}}


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIEDE2000 colour difference between two CIELAB colours (L 0..100)."""
    L1, a1, b1 = (float(v) for v in lab1)
    L2, a2, b2 = (float(v) for v in lab2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = 0.5 * (C1 + C2)
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    if C1p * C2p == 0:
        dh = 0.0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2))
    Lbp, Cbp = 0.5 * (L1 + L2), 0.5 * (C1p + C2p)
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = 0.5 * (h1p + h2p)
    else:
        hbp = 0.5 * (h1p + h2p + (360 if h1p + h2p < 360 else -360))
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(np.clip(rgb, 0, 1), np.float32), cv2.COLOR_RGB2Lab)


def _pore_peaks(hp: np.ndarray, thr: float) -> np.ndarray:
    """Dark pore-scale peaks: local minima of the fine band below -thr."""
    neg = -hp
    mx = cv2.dilate(neg, _ellipse3)
    return (neg >= mx) & (neg > thr)


_ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass
class Window:
    """The pixels one mark's checks look at: a crop around the repair, before and after."""

    x0: int
    y0: int
    before: np.ndarray
    after: np.ndarray
    skin: np.ndarray
    others: np.ndarray  # other repairs and features in this window (bool): never counted as ring
    d: np.ndarray  # distance from the mark centre, px


def window(before: np.ndarray, after: np.ndarray, skin: np.ndarray, holes: np.ndarray, x: float, y: float, R: float) -> Window | None:
    h, w = before.shape[:2]
    reach = int(math.ceil(RING_OUT * R + 3))
    x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 1)
    y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(xx - x, yy - y)
    own = d <= R + 1.0
    others = (holes[y0:y1, x0:x1] > 0) & ~own
    return Window(x0, y0, before[y0:y1, x0:x1], after[y0:y1, x0:x1], skin[y0:y1, x0:x1], others, d)


def assess(win: Window | None, R: float, F: float, residual: tuple[float, float] | None,
           texture_keep: float = 1.0, pore_keep: float = 1.0) -> Quality:
    """Checks 1-5 and 7 for one repair (repetition is measured across repairs: `repetition`).

    residual: (contrast now / before at the mark's scale, same at a finer scale), or None when
    the mark has no measurable contrast (a manual mark). texture_keep / pore_keep (0..1): the
    user's texture and pore preservation; a smoother repair the user asked for is not a failure.
    """
    q = Quality(values={k: None for k in CHECKS})
    if residual is not None:
        q.values["residual"] = max(residual)
    if win is None:
        return q
    d = win.d
    inner = d <= max(1.0, INNER * R)
    ring = (d >= RING_IN * R) & (d <= RING_OUT * R) & (win.skin > 0.5) & ~win.others
    if ring.sum() < 8 or inner.sum() < 1:
        return q
    sf = max(0.7, 0.0012 * F)  # pore band, as the correction uses
    # Colour: the repair's tone against the ring's (both low-passed below the pore band, so pores
    # and noise do not count as colour).
    lab_a = _lab(_gauss(win.after, max(0.8, 1.5 * sf)))
    rep_c, ring_c = lab_a[inner].mean(0), np.median(lab_a[ring], 0)
    q.values["colour"] = delta_e2000(rep_c, ring_c)
    mark_c = _lab(_gauss(win.before, max(0.8, 1.5 * sf)))[inner].mean(0)
    left, was = rep_c - ring_c, mark_c - ring_c
    nl, nw = float(np.linalg.norm(left)), float(np.linalg.norm(was))
    q.incomplete = bool(nl > 0 and nw > 0 and float(left @ was) >= 0.7 * nl * nw and nl < nw)
    # Surround: the ring itself must be (almost) unchanged.
    lab_b = _lab(win.before)
    lab_f = _lab(win.after)
    q.values["surround"] = float(np.mean(np.abs(lab_f[..., 0][ring] - lab_b[..., 0][ring])))
    # Edges: the boundary of the repair against the ring's own gradients.
    ya = _luma(_gauss(win.after, 0.7))
    g = np.hypot(cv2.Sobel(ya, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ya, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    edge = (np.abs(d - R) <= 1.0) & (win.skin > 0.5) & ~win.others
    if edge.sum() >= 6:
        p90 = float(np.percentile(g[ring], 90))
        q.values["edges"] = float(np.mean(g[edge])) / max(p90, 1e-4)
    if R >= MIN_TEXTURE_PX:
        hp = _luma(win.after) - _luma(_gauss(win.after, 2.0 * sf))
        # The invisible floor is added to both sides, so a difference nobody could see (on skin
        # with almost no fine texture) does not count.
        n2 = VISIBLE_FINE ** 2
        e_in, e_ring = float(np.mean(hp[inner] ** 2)), float(np.mean(hp[ring] ** 2))
        q.values["texture"] = (e_in + n2) / (e_ring + n2)
        # Pores: only where the ring has visible pores to compare with.
        thr = float(np.std(hp[ring]))
        if thr >= VISIBLE_FINE:
            peaks = _pore_peaks(hp, thr)
            dens_ring = float(peaks[ring].mean())
            if dens_ring * inner.sum() >= MIN_EXPECTED_PORES:
                q.values["pores"] = float(peaks[inner].mean()) / dens_ring
    q.failed = failures(q.values, texture_keep, pore_keep)
    return q


def failures(v: dict[str, float | None], texture_keep: float = 1.0, pore_keep: float = 1.0, verify: float = 1.0) -> list[str]:
    """Which checks fail. residual is a ratio to VERIFY (the engine passes contrast/before/VERIFY_RATIO)."""
    lo_tex = TEXTURE_RANGE[0] * min(1.0, texture_keep / 0.9)
    lo_pore = MIN_PORE_SHARE * min(1.0, pore_keep / 0.9)
    out = []
    if v.get("residual") is not None and v["residual"] > verify:
        out.append("residual")
    if v.get("colour") is not None and v["colour"] > MAX_DELTA_E:
        out.append("colour")
    if v.get("texture") is not None and not (lo_tex <= v["texture"] <= TEXTURE_RANGE[1]):
        out.append("texture")
    if v.get("pores") is not None and v["pores"] < lo_pore:
        out.append("pores")
    if v.get("edges") is not None and v["edges"] > MAX_EDGE_RATIO:
        out.append("edges")
    if v.get("repetition") is not None and v["repetition"] > MAX_REPEAT_NCC:
        out.append("repetition")
    if v.get("surround") is not None and v["surround"] > MAX_SURROUND_DL:
        out.append("surround")
    return out


def texture_signature(after: np.ndarray, x: float, y: float, R: float, F: float, n: int = 12) -> np.ndarray | None:
    """The repair's fine texture as a zero-mean unit vector (for the repetition check)."""
    if R < MIN_TEXTURE_PX:
        return None
    h, w = after.shape[:2]
    s = int(math.floor(INNER * R / math.sqrt(2)))  # square inside the repair core
    if s < 2:
        return None
    x0, x1, y0, y1 = int(round(x)) - s, int(round(x)) + s + 1, int(round(y)) - s, int(round(y)) + s + 1
    pad = int(math.ceil(4 * max(0.7, 0.0012 * F))) + 1
    if x0 - pad < 0 or y0 - pad < 0 or x1 + pad > w or y1 + pad > h:
        return None
    big = after[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
    sf = max(0.7, 0.0012 * F)
    hp = (_luma(big) - _luma(_gauss(big, 2.0 * sf)))[pad:-pad, pad:-pad]
    v = cv2.resize(hp, (n, n), interpolation=cv2.INTER_AREA).ravel().astype(np.float64)
    v -= v.mean()
    nv = np.linalg.norm(v)
    if nv < 1e-6 * n:
        return None
    return v / nv


def repetition(sigs: list[np.ndarray | None], xs: list[float], ys: list[float], rs: list[float]) -> list[float | None]:
    """Highest texture correlation of each repair with another, non-overlapping repair of similar size."""
    idx = [i for i, s in enumerate(sigs) if s is not None]
    out: list[float | None] = [None] * len(sigs)
    if len(idx) < 2:
        return out
    M = np.stack([sigs[i] for i in idx])
    C = M @ M.T
    x, y, r = (np.array([v[i] for i in idx], np.float64) for v in (xs, ys, rs))
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    size_ok = np.maximum(r[:, None], r[None, :]) <= 1.5 * np.minimum(r[:, None], r[None, :])
    valid = (dist > r[:, None] + r[None, :]) & size_ok
    np.fill_diagonal(valid, False)
    C = np.where(valid, C, -1.0)
    best = C.max(1)
    for k, i in enumerate(idx):
        if best[k] > -1.0:
            out[i] = float(best[k])
    return out
