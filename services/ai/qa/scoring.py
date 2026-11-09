"""Scoring one processed photo against its labels (synthetic truths or real-photo labels).

Definitions (identical for synthetic and real photos):

- A *label* is a point mark: centre (px), radius (px) and kind. Kinds in REMOVE_KINDS are
  imperfections to remove; kinds in KEEP_KINDS are natural features to keep.
- An engine mark *matches* a label when its centre is within max(1.5 r_label, 3 px, r_mark).
- *Flagged*: the engine reported the mark as an imperfection (corrected, remaining or review).
- Precision = flagged marks matching an imperfection / flagged marks.
  Recall = imperfections with at least one flagged match / imperfections. F1 as usual.
- *False positive*: a flagged mark that matches no imperfection; split into those on a natural
  feature and those on plain skin. A *false correction* is a false positive the engine changed.
  Rates are per megapixel of skin bounding box so different photo sizes compare.
- *Removed*: the imperfection's contrast against its ring fell to <= 40 % of before (measured on
  the output pixels, never taken from the engine's report). *Residual rate* = 1 - removal.
- *Claimed but visible*: the engine reported a matching mark "corrected" while the imperfection
  is still visible. This measures whether the report can be trusted.
- *Natural feature damaged*: its contrast fell below 50 %. *Wrongly flagged*: the engine flagged it.
- *Texture retention*: pore-band energy on clean skin (away from every label and mark), output /
  input; 1.0 = untouched. *Clean-skin change*: mean |dL*| there (skin the engine had no reason to
  touch).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

REMOVE_KINDS = ("acne", "pimple", "blackhead", "whitehead", "darkSpot", "postAcne", "smallDot", "blemish", "scar")
KEEP_KINDS = ("mole", "freckle", "birthmark", "follicle", "hair")
FLAGGED = ("corrected", "remaining", "review")
REMOVED_RATIO = 0.4
DAMAGED_RATIO = 0.5


@dataclass
class Label:
    x: float
    y: float
    r: float
    kind: str

    @property
    def keep(self) -> bool:
        return self.kind in KEEP_KINDS


def contrast_at(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Relative luma + redness contrast of a mark against the ring around it."""
    h, w = img.shape[:2]
    R = max(1.5, r)
    rr = int(np.ceil(3.0 * R)) + 2
    x0, x1, y0, y1 = max(0, int(x) - rr), min(w, int(x) + rr + 1), max(0, int(y) - rr), min(h, int(y) + rr + 1)
    crop = cv2.GaussianBlur(np.ascontiguousarray(img[y0:y1, x0:x1], np.float32), (0, 0), max(0.6, 0.3 * R))
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2Lab)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(xx - x, yy - y)
    inner, ring = d <= 0.6 * R, (d >= 2.0 * R) & (d <= 2.8 * R)
    if not inner.any() or not ring.any():
        return 0.0
    Lr = float(np.median(lab[..., 0][ring]))
    dL = abs(float(np.mean(lab[..., 0][inner])) - Lr) / max(Lr, 1.0)
    dA = max(0.0, float(np.mean(lab[..., 1][inner])) - float(np.median(lab[..., 1][ring])))
    return dL + 0.012 * dA


def _l_star(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img, np.float32), cv2.COLOR_RGB2Lab)[..., 0]


def clean_mask(region: np.ndarray, labels: list[Label], marks: list[dict]) -> np.ndarray:
    """region (bool HxW) minus discs around every label and every engine mark."""
    m = region.copy()
    h, w = m.shape
    block = np.zeros((h, w), np.uint8)
    for t in labels:
        cv2.circle(block, (int(round(t.x)), int(round(t.y))), int(np.ceil(4 * t.r + 3)), 1, -1)
    for k in marks:
        cv2.circle(block, (int(round(k["x"])), int(round(k["y"]))), int(np.ceil(3 * k["r"] + 3)), 1, -1)
    return m & (block == 0)


def score_photo(
    before: np.ndarray, after: np.ndarray, labels: list[Label], marks: list[dict], region: np.ndarray,
    F: float, ms: float | None = None, peak_mb: float | None = None,
) -> dict:
    """Score one photo. before / after: float32 RGB 0..1. marks: engine marks in px
    ({x, y, r, kind, status, confidence}). region: bool mask of skin that was scored.
    F: face width (or a typical skin scale) in px, for the texture band."""
    imps = [t for t in labels if not t.keep]
    nats = [t for t in labels if t.keep]
    flagged = [m for m in marks if m["status"] in FLAGGED]

    def matches(t: Label, m: dict) -> bool:
        lim = max(1.5 * t.r, 3.0, m["r"])
        return (m["x"] - t.x) ** 2 + (m["y"] - t.y) ** 2 <= lim * lim

    cats: dict[str, dict] = {}
    found = removed = claimed_visible = 0
    for t in imps:
        hits = [m for m in flagged if matches(t, m)]
        b, a = contrast_at(before, t.x, t.y, t.r), contrast_at(after, t.x, t.y, t.r)
        gone = a <= REMOVED_RATIO * max(b, 1e-6)
        c = cats.setdefault(t.kind, {"n": 0, "found": 0, "removed": 0, "claimedButVisible": 0})
        c["n"] += 1
        c["found"] += bool(hits)
        c["removed"] += bool(gone)
        found += bool(hits)
        removed += bool(gone)
        if not gone and any(m["status"] == "corrected" for m in hits):
            c["claimedButVisible"] += 1
            claimed_visible += 1
    for t in nats:
        b, a = contrast_at(before, t.x, t.y, t.r), contrast_at(after, t.x, t.y, t.r)
        c = cats.setdefault(t.kind, {"n": 0, "kept": 0, "damaged": 0, "wronglyFlagged": 0})
        c["n"] += 1
        dmg = a < DAMAGED_RATIO * b
        c["damaged"] += bool(dmg)
        c["kept"] += not dmg
        c["wronglyFlagged"] += any(matches(t, m) for m in flagged)

    tp = sum(any(matches(t, m) for t in imps) for m in flagged)
    fp = [m for m in flagged if not any(matches(t, m) for t in imps)]
    fp_nat = sum(any(matches(t, m) for t in nats) for m in fp)
    fp_corrected = sum(m["status"] == "corrected" for m in fp)
    ys, xs = np.nonzero(region)
    mp = max(1e-6, ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / 1e6) if ys.size else 1.0

    precision = tp / len(flagged) if flagged else (1.0 if not imps else 0.0)
    recall = found / len(imps) if imps else 1.0
    clean = clean_mask(region, labels, marks)
    sigma = max(0.8, 0.0025 * F)
    if clean.sum() > 100:
        lb, la = _l_star(before), _l_star(after)
        fb = lb - cv2.GaussianBlur(lb, (0, 0), sigma)
        fa = la - cv2.GaussianBlur(la, (0, 0), sigma)
        texture = float(np.mean(fa[clean] ** 2) / max(np.mean(fb[clean] ** 2), 1e-12))
        clean_change = float(np.mean(np.abs(la[clean] - lb[clean])))
    else:
        texture = clean_change = None
    n_nat = len(nats)
    damaged = sum(c.get("damaged", 0) for k, c in cats.items() if k in KEEP_KINDS)
    return {
        "imperfections": len(imps),
        "natural": n_nat,
        "flagged": len(flagged),
        "truePositives": tp,
        "found": found,
        "removed": removed,
        "skinMP": mp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "removalRate": removed / len(imps) if imps else 1.0,
        "residualRate": 1.0 - (removed / len(imps) if imps else 1.0),
        "claimedButVisible": claimed_visible,
        "falsePositives": len(fp),
        "falsePositivesOnNatural": fp_nat,
        "falseCorrections": fp_corrected,
        "falsePositivesPerMP": len(fp) / mp,
        "naturalPreserved": (n_nat - damaged) / n_nat if n_nat else 1.0,
        "naturalDamaged": damaged,
        "textureRetention": texture,
        "cleanSkinChangeL": clean_change,
        "ms": ms,
        "peakMB": peak_mb,
        "categories": cats,
    }
