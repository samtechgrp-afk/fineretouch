"""Stray / flyaway hair cleanup around the face outline and forehead.

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
