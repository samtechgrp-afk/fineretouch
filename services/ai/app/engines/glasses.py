"""Eyeglasses detection (phase 3b): is this face wearing glasses?

The Glasses glare tool (studio.reduce_glasses_glare) reads bright structures inside the lens
area as reflections. On a face without glasses those are skin highlights and rim light, and
removing them as "glare" turned the outer eye blue and orange. So the tool now runs only on
faces whose glasses are confidently found here (or when the user forces it).

What is measured, on the eye area turned level and scaled so the eyes are D px apart:

  bridge   the frame's bridge over the nose between the eyes: a thin, level edge running across
           (almost) the whole width between the inner eye corners. Skin there shades softly.
  rims     the lens rims: a thin edge below each eye (the lower rim, on the upper cheek) and
           beside each eye's outer corner, each running along most of its length.

A score is the edge's strength (thin-scale gradient, relative to the face's own contrast)
times how consistently it runs (share of columns / rows that carry it at about the same place).
Glasses need the bridge plus a rim on both sides, so one strong cue alone (a deep crease, the
nasion shadow, the edge of the face) is never enough. Rimless glasses have no frame to find: the
detector then says "not sure" and the tool is skipped unless the user turns it on by hand.
"""

from __future__ import annotations

import cv2
import numpy as np

D = 100.0  # eye centre distance in the working crop, px
GLASSES_CONFIDENT = 0.6  # at or above: glasses (Glasses glare runs)
BRIDGE_ZONE = (-0.08, 0.08, -0.38, 0.08)  # x0, x1, y0, y1 in eye distances from the eyes' midpoint
LOWER_ZONE = (-0.32, 0.32, 0.14, 0.5)  # around each eye centre
OUTER_ZONE = (0.38, 0.66, -0.3, 0.3)  # outward from each eye centre
EDGE_SIGMA = 1.0  # px at D: the frame is a thin line
EDGE_LO, EDGE_HI = 0.05, 0.14  # edge strength (share of the face's contrast) for no / full score
RUN_LO, RUN_HI = 0.45, 0.8  # share of the zone the edge must run along


def _smoothstep(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _level_crop(grey: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The eye area, turned so the eyes are level, scaled so they are D px apart and centred."""
    mid = (left + right) / 2.0
    v = right - left
    dist = float(np.hypot(*v))
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    scale = D / max(dist, 1e-3)
    M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), ang, scale)
    W, H = int(2.4 * D), int(1.6 * D)
    M[0, 2] += W / 2 - mid[0]
    M[1, 2] += H / 2 - mid[1]
    return cv2.warpAffine(grey, M, (W, H), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _zone(img: np.ndarray, cx: float, cy: float, z: tuple[float, float, float, float], flip: bool = False) -> np.ndarray:
    x0, x1, y0, y1 = z
    if flip:
        x0, x1 = -x1, -x0
    a, b = round(cx + x0 * D), round(cx + x1 * D)
    c, d = round(cy + y0 * D), round(cy + y1 * D)
    return img[max(0, c):max(0, d), max(0, a):max(0, b)]


def _run_score(g: np.ndarray, contrast: float, axis: int) -> float:
    """A thin edge running along `axis` (0: level edge, read per column; 1: upright, per row):
    for each column (row) its strongest edge and where it sits; the score is the typical
    strength times the share of columns whose edge sits near the others' (a frame is one line)."""
    if g.size == 0 or min(g.shape) < 4:
        return 0.0
    a = np.abs(g) / max(contrast, 1e-3)
    if axis == 1:
        a = a.T
    pos = a.argmax(axis=0).astype(np.float32)
    peak = a.max(axis=0)
    # A frame's edge moves smoothly from one column to the next; skin texture jumps about.
    med = np.median(pos)
    near = np.abs(pos - cv2.GaussianBlur(pos[None, :], (0, 0), 3.0)[0]) <= 2.0
    near &= np.abs(pos - med) <= 0.12 * D
    strength = float(np.median(peak[near])) if near.any() else 0.0
    return float(_smoothstep(EDGE_LO, EDGE_HI, strength) * _smoothstep(RUN_LO, RUN_HI, near.mean()))


def glasses_score(rgb: np.ndarray, left_eye: list, right_eye: list) -> dict:
    """Cue scores (0..1) and the overall confidence for one face. left_eye / right_eye: the
    eye-opening landmark rings, normalised to the image."""
    h, w = rgb.shape[:2]
    le = np.array(left_eye, np.float32) * [w, h]
    re = np.array(right_eye, np.float32) * [w, h]
    cl, cr = le.mean(axis=0), re.mean(axis=0)
    if cl[0] > cr[0]:  # left in the image first
        cl, cr = cr, cl
    if np.hypot(*(cr - cl)) < 12:  # too small to see a frame
        return {"bridge": 0.0, "lower": 0.0, "outer": 0.0, "confidence": 0.0}
    # Only the eye area is read: crop first, so a 30 MP frame is never converted whole.
    d = float(np.hypot(*(cr - cl)))
    x0, x1 = int(max(0, min(cl[0], cr[0]) - 1.4 * d)), int(min(w, max(cl[0], cr[0]) + 1.4 * d))
    y0, y1 = int(max(0, min(cl[1], cr[1]) - 1.2 * d)), int(min(h, max(cl[1], cr[1]) + 1.2 * d))
    sub = rgb[y0:y1, x0:x1]
    grey = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype(np.float32) / (65535.0 if sub.dtype == np.uint16 else 255.0)
    off = np.array([x0, y0], np.float32)
    img = _level_crop(grey, cl - off, cr - off)
    img = cv2.GaussianBlur(img, (0, 0), EDGE_SIGMA)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    H, W = img.shape
    mx, my = W / 2.0, H / 2.0
    face = img[int(my - 0.5 * D):int(my + 0.6 * D), int(mx - 0.9 * D):int(mx + 0.9 * D)]
    contrast = float(np.percentile(face, 90) - np.percentile(face, 10)) if face.size else 0.0
    if contrast < 0.02:
        return {"bridge": 0.0, "lower": 0.0, "outer": 0.0, "confidence": 0.0}
    bridge = _run_score(_zone(gy, mx, my, BRIDGE_ZONE), contrast, 0)
    lower = [_run_score(_zone(gy, mx + s * D / 2, my, LOWER_ZONE), contrast, 0) for s in (-1, 1)]
    outer = [_run_score(_zone(gx, mx + s * D / 2, my, OUTER_ZONE, flip=s < 0), contrast, 1) for s in (-1, 1)]
    rim = [max(lo, ou) for lo, ou in zip(lower, outer)]
    confidence = float(min(bridge, min(rim)) ** 0.5 * max(bridge, min(rim)) ** 0.5)
    return {"bridge": round(bridge, 3), "lower": round(min(lower), 3), "outer": round(min(outer), 3),
            "confidence": round(confidence, 3)}


def glasses_confidence(rgb: np.ndarray, left_eye: list, right_eye: list) -> float | None:
    """0..1 how sure we are this face wears glasses; None without eye landmarks."""
    if len(left_eye) < 3 or len(right_eye) < 3:
        return None
    return glasses_score(rgb, left_eye, right_eye)["confidence"]


def wears_glasses(face, force: bool = False) -> bool:
    """Glasses glare runs on this face: glasses confidently found, or the user's override."""
    if force:
        return len(face.regions.leftEye) >= 3 and len(face.regions.rightEye) >= 3
    g = getattr(face, "glasses", None)
    return g is not None and g >= GLASSES_CONFIDENT
