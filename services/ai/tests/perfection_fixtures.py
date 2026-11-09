"""Synthetic skin with labelled imperfections for the Skin Perfection benchmark and tests.

Each scene is built in CIELAB so skin tone, redness (a*) and brownness (b*) are controlled:
a skin tone (Fitzpatrick-like L*/a*/b*), a lighting gradient, multi-octave skin relief, a pore
field (thousands of tiny low-contrast dark dots), sensor noise, and labelled marks:

  imperfections (must be removed): acne, pimple, blackhead, whitehead, darkSpot, postAcne
  natural features (must be kept):  mole, freckle (in clusters), follicle (dense stubble field)

Sizes are shares of the face width F, so the same scene can be rendered at any resolution.
Ground truth comes back as a list of Truth (x, y, r in px, kind).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas import FaceAnalysis, FaceRegions

# L*, a*, b* of the skin (light -> deep).
SKIN_TONES = {
    "I": (78.0, 9.0, 14.0), "II": (70.0, 11.0, 17.0), "III": (62.0, 13.0, 20.0),
    "IV": (52.0, 14.0, 22.0), "V": (40.0, 13.0, 19.0), "VI": (28.0, 10.0, 13.0),
}
IMPERFECTIONS = ("acne", "pimple", "blackhead", "whitehead", "darkSpot", "postAcne")
NATURAL = ("mole", "freckle", "follicle")


@dataclass
class Truth:
    x: float
    y: float
    r: float
    kind: str


def _add(L, A, B, cx, cy, r, dL, dA, dB, soft=0.35):
    """Soft disc of Lab deltas (soft: edge width as a share of r)."""
    h, w = L.shape
    rr = int(np.ceil(r * 2.5)) + 2
    x0, x1, y0, y1 = max(0, int(cx) - rr), min(w, int(cx) + rr + 1), max(0, int(cy) - rr), min(h, int(cy) + rr + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy)
    a = np.clip((r - d) / max(0.6, soft * r) + 0.5, 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    L[y0:y1, x0:x1] += dL * a
    A[y0:y1, x0:x1] += dA * a
    B[y0:y1, x0:x1] += dB * a


def oval_face(w: int, h: int) -> FaceAnalysis:
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    oval = [(0.5 + 0.47 * np.cos(a), 0.5 + 0.47 * np.sin(a)) for a in t]
    return FaceAnalysis(
        box=(0.03, 0.03, 0.94, 0.94), landmarkCount=478, eyeOpenness=0.8,
        regions=FaceRegions(faceOval=oval, leftEye=[], rightEye=[], leftUnderEye=[], rightUnderEye=[],
                            innerMouth=[], lips=[], leftBrow=[], rightBrow=[], nose=[]),
    )


def oval_mask(w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (int(0.47 * w), int(0.47 * h)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(2.0, 0.01 * w))


def make_scene(
    size: int = 1200, tone: str = "III", seed: int = 0, density: str = "normal", lighting: float = 0.15,
    with_natural: bool = True, face_share: float = 0.94, marks: bool = True,
) -> tuple[np.ndarray, list[Truth], float]:
    """(float32 sRGB image size x size, truths, face width px). density: normal | dense (acne field)."""
    rng = np.random.default_rng(seed)
    h = w = size
    F = face_share * size
    L0, A0, B0 = SKIN_TONES[tone]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Lighting: a soft key from the upper left.
    light = 1.0 + lighting * (0.5 - (xx + yy) / (w + h))
    L = np.full((h, w), L0, np.float32) * light
    A = np.full((h, w), A0, np.float32)
    B = np.full((h, w), B0, np.float32)
    # Skin relief: a few octaves of smooth noise (blotchiness and micro relief).
    for scale, amp in ((0.08, 1.2), (0.02, 0.7), (0.006, 0.5)):
        s = max(1.0, scale * F)
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), s)
        n /= max(float(n.std()), 1e-6)
        L += amp * n
    # Pores: a dense field of tiny, faint dark dots.
    n_pores = int(0.012 * h * w / max(1.0, (0.004 * F) ** 2) * 0.35)
    pr = max(0.6, 0.0013 * F)
    pores = np.zeros((h, w), np.float32)
    px, py = rng.uniform(0, w, n_pores).astype(int), rng.uniform(0, h, n_pores).astype(int)
    pores[py, px] = 1.0
    pores = cv2.GaussianBlur(pores, (0, 0), pr)
    pores /= max(float(pores.max()), 1e-6)
    L -= pores * L0 * 0.035
    truths: list[Truth] = []

    def place(n: int, kind: str, rmin: float, rmax: float, margin: float = 0.12) -> list[tuple[float, float, float]]:
        out = []
        tries = 0
        while len(out) < n and tries < n * 200:
            tries += 1
            cx, cy = rng.uniform(margin * w, (1 - margin) * w), rng.uniform(margin * h, (1 - margin) * h)
            if (cx - w / 2) ** 2 / (0.4 * w) ** 2 + (cy - h / 2) ** 2 / (0.4 * h) ** 2 > 1:
                continue
            r = rng.uniform(rmin, rmax) * F
            if any((t.x - cx) ** 2 + (t.y - cy) ** 2 < (2.6 * max(t.r, r)) ** 2 for t in truths) or any(
                (o[0] - cx) ** 2 + (o[1] - cy) ** 2 < (2.6 * max(o[2], r)) ** 2 for o in out
            ):
                continue
            out.append((cx, cy, r))
            truths.append(Truth(cx, cy, r, kind))
        return out

    k = (2.5 if density == "dense" else 1.0) * (1.0 if marks else 0.0)
    for cx, cy, r in place(int(10 * k), "acne", 0.004, 0.008):
        _add(L, A, B, cx, cy, r * 1.8, -0.02 * L0, 4.0, 0.0, soft=0.8)  # inflamed halo
        _add(L, A, B, cx, cy, r, -0.05 * L0, 9.0, 1.0)
    for cx, cy, r in place(int(5 * k), "pimple", 0.008, 0.013):
        _add(L, A, B, cx, cy, r * 1.6, -0.02 * L0, 5.0, 0.0, soft=0.8)
        _add(L, A, B, cx, cy, r, -0.03 * L0, 10.0, 1.0)
        _add(L, A, B, cx, cy, r * 0.35, 0.06 * L0, -3.0, -1.0)  # pus head
    for cx, cy, r in place(int(12 * k), "blackhead", 0.0018, 0.003):
        _add(L, A, B, cx, cy, r, -0.32 * L0, -1.0, 0.5, soft=0.25)
    for cx, cy, r in place(int(6 * k), "whitehead", 0.0025, 0.0045):
        _add(L, A, B, cx, cy, r, 0.08 * L0, -1.5, 0.5, soft=0.5)
    for cx, cy, r in place(int(8 * k), "darkSpot", 0.004, 0.01):
        _add(L, A, B, cx, cy, r, -0.16 * L0, 1.0, 5.0, soft=0.4)
    for cx, cy, r in place(int(5 * k), "postAcne", 0.006, 0.012):
        _add(L, A, B, cx, cy, r, -0.08 * L0, 4.5, 3.5, soft=0.6)
    if with_natural:
        for cx, cy, r in place(2, "mole", 0.008, 0.012):
            _add(L, A, B, cx, cy, r, -0.55 * L0, 0.5, 3.0, soft=0.2)
        # A freckle cluster.
        fx, fy = 0.32 * w, 0.62 * h
        for _ in range(14):
            cx, cy = fx + rng.normal(0, 0.05 * F), fy + rng.normal(0, 0.04 * F)
            r = rng.uniform(0.003, 0.0055) * F
            if any((t.x - cx) ** 2 + (t.y - cy) ** 2 < (2.6 * max(t.r, r)) ** 2 for t in truths):
                continue
            _add(L, A, B, cx, cy, r, -0.07 * L0, 1.0, 4.0, soft=0.5)
            truths.append(Truth(cx, cy, r, "freckle"))
        # A stubble field (hair follicles): dense tiny dark dots.
        sx, sy = 0.62 * w, 0.75 * h
        for _ in range(70):
            cx, cy = sx + rng.uniform(-0.07, 0.07) * F, sy + rng.uniform(-0.05, 0.05) * F
            r = 0.0012 * F
            if any((t.x - cx) ** 2 + (t.y - cy) ** 2 < (3 * max(t.r, r)) ** 2 for t in truths):
                continue
            _add(L, A, B, cx, cy, r, -0.2 * L0, -0.5, -0.5, soft=0.3)
            truths.append(Truth(cx, cy, r, "follicle"))
    if not marks:
        # Faint marks only (for sensitivity tests): barely-visible dark spots.
        for cx, cy, r in place(14, "darkSpot", 0.005, 0.009):
            _add(L, A, B, cx, cy, r, -0.05 * L0, 0.5, 2.0, soft=0.5)
    lab = np.stack([np.clip(L, 0, 100), A, B], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb += rng.normal(0, 0.004, rgb.shape).astype(np.float32)  # sensor noise
    return np.clip(rgb, 0.0, 1.0).astype(np.float32), truths, F


def contrast_at(img: np.ndarray, t: Truth) -> float:
    """Relative luma + redness contrast of a truth mark against its ring (for scoring)."""
    h, w = img.shape[:2]
    R = max(1.5, t.r)
    rr = int(np.ceil(3.0 * R)) + 2
    x0, x1, y0, y1 = max(0, int(t.x) - rr), min(w, int(t.x) + rr + 1), max(0, int(t.y) - rr), min(h, int(t.y) + rr + 1)
    crop = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), max(0.6, 0.3 * R))
    lab = cv2.cvtColor(np.ascontiguousarray(crop, np.float32), cv2.COLOR_RGB2Lab)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(xx - t.x, yy - t.y)
    inner, ring = d <= 0.6 * R, (d >= 2.0 * R) & (d <= 2.8 * R)
    Lr = float(np.median(lab[..., 0][ring]))
    dL = abs(float(np.mean(lab[..., 0][inner])) - Lr) / max(Lr, 1.0)
    dA = max(0.0, float(np.mean(lab[..., 1][inner])) - float(np.median(lab[..., 1][ring])))
    return dL + 0.012 * dA


def pore_energy(img: np.ndarray, mask: np.ndarray, sigma: float) -> float:
    y = cv2.cvtColor(np.ascontiguousarray(img, np.float32), cv2.COLOR_RGB2Lab)[..., 0]
    fine = y - cv2.GaussianBlur(y, (0, 0), sigma)
    return float(np.mean(fine[mask] ** 2))
