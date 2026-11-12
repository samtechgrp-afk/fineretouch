"""Phase 3b: eyeglasses detection gates the Glasses glare tool."""

import math

import cv2
import numpy as np

from app.engines.glasses import GLASSES_CONFIDENT, glasses_score, wears_glasses
from app.schemas import FaceAnalysis, FaceRegions

H = W = 600
rng = np.random.default_rng(3)


def ring(cx, cy, rx, ry, n=16):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


EYE_L, EYE_R = ring(0.4, 0.42, 0.045, 0.015), ring(0.6, 0.42, 0.045, 0.015)


def bare_face(tone=(150, 105, 80)) -> np.ndarray:
    img = np.full((H, W, 3), tone, np.float32)
    _yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img *= (1.0 - 0.25 * ((xx - 300) / 300) ** 2)[..., None]  # soft shading toward the sides
    img += rng.normal(0, 3, (H, W, 1))
    for cx in (240, 360):
        cv2.ellipse(img, (cx, 252), (27, 9), 0, 0, 360, (215, 205, 200), -1)
        cv2.circle(img, (cx, 252), 7, (40, 25, 20), -1)
        cv2.ellipse(img, (cx, 205), (40, 7), 0, 0, 360, (45, 30, 25), -1)  # brow
    cv2.GaussianBlur(img, (0, 0), 1.0, dst=img)
    return np.clip(img, 0, 255).astype(np.uint8)


def with_frames(img: np.ndarray, thick: int, colour=(25, 20, 20)) -> np.ndarray:
    out = img.copy()
    for cx in (240, 360):
        cv2.rectangle(out, (cx - 45, 228), (cx + 45, 290), colour, thick, cv2.LINE_AA)
    cv2.line(out, (285, 240), (315, 240), colour, thick, cv2.LINE_AA)  # bridge
    return out


def test_bare_faces_are_not_glasses():
    for tone in ((210, 170, 150), (150, 105, 80), (80, 52, 40)):
        assert glasses_score(bare_face(tone), EYE_L, EYE_R)["confidence"] < 0.3, tone


def test_dark_frames_are_found():
    for tone in ((210, 170, 150), (150, 105, 80), (80, 52, 40)):
        for thick in (3, 6):
            s = glasses_score(with_frames(bare_face(tone), thick), EYE_L, EYE_R)
            assert s["confidence"] >= GLASSES_CONFIDENT, (tone, thick, s)


def test_one_cue_alone_is_not_glasses():
    img = bare_face()
    cv2.line(img, (285, 240), (315, 240), (25, 20, 20), 4)  # a bridge-like line only
    assert glasses_score(img, EYE_L, EYE_R)["confidence"] < GLASSES_CONFIDENT


def test_wears_glasses_gate_and_override():
    def face(g):
        regions = {"faceOval": ring(0.5, 0.5, 0.3, 0.4), "leftEye": EYE_L, "rightEye": EYE_R, "leftUnderEye": [], "rightUnderEye": [],
                       "innerMouth": [], "lips": [], "leftBrow": [], "rightBrow": []}
        return FaceAnalysis(box=(0.2, 0.1, 0.6, 0.8), landmarkCount=478, regions=FaceRegions(**regions), eyeOpenness=1.0, glasses=g)
    assert wears_glasses(face(0.9)) and not wears_glasses(face(0.3)) and not wears_glasses(face(None))
    assert wears_glasses(face(None), force=True) and wears_glasses(face(0.0), force=True)
