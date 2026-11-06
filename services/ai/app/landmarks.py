"""Face landmark extraction (MediaPipe Face Landmarker, 478-point mesh) and region building.

MediaPipe returns the 468-point canonical face mesh plus 10 iris points. We turn
those into the closed polygons the renderers need (skin oval, eyes, under-eye
bands, mouth) so neither the browser nor the worker has to know mesh indices.

If the MediaPipe model file is missing we fall back to OpenCV's Haar cascade,
which yields a face box and an approximate oval but no feature regions.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .schemas import FaceAnalysis, FaceRegions, Point

log = logging.getLogger(__name__)

# Canonical MediaPipe face-mesh index rings.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# "Left"/"right" are the subject's, so the subject's left eye is on the image's right.
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
# Lower nose: subnasale, columella, nostril openings, alae and the sides just above them.
# Spot healing treats the convex hull of these as off-limits (nostril shadows read as spots).
NOSE_BOTTOM = [2, 19, 97, 98, 64, 48, 326, 327, 294, 278]
# Eye-aspect-ratio pairs: (top, bottom) x2 and (outer, inner) corners.
RIGHT_EAR = ((159, 145), (158, 153), (33, 133))
LEFT_EAR = ((386, 374), (385, 380), (263, 362))

MODEL_PATH = Path(os.environ.get("FACE_LANDMARKER_MODEL", Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"))


def _ring(pts: np.ndarray, idx: list[int]) -> list[Point]:
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in idx]


def _hull(pts: np.ndarray, idx: list[int], aspect: float) -> list[Point]:
    """Convex hull (normalised coords) of a landmark subset; hull computed in pixel-proportional space."""
    sub = pts[idx, :2].astype(np.float32) * np.array([aspect, 1.0], np.float32)
    hull = cv2.convexHull(sub).reshape(-1, 2) / np.array([aspect, 1.0], np.float32)
    return [(float(x), float(y)) for x, y in hull]


def _under_eye(pts: np.ndarray, lid: list[int], aspect: float) -> list[Point]:
    """A crescent below the lower lid: the lid curve plus the same curve pushed down."""
    lid_pts = pts[lid, :2].copy()
    width = float(np.linalg.norm(lid_pts[0] - lid_pts[-1]))
    # Push down by ~45% of the eye width (in pixel space, so compensate for aspect).
    offset = np.array([0.0, 0.45 * width * aspect], dtype=np.float32)
    lower = lid_pts[::-1] + offset
    # Pull the lower curve's ends toward the centre so the band tapers like a real crescent.
    centre_x = lid_pts[:, 0].mean()
    lower[:, 0] = centre_x + (lower[:, 0] - centre_x) * 0.85
    poly = np.concatenate([lid_pts, lower], axis=0)
    return [(float(x), float(y)) for x, y in poly]


def _eye_openness(pts_px: np.ndarray) -> float:
    def ear(spec: tuple[tuple[int, int], ...]) -> float:
        (a1, b1), (a2, b2), (c1, c2) = spec
        v = np.linalg.norm(pts_px[a1] - pts_px[b1]) + np.linalg.norm(pts_px[a2] - pts_px[b2])
        h = np.linalg.norm(pts_px[c1] - pts_px[c2])
        return float(v / (2.0 * h + 1e-6))

    e = (ear(RIGHT_EAR) + ear(LEFT_EAR)) / 2.0
    return float(np.clip((e - 0.12) / (0.28 - 0.12), 0.0, 1.0))


class FaceLandmarkService:
    """Thread-safe lazy wrapper around MediaPipe's FaceLandmarker."""

    def __init__(self, model_path: Path = MODEL_PATH, max_faces: int = 6) -> None:
        self.model_path = Path(model_path)
        self.max_faces = max_faces
        self._landmarker = None
        self._lock = threading.Lock()
        self._haar = None

        self._mediapipe_failed = False

    @property
    def backend(self) -> str:
        return "mediapipe" if self.model_path.exists() and not self._mediapipe_failed else "opencv-haar"

    def _get_landmarker(self):
        if self._landmarker is None:
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.max_faces,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    def detect(self, rgb: np.ndarray) -> list[FaceAnalysis]:
        """rgb: HxWx3 uint8. Coordinates in the result are normalised to [0, 1]."""
        h, w = rgb.shape[:2]
        # Landmarks are resolution independent; detect on a bounded proxy for speed.
        scale = min(1.0, 1280.0 / max(h, w))
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        if self.backend == "mediapipe":
            try:
                return self._detect_mediapipe(np.ascontiguousarray(small), w / h)
            except (OSError, RuntimeError) as exc:  # e.g. missing libEGL/GLES in a slim image
                log.error("MediaPipe unavailable, falling back to Haar cascade: %s", exc)
                self._mediapipe_failed = True
        return self._detect_haar(small)

    def _detect_mediapipe(self, rgb: np.ndarray, aspect: float) -> list[FaceAnalysis]:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:  # FaceLandmarker instances are not re-entrant.
            result = self._get_landmarker().detect(image)
        h, w = rgb.shape[:2]
        faces: list[FaceAnalysis] = []
        for lms in result.face_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            pts_px = pts[:, :2] * np.array([w, h], dtype=np.float32)
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            regions = FaceRegions(
                faceOval=_ring(pts, FACE_OVAL),
                leftEye=_ring(pts, LEFT_EYE),
                rightEye=_ring(pts, RIGHT_EYE),
                leftUnderEye=_under_eye(pts, LEFT_LOWER_LID, aspect),
                rightUnderEye=_under_eye(pts, RIGHT_LOWER_LID, aspect),
                innerMouth=_ring(pts, INNER_MOUTH),
                lips=_ring(pts, LIPS),
                leftBrow=_ring(pts, LEFT_BROW),
                rightBrow=_ring(pts, RIGHT_BROW),
                nose=_hull(pts, NOSE_BOTTOM, aspect),
            )
            faces.append(
                FaceAnalysis(
                    box=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
                    landmarkCount=int(pts.shape[0]),
                    regions=regions,
                    eyeOpenness=_eye_openness(pts_px),
                )
            )
        return faces

    def _detect_haar(self, rgb: np.ndarray) -> list[FaceAnalysis]:
        if self._haar is None:
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self._haar.empty():  # some OpenCV builds ship without the cascade files
            return []
        h, w = rgb.shape[:2]
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._haar.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        faces: list[FaceAnalysis] = []
        for (x, y, bw, bh) in boxes[: self.max_faces]:
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            rx, ry = bw / 2 / w * 0.9, bh / 2 / h * 1.15
            oval = [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 36, endpoint=False)]
            oval_pts = [(float(a), float(b)) for a, b in oval]
            faces.append(
                FaceAnalysis(
                    box=(x / w, y / h, bw / w, bh / h),
                    landmarkCount=0,
                    regions=FaceRegions(
                        faceOval=oval_pts, leftEye=[], rightEye=[], leftUnderEye=[], rightUnderEye=[],
                        innerMouth=[], lips=[], leftBrow=[], rightBrow=[],
                    ),
                    eyeOpenness=1.0,
                )
            )
        return faces
