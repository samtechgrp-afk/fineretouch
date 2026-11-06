"""Single-image relative depth for the depth-aware bokeh.

Output convention (shared with the WebGL preview): float32 HxW in [0, 1] where
0 is the subject's depth (in focus) and 1 is the far background.

Backends, best first; each one is optional and the service degrades gracefully:
  1. MiDaS v2.1 small (ONNX, run with OpenCV DNN, no PyTorch needed): true monocular
     depth, normalised so the detected faces sit at 0.
  2. MediaPipe selfie multiclass segmenter: person vs background. Used on its own
     when MiDaS is missing, and to pin the whole person to the focal plane when both exist.
  3. A geometric portrait prior built from face boxes (head ellipse + shoulders).

The result is refined with a guided filter on the photo itself so the focus
transition follows hair and shoulder edges instead of the coarse model grid.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .schemas import FaceAnalysis

log = logging.getLogger(__name__)

MODELS = Path(__file__).resolve().parent.parent / "models"
MIDAS_PATH = Path(os.environ.get("MIDAS_MODEL", MODELS / "midas_small.onnx"))
SEGMENTER_PATH = Path(os.environ.get("SEGMENTER_MODEL", MODELS / "selfie_multiclass.tflite"))
WORK_LONG_EDGE = 512
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float, subsample: int = 1) -> np.ndarray:
    """He et al. guided filter with a grey guide (fast version when subsample > 1).

    Edge-preserving smoothing: the output is locally a linear function of the guide,
    so edges in the guide survive while flat regions of `src` are averaged.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    h, w = guide.shape[:2]
    s = max(1, int(subsample))
    if s > 1:
        size = (max(1, w // s), max(1, h // s))
        g = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
        p = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        rr = max(1, r // s)
    else:
        g, p, rr = guide, src, max(1, r)
    color = p.ndim == 3
    ex = (lambda a: a[..., None]) if color else (lambda a: a)
    mean_i = box(g, rr)
    var_i = box(g * g, rr) - mean_i * mean_i
    mean_p = box(p, rr)
    cov_ip = box(ex(g) * p, rr) - ex(mean_i) * mean_p
    a = cov_ip / (ex(var_i) + eps)
    b = mean_p - a * ex(mean_i)
    mean_a = box(a, rr)
    mean_b = box(b, rr)
    if s > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    full_g = guide[..., None] if src.ndim == 3 else guide
    return mean_a * full_g + mean_b


def portrait_prior(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    """Soft person mask (1 = person) from face boxes: head ellipse plus a shoulders trapezoid."""
    m = np.zeros((h, w), np.float32)
    for f in faces:
        x, y, bw, bh = f.box
        cx, cy = (x + bw / 2) * w, (y + bh / 2) * h
        fw, fh = bw * w, bh * h
        cv2.ellipse(m, (int(cx), int(cy - fh * 0.1)), (int(fw * 0.75), int(fh * 0.85)), 0, 0, 360, 1.0, -1)
        neck_y = cy + fh * 0.55
        torso = np.array(
            [[cx - fw * 0.45, neck_y], [cx + fw * 0.45, neck_y], [cx + fw * 1.9, h], [cx - fw * 1.9, h]], np.int32
        )
        cv2.fillPoly(m, [torso], 1.0)
    if faces:
        m = cv2.GaussianBlur(m, (0, 0), max(h, w) * 0.01)
    return m


class DepthEstimator:
    """Thread-safe lazy wrapper around the optional depth / segmentation models."""

    def __init__(self, midas_path: Path = MIDAS_PATH, segmenter_path: Path = SEGMENTER_PATH) -> None:
        self.midas_path = Path(midas_path)
        self.segmenter_path = Path(segmenter_path)
        self._midas = None
        self._segmenter = None
        self._midas_failed = False
        self._segmenter_failed = False
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        parts = []
        if self.midas_path.exists() and not self._midas_failed:
            parts.append("midas")
        if self.segmenter_path.exists() and not self._segmenter_failed:
            parts.append("segmenter")
        return "+".join(parts) or "prior"

    # --- backends -------------------------------------------------------------

    def _midas_inverse_depth(self, rgb: np.ndarray) -> np.ndarray | None:
        if not self.midas_path.exists() or self._midas_failed:
            return None
        try:
            with self._lock:
                if self._midas is None:
                    self._midas = cv2.dnn.readNetFromONNX(str(self.midas_path))
                x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                x = ((x - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
                self._midas.setInput(np.ascontiguousarray(x))
                out = self._midas.forward()
            inv = np.squeeze(out).astype(np.float32)
            return cv2.resize(inv, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        except Exception as exc:  # bad model file, unsupported op in this OpenCV build
            log.error("MiDaS depth unavailable: %s", exc)
            self._midas_failed = True
            return None

    def _person_mask(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(person probability, hair probability), both HxW float32, or None."""
        if not self.segmenter_path.exists() or self._segmenter_failed:
            return None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            with self._lock:
                if self._segmenter is None:
                    opts = vision.ImageSegmenterOptions(
                        base_options=BaseOptions(model_asset_path=str(self.segmenter_path)),
                        running_mode=vision.RunningMode.IMAGE,
                        output_confidence_masks=True,
                        output_category_mask=False,
                    )
                    self._segmenter = vision.ImageSegmenter.create_from_options(opts)
                result = self._segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            masks = result.confidence_masks
            background = masks[0].numpy_view().astype(np.float32)
            hair = masks[1].numpy_view().astype(np.float32) if len(masks) > 1 else np.zeros_like(background)
            size = (rgb.shape[1], rgb.shape[0])
            person = 1.0 - cv2.resize(background, size, interpolation=cv2.INTER_LINEAR)
            hair = cv2.resize(hair, size, interpolation=cv2.INTER_LINEAR)
            return np.clip(person, 0, 1), np.clip(hair, 0, 1)
        except Exception as exc:
            log.error("Selfie segmenter unavailable: %s", exc)
            self._segmenter_failed = True
            return None

    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Person and hair probability at the image's resolution (computed on a <=512px proxy)."""
        h, w = rgb.shape[:2]
        s = min(1.0, WORK_LONG_EDGE / max(h, w))
        small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
        out = self._person_mask(small)
        if out is None:
            return None
        return tuple(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in out)  # type: ignore[return-value]

    # --- public ---------------------------------------------------------------

    def estimate(self, rgb: np.ndarray, faces: list[FaceAnalysis], long_edge: int = WORK_LONG_EDGE) -> np.ndarray:
        """rgb: HxWx3 uint8. Returns depth (0 subject .. 1 far) at <= `long_edge` px, same aspect."""
        h, w = rgb.shape[:2]
        s = min(1.0, long_edge / max(h, w))
        size = (max(1, round(w * s)), max(1, round(h * s)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if s < 1 else rgb.copy()
        sh, sw = small.shape[:2]

        seg = self._person_mask(small)
        person = seg[0] if seg is not None else None
        inv = self._midas_inverse_depth(small)

        if inv is not None:
            if faces:
                ref_mask = np.zeros((sh, sw), bool)
                for f in faces:
                    x, y, bw, bh = f.box
                    ref_mask[int(y * sh): int((y + bh) * sh) + 1, int(x * sw): int((x + bw) * sw) + 1] = True
            elif person is not None and (person > 0.5).any():
                ref_mask = person > 0.5
            else:
                ref_mask = np.zeros((sh, sw), bool)
                ref_mask[sh // 3: 2 * sh // 3, sw // 3: 2 * sw // 3] = True
            subject = float(np.median(inv[ref_mask])) if ref_mask.any() else float(np.percentile(inv, 90))
            far = float(np.percentile(inv, 2))
            depth = np.clip((subject - inv) / max(subject - far, 1e-6), 0.0, 1.0)
            if person is not None:
                depth = depth * (1.0 - np.clip(person * 1.2, 0, 1))
        else:
            if person is None:
                person = portrait_prior(faces, sh, sw) if faces else None
            if person is None:
                # No subject information at all: gentle centre-weighted falloff.
                yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
                d = np.sqrt(((xx + 0.5) / sw - 0.5) ** 2 + ((yy + 0.5) / sh - 0.45) ** 2) / 0.6
                depth = np.clip(d, 0.0, 1.0)
            else:
                # Background behind a portrait: further toward the top (sky, walls), a bit nearer at the floor.
                yy = (np.arange(sh, dtype=np.float32)[:, None] + 0.5) / sh
                bg = np.broadcast_to(1.0 - 0.35 * np.clip((yy - 0.6) / 0.4, 0, 1), (sh, sw))
                depth = bg * (1.0 - person)

        grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        r = max(2, round(max(sh, sw) * 0.012))
        depth = guided_filter(grey, depth.astype(np.float32), r, 1e-3)
        return np.clip(depth, 0.0, 1.0).astype(np.float32)
