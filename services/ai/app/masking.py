"""Semantic skin masks and user mask edits (v0.8, Phase 3).

Two inputs refine the landmark-polygon skin mask built in pipeline.build_masks:

* Face parsing from the MediaPipe selfie multiclass segmenter (Apache-2.0, already shipped for
  bokeh and stray hair): per-pixel face skin, body skin (neck, chest, arms) and hair. Semantic
  face skin reaches the hairline, ears and jaw that the landmark oval misses, and hair lying
  across the face is taken out of the mask.
* User mask strokes (SkinMaskSettings.strokes): "add" paints skin the AI missed, "erase"
  protects an area from every skin tool. Strokes are applied to the final skin weight, so an
  erased area is untouched whatever the sliders say.

Mirrored in apps/web/src/engine/masks.ts (buildFaceMask / rasterizeMaskStrokes).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .schemas import MaskStroke, SkinMaskSettings

# Feather of the user's strokes at feather=100, as a share of the stroke radius.
STROKE_FEATHER_MAX = 1.0
# Hair probability above which skin is removed (soft ramp from HAIR_CUT[0] to HAIR_CUT[1]).
HAIR_CUT = (0.35, 0.75)
# Semantic face skin only counts within this many face widths of a detected face's centre.
FACE_REACH = 1.0


@dataclass
class FaceParse:
    """Per-pixel probabilities (HxW float32, 0..1) from the selfie multiclass segmenter."""

    face_skin: np.ndarray
    body_skin: np.ndarray
    hair: np.ndarray

    def resized(self, h: int, w: int) -> "FaceParse":
        if self.face_skin.shape == (h, w):
            return self
        r = lambda m: cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)  # noqa: E731
        return FaceParse(r(self.face_skin), r(self.body_skin), r(self.hair))


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hair_keep(hair: np.ndarray) -> np.ndarray:
    """1 where there is no hair, 0 on hair (soft)."""
    return 1.0 - smoothstep(HAIR_CUT[0], HAIR_CUT[1], hair)


def rasterize_strokes(strokes: list[MaskStroke], h: int, w: int, feather: float) -> tuple[np.ndarray, np.ndarray]:
    """(add, erase) masks, HxW float32 0..1. Stroke size is a radius as a share of the long edge;
    points are normalised [x0, y0, x1, y1, ...]. Feather 0..100 blurs each by up to its radius."""
    add = np.zeros((h, w), np.float32)
    erase = np.zeros((h, w), np.float32)
    if not strokes:
        return add, erase
    L = float(max(h, w))
    f = float(np.clip(feather / 100.0, 0.0, 1.0)) * STROKE_FEATHER_MAX
    for kind, canvas in (("add", add), ("erase", erase)):
        group = [s for s in strokes if s.mode == kind]
        if not group:
            continue
        sigma = 0.0
        for s in group:
            r = max(0.5, s.size * L)
            flat = s.points[: len(s.points) // 2 * 2]
            pts = np.asarray(flat, np.float32).reshape(-1, 2) * np.array([w, h], np.float32)
            layer = np.zeros((h, w), np.float32)
            ipts = np.round(pts).astype(np.int32)
            if len(ipts) == 1:
                cv2.circle(layer, tuple(int(v) for v in ipts[0]), int(round(r)), 1.0, -1, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(layer, [ipts], False, 1.0, thickness=max(1, int(round(2 * r))), lineType=cv2.LINE_AA)
                for p in (ipts[0], ipts[-1]):
                    cv2.circle(layer, tuple(int(v) for v in p), int(round(r)), 1.0, -1, lineType=cv2.LINE_AA)
            np.maximum(canvas, layer, out=canvas)
            sigma = max(sigma, f * r * 0.5)
        if sigma >= 0.5:
            canvas[:] = cv2.GaussianBlur(canvas, (0, 0), sigma)
    return np.clip(add, 0, 1), np.clip(erase, 0, 1)


def apply_mask_edits(skin: np.ndarray, settings: SkinMaskSettings) -> np.ndarray:
    """Final skin weight with the user's add / erase strokes (absolute, not colour-gated)."""
    if not settings.strokes:
        return skin
    add, erase = rasterize_strokes(settings.strokes, skin.shape[0], skin.shape[1], settings.feather)
    return (np.maximum(skin, add) * (1.0 - erase)).astype(np.float32)


def protect_weight(settings: SkinMaskSettings, h: int, w: int) -> np.ndarray | None:
    """1 - erase strokes (HxW), for landmark tools outside the skin mask (under-eye); None
    when there are no erase strokes."""
    if not any(s.mode == "erase" for s in settings.strokes):
        return None
    _, erase = rasterize_strokes(settings.strokes, h, w, settings.feather)
    return 1.0 - erase


def face_zone(boxes_px: list[tuple[float, float, float, float]], h: int, w: int) -> np.ndarray:
    """Soft region within FACE_REACH face widths of each face centre (limits semantic face skin
    to people whose face was detected, e.g. not a poster in the background)."""
    zone = np.zeros((h, w), np.float32)
    for x, y, bw, bh in boxes_px:
        cx, cy = x + bw / 2, y + bh / 2
        axes = (max(1, int(FACE_REACH * bw)), max(1, int(FACE_REACH * bh * 1.1)))
        cv2.ellipse(zone, (int(cx), int(cy)), axes, 0, 0, 360, 1.0, -1)
    return zone
