"""Clothing virtual iron and recolour.

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
from .studio import WRINKLE_WORK_LONG, _bbox, _resize
from .style import _skin_probability

def clothing_mask(
    image: np.ndarray,
    faces: list[FaceAnalysis],
    clothes_prob: np.ndarray | None = None,
    person_prob: np.ndarray | None = None,
) -> np.ndarray | None:
    """Where the garments are. HxW float32, or None when nothing can be found.

    Best: the selfie segmenter's clothes class (clothes_prob). Without the model, a portrait
    prior: person (segmenter or a head-and-shoulders shape from the face boxes) below the
    chin, minus anything skin-coloured.
    """
    h, w = image.shape[:2]
    if clothes_prob is not None:
        m = np.clip(clothes_prob.astype(np.float32), 0.0, 1.0)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m if (m > 0.5).any() else None
    if not faces:
        return None
    if person_prob is None:
        from ..depth import portrait_prior

        person_prob = portrait_prior(faces, h, w)
    person = cv2.resize(person_prob.astype(np.float32), (w, h)) if person_prob.shape != (h, w) else person_prob
    chin = max((f.box[1] + f.box[3]) * h for f in faces)
    below = np.zeros((h, w), np.float32)
    below[min(h, int(chin + 0.04 * h)):, :] = 1.0
    skin = _skin_probability(_gauss(image, 2.0))
    m = person * below * (1.0 - skin)
    m = _gauss(m.astype(np.float32), 0.004 * max(h, w))
    return m if (m > 0.5).any() else None


def _log_luma(c: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(_luma(c), 1e-3)).astype(np.float32)


def remove_clothing_wrinkles(
    image: np.ndarray,
    mask: np.ndarray,
    strength: float = 60.0,
    texture_preservation: float = 80.0,
    face_width: float | None = None,
) -> np.ndarray:
    """"Virtual iron": smooth creases and folds in clothing, keep the weave and the drape.

    Wrinkles are shading. They change how much light a patch of fabric gets, not its colour,
    and they live at a scale between the fabric's threads and the garment's overall drape.
    So the filter works on log-luminance only (shading is multiplicative; chromaticity is
    kept exactly, which also means prints and dyed patterns are untouched) and splits it,
    using normalised convolution inside the garment mask so skin and background never
    bleed in, into:

      weave     finer than ~0.6% of the face width: threads, knit, twill. Kept; only lowered
                by (1 - texture_preservation) at full strength.
      creases   up to ~6% of the face width: sharp folds and crumpled creases. Attenuated by
                up to 90%.
      folds     up to ~30% of the face width: soft rolls and bunching. Attenuated by up to 45%.
      drape     everything larger: the garment's form and its big shadows. Never touched.

    Garment edges, seams and luminance-only prints (strong steps) are protected by an edge
    gate on the band amplitude, and the effect fades out near the mask border. Colour is
    rebuilt as rgb * (new luminance / old luminance).

    image: HxWx3 float32 sRGB; mask: HxW garment weight; strength and texture_preservation
    0..100; face_width (px) sets the scales, defaulting to 18% of the long edge.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    box = _bbox(mask, 8)
    if box is None:
        return image
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    m_full = np.clip(mask[y0:y1, x0:x1].astype(np.float32), 0, 1)
    ch, cw = crop.shape[:2]
    s = min(1.0, WRINKLE_WORK_LONG / max(ch, cw))
    size = (max(1, round(cw * s)), max(1, round(ch * s)))
    work = _resize(crop, size, True) if s < 1 else crop
    m = _resize(m_full, size, True) if s < 1 else m_full
    fw = (face_width if face_width and face_width > 0 else 0.18 * max(h, w)) * s

    hard = (m > 0.5).astype(np.float32)
    l = _log_luma(work)
    l_w = _masked_blur(l, hard, max(0.6, 0.006 * fw))   # weave removed
    l_c = _masked_blur(l, hard, max(1.0, 0.06 * fw))    # creases removed
    l_d = _masked_blur(l, hard, max(2.0, 0.3 * fw))     # folds removed: the drape
    weave = l - l_w
    creases = l_w - l_c
    folds = l_c - l_d

    k = float(np.clip(strength / 100.0, 0, 1))
    pres = float(np.clip(texture_preservation / 100.0, 0, 1))
    # Edges, seams and bold prints: big, abrupt steps. Wrinkle shading stays well below
    # ~0.35 log units (a 1.4x brightness change) per band.
    edge = _smoothstep(0.35, 0.6, np.abs(creases)) + _smoothstep(0.45, 0.8, np.abs(folds))
    gx = cv2.Sobel(l_w, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_w, cv2.CV_32F, 0, 1, ksize=3)
    step = _smoothstep(0.25, 0.5, np.hypot(gx, gy))
    keep = np.clip(edge + step, 0, 1)
    inner = cv2.erode(hard, _ellipse(max(3, int(0.02 * fw)) | 1))
    inner = _gauss(inner, max(1.0, 0.015 * fw))
    gate = k * inner * m * (1.0 - _gauss(keep, max(1.0, 0.01 * fw)))
    l_new = l_d + folds * (1.0 - 0.45 * gate) + creases * (1.0 - 0.9 * gate) + weave * (1.0 - gate * (1.0 - pres))
    ratio = np.exp(l_new - l)
    if s < 1:
        ratio = cv2.resize(ratio, (cw, ch), interpolation=cv2.INTER_CUBIC)
    out = image.copy()
    out[y0:y1, x0:x1] = crop * ratio[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def shift_clothing_color(image: np.ndarray, mask: np.ndarray, hue_shift: float = 0.0, saturation: float = 0.0) -> np.ndarray:
    """Recolour garments: rotate CIELAB hue by hue_shift degrees and scale chroma by
    (1 + saturation / 100). L* (and with it the fabric's shading and texture) is kept.
    Skin-coloured pixels are excluded from the mask so the edit never bleeds onto skin."""
    if hue_shift == 0 and saturation == 0:
        return image
    box = _bbox(mask, 2)
    if box is None:
        return image
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    m = np.clip(mask[y0:y1, x0:x1], 0, 1) * (1.0 - _skin_probability(_gauss(crop, 1.5)))
    lab = rgb_to_lab(crop.reshape(-1, 3)).reshape(crop.shape)
    t = math.radians(hue_shift)
    cos_t, sin_t = math.cos(t), math.sin(t)
    a, b = lab[..., 1], lab[..., 2]
    k = max(0.0, 1.0 + saturation / 100.0)
    lab[..., 1], lab[..., 2] = (a * cos_t - b * sin_t) * k, (a * sin_t + b * cos_t) * k
    rgb = lab_to_rgb(lab.reshape(-1, 3)).reshape(crop.shape)
    out = image.copy()
    out[y0:y1, x0:x1] = crop + (rgb - crop) * m[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
