"""Pro studio retouch: eyeglass glare and teeth.

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
# 5. Pro studio retouch (v0.6): glasses glare, teeth, clothing "virtual iron"
# ---------------------------------------------------------------------------
#
# All three work on a crop around their region at a face-scaled working resolution, so the
# cost does not grow with the megapixels of the camera; results are resampled and
# feather-blended back into the full-resolution image.

GLARE_WORK_FACE = 480.0  # face width (px) the glare detector works at
GLARE_MAX_VEIL = 0.75  # strongest veil the de-veiling model inverts; brighter cores are inpainted
TEETH_TARGET_B = 3.0  # CIELAB b* that whitening aims for (slightly warm; 0 or below looks blue / fake)
TEETH_MAX_L = 92.0  # whitening never lifts teeth past this L*
WRINKLE_WORK_LONG = 1400.0  # long edge of the clothing crop the wrinkle filter works at


def _face_width(faces: list[FaceAnalysis], w: int) -> float:
    return max((f.box[2] * w for f in faces), default=0.0)


def _poly_px(points: Sequence[tuple[float, float]], w: int, h: int) -> np.ndarray:
    return np.round(np.array(points, np.float32) * np.array([w, h], np.float32)).astype(np.int32)


def _bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 1e-3)
    if ys.size == 0:
        return None
    h, w = mask.shape[:2]
    return max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)


def _resize(img: np.ndarray, size: tuple[int, int], down: bool) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)


def glasses_lens_roi(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    """Where eyeglass lenses sit: an ellipse around each eye, larger than the eye itself.

    Lenses are typically 2-2.5x the width of the eye opening and reach from the brow to the
    upper cheek, so the ellipse is sized from the face width and centred slightly below the
    eye. HxW float32 in [0, 1]; empty when no face has eye landmarks.
    """
    roi = np.zeros((h, w), np.float32)
    for f in faces:
        fw = f.box[2] * w
        for eye in (f.regions.leftEye, f.regions.rightEye):
            if len(eye) < 3:
                continue
            c = np.array(eye, np.float32).mean(axis=0) * np.array([w, h], np.float32)
            centre = (int(round(c[0])), int(round(c[1] + 0.02 * fw)))
            axes = (max(2, int(0.2 * fw)), max(2, int(0.15 * fw)))
            cv2.ellipse(roi, centre, axes, 0, 0, 360, 1.0, -1)
    return roi


def _eye_mask(faces: list[FaceAnalysis], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for f in faces:
        for eye in (f.regions.leftEye, f.regions.rightEye):
            if len(eye) >= 3:
                cv2.fillPoly(m, [_poly_px(eye, w, h)], 1)
    return m


def detect_glare(crop: np.ndarray, roi: np.ndarray, eyes: np.ndarray, face_w: float, sensitivity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lens reflections in a working-resolution crop.

    crop: HxWx3 float32; roi: lens region; eyes: uint8 eye-opening mask; face_w in crop px.
    Returns (veil, background, glare_colour): veil is the estimated reflection opacity
    (0 = clean, ~1 = blown-out reflection), background the scene without the reflection
    (a grey morphological opening, which removes bright structures smaller than a lens
    while keeping the darker scene behind them), glare_colour the reflection's RGB.

    Catchlights (small highlights entirely inside the eye opening) and thin rim highlights
    on the frames are not glare and are left out, so eyes keep their sparkle and frames
    keep their shape.
    """
    # Pre-smooth so sensor noise neither biases the opening low nor reads as reflection.
    soft = _gauss(crop, max(0.8, 0.005 * face_w))
    y = _luma(soft).astype(np.float32)
    k = _ellipse(int(0.18 * face_w) | 1)
    bg = cv2.morphologyEx(soft, cv2.MORPH_OPEN, k)
    bg = _gauss(bg, max(0.8, 0.005 * face_w))
    bg_y = _luma(bg)
    excess = y - bg_y
    thr = 0.11 - 0.07 * sensitivity
    binary = ((excess > thr) & (roi > 0.5)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool)
    catch_area = (0.035 * face_w) ** 2
    min_side = 0.02 * face_w
    for i in range(1, n):
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if min(bw, bh) < min_side or area < (0.015 * face_w) ** 2:
            continue  # rim highlight on the frame, or noise
        comp = labels == i
        inside_eye = float(eyes[comp].mean()) if area else 0.0
        if inside_eye > 0.6 or (inside_eye > 0.3 and area < catch_area):
            continue  # catchlight, or the white of the eye: not a reflection on the lens
        keep[i] = True
    glare = keep[labels]
    zero = np.zeros(y.shape, np.float32)
    if not glare.any():
        return zero, bg, np.ones(3, np.float32)
    ex = excess[glare]
    top = glare & (excess >= np.percentile(ex, 80))
    # Flash and window reflections are close to clipping: keep the measured tint, at full brightness.
    tint = soft[top].mean(axis=0).astype(np.float32)
    colour = (tint / max(float(tint.max()), 1e-3)).astype(np.float32)
    g_y = float(colour @ LUMA)
    # Opacity from the smooth brightness lift over the scene around the reflection: the scene
    # brightness is measured outside the reflection (normalised convolution), the lift at a
    # scale coarser than eye detail, so an iris behind the veil does not read as "no veil".
    support = cv2.dilate(glare.astype(np.uint8), _ellipse(int(0.15 * face_w) | 1)).astype(np.float32)
    clear = (1.0 - support) * (roi > 0.5) + (1.0 - support) * 0.25
    ref_y = _masked_blur(y, np.clip(clear, 0, 1).astype(np.float32), max(2.0, 0.12 * face_w))
    lift_y = _gauss(cv2.morphologyEx(y, cv2.MORPH_CLOSE, _ellipse(int(0.08 * face_w) | 1)), max(1.0, 0.03 * face_w))
    veil = np.clip((lift_y - ref_y) / np.maximum(g_y - ref_y, 0.05), 0.0, 1.0) * support
    veil = cv2.morphologyEx(veil.astype(np.float32), cv2.MORPH_CLOSE, _ellipse(int(0.08 * face_w) | 1))
    veil = _gauss(veil, max(0.8, 0.02 * face_w))
    support = _gauss(support, 0.03 * face_w)
    return (veil * np.clip(support, 0, 1)).astype(np.float32), bg.astype(np.float32), colour


def reduce_glasses_glare(image: np.ndarray, faces: list[FaceAnalysis], strength: float = 60.0) -> np.ndarray:
    """Remove flash / window reflections on eyeglass lenses while keeping the eyes visible.

    Reflection model: observed = scene * (1 - a) + a * G, where G is the reflection colour
    and a its opacity. a is estimated per pixel from how far the lens is lifted above the
    scene behind it (detect_glare); where a <= GLARE_MAX_VEIL the scene is recovered
    exactly by inverting the model, which keeps the iris, lashes and skin seen through a
    hazy reflection. Blown-out cores (a > GLARE_MAX_VEIL hold too little of the scene) are
    inpainted from the recovered surroundings. strength 0..100 sets detection sensitivity
    and how much of the correction is applied.
    """
    if strength <= 0 or not faces:
        return image
    h, w = image.shape[:2]
    roi_full = glasses_lens_roi(faces, h, w)
    fw_full = _face_width(faces, w)
    box = _bbox(roi_full, int(0.05 * fw_full) + 4)
    if box is None or fw_full <= 0:
        return image
    y0, y1, x0, x1 = box
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = min(1.0, GLARE_WORK_FACE / fw_full)
    size = (max(1, round(cw * s)), max(1, round(ch * s)))
    work = _resize(crop, size, True) if s < 1 else crop
    roi = _resize(roi_full[y0:y1, x0:x1], size, True) if s < 1 else roi_full[y0:y1, x0:x1]
    eyes = _eye_mask(faces, h, w)[y0:y1, x0:x1]
    eyes = cv2.resize(eyes, size, interpolation=cv2.INTER_NEAREST) if s < 1 else eyes
    sens = float(np.clip(strength / 100.0, 0.0, 1.0))
    veil, _bg, colour = detect_glare(work, roi, eyes, fw_full * s, sens)
    if veil.max() <= 1e-3:
        return image

    a = np.minimum(veil, GLARE_MAX_VEIL)[..., None]
    scene = np.clip((work - a * colour) / (1.0 - a), 0.0, 1.0)
    core = (veil > GLARE_MAX_VEIL).astype(np.uint8)
    if core.any():
        core = cv2.dilate(core, _ellipse(3))
        radius = max(3, int(0.02 * fw_full * s))
        scene = np.stack([_inpaint_unit(scene[..., i], core, radius) for i in range(3)], axis=-1)
    # Only the correction is resampled, so the crop keeps its full-resolution detail.
    delta = (scene - work) * (0.5 + 0.5 * sens)
    alpha = np.clip(veil * 4.0, 0.0, 1.0) * np.clip(_gauss(roi, 0.01 * fw_full * s) * 1.5, 0.0, 1.0)
    if s < 1:
        delta = cv2.resize(delta, (cw, ch), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
    fixed = np.clip(crop + delta * alpha[..., None], 0.0, 1.0).astype(np.float32)
    # Phase 3b: skin seen through (or around) the lens keeps its own colour. Removing a near-
    # neutral reflection may restore saturation but never turns a hue: outside the eye openings
    # a* / b* keep their direction, and their size may only grow by what the veil took away.
    eye_full = _eye_mask(faces, h, w)[y0:y1, x0:x1].astype(np.float32)
    fixed = _keep_skin_hue(crop.astype(np.float32), fixed, alpha, eye_full)
    out = image.copy()
    out[y0:y1, x0:x1] = fixed
    return np.clip(out, 0.0, 1.0).astype(np.float32)


GLARE_CHROMA_GAIN = 1.0 / (1.0 - GLARE_MAX_VEIL)  # most a reflection can have desaturated a colour


def _keep_skin_hue(before: np.ndarray, after: np.ndarray, alpha: np.ndarray, eyes: np.ndarray) -> np.ndarray:
    """after with its a*/b* outside the eye openings put back on before's hue, the chroma scaled
    by at most GLARE_CHROMA_GAIN (and no less than before's). Lightness is kept from after."""
    lab0 = cv2.cvtColor(before, cv2.COLOR_RGB2Lab)
    lab1 = cv2.cvtColor(after, cv2.COLOR_RGB2Lab)
    c0 = np.hypot(lab0[..., 1], lab0[..., 2])
    c1 = np.hypot(lab1[..., 1], lab1[..., 2])
    k = np.clip(c1 / np.maximum(c0, 1e-3), 1.0, GLARE_CHROMA_GAIN)
    k = np.where(c0 < 1.0, 1.0, k)  # near-grey: no hue to keep, keep before's colour
    locked = lab1.copy()
    locked[..., 1] = lab0[..., 1] * k
    locked[..., 2] = lab0[..., 2] * k
    keep = (1.0 - np.clip(cv2.GaussianBlur(eyes, (0, 0), 2.0), 0.0, 1.0))[..., None]
    lab = lab1 * (1.0 - keep) + locked * keep
    out = after.copy()
    moved = alpha > 1e-4  # pixels the correction left alone come back bit for bit
    out[moved] = np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0.0, 1.0)[moved]
    return out


def teeth_mask(image: np.ndarray, faces: list[FaceAnalysis]) -> np.ndarray:
    """Soft mask of visible teeth: inside the inner-mouth outline, bright and not red.

    Gums, tongue and lips are strongly red (CIELAB a* > ~20) and the gaps between teeth are
    dark, so both are excluded; what remains is enamel. HxW float32.
    """
    h, w = image.shape[:2]
    mouth = np.zeros((h, w), np.uint8)
    for f in faces:
        if len(f.regions.innerMouth) >= 3:
            cv2.fillPoly(mouth, [_poly_px(f.regions.innerMouth, w, h)], 1)
    box = _bbox(mouth, 2)
    out = np.zeros((h, w), np.float32)
    if box is None:
        return out
    y0, y1, x0, x1 = box
    lab = rgb_to_lab(image[y0:y1, x0:x1].reshape(-1, 3)).reshape(y1 - y0, x1 - x0, 3)
    sel = mouth[y0:y1, x0:x1] > 0
    l_ref = float(np.percentile(lab[..., 0][sel], 90)) if sel.any() else 70.0
    bright = _smoothstep(max(25.0, 0.55 * l_ref), max(35.0, 0.75 * l_ref), lab[..., 0])
    not_red = 1.0 - _smoothstep(12.0, 22.0, lab[..., 1])
    m = bright * not_red * sel
    fw = _face_width(faces, w)
    out[y0:y1, x0:x1] = _gauss(m.astype(np.float32), max(0.6, 0.002 * fw)) * sel
    return out


def correct_teeth(image: np.ndarray, faces: list[FaceAnalysis], whiten: float = 0.0, stain_removal: float = 0.0) -> np.ndarray:
    """Whiten teeth and even out stains while keeping enamel texture.

    Works in CIELAB on the teeth mask. Every change is computed from a smoothed (masked
    Gaussian) copy of the teeth, then added as an offset, so the fine detail (enamel
    texture, edges between teeth) is carried over untouched:
      stain_removal  pulls local yellow / dark patches toward the teeth's own average.
      whiten         lowers yellow (b*) toward TEETH_TARGET_B, takes out a little red, and
                     lifts L* by a share of the headroom to TEETH_MAX_L.
    Both 0..100.
    """
    if (whiten <= 0 and stain_removal <= 0) or not faces:
        return image
    h, w = image.shape[:2]
    mask_full = teeth_mask(image, faces)
    box = _bbox(mask_full, 4)
    if box is None:
        return image
    y0, y1, x0, x1 = box
    m = mask_full[y0:y1, x0:x1]
    crop = image[y0:y1, x0:x1]
    lab = rgb_to_lab(crop.reshape(-1, 3)).reshape(crop.shape)
    fw = _face_width(faces, w)
    enamel = (m > 0.3).astype(np.float32)
    sm = _masked_blur(lab, enamel, max(1.0, 0.03 * fw))  # broad: whole-tooth colour
    new = lab.copy()
    st = float(np.clip(stain_removal / 100.0, 0, 1))
    if st > 0 and enamel.any():
        local = _masked_blur(lab, enamel, max(0.8, 0.006 * fw))  # stain-sized, finer than a tooth
        sel = enamel > 0
        mean_b = float(np.median(local[..., 2][sel]))
        mean_l = float(np.median(local[..., 0][sel]))
        # Stains: patches yellower or darker than the teeth's own typical enamel.
        new[..., 2] -= st * np.maximum(local[..., 2] - mean_b, 0.0)
        new[..., 0] += st * 0.8 * np.maximum(mean_l - local[..., 0], 0.0)
    wt = float(np.clip(whiten / 100.0, 0, 1))
    if wt > 0:
        b_s = sm[..., 2] - (lab[..., 2] - new[..., 2])
        new[..., 2] -= wt * 0.75 * np.maximum(b_s - TEETH_TARGET_B, 0.0)
        new[..., 1] -= wt * 0.4 * np.maximum(sm[..., 1], 0.0)
        new[..., 0] += wt * 0.35 * np.maximum(TEETH_MAX_L - sm[..., 0], 0.0)
    new[..., 0] = np.minimum(new[..., 0], np.maximum(lab[..., 0], TEETH_MAX_L + 4.0))
    rgb = lab_to_rgb(new.reshape(-1, 3)).reshape(crop.shape)
    out = image.copy()
    out[y0:y1, x0:x1] = crop + (rgb - crop) * m[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
