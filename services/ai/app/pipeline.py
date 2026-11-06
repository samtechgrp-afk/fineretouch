"""Full-resolution renderer. A NumPy/OpenCV port of the WebGL2 shader chain in
apps/web/src/engine/shaders.ts; any change to one must be mirrored in the other.

Stages (all on float32 sRGB in [0, 1]):
  1. Three Gaussian bands at radii proportional to the long edge (s0 < s1 < s2).
  2. Skin probability (YCbCr box) gated by the face-oval mask from landmarks.
  3. Frequency separation: suppress the mid band (blotches) and part of the fine
     band (texture) on skin, keeping strong edges.
  4. Blemish removal: small dark/red spots relative to the surrounding tone are
     replaced with the low band plus preserved texture.
  5. Skin tone unification: chroma pulled toward the local average.
  5b. Boosted skin (2x/3x) and stray-hair cleanup (retouch_engine; the preview
      approximates these, so they are excluded from the parity guarantee).
  6. Landmark-driven features: eyes, dark circles, teeth.
  7. L*a*b* colour match (covariance transform + tone curve when available).
  8. Global tone: exposure / white balance (linear light), highlights, shadows,
     contrast, saturation, vibrance, sharpening, vignette.
  9. Depth-aware bokeh (retouch_engine.apply_depth_bokeh), when enabled and a depth map is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import color_stats, linear_to_srgb, srgb_to_linear, transfer_lab
from .retouch_engine import apply_boosted_skin_retouch, apply_depth_bokeh, remove_stray_hairs
from .schemas import Adjustments, ColorStats, FaceAnalysis

S0, S1, S2 = 0.0012, 0.004, 0.012
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luma(c: np.ndarray) -> np.ndarray:
    return c @ LUMA


def cbcr(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return cb, cr


def from_ycbcr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb, cr = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def skin_probability(c: np.ndarray) -> np.ndarray:
    cb, cr = cbcr(c)
    return (
        smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.66, 0.71, cr))
        * smoothstep(0.27, 0.32, cb) * (1.0 - smoothstep(0.48, 0.53, cb))
    )


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sigma, 0.6), borderType=cv2.BORDER_REFLECT)


@dataclass
class FaceMasks:
    skin: np.ndarray
    eyes: np.ndarray
    under_eye: np.ndarray
    mouth: np.ndarray
    has_face: bool


def build_masks(faces: list[FaceAnalysis], h: int, w: int) -> FaceMasks:
    """Rasterise normalised landmark polygons into feathered float masks."""
    skin = np.zeros((h, w), np.float32)
    eyes = np.zeros((h, w), np.float32)
    under = np.zeros((h, w), np.float32)
    mouth = np.zeros((h, w), np.float32)
    if not faces:
        return FaceMasks(skin, eyes, under, mouth, False)

    def poly(points) -> np.ndarray:
        return np.round(np.array(points, np.float32) * np.array([w, h], np.float32)).astype(np.int32)

    face_w = 1.0
    for f in faces:
        face_w = max(face_w, f.box[2] * w)
        r = f.regions
        cv2.fillPoly(skin, [poly(r.faceOval)], 1.0)
        cutouts = [p for p in (r.leftEye, r.rightEye, r.lips, r.leftBrow, r.rightBrow) if len(p) >= 3]
        if cutouts:
            pts = [poly(p) for p in cutouts]
            cv2.fillPoly(skin, pts, 0.0)
            cv2.polylines(skin, pts, True, 0.0, thickness=max(1, int(face_w * 0.012)))
        for region, canvas in ((r.leftEye, eyes), (r.rightEye, eyes), (r.leftUnderEye, under),
                               (r.rightUnderEye, under), (r.innerMouth, mouth)):
            if len(region) >= 3:
                cv2.fillPoly(canvas, [poly(region)], 1.0)

    return FaceMasks(
        skin=blur(skin, face_w * 0.025),
        eyes=blur(eyes, face_w * 0.006),
        under_eye=blur(under, face_w * 0.02),
        mouth=blur(mouth, face_w * 0.004),
        has_face=True,
    )


def render(
    rgb8: np.ndarray,
    adj: Adjustments,
    faces: list[FaceAnalysis],
    source_stats: ColorStats | None = None,
    depth: np.ndarray | None = None,
    hair_prob: np.ndarray | None = None,
) -> np.ndarray:
    """rgb8: HxWx3 uint8 (or uint16, e.g. a demosaiced camera RAW) sRGB -> HxWx3 uint8 sRGB.

    depth: optional relative depth (0 subject .. 1 far, any size) for the bokeh stage.
    hair_prob: optional hair probability (HxW) that keeps stray-hair cleanup out of the hairstyle.
    """
    src = rgb8.astype(np.float32) / (65535.0 if rgb8.dtype == np.uint16 else 255.0)
    h, w = src.shape[:2]
    L = float(max(h, w))
    low0 = blur(src, S0 * L)
    low1 = blur(src, S1 * L)
    low2 = blur(src, S2 * L)
    masks = build_masks(faces, h, w)

    p = skin_probability(low0)
    if masks.has_face:
        skin = masks.skin * (0.35 + 0.65 * p)
    else:
        skin = p * 0.8
    skin = skin[..., None]

    c = src.copy()
    smooth = adj.skinSmoothing / 100.0
    texture = adj.skinTexture / 100.0
    fine = src - low1
    mid = low1 - low2

    # 3. Frequency separation.
    if smooth > 0:
        edge = smoothstep(0.08, 0.22, np.abs(luma(fine)) + np.abs(luma(mid)))[..., None]
        k = skin * smooth * (1.0 - edge)
        c = c - k * ((1.0 - texture) * fine + 0.85 * mid)

    # 4. Blemishes: darker or redder than the surrounding tone at the s1..s2 scale.
    if adj.blemishRemoval > 0:
        dark = smoothstep(0.015, 0.05, luma(low2) - luma(low1))
        _, cr1 = cbcr(low1)
        _, cr2 = cbcr(low2)
        red = smoothstep(0.012, 0.035, cr1 - cr2)
        spot = (np.maximum(dark, red) * skin[..., 0])[..., None]
        clean = low2 + fine * texture
        c = c + (clean - c) * spot * (adj.blemishRemoval / 100.0)

    # 5. Skin tone unification.
    if adj.skinToneUnify > 0:
        y = luma(c)
        cb, cr = cbcr(c)
        cb2, cr2 = cbcr(low2)
        t = (adj.skinToneUnify / 100.0) * 0.8 * skin[..., 0]
        c = from_ycbcr(y, cb + (cb2 - cb) * t, cr + (cr2 - cr) * t)

    # 5b. Boosted retouch and flyaway hair.
    if adj.skinMultiplier > 1:
        c = apply_boosted_skin_retouch(np.clip(c, 0.0, 1.0).astype(np.float32), skin[..., 0], float(adj.skinMultiplier))
    if adj.strayHairRemoval.enabled and faces:
        c = remove_stray_hairs(np.clip(c, 0.0, 1.0).astype(np.float32), faces, adj.strayHairRemoval.strength, hair_prob)

    # 6. Face features.
    if masks.has_face:
        if adj.eyeBrighten > 0:
            e = (adj.eyeBrighten / 100.0) * masks.eyes[..., None]
            c = c * (1.0 + 0.35 * e) + 0.6 * e * (src - low0)
        if adj.darkCircles > 0:
            d = (adj.darkCircles / 100.0) * masks.under_eye
            y = luma(c)
            cb, cr = cbcr(c)
            cb2, cr2 = cbcr(low2)
            y = y + np.maximum(0.0, luma(low2) * 1.06 - y) * d
            c = from_ycbcr(y, cb + (cb2 - cb) * 0.5 * d, cr + (cr2 - cr) * 0.5 * d)
        if adj.teethWhiten > 0:
            y = luma(c)
            tp = smoothstep(0.30, 0.55, y) * masks.mouth * (adj.teethWhiten / 100.0)
            neutral = np.maximum(c[..., 2], (c[..., 0] + c[..., 1]) * 0.5)
            c[..., 2] = c[..., 2] + (neutral - c[..., 2]) * tp
            c = c * (1.0 + 0.08 * tp)[..., None]

    c = np.clip(c, 0.0, 1.0)

    # 7. Colour match.
    if adj.colorMatch is not None and adj.colorMatch.amount > 0:
        stats = source_stats or color_stats(src)
        c = transfer_lab(c, stats, adj.colorMatch.reference, adj.colorMatch.amount / 100.0)

    # 8. Global tone. Exposure and white balance in linear light.
    lin = srgb_to_linear(c)
    lin = lin * (2.0 ** (adj.exposure / 50.0))
    T, Ti = adj.temperature / 100.0, adj.tint / 100.0
    lin = lin * np.array([1.0 + 0.2 * T, 1.0 - 0.12 * Ti, 1.0 - 0.2 * T], np.float32)
    c = linear_to_srgb(lin)

    y = luma(c)[..., None]
    H, S = adj.highlights / 100.0, adj.shadows / 100.0
    c = c + 0.25 * H * smoothstep(0.5, 1.0, y) + 0.25 * S * (1.0 - smoothstep(0.0, 0.5, y))
    c = (c - 0.5) * (1.0 + 0.5 * adj.contrast / 100.0) + 0.5

    y = luma(c)[..., None]
    sat = (c.max(axis=-1) - c.min(axis=-1))[..., None]
    c = y + (c - y) * (1.0 + adj.saturation / 100.0)
    c = y + (c - y) * (1.0 + (adj.vibrance / 100.0) * (1.0 - np.clip(sat, 0.0, 1.0)))

    if adj.sharpen > 0:
        c = c + (adj.sharpen / 100.0) * 1.2 * (src - low0) * (1.0 - 0.8 * skin * smooth)

    if adj.vignette != 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = (xx + 0.5) / w - 0.5
        dy = (yy + 0.5) / h - 0.5
        d = np.sqrt(dx * dx + dy * dy) / np.sqrt(0.5)
        c = c * (1.0 + (adj.vignette / 100.0) * 0.6 * smoothstep(0.35, 1.0, d))[..., None]

    # 9. Depth-of-field.
    b = adj.depthBokeh
    if b.enabled and depth is not None and b.blurRadius > 0:
        c = apply_depth_bokeh(
            np.clip(c, 0.0, 1.0).astype(np.float32), depth, b.bokehMultiplier / 100.0, b.blurRadius,
            b.apertureShape, b.highlightBoost,
        )

    return (np.clip(c, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def sharpness(rgb8: np.ndarray) -> float:
    """Variance of the Laplacian on a 1024px grey proxy (scale-normalised blur metric)."""
    h, w = rgb8.shape[:2]
    s = min(1.0, 1024.0 / max(h, w))
    grey = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    if s < 1:
        grey = cv2.resize(grey, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())
