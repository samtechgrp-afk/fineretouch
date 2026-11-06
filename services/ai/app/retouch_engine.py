"""Advanced retouch stages for the full-resolution renderer (FineRetouch v0.2).

  apply_boosted_skin_retouch  2x/3x multi-stage frequency separation: recursive guided
                              filtering of the low-frequency colour layer plus adaptive
                              blemish detection and patch-based repair of the detail layer.
  remove_stray_hairs          flyaway hairs around the face outline and forehead:
                              morphological + Gabor line detection, then inpainting.
  apply_depth_bokeh           layered disk / hexagon depth-of-field from a depth map,
                              with highlight bloom and a 100-300% radius multiplier.
  style_profile /             reference-photo metrics and the mapping to editable
  extract_and_sync_style      adjustments for a whole batch (port of packages/shared/src/style.ts).

All functions take and return float32 sRGB in [0, 1] unless stated. The WebGL preview
(apps/web/src/engine/shaders.ts) approximates these stages in real time; exports from
this service run the full versions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .color import color_stats, linear_to_srgb, rgb_to_lab, srgb_to_linear
from .depth import guided_filter
from .schemas import Adjustments, ColorMatch, FaceAnalysis, StyleProfile, StyleSyncResponse, StyleSyncTarget

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luma(c: np.ndarray) -> np.ndarray:
    return c @ LUMA


def _cbcr(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5, 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5


def _from_ycbcr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb, cr = cb - 0.5, cr - 0.5
    return np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], axis=-1)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sigma, 0.6), borderType=cv2.BORDER_REFLECT)


def _masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: a blur that only averages pixels inside `mask`."""
    m = mask[..., None] if img.ndim == 3 else mask
    num = _gauss(img * m, sigma)
    den = _gauss(mask, sigma)
    den = den[..., None] if img.ndim == 3 else den
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), img)


def _ellipse(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _inpaint_unit(x: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Telea inpainting of a single float channel in [0, 1] via a 16-bit round trip."""
    u16 = (np.clip(x, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    return cv2.inpaint(u16, mask, radius, cv2.INPAINT_TELEA).astype(np.float32) / 65535.0


def _compact_components(binary: np.ndarray, max_side: float, min_fill: float) -> np.ndarray:
    """Float mask of connected components that are small (bbox side <= max_side) and blob-like
    (area / bbox area >= min_fill). Rejects lines, folds and broad shading."""
    b = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return np.zeros(b.shape, np.float32)
    bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    ok = (np.maximum(bw, bh) <= max_side) & (area / (bw * bh) >= min_fill) & (np.minimum(bw, bh) / np.maximum(bw, bh) > 0.3)
    keep = np.concatenate([[False], ok])
    return keep[labels].astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Boosted skin retouch
# ---------------------------------------------------------------------------


def apply_boosted_skin_retouch(image: np.ndarray, mask: np.ndarray, multiplier: float = 2.0) -> np.ndarray:
    """Multi-stage frequency separation for stubborn skin problems.

    image: HxWx3 float32 sRGB; mask: HxW float32 skin weight in [0, 1].
    multiplier 1.0 returns the image unchanged; 2.0 and 3.0 run 2 and 3 recursive passes.

    Low-frequency layer (colour and tone): recursive guided filtering with the original
    luma as guide (edges and facial contours survive), then dark spots and redness are
    found with an adaptive z-score against a skin-only base and pulled toward it.
    High-frequency layer (texture): blemishes are outliers relative to the local texture
    energy; they are opened (pores are smaller than the kernel), inpainted, and given
    back real grain from a nearby skin patch, so repaired spots never look airbrushed.
    """
    extra = float(np.clip(multiplier - 1.0, 0.0, 2.0))
    if extra <= 0 or mask.max() <= 1e-3:
        return image
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    # Every radius is relative to the size of the skin area (~face width), not the frame,
    # so a head-and-shoulders crop and a full-length shot get the same treatment.
    F = float(np.clip(np.sqrt(float((mask > 0.5).sum())) * 1.1, 32.0, max(image.shape[:2])))
    guide = _luma(image)

    # --- split ---------------------------------------------------------------
    # Classic Gaussian frequency separation puts spots and pores in the detail layer;
    # the colour layer is then smoothed recursively with an edge-preserving guided filter
    # (guide = original luma), so blotches melt while jawline, nose and lips stay put.
    low = _gauss(image, max(1.0, 0.02 * F))
    high = image - low
    passes = 1 + math.ceil(extra)
    r0 = max(2, round(0.015 * F))
    eps = 0.015**2
    sub = max(1, r0 // 4)
    for i in range(passes):
        low = guided_filter(guide, low, r0 * (i + 1), eps * (1 + i), subsample=sub)

    # --- low frequency: hyperpigmentation, redness, uneven tone ----------------
    base = _masked_blur(low, mask, 0.08 * F)
    y_low, y_base = _luma(low), _luma(base)
    cb_low, cr_low = _cbcr(low)
    cb_base, cr_base = _cbcr(base)
    sigma_ctx = 0.12 * F

    def zscore(d: np.ndarray) -> np.ndarray:
        spread = np.sqrt(_masked_blur(d * d, mask, sigma_ctx))
        return d / (spread + 4e-3)

    d_dark = y_base - y_low
    d_red = cr_low - cr_base
    t0 = 1.4 - 0.35 * extra
    dark = _smoothstep(t0, t0 + 1.2, zscore(d_dark)) * (d_dark > 0.006)
    red = _smoothstep(t0, t0 + 1.2, zscore(d_red)) * (d_red > 0.004)
    # Pigment spots are compact; natural shading (eye sockets, folds, jaw shadow) is large
    # or elongated. Keep only compact blobs so contouring survives and nothing goes flat.
    dark = dark * _compact_components(dark > 0.3, max_side=0.12 * F, min_fill=0.3)
    strength = min(1.0, 0.55 + 0.25 * extra)
    lift = np.minimum(np.maximum(d_dark, 0.0), 0.08)
    y_new = y_low + lift * dark * strength
    spot = np.maximum(dark, red) * strength
    even = 0.12 * extra  # gentle global tone evening on top of spot repair
    t_chroma = np.clip(spot + even, 0.0, 1.0)
    low_fixed = _from_ycbcr(
        y_new,
        cb_low + (cb_base - cb_low) * t_chroma,
        cr_low + (cr_base - cr_low) * t_chroma,
    ).astype(np.float32)

    # --- high frequency: acne / blemish patches --------------------------------
    hy = _luma(high)
    energy = np.sqrt(_gauss(hy * hy, 0.05 * F)) + 1e-4
    k = 2.6 - 0.35 * extra
    # Blemishes are darker than their surroundings in the detail layer; strong bright/dark
    # edges (nostrils, folds, lash lines) are long or sit on contour edges and are rejected below.
    core = mask > 0.85
    cand = ((hy < -k * energy) & core).astype(np.uint8)
    pore = round(0.006 * F) | 1
    if pore >= 3:
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _ellipse(pore))  # drop pore-sized specks
    blem = (_compact_components(cand > 0, max_side=0.06 * F, min_fill=0.35) > 0).astype(np.uint8)
    blem = cv2.dilate(blem, _ellipse(max(3, round(0.008 * F) | 1)))

    high_fixed = high
    if blem.any():
        radius = max(2, round(0.015 * F))
        # Inpaint as 16-bit: OpenCV's float path is unreliable across versions, and the
        # detail layer is centred on 0, so offset it into [0, 1] first.
        filled = np.stack([_inpaint_unit(high[..., ch] + 0.5, blem, radius) - 0.5 for ch in range(3)], axis=-1)
        filled = np.clip(filled, -0.5, 0.5)
        # Re-grain: borrow fine luma texture from skin a little way off so the patch keeps pores.
        shift = max(4, round(0.05 * F))
        donor = np.roll(hy, (shift, shift), axis=(0, 1))
        donor_ok = (np.roll(mask, (shift, shift), axis=(0, 1)) > 0.85) & (np.roll(blem, (shift, shift), axis=(0, 1)) == 0)
        donor_fine = donor - _gauss(donor, max(1.0, 0.006 * F))
        donor_fine = np.clip(donor_fine, -1.5 * energy, 1.5 * energy)
        regrain = np.where(donor_ok, donor_fine * 0.85, 0.0)[..., None]
        soft = _gauss(blem.astype(np.float32), max(0.8, 0.004 * F))[..., None]
        high_fixed = high * (1.0 - soft) + (filled + regrain) * soft

    out = low_fixed + high_fixed
    m = mask[..., None]
    return np.clip(image * (1.0 - m) + out * m, 0.0, 1.0).astype(np.float32)


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


# ---------------------------------------------------------------------------
# 3. Depth-aware bokeh
# ---------------------------------------------------------------------------

BOKEH_RADIUS_SHARE = 0.02  # blurRadius 100 => 2% of the long edge, before the multiplier
FOCUS_START, FOCUS_END = 0.04, 0.45  # depth band over which blur ramps up from the subject


def aperture_kernel(radius: float, shape: str = "circle") -> np.ndarray:
    """Normalised, anti-aliased disk or hexagon kernel."""
    r = max(radius, 0.5)
    n = int(math.ceil(r)) + 1
    yy, xx = np.mgrid[-n: n + 1, -n: n + 1].astype(np.float32)
    if shape == "hex":
        ax, ay = np.abs(xx), np.abs(yy)
        dist = np.maximum(ax * math.cos(math.pi / 6) + ay * 0.5, ay) / math.cos(math.pi / 6)
    else:
        dist = np.sqrt(xx * xx + yy * yy)
    k = np.clip(r + 0.5 - dist, 0.0, 1.0)
    return (k / k.sum()).astype(np.float32)


def circle_of_confusion(depth: np.ndarray, max_coc: float) -> np.ndarray:
    return max_coc * _smoothstep(FOCUS_START, FOCUS_END, depth)


def apply_depth_bokeh(
    image: np.ndarray,
    depth_map: np.ndarray,
    blur_factor: float = 2.0,
    blur_radius: float = 50.0,
    aperture: str = "circle",
    highlight_boost: float = 40.0,
    layers: int = 8,
) -> np.ndarray:
    """Layered depth-of-field.

    depth_map: HxW (any size; resized) in [0, 1], 0 = subject plane, 1 = far.
    blur_factor: 1.0-3.0 multiplier on the base radius (the UI's 100-300%).

    Each pixel's circle of confusion picks a pair of radius layers; each layer is
    convolved with the aperture kernel in linear light using normalised convolution,
    so the in-focus subject never bleeds into the background (no halo). Highlights are
    boosted before the blur so bright points bloom into aperture-shaped discs. Large
    radii are processed at a reduced scale (kernels capped at ~32 px) and upsampled,
    which keeps a 24 MP export to a few hundred milliseconds of filtering.
    """
    h, w = image.shape[:2]
    L = float(max(h, w))
    max_coc = (blur_radius / 100.0) * BOKEH_RADIUS_SHARE * L * float(np.clip(blur_factor, 1.0, 3.0))
    if max_coc < 0.75:
        return image
    depth = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coc_full = circle_of_confusion(np.clip(depth, 0, 1), max_coc)

    s = min(1.0, 32.0 / max_coc)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if s < 1 else image
    coc = cv2.resize(coc_full, size, interpolation=cv2.INTER_AREA) * s if s < 1 else coc_full
    max_r = max_coc * s

    lin = srgb_to_linear(small)
    gain = 1.0 + (highlight_boost / 100.0) * 6.0 * _smoothstep(0.55, 0.95, _luma(lin))
    lin = lin * gain[..., None]

    radii = np.linspace(0.0, max_r, layers)
    pos = np.clip(coc / max(max_r, 1e-6) * (layers - 1), 0, layers - 1)
    num = np.zeros_like(lin)
    den = np.zeros(lin.shape[:2], np.float32)
    for i, r in enumerate(radii):
        wgt = np.clip(1.0 - np.abs(pos - i), 0.0, 1.0).astype(np.float32)
        if not wgt.any():
            continue
        if r < 0.5:
            num += lin * wgt[..., None]
            den += wgt
            continue
        k = aperture_kernel(r, aperture)
        num += cv2.filter2D(lin * wgt[..., None], -1, k, borderType=cv2.BORDER_REFLECT)
        den += cv2.filter2D(wgt, -1, k, borderType=cv2.BORDER_REFLECT)
    blurred = linear_to_srgb(np.clip(num / np.maximum(den, 1e-4)[..., None], 0.0, 1.0))
    if s < 1:
        blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = _smoothstep(0.35, 1.5, coc_full)[..., None]
    return np.clip(image * (1.0 - alpha) + blurred * alpha, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Style profile and sync
# ---------------------------------------------------------------------------

PROFILE_LONG_EDGE = 512


def _skin_probability(c: np.ndarray) -> np.ndarray:
    cb, cr = _cbcr(c)
    return (
        _smoothstep(0.50, 0.55, cr) * (1.0 - _smoothstep(0.66, 0.71, cr))
        * _smoothstep(0.27, 0.32, cb) * (1.0 - _smoothstep(0.48, 0.53, cb))
    )


def style_profile(rgb8: np.ndarray) -> StyleProfile:
    """Metrics of a photo's look. Must match profileFromRgba in apps/web/src/engine/style.ts."""
    h, w = rgb8.shape[:2]
    s = min(1.0, PROFILE_LONG_EDGE / max(h, w))
    small = cv2.resize(rgb8, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb8
    f = small.astype(np.float32) / 255.0
    stats = color_stats(f)
    lab = rgb_to_lab(f.reshape(-1, 3))
    chroma = float(np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2).mean())
    grey = (_luma(f) * 255.0).astype(np.float32)
    sharp = float(cv2.Laplacian(grey, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REPLICATE).var())
    hp = np.abs(grey - cv2.blur(grey, (5, 5), borderType=cv2.BORDER_REPLICATE))
    skin = (_skin_probability(f) > 0.5).reshape(-1)
    if skin.sum() >= max(64, 0.01 * skin.size):
        skin_luma: float | None = float(lab[skin, 0].mean())
        skin_tex: float | None = float(hp.reshape(-1)[skin].mean())
    else:
        skin_luma = skin_tex = None
    return StyleProfile(
        stats=stats,
        meanL=float(stats.mean[0]),
        contrast=float(stats.std[0]),
        warmth=float(stats.mean[2]),
        tintAxis=float(stats.mean[1]),
        chroma=chroma,
        sharpness=sharp,
        skinLuma=skin_luma,
        skinTexture=skin_tex,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def exposure_for_lightness(from_l: float, to_l: float) -> float:
    k = ((to_l + 16) / max(from_l + 16, 1)) ** 3
    return _clamp(50 * math.log2(max(k, 1e-3)), -100, 100)


def temperature_for_warmth(db: float, L: float) -> float:
    return _clamp(100 * db / (0.15 * max(L, 10)), -100, 100)


def tint_for_magenta(da: float, L: float) -> float:
    return _clamp(100 * da / (0.078 * max(L, 10)), -100, 100)


def map_style_to_adjustments(ref: StyleProfile, target: StyleProfile, mode: str = "palette", amount: float = 100) -> dict:
    """Port of mapStyleToAdjustments (packages/shared/src/style.ts). Returns a partial adjustments dict."""
    out: dict = {}
    if ref.skinTexture is not None and target.skinTexture is not None and target.skinTexture > 0:
        ratio = ref.skinTexture / target.skinTexture
        out["skinSmoothing"] = round(_clamp((1 - ratio) * 130, 0, 85))
        out["skinTexture"] = round(_clamp(40 + ratio * 30, 35, 80))
        if ratio < 0.75:
            out["blemishRemoval"] = round(_clamp((1 - ratio) * 120, 0, 80))
        if ratio < 0.45:
            out["skinMultiplier"] = 2
    if target.sharpness > 0:
        out["sharpen"] = round(_clamp((ref.sharpness / target.sharpness - 1) * 35, 0, 60))
    amount = _clamp(amount, 0, 100)
    if mode == "palette":
        out["colorMatch"] = ColorMatch(reference=ref.stats, amount=amount).model_dump()
        return out
    L = (ref.meanL + target.meanL) / 2
    k = amount / 100
    tone = {
        "exposure": exposure_for_lightness(target.meanL, ref.meanL),
        "contrast": _clamp(((ref.contrast / max(target.contrast, 1) - 1) / 0.5) * 100, -60, 60),
        "temperature": temperature_for_warmth(ref.warmth - target.warmth, L),
        "tint": tint_for_magenta(ref.tintAxis - target.tintAxis, L),
        "saturation": _clamp((ref.chroma / max(target.chroma, 1) - 1) * 100, -80, 60),
    }
    out.update({key: round(round(v) * k) for key, v in tone.items()})
    out["colorMatch"] = None
    return out


def _read_rgb8(path: str | Path) -> np.ndarray:
    from . import imageio

    return imageio.decode(Path(path).read_bytes(), 2048)


def extract_and_sync_style(
    reference_image_path: str | Path,
    target_image_paths: Sequence[str | Path],
    mode: str = "palette",
    amount: float = 100,
) -> StyleSyncResponse:
    """Profile a reference photo and compute per-target adjustment patches ("Sync to All").

    Each patch is validated against the Adjustments schema, so it can go straight into a
    batch job's per-image `adjustments` override.
    """
    return sync_style_arrays(
        _read_rgb8(reference_image_path),
        [(Path(p).name, _read_rgb8(p)) for p in target_image_paths],
        mode,
        amount,
    )


def sync_style_arrays(
    reference_rgb8: np.ndarray,
    targets_rgb8: Sequence[tuple[str, np.ndarray]],
    mode: str = "palette",
    amount: float = 100,
) -> StyleSyncResponse:
    """In-memory variant of extract_and_sync_style (used by the HTTP endpoint)."""
    ref = style_profile(reference_rgb8)
    targets: list[StyleSyncTarget] = []
    for name, rgb in targets_rgb8:
        prof = style_profile(rgb)
        patch = map_style_to_adjustments(ref, prof, mode, amount)
        Adjustments.model_validate(patch)  # raises if the mapping ever produced an invalid value
        targets.append(StyleSyncTarget(name=name, profile=prof, adjustments=patch))
    return StyleSyncResponse(reference=ref, targets=targets)
