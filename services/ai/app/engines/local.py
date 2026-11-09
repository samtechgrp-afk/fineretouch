"""Professional retouching tools (v0.10, Phase 5).

* Healing and clone brushes (apply_heal_strokes), replayed on the original pixels before every
  other stage, so the automatic tools see the cleaned photo:
  - clone copies the source area (destination + offset);
  - heal takes the source's texture and the destination's tone and colour: the difference
    between destination and source is known around the stroke and filled smoothly across it
    (push-pull membrane interpolation), so healed = source + filled difference. Inside the
    stroke the pores come from the source while the tone meets the surroundings seamlessly.
* Local retouch layers (apply_local_layers), each with its own mask (AI skin / everything /
  nothing as a base, add and erase strokes, invert, feather, luminance range, density):
  - frequency separation: split at a user radius into a tone layer and a detail layer; the
    tone layer is smoothed edge-aware and the detail layer is kept, reduced or boosted;
  - dodge & burn: painted exposure in linear light (colour-preserving), limited to a tonal range;
  - correction brush: local exposure, contrast, saturation, temperature and tint.

Mirrored in apps/web/src/engine/heal.ts and LOCAL_FRAG (apps/web/src/engine/shaders.ts); the
data model is packages/shared/src/localRetouch.ts.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..color import linear_to_srgb, srgb_to_linear
from ..schemas import HealStroke, LayerStroke, LocalRetouch, RetouchLayer
from .common import _luma, _smoothstep

STROKE_FEATHER_MAX = 1.0  # same as masking.STROKE_FEATHER_MAX
DODGE_BURN_MAX_EV = 1.5
FS_EDGE = (0.04, 0.12)  # tone-layer step (luma) above which smoothing stops at an edge
FS_SMOOTH_RATIO = 5.0  # tone layer smoothed at this many split radii
HEAL_UNKNOWN = 0.002  # coverage above which a pixel is filled rather than known


def split_sigma(split_radius: float, L: float) -> float:
    """Split radius in px (same formula as splitSigma in localRetouch.ts)."""
    return max(0.6, L * 0.0008 * 30.0 ** (min(100.0, max(0.0, split_radius)) / 100.0))


# --- membrane fill ----------------------------------------------------------------------------


def _upsample_bilinear(c: np.ndarray, h: int, w: int) -> np.ndarray:
    """2x bilinear upsample of c (h2 x w2 x C) to h x w, pixel centres aligned, edges clamped."""
    h2, w2 = c.shape[:2]
    sy = np.clip((np.arange(h, dtype=np.float32) + 0.5) / 2.0 - 0.5, 0.0, h2 - 1)
    sx = np.clip((np.arange(w, dtype=np.float32) + 0.5) / 2.0 - 0.5, 0.0, w2 - 1)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h2 - 1)
    x1 = np.minimum(x0 + 1, w2 - 1)
    fy = (sy - y0)[:, None, None]
    fx = (sx - x0)[None, :, None]
    top = c[y0][:, x0] * (1 - fx) + c[y0][:, x1] * fx
    bot = c[y1][:, x0] * (1 - fx) + c[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def push_pull(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown pixels (known == 0) smoothly from the known ones (HxWxC float32)."""
    V = [(values * known[..., None]).astype(np.float32)]
    W = [known.astype(np.float32)]
    while W[-1].shape[0] > 1 or W[-1].shape[1] > 1:
        v, wt = V[-1], W[-1]
        h, w = wt.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        vp = np.zeros((h2 * 2, w2 * 2, v.shape[2]), np.float32)
        wp = np.zeros((h2 * 2, w2 * 2), np.float32)
        vp[:h, :w] = v
        wp[:h, :w] = wt
        v2 = vp.reshape(h2, 2, w2, 2, -1).sum(axis=(1, 3))
        w2_ = wp.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        over = w2_ > 1.0
        v2[over] /= w2_[over][:, None]
        w2_[over] = 1.0
        V.append(v2)
        W.append(w2_)
    c = np.where(W[-1][..., None] > 0, V[-1] / np.maximum(W[-1][..., None], 1e-8), 0.0).astype(np.float32)
    for k in range(len(V) - 2, -1, -1):
        h, w = W[k].shape
        c = V[k] + (1.0 - W[k])[..., None] * _upsample_bilinear(c, h, w)
    return c


# --- brush geometry ---------------------------------------------------------------------------


def _polyline_px(points: list[float], w: int, h: int) -> np.ndarray:
    """Pixel coordinates in float64, like the browser port (same crop bounds to the pixel)."""
    flat = points[: len(points) // 2 * 2]
    return np.asarray(flat, np.float64).reshape(-1, 2) * np.array([w, h], np.float64)


def _distance_to_polyline(pts: np.ndarray, x0: int, y0: int, cw: int, ch: int, reach: float) -> np.ndarray:
    """Distance (px) from each pixel centre of the crop to the polyline; inf beyond reach."""
    d = np.full((ch, cw), np.inf, np.float32)
    segs = [(pts[0], pts[0])] if len(pts) == 1 else list(zip(pts[:-1], pts[1:]))
    for a, b in segs:
        sx0 = max(0, int(math.floor(min(a[0], b[0]) - reach)) - x0)
        sx1 = min(cw, int(math.ceil(max(a[0], b[0]) + reach)) - x0 + 1)
        sy0 = max(0, int(math.floor(min(a[1], b[1]) - reach)) - y0)
        sy1 = min(ch, int(math.ceil(max(a[1], b[1]) + reach)) - y0 + 1)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        py, px = np.mgrid[sy0:sy1, sx0:sx1].astype(np.float32)
        px += x0 + 0.5
        py += y0 + 0.5
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        len2 = dx * dx + dy * dy
        t = np.clip(((px - a[0]) * dx + (py - a[1]) * dy) / len2, 0.0, 1.0) if len2 > 0 else 0.0
        ex = px - (a[0] + t * dx)
        ey = py - (a[1] + t * dy)
        np.minimum(d[sy0:sy1, sx0:sx1], np.sqrt(ex * ex + ey * ey), out=d[sy0:sy1, sx0:sx1])
    return d


def brush_coverage(d: np.ndarray, r: float, hardness: float) -> np.ndarray:
    """Soft brush tip from the distance field: 1 inside hardness * r, smooth falloff to r."""
    inner = r * min(1.0, max(0.0, hardness / 100.0))
    outer = r + 0.5
    t = np.clip((d - inner) / max(outer - inner, 1e-3), 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """img (HxWx3) at pixel-index coordinates xs, ys (same shape), edges clamped."""
    h, w = img.shape[:2]
    xs = np.clip(xs, 0.0, w - 1)
    ys = np.clip(ys, 0.0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


# --- healing and clone ------------------------------------------------------------------------


def apply_heal_stroke(img: np.ndarray, s: HealStroke) -> np.ndarray:
    """One heal / clone stroke on img (HxWx3 float32, modified in place and returned)."""
    h, w = img.shape[:2]
    L = float(max(h, w))
    r = max(1.0, s.size * L)
    pts = _polyline_px(s.points, w, h)
    if pts.size == 0:
        return img
    pad = r + max(3.0, 0.5 * r)
    x0 = max(0, int(math.floor(pts[:, 0].min() - pad)))
    x1 = min(w, int(math.ceil(pts[:, 0].max() + pad)) + 1)
    y0 = max(0, int(math.floor(pts[:, 1].min() - pad)))
    y1 = min(h, int(math.ceil(pts[:, 1].max() + pad)) + 1)
    if x0 >= x1 or y0 >= y1:
        return img
    cw, ch = x1 - x0, y1 - y0
    cov = brush_coverage(_distance_to_polyline(pts, x0, y0, cw, ch, r + 1.0), r, s.hardness)
    a = cov * (s.opacity / 100.0)
    if not np.any(a > 0):
        return img
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    source = sample_bilinear(img, gx + s.offset[0] * w, gy + s.offset[1] * h)
    dest = img[y0:y1, x0:x1]
    if s.mode == "heal":
        known = (cov <= HEAL_UNKNOWN).astype(np.float32)
        if known.any():
            source = source + push_pull(dest - source, known)
    img[y0:y1, x0:x1] = dest + (source - dest) * a[..., None]
    return img


def apply_heal_strokes(img: np.ndarray, strokes: list[HealStroke]) -> np.ndarray:
    """All strokes in order (each sees the result of the previous ones). Returns a new array."""
    if not strokes:
        return img
    out = np.ascontiguousarray(img, dtype=np.float32).copy()
    for s in strokes:
        apply_heal_stroke(out, s)
    return np.clip(out, 0.0, 1.0)


# --- layer masks ------------------------------------------------------------------------------


def paint_strokes(strokes: list[LayerStroke], mode: str, h: int, w: int, feather: float) -> np.ndarray | None:
    """Coverage (HxW 0..1) of the strokes with this mode; flow builds up across strokes."""
    group = [s for s in strokes if s.mode == mode]
    if not group:
        return None
    L = float(max(h, w))
    canvas = np.zeros((h, w), np.float32)
    f = float(np.clip(feather / 100.0, 0.0, 1.0)) * STROKE_FEATHER_MAX
    sigma = 0.0
    for s in group:
        r = max(0.5, s.size * L)
        pts = _polyline_px(s.points, w, h)
        x0 = max(0, int(math.floor(pts[:, 0].min() - r - 2)))
        x1 = min(w, int(math.ceil(pts[:, 0].max() + r + 2)) + 1)
        y0 = max(0, int(math.floor(pts[:, 1].min() - r - 2)))
        y1 = min(h, int(math.ceil(pts[:, 1].max() + r + 2)) + 1)
        sigma = max(sigma, f * r * 0.5)
        if x0 >= x1 or y0 >= y1:
            continue
        d = _distance_to_polyline(pts, x0, y0, x1 - x0, y1 - y0, r + 1.0)
        layer = np.clip(r + 0.5 - d, 0.0, 1.0)
        sub = canvas[y0:y1, x0:x1]
        np.minimum(sub + layer * (s.flow / 100.0), 1.0, out=sub)
    if sigma >= 0.5:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return np.clip(canvas, 0.0, 1.0)


def layer_masks(layer: RetouchLayer, h: int, w: int, skin: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """(coverage, dodge, burn) before the luminance range, density and opacity."""
    m = layer.mask
    add = paint_strokes(m.strokes, "add", h, w, m.feather)
    erase = paint_strokes(m.strokes, "erase", h, w, m.feather)
    keep = 1.0 if erase is None else 1.0 - erase
    if m.base == "all":
        base = np.ones((h, w), np.float32)
    elif m.base == "skin" and skin is not None:
        base = skin.astype(np.float32)
    else:
        base = np.zeros((h, w), np.float32)
    cov = base if add is None else np.maximum(base, add)
    cov = cov * keep
    if m.invert:
        cov = 1.0 - cov
    dodge = burn = None
    if layer.kind == "dodgeBurn":
        gate = skin.astype(np.float32) if (m.base == "skin" and skin is not None) else 1.0
        d = paint_strokes(m.strokes, "dodge", h, w, m.feather)
        b = paint_strokes(m.strokes, "burn", h, w, m.feather)
        dodge = None if d is None else d * keep * gate
        burn = None if b is None else b * keep * gate
    return np.asarray(cov, np.float32), dodge, burn


def luma_gate(y: np.ndarray, lo: float, hi: float, softness: float) -> np.ndarray | float:
    lo, hi = lo / 100.0, hi / 100.0
    if lo <= 0.0 and hi >= 1.0:
        return 1.0
    s = max(0.005, softness / 100.0 * 0.25)
    g = np.ones_like(y)
    if lo > 0.0:
        g = g * _smoothstep(lo - s, lo, y)
    if hi < 1.0:
        g = g * (1.0 - _smoothstep(hi, hi + s, y))
    return g


def tone_range_weight(y: np.ndarray, rng: str) -> np.ndarray | float:
    if rng == "shadows":
        return 1.0 - _smoothstep(0.2, 0.6, y)
    if rng == "highlights":
        return _smoothstep(0.4, 0.8, y)
    if rng == "midtones":
        return 1.0 - _smoothstep(0.15, 0.5, np.abs(y - 0.5))
    return 1.0


# --- layer operations -------------------------------------------------------------------------


def frequency_separation(c: np.ndarray, m: np.ndarray, sigma: float, smoothing: float, texture: float) -> np.ndarray:
    """Split at sigma; smooth the tone layer edge-aware; scale the detail layer; blend by m."""
    ys, xs = np.nonzero(m > 1e-3)
    if ys.size == 0:
        return c
    h, w = m.shape
    p = int(math.ceil(3.0 * FS_SMOOTH_RATIO * sigma)) + 2
    y0, y1 = max(0, ys.min() - p), min(h, ys.max() + p + 1)
    x0, x1 = max(0, xs.min() - p), min(w, xs.max() + p + 1)
    crop = c[y0:y1, x0:x1]
    low = cv2.GaussianBlur(crop, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    low2 = cv2.GaussianBlur(crop, (0, 0), FS_SMOOTH_RATIO * sigma, borderType=cv2.BORDER_REFLECT)
    delta = low2 - low
    edge = 1.0 - _smoothstep(FS_EDGE[0], FS_EDGE[1], np.abs(_luma(delta)))
    tone = low + delta * ((smoothing / 100.0) * edge)[..., None]
    out = tone + (crop - low) * (texture / 100.0)
    c = c.copy()
    c[y0:y1, x0:x1] = crop + (out - crop) * m[y0:y1, x0:x1, None]
    return c


def correct(c: np.ndarray, m: np.ndarray, layer: RetouchLayer) -> np.ndarray:
    lin = srgb_to_linear(c) * (2.0 ** (layer.exposure / 50.0))
    T, Ti = layer.temperature / 100.0, layer.tint / 100.0
    lin = lin * np.array([1.0 + 0.2 * T, 1.0 - 0.12 * Ti, 1.0 - 0.2 * T], np.float32)
    o = linear_to_srgb(lin)
    o = (o - 0.5) * (1.0 + 0.5 * layer.contrast / 100.0) + 0.5
    y = _luma(o)[..., None]
    o = np.clip(y + (o - y) * (1.0 + layer.saturation / 100.0), 0.0, 1.0)
    return c + (o - c) * m[..., None]


def dodge_burn(c: np.ndarray, signed: np.ndarray, layer: RetouchLayer) -> np.ndarray:
    y = _luma(np.clip(c, 0.0, 1.0))
    stops = DODGE_BURN_MAX_EV * (layer.dodgeBurnStrength / 100.0) * signed * tone_range_weight(y, layer.dodgeBurnRange)
    return linear_to_srgb(srgb_to_linear(c) * np.exp2(stops)[..., None])


def layer_is_active(l: RetouchLayer) -> bool:
    """Same rule as layerIsActive in localRetouch.ts."""
    m = l.mask
    if not l.enabled or l.opacity <= 0 or m.density <= 0:
        return False
    if l.kind == "dodgeBurn":
        return l.dodgeBurnStrength > 0 and any(s.mode in ("dodge", "burn") for s in m.strokes)
    fs = l.toneSmoothing > 0 or l.texture != 100
    corr = l.exposure != 0 or l.contrast != 0 or l.saturation != 0 or l.temperature != 0 or l.tint != 0
    if not fs and not corr:
        return False
    return m.base != "none" or m.invert or any(s.mode == "add" for s in m.strokes)


def apply_local_layers(c: np.ndarray, local: LocalRetouch, skin: np.ndarray | None) -> np.ndarray:
    """Every enabled layer in order, bottom first (c: HxWx3 float32 sRGB 0..1)."""
    layers = [l for l in local.layers if layer_is_active(l)]
    if not layers:
        return c
    h, w = c.shape[:2]
    L = float(max(h, w))
    c = np.clip(c, 0.0, 1.0).astype(np.float32)
    for layer in layers:
        cov, dodge, burn = layer_masks(layer, h, w, skin)
        m = layer.mask
        gate = luma_gate(_luma(c), m.lumaMin, m.lumaMax, m.lumaSoftness) * (m.density / 100.0) * (layer.opacity / 100.0)
        if layer.kind == "dodgeBurn":
            signed = np.zeros((h, w), np.float32)
            if dodge is not None:
                signed += dodge
            if burn is not None:
                signed -= burn
            c = dodge_burn(c, signed * gate, layer)
            continue
        wgt = (cov * gate).astype(np.float32)
        if layer.toneSmoothing > 0 or layer.texture != 100:
            c = frequency_separation(c, wgt, split_sigma(layer.splitRadius, L), layer.toneSmoothing, layer.texture)
        if layer.exposure or layer.contrast or layer.saturation or layer.temperature or layer.tint:
            c = correct(c, wgt, layer)
        c = np.clip(c, 0.0, 1.0).astype(np.float32)
    return c
