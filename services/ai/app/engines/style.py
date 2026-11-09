"""Reference-photo style profile and batch style sync.

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
    from .. import imageio

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
