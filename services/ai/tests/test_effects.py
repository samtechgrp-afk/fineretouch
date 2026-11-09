"""Per-effect strength, 1x/2x/3x multipliers, opacity, enable and global strength.

The curve vectors are the same as packages/shared/src/effects.test.ts so the preview and
export resolve identical effective values. The render tests prove the multiplier changes the
actual exported pixels, not just a number in the UI.
"""

import numpy as np
import pytest

from app.effects import EFFECT_CURVES, EFFECT_SOURCE, boost_strength, resolve_adjustments
from app.pipeline import render
from app.schemas import Adjustments

from .test_natural_retouch import skin_patch

CURVE_VECTORS = [
    (30, 1, 0.8, 100, 30),
    (30, 2, 0.8, 100, 60),
    (30, 3, 0.8, 100, 87.86938680575),
    (40, 2, 0.8, 100, 80),
    (40, 3, 0.8, 100, 97.29329433527),
    (90, 1, 0.8, 100, 90),
    (90, 2, 0.8, 100, 99.86524106002),
    (50, 3, 0.6, 85, 83.15109084593),
    (95, 2, 0.6, 85, 95),
    (0, 3, 0.8, 100, 0),
]

TS_CURVES = {  # EFFECT_CURVES in packages/shared/src/effects.ts
    "skinSmoothing": (0.8, 100), "blemishRemoval": (0.85, 100), "acneRemoval": (0.85, 100),
    "darkSpotRemoval": (0.75, 100), "skinToneUnify": (0.75, 100),
    "darkCircles": (0.75, 100), "eyeBrighten": (0.6, 85), "teethWhiten": (0.6, 85),
    "teethStainRemoval": (0.8, 100), "glareReduction": (0.85, 100), "sharpen": (0.6, 90),
    "strayHair": (0.8, 100), "clothingIron": (0.8, 100),
}


def ctl(multiplier=1, opacity=100, enabled=True):
    return {"enabled": enabled, "multiplier": multiplier, "opacity": opacity}


@pytest.mark.parametrize("value,mult,knee,cap,want", CURVE_VECTORS)
def test_curve_vectors_match_typescript(value, mult, knee, cap, want):
    assert boost_strength(value, mult, knee, cap) == pytest.approx(want, abs=1e-8)


def test_curve_table_matches_typescript():
    assert {k: (c.knee, c.cap) for k, c in EFFECT_CURVES.items()} == TS_CURVES
    assert set(EFFECT_SOURCE) == set(TS_CURVES)


def test_old_edits_resolve_unchanged():
    adj = Adjustments(skinSmoothing=42, blemishRemoval=70, sharpen=33)
    r, spot = resolve_adjustments(adj)
    assert (r.skinSmoothing, r.blemishRemoval, r.sharpen, spot) == (42, 70, 33, 1)


def test_resolve_applies_controls_and_keeps_sliders():
    adj = Adjustments(
        skinSmoothing=30, darkCircles=40, sharpen=50, globalStrength=50, exposure=20,
        effects={"skinSmoothing": ctl(2), "darkCircles": ctl(opacity=50), "sharpen": ctl(3, enabled=False),
                 "blemishRemoval": ctl(3)},
    )
    r, spot = resolve_adjustments(adj)
    assert r.skinSmoothing == pytest.approx(30)
    assert r.darkCircles == pytest.approx(10)
    assert r.sharpen == 0
    assert r.exposure == 20
    assert spot == 3
    assert adj.skinSmoothing == 30 and adj.effects.skinSmoothing.multiplier == 2


def _face_render(adj: Adjustments) -> np.ndarray:
    img, _, _ = skin_patch()
    rgb8 = (img * 255 + 0.5).astype(np.uint8)
    return render(rgb8, adj, []).astype(np.float32)


def test_multiplier_changes_rendered_pixels_monotonically():
    img, _, _ = skin_patch()
    base = (img * 255 + 0.5).astype(np.float32)
    diffs = []
    for m in (1, 2, 3):
        out = _face_render(Adjustments(skinSmoothing=30, skinTexture=60, effects={"skinSmoothing": ctl(m)}))
        diffs.append(float(np.abs(out - base).mean()))
    assert diffs[0] > 0.1
    assert diffs[1] > diffs[0] * 1.3, diffs
    assert diffs[2] > diffs[1] * 1.05, diffs


def test_multiplier_equals_the_matching_plain_slider():
    """2x at 30 renders exactly like 60 at 1x: the multiplier drives the real engine parameter."""
    a = _face_render(Adjustments(skinSmoothing=30, effects={"skinSmoothing": ctl(2)}))
    b = _face_render(Adjustments(skinSmoothing=60))
    assert np.array_equal(a, b)


def test_disable_opacity_and_global_strength_change_output():
    on = _face_render(Adjustments(sharpen=60))
    off = _face_render(Adjustments(sharpen=60, effects={"sharpen": ctl(enabled=False)}))
    none = _face_render(Adjustments())
    assert np.array_equal(off, none)
    assert not np.array_equal(on, none)
    half = _face_render(Adjustments(sharpen=60, effects={"sharpen": ctl(opacity=50)}))
    g_half = _face_render(Adjustments(sharpen=60, globalStrength=50))
    assert np.array_equal(half, g_half)
    d_on, d_half = np.abs(on - none).mean(), np.abs(half - none).mean()
    assert 0.35 * d_on < d_half < 0.65 * d_on
    zero = _face_render(Adjustments(sharpen=60, skinSmoothing=50, globalStrength=0))
    assert np.array_equal(zero, none)


def test_spot_healing_multiplier_reaches_more_spots():
    img, _, _ = skin_patch(5)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for k in range(18):  # faint to strong dark spots
        cy, cx = rng.integers(30, img.shape[0] - 30, 2)
        depth = 0.03 + 0.012 * k
        img -= depth * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))[..., None]
    rgb8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    outs = [render(rgb8, Adjustments(blemishRemoval=45, effects={"blemishRemoval": ctl(m)}), []).astype(np.float32)
            for m in (1, 2, 3)]
    d = [float(np.abs(o - rgb8).sum()) for o in outs]
    assert d[0] > 0 and d[1] > d[0] and d[2] > d[1], d
