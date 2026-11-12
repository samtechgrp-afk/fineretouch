"""Dark circles (v0.14, engines/under_eye.correct_dark_circles): the under-eye shadow is lifted as a
whole toward the person's own surrounding skin, so texture, the socket's natural shadow and the
skin's colour are kept, on deep and light skin, flat and side lighting."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engines.common import _gauss
from app.engines.under_eye import DARK_CIRCLE_LIFT, correct_dark_circles
from tests.perfection_fixtures import make_scene

SIZE = 600


def scene(tone: str, lighting: float, seed: int = 3):
    """Textured skin with a dark (and slightly cool) under-eye patch, a near-black grey eye above it
    and a grey background strip: the old fill pulled those into the under-eye colour."""
    rgb, _, F = make_scene(size=SIZE, tone=tone, seed=seed, with_natural=False, marks=False, lighting=lighting)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    patch = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(patch, (300, 330), (110, 45), 0, 0, 360, 1.0, -1)
    patch = cv2.GaussianBlur(patch, (0, 0), 18)
    lab[..., 0] -= 12.0 * patch
    lab[..., 2] -= 4.0 * patch  # dark circles are cooler than the cheek
    eye = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(eye, (300, 250), (90, 30), 0, 0, 360, 1.0, -1)
    lab[..., 0] = lab[..., 0] * (1 - eye) + 18.0 * eye
    lab[..., 1] *= 1 - eye
    lab[..., 2] *= 1 - eye
    lab[:60, :, 0], lab[:60, :, 1], lab[:60, :, 2] = 30.0, 0.0, 0.0  # grey background
    img = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    skin = np.ones((SIZE, SIZE), np.float32)
    skin[:60] = 0.0
    skin = skin * (1.0 - cv2.dilate(eye, np.ones((15, 15), np.uint8)))
    skin = cv2.GaussianBlur(skin, (0, 0), 4)
    region = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(region, (300, 330), (120, 55), 0, 0, 360, 1.0, -1)
    region = cv2.GaussianBlur(region, (0, 0), 12)
    return img, skin, region, patch, F


def old_fill(img, amount):
    """The pre-v0.14 stage 6 fill, for comparison (wide blur of the whole photo, per-pixel max)."""
    from app.pipeline import S2, cbcr, from_ycbcr, luma

    low2 = _gauss(img, S2 * SIZE)
    return low2, (lambda d: _apply_old(img, low2, d, luma, cbcr, from_ycbcr))


def _apply_old(c, low2, d, luma, cbcr, from_ycbcr):
    y = luma(c)
    cb, cr = cbcr(c)
    cb2, cr2 = cbcr(low2)
    y = y + np.maximum(0.0, luma(low2) * 1.06 - y) * d
    return from_ycbcr(y, cb + (cb2 - cb) * 0.5 * d, cr + (cr2 - cr) * 0.5 * d)


def run(img, skin, region, amount):
    from app.engines.under_eye import UNDER_REF_SIGMA
    from app.pipeline import S1

    F = 0.94 * SIZE
    return correct_dark_circles(img, img, _gauss(img, S1 * SIZE), skin, region, amount, sigma_ref=UNDER_REF_SIGMA * F)


def lab_of(img):
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def texture(L, sel, F):
    hi = L - _gauss(L, max(0.6, 0.0045 * F))
    return float(np.sqrt(np.mean(hi[sel] ** 2)))


CASES = [("II", 0.15), ("II", 0.5), ("V", 0.15), ("VI", 0.5)]


@pytest.mark.parametrize(("tone", "lighting"), CASES)
def test_lifts_the_shadow_and_keeps_texture_colour_and_some_shadow(tone, lighting):
    img, skin, region, patch, F = scene(tone, lighting)
    core = (patch > 0.8) & (region > 0.9)
    cheek = np.zeros_like(core)
    cheek[420:520, 200:400] = True
    lab0 = lab_of(img)
    out = lab_of(run(img, skin, region, 100))
    deficit = 12.0 * patch[core].mean()  # the shadow the fixture added
    lift = out[..., 0][core].mean() - lab0[..., 0][core].mean()
    assert lift > 0.5 * deficit  # visibly lifted
    assert lift < 0.95 * deficit  # part of the natural shadow stays
    # Texture: the high-pass of L* keeps at least 90% of its energy (no flat patch).
    assert texture(out[..., 0], core, F) > 0.9 * texture(lab0[..., 0], core, F)
    # Colour moves toward the cheek, never toward grey: chroma does not drop.
    c0 = np.hypot(lab0[..., 1], lab0[..., 2])[core].mean()
    c1 = np.hypot(out[..., 1], out[..., 2])[core].mean()
    assert c1 > c0 - 0.5
    db_cheek = lab0[..., 2][cheek].mean() - lab0[..., 2][core].mean()
    assert out[..., 2][core].mean() - lab0[..., 2][core].mean() > 0.3 * db_cheek


@pytest.mark.parametrize(("tone", "lighting"), CASES)
def test_old_fill_flattened_or_greyed_what_the_new_one_keeps(tone, lighting):
    img, skin, region, patch, F = scene(tone, lighting)
    core = (patch > 0.8) & (region > 0.9)
    _, apply = old_fill(img, 100)
    old = lab_of(apply(region))
    new = lab_of(run(img, skin, region, 100))
    lab0 = lab_of(img)
    t0 = texture(lab0[..., 0], core, F)
    c0 = np.hypot(lab0[..., 1], lab0[..., 2])[core].mean()
    old_bad = texture(old[..., 0], core, F) < 0.8 * t0 or np.hypot(old[..., 1], old[..., 2])[core].mean() < c0 - 1.0
    assert old_bad  # the fixture reproduces the reported flat / grey patch
    assert texture(new[..., 0], core, F) > 0.9 * t0


def test_amount_scales_and_zero_is_identity():
    img, skin, region, patch, _ = scene("IV", 0.15)
    core = (patch > 0.8) & (region > 0.9)
    assert run(img, skin, region, 0) is img
    L = [lab_of(run(img, skin, region, a))[..., 0][core].mean() for a in (25, 50, 100)]
    assert L[0] < L[1] < L[2]
    out = run(img, skin, region, 100)
    outside = region <= 1e-3
    assert np.abs(out - img)[outside].max() < 1e-6


def test_lift_never_exceeds_the_cheek():
    img, skin, region, patch, _ = scene("III", 0.15)
    core = (patch > 0.8) & (region > 0.9)
    out = lab_of(run(img, skin, region, 100))
    cheek = lab_of(img)[..., 0][420:520, 200:400].mean()
    assert out[..., 0][core].mean() < cheek
    assert DARK_CIRCLE_LIFT < 1.0
