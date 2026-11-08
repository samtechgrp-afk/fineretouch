"""Separate skin tools (v0.7): acne removal, dark spot removal (with pigmentation) and general
blemish removal each act only on their own targets."""

import numpy as np

from app.pipeline import S1, S2, render
from app.retouch_engine import correct_pigmentation, detect_spots
from app.schemas import Adjustments

from .test_natural_retouch import SKIN, skin_patch as _patch


def skin_patch(seed: int = 3):
    """Even skin with pores (no colour blotches, which would read as faint acne everywhere)."""
    _, pores, blotch = _patch(seed)
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[:] = SKIN
    img += pores[..., None]
    return np.clip(img, 0, 1), pores, blotch

SIZE = 384
yy, xx = np.mgrid[0:SIZE, 0:SIZE]


def disc(cy, cx, r):
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))[..., None]


RED_SPOTS = [(80, 90), (150, 260), (260, 120)]
DARK_SPOTS = [(90, 250), (220, 220), (300, 300)]


def portrait_with_spots() -> np.ndarray:
    img, _, _ = skin_patch(11)
    img = img.copy()
    for cy, cx in RED_SPOTS:  # inflamed acne: redder, slightly darker
        img += disc(cy, cx, 4.0) * np.array([0.06, -0.09, -0.07], np.float32)
    for cy, cx in DARK_SPOTS:  # brown marks: darker, not red
        img += disc(cy, cx, 4.0) * np.array([-0.16, -0.15, -0.13], np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def change_at(out: np.ndarray, src: np.ndarray, pts) -> float:
    return float(np.mean([np.abs(out[cy - 3:cy + 4, cx - 3:cx + 4] - src[cy - 3:cy + 4, cx - 3:cx + 4]).mean() for cy, cx in pts]))


def run(**kw) -> tuple[np.ndarray, np.ndarray]:
    src = (portrait_with_spots() * 255 + 0.5).astype(np.uint8)
    return render(src, Adjustments(**kw), []).astype(np.float32), src.astype(np.float32)


def test_detector_classifies_acne_and_dark_spots():
    img = portrait_with_spots()
    mask = np.ones(img.shape[:2], np.float32)
    acne = detect_spots(img, mask, [], 2, 100, True, kind="acne")
    dark = detect_spots(img, mask, [], 2, 100, True, kind="dark")
    for cy, cx in RED_SPOTS:
        assert acne.heal[cy, cx] and not dark.heal[cy, cx]
    for cy, cx in DARK_SPOTS:
        assert dark.heal[cy, cx] and not acne.heal[cy, cx]


def test_acne_removal_heals_red_spots_only():
    out, src = run(acneRemoval=100)
    assert change_at(out, src, RED_SPOTS) > 4 * change_at(out, src, DARK_SPOTS)
    # Clear skin away from any spot is untouched.
    assert np.abs(out[180:200, 20:40] - src[180:200, 20:40]).max() <= 1


def test_dark_spot_removal_heals_dark_spots_only():
    out, src = run(darkSpotRemoval=100)
    assert change_at(out, src, DARK_SPOTS) > 4 * change_at(out, src, RED_SPOTS)


def test_general_blemish_removal_heals_both():
    out, src = run(blemishRemoval=100)
    assert change_at(out, src, DARK_SPOTS) > 3 and change_at(out, src, RED_SPOTS) > 3


def test_keep_freckles_leaves_faint_small_dark_spots():
    img, _, _ = skin_patch(4)
    freckles = [(60 + 40 * i, 80 + 25 * (i % 3)) for i in range(7)]
    for cy, cx in freckles:
        img = img + disc(cy, cx, 1.6) * np.array([-0.05, -0.055, -0.05], np.float32)
    img = np.clip(img, 0, 1).astype(np.float32)
    mask = np.ones(img.shape[:2], np.float32)
    off = detect_spots(img, mask, [], 3, 100, True, kind="dark")
    on = detect_spots(img, mask, [], 3, 100, True, kind="dark", keep_freckles=True)
    assert on.heal.sum() < off.heal.sum()
    assert sum(int(on.heal[cy, cx]) for cy, cx in freckles) < sum(int(off.heal[cy, cx]) for cy, cx in freckles)


def test_pigmentation_evens_brown_patch_but_keeps_neutral_shading():
    img, _, _ = skin_patch(8)
    L = float(SIZE)
    brown = disc(120, 120, 10.0) * np.array([-0.03, -0.05, -0.07], np.float32)  # darker and yellower
    shade = disc(260, 260, 10.0) * np.array([-0.06, -0.06, -0.06], np.float32)  # neutral darkening
    img = np.clip(img + brown + shade, 0, 1).astype(np.float32)
    out = correct_pigmentation(img, np.ones(img.shape[:2], np.float32), S2 * L, 100)
    d_brown = np.abs(out[115:126, 115:126] - img[115:126, 115:126]).mean()
    d_shade = np.abs(out[255:266, 255:266] - img[255:266, 255:266]).mean()
    assert d_brown > 0.004
    assert d_shade < 0.35 * d_brown


def test_new_tools_respond_to_multiplier_and_switch():
    base, src = run()
    a1, _ = run(acneRemoval=40)
    a3, _ = run(acneRemoval=40, effects={"acneRemoval": {"multiplier": 3}})
    off, _ = run(acneRemoval=40, effects={"acneRemoval": {"enabled": False}})
    assert np.abs(a3 - src).sum() > np.abs(a1 - src).sum() > 0
    assert np.array_equal(off, base)
