import cv2
import numpy as np
import pytest

from app.color import rgb_to_lab
from app.pipeline import S1, S2, render
from app.retouch_engine import (
    TONE_GUARD_CHROMA_KEEP,
    TONE_GUARD_HUE_DEG,
    adaptive_tone_median,
    natural_skin_retouch,
    skin_tone_guard,
    split_frequencies,
)
from app.schemas import Adjustments

SIZE = 384
L = float(SIZE)
SKIN = np.array([0.82, 0.62, 0.52], np.float32)


def blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigma)


def skin_patch(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat skin + pore texture (fine, high frequency) + red blotches (low frequency)."""
    rng = np.random.default_rng(seed)
    pores = blur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), 0.8)
    pores *= 0.03 / pores.std()
    blotch = blur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), 0.008 * L)
    blotch *= 0.035 / blotch.std()
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[:] = SKIN
    img += pores[..., None]
    img[..., 0] += blotch  # redder
    img[..., 1] -= 0.6 * blotch
    img[..., 2] -= 0.4 * blotch
    return np.clip(img, 0, 1), pores, blotch


def retouch(img: np.ndarray, smoothing: float, retention: float, **kw) -> np.ndarray:
    return natural_skin_retouch(
        img, np.ones(img.shape[:2], np.float32), [], smoothing=smoothing, texture_retention=retention,
        sigma_tone=S1 * L, sigma_base=S2 * L, **kw,
    )


def detail_of(img: np.ndarray) -> np.ndarray:
    return (img - blur(img, S1 * L)) @ np.array([0.299, 0.587, 0.114], np.float32)


def test_split_is_exact():
    img, _, _ = skin_patch()
    tone, detail = split_frequencies(img, 4.0)
    assert np.abs(tone + detail - img).max() < 1e-6
    assert np.abs(detail.mean()) < 1e-3


def test_full_retention_keeps_every_pore_and_still_evens_tone():
    img, _, _ = skin_patch()
    out = retouch(img, smoothing=100, retention=100)
    d_in, d_out = detail_of(img)[16:-16, 16:-16], detail_of(out)[16:-16, 16:-16]
    assert np.corrcoef(d_in.ravel(), d_out.ravel())[0, 1] > 0.97
    assert d_out.std() == pytest.approx(d_in.std(), rel=0.15)
    # Blotchy redness (low frequency) is reduced on the tone layer.
    red = lambda x: blur(x[..., 0] - x[..., 1], 0.006 * L)[16:-16, 16:-16].std()
    assert red(out) < 0.75 * red(img)


def test_texture_retention_sets_how_much_grain_comes_back():
    img, _, _ = skin_patch()
    stds = [detail_of(retouch(img, smoothing=100, retention=r))[16:-16, 16:-16].std() for r in (0, 50, 100)]
    assert stds[0] < stds[1] < stds[2]
    assert stds[0] < 0.4 * stds[2]  # 0% gives the smooth (plastic) look, 100% keeps the grain


def test_zero_smoothing_and_no_spots_is_identity():
    img, _, _ = skin_patch()
    assert retouch(img, smoothing=0, retention=50) is img


def test_median_keeps_moles_but_evens_faint_blotches():
    tone = np.full((200, 200, 3), SKIN, np.float32)
    cv2.circle(tone, (60, 100), 5, (0.45, 0.32, 0.27), -1)  # dark mole
    cv2.circle(tone, (140, 100), 3, tuple(float(v) for v in SKIN + [0.02, -0.012, -0.008]), -1)  # faint blotch
    tone = blur(tone, 1.0)
    out = adaptive_tone_median(tone, np.ones((200, 200), np.float32), sigma=3.0)
    assert np.abs(out[100, 60] - tone[100, 60]).max() < 1e-3
    assert np.abs(out[100, 140] - SKIN).max() < 0.5 * np.abs(tone[100, 140] - SKIN).max()


def test_tone_guard_restores_ashen_skin_and_keeps_lightness():
    orig = np.full((8, 8, 3), SKIN, np.float32)
    grey = np.full_like(orig, float(orig[0, 0] @ [0.299, 0.587, 0.114]))
    out = skin_tone_guard(grey, orig, orig, np.ones((8, 8), np.float32))
    lab_o, lab_g, lab_out = rgb_to_lab(orig[0, 0]), rgb_to_lab(grey[0, 0]), rgb_to_lab(out[0, 0])
    c = lambda lab: float(np.hypot(lab[1], lab[2]))
    assert c(lab_g) < 1.0
    assert c(lab_out) == pytest.approx(TONE_GUARD_CHROMA_KEEP * c(lab_o), rel=0.02)
    assert lab_out[0] == pytest.approx(lab_g[0], abs=0.3)
    hue = lambda lab: np.degrees(np.arctan2(lab[2], lab[1]))
    assert abs(hue(lab_out) - hue(lab_o)) <= TONE_GUARD_HUE_DEG + 0.5


def test_tone_guard_limits_hue_drift_but_leaves_untouched_pixels():
    orig = np.full((4, 4, 3), SKIN, np.float32)
    lab = rgb_to_lab(orig[0, 0])
    ch, h = np.hypot(lab[1], lab[2]), np.arctan2(lab[2], lab[1])
    from app.color import lab_to_rgb

    turned = lab_to_rgb(np.array([lab[0], ch * np.cos(h + np.radians(25)), ch * np.sin(h + np.radians(25))], np.float32))
    img = np.full_like(orig, turned)
    img[0, 0] = orig[0, 0]  # an untouched pixel
    out = skin_tone_guard(img, orig, orig, np.ones((4, 4), np.float32))
    assert np.array_equal(out[0, 0], orig[0, 0])
    lab_out = rgb_to_lab(out[1, 1])
    drift = np.degrees(np.arctan2(lab_out[2], lab_out[1]) - h)
    assert drift == pytest.approx(TONE_GUARD_HUE_DEG, abs=0.6)


def test_render_runs_guard_by_default_and_it_can_be_turned_off():
    img, _, _ = skin_patch()
    rgb8 = (img * 255).astype(np.uint8)
    adj = Adjustments(skinSmoothing=80, skinTexture=70, skinToneUnify=60)
    on = render(rgb8, adj, [])
    off = render(rgb8, adj.model_copy(update={"skinToneGuard": False}), [])
    assert Adjustments().skinToneGuard is True
    # Skin colour stays the person's own: the guard never lowers the mean chroma.
    chroma = lambda x: np.hypot(*rgb_to_lab(x.astype(np.float32) / 255)[..., 1:].transpose(2, 0, 1)).mean()
    assert chroma(on) >= chroma(off) - 0.05
    assert chroma(on) > 0.85 * chroma(rgb8)


def test_sharpen_after_smoothing_masks_skin():
    img, _, _ = skin_patch()
    rgb8 = (img * 255).astype(np.uint8)
    out = render(rgb8, Adjustments(skinSmoothing=60, sharpen=50), [])
    assert out.shape == rgb8.shape


def test_guard_constants_match_the_preview():
    import re
    from pathlib import Path

    from app.retouch_engine import TONE_GUARD_MIN_CHROMA

    shared = Path(__file__).resolve().parents[3] / "packages" / "shared" / "src" / "index.ts"
    if not shared.exists():
        pytest.skip("shared package not present")
    m = re.search(r"SKIN_TONE_GUARD = Object\.freeze\(\{ chromaKeep: ([\d.]+), hueTolerance: ([\d.]+), minChroma: \[([\d.]+), ([\d.]+)\]", shared.read_text())
    assert m, "SKIN_TONE_GUARD not found in packages/shared"
    keep, tol, c0, c1 = map(float, m.groups())
    assert (keep, tol, (c0, c1)) == (TONE_GUARD_CHROMA_KEEP, TONE_GUARD_HUE_DEG, TONE_GUARD_MIN_CHROMA)
