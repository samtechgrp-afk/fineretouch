import colorsys

import numpy as np
import pytest

from app.color_engine import (
    HSL_CHANNELS,
    apply_color_grading,
    apply_hsl_mixer,
    grade_bounds,
    hsv_to_rgb,
    hue_weights,
    rgb_to_hsv,
    wheel_offset,
    zone_weights,
)
from app.pipeline import render
from app.schemas import Adjustments, ColorGrading, GradeWheel, HslMixer, HslValue

rng = np.random.default_rng(5)


def swatch(h_deg: float, s: float = 0.8, v: float = 0.8) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, s, v)
    return np.full((4, 4, 3), [r, g, b], np.float32)


def mixer(**channels: dict) -> HslMixer:
    return HslMixer(**{k: HslValue(**v) for k, v in channels.items()})


def test_hsv_round_trip_matches_colorsys():
    c = rng.random((32, 32, 3)).astype(np.float32)
    h, s, v = rgb_to_hsv(c)
    for (y, x) in [(0, 0), (5, 7), (31, 31)]:
        eh, es, ev = colorsys.rgb_to_hsv(*c[y, x])
        assert h[y, x] == pytest.approx(eh * 360.0, abs=1e-3) or abs(h[y, x] - eh * 360) > 359.99
        assert s[y, x] == pytest.approx(es, abs=1e-5) and v[y, x] == pytest.approx(ev, abs=1e-5)
    assert np.abs(hsv_to_rgb(h, s, v) - c).max() < 1e-5


def test_hue_weights_are_a_partition_of_unity_and_peak_at_centres():
    hues = np.linspace(0, 359.9, 721, dtype=np.float32)
    w = hue_weights(hues)
    assert np.allclose(w.sum(-1), 1.0, atol=1e-5)
    for i, centre in enumerate([0, 30, 60, 120, 180, 240, 270, 300]):
        assert hue_weights(np.array([centre], np.float32))[0, i] == pytest.approx(1.0)
    # Halfway between yellow (60) and green (120) both share the pixel.
    mid = hue_weights(np.array([90.0], np.float32))[0]
    assert mid[2] == pytest.approx(0.5) and mid[3] == pytest.approx(0.5)


def test_identity_mixer_and_grading_return_input():
    c = rng.random((8, 8, 3)).astype(np.float32)
    assert apply_hsl_mixer(c, HslMixer()) is c
    assert apply_color_grading(c, ColorGrading()) is c


def test_channel_saturation_only_touches_its_band():
    blue, red = swatch(240), swatch(0)
    m = mixer(blue={"saturation": -100})
    out_blue = apply_hsl_mixer(blue, m)
    assert np.ptp(out_blue[0, 0]) < 1e-3  # fully desaturated
    assert np.abs(apply_hsl_mixer(red, m) - red).max() < 1e-6


def test_hue_shift_moves_toward_the_neighbour():
    out = apply_hsl_mixer(swatch(120), mixer(green={"hue": 100}))
    h, _, _ = rgb_to_hsv(out)
    assert h[0, 0] == pytest.approx(150.0, abs=0.5)


def test_luminance_darkens_one_colour_and_spares_greys():
    orange, grey = swatch(30), np.full((4, 4, 3), 0.5, np.float32)
    m = mixer(orange={"luminance": -100})
    assert apply_hsl_mixer(orange, m)[0, 0].max() < orange[0, 0].max() * 0.6
    assert np.abs(apply_hsl_mixer(grey, m) - grey).max() < 1e-6


def test_grading_zones_follow_luma_and_balance():
    g = ColorGrading()
    y = np.array([0.05, 0.5, 0.95], np.float32)
    ws, wm, wh = zone_weights(y, g)
    assert ws[0] > 0.99 and wm[1] > 0.99 and wh[2] > 0.99
    lo, hi, _ = grade_bounds(ColorGrading(balance=100))
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.55)


def test_wheel_offsets_have_zero_luma():
    for hue in (0, 45, 200, 330):
        off = wheel_offset(hue, 100, 0)
        assert float(off @ [0.299, 0.587, 0.114]) == pytest.approx(0.0, abs=1e-6)
        assert np.abs(off).max() > 0.05


def test_split_toning_tints_shadows_and_highlights_separately():
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 3, axis=2).repeat(4, axis=0)
    g = ColorGrading(shadows=GradeWheel(hue=220, saturation=60), highlights=GradeWheel(hue=40, saturation=60))
    out = apply_color_grading(ramp, g)
    dark, bright, mid = out[0, 8], out[0, 56], out[0, 32]
    assert dark[2] > dark[0]  # blue shadows
    assert bright[0] > bright[2]  # warm highlights
    assert np.ptp(mid) < np.ptp(dark) and np.ptp(mid) < np.ptp(bright)


def test_render_applies_mixer_and_grading_and_validates():
    img = (np.stack([swatch(240, v=0.7)[0, 0]] * 64 * 64).reshape(64, 64, 3) * 255).astype(np.uint8)
    adj = Adjustments.model_validate({"hsl": {"blue": {"saturation": -100}}})
    out = render(img, adj, [])
    assert np.ptp(out[10, 10].astype(int)) <= 2
    graded = render(img, Adjustments(colorGrading=ColorGrading(midtones=GradeWheel(hue=0, saturation=100))), [])
    assert graded[10, 10, 0] > img[10, 10, 0] + 10
    with pytest.raises(ValueError):
        Adjustments.model_validate({"hsl": {"red": {"hue": 101}}})
    with pytest.raises(ValueError):
        Adjustments.model_validate({"colorGrading": {"shadows": {"hue": 400}}})
    assert set(HslMixer.model_fields) == set(HSL_CHANNELS)


def test_desaturating_keeps_luma():
    navy = np.full((4, 4, 3), [0.1, 0.13, 0.3], np.float32)
    out = apply_hsl_mixer(navy, mixer(blue={"saturation": -100}))
    luma = np.array([0.299, 0.587, 0.114], np.float32)
    assert float(out[0, 0] @ luma) == pytest.approx(float(navy[0, 0] @ luma), abs=1e-4)
    assert np.ptp(out[0, 0]) < 0.1 * np.ptp(navy[0, 0])
