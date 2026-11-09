"""Phase 5 pro tools (v0.10): healing, clone, frequency separation, dodge & burn, correction
brushes and layer masks (app/engines/local.py).

Each test checks the tool does its job and leaves alone what it must not touch: pixels outside
the brush, skin texture, colour when dodging, edges when smoothing.
"""

import cv2
import numpy as np
from app.color import srgb_to_linear
from app.engines.common import _luma
from app.engines.local import (
    apply_heal_strokes,
    apply_local_layers,
    layer_masks,
    push_pull,
    split_sigma,
)
from app.pipeline import render
from app.schemas import Adjustments, HealStroke, LayerMask, LayerStroke, LocalRetouch, RetouchLayer

SIZE = 256
SKIN = np.array([0.78, 0.58, 0.48], np.float32)
yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def pores(seed=3, amp=0.03):
    p = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), 1.2)
    return p * (amp / p.std())


def skin_image(seed=3):
    shade = 0.06 * (xx / SIZE - 0.5)  # gentle gradient, so tone matching matters
    return np.clip(SKIN + (shade + pores(seed))[..., None], 0, 1).astype(np.float32)


def disc(cx, cy, r):
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)


def hp(img, s=2.0):
    """Fine detail (what reads as pores)."""
    y = _luma(img)
    return y - cv2.GaussianBlur(y, (0, 0), s)


def test_push_pull_fills_a_smooth_field_from_its_border():
    field = np.stack([xx / SIZE, yy / SIZE, 0.5 + 0 * xx], axis=-1).astype(np.float32)
    known = 1.0 - disc(128, 128, 40)
    filled = push_pull(field, known)
    assert np.allclose(filled[known > 0], field[known > 0], atol=1e-5)  # known pixels kept exactly
    hole = known == 0
    assert np.abs(filled[hole] - field[hole]).max() < 0.06  # a ramp is continued, roughly


def test_heal_removes_a_blemish_keeps_texture_and_matches_surrounding_tone():
    clean = skin_image()
    img = clean.copy()
    blemish = cv2.GaussianBlur(disc(100, 128, 9), (0, 0), 2.0)
    img -= (blemish * 0.18)[..., None]  # dark spot
    img[..., 0] += blemish * 0.05  # a little red
    # Source 60 px to the right is brighter (gradient + a lift), so a plain copy would show a patch.
    img[:, 150:] += 0.04
    s = HealStroke(mode="heal", size=12 / SIZE, hardness=40, opacity=100, offset=(60 / SIZE, 0.0), points=[100 / SIZE, 128 / SIZE])
    out = apply_heal_strokes(img, [s])
    inner = disc(100, 128, 6) > 0
    # Blemish gone: the centre is back to the clean skin's tone (not the brighter source's).
    assert abs(_luma(out)[inner].mean() - _luma(clean)[inner].mean()) < 0.012
    # Texture is there (from the source), not a flat fill.
    assert hp(out)[inner].std() > 0.6 * hp(clean)[inner].std()
    # Nothing outside the brush (+ its soft edge) changed.
    far = disc(100, 128, 14) == 0
    assert np.array_equal(out[far], np.clip(img, 0, 1)[far])


def test_clone_copies_the_source_and_respects_opacity():
    img = skin_image()
    img[120:136, 180:196] = [0.1, 0.2, 0.9]  # a distinct patch to clone
    s = HealStroke(mode="clone", size=6 / SIZE, hardness=100, opacity=100, offset=(60 / SIZE, 0.0), points=[128 / SIZE, 128 / SIZE])
    out = apply_heal_strokes(img, [s])
    assert np.allclose(out[126:130, 126:130], [0.1, 0.2, 0.9], atol=1e-4)
    half = apply_heal_strokes(img, [s.model_copy(update={"opacity": 50})])
    expect = 0.5 * img[128, 128] + 0.5 * np.array([0.1, 0.2, 0.9])
    assert np.allclose(half[128, 128], expect, atol=1e-3)


def test_heal_strokes_are_replayed_in_order():
    img = skin_image()
    img[100:110, 200:210] = [0.9, 0.1, 0.1]
    a = HealStroke(mode="clone", size=4 / SIZE, hardness=100, offset=(80 / SIZE, 0.0), points=[125 / SIZE, 105 / SIZE])
    b = HealStroke(mode="clone", size=4 / SIZE, hardness=100, offset=(80 / SIZE, 0.0), points=[45 / SIZE, 105 / SIZE])
    out = apply_heal_strokes(img, [a, b])  # b clones what a just painted
    assert np.allclose(out[105, 45], [0.9, 0.1, 0.1], atol=1e-3)


def fs_layer(**kw):
    mask = kw.pop("mask", LayerMask(base="all"))
    return RetouchLayer(kind="frequency", mask=mask, **kw)


def test_frequency_separation_smooths_tone_and_keeps_pores():
    blotch = 0.05 * np.sin(xx / 9.0) * np.sin(yy / 11.0)
    img = np.clip(skin_image() + blotch[..., None], 0, 1).astype(np.float32)
    out = apply_local_layers(img, LocalRetouch(layers=[fs_layer(splitRadius=100, toneSmoothing=100, texture=100)]), None)
    sigma = split_sigma(100, SIZE)
    tone = lambda im: cv2.GaussianBlur(_luma(im), (0, 0), sigma)
    core = (slice(40, 216), slice(40, 216))
    assert tone(out)[core].std() < 0.75 * tone(img)[core].std()  # blotches evened
    assert hp(out)[core].std() > 0.9 * hp(img)[core].std()  # pores untouched
    flat = apply_local_layers(img, LocalRetouch(layers=[fs_layer(splitRadius=100, texture=0)]), None)
    assert hp(flat)[core].std() < 0.3 * hp(img)[core].std()  # texture slider really drives detail


def test_frequency_separation_does_not_smear_a_strong_edge():
    img = np.where((xx < 128)[..., None], SKIN, SKIN * 0.35).astype(np.float32)
    out = apply_local_layers(img, LocalRetouch(layers=[fs_layer(splitRadius=70, toneSmoothing=100)]), None)
    # Smoothing the tone layer would pull the two sides together across the jawline.
    assert abs(_luma(out)[:, 100].mean() - _luma(img)[:, 100].mean()) < 0.01
    assert abs(_luma(out)[:, 156].mean() - _luma(img)[:, 156].mean()) < 0.01


def test_layer_mask_strokes_invert_and_luminance_range():
    img = skin_image() * 0.5  # mid-grey skin, so +1 EV does not clip
    stroke = LayerStroke(mode="add", size=20 / SIZE, flow=100, points=[64 / SIZE, 64 / SIZE])
    layer = RetouchLayer(kind="correction", exposure=50, mask=LayerMask(base="none", feather=0, strokes=[stroke]))
    out = apply_local_layers(img, LocalRetouch(layers=[layer]), None)
    gain = srgb_to_linear(out[64, 64]) / srgb_to_linear(img[64, 64])
    assert np.allclose(gain, 2.0, atol=0.03)  # +50 = +1 EV, like the global exposure slider
    assert np.array_equal(out[200, 200], img[200, 200])
    inv = apply_local_layers(img, LocalRetouch(layers=[layer.model_copy(update={"mask": layer.mask.model_copy(update={"invert": True})})]), None)
    assert np.allclose(inv[64, 64], img[64, 64], atol=1e-6) and inv[200, 200].mean() > img[200, 200].mean() + 0.05
    # Luminance range 0..20: only dark pixels change.
    dark = img.copy()
    dark[:, :128] *= 0.3
    ranged = RetouchLayer(kind="correction", exposure=50, mask=LayerMask(base="all", lumaMin=0, lumaMax=20, lumaSoftness=10))
    o2 = apply_local_layers(dark, LocalRetouch(layers=[ranged]), None)
    assert o2[128, 40].mean() > dark[128, 40].mean() + 0.03
    assert np.allclose(o2[128, 200], dark[128, 200], atol=1e-4)


def test_flow_builds_up_and_erase_removes():
    a = LayerStroke(mode="add", size=10 / SIZE, flow=40, points=[0.5, 0.5])
    layer = RetouchLayer(kind="correction", exposure=10, mask=LayerMask(feather=0, strokes=[a]))
    cov1, _, _ = layer_masks(layer, SIZE, SIZE, None)
    layer2 = layer.model_copy(update={"mask": layer.mask.model_copy(update={"strokes": [a, a]})})
    cov2, _, _ = layer_masks(layer2, SIZE, SIZE, None)
    assert abs(cov1[128, 128] - 0.4) < 1e-5 and abs(cov2[128, 128] - 0.8) < 1e-5
    e = LayerStroke(mode="erase", size=10 / SIZE, points=[0.5, 0.5])
    layer3 = layer.model_copy(update={"mask": layer.mask.model_copy(update={"strokes": [a, e]})})
    assert layer_masks(layer3, SIZE, SIZE, None)[0][128, 128] == 0


def test_dodge_and_burn_change_exposure_but_keep_colour():
    img = skin_image()
    d = LayerStroke(mode="dodge", size=16 / SIZE, points=[64 / SIZE, 128 / SIZE])
    b = LayerStroke(mode="burn", size=16 / SIZE, points=[192 / SIZE, 128 / SIZE])
    layer = RetouchLayer(kind="dodgeBurn", dodgeBurnStrength=50, mask=LayerMask(base="all", feather=0, strokes=[d, b]))
    out = apply_local_layers(img, LocalRetouch(layers=[layer]), None)
    assert _luma(out)[128, 64] > _luma(img)[128, 64] + 0.05
    assert _luma(out)[128, 192] < _luma(img)[128, 192] - 0.05
    # Linear-light scaling: channel ratios (chromaticity) unchanged.
    for x in (64, 192):
        r0 = srgb_to_linear(img[128, x])
        r1 = srgb_to_linear(out[128, x])
        assert np.allclose(r1 / r1[1], r0 / r0[1], atol=0.01)
    assert np.allclose(out[20, 20], img[20, 20], atol=1e-5)
    # Highlights range: a dark pixel under the dodge brush stays put.
    dark = img * 0.25
    hl = layer.model_copy(update={"dodgeBurnRange": "highlights"})
    o2 = apply_local_layers(dark, LocalRetouch(layers=[hl]), None)
    assert abs(_luma(o2)[128, 64] - _luma(dark)[128, 64]) < 0.005


def test_skin_base_follows_the_skin_mask_and_disabled_layers_do_nothing():
    img = skin_image()
    skin = np.zeros((SIZE, SIZE), np.float32)
    skin[:, :128] = 1.0
    layer = RetouchLayer(kind="correction", saturation=-100, mask=LayerMask(base="skin"))
    out = apply_local_layers(img, LocalRetouch(layers=[layer]), skin)
    assert np.ptp(out[100, 50]) < 1e-3 and np.ptp(out[100, 200]) > 0.1
    off = apply_local_layers(img, LocalRetouch(layers=[layer.model_copy(update={"enabled": False})]), skin)
    assert np.array_equal(off, img)


def test_render_applies_pro_tools_and_keeps_the_rest():
    rgb8 = (skin_image() * 255 + 0.5).astype(np.uint8)
    rgb8[60:70, 60:70] = [30, 20, 20]
    adj = Adjustments(localRetouch=LocalRetouch(
        heal=[HealStroke(mode="heal", size=10 / SIZE, offset=(0.0, 0.3), points=[65 / SIZE, 65 / SIZE])],
        layers=[RetouchLayer(kind="correction", exposure=30, mask=LayerMask(feather=0, strokes=[
            LayerStroke(mode="add", size=12 / SIZE, points=[190 / SIZE, 190 / SIZE])]))],
    ))
    out = render(rgb8, adj, [])
    assert out[65, 65].mean() > 120  # the dark square is healed
    assert out[190, 190].mean() > rgb8[190, 190].mean() + 10  # the correction brush brightened
    assert np.abs(out[10:40, 150:250].astype(int) - rgb8[10:40, 150:250].astype(int)).max() <= 1
    # Default edit: identical to rendering without the field.
    assert np.array_equal(render(rgb8, Adjustments(), []), render(rgb8, Adjustments(localRetouch=LocalRetouch()), []))


def test_schema_accepts_the_shared_model_json():
    data = {
        "localRetouch": {
            "heal": [{"mode": "clone", "size": 0.03, "hardness": 70, "opacity": 80, "offset": [0.1, -0.05], "points": [0.2, 0.2, 0.25, 0.22]}],
            "layers": [{
                "id": "l1", "name": "D&B", "kind": "dodgeBurn", "enabled": True, "opacity": 90,
                "mask": {"base": "skin", "feather": 30, "invert": False, "lumaMin": 10, "lumaMax": 90, "lumaSoftness": 20, "density": 80,
                         "strokes": [{"mode": "burn", "size": 0.02, "flow": 25, "points": [0.5, 0.5]}]},
                "splitRadius": 40, "toneSmoothing": 0, "texture": 100, "dodgeBurnStrength": 40, "dodgeBurnRange": "midtones",
                "exposure": 0, "contrast": 0, "saturation": 0, "temperature": 0, "tint": 0,
            }],
        }
    }
    a = Adjustments.model_validate(data)
    assert a.localRetouch.heal[0].offset == (0.1, -0.05)
    assert a.localRetouch.layers[0].mask.strokes[0].flow == 25


# Same image, strokes and expected pixels as apps/web/src/engine/heal.test.ts, so the browser
# preview / export and this engine heal identically (to uint8 rounding).
HEAL_VECTOR_STROKES = [
    {"mode": "heal", "size": 0.12, "hardness": 30, "opacity": 100, "offset": [0.4, 0.05], "points": [0.29, 0.5, 0.31, 0.52]},
    {"mode": "clone", "size": 0.08, "hardness": 70, "opacity": 60, "offset": [-0.3, 0.1], "points": [0.75, 0.3]},
]
HEAL_VECTORS = {
    (14, 20): [206.48, 166.48, 146.48],
    (12, 22): [168.12, 131.09, 112.57],
    (16, 19): [187.13, 147.47, 127.64],
    (36, 12): [191.56, 151.56, 131.56],
    (34, 13): [183.90, 143.90, 123.90],
    (5, 5): [190.0, 150.0, 130.0],
}


def test_heal_vectors_match_typescript():
    h, w = 40, 48
    y, x = np.mgrid[0:h, 0:w]
    base = (120 + ((x * 37 + y * 91) % 61)).astype(np.float32)
    img = np.stack([base + 40, base, base - 20], -1)
    img[(x - 14) ** 2 + (y - 20) ** 2 <= 16] = [60, 40, 30]
    img = np.clip(img, 0, 255).astype(np.uint8).astype(np.float32) / 255
    out = apply_heal_strokes(img, [HealStroke(**s) for s in HEAL_VECTOR_STROKES]) * 255
    for (px, py), want in HEAL_VECTORS.items():
        assert np.allclose(out[py, px], want, atol=0.01), (px, py, out[py, px])
