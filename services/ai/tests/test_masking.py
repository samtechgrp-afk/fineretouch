"""Semantic skin mask (face parsing) and user mask strokes (Phase 3)."""

import numpy as np

from app.masking import FaceParse, apply_mask_edits, rasterize_strokes
from app.pipeline import build_masks, render
from app.schemas import Adjustments, MaskStroke, SkinMaskSettings

from .test_natural_retouch import skin_patch
from .test_pipeline import oval_face


def stroke(mode, pts, size=0.05):
    return MaskStroke(mode=mode, size=size, points=[v for p in pts for v in p])


def test_strokes_rasterize_where_drawn_and_feather():
    add, erase = rasterize_strokes([stroke("add", [(0.2, 0.5), (0.8, 0.5)]), stroke("erase", [(0.5, 0.2)])], 100, 100, 0)
    assert add[50, 50] == 1 and add[50, 5] == 0 and add[10, 50] == 0
    assert erase[20, 50] == 1 and erase[80, 50] == 0
    soft, _ = rasterize_strokes([stroke("add", [(0.2, 0.5), (0.8, 0.5)])], 100, 100, 100)
    assert 0 < soft[44, 50] < 1  # feathered edge


def test_odd_point_lists_are_trimmed():
    add, _ = rasterize_strokes([MaskStroke(mode="add", size=0.05, points=[0.5, 0.5, 0.7])], 50, 50, 0)
    assert add[25, 25] == 1


def test_apply_mask_edits_add_then_erase_wins():
    skin = np.zeros((100, 100), np.float32)
    sm = SkinMaskSettings(feather=0, strokes=[stroke("add", [(0.5, 0.5)], 0.2), stroke("erase", [(0.5, 0.5)], 0.05)])
    out = apply_mask_edits(skin, sm)
    assert out[50, 50] == 0 and out[50, 30] == 1


def _rgb():
    img, _, _ = skin_patch(2)
    return (img * 255 + 0.5).astype(np.uint8)


def test_erase_stroke_protects_area_from_every_skin_tool():
    rgb = _rgb()
    kw = dict(skinSmoothing=80, blemishRemoval=80, skinToneUnify=60, acneRemoval=60, darkSpotRemoval=60)
    plain = render(rgb, Adjustments(**kw), [])
    erased = render(rgb, Adjustments(**kw, skinMask={"feather": 0, "strokes": [stroke("erase", [(0.5, 0.5)], 0.15)]}), [])
    c = slice(170, 214)
    assert np.abs(plain[c, c].astype(int) - rgb[c, c]).mean() > 0.5
    assert np.abs(erased[c, c].astype(int) - rgb[c, c]).max() <= 1
    # Outside the stroke the retouch is unchanged.
    assert np.array_equal(plain[:60, :60], erased[:60, :60])


def test_add_stroke_retouches_non_skin_coloured_area():
    grey = np.full((200, 200, 3), 128, np.uint8)
    grey += np.random.default_rng(0).integers(0, 30, (200, 200, 3), dtype=np.uint8)
    adj = dict(skinSmoothing=100, skinTexture=0)
    off = render(grey, Adjustments(**adj), [])
    on = render(grey, Adjustments(**adj, skinMask={"strokes": [stroke("add", [(0.5, 0.5)], 0.2)]}), [])
    d_off = np.abs(off[90:110, 90:110].astype(int) - grey[90:110, 90:110]).mean()
    d_on = np.abs(on[90:110, 90:110].astype(int) - grey[90:110, 90:110]).mean()
    assert d_on > 1 and d_on > 4 * d_off  # grey is barely "skin" until the user paints it in


def test_face_parsing_extends_skin_and_removes_hair():
    f = oval_face()
    h = w = 200
    base = build_masks([f], h, w)
    face_skin = np.zeros((h, w), np.float32)
    face_skin[20:180, 20:180] = 1.0  # semantic skin reaches beyond the landmark oval
    hair = np.zeros((h, w), np.float32)
    hair[95:105, :] = 1.0  # a strand of hair across the face
    parsed = build_masks([f], h, w, FaceParse(face_skin, np.zeros_like(hair), hair))
    ys, xs = np.nonzero(parsed.skin > 0.5)
    ys0, xs0 = np.nonzero(base.skin > 0.5)
    assert ys.size > ys0.size
    assert parsed.skin[100, 100] < 0.2 < base.skin[100, 100]


def test_body_skin_only_when_requested():
    f = oval_face()
    body = np.zeros((200, 200), np.float32)
    body[185:, :] = 1.0
    p = FaceParse(np.zeros_like(body), body, np.zeros_like(body))
    assert build_masks([f], 200, 200, p).skin[197, 5] < 0.05
    assert build_masks([f], 200, 200, p, include_body=True).skin[197, 5] > 0.9


def test_parse_ignored_when_segmentation_off():
    rgb = _rgb()
    f = oval_face()
    fs = np.ones(rgb.shape[:2], np.float32)
    p = FaceParse(fs, np.zeros_like(fs), np.zeros_like(fs))
    a = render(rgb, Adjustments(skinSmoothing=60, skinMask={"useSegmentation": False}), [f], parse=p)
    b = render(rgb, Adjustments(skinSmoothing=60), [f])
    assert np.array_equal(a, b)
