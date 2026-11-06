import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.color import color_stats, match_matrix, rgb_to_lab, transfer_lab
from app.depth import DepthEstimator, guided_filter, portrait_prior
from app.main import app
from app.pipeline import render
from app.retouch_engine import (
    aperture_kernel,
    apply_boosted_skin_retouch,
    apply_depth_bokeh,
    map_style_to_adjustments,
    remove_stray_hairs,
    stray_hair_roi,
    style_profile,
    sync_style_arrays,
)
from app.schemas import Adjustments, FaceAnalysis, FaceRegions

client = TestClient(app)
rng = np.random.default_rng(11)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def skin(h=256, w=256, grain=3.0) -> np.ndarray:
    base = np.array([214, 160, 130], np.float32)
    return np.clip(base + rng.normal(0, grain, (h, w, 1)), 0, 255).astype(np.float32) / 255.0


def face(cx=0.5, cy=0.5, rx=0.3, ry=0.38) -> FaceAnalysis:
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    oval = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]
    lips = [(cx - 0.08, cy + 0.2), (cx + 0.08, cy + 0.2), (cx + 0.08, cy + 0.24), (cx - 0.08, cy + 0.24)]
    return FaceAnalysis(
        box=(cx - rx, cy - ry, 2 * rx, 2 * ry), landmarkCount=478, eyeOpenness=0.8,
        regions=FaceRegions(faceOval=oval, leftEye=[], rightEye=[], leftUnderEye=[], rightUnderEye=[],
                            innerMouth=[], lips=lips, leftBrow=[], rightBrow=[]),
    )


# --- boosted skin -------------------------------------------------------------


def test_boost_multiplier_one_is_identity():
    img = skin()
    assert np.array_equal(apply_boosted_skin_retouch(img, np.ones(img.shape[:2], np.float32), 1.0), img)


def test_boost_removes_dark_spots_and_keeps_texture():
    img = skin(grain=3.0)
    yy, xx = np.mgrid[0:256, 0:256]
    spots = [(70, 80), (150, 170), (190, 60)]
    for y, x in spots:
        img[(yy - y) ** 2 + (xx - x) ** 2 < 4**2] *= 0.72
    mask = np.ones((256, 256), np.float32)
    grey = lambda a: a @ [0.299, 0.587, 0.114]  # noqa: E731
    before = [grey(img[y, x]) for y, x in spots]
    out2 = apply_boosted_skin_retouch(img, mask, 2.0)
    out3 = apply_boosted_skin_retouch(img, mask, 3.0)
    ref = grey(img[20:40, 20:40]).mean()
    for (y, x), b in zip(spots, before):
        assert ref - grey(out3[y, x]) < (ref - b) * 0.4, "spot should be mostly gone at 3x"
        assert grey(out3[y, x]) >= grey(out2[y, x]) - 0.02
    # Not plastic: fine texture away from the spots survives.
    patch = (slice(10, 50), slice(200, 240))
    hp = lambda a: grey(a) - cv2.GaussianBlur(grey(a).astype(np.float32), (0, 0), 2)  # noqa: E731
    assert hp(out3)[patch].std() > hp(img)[patch].std() * 0.6


def test_boost_respects_mask():
    img = skin()
    img[100:110, 100:110] *= 0.6
    out = apply_boosted_skin_retouch(img, np.zeros((256, 256), np.float32), 3.0)
    assert np.array_equal(out, img)


# --- stray hair ---------------------------------------------------------------


def test_stray_hair_roi_excludes_face_centre_and_below_mouth():
    roi = stray_hair_roi([face()], 200, 200)
    assert roi[100, 100] == 0  # face centre
    assert roi[195, 100] == 0  # below the mouth
    assert roi[20, 100] == 1  # forehead / hairline band


def test_stray_hair_removes_thin_strand_inside_band_only():
    h = w = 400
    img = np.full((h, w, 3), [0.92, 0.9, 0.88], np.float32)  # bright studio background
    f = face(0.5, 0.55, 0.22, 0.3)
    oval = np.round(np.array(f.regions.faceOval) * [w, h]).astype(np.int32)
    cv2.fillPoly(img, [oval], (0.84, 0.63, 0.51))
    strand = np.zeros((h, w), np.uint8)
    cv2.line(strand, (150, 40), (215, 120), 1, 1)  # flyaway crossing the forehead outline
    outside = np.zeros((h, w), np.uint8)
    cv2.line(outside, (20, 380), (90, 330), 1, 1)  # a line far from the face must survive
    img[strand > 0] = 0.12
    img[outside > 0] = 0.12
    out = remove_stray_hairs(img, [f], 80)
    on = strand > 0
    assert out[on].mean() > img[on].mean() + 0.35
    assert np.allclose(out[outside > 0], img[outside > 0])


def test_stray_hair_without_faces_is_noop():
    img = skin()
    assert remove_stray_hairs(img, [], 100) is img


# --- bokeh --------------------------------------------------------------------


def test_aperture_kernels_are_normalised():
    for shape in ("circle", "hex"):
        k = aperture_kernel(6.0, shape)
        assert k.sum() == pytest.approx(1.0, abs=1e-5)
    hexk = aperture_kernel(10, "hex")
    c = hexk.shape[0] // 2
    # Flat-topped hexagon: the corner at 0 deg reaches the radius, the flat side at 90 deg does not.
    assert hexk[c, c + 10] > 0 and hexk[c - 10, c] == 0


def test_bokeh_blurs_background_and_keeps_subject_sharp():
    h, w = 300, 400
    img = np.repeat(rng.random((h, w, 1), dtype=np.float32), 3, axis=2) * 0.8 + 0.1
    depth = np.ones((h, w), np.float32)
    depth[100:200, 150:250] = 0.0  # subject in the middle
    out = apply_depth_bokeh(img, depth, blur_factor=2.0, blur_radius=60)
    subj = (slice(110, 190), slice(160, 240))
    bg = (slice(10, 80), slice(10, 120))
    assert np.abs(out[subj] - img[subj]).max() < 1e-4
    assert out[bg].std() < img[bg].std() * 0.25
    triple = apply_depth_bokeh(img, depth, blur_factor=3.0, blur_radius=60)
    assert triple[bg].std() <= out[bg].std()


def test_bokeh_highlights_bloom_into_discs():
    img = np.full((200, 200, 3), 0.15, np.float32)
    img[100, 100] = 1.0
    depth = np.ones((200, 200), np.float32)
    plain = apply_depth_bokeh(img, depth, 1.0, 100, highlight_boost=0)
    bloom = apply_depth_bokeh(img, depth, 1.0, 100, highlight_boost=100)
    assert bloom[100, 100].mean() > plain[100, 100].mean()
    assert bloom[100, 102].mean() > plain[100, 102].mean() + 0.02  # a bright disc, not a faint smear


def test_bokeh_zero_radius_is_identity():
    img = skin(64, 64)
    assert apply_depth_bokeh(img, np.ones((64, 64), np.float32), 3.0, 0) is img


# --- depth --------------------------------------------------------------------


def test_guided_filter_preserves_edges():
    g = np.zeros((64, 64), np.float32)
    g[:, 32:] = 1.0
    noisy = g + rng.normal(0, 0.1, g.shape).astype(np.float32)
    out = guided_filter(g, noisy, 4, 1e-3)
    assert out[:, 36:].mean() - out[:, :28].mean() > 0.9
    assert out[:, 40:].std() < noisy[:, 40:].std() * 0.5


def test_depth_prior_puts_subject_near():
    est = DepthEstimator(midas_path="/nonexistent", segmenter_path="/nonexistent")
    rgb = (skin(240, 320) * 255).astype(np.uint8)
    f = face(0.5, 0.35, 0.12, 0.18)
    d = est.estimate(rgb, [f])
    assert est.backend == "prior"
    assert d[int(0.35 * d.shape[0]), d.shape[1] // 2] < 0.15
    assert d[5, 5] > 0.6
    assert portrait_prior([], 10, 10).max() == 0


def test_depth_endpoint_returns_png():
    r = client.post("/v1/depth", files={"file": ("a.png", png_bytes((skin(120, 160) * 255).astype(np.uint8)), "image/png")})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(r.content))
    assert im.mode == "L" and im.size == (160, 120)


# --- colour / style -------------------------------------------------------------


def test_covariance_transfer_matches_reference_covariance():
    src = np.clip(rng.normal([0.5, 0.45, 0.4], 0.12, (96, 96, 3)), 0, 1).astype(np.float32)
    mix = np.array([[1, 0.3, 0], [0, 1, 0.4], [0.2, 0, 1]], np.float32)
    ref = np.clip(rng.normal(0, 0.08, (96, 96, 3)) @ mix + [0.65, 0.5, 0.3], 0, 1).astype(np.float32)
    s, r = color_stats(src), color_stats(ref)
    m = match_matrix(s, r).astype(np.float64)
    cs = np.array(s.cov).reshape(3, 3) + np.eye(3)
    cr = np.array(r.cov).reshape(3, 3) + np.eye(3)
    assert np.allclose(m @ cs @ m.T, cr, rtol=1e-3, atol=1e-2)
    out = color_stats(transfer_lab(src, s, r, 1.0))
    assert out.mean[1:] == pytest.approx(r.mean[1:], abs=2.0)
    assert out.lQuantiles[8] == pytest.approx(r.lQuantiles[8], abs=2.0)  # median L* via tone curve


def test_style_profile_and_mapping():
    warm = np.clip(skin(128, 128, grain=8) * 255 * [1.05, 0.95, 0.8], 0, 255).astype(np.uint8)
    flat = (skin(128, 128, grain=1.5) * 255 * 0.8).astype(np.uint8)
    pw, pf = style_profile(warm), style_profile(flat)
    assert pw.warmth > pf.warmth
    assert pw.skinTexture is not None and pf.skinTexture is not None and pf.skinTexture < pw.skinTexture
    patch = map_style_to_adjustments(pf, pw, "palette")
    assert patch["skinSmoothing"] > 30 and patch["colorMatch"]["amount"] == 100
    sliders = map_style_to_adjustments(pw, pf, "sliders", 50)
    assert sliders["exposure"] > 0 and sliders["colorMatch"] is None
    Adjustments.model_validate(sliders)
    res = sync_style_arrays(warm, [("a.jpg", flat), ("b.jpg", warm)], "sliders")
    assert [t.name for t in res.targets] == ["a.jpg", "b.jpg"]
    assert res.targets[1].adjustments["exposure"] == 0


def test_style_endpoints():
    ref = png_bytes((skin(80, 80, 6) * 255).astype(np.uint8))
    r = client.post("/v1/style/profile", files={"file": ("r.png", ref, "image/png")})
    assert r.status_code == 200 and r.json()["skinTexture"] is not None
    t = png_bytes((skin(60, 90) * 200).astype(np.uint8))
    r = client.post(
        "/v1/style/sync",
        files=[("reference", ("r.png", ref, "image/png")), ("targets", ("t1.png", t, "image/png")),
               ("targets", ("t2.png", t, "image/png"))],
        data={"mode": "sliders", "amount": "80"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["targets"]) == 2 and body["targets"][0]["adjustments"]["exposure"] > 0
    bad = client.post("/v1/style/sync", files=[("reference", ("r.png", ref, "image/png")), ("targets", ("t.png", t, "image/png"))],
                      data={"mode": "nope"})
    assert bad.status_code == 422


# --- end to end -----------------------------------------------------------------


def test_render_with_all_new_features_end_to_end():
    img = (skin(200, 160, 5) * 255).astype(np.uint8)
    adj = Adjustments(
        skinSmoothing=40, skinMultiplier=3,
        strayHairRemoval={"enabled": True, "strength": 70},
        depthBokeh={"enabled": True, "blurRadius": 80, "bokehMultiplier": 250, "apertureShape": "hex"},
    )
    depth = np.ones((50, 40), np.float32)
    out = render(img, adj, [face()], depth=depth)
    assert out.shape == img.shape and out.dtype == np.uint8
    r = client.post("/v1/render", files={"file": ("a.png", png_bytes(img), "image/png")},
                    data={"adjustments": json.dumps(adj.model_dump())})
    assert r.status_code == 200, r.text


def test_new_fields_are_validated():
    with pytest.raises(Exception):
        Adjustments(skinMultiplier=4)
    with pytest.raises(Exception):
        Adjustments(depthBokeh={"bokehMultiplier": 50})
    assert Adjustments().depthBokeh.enabled is False


def test_lab_helpers_still_round_trip():
    lab = rgb_to_lab(np.array([[0.5, 0.4, 0.3]], np.float32))
    assert lab.shape == (1, 3)
