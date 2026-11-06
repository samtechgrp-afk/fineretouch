import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.color import color_stats, lab_to_rgb, rgb_to_lab, transfer_lab
from app.main import app
from app.pipeline import build_masks, render, skin_probability
from app.schemas import Adjustments, ColorStats, FaceAnalysis, FaceRegions

client = TestClient(app)
rng = np.random.default_rng(7)


def jpeg_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def skin_patch(h: int = 256, w: int = 256, noise: float = 14.0, block: int = 4) -> np.ndarray:
    base = np.array([214, 160, 130], np.float32)  # typical light skin tone
    blotches = np.kron(rng.normal(0, noise, (-(-h // block), -(-w // block), 1)), np.ones((block, block, 1)))[:h, :w]
    grain = rng.normal(0, 4, (h, w, 1))
    return np.clip(base + blotches + grain, 0, 255).astype(np.uint8)


def oval_face() -> FaceAnalysis:
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    oval = [(0.5 + 0.4 * np.cos(a), 0.5 + 0.45 * np.sin(a)) for a in t]
    empty: list = []
    return FaceAnalysis(
        box=(0.1, 0.05, 0.8, 0.9),
        landmarkCount=478,
        regions=FaceRegions(faceOval=oval, leftEye=empty, rightEye=empty, leftUnderEye=empty, rightUnderEye=empty,
                            innerMouth=[(0.4, 0.7), (0.6, 0.7), (0.6, 0.78), (0.4, 0.78)], lips=empty,
                            leftBrow=empty, rightBrow=empty),
        eyeOpenness=0.8,
    )


def test_lab_round_trip():
    rgb = rng.random((64, 64, 3), dtype=np.float32)
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(back - rgb).max() < 2e-3


def test_lab_reference_values():
    lab = rgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32))
    assert lab[0] == pytest.approx([100, 0, 0], abs=0.05)
    assert lab[1] == pytest.approx([0, 0, 0], abs=0.05)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def test_color_transfer_matches_reference_statistics():
    src = rng.random((128, 128, 3), dtype=np.float32) * 0.5 + 0.25
    ref = np.clip(rng.normal([0.7, 0.5, 0.3], 0.08, (128, 128, 3)), 0, 1).astype(np.float32)
    s, r = color_stats(src), color_stats(ref)
    out = color_stats(transfer_lab(src, s, r, 1.0))
    assert out.mean == pytest.approx(r.mean, abs=1.5)
    assert out.std == pytest.approx(r.std, rel=0.2)


def test_default_adjustments_are_identity():
    img = skin_patch()
    out = render(img, Adjustments(), [])
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_skin_probability_separates_skin_from_background():
    skin = np.array([[[214, 160, 130]]], np.float32) / 255
    sky = np.array([[[90, 140, 220]]], np.float32) / 255
    assert skin_probability(skin)[0, 0] > 0.9
    assert skin_probability(sky)[0, 0] < 0.05


def test_frequency_separation_reduces_blotches_but_keeps_mean():
    img = skin_patch()
    out = render(img, Adjustments(skinSmoothing=100, skinTexture=50), [oval_face()])
    centre = (slice(80, 176), slice(80, 176))
    before = img[centre].astype(np.float32) @ [0.299, 0.587, 0.114]
    after = out[centre].astype(np.float32) @ [0.299, 0.587, 0.114]
    assert after.std() < before.std() * 0.7
    assert abs(after.mean() - before.mean()) < 3


def test_blemish_removal_lightens_dark_spot():
    img = np.full((256, 256, 3), [214, 160, 130], np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    spot = (yy - 128) ** 2 + (xx - 128) ** 2 < 3**2
    img[spot] = [150, 95, 80]
    # A spot this dark and neutral reads as a beauty mark, so it is kept unless protection is off.
    kept = render(img, Adjustments(blemishRemoval=100), [oval_face()])
    assert abs(kept[128, 128].astype(int).sum() - img[128, 128].astype(int).sum()) < 25
    out = render(img, Adjustments(blemishRemoval=100, protectBeautyMarks=False), [oval_face()])
    assert out[128, 128].astype(int).sum() > img[128, 128].astype(int).sum() + 60


def test_masks_are_empty_without_faces_and_feathered_with_one():
    empty = build_masks([], 100, 100)
    assert not empty.has_face and empty.skin.max() == 0
    m = build_masks([oval_face()], 200, 200)
    assert m.has_face
    assert m.skin[100, 100] > 0.95 and m.skin[2, 2] < 0.01
    assert 0.05 < m.skin[100, 18] < 0.95  # soft edge near the oval boundary
    assert m.mouth[148, 100] > 0.5


def test_saturation_minus_100_is_monochrome():
    out = render(skin_patch(), Adjustments(saturation=-100), []).astype(int)
    assert np.abs(out[..., 0] - out[..., 2]).max() <= 1


def test_adjustments_reject_out_of_range():
    with pytest.raises(Exception):
        Adjustments(exposure=500)
    assert Adjustments.model_validate({"unknown": 1}).exposure == 0


# --- HTTP API -----------------------------------------------------------------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_without_face():
    r = client.post("/v1/analyze", files={"file": ("a.jpg", jpeg_bytes(skin_patch(240, 320)), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["width"], body["height"]) == (320, 240)
    assert body["cull"]["noFace"] is True
    assert len(body["colorStats"]["mean"]) == 3


def test_render_returns_jpeg_with_same_size():
    adj = json.dumps({"exposure": 20, "skinSmoothing": 50})
    r = client.post("/v1/render", files={"file": ("a.jpg", jpeg_bytes(skin_patch(200, 300)), "image/jpeg")},
                    data={"adjustments": adj, "quality": "90"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (300, 200)


def test_render_respects_max_long_edge_and_format():
    r = client.post("/v1/render", files={"file": ("a.jpg", jpeg_bytes(skin_patch(200, 400)), "image/jpeg")},
                    data={"format": "png", "maxLongEdge": "100"})
    assert r.status_code == 200
    assert Image.open(io.BytesIO(r.content)).size == (100, 50)


def test_render_rejects_bad_input():
    bad = client.post("/v1/render", files={"file": ("a.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 415
    bad_adj = client.post("/v1/render", files={"file": ("a.jpg", jpeg_bytes(skin_patch(64, 64)), "image/jpeg")},
                          data={"adjustments": '{"exposure": 999}'})
    assert bad_adj.status_code == 422


def test_color_stats_endpoint():
    r = client.post("/v1/color-stats", files={"file": ("a.jpg", jpeg_bytes(skin_patch(64, 64)), "image/jpeg")})
    assert r.status_code == 200
    ColorStats.model_validate(r.json())


FIXTURE = os.environ.get("FINERETOUCH_FACE_FIXTURE")


@pytest.mark.skipif(not FIXTURE, reason="set FINERETOUCH_FACE_FIXTURE to a portrait photo to run")
def test_face_landmarks_on_real_portrait():
    with open(FIXTURE, "rb") as fh:
        r = client.post("/v1/analyze", files={"file": ("p.jpg", fh.read(), "image/jpeg")})
    face = r.json()["faces"][0]
    assert face["landmarkCount"] >= 468
    assert len(face["regions"]["leftEye"]) == 16
    assert 0 <= face["eyeOpenness"] <= 1
