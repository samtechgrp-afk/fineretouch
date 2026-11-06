import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import render
from app.retouch_engine import detect_spots, feature_protection_mask, heal_spots
from app.schemas import Adjustments, FaceAnalysis, FaceRegions

client = TestClient(app)
rng = np.random.default_rng(23)
H = W = 400


def face(lips=None, nose=None) -> FaceAnalysis:
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    oval = [(0.5 + 0.42 * np.cos(a), 0.5 + 0.46 * np.sin(a)) for a in t]
    return FaceAnalysis(
        box=(0.08, 0.04, 0.84, 0.92), landmarkCount=478, eyeOpenness=0.8,
        regions=FaceRegions(faceOval=oval, leftEye=[], rightEye=[], leftUnderEye=[], rightUnderEye=[],
                            innerMouth=[], lips=lips or [], leftBrow=[], rightBrow=[], nose=nose or []),
    )


def textured_skin() -> np.ndarray:
    """Skin tone with pores (fine dark specks) and soft grain, float32 sRGB."""
    base = np.array([0.84, 0.63, 0.52], np.float32)
    img = np.ones((H, W, 3), np.float32) * base
    img += rng.normal(0, 0.008, (H, W, 1)).astype(np.float32)
    pores = (rng.random((H, W)) < 0.01).astype(np.float32)
    pores = cv2.GaussianBlur(pores, (0, 0), 0.7)[..., None]
    img -= pores * 0.12
    return np.clip(img, 0, 1)


def disc(img, cy, cx, r, color, soft=1.5):
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    a = np.clip((r - d) / soft + 0.5, 0, 1)[..., None]
    img[:] = img * (1 - a) + np.array(color, np.float32) * a


def skin_mask() -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (W // 2, H // 2), (int(0.42 * W), int(0.46 * H)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


PIMPLES = [(120, 140), (150, 260), (250, 120), (280, 280)]
MOLE = (200, 300)


def scene():
    img = textured_skin()
    for cy, cx in PIMPLES:
        disc(img, cy, cx, 5, [0.80, 0.42, 0.40])  # inflamed red
    disc(img, 210, 190, 4, [0.70, 0.52, 0.43])  # post-acne dark spot
    disc(img, *MOLE, 5, [0.36, 0.24, 0.19], soft=0.8)  # beauty mark
    return img


def redness(img, cy, cx, r=4):
    p = img[cy - r:cy + r + 1, cx - r:cx + r + 1]
    return float((p[..., 0] - p[..., 1]).mean())


def test_detects_pimples_and_dark_spots_and_keeps_beauty_mark():
    img = scene()
    det = detect_spots(img, skin_mask(), [face()], multiplier=1)
    for cy, cx in PIMPLES:
        assert det.heal[cy, cx], (cy, cx)
    assert det.heal[210, 190]
    assert det.protected[MOLE] and not det.heal[MOLE]
    assert det.protected_count == 1
    # Pores are below the finest scale: nowhere near every pore is flagged.
    assert det.heal.mean() < 0.02


def test_healing_removes_spots_locally_and_keeps_texture():
    img = scene()
    out = heal_spots(img, skin_mask(), [face()], amount=100, multiplier=1)
    clean = textured_skin()
    base_red = float((clean[..., 0] - clean[..., 1]).mean())
    for cy, cx in PIMPLES:
        assert redness(out, cy, cx) < base_red + 0.05 < redness(img, cy, cx)
    assert abs(out[MOLE] - img[MOLE]).max() < 1e-6  # beauty mark untouched
    # Far from any spot, pixels are bit-identical: healing is local, not a global blur.
    far = np.ones((H, W), bool)
    for cy, cx in PIMPLES + [MOLE, (210, 190)]:
        far[max(0, cy - 30):cy + 30, max(0, cx - 30):cx + 30] = False
    assert np.abs(out[far] - img[far]).max() < 1e-6
    # Healed spots keep grain (cloned texture), they are not flat holes.
    cy, cx = PIMPLES[0]
    assert out[cy - 3:cy + 4, cx - 3:cx + 4, 1].std() > 0.002


def test_beauty_mark_heals_when_protection_is_off():
    img = scene()
    out = heal_spots(img, skin_mask(), [face()], amount=100, multiplier=1, protect_beauty_marks=False)
    assert out[MOLE][1] > img[MOLE][1] + 0.15


def test_multiplier_reaches_fainter_and_bigger_spots():
    img = textured_skin()
    disc(img, 140, 200, 4, [0.81, 0.58, 0.49])  # faint
    disc(img, 260, 200, 9, [0.78, 0.50, 0.44], soft=3)  # broad
    counts = [detect_spots(img, skin_mask(), [face()], multiplier=m).count for m in (1, 2, 3)]
    assert counts[0] <= counts[1] <= counts[2]
    d3 = detect_spots(img, skin_mask(), [face()], multiplier=3)
    assert d3.heal[140, 200] and d3.heal[260, 200]
    d1 = detect_spots(img, skin_mask(), [face()], multiplier=1)
    assert not d1.heal[260, 200]


def test_lips_and_nostrils_are_protected():
    lips = [(0.42, 0.70), (0.58, 0.70), (0.58, 0.76), (0.42, 0.76)]
    nose = [(0.45, 0.55), (0.55, 0.55), (0.55, 0.6), (0.45, 0.6)]
    f = face(lips=lips, nose=nose)
    img = textured_skin()
    disc(img, int(0.73 * H), int(0.5 * W), 5, [0.8, 0.42, 0.4])  # on the lips
    disc(img, int(0.575 * H), int(0.47 * W), 5, [0.45, 0.3, 0.28])  # nostril shadow
    prot = feature_protection_mask([f], H, W)
    assert prot[int(0.73 * H), int(0.5 * W)] and prot[int(0.575 * H), int(0.47 * W)]
    det = detect_spots(img, skin_mask(), [f], multiplier=3)
    assert not det.heal[int(0.73 * H), int(0.5 * W)]
    assert not det.heal[int(0.575 * H), int(0.47 * W)]


def test_zero_amount_and_clear_skin_are_noops():
    img = textured_skin()
    assert heal_spots(img, skin_mask(), [face()], amount=0) is img
    out = heal_spots(img, skin_mask(), [face()], amount=100, multiplier=1)
    assert np.abs(out - img).max() < 0.05


def test_render_runs_spot_healing_with_multiplier():
    img8 = (scene() * 255 + 0.5).astype(np.uint8)
    out = render(img8, Adjustments(blemishRemoval=100, skinMultiplier=2), [face()])
    cy, cx = PIMPLES[1]
    assert redness(out.astype(np.float32) / 255, cy, cx) < redness(img8.astype(np.float32) / 255, cy, cx) - 0.05


def test_older_clients_without_nose_region_still_validate():
    body = face().model_dump()
    del body["regions"]["nose"]
    assert FaceAnalysis.model_validate(body).regions.nose == []
