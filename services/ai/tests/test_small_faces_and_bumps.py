"""Regressions from the full-length portrait case study (docs/case-studies/full-length-portrait.md).

The face there is ~9% of the frame width on deeper, glossy skin: the detector missed it, skin
radii were sized from the long edge, raised comedones read lighter (not darker or redder) than
the skin, and the 2x boost wrote a guided-filtered tone layer back over the whole face.
"""

import cv2
import numpy as np

from app import landmarks
from app.landmarks import FaceLandmarkService, _remap_face, _same_face
from app.pipeline import build_masks, render
from app.retouch_engine import apply_boosted_skin_retouch, detect_spots, faces_in_crop, heal_spots
from app.schemas import Adjustments, FaceAnalysis, FaceRegions

from .test_spot_healing import H, W, disc, face, skin_mask, textured_skin

rng = np.random.default_rng(7)


def oval(cx: float, cy: float, rx: float, ry: float) -> list[tuple[float, float]]:
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


def small_face(cx: float, cy: float, rx: float, ry: float) -> FaceAnalysis:
    return FaceAnalysis(
        box=(cx - rx, cy - ry, 2 * rx, 2 * ry), landmarkCount=478, eyeOpenness=0.8,
        regions=FaceRegions(faceOval=oval(cx, cy, rx, ry), leftEye=[], rightEye=[], leftUnderEye=[],
                            rightUnderEye=[], innerMouth=[], lips=[], leftBrow=[], rightBrow=[]),
    )


def test_remap_and_same_face_round_trip():
    f = small_face(0.5, 0.5, 0.1, 0.12)
    full = _remap_face(f, 100, 200, 400, 300, 1000, 2000)
    assert np.allclose(full.box, ((100 + 0.4 * 400) / 1000, (200 + 0.38 * 300) / 2000, 0.08, 0.036))
    back = faces_in_crop([full], 100, 200, 400, 300, 1000, 2000)[0]
    assert np.allclose(back.box, f.box) and np.allclose(back.regions.faceOval, f.regions.faceOval)
    assert _same_face(full, full) and not _same_face(full, small_face(0.9, 0.9, 0.02, 0.02))


def test_detector_falls_back_to_tiles_for_small_faces(monkeypatch):
    """No face on the downscaled frame -> overlapping full-res windows find it, in frame coordinates."""
    rgb = np.zeros((3000, 2000, 3), np.uint8)
    rgb[600:700, 900:980] = 255  # the "face"
    svc = FaceLandmarkService()
    monkeypatch.setattr(FaceLandmarkService, "backend", property(lambda self: "mediapipe"))

    def fake(self, img, aspect):
        ys, xs = np.nonzero(img[..., 0] > 128)
        # The 1280px proxy of the whole frame is too coarse: only full-resolution crops see it.
        if ys.size == 0 or img.shape[:2] == (1280, 853):
            return []
        h, w = img.shape[:2]
        return [small_face((xs.mean() + 0.5) / w, (ys.mean() + 0.5) / h, (np.ptp(xs) + 1) / 2 / w, (np.ptp(ys) + 1) / 2 / h)]

    monkeypatch.setattr(FaceLandmarkService, "_detect_mediapipe", fake)
    faces = svc.detect(rgb)
    assert len(faces) == 1
    x, y, bw, bh = faces[0].box
    assert abs((x + bw / 2) * 2000 - 940) < 6 and abs((y + bh / 2) * 3000 - 650) < 6
    assert landmarks.TILE_FRACTIONS[0] < 1


def test_forehead_mask_reaches_above_the_landmark_oval():
    m = build_masks([face()], H, W)
    top = int((0.5 - 0.46) * H)  # top of the oval
    assert m.forehead is not None and m.forehead[top - 10, W // 2] > 0.5
    assert m.forehead[H // 2, W // 2] == 0  # the oval itself is the skin mask's job


def test_raised_light_bumps_are_found_and_healed():
    img = textured_skin()
    bumps = [(120, 150), (130, 220), (160, 260), (250, 140)]
    for cy, cx in bumps:
        disc(img, cy, cx, 2.5, [0.93, 0.74, 0.63], soft=1.0)  # small bright dome
    det = detect_spots(img, skin_mask(), [face()], multiplier=2)
    found = sum(bool(det.heal[cy, cx]) for cy, cx in bumps)
    assert found >= 3
    out = heal_spots(img, skin_mask(), [face()], amount=100, multiplier=2, detection=det)
    for cy, cx in bumps:
        if det.heal[cy, cx]:
            assert out[cy, cx, 1] < img[cy, cx, 1] - 0.04
    # Multiplier 1 is conservative: never more bumps than 2x.
    assert detect_spots(img, skin_mask(), [face()], multiplier=1).count <= det.count


def test_soft_shading_hollow_is_not_a_spot():
    img = textured_skin()
    yy, xx = np.mgrid[0:H, 0:W]
    hollow = np.exp(-(((yy - 200) / 14.0) ** 2 + ((xx - 200) / 10.0) ** 2) / 2)[..., None]
    img = np.clip(img - 0.12 * hollow, 0, 1).astype(np.float32)
    det = detect_spots(img, skin_mask(), [face()], multiplier=3, protect_beauty_marks=False)
    assert not det.heal[200, 200]


def test_one_sided_shading_edge_is_not_a_spot():
    img = textured_skin()
    yy, xx = np.mgrid[0:H, 0:W]
    edge = (1.0 / (1.0 + np.exp(-(xx - 200) / 3.0)))[..., None]  # darker right half
    img = np.clip(img - 0.1 * edge, 0, 1).astype(np.float32)
    det = detect_spots(img, skin_mask(), [face()], multiplier=2, protect_beauty_marks=False)
    assert det.heal[:, 185:215].mean() < 0.01


def test_boost_keeps_facial_shading():
    """2x/3x repair spots; they do not rewrite the face's highlights and shadows."""
    img = textured_skin()
    yy, xx = np.mgrid[0:H, 0:W]
    shade = (0.12 * np.sin(xx / 25.0) * np.cos(yy / 31.0))[..., None]
    img = np.clip(img + shade, 0, 1).astype(np.float32)
    out = apply_boosted_skin_retouch(img, skin_mask(), 3.0)
    core = skin_mask() > 0.9
    lo = lambda a: cv2.GaussianBlur(a @ np.array([0.299, 0.587, 0.114], np.float32), (0, 0), 6)
    assert np.abs(lo(out) - lo(img))[core].mean() < 0.006


def test_full_length_frame_only_touches_the_face():
    """Skin stages run on the face box and radii follow the face, not the long edge."""
    h, w = 1800, 900
    img = np.full((h, w, 3), [0.25, 0.45, 0.30], np.float32)
    img += rng.normal(0, 0.01, (h, w, 3)).astype(np.float32)
    skin = np.array([0.55, 0.36, 0.26], np.float32)
    cv2.ellipse(img, (450, 300), (40, 50), 0, 0, 360, skin.tolist(), -1)
    yy, xx = np.mgrid[0:h, 0:w]
    img[((yy - 300) / 50.0) ** 2 + ((xx - 450) / 40.0) ** 2 < 1] += rng.normal(0, 0.02, (1, 3)).astype(np.float32)
    rgb8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    f = small_face(450 / w, 300 / h, 40 / w, 50 / h)
    out = render(rgb8, Adjustments(skinSmoothing=60, blemishRemoval=100, skinMultiplier=2), [f])
    diff = np.abs(out.astype(int) - rgb8.astype(int)).sum(-1)
    ys, xs = np.nonzero(diff)
    assert ys.size > 0
    assert ys.min() > 150 and ys.max() < 450 and xs.min() > 300 and xs.max() < 600


def test_long_merged_repair_keeps_grain():
    """A run of spots that merges into one long repair is re-textured in tiles, not left flat."""
    img = textured_skin()
    heal = np.zeros((H, W), np.uint8)
    heal[196:206, 110:290] = 1  # 180 px long: no single clean donor this size exists nearby
    from app.retouch_engine import SpotDetection
    det = SpotDetection(heal, np.zeros_like(heal), 1, 0, float(W) * 0.84)
    out = heal_spots(img, skin_mask(), [face()], amount=100, multiplier=2, detection=det)
    grain = lambda a: (a - cv2.GaussianBlur(a, (0, 0), 2.0))[198:204, 120:280, 1].std()
    assert grain(out) > 0.5 * grain(img)
