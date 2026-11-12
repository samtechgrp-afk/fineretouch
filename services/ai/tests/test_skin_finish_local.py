"""Professional Skin Finish phase 2 (engines/finish_local.py): each local correction changes the
rendered pixels it is about, scales with its amount, and leaves contours, deep cores (moles,
nostrils), pores and complexion alone, on light and deep skin."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.engines import finish as FIN
from app.engines import finish_local as LOC
from app.schemas import SkinFinishSettings
from qa.finish_eval import contour_scene
from tests.perfection_fixtures import make_scene, oval_mask

SIZE = 700
LOCAL = ("shine", "darkSpots", "toneEvenness", "underEye", "dodgeBurn")


def st(**amounts) -> SkinFinishSettings:
    """Only the named controls on (phase 1 smoothing off, so each test sees one correction)."""
    d = {"enabled": True, "smoothing": {"amount": amounts.pop("smoothing", 0)},
         "texturePreservation": {"amount": 100}, "porePreservation": {"amount": 100}}
    for k in LOCAL:
        d[k] = {"amount": amounts.get(k, 0)}
    return SkinFinishSettings(**d)


def to_lab(img):
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)


def base(tone="III", seed=1, lighting=0.15):
    rgb, _, F = make_scene(size=SIZE, tone=tone, seed=seed, with_natural=False, marks=False, lighting=lighting)
    return to_lab(rgb), oval_mask(SIZE, SIZE), F


def disk(cx, cy, r, blur):
    m = np.zeros((SIZE, SIZE), np.float32)
    cv2.circle(m, (cx, cy), r, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), blur)


def fine_rms(L, sel, F):
    hi = L - cv2.GaussianBlur(L, (0, 0), max(0.6, 0.0045 * F))
    return float(np.sqrt(np.mean(hi[sel] ** 2)))


# --- off is phase 1 -----------------------------------------------------------------------------

def test_all_local_off_is_exactly_phase_1():
    lab, mask, F = base()
    img = from_lab(lab)
    lab = to_lab(img)  # what the engine sees (deep tones clip)
    s1 = SkinFinishSettings(enabled=True)  # only phase 1 controls (local default to 0)
    a, rep = FIN.finish_skin(img, mask, s1, F)
    s2 = st(smoothing=55)
    s2.texturePreservation.amount, s2.porePreservation.amount = 75, 65
    for k in LOCAL:  # switched off with a non-zero amount: still off
        getattr(s2, k).amount, getattr(s2, k).enabled = 80, False
    b, _ = FIN.finish_skin(img, mask, s2, F)
    assert np.array_equal(a, b)
    assert rep.local == {}


@pytest.mark.parametrize("name", LOCAL)
def test_each_control_changes_the_image_and_scales(name):
    lab, mask, F = base()
    lab = lab.copy()
    # Something for every control to act on: a mark, a colour blotch, a sheen, a dark under-eye.
    lab[..., 0] -= 6 * disk(250, 300, 14, 4)
    lab[..., 1] += 4 * disk(450, 420, 40, 15)
    sheen = disk(400, 250, 30, 12)
    lab[..., 0] += 14 * sheen
    lab[..., 1] *= 1 - 0.4 * sheen
    lab[..., 2] *= 1 - 0.4 * sheen
    under = disk(300, 470, 45, 10)
    lab[..., 0] -= 8 * under
    img = from_lab(lab)
    lab = to_lab(img)  # what the engine sees (deep tones clip)
    under_w = cv2.GaussianBlur((disk(300, 470, 55, 1) > 0.5).astype(np.float32), (0, 0), 8)
    outs = [FIN.finish_skin(img, mask, st(**{name: a}), F, under_eye=under_w)[0] for a in (0, 40, 100)]
    assert np.array_equal(outs[0], FIN.finish_skin(img, mask, st(), F, under_eye=under_w)[0])
    d40, d100 = (float(np.abs(o - img).sum()) for o in outs[1:])
    assert d40 > 0 and d100 > d40 * 1.3, (name, d40, d100)


# --- dark spots -------------------------------------------------------------------------------

@pytest.mark.parametrize("tone", ["I", "III", "V", "VI"])
def test_dark_spots_lifted_creases_moles_and_pores_kept(tone):
    lab, mask, F = base(tone)
    lab = lab.copy()
    spots = [(220, 260, 7), (300, 420, 12), (470, 300, 18)]
    sm = np.zeros((SIZE, SIZE), np.float32)
    for x, y, r in spots:
        d = disk(x, y, r, 0.4 * r)
        sm = np.maximum(sm, d)
        lab[..., 0] -= 6 * d
        lab[..., 2] += 2.5 * d  # brown: warmer than the skin
    crease = np.zeros((SIZE, SIZE), np.float32)
    cv2.line(crease, (380, 520), (560, 470), 1.0, 5)
    crease = cv2.GaussianBlur(crease, (0, 0), 2.5)
    lab[..., 0] -= 7 * crease
    mole = disk(520, 230, 6, 1.5)
    lab[..., 0] -= 30 * mole
    img = from_lab(lab)
    lab = to_lab(img)  # what the engine sees (deep tones clip)
    out, rep = FIN.finish_skin(img, mask, st(darkSpots=100), F)
    L0, L1 = lab[..., 0], to_lab(out)[..., 0]
    ref = to_lab(from_lab(base(tone)[0]))[..., 0]  # the same skin without the marks
    for x, y, r in spots:
        core = disk(x, y, r, 0.4 * r) > 0.8
        depth = (ref - L0)[core].mean()
        left = (ref - L1)[core].mean()
        assert left < 0.4 * depth, (tone, r, depth, left)
    ccore = crease > 0.6
    assert (L1 - L0)[ccore].mean() < 0.3 * 7  # a crease is not a spot
    assert np.abs(L1 - L0)[mole > 0.8].mean() < 1.0  # a mole is kept
    clean = (mask > 0.5) & (sm < 0.01) & (crease < 0.01) & (mole < 0.01)
    assert fine_rms(L1, clean, F) > 0.93 * fine_rms(L0, clean, F)  # pores kept
    assert rep.local["darkSpots"]["meanDepthAfter"] < rep.local["darkSpots"]["meanDepthBefore"]


# --- shine ------------------------------------------------------------------------------------

def test_shine_reduces_glare_more_than_plain_light():
    lab, mask, F = base("IV")
    lab = lab.copy()
    glare = disk(250, 300, 35, 14)
    lab[..., 0] += 16 * glare
    lab[..., 1] *= 1 - 0.5 * glare
    lab[..., 2] *= 1 - 0.5 * glare
    lit = disk(470, 400, 35, 14)
    lab[..., 0] += 16 * lit  # just as bright, colour kept: better-lit skin, not glare
    img = from_lab(lab)
    lab = to_lab(img)  # what the engine sees (deep tones clip)
    out, rep = FIN.finish_skin(img, mask, st(shine=100), F)
    L0, L1 = lab[..., 0], to_lab(out)[..., 0]
    g, l_ = glare > 0.8, lit > 0.8
    drop_g, drop_l = (L0 - L1)[g].mean(), (L0 - L1)[l_].mean()
    assert drop_g > 5.0 and drop_g > 1.8 * drop_l, (drop_g, drop_l)
    c0 = np.hypot(lab[..., 1], lab[..., 2])[g].mean()
    c1 = np.hypot(*(to_lab(out)[..., c] for c in (1, 2)))[g].mean()
    assert c1 > c0 + 0.5  # the washed-out colour comes back
    assert (L1 - L0)[g].max() <= 0.01  # it only lowers glare
    assert rep.local["shine"]["p99After"] < rep.local["shine"]["p99Before"]


# --- tone evenness ----------------------------------------------------------------------------

@pytest.mark.parametrize("tone", ["II", "V"])
def test_tone_evenness_reduces_colour_blotches_not_lighting(tone):
    lab, mask, F = base(tone, lighting=0.4)
    lab = lab.copy()
    rng = np.random.default_rng(4)
    for _ in range(8):
        x, y = rng.integers(150, 550, 2)
        lab[..., 1] += 4 * disk(int(x), int(y), 30, 12)  # red blotches
    img = from_lab(lab)
    lab = to_lab(img)  # what the engine sees (deep tones clip)
    out, rep = FIN.finish_skin(img, mask, st(toneEvenness=100), F)
    m = rep.local["toneEvenness"]
    assert m["colourVariationAfter"] < 0.7 * m["colourVariationBefore"]
    # The key light's gradient (large scale) is unchanged.
    L0, L1 = lab[..., 0], to_lab(out)[..., 0]
    big0 = cv2.GaussianBlur(L0, (0, 0), 60)
    big1 = cv2.GaussianBlur(L1, (0, 0), 60)
    sel = mask > 0.9
    assert np.abs(big1 - big0)[sel].mean() < 0.4


# --- under-eye --------------------------------------------------------------------------------

def test_under_eye_lifts_shadow_keeps_texture():
    lab, mask, F = base("V")
    lab = lab.copy()
    shadow = disk(350, 330, 50, 14)
    lab[..., 0] -= 10 * shadow
    img = from_lab(lab)
    lab = to_lab(img)  # what the engine sees (deep tones clip)
    under = cv2.GaussianBlur((disk(350, 330, 60, 1) > 0.5).astype(np.float32), (0, 0), 10)
    out, rep = FIN.finish_skin(img, mask, st(underEye=100), F, under_eye=under)
    m = rep.local["underEye"]
    assert m["deficitAfter"] < 0.6 * m["deficitBefore"]
    core = shadow > 0.8
    L0, L1 = lab[..., 0], to_lab(out)[..., 0]
    assert fine_rms(L1, core, F) > 0.9 * fine_rms(L0, core, F)
    assert (L1 - L0)[core].mean() < 10  # not lifted past the surrounding skin
    out_none, _ = FIN.finish_skin(img, mask, st(underEye=100), F)  # no landmarks: nothing to do
    assert np.array_equal(out_none, FIN.finish_skin(img, mask, st(), F)[0])


# --- dodge and burn ---------------------------------------------------------------------------

def test_dodge_burn_evens_irregular_light_keeps_contours():
    src, mask, F = contour_scene("III", 12, size=SIZE)
    lab = to_lab(src)
    rng = np.random.default_rng(9)
    blobs = np.zeros((SIZE, SIZE), np.float32)
    for _ in range(10):  # uneven light on plain skin, away from the nose-side shadow and nostril
        x, y = int(rng.integers(130, 340)), int(rng.integers(150, 320))
        d = disk(x, y, 20, 8)
        blobs = np.maximum(blobs, d)
        lab[..., 0] += float(rng.choice([-3.0, 3.0])) * d
    img = from_lab(lab)
    lab = to_lab(img)
    out, rep = FIN.finish_skin(img, mask, st(dodgeBurn=100), F)
    sel = (blobs > 0.05) & (mask > 0.5)
    before, after = LOC.micro_rms(lab[..., 0], sel, F), LOC.micro_rms(to_lab(out)[..., 0], sel, F)
    assert after < 0.7 * before, (before, after)
    assert rep.local["dodgeBurn"]["microAfter"] < rep.local["dodgeBurn"]["microBefore"]
    assert rep.edge_kept >= 0.98  # the nose-side shadow and the nostril keep their contrast


# --- all together, every tone ----------------------------------------------------------------

@pytest.mark.parametrize("tone", ["I", "II", "III", "IV", "V", "VI"])
def test_professional_finish_keeps_complexion_and_contours(tone):
    src, mask, F = contour_scene(tone, 30, size=SIZE)
    s = SkinFinishSettings(enabled=True, smoothing={"amount": 55}, texturePreservation={"amount": 75},
                           porePreservation={"amount": 65}, shine={"amount": 55}, darkSpots={"amount": 70},
                           toneEvenness={"amount": 45}, underEye={"amount": 45}, dodgeBurn={"amount": 30})
    out, rep = FIN.finish_skin(src, mask, s, F)
    dL, da, db = rep.complexion_shift
    assert abs(dL) < 1.0 and abs(da) < 0.5 and abs(db) < 0.5, rep.complexion_shift
    assert rep.edge_kept >= 0.93, rep.edge_kept
    assert rep.texture_kept is not None and rep.texture_kept > 0.8
    out2, _ = FIN.finish_skin(src, mask, s, F)
    assert np.array_equal(out, out2)  # deterministic


# --- landmark folds ----------------------------------------------------------------------------

def test_irregular_patch_lifted_away_from_folds_but_fold_shadow_kept():
    """With the nose / nasolabial folds known, a large elongated pigment patch on the cheek is
    lifted, while the same-shaped shadow inside the fold mask is left as it is."""
    lab, mask, F = base("V")
    lab = lab.copy()
    band = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(band, (230, 300), (70, 18), 30, 0, 360, 1.0, -1)
    band = cv2.GaussianBlur(band, (0, 0), 6)
    fold = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(fold, (470, 420), (70, 18), -60, 0, 360, 1.0, -1)
    fold = cv2.GaussianBlur(fold, (0, 0), 6)
    lab[..., 0] -= 8 * band + 8 * fold
    img = from_lab(lab)
    lab = to_lab(img)
    folds = cv2.GaussianBlur(cv2.dilate((fold > 0.05).astype(np.float32), np.ones((41, 41), np.uint8)), (0, 0), 8)
    out, _ = FIN.finish_skin(img, mask, st(darkSpots=100), F, folds=folds)
    L0, L1 = lab[..., 0], to_lab(out)[..., 0]
    assert (L1 - L0)[band > 0.8].mean() > 0.5 * 8
    assert (L1 - L0)[fold > 0.8].mean() < 0.15 * 8
    # Without landmarks the elongated patch is not taken for a spot (it could be a fold).
    out2, _ = FIN.finish_skin(img, mask, st(darkSpots=100), F)
    assert (to_lab(out2)[..., 0] - L0)[band > 0.8].mean() < 0.5 * 8


def test_fold_mask_from_landmarks():
    from app.pipeline import build_masks, fold_mask
    from app.schemas import FaceAnalysis, FaceRegions

    def ring(cx, cy, rx, ry, n=12):
        return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]

    regions = dict(faceOval=ring(0.5, 0.5, 0.3, 0.4), leftEye=ring(0.38, 0.4, 0.05, 0.02),
                   rightEye=ring(0.62, 0.4, 0.05, 0.02), leftUnderEye=ring(0.38, 0.45, 0.05, 0.02),
                   rightUnderEye=ring(0.62, 0.45, 0.05, 0.02), innerMouth=ring(0.5, 0.72, 0.06, 0.01),
                   lips=ring(0.5, 0.72, 0.09, 0.03), leftBrow=ring(0.38, 0.33, 0.06, 0.01),
                   rightBrow=ring(0.62, 0.33, 0.06, 0.01), nose=ring(0.5, 0.6, 0.06, 0.03))
    face = FaceAnalysis(box=(0.2, 0.1, 0.6, 0.8), landmarkCount=478, regions=FaceRegions(**regions), eyeOpenness=1.0)
    m = build_masks([face], 400, 400)
    assert m.folds is not None and m.folds.shape == (400, 400)
    assert m.folds[240, 200] > 0.9  # the nose
    assert m.folds[265, 175] > 0.5  # between the ala and the mouth corner
    assert m.folds[150, 100] < 0.01 and m.folds[330, 200] < 0.5  # cheek, chin
    del regions["nose"]
    assert fold_mask([FaceAnalysis(box=(0.2, 0.1, 0.6, 0.8), landmarkCount=478,
                                   regions=FaceRegions(**regions), eyeOpenness=1.0)], 400, 400, 240.0) is None


# --- styles, the scan endpoint (preview) and the heal order -------------------------------------

def test_styles_mirror_the_typescript_panel_and_grow_in_strength():
    import re
    from pathlib import Path

    from app.schemas import FINISH_STYLES

    ts = (Path(__file__).resolve().parents[3] / "packages/shared/src/finish.ts").read_text()
    for style, amounts in FINISH_STYLES.items():
        line = re.search(rf"^\s*{style}: \{{(.*)\}},$", ts, re.M)
        assert line, style
        parsed = {k: float(v) for k, v in re.findall(r"(\w+): (\d+)", line.group(1))}
        assert parsed == amounts, style
    for k in ("smoothing", "shine", "darkSpots", "toneEvenness", "underEye", "dodgeBurn"):
        assert FINISH_STYLES["natural"][k] < FINISH_STYLES["professional"][k] < FINISH_STYLES["advanced"][k]


def test_scan_endpoint_returns_the_finished_preview_and_its_report():
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from app.main import app
    from app.schemas import Adjustments

    client = TestClient(app)
    rgb, _, _ = make_scene(size=480, tone="V", seed=5, with_natural=False, marks=True)
    buf = io.BytesIO()
    Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(buf, "PNG")
    adj = Adjustments(skinFinish=SkinFinishSettings.styled("professional"))
    res = client.post("/v1/perfection", files={"file": ("x.png", buf.getvalue(), "image/png")},
                      data={"adjustments": adj.model_dump_json(), "previewLongEdge": "320"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"] is None  # Skin Perfection is off in this edit
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert max(Image.open(io.BytesIO(base64.b64decode(body["image"].split(",", 1)[1]))).size) == 320
    # No face in the synthetic patch: no face skin, so the finish reports an empty run or none.
    assert "finish" in body


def test_heal_strokes_follow_the_finish_like_the_preview(monkeypatch):
    from app import pipeline
    from app.pipeline import render
    from app.schemas import Adjustments
    from tests.perfection_fixtures import oval_face

    order: list[str] = []
    real_finish, real_heal = pipeline.finish_photo, pipeline.apply_heal_strokes
    monkeypatch.setattr(pipeline, "finish_photo", lambda *a, **k: (order.append("finish"), real_finish(*a, **k))[1])
    monkeypatch.setattr(pipeline, "apply_heal_strokes", lambda *a, **k: (order.append("heal"), real_heal(*a, **k))[1])
    rgb, _, _ = make_scene(size=400, tone="IV", seed=2, with_natural=False, marks=False, face_share=0.4)
    rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
    stroke = {"mode": "heal", "size": 0.02, "offset": [0.05, 0.0], "points": [0.5, 0.5, 0.52, 0.5]}
    adj = Adjustments.model_validate({"localRetouch": {"heal": [stroke]}, "skinFinish": SkinFinishSettings.styled("natural").model_dump()})
    render(rgb8, adj, [oval_face(400, 400)])
    assert order == ["finish", "heal"]
    order.clear()
    render(rgb8, Adjustments.model_validate({"localRetouch": {"heal": [stroke]}}), [oval_face(400, 400)])
    assert order == ["heal"]  # finish off: unchanged, the strokes run first
