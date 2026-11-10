"""Skin Perfection engine (app/engines/perfection.py): detection quality on labelled skin,
natural-feature safety, tiling, body coverage, user overrides, verified passes and that every
control changes the output."""

import base64
import io
import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.engines import perfection as P
from app.engines.perfection import perfect_skin
from app.main import app
from app.pipeline import render
from app.schemas import Adjustments, PerfectionCategory, PerfectionOverride, SkinPerfectionSettings
from scripts.perfection_benchmark import evaluate, run_new

from .perfection_fixtures import NATURAL, Truth, _add, contrast_at, make_scene, oval_face, oval_mask

client = TestClient(app)


def on(**kw) -> SkinPerfectionSettings:
    return SkinPerfectionSettings(enabled=True, **kw)


@pytest.fixture(scope="module")
def scene3():
    img, truths, F = make_scene(size=1200, tone="III", seed=1)
    out, marks, rep = run_new(img, F)
    return img, truths, F, out, marks, rep


def test_finds_and_removes_imperfections_without_touching_natural_features(scene3):
    img, truths, F, out, marks, rep = scene3
    r = evaluate(img, out, truths, F, marks)
    assert r["recall"] >= 0.9 and r["precision"] >= 0.9, r
    assert r["removal_rate"] >= 0.88, r["removed"]
    # Every small blackhead found (the old 420 px detector found none of them).
    assert r["found"]["blackhead"][0] >= r["found"]["blackhead"][1] - 1
    assert r["natural_damaged"] == 0, r["damaged"]
    assert r["texture_retention"] >= 0.97


def test_deep_skin_tone_and_dense_acne():
    img, truths, F = make_scene(size=1600, tone="VI", seed=6, density="dense")
    out, marks, rep = run_new(img, F)
    r = evaluate(img, out, truths, F, marks)
    assert r["removal_rate"] >= 0.78 and r["precision"] >= 0.9, (r["removed"], r["precision"])
    # 4 = a group of 4 stubble follicles cut off from their field by an acne mark the scene put
    # inside it: alone they look exactly like a cluster of blackheads (and are repaired as such).
    # Everything else natural (moles, the rest of the stubble, all but one freckle) is kept.
    assert r["natural_damaged"] <= 4
    assert r["damaged"]["mole"][0] == 0
    # Dense acne stays separate marks, each one counted.
    assert r["found"]["acne"][0] >= 0.8 * r["found"]["acne"][1]


def test_report_counts_are_verified(scene3):
    img, truths, F, out, marks, rep = scene3
    c = rep.counts()
    assert c["detected"] == c["corrected"] + c["remaining"] + c["ignored"]
    assert c["protected"] > 0
    for m in rep.marks:
        assert m.attempts <= P.MAX_ATTEMPTS
        if m.status == "corrected" and m.kind != "manual":
            # Never claimed without measuring: the contrast after correction is recorded and low.
            assert m.after is not None and m.after <= max(P.VERIFY_RATIO * m.contrast, 0.5 * P.MIN_REL_CONTRAST) + 1e-6
    assert rep.passes <= 2
    j = rep.to_json()
    assert j["counts"] == c and all(0 <= m["x"] <= 1 and 0 <= m["y"] <= 1 for m in j["marks"])


def test_mark_on_a_tile_seam_is_found_once():
    size = P.TILE + 400
    img, _, _ = make_scene(size=size, tone="II", seed=9, with_natural=False)
    # Face width so the analysis runs at full size (no shrink): tiles split at x = TILE.
    face_share = P.WORK_FACE / size
    img, truths, F = make_scene(size=size, tone="II", seed=9, with_natural=False, face_share=face_share)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = [np.ascontiguousarray(lab[..., k]) for k in range(3)]
    t = Truth(float(P.TILE), size * 0.5, 0.008 * F, "darkSpot")
    _add(L, A, B, t.x, t.y, t.r, -0.18 * 70, 1.0, 5.0)
    img = np.clip(cv2.cvtColor(np.stack([L, A, B], -1), cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    face = oval_face(size, size).model_copy(update={"box": (0.5 - face_share / 2, 0.03, face_share, 0.94)})
    out, rep = perfect_skin(img, oval_mask(size, size), [face], on())
    assert rep.tiles >= 2 and rep.work_scale == 1.0
    hits = [m for m in rep.marks if (m.x - t.x) ** 2 + (m.y - t.y) ** 2 <= (1.5 * t.r) ** 2]
    assert len(hits) == 1 and hits[0].status == "corrected"
    assert contrast_at(out, t) < 0.4 * contrast_at(img, t)


def test_body_only_photo_is_corrected_inside_body_skin_only():
    img, truths, F = make_scene(size=900, tone="IV", seed=4, with_natural=False)
    body = oval_mask(900, 900)
    out, rep = perfect_skin(img, np.zeros_like(body), [], on(), body_skin=body)
    assert rep.counts()["corrected"] >= 10
    assert all(m.region == "body" for m in rep.marks)
    outside = body < 1e-3
    assert np.array_equal(out[outside], img[outside])  # background / clothing untouched
    # Body sensitivity is a real control: faint marks are found at high sensitivity only.
    faint, _, _ = make_scene(size=900, tone="IV", seed=4, with_natural=False, marks=False)
    _, hi = perfect_skin(faint, np.zeros_like(body), [], on(bodySensitivity=100), body_skin=body)
    _, lo = perfect_skin(faint, np.zeros_like(body), [], on(bodySensitivity=10), body_skin=body)
    assert lo.counts()["detected"] < hi.counts()["detected"]
    # And body coverage off means nothing is done at all.
    same, none = perfect_skin(img, np.zeros_like(body), [], on(bodyCoverage=False), body_skin=body)
    assert np.array_equal(same, img) and none.counts()["detected"] == 0


def test_moles_and_freckles_are_kept_unless_the_user_says_otherwise(scene3):
    img, truths, F, out, marks, rep = scene3
    moles = [t for t in truths if t.kind == "mole"]
    for t in moles:
        assert contrast_at(out, t) > 0.8 * contrast_at(img, t)
    out2, _ = perfect_skin(img, oval_mask(1200, 1200), [oval_face(1200, 1200)], on(preserveMoles=False))
    assert sum(contrast_at(out2, t) < 0.5 * contrast_at(img, t) for t in moles) >= 1
    freckles = [t for t in truths if t.kind == "freckle"]
    out3, _ = perfect_skin(img, oval_mask(1200, 1200), [oval_face(1200, 1200)], on(preserveFreckles=False))
    kept = np.mean([contrast_at(out, t) / contrast_at(img, t) for t in freckles])
    removed = np.mean([contrast_at(out3, t) / contrast_at(img, t) for t in freckles])
    assert kept > 0.85 and removed < kept - 0.15


def test_ignore_and_correct_overrides(scene3):
    img, truths, F, out, marks, rep = scene3
    h, w = img.shape[:2]
    target = next(m for m in rep.marks if m.status == "corrected" and m.kind == "acne")
    ign = PerfectionOverride(x=target.x / w, y=target.y / h, radius=target.r / w, action="ignore")
    out2, rep2 = perfect_skin(img, oval_mask(w, h), [oval_face(w, h)], on(overrides=[ign]))
    t = Truth(target.x, target.y, target.r, "acne")
    assert contrast_at(out2, t) > 0.9 * contrast_at(img, t)
    assert rep2.counts()["ignored"] >= 1
    # Force a correction where nothing was detected: a freckle the user wants gone.
    fr = next(t for t in truths if t.kind == "freckle")
    cor = PerfectionOverride(x=fr.x / w, y=fr.y / h, radius=1.4 * fr.r / w, action="correct")
    out3, rep3 = perfect_skin(img, oval_mask(w, h), [oval_face(w, h)], on(overrides=[cor]))
    assert contrast_at(out3, fr) < 0.5 * contrast_at(img, fr)


@pytest.fixture(scope="module")
def control_scene(scene3):
    """scene3 plus a broad brownish patch (discoloration) and a strong shading edge with marks
    right beside it (jaw / nose side), so area tools and edge protection have work to do."""
    img = scene3[0].copy()
    F = scene3[2]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = [np.ascontiguousarray(lab[..., k]) for k in range(3)]
    _add(L, A, B, 420, 760, 0.07 * F, -4.0, 1.0, 4.0, soft=1.0)
    L[:, 900:] -= 14.0
    for y in (380, 520, 660):
        _add(L, A, B, 896, y, 0.006 * F, -9.0, 8.0, 1.0)
    img = np.clip(cv2.cvtColor(np.stack([L, A, B], -1), cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    base, _ = perfect_skin(img, oval_mask(1200, 1200), [oval_face(1200, 1200)], on())
    return img, base


@pytest.mark.parametrize("change", [
    dict(strength=40), dict(multiplier=3), dict(texturePreservation=0), dict(porePreservation=0),
    dict(correctionRadius=100), dict(edgeProtection=0), dict(confidenceThreshold=95),
    dict(residualIntensity=0, passes=1), dict(passes=1),
    dict(categories={"acne": PerfectionCategory(sensitivity=10)}),
    dict(categories={"blackhead": PerfectionCategory(strength=0)}),
    dict(categories={"darkSpot": PerfectionCategory(enabled=False)}),
    dict(categories={"redness": PerfectionCategory(strength=100)}),
    dict(categories={"discoloration": PerfectionCategory(strength=100)}),
    dict(categories={"texture": PerfectionCategory(strength=100)}),
    dict(categories={"scar": PerfectionCategory(strength=100, sensitivity=100)}),
])
def test_every_control_changes_the_result(control_scene, change):
    img, out = control_scene
    cats = {**SkinPerfectionSettings().categories, **change.pop("categories", {})}
    s = on(categories=cats, **change)
    out2, _ = perfect_skin(img, oval_mask(1200, 1200), [oval_face(1200, 1200)], s)
    assert np.abs(out2 - out).max() > 1e-3


def test_under_eye_category_uses_the_under_eye_region():
    img, _, F = make_scene(size=600, tone="III", seed=2, with_natural=False)
    region = np.zeros((600, 600), np.float32)
    region[200:240, 150:260] = 1.0
    img[200:240, 150:260] *= 0.85  # darker band
    cats = {**SkinPerfectionSettings().categories, "underEye": PerfectionCategory(strength=100)}
    base, _ = perfect_skin(img, oval_mask(600, 600), [oval_face(600, 600)], on())
    lifted, rep = perfect_skin(img, oval_mask(600, 600), [oval_face(600, 600)], on(categories=cats), under_eye=region)
    assert lifted[200:240, 150:260].mean() > base[200:240, 150:260].mean() + 0.01
    assert "underEye" in rep.areas


def test_disabled_is_identity_and_input_is_never_modified(scene3):
    img = scene3[0]
    before = img.copy()
    same, rep = perfect_skin(img, oval_mask(1200, 1200), [oval_face(1200, 1200)], SkinPerfectionSettings())
    assert same is img and rep.counts()["detected"] == 0
    perfect_skin(img, oval_mask(1200, 1200), [oval_face(1200, 1200)], on())
    assert np.array_equal(img, before)


def test_render_runs_the_engine_and_reports():
    img, truths, F = make_scene(size=700, tone="II", seed=5, with_natural=False)
    rgb8 = (img * 255 + 0.5).astype(np.uint8)
    reports: list = []
    adj = Adjustments(skinPerfection=on())
    out = render(rgb8, adj, [oval_face(700, 700)], report=reports)
    assert len(reports) == 1 and reports[0].counts()["corrected"] > 5
    plain = render(rgb8, Adjustments(), [oval_face(700, 700)])
    assert np.abs(out.astype(int) - plain.astype(int)).max() > 20


def test_perfection_endpoint_returns_report_and_preview():
    img, truths, F = make_scene(size=640, tone="IV", seed=8, with_natural=False)
    buf = io.BytesIO()
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(buf, "PNG")
    res = client.post("/v1/perfection", files={"file": ("x.png", buf.getvalue(), "image/png")},
                      data={"adjustments": Adjustments(skinPerfection=on()).model_dump_json(), "previewLongEdge": "320"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["image"].startswith("data:image/jpeg;base64,")
    prev = Image.open(io.BytesIO(base64.b64decode(body["image"].split(",", 1)[1])))
    assert max(prev.size) == 320
    # No face in a synthetic patch and no body parse model hit -> the report still comes back.
    assert body["report"] is not None and "counts" in body["report"]


def test_shared_typescript_defaults_match_the_engine():
    """packages/shared/src/perfection.ts mirrors schemas.py: same categories and defaults."""
    import re
    from pathlib import Path

    from app.schemas import PERFECTION_CATEGORIES, PERFECTION_DEFAULTS

    ts = (Path(__file__).resolve().parents[3] / "packages/shared/src/perfection.ts").read_text()
    block = ts.split("PERFECTION_DEFAULTS: Readonly", 1)[1].split("});", 1)[0]
    pairs = dict((k, (float(a), float(b))) for k, a, b in re.findall(r"(\w+): \[(\d+), (\d+)\]", block))
    assert pairs == {k: (float(a), float(b)) for k, (a, b) in PERFECTION_DEFAULTS.items()}
    assert list(pairs) == list(PERFECTION_CATEGORIES)
    d = SkinPerfectionSettings()
    dflt = ts.split("DEFAULT_SKIN_PERFECTION: Readonly<SkinPerfectionSettings> = Object.freeze({", 1)[1].split("});", 1)[0]
    for key, val in re.findall(r"^\s+(\w+): (true|false|\d+),", dflt, re.M):
        expect = {"true": True, "false": False}.get(val, None)
        assert getattr(d, key) == (expect if expect is not None else float(val)), key
