"""B1: "Correct all remaining" respects the quality gate. B3: preview and export run the heal
strokes in the same order (right after the Skin Perfection engine)."""

from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.engines import perfection as P
from app.engines import perfection_quality as Q
from app.engines.local import apply_heal_strokes
from app.engines.perfection import perfect_skin
from app.pipeline import render
from app.schemas import Adjustments, HealStroke, PerfectionOverride, SkinPerfectionSettings

from .perfection_fixtures import _add, make_scene, oval_face, oval_mask


def on(**kw) -> SkinPerfectionSettings:
    return SkinPerfectionSettings(enabled=True, **kw)


def bulk_for(marks, w: int, h: int) -> list[PerfectionOverride]:
    """The overrides "Correct all remaining" sends (packages/shared correctRemaining), one per mark."""
    L = float(max(w, h))
    return [PerfectionOverride(x=m.x / w, y=m.y / h, radius=max(0.0005, m.r / L * 1.3), action="correct", bulk=True) for m in marks]


def disc(shape, x: float, y: float, r: float) -> np.ndarray:
    """Pixels whose centre lies inside the circle (float centre)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


@pytest.fixture(scope="module")
def scene():
    img, truths, F = make_scene(size=900, tone="V", seed=21)
    h, w = img.shape[:2]
    mask, faces = oval_mask(w, h), [oval_face(w, h)]
    out, rep = perfect_skin(img, mask, faces, on())
    return img, mask, faces, out, rep


def test_bulk_correction_that_passes_the_gate_is_kept(scene):
    img, mask, faces, out, rep = scene
    h, w = img.shape[:2]
    chosen = [m for m in rep.marks if m.status == "corrected"][:6]
    assert len(chosen) == 6
    out2, rep2 = perfect_skin(img, mask, faces, on(overrides=bulk_for(chosen, w, h)))
    kept = [m for m in rep2.marks if m.bulk]
    assert kept and all(m.status == "corrected" and not m.quality["failed"] for m in kept)
    for m in kept:  # the repair is in the output: the mark's contrast fell
        assert P._mark_contrast(out2, oval_mask(w, h), m) < 0.5 * m.contrast
    # Forcing marks the engine corrects anyway changes nothing.
    assert np.array_equal(out2, out)


def _fail_near(monkeypatch, points: list[tuple[float, float]], reach: float, check: str = "texture") -> None:
    """Make the quality gate fail `check` for repairs centred near the given points (a stand-in for a
    repair that looks wrong, as on the knuckles in the 25 MP photo)."""
    real = P._verify

    def verify(image, out, skin, todo, *a, **k):
        res = real(image, out, skin, todo, *a, **k)
        for m in todo:
            if m.quality is not None and any(math.hypot(m.x - x, m.y - y) <= reach for x, y in points):
                m.quality["failed"] = sorted(set(m.quality["failed"]) | {check})
                m.quality["values"][check] = 99.0
                if m.status == "corrected":
                    m.status = "remaining"
        return res

    monkeypatch.setattr(P, "_verify", verify)


def test_bulk_correction_that_fails_the_gate_is_undone_and_reported(scene, monkeypatch):
    img, mask, faces, out, rep = scene
    h, w = img.shape[:2]
    chosen = [m for m in rep.marks if m.status == "corrected" and m.region == "face"][:4]
    pts = [(m.x, m.y) for m in chosen]
    _fail_near(monkeypatch, pts, reach=1.0)
    out2, rep2 = perfect_skin(img, mask, faces, on(overrides=bulk_for(chosen, w, h)))
    failed = [m for m in rep2.marks if m.bulk]
    assert len(failed) == len(chosen)
    for m in failed:
        assert m.status == "review" and m.reason == "repairFailed"
        assert "texture" in m.quality["failed"] and m.quality["kept"] is False
    assert rep2.counts()["corrected"] < rep.counts()["corrected"]


def test_failed_bulk_repair_restores_the_original_pixels_exactly(scene, monkeypatch):
    img, mask, faces, out, rep = scene
    h, w = img.shape[:2]
    # Marks with a corrected neighbour close by, whose feathered repair would otherwise leak in.
    corrected = [m for m in rep.marks if m.status == "corrected"]
    chosen = corrected[:8]
    _fail_near(monkeypatch, [(m.x, m.y) for m in chosen], reach=1.0)
    out2, rep2 = perfect_skin(img, mask, faces, on(overrides=bulk_for(chosen, w, h)))
    failed = [m for m in rep2.marks if m.bulk and m.reason == "repairFailed"]
    assert failed
    for m in failed:
        k = disc(img.shape, m.x, m.y, m.r)
        assert np.array_equal(out2[k], img[k]), f"mark {m.id} at ({m.x:.0f}, {m.y:.0f}) not restored"


def test_a_hand_picked_correction_still_keeps_a_failing_repair(scene, monkeypatch):
    """Unchanged behaviour: choosing Correct on one mark keeps its repair and shows the failed checks."""
    img, mask, faces, out, rep = scene
    h, w = img.shape[:2]
    m0 = next(m for m in rep.marks if m.status == "corrected" and m.region == "face")
    _fail_near(monkeypatch, [(m0.x, m0.y)], reach=1.0)
    L = float(max(w, h))
    own = PerfectionOverride(x=m0.x / w, y=m0.y / h, radius=max(0.0005, m0.r / L * 1.3), action="correct")
    out2, rep2 = perfect_skin(img, mask, faces, on(overrides=[own]))
    m = min(rep2.marks, key=lambda a: math.hypot(a.x - m0.x, a.y - m0.y))
    assert m.status == "corrected" and m.quality["kept"] is True and "texture" in m.quality["failed"]
    assert not np.array_equal(out2[disc(img.shape, m.x, m.y, m.r)], img[disc(img.shape, m.x, m.y, m.r)])
    # The user's own choice also wins over an earlier bulk choice on the same mark.
    out3, rep3 = perfect_skin(img, mask, faces, on(overrides=bulk_for([m0], w, h) + [own]))
    m3 = min(rep3.marks, key=lambda a: math.hypot(a.x - m0.x, a.y - m0.y))
    assert m3.status == "corrected" and m3.quality["kept"] is True


def test_bulk_never_forces_a_protected_feature_or_creates_a_manual_spot(scene):
    img, mask, faces, out, rep = scene
    h, w = img.shape[:2]
    prot = [m for m in rep.marks if m.status == "protected"][:5]
    assert prot
    nowhere = PerfectionOverride(x=0.02, y=0.02, radius=0.004, action="correct", bulk=True)
    out2, rep2 = perfect_skin(img, mask, faces, on(overrides=bulk_for(prot, w, h) + [nowhere]))
    assert not any(m.kind == "manual" for m in rep2.marks)
    assert not any(m.bulk for m in rep2.marks)
    assert np.array_equal(out2, out)


def test_every_bulk_repair_in_the_output_passes_every_check():
    """The gate holds with no monkeypatching: marks right beside a strong shading edge (a jaw, a
    knuckle crease, a nail edge), where repairs are most likely to fail, are either corrected
    with no failed check or put back exactly."""
    img, truths, F = make_scene(size=900, tone="VI", seed=33)
    h, w = img.shape[:2]
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = [np.ascontiguousarray(lab[..., k]) for k in range(3)]
    L[:, 640:] -= 16.0  # hard shading edge
    L[300:310, 200:600] += 22.0  # a bright nail-like ridge
    for y in range(260, 680, 45):
        _add(L, A, B, 636, y, 0.006 * F, -9.0, 7.0, 1.0)
        _add(L, A, B, 400, y if y < 300 else y + 20, 0.004 * F, -7.0, 6.0, 1.0)
    img = np.clip(cv2.cvtColor(np.stack([L, A, B], -1), cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    mask, faces = oval_mask(w, h), [oval_face(w, h)]
    out, rep = perfect_skin(img, mask, faces, on())
    todo = [m for m in rep.marks if m.status in ("remaining", "review") and m.reason in (None, "lowConfidence", "repairFailed")]
    assert todo
    out2, rep2 = perfect_skin(img, mask, faces, on(overrides=bulk_for(todo, w, h)))
    bulk = [m for m in rep2.marks if m.bulk]
    assert bulk
    for m in bulk:
        if m.status == "corrected":
            assert not m.quality["failed"], (m.kind, m.quality)
        else:
            assert m.reason == "repairFailed"
            k = disc(img.shape, m.x, m.y, m.r)
            assert np.array_equal(out2[k], img[k])


def test_undoing_the_bulk_action_gives_back_the_default_result_exactly(scene, monkeypatch):
    """Undo / redo in the app swap the settings; the engine keeps no state between renders."""
    img, mask, faces, out, rep = scene
    h, w = img.shape[:2]
    chosen = [m for m in rep.marks if m.status == "corrected"][:5]
    _fail_near(monkeypatch, [(m.x, m.y) for m in chosen], reach=1.0)
    s_bulk = on(overrides=bulk_for(chosen, w, h))
    first, _ = perfect_skin(img, mask, faces, s_bulk)
    monkeypatch.undo()
    undone, _ = perfect_skin(img, mask, faces, on())
    assert np.array_equal(undone, out)
    _fail_near(monkeypatch, [(m.x, m.y) for m in chosen], reach=1.0)
    redone, _ = perfect_skin(img, mask, faces, s_bulk)
    assert np.array_equal(redone, first)


# --- the 25 MP photo from the review (opt-in: the photo is not in the repository) ----------------

FINGER_PHOTO = os.environ.get("FINERETOUCH_BULK_FIXTURE")


@pytest.mark.skipif(not FINGER_PHOTO, reason="set FINERETOUCH_BULK_FIXTURE to the 25 MP full-body photo (CRPixels-5) to run")
def test_bulk_correction_leaves_fingers_and_nails_alone_on_the_25mp_photo():
    from app import imageio, raw_engine

    data = Path(FINGER_PHOTO).read_bytes()
    hq = imageio.decode_hq(data, None)
    rgb = raw_engine.to_uint8(hq)
    faces = main.faces_service.detect(rgb)
    parse = main.depth_service.parse(rgb)
    rep: list = []
    base = np.asarray(render(hq, Adjustments(skinPerfection=on()), faces, parse=parse, report=rep))
    h, w = rgb.shape[:2]
    L = float(max(w, h))
    # What "Correct all remaining" sends at the default threshold (isBulkCorrectable).
    pick = [m for m in rep[0].marks if m.confidence >= 0.5 and (m.status == "remaining" or (m.status == "review" and m.reason in (None, "lowConfidence")))]
    ov = [PerfectionOverride(x=m.x / w, y=m.y / h, radius=max(0.0005, m.r / L * 1.3), action="correct", bulk=True) for m in pick]
    rep2: list = []
    out = np.asarray(render(hq, Adjustments(skinPerfection=on(overrides=ov)), faces, parse=parse, report=rep2))
    o = rgb.astype(np.int16)
    # The hands: the crop where the bulk click used to leave a smudge (x 1900-2110, y 2690-2900).
    sy, sx = h / 6148.0, w / 4099.0
    y0, y1, x0, x1 = int(2690 * sy), int(2900 * sy), int(1900 * sx), int(2110 * sx)
    d_bulk = np.abs(out[y0:y1, x0:x1].astype(np.int16) - o[y0:y1, x0:x1]).max(-1)
    d_base = np.abs(base[y0:y1, x0:x1].astype(np.int16) - o[y0:y1, x0:x1]).max(-1)
    assert (d_bulk > 8).sum() <= (d_base > 8).sum() and d_bulk.max() <= d_base.max()
    for m in rep2[0].marks:
        if m.bulk:
            assert m.status == "corrected" and not m.quality["failed"] or m.reason == "repairFailed"
            if m.reason == "repairFailed":
                k = disc(rgb.shape, m.x, m.y, m.r - 0.5)
                assert np.array_equal(out[k], rgb[k])


# --- B3: heal strokes in the same order for preview and export ----------------------------------


@pytest.fixture(scope="module")
def healed_scene():
    img, truths, F = make_scene(size=700, tone="IV", seed=12, with_natural=False)
    rgb8 = (img * 255 + 0.5).astype(np.uint8)
    face = oval_face(700, 700)
    rep: list = []
    perfected = np.asarray(render(rgb8, Adjustments(skinPerfection=on()), [face], report=rep))
    m = next(m for m in rep[0].marks if m.status == "corrected" and m.r >= 2.5 and 200 < m.x < 500 and 200 < m.y < 500)
    # A heal stroke right across that repair, sampling skin 40 px to the right.
    stroke = HealStroke(mode="heal", size=4.0 * m.r / 700, hardness=50, opacity=100, offset=(40 / 700, 0.0),
                        points=[(m.x - 2 * m.r) / 700, m.y / 700, (m.x + 2 * m.r) / 700, m.y / 700])
    return rgb8, face, perfected, m, stroke


def _overlap(m, shape) -> np.ndarray:
    return disc(shape, m.x, m.y, 4.0 * m.r)


def test_export_heals_after_skin_perfection_as_the_preview_does(healed_scene):
    """Export (one render with both) equals the preview's order: the perfected photo, then the
    stroke. The only difference allowed is rounding: the preview heals 8-bit pixels."""
    rgb8, face, perfected, m, stroke = healed_scene
    export = np.asarray(render(rgb8, Adjustments(skinPerfection=on(), localRetouch={"heal": [stroke.model_dump()]}), [face]))
    preview = (apply_heal_strokes(perfected.astype(np.float32) / 255.0, [stroke]) * 255 + 0.5).astype(np.uint8)
    k = _overlap(m, rgb8.shape)
    assert np.abs(export[k].astype(int) - perfected[k].astype(int)).max() > 3, "the stroke must change the repaired area"
    assert np.abs(export[k].astype(int) - preview[k].astype(int)).max() <= 1
    # And the old order (stroke first, then the engine) is a different picture there.
    healed_first = (apply_heal_strokes(rgb8.astype(np.float32) / 255.0, [stroke]) * 255 + 0.5).astype(np.uint8)
    old = np.asarray(render(healed_first, Adjustments(skinPerfection=on()), [face]))
    assert np.abs(old[k].astype(int) - export[k].astype(int)).max() > 1


def test_preview_and_export_endpoints_agree_where_a_stroke_meets_a_repair(healed_scene, monkeypatch):
    """Through the app's two endpoints: POST /v1/perfection (the preview base; the browser then
    replays the stroke, engines/local.py being the port it mirrors) and POST /v1/render (export).
    Documented difference: the preview image is a JPEG (quality 95), the export here a PNG."""
    rgb8, face, perfected, m, stroke = healed_scene
    monkeypatch.setattr(main.faces_service, "detect", lambda rgb: [face])
    monkeypatch.setattr(main.depth_service, "parse", lambda rgb: None)
    client = TestClient(main.app)
    buf = io.BytesIO()
    Image.fromarray(rgb8).save(buf, "PNG")
    sp = on()
    res = client.post("/v1/perfection", files={"file": ("x.png", buf.getvalue(), "image/png")},
                      data={"adjustments": Adjustments(skinPerfection=sp).model_dump_json(), "previewLongEdge": "700"})
    assert res.status_code == 200, res.text
    base = np.asarray(Image.open(io.BytesIO(base64.b64decode(res.json()["image"].split(",", 1)[1]))).convert("RGB"))
    preview = (apply_heal_strokes(base.astype(np.float32) / 255.0, [stroke]) * 255 + 0.5).astype(np.uint8)
    adj = Adjustments(skinPerfection=sp, localRetouch={"heal": [stroke.model_dump()]})
    exp = client.post("/v1/render", files={"file": ("x.png", buf.getvalue(), "image/png")},
                      data={"adjustments": adj.model_dump_json(), "format": "png"})
    assert exp.status_code == 200, exp.text
    export = np.asarray(Image.open(io.BytesIO(exp.content)).convert("RGB"))
    k = _overlap(m, rgb8.shape)
    d = np.abs(export[k].astype(int) - preview[k].astype(int))
    # The JPEG's own error on the perfected photo, in the same pixels, is the allowance.
    jpeg = np.abs(base[k].astype(int) - perfected[k].astype(int))
    assert d.mean() <= jpeg.mean() + 0.1 and d.max() <= jpeg.max() + 2
    assert np.array_equal(export, np.asarray(render(rgb8, adj, [face])))


# --- B2: the count shows only what the bulk action can attempt -------------------------------


def test_bulk_choice_on_a_mark_the_last_rescan_found_does_nothing(scene):
    """Marks the residual scan finds are created after the choices are applied, so a bulk choice
    there cannot reach them (correctRemaining leaves them out, and the count does too)."""
    img, mask, faces, _, _ = scene
    h, w = img.shape[:2]
    out, rep = perfect_skin(img, mask, faces, on(passes=1))
    late = [m for m in rep.marks if m.residual and m.status == "remaining"]
    assert late and all(m.attempts == 0 for m in late)
    out2, rep2 = perfect_skin(img, mask, faces, on(passes=1, overrides=bulk_for(late, w, h)))
    assert np.array_equal(out2, out)
    assert [(m.id, m.status) for m in rep2.marks] == [(m.id, m.status) for m in rep.marks]
    # Choosing Correct on one of them still repairs it (a hand-placed correction).
    one = late[0]
    L = float(max(w, h))
    own = PerfectionOverride(x=one.x / w, y=one.y / h, radius=max(0.0005, one.r / L * 1.3), action="correct")
    out3, _ = perfect_skin(img, mask, faces, on(passes=1, overrides=[own]))
    assert np.abs(out3 - out)[disc(img.shape, one.x, one.y, one.r)].max() > 0.01
