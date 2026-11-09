"""Skin Perfection phase 2D: the correction quality gate. Each check must fail a repair that is
bad in its way and pass a good one; a mark is not "corrected" just because its contrast fell."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engines import perfection as P
from app.engines import perfection_quality as Q
from app.schemas import PerfectionOverride, SkinPerfectionSettings

from .perfection_fixtures import contrast_at, make_scene, oval_face, oval_mask

N, CX, CY, R, F = 121, 60.0, 60.0, 12.0, 1000.0


def _skin(seed=0, pores=True) -> np.ndarray:
    """Plain skin with visible fine texture and pores (float32 RGB)."""
    rng = np.random.default_rng(seed)
    base = np.array([0.78, 0.6, 0.5], np.float32)
    img = np.ones((N, N, 3), np.float32) * base
    grain = cv2.GaussianBlur(rng.normal(0, 0.012, (N, N)).astype(np.float32), (0, 0), 0.8)
    img += grain[..., None]
    if pores:
        for _ in range(140):
            x, y = rng.uniform(2, N - 2, 2)
            cv2.circle(img, (int(x), int(y)), 1, tuple(float(v) * 0.94 for v in base), -1)
    return np.clip(img, 0, 1)


def _disc(r=R):
    yy, xx = np.mgrid[0:N, 0:N]
    return np.hypot(xx - CX, yy - CY) <= r


def _mark_with(q):
    return P.Mark(0, CX, CY, R, "acne", 0.9, "face", 0.2, status="remaining", quality=q.to_json())


def _assess(before, after, residual=(0.1, 0.1), skin=None):
    skin = np.ones((N, N), np.float32) if skin is None else skin
    holes = np.zeros((N, N), np.uint8)
    win = Q.window(before, after, skin, holes, CX, CY, R)
    return Q.assess(win, R, F, residual)


@pytest.fixture(scope="module")
def pair():
    """(before with a mark, a good repair: other skin of the same kind pasted over the mark)."""
    good = _skin(0)
    other = _skin(7)
    d = _disc(R + 1)
    good_rep = good.copy()
    good_rep[d] = other[d]
    before = good.copy()
    m = _disc(0.6 * R)
    before[m] = before[m] * 0.75
    return before, good_rep


def test_ciede2000_matches_reference_data():
    # Sharma, Wu & Dalal (2005), test pairs 1, 7, 17 and 19.
    assert Q.delta_e2000(np.array([50, 2.6772, -79.7751]), np.array([50, 0, -82.7485])) == pytest.approx(2.0425, abs=1e-4)
    assert Q.delta_e2000(np.array([50, 0, 0]), np.array([50, -1, 2])) == pytest.approx(2.3669, abs=1e-4)
    assert Q.delta_e2000(np.array([50, 2.5, 0]), np.array([73, 25, -18])) == pytest.approx(27.1492, abs=1e-4)
    assert Q.delta_e2000(np.array([50, 2.5, 0]), np.array([56, -27, -3])) == pytest.approx(31.9030, abs=1e-4)


def test_a_good_repair_passes_every_check(pair):
    before, good = pair
    q = _assess(before, good)
    assert q.failed == [], q.values
    assert all(q.values[k] is not None for k in ("colour", "texture", "pores", "edges", "surround"))


def test_a_mark_left_visible_fails_residual(pair):
    before, _ = pair
    q = _assess(before, before, residual=(0.9, 1.4))
    assert "residual" in q.failed


def test_a_wrong_tone_fails_colour_and_is_not_an_unfinished_repair(pair):
    before, good = pair
    bad = good.copy()
    d = _disc(R)
    bad[d] = bad[d] + np.array([0.05, 0.05, 0.05], np.float32)  # a pale disc: a tone the mark never had
    q = _assess(before, bad)
    assert "colour" in q.failed and not q.incomplete
    assert "colour" in P._natural_failures(_mark_with(q))  # wrong skin: undone


def test_colour_left_from_the_mark_is_an_unfinished_repair(pair):
    before, good = pair
    bad = good.copy()
    d = _disc(R)
    bad[d] = 0.6 * bad[d] + 0.4 * before[d] * 0.85  # a share of the mark's own (darker) colour left
    q = _assess(before, bad)
    assert "colour" in q.failed and q.incomplete
    assert "colour" not in P._natural_failures(_mark_with(q))  # unfinished, not damage: not undone for colour


def test_a_flat_repair_fails_texture_and_pores(pair):
    before, good = pair
    flat = good.copy()
    d = _disc(R + 1)
    flat[d] = cv2.GaussianBlur(good, (0, 0), 6)[d]
    q = _assess(before, flat)
    assert "texture" in q.failed and "pores" in q.failed


def test_a_seam_fails_edges(pair):
    before, good = pair
    seam = good.copy()
    yy, xx = np.mgrid[0:N, 0:N]
    ring = np.abs(np.hypot(xx - CX, yy - CY) - R) <= 1.0
    seam[ring] = seam[ring] * 0.8
    q = _assess(before, seam)
    assert "edges" in q.failed


def test_changed_surrounding_skin_fails_surround(pair):
    before, good = pair
    bad = good.copy()
    yy, xx = np.mgrid[0:N, 0:N]
    around = np.hypot(xx - CX, yy - CY) <= 2.4 * R
    bad[around] = bad[around] * 0.96  # about -2.5 L* over the ring
    q = _assess(before, bad)
    assert "surround" in q.failed


def test_a_stamped_texture_fails_repetition():
    img = _skin(3)
    sig_a = Q.texture_signature(img, 30, 30, 8, F)
    copy = img.copy()
    copy[72:89, 72:89] = img[22:39, 22:39]  # the same patch, stamped elsewhere
    sig_b = Q.texture_signature(copy, 80, 80, 8, F)
    sig_c = Q.texture_signature(_skin(9), 80, 80, 8, F)
    rep = Q.repetition([sig_a, sig_b], [30, 80], [30, 80], [8, 8])
    assert rep[0] > Q.MAX_REPEAT_NCC and "repetition" in Q.failures({"repetition": rep[0]})
    assert Q.repetition([sig_a, sig_c], [30, 80], [30, 80], [8, 8])[0] < Q.MAX_REPEAT_NCC


def test_texture_preservation_setting_moves_the_smooth_limit():
    v = {"texture": 0.45}
    assert "texture" in Q.failures(v, texture_keep=0.9)
    assert "texture" not in Q.failures(v, texture_keep=0.3)  # a smoother repair the user asked for


# --- In the engine -------------------------------------------------------------------------------

SIZE = 900


@pytest.fixture(scope="module")
def scene():
    img, truths, Fw = make_scene(size=SIZE, tone="III", seed=31, with_natural=False)
    return img, truths, Fw


def _run(img, **kw):
    return P.perfect_skin(img, oval_mask(SIZE, SIZE), [oval_face(SIZE, SIZE)], SkinPerfectionSettings(enabled=True, **kw))


def _pale_repairs(monkeypatch, amount):
    """Make every repair also lift the tone of its disc: contrast falls, the tone is wrong."""
    orig = P._correct_marks

    def bad(image, skin, marks, settings, Fw, edge_guard, avoid=()):
        out = orig(image, skin, marks, settings, Fw, edge_guard, avoid).copy()
        for m in marks:
            mask = np.zeros(out.shape[:2], np.uint8)
            cv2.circle(mask, (int(round(m.x)), int(round(m.y))), int(np.ceil(P._repair_radius(m, settings))), 1, -1)
            out[mask > 0] = np.clip(out[mask > 0] + amount, 0, 1)
        return out

    monkeypatch.setattr(P, "_correct_marks", bad)


def test_a_repair_with_the_wrong_tone_is_not_corrected_and_is_undone(scene, monkeypatch):
    img, truths, Fw = scene
    _pale_repairs(monkeypatch, 0.08)
    out, rep = _run(img)
    tried = [m for m in rep.marks if m.attempts > 0]
    assert tried and not [m for m in tried if m.status == "corrected"]
    failed = [m for m in tried if m.reason == "repairFailed"]
    assert len(failed) >= 0.8 * len(tried)
    assert all(m.status == "review" and "colour" in m.quality["failed"] and not m.quality["kept"] for m in failed)
    # Undone: no pale disc is left (the redness tool, which runs after, may still tone the mark down).
    for m in failed:
        y, x = int(round(m.y)), int(round(m.x))
        assert (out[y, x] - img[y, x]).max() < 0.02
    assert rep.quality.get("colour", 0) >= len(failed)
    assert rep.to_json()["quality"]["colour"] >= len(failed)


def test_a_small_tone_error_is_undone_too(scene, monkeypatch):
    """No failed repair is left in place for review: an uncertain repair must not change skin."""
    img, truths, Fw = scene
    _pale_repairs(monkeypatch, 0.025)  # ΔE just over the limit
    out, rep = _run(img)
    failed = [m for m in rep.marks if m.reason == "repairFailed"]
    assert failed
    s = SkinPerfectionSettings(enabled=True)
    passed = [m for m in rep.marks if m.status == "corrected"]
    for m in failed:
        assert m.status == "review" and not m.quality["kept"]
        if any((k.x - m.x) ** 2 + (k.y - m.y) ** 2 < (P._repair_radius(k, s) + P._repair_radius(m, s)) ** 2 for k in passed):
            continue  # a passing (here also lifted) repair overlaps it, and is rightly kept
        y, x = int(round(m.y)), int(round(m.x))
        assert (out[y, x] - img[y, x]).mean() < 0.01  # no lift (the redness tool may still shift red to green)


def test_a_mark_the_user_corrects_keeps_its_repair_and_shows_failed_checks(scene, monkeypatch):
    img, truths, Fw = scene
    _pale_repairs(monkeypatch, 0.08)
    _, rep = _run(img)
    m = next(m for m in rep.marks if m.reason == "repairFailed")
    cor = PerfectionOverride(x=m.x / SIZE, y=m.y / SIZE, radius=1.3 * m.r / SIZE, action="correct")
    _, rep2 = _run(img, overrides=[cor])
    m2 = min(rep2.marks, key=lambda k: (k.x - m.x) ** 2 + (k.y - m.y) ** 2)
    assert m2.status == "corrected" and "colour" in m2.quality["failed"] and m2.quality["kept"]


def test_good_repairs_pass_the_gate(scene):
    img, truths, Fw = scene
    out, rep = _run(img)
    tried = [m for m in rep.marks if m.attempts > 0]
    assert sum(m.status == "corrected" for m in tried) >= 0.9 * len(tried)
    for m in tried:
        assert set(m.quality["values"]) == set(Q.CHECKS)


def test_a_donor_never_carries_a_neighbouring_mark_into_the_repair():
    """Dark dots everywhere except under the repair: the donor must bring skin, not dots."""
    rng = np.random.default_rng(5)
    img = np.ones((200, 200, 3), np.float32) * np.array([0.75, 0.58, 0.48], np.float32)
    img += rng.normal(0, 0.004, (200, 200, 1)).astype(np.float32)
    taken = np.zeros((200, 200), np.uint8)
    for y in range(6, 200, 14):
        for x in range(6, 200, 14):
            if abs(x - 100) < 12 and abs(y - 100) < 12:
                continue
            cv2.circle(img, (x, y), 2, (0.45, 0.33, 0.28), -1)
            cv2.circle(taken, (x, y), 4, 1, -1)
    skin = np.ones((200, 200), np.float32)
    patch = P._donor_patch(img, skin, taken, (94, 94, 13, 13), 13, 4.0, 0)
    assert patch is not None
    assert P._luma(patch).min() > -0.05  # no dot (about -0.25 in luma) came along


def test_contrast_alone_does_not_make_a_mark_corrected(scene, monkeypatch):
    """The phase 2D rule: contrast fell (the old test passes), but the repair is a grey disc."""
    img, truths, Fw = scene
    orig = P._correct_marks

    def grey(image, skin, marks, settings, Fw, edge_guard, avoid=()):
        out = orig(image, skin, marks, settings, Fw, edge_guard, avoid).copy()
        for m in marks:
            mask = np.zeros(out.shape[:2], np.uint8)
            cv2.circle(mask, (int(round(m.x)), int(round(m.y))), int(np.ceil(P._repair_radius(m, settings))), 1, -1)
            px = out[mask > 0]
            out[mask > 0] = px.mean(1, keepdims=True)  # desaturated
        return out

    monkeypatch.setattr(P, "_correct_marks", grey)
    out, rep = _run(img)
    tried = [m for m in rep.marks if m.attempts > 0 and m.kind not in ("manual",)]
    old_rule = [m for m in tried if m.quality["values"]["residual"] is not None and m.quality["values"]["residual"] <= 1.0]
    assert old_rule  # these would have counted as corrected before 2D
    assert not [m for m in old_rule if m.status == "corrected"]
    assert any(contrast_at(out, t) >= 0.5 * contrast_at(img, t) for t in truths if t.kind != "mole")  # undone, not grey
