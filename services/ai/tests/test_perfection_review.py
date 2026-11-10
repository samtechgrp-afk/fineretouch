"""Skin Perfection phase 2C: ambiguous marks go to review (never corrected unasked), stubble and
freckle groups, and clusters of marks that merge into one patch at a coarse scale."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.engines import perfection as P
from app.engines.perfection import perfect_skin
from app.schemas import PerfectionOverride, SkinPerfectionSettings

from .perfection_fixtures import SKIN_TONES, Truth, _add, contrast_at, make_scene, oval_face, oval_mask

SIZE = 1000


def on(**kw) -> SkinPerfectionSettings:
    return SkinPerfectionSettings(enabled=True, **kw)


def _scene(tone: str, seed: int, paint) -> tuple[np.ndarray, float, dict]:
    """Plain generated skin (no labelled marks) with features painted by paint(L, A, B, F, rng)."""
    img, _, F = _clean(tone, seed)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = [np.ascontiguousarray(lab[..., k]) for k in range(3)]
    extra = paint(L, A, B, F, np.random.default_rng(seed))
    out = np.clip(cv2.cvtColor(np.stack([L, A, B], -1), cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    return out, F, extra


def _clean(tone: str, seed: int):
    img, truths, F = make_scene(size=SIZE, tone=tone, seed=seed, with_natural=False, marks=False)
    # marks=False paints a few barely visible dark spots for sensitivity tests; paint them out.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = [np.ascontiguousarray(lab[..., k]) for k in range(3)]
    L0 = SKIN_TONES[tone][0]
    for t in truths:
        _add(L, A, B, t.x, t.y, t.r, 0.05 * L0, -0.5, -2.0, soft=0.5)
    return np.clip(cv2.cvtColor(np.stack([L, A, B], -1), cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32), truths, F


def _run(img, **kw):
    return perfect_skin(img, oval_mask(SIZE, SIZE), [oval_face(SIZE, SIZE)], on(**kw))


def _near(rep, t: Truth, k: float = 1.5):
    return [m for m in rep.marks if (m.x - t.x) ** 2 + (m.y - t.y) ** 2 <= (max(k * t.r, 3.0, m.r)) ** 2]


@pytest.fixture(scope="module")
def stubble():
    """A stubble field (alike tiny dark dots) with one clearly bigger, darker dot in it."""
    def paint(L, A, B, F, rng):
        L0 = float(np.median(L))
        sx, sy = 0.6 * SIZE, 0.66 * SIZE
        dots = []
        for _ in range(90):
            cx, cy = sx + rng.uniform(-0.08, 0.08) * F, sy + rng.uniform(-0.06, 0.06) * F
            if any((cx - x) ** 2 + (cy - y) ** 2 < (0.012 * F) ** 2 for x, y in dots) or (cx - sx) ** 2 + (cy - sy) ** 2 < (0.02 * F) ** 2:
                continue
            dots.append((cx, cy))
            _add(L, A, B, cx, cy, 0.0012 * F, -0.2 * L0, -0.5, -0.5, soft=0.3)
        odd = Truth(sx, sy, 0.0026 * F, "blackhead")
        _add(L, A, B, odd.x, odd.y, odd.r, -0.34 * L0, -1.0, 0.5, soft=0.25)
        return {"odd": odd, "dots": [Truth(x, y, 0.0012 * F, "follicle") for x, y in dots]}

    img, F, extra = _scene("II", 21, paint)
    out, rep = _run(img)
    return img, F, extra, out, rep


def test_a_dot_unlike_its_stubble_field_is_reviewed_not_corrected(stubble):
    img, F, extra, out, rep = stubble
    odd = extra["odd"]
    hits = _near(rep, odd)
    assert hits and all(m.status == "review" and m.reason == "mayBeFollicle" for m in hits), [(m.kind, m.status, m.reason) for m in hits]
    assert contrast_at(out, odd) > 0.9 * contrast_at(img, odd)  # untouched
    # The field itself stays protected stubble, and none of it is corrected.
    corrected = [t for t in extra["dots"] if any(m.status == "corrected" for m in _near(rep, t))]
    assert not corrected
    assert sum(any(m.kind == "follicle" for m in _near(rep, t)) for t in extra["dots"]) >= 0.6 * len(extra["dots"])


def test_a_reviewed_mark_is_corrected_when_the_user_chooses_it(stubble):
    img, F, extra, out, rep = stubble
    odd = extra["odd"]
    m = _near(rep, odd)[0]
    cor = PerfectionOverride(x=m.x / SIZE, y=m.y / SIZE, radius=1.3 * m.r / SIZE, action="correct")
    out2, rep2 = _run(img, overrides=[cor])
    assert contrast_at(out2, odd) < 0.5 * contrast_at(img, odd)
    assert all(m2.reason is None for m2 in _near(rep2, odd) if m2.status == "corrected")


@pytest.mark.parametrize("r_k,d_k,expect", [(0.0062, 0.14, "review"), (0.0075, 0.18, "corrected")])
def test_a_spot_unlike_its_freckle_group_is_reviewed_not_corrected(r_k, d_k, expect):
    """A spot read as a freckle but bigger and darker than its group is reviewed; one far darker
    than any freckle is a dark spot and is corrected, as before."""
    def paint(L, A, B, F, rng):
        L0 = float(np.median(L))
        fx, fy = 0.4 * SIZE, 0.55 * SIZE
        frk = []
        for _ in range(40):
            cx, cy = fx + rng.normal(0, 0.05 * F), fy + rng.normal(0, 0.04 * F)
            r = rng.uniform(0.003, 0.0045) * F
            if any((cx - t.x) ** 2 + (cy - t.y) ** 2 < (2.8 * max(t.r, r)) ** 2 for t in frk) or (cx - fx) ** 2 + (cy - fy) ** 2 < (0.025 * F) ** 2:
                continue
            frk.append(Truth(cx, cy, r, "freckle"))
            _add(L, A, B, cx, cy, r, -0.07 * L0, 1.0, 4.0, soft=0.5)
            if len(frk) == 14:
                break
        spot = Truth(fx, fy, r_k * F, "darkSpot")
        _add(L, A, B, spot.x, spot.y, spot.r, -d_k * L0, 1.0, 5.0, soft=0.4)
        return {"spot": spot, "freckles": frk}

    img, F, extra = _scene("VI", 22, paint)
    out, rep = _run(img)
    spot = extra["spot"]
    hits = _near(rep, spot)
    if expect == "corrected":
        assert any(m.status == "corrected" and m.kind == "darkSpot" for m in hits)
        assert contrast_at(out, spot) < 0.5 * contrast_at(img, spot)
    else:
        assert hits and all(m.status == "review" and m.reason == "mayBeFreckle" for m in hits), [(m.kind, m.status, m.reason) for m in hits]
        assert contrast_at(out, spot) > 0.9 * contrast_at(img, spot)
    assert not [t for t in extra["freckles"] if any(m.status == "corrected" for m in _near(rep, t))]
    assert sum(contrast_at(out, t) >= 0.5 * contrast_at(img, t) for t in extra["freckles"]) == len(extra["freckles"])


def test_marks_merged_into_one_patch_at_a_coarse_scale_are_corrected_one_by_one():
    def paint(L, A, B, F, rng):
        L0 = float(np.median(L))
        cx, cy = 0.45 * SIZE, 0.5 * SIZE
        _add(L, A, B, cx, cy, 0.034 * F, -0.03 * L0, 1.0, 0.5, soft=1.0)  # broad, faint, flat patch
        acne = [Truth(cx + dx * F, cy + dy * F, 0.006 * F, "acne") for dx, dy in ((-0.015, -0.006), (0.012, -0.01), (0.002, 0.014))]
        for t in acne:
            _add(L, A, B, t.x, t.y, t.r * 1.8, -0.02 * L0, 4.0, 0.0, soft=0.8)
            _add(L, A, B, t.x, t.y, t.r, -0.06 * L0, 9.0, 1.0)
        return {"acne": acne}

    img, F, extra = _scene("IV", 23, paint)
    out, rep = _run(img)
    for t in extra["acne"]:
        assert any(m.status == "corrected" and m.kind in ("acne", "pimple") for m in _near(rep, t)), [(m.kind, m.status, round(m.r, 1)) for m in _near(rep, t, 4)]
        assert contrast_at(out, t) < 0.5 * contrast_at(img, t)
    assert not [m for m in rep.marks if m.kind == "birthmark" and m.r > 0.025 * F]


def test_report_carries_the_review_reason(stubble):
    img, F, extra, out, rep = stubble
    d = rep.to_json()
    reasons = {m["reason"] for m in d["marks"] if m["status"] == "review"}
    assert "mayBeFollicle" in reasons and reasons <= set(P.REVIEW_REASONS)
    assert all(m["reason"] is None for m in d["marks"] if m["status"] != "review")
    assert "r0" not in d["marks"][0]


def test_a_correct_override_for_one_mark_leaves_neighbouring_features_alone():
    """"Correct all remaining" places one override per blemish. One whose circle also covers a
    follicle, a freckle or a mark waiting for review must force only its own mark."""
    mk = lambda i, x, kind, status, reason=None: P.Mark(i, x, 100.0, 3.0, kind, 0.9, "face", 0.3, status=status, reason=reason)  # noqa: E731
    marks = [mk(0, 100.0, "blemish", "remaining"), mk(1, 104.0, "blackhead", "review", "mayBeFollicle"),
             mk(2, 96.0, "freckle", "protected"), mk(3, 103.0, "acne", "review", "lowConfidence")]
    o = PerfectionOverride(x=100.0 / 1000, y=100.0 / 1000, radius=6.0 / 1000, action="correct")
    out, _ = P._apply_overrides(marks, [o], 1000, 1000, 4)
    assert [m.status for m in out] == ["forced", "review", "protected", "forced"]
    assert out[1].reason == "mayBeFollicle"
    # Choosing the feature itself still corrects it.
    o2 = PerfectionOverride(x=104.0 / 1000, y=100.0 / 1000, radius=4.0 / 1000, action="correct")
    out2, _ = P._apply_overrides(out, [o2], 1000, 1000, 4)
    assert out2[1].status == "forced" and out2[1].reason is None and out2[2].status == "protected"


def test_a_correct_override_forces_a_neighbour_only_if_its_centre_is_inside_the_circle():
    """One bulk override per remaining blemish: a separate mark that only touches the circle (often
    one whose repair failed the quality checks) is not corrected because its neighbour was."""
    mk = lambda i, x: P.Mark(i, x, 100.0, 3.0, "blemish", 0.9, "face", 0.3, status="detected")  # noqa: E731
    marks = [mk(0, 100.0), mk(1, 105.0), mk(2, 108.0)]
    o = PerfectionOverride(x=100.0 / 1000, y=100.0 / 1000, radius=6.0 / 1000, action="correct")
    out, _ = P._apply_overrides(marks, [o], 1000, 1000, 3)
    assert [m.status for m in out] == ["forced", "forced", "detected"]
