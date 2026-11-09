"""Skin Perfection QA: scoring, real-photo labels and splits, the report and the regression gate."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from qa import report as R
from qa.dataset import (
    ALL_KINDS, CONDITIONS, LIGHTING, TONE_GROUPS, assign_split, check_subject_separation, coverage, load_labels,
    load_manifest, required_photos, tone_group,
)
from qa.memory_probe import peak_memory
from qa.runners import run_real, run_synthetic
from qa.scoring import KEEP_KINDS, REMOVE_KINDS, Label, score_photo
from tests.perfection_fixtures import make_scene, oval_mask

ROOT = Path(__file__).resolve().parents[3]
BASELINES = Path(__file__).resolve().parents[1] / "qa" / "baselines"


@pytest.fixture(scope="module")
def scene():
    img, truths, F = make_scene(size=600, tone="IV", seed=3)
    labels = [Label(t.x, t.y, t.r, t.kind) for t in truths]
    return img, labels, F, oval_mask(600, 600) > 0.9


def _paint_out(img, labels):
    """An 'ideal' result: every imperfection replaced by the skin around it."""
    m = np.zeros(img.shape[:2], np.uint8)
    for t in labels:
        if not t.keep:
            cv2.circle(m, (int(t.x), int(t.y)), int(2.2 * t.r + 2), 255, -1)
    out8 = cv2.inpaint((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), m, 3, cv2.INPAINT_TELEA)
    return np.where(m[..., None] > 0, out8.astype(np.float32) / 255, img)


def test_untouched_photo_scores_zero_removal_and_full_preservation(scene):
    img, labels, F, region = scene
    s = score_photo(img, img, labels, [], region, F)
    assert s["removalRate"] == 0 and s["residualRate"] == 1
    assert s["naturalPreserved"] == 1 and s["naturalDamaged"] == 0
    assert s["recall"] == 0 and s["falsePositives"] == 0
    assert s["textureRetention"] == pytest.approx(1.0) and s["cleanSkinChangeL"] == pytest.approx(0.0)


def test_ideal_result_and_exact_marks(scene):
    img, labels, F, region = scene
    out = _paint_out(img, labels)
    marks = [{"x": t.x, "y": t.y, "r": t.r, "kind": t.kind, "status": "corrected", "confidence": 1.0} for t in labels if not t.keep]
    s = score_photo(img, out, labels, marks, region, F)
    assert s["precision"] == 1 and s["recall"] == 1 and s["f1"] == 1
    assert s["removalRate"] >= 0.9
    assert s["naturalPreserved"] == 1


def test_false_positives_and_claims_are_counted(scene):
    img, labels, F, region = scene
    nat = next(t for t in labels if t.kind == "mole")
    imp = next(t for t in labels if not t.keep)
    marks = [
        {"x": nat.x, "y": nat.y, "r": nat.r, "kind": "darkSpot", "status": "corrected", "confidence": 0.9},  # on a mole
        {"x": 300.0, "y": 300.0, "r": 3.0, "kind": "blemish", "status": "review", "confidence": 0.4},  # plain skin
        {"x": imp.x, "y": imp.y, "r": imp.r, "kind": imp.kind, "status": "corrected", "confidence": 0.9},  # but untouched
    ]
    s = score_photo(img, img, labels, marks, region, F)
    assert s["falsePositives"] == 2 and s["falsePositivesOnNatural"] == 1 and s["falseCorrections"] == 1
    assert s["claimedButVisible"] == 1  # reported corrected, still visible on the output
    assert s["categories"]["mole"]["wronglyFlagged"] == 1


def test_kinds_match_the_engine_categories():
    from app.schemas import PERFECTION_CATEGORIES
    from app.engines.perfection import PROTECTED_KINDS

    assert set(REMOVE_KINDS) - {"scar"} <= set(PERFECTION_CATEGORIES)
    assert set(PROTECTED_KINDS) <= set(KEEP_KINDS)


def test_label_kinds_match_the_app():
    ts = (ROOT / "packages/shared/src/labels.ts").read_text()
    def arr(name):
        body = re.search(rf"export const {name} = \[(.*?)\] as const", ts, re.S).group(1)
        return tuple(re.findall(r"'(\w+)'", body))
    assert arr("LABEL_REMOVE_KINDS") == REMOVE_KINDS
    assert arr("LABEL_KEEP_KINDS") == KEEP_KINDS
    assert "'fineretouch-labels'" in ts


def _write_labels(path: Path, w: int, h: int, labels: list[dict]):
    path.write_text(json.dumps({"format": "fineretouch-labels", "version": 1, "image": {"name": path.name, "width": w, "height": h},
                                "annotator": "test", "reviewer": None, "labels": labels}))


def test_label_file_validation_and_unconfirmed(tmp_path):
    p = tmp_path / "a.jpg.labels.json"
    _write_labels(p, 400, 200, [
        {"id": "1", "x": 0.5, "y": 0.5, "r": 0.01, "kind": "whitehead", "confirmed": True},
        {"id": "2", "x": 0.2, "y": 0.5, "r": 0.01, "kind": "mole", "source": "engine", "confirmed": False},
    ])
    lf = load_labels(p)
    px = lf.to_px()
    assert len(px) == 1 and px[0].x == 200 and px[0].y == 100 and px[0].r == pytest.approx(4.0)
    assert lf.unconfirmed == 1 and len(lf.to_px(include_unconfirmed=True)) == 2
    _write_labels(p, 400, 200, [{"id": "1", "x": 0.5, "y": 0.5, "r": 0.01, "kind": "zit"}])
    with pytest.raises(ValueError, match="unknown kind"):
        load_labels(p)
    p.write_text(json.dumps({"format": "other", "version": 1}))
    with pytest.raises(ValueError, match="not a FineRetouch label file"):
        load_labels(p)


def test_splits_are_by_subject_and_deterministic(tmp_path):
    assert assign_split("S01") == assign_split("S01")
    splits = [assign_split(f"S{i:03d}") for i in range(400)]
    assert 0.33 < splits.count("validation") / 400 < 0.47
    with pytest.raises(ValueError, match="both"):
        check_subject_separation([{"subject": "A", "split": "test"}, {"subject": "A", "split": "validation"}])
    (tmp_path / "manifest.json").write_text(json.dumps({"photos": [{"file": "x.jpg", "subject": "A", "consent": "none"}]}))
    with pytest.raises(ValueError, match="consent"):
        load_manifest(tmp_path)


def test_coverage_gaps_and_required_size():
    assert tone_group("monk-8") == "monk-7-8" and tone_group("monk-1") == "monk-1-3" and tone_group("fitz-IV") is None
    photos = [{"file": "a", "subject": "A", "split": "test", "tone": "monk-9", "lighting": "hard", "conditions": ["whitehead", "stubble"]}]
    cov = coverage(photos)
    assert cov["test"]["conditions"]["monk-9-10"]["whitehead"] == 1
    gaps = cov["test"]["gaps"]
    assert len(gaps) == len(TONE_GROUPS) * (len(CONDITIONS) + len(LIGHTING))  # one photo fills no cell
    need = required_photos()
    assert need == {"test": 64, "validation": 39, "total": 103}


def test_memory_probe_sees_an_allocation():
    with peak_memory() as p:
        a = np.ones((64, 1024, 1024), np.uint8)  # 64 MB, touched
        time.sleep(0.1)  # the probe samples every 10 ms
        del a
    assert p.method in ("sampled", "lifetime-peak", "unavailable")
    if p.method == "sampled":
        assert p.added_mb is not None and p.added_mb >= 40


def test_report_compare_flags_regressions():
    rows = [{"tone": "fitz-I", "score": {"imperfections": 10, "natural": 4, "flagged": 10, "truePositives": 9, "found": 9, "removed": 9,
                                         "skinMP": 1.0, "falsePositives": 1, "falseCorrections": 0, "falsePositivesOnNatural": 0,
                                         "naturalDamaged": 0, "claimedButVisible": 0, "textureRetention": 1.0, "cleanSkinChangeL": 0.1,
                                         "ms": 1000.0, "peakMB": 500.0, "categories": {"acne": {"n": 10, "found": 9, "removed": 9, "claimedButVisible": 0}}}}]
    base = R.build("synthetic", rows)
    worse = json.loads(json.dumps(rows))
    worse[0]["score"].update(found=7, removed=7, naturalDamaged=1)
    cmp_ = R.compare(R.build("synthetic", worse), base)
    joined = " ".join(cmp_["regressions"])
    assert "recall" in joined and "removalRate" in joined and "naturalDamaged" in joined
    assert R.compare(base, base)["regressions"] == []
    with pytest.raises(ValueError):
        R.compare(R.build("real", rows), base)
    html = R.to_html(base, cmp_)
    assert "not evidence of accuracy on real photographs" in html and "Results by skin tone" in html


def test_quick_suite_matches_its_baseline():
    """The regression gate: the 4 quick scenes must not fall below the committed baseline."""
    base = json.loads((BASELINES / "synthetic-quick.json").read_text())
    rep = R.build("synthetic", run_synthetic(quick=True, log=lambda *_: None))
    cmp_ = R.compare(rep, base)
    assert cmp_["regressions"] == [], cmp_["regressions"]


def test_real_suite_plumbing(tmp_path):
    """A labelled photo folder runs end to end through the app's render path (labels round-trip,
    sizes are checked, unconfirmed labels are skipped and reported)."""
    img, truths, F = make_scene(size=500, tone="V", seed=4)
    rgb8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    (tmp_path / "S01").mkdir()
    photo = tmp_path / "S01" / "p1.png"
    cv2.imwrite(str(photo), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    labels = [{"id": str(i), "x": t.x / 500, "y": t.y / 500, "r": t.r / 500, "kind": t.kind, "confirmed": True}
              for i, t in enumerate(truths)]
    labels.append({"id": "u", "x": 0.1, "y": 0.1, "r": 0.01, "kind": "acne", "source": "engine", "confirmed": False})
    _write_labels(photo.with_name("p1.png.labels.json"), 500, 500, labels)
    (tmp_path / "manifest.json").write_text(json.dumps({"photos": [
        {"file": "S01/p1.png", "subject": "S01", "tone": "monk-8", "lighting": "soft", "conditions": ["whitehead"], "consent": "internal-qa"}]}))
    rows = run_real(tmp_path, log=lambda *_: None)
    assert len(rows) == 1
    s = rows[0]["score"]
    assert s["imperfections"] == sum(t.kind in REMOVE_KINDS for t in truths)
    assert s["unconfirmedLabelsSkipped"] == 1
    assert rows[0]["toneGroup"] == "monk-7-8" and rows[0]["split"] in ("validation", "test")
    rep = R.build("real", rows)
    assert rep["suite"] == "real" and "Real photographs" in rep["note"]
    assert set(ALL_KINDS) >= {k for k in rep["byCategory"]}
