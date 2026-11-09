"""Skin Perfection QA benchmark (metric definitions: qa/scoring.py).

    cd services/ai
    python -m scripts.perfection_benchmark                       # synthetic suite, 12 scenes
    python -m scripts.perfection_benchmark --quick               # 4 scenes (what CI runs)
    python -m scripts.perfection_benchmark --json r.json --html r.html --baseline qa/baselines/synthetic.json
    python -m scripts.perfection_benchmark --suite real --dataset D:/qa-photos --split validation --html real.html
    python -m scripts.perfection_benchmark --suite real --dataset D:/qa-photos --coverage
    python -m scripts.perfection_benchmark --perf D:/photos/IMG_0012.jpg --json perf.json   # time + memory

Synthetic and real results are separate reports; synthetic numbers are not evidence of
accuracy on real photographs. The functions below (evaluate, run_new) are kept for the tests.

A detection matches a truth when its centre is within max(1.5 r, 3 px) of it. Removal of an
imperfection is verified on the output (contrast against its ring fell to <= 40%); a natural
feature counts as damaged when its contrast fell below 50%. Texture: pore-band energy on clean
skin (away from every mark), output / input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engines.perfection import perfect_skin  # noqa: E402
from app.engines.skin import natural_skin_retouch, remove_acne, remove_dark_spots  # noqa: E402
from app.schemas import SkinPerfectionSettings  # noqa: E402
from tests.perfection_fixtures import (  # noqa: E402
    IMPERFECTIONS, NATURAL, contrast_at, make_scene, oval_face, oval_mask, pore_energy,
)

FLAGGED = ("corrected", "remaining", "review")


def _clean_mask(shape, truths, F):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = oval_mask(w, h) > 0.9
    for t in truths:
        m &= (xx - t.x) ** 2 + (yy - t.y) ** 2 > (4 * t.r + 3) ** 2
    return m


def evaluate(img, out, truths, F, marks=None) -> dict:
    res: dict = {"removed": {}, "found": {}, "damaged": {}}
    for kind in IMPERFECTIONS + NATURAL:
        ts = [t for t in truths if t.kind == kind]
        if not ts:
            continue
        before = np.array([contrast_at(img, t) for t in ts])
        after = np.array([contrast_at(out, t) for t in ts])
        gone = after <= 0.4 * np.maximum(before, 1e-6)
        if kind in IMPERFECTIONS:
            res["removed"][kind] = (int(gone.sum()), len(ts))
        else:
            res["damaged"][kind] = (int((after < 0.5 * before).sum()), len(ts))
        if marks is not None:
            hit = 0
            for t in ts:
                lim = max(1.5 * t.r, 3.0)
                if any((m["x"] - t.x) ** 2 + (m["y"] - t.y) ** 2 <= lim * lim and m["status"] in FLAGGED for m in marks):
                    hit += 1
            res["found"][kind] = (hit, len(ts))
    if marks is not None:
        flagged = [m for m in marks if m["status"] in FLAGGED]
        tp = 0
        for m in flagged:
            if any((m["x"] - t.x) ** 2 + (m["y"] - t.y) ** 2 <= max(1.5 * t.r, 3.0, m["r"]) ** 2
                   for t in truths if t.kind in IMPERFECTIONS):
                tp += 1
        imps = [t for t in truths if t.kind in IMPERFECTIONS]
        found = sum(res["found"][k][0] for k in IMPERFECTIONS if k in res["found"])
        precision = tp / max(len(flagged), 1)
        recall = found / max(len(imps), 1)
        res.update(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-9))
    rem = sum(v[0] for v in res["removed"].values())
    tot = sum(v[1] for v in res["removed"].values())
    res["removal_rate"] = rem / max(tot, 1)
    res["missed_rate"] = 1 - res["removal_rate"]
    nat = sum(v[0] for v in res["damaged"].values())
    res["natural_damaged"] = nat
    clean = _clean_mask(img.shape[:2], truths, F)
    res["texture_retention"] = pore_energy(out, clean, max(0.8, 0.0025 * F)) / max(pore_energy(img, clean, max(0.8, 0.0025 * F)), 1e-12)
    return res


def run_new(img, F, settings=None):
    h, w = img.shape[:2]
    s = settings or SkinPerfectionSettings(enabled=True)
    out, rep = perfect_skin(img, oval_mask(w, h), [oval_face(w, h)], s)
    marks = [{"x": m.x, "y": m.y, "r": m.r, "status": m.status, "kind": m.kind} for m in rep.marks]
    return out, marks, rep


def run_old(img, F):
    h, w = img.shape[:2]
    m, f = oval_mask(w, h), [oval_face(w, h)]
    c = remove_acne(img, m, f, 100, 1)
    c = remove_dark_spots(c, m, f, 100, 1, sigma_tone=0.012 * F, sigma_base=0.036 * F)
    return natural_skin_retouch(c, m, f, smoothing=0, texture_retention=100, spot_amount=100, sigma_tone=0.012 * F, sigma_base=0.036 * F)


SCENES_FULL = [
    dict(size=s, tone=t, seed=i, density=d)
    for i, (s, t, d) in enumerate([
        (1200, "I", "normal"), (1200, "III", "normal"), (1200, "V", "normal"), (1200, "VI", "normal"),
        (1600, "II", "dense"), (1600, "IV", "dense"), (1600, "VI", "dense"), (2400, "III", "normal"),
        (2400, "V", "normal"), (800, "IV", "normal"),
    ])
]
SCENES_QUICK = SCENES_FULL[:4]


def main() -> None:
    from qa import report as qa_report
    from qa.dataset import coverage, load_manifest, required_photos
    from qa.runners import run_real, run_synthetic

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--quick", action="store_true", help="synthetic: 4 scenes (light, deep, deep dense, deep hard light)")
    ap.add_argument("--dataset", help="real: folder with manifest.json, photos and .labels.json files")
    ap.add_argument("--split", choices=("validation", "test"), help="real: score one split only")
    ap.add_argument("--dev", action="store_true", help="synthetic: the tuning scenes (other seeds), not the benchmark scenes")
    ap.add_argument("--old", action="store_true", help="synthetic: also score the pre-v0.13 tools")
    ap.add_argument("--json", help="write the report as JSON")
    ap.add_argument("--html", help="write the report as HTML")
    ap.add_argument("--baseline", help="compare with this report JSON; exit 1 on a regression")
    ap.add_argument("--label", default="", help="free text shown in the report (e.g. a commit)")
    ap.add_argument("--coverage", action="store_true", help="real: print photos per coverage cell and the gaps, then stop")
    ap.add_argument("--perf", metavar="PHOTO", help="time one full-resolution render of PHOTO on this computer (no labels)")
    ap.add_argument("--repeats", type=int, default=2, help="--perf: timed runs after a warm-up run")
    a = ap.parse_args()

    if a.perf:
        from qa.runners import run_perf

        res = run_perf(a.perf, a.repeats)
        print(f"{res['photo']} ({res['megapixels']} MP): median {res['msMedian'] / 1000:.1f} s, peak {res['peakMB']} MB")
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, default=float))
        return

    if a.suite == "real":
        if not a.dataset:
            ap.error("--suite real needs --dataset")
        if a.coverage:
            cov = coverage(load_manifest(a.dataset))
            need = required_photos()
            for split, c in cov.items():
                print(f"{split}: {c['photos']} photos, {c['subjects']} subjects; about {need[split]} photos needed")
                for g in c["gaps"]:
                    print("  gap:", g)
            return
        rows = run_real(a.dataset, a.split)
    else:
        rows = run_synthetic(quick=a.quick, old=a.old, dev=a.dev)
    rep = qa_report.build(a.suite, rows, a.label)
    o = rep["overall"]
    print(f"\n{a.suite} ({o['photos']} photos): P {o['precision']:.3f} R {o['recall']:.3f} F1 {o['f1']:.3f} "
          f"removal {o['removalRate']:.3f} FP {o['falsePositives']} natural preserved {o['naturalPreserved']:.3f} "
          f"texture {o['textureRetention'] or 0:.3f} time {o['msMean'] / 1000:.1f}s peak {o['peakMB']} MB")
    print("   ", qa_report.SUITE_NOTE[a.suite])
    if a.old:
        old_rows = [r | {"score": r["old"] | {"ms": None, "peakMB": None}} for r in rows]
        oo = qa_report.aggregate(old_rows)
        print(f"pre-v0.13 tools: removal {oo['removalRate']:.3f} natural preserved {oo['naturalPreserved']:.3f} texture {oo['textureRetention'] or 0:.3f}")
        for r in rows:
            r["old"] = {k: v for k, v in r["old"].items() if k != "categories"}
    cmp_ = None
    if a.baseline:
        cmp_ = qa_report.compare(rep, json.loads(Path(a.baseline).read_text()))
        for x in cmp_["regressions"]:
            print("REGRESSION", x)
    qa_report.write(rep, a.json, a.html, cmp_)
    if cmp_ and cmp_["regressions"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
