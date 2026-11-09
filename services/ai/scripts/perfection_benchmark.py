"""Skin Perfection benchmark: precision / recall / F1, removal, natural-feature safety and
texture retention on labelled synthetic skin, for the new engine and the previous tools.

    cd services/ai && python -m scripts.perfection_benchmark [--quick] [--json out.json]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--old", action="store_true", help="also score the previous acne / dark spot / blemish tools")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for sc in SCENES_QUICK if a.quick else SCENES_FULL:
        img, truths, F = make_scene(**sc)
        t0 = time.perf_counter()
        out, marks, rep = run_new(img, F)
        dt = time.perf_counter() - t0
        r = evaluate(img, out, truths, F, marks)
        row = {"scene": sc, "new": r, "ms": round(dt * 1000), "counts": rep.counts(), "passes": rep.passes}
        if a.old:
            row["old"] = evaluate(img, run_old(img, F), truths, F)
        rows.append(row)
        old = f" | old removal {row['old']['removal_rate']:.2f} nat {row['old']['natural_damaged']} tex {row['old']['texture_retention']:.2f}" if a.old else ""
        print(f"{sc['size']:>5} {sc['tone']:>3} {sc['density']:>6}: P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f} "
              f"removal {r['removal_rate']:.2f} nat {r['natural_damaged']} tex {r['texture_retention']:.2f} "
              f"{dt:.1f}s{old}  {r['removed']}")
    keys = ("precision", "recall", "f1", "removal_rate", "texture_retention")
    mean = {k: round(float(np.mean([r["new"][k] for r in rows])), 3) for k in keys}
    mean["natural_damaged"] = int(sum(r["new"]["natural_damaged"] for r in rows))
    print("mean new:", mean)
    if a.old:
        mo = {k: round(float(np.mean([r["old"][k] for r in rows])), 3) for k in ("removal_rate", "texture_retention")}
        mo["natural_damaged"] = int(sum(r["old"]["natural_damaged"] for r in rows))
        print("mean old:", mo)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "mean": mean}, indent=1, default=str))


if __name__ == "__main__":
    main()
