"""Aggregate per-photo scores into the QA report, compare with a baseline, write JSON / HTML.

Synthetic and real results are never pooled: a report holds one suite, and its header says
which. Synthetic numbers measure the engine on generated skin only; they are not evidence of
accuracy on real photographs.
"""

from __future__ import annotations

import html
import json
import platform
import time
from pathlib import Path

import numpy as np

from .scoring import KEEP_KINDS

SUITE_NOTE = {
    "synthetic": "Synthetic skin with exact ground truth. Measures the engine on generated scenes only; "
                 "it is not evidence of accuracy on real photographs.",
    "real": "Real photographs with human labels (confirmed labels only).",
}

# Allowed change before a metric counts as a regression (higher is better unless marked).
TOLERANCE = {
    "precision": 0.02, "recall": 0.02, "f1": 0.02, "removalRate": 0.02, "naturalPreserved": 0.01,
    "textureRetention": 0.03,
}
LOWER_IS_BETTER = {"falsePositives": 2, "falseCorrections": 1, "naturalDamaged": 0, "claimedButVisible": 1}
METRICS = ("precision", "recall", "f1", "removalRate", "residualRate", "falsePositives", "falseCorrections", "falseReviews",
           "falsePositivesPerMP", "naturalPreserved", "naturalDamaged", "claimedButVisible", "textureRetention",
           "cleanSkinChangeL", "msMean", "msMax", "peakMB")


def aggregate(rows: list[dict]) -> dict:
    """Pool photos by counts (micro-average), not by averaging per-photo rates."""
    sc = [r["score"] for r in rows]
    if not sc:
        return {}
    s = lambda k: sum(x[k] for x in sc)  # noqa: E731
    imps, flagged, nat = s("imperfections"), s("flagged"), s("natural")
    p = s("truePositives") / flagged if flagged else (1.0 if not imps else 0.0)
    rc = s("found") / imps if imps else 1.0
    tex = [x["textureRetention"] for x in sc if x["textureRetention"] is not None]
    cl = [x["cleanSkinChangeL"] for x in sc if x["cleanSkinChangeL"] is not None]
    ms = [x["ms"] for x in sc if x["ms"] is not None]
    pk = [x["peakMB"] for x in sc if x["peakMB"] is not None]
    return {
        "photos": len(sc), "imperfections": imps, "natural": nat,
        "precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9),
        "removalRate": s("removed") / imps if imps else 1.0,
        "residualRate": 1 - (s("removed") / imps if imps else 1.0),
        "falsePositives": s("falsePositives"), "falseCorrections": s("falseCorrections"),
        "falseReviews": sum(x.get("falseReviews", 0) for x in sc),
        "falsePositivesOnNatural": s("falsePositivesOnNatural"),
        "falsePositivesPerMP": s("falsePositives") / max(s("skinMP"), 1e-6),
        "naturalPreserved": (nat - s("naturalDamaged")) / nat if nat else 1.0,
        "naturalDamaged": s("naturalDamaged"), "claimedButVisible": s("claimedButVisible"),
        "textureRetention": float(np.mean(tex)) if tex else None,
        "cleanSkinChangeL": float(np.mean(cl)) if cl else None,
        "msMean": float(np.mean(ms)) if ms else None, "msMax": float(np.max(ms)) if ms else None,
        "peakMB": float(np.max(pk)) if pk else None,
    }


def by_category(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        for k, c in r["score"]["categories"].items():
            o = out.setdefault(k, {key: 0 for key in c})
            for key, v in c.items():
                o[key] = o.get(key, 0) + v
    for k, o in out.items():
        n = max(o["n"], 1)
        if k in KEEP_KINDS:
            o["preserved"] = o["kept"] / n
        else:
            o["recall"] = o["found"] / n
            o["removal"] = o["removed"] / n
    return dict(sorted(out.items()))


def build(suite: str, rows: list[dict], label: str = "") -> dict:
    tones = sorted({r["tone"] for r in rows if r.get("tone")})
    return {
        "suite": suite, "note": SUITE_NOTE[suite], "label": label,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                    "python": platform.python_version()},
        "overall": aggregate(rows),
        "byTone": {t: aggregate([r for r in rows if r.get("tone") == t]) for t in tones},
        "byCategory": by_category(rows),
        "byToneCategory": {t: by_category([r for r in rows if r.get("tone") == t]) for t in tones},
        "rows": rows,
    }


def compare(report: dict, baseline: dict) -> dict:
    """Regressions and changes against a baseline of the same suite (by overall and by tone)."""
    if report["suite"] != baseline["suite"]:
        raise ValueError(f"cannot compare a {report['suite']} report with a {baseline['suite']} baseline")
    regressions, changes = [], []

    def check(scope: str, now: dict, base: dict) -> None:
        for k in METRICS:
            a, b = now.get(k), base.get(k)
            if a is None or b is None:
                continue
            changes.append({"scope": scope, "metric": k, "baseline": b, "now": a, "delta": a - b})
            if k in TOLERANCE and a < b - TOLERANCE[k]:
                regressions.append(f"{scope} {k}: {b:.3f} -> {a:.3f}")
            if k in LOWER_IS_BETTER and a > b + LOWER_IS_BETTER[k]:
                regressions.append(f"{scope} {k}: {b} -> {a}")

    check("overall", report["overall"], baseline["overall"])
    for t, agg in report["byTone"].items():
        if t in baseline.get("byTone", {}):
            check(f"tone {t}", agg, baseline["byTone"][t])
    return {"regressions": regressions, "changes": changes}


def _fmt(v, k: str = "") -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if k in ("msMean", "msMax"):
            return f"{v / 1000:.1f} s"
        if k == "peakMB":
            return f"{v:,.0f} MB"
        if k in ("falsePositivesPerMP", "cleanSkinChangeL"):
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)


LABELS = {
    "precision": "Precision", "recall": "Recall", "f1": "F1", "removalRate": "Removal rate", "residualRate": "Residual rate",
    "falsePositives": "False positives", "falseCorrections": "False corrections", "falseReviews": "False flags left for review (untouched)",
    "falsePositivesPerMP": "False positives / MP",
    "naturalPreserved": "Natural features preserved", "naturalDamaged": "Natural features damaged",
    "claimedButVisible": "Reported corrected but still visible", "textureRetention": "Texture retention",
    "cleanSkinChangeL": "Clean-skin change (ΔL*)", "msMean": "Time (mean)", "msMax": "Time (max)", "peakMB": "Peak memory",
}


def to_html(report: dict, comparison: dict | None = None) -> str:
    e = html.escape
    groups = [("Overall", report["overall"])] + [(f"Tone {t}", a) for t, a in report["byTone"].items()]
    head = "".join(f"<th>{e(g)}</th>" for g, _ in groups)
    body = "".join(
        f"<tr><th>{e(LABELS[k])}</th>" + "".join(f"<td>{_fmt(a.get(k), k)}</td>" for _, a in groups) + "</tr>"
        for k in METRICS
    )
    cat_rows = ""
    for k, c in report["byCategory"].items():
        if k in KEEP_KINDS:
            cat_rows += f"<tr><td>{e(k)}</td><td>keep</td><td>{c['n']}</td><td>–</td><td>–</td><td>{c['preserved']:.3f}</td><td>{c['wronglyFlagged']}</td></tr>"
        else:
            cat_rows += (f"<tr><td>{e(k)}</td><td>remove</td><td>{c['n']}</td><td>{c['recall']:.3f}</td><td>{c['removal']:.3f}</td>"
                         f"<td>–</td><td>{c['claimedButVisible']} visible but reported corrected</td></tr>")
    kinds = sorted({k for per in report.get("byToneCategory", {}).values() for k in per if k not in KEEP_KINDS})
    tc_head = "".join(f"<th>{e(k)}</th>" for k in kinds)
    tc_rows = ""
    for t, per in report.get("byToneCategory", {}).items():
        cells = "".join(
            f"<td>{per[k]['found']}/{per[k]['n']} found · {per[k]['removed']} removed</td>" if k in per else "<td>–</td>" for k in kinds
        )
        tc_rows += f"<tr><td>{e(t)}</td>{cells}</tr>"
    reg = ""
    if comparison is not None:
        items = "".join(f"<li>{e(x)}</li>" for x in comparison["regressions"]) or "<li>None</li>"
        reg = f"<h2>Regressions against the baseline</h2><ul>{items}</ul>"
    m = report["machine"]
    return f"""<!doctype html><meta charset="utf-8"><title>Skin Perfection QA ({e(report['suite'])})</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#1b1b1f;background:#fff}}table{{border-collapse:collapse;margin:8px 0 20px}}
td,th{{border:1px solid #d4d4d8;padding:4px 8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.note{{background:#fff7e6;border:1px solid #f5c26b;padding:8px 12px;border-radius:6px;max-width:820px}}</style>
<h1>Skin Perfection QA report: {e(report['suite'])} suite</h1>
<p class="note">{e(report['note'])}</p>
<p>{e(report['created'])} · {report['overall'].get('photos', 0)} photos · {e(m['platform'])} · {e(m['processor'])} {e(report.get('label') or '')}</p>
{reg}
<h2>Results by skin tone</h2><table><tr><th>Metric</th>{head}</tr>{body}</table>
<h2>Results by category</h2><table><tr><th>Category</th><th>Goal</th><th>Labels</th><th>Recall</th><th>Removal</th><th>Preserved</th><th>Notes</th></tr>{cat_rows}</table>
<h2>Imperfections by skin tone and category</h2><table><tr><th>Tone</th>{tc_head}</tr>{tc_rows}</table>
<p>Definitions: services/ai/qa/scoring.py. Timing and memory are machine-dependent; compare runs from the same computer.</p>
"""


def write(report: dict, json_path: str | Path | None, html_path: str | Path | None, comparison: dict | None = None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(report | ({"comparison": comparison} if comparison else {}), indent=1, default=float))
    if html_path:
        Path(html_path).write_text(to_html(report, comparison))
