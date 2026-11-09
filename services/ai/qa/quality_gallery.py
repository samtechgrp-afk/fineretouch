"""Calibrating the phase 2D correction quality checks on real photographs.

Labels say what a mark is, not whether its repair looks natural, so the quality thresholds are
validated by a person looking at repairs. This page shows every repair that failed a check and
a sample of repairs that passed, before | after, with each check's value against its limit.
The reviewer notes which failures are false alarms and which passes look wrong; the thresholds
in app/engines/perfection_quality.py are then set on the validation split, never the test split.

    python -m scripts.perfection_benchmark --suite real --dataset D:/qa-photos --split validation \\
        --quality-gallery quality-review.html
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import cv2
import numpy as np

from app.engines import perfection_quality as Q

PASSED_SAMPLE = 4  # passed repairs shown per photo (randomly chosen, seeded)


def _limit(k: str) -> str:
    return {
        "residual": "≤ 1 (contrast left / allowed)", "colour": f"≤ {Q.MAX_DELTA_E} ΔE2000",
        "texture": f"{Q.TEXTURE_RANGE[0]}–{Q.TEXTURE_RANGE[1]}", "pores": f"≥ {Q.MIN_PORE_SHARE}",
        "edges": f"≤ {Q.MAX_EDGE_RATIO}", "repetition": f"≤ {Q.MAX_REPEAT_NCC}", "surround": f"≤ {Q.MAX_SURROUND_DL} L*",
    }[k]


def _crop_png(before: np.ndarray, after: np.ndarray, x: float, y: float, r: float, px: int = 140) -> str | None:
    h, w = before.shape[:2]
    half = int(max(10, 4.0 * r))
    x0, x1, y0, y1 = int(x) - half, int(x) + half, int(y) - half, int(y) + half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    pair = np.concatenate([before[y0:y1, x0:x1], after[y0:y1, x0:x1]], 1)
    pair = cv2.resize(np.clip(pair, 0, 1), (2 * px, px), interpolation=cv2.INTER_NEAREST if half < px // 2 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor((pair * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def collect(photo_id: str, before: np.ndarray, after: np.ndarray, marks: list, seed: int = 0) -> list[dict]:
    """Gallery entries for one photo: every failed repair and a few passed ones."""
    repaired = [m for m in marks if getattr(m, "quality", None)]
    failed = [m for m in repaired if m.quality["failed"]]
    passed = [m for m in repaired if not m.quality["failed"]]
    rng = np.random.default_rng(seed)
    if len(passed) > PASSED_SAMPLE:
        passed = [passed[i] for i in sorted(rng.choice(len(passed), PASSED_SAMPLE, replace=False))]
    out = []
    for m in failed + passed:
        png = _crop_png(before, after, m.x, m.y, m.r)
        if png is None:
            continue
        out.append({"photo": photo_id, "id": m.id, "kind": m.kind, "r": round(float(m.r), 1), "status": m.status,
                    "failed": list(m.quality["failed"]), "kept": m.quality.get("kept"), "incomplete": m.quality.get("incomplete"),
                    "values": m.quality["values"], "png": png})
    return out


def summary(entries_marks: list) -> dict:
    """Failure count per check and value percentiles over every repaired mark."""
    vals: dict[str, list[float]] = {k: [] for k in Q.CHECKS}
    fails = {k: 0 for k in Q.CHECKS}
    n = 0
    for m in entries_marks:
        if not getattr(m, "quality", None):
            continue
        n += 1
        for k, v in m.quality["values"].items():
            if v is not None:
                vals[k].append(float(v))
        for k in m.quality["failed"]:
            fails[k] += 1
    pct = {k: (dict(zip(("p5", "p50", "p95", "p99"), np.percentile(v, [5, 50, 95, 99]).round(3).tolist())) if v else None)
           for k, v in vals.items()}
    return {"repairs": n, "failed": fails, "percentiles": pct}


def write_html(entries: list[dict], stats: dict, path: str | Path) -> None:
    e = html.escape
    rows = []
    for x in entries:
        cells = "".join(
            f"<td class='{'bad' if k in x['failed'] else ''}'>{'–' if x['values'].get(k) is None else f'{x['values'][k]:.2f}'}</td>"
            for k in Q.CHECKS
        )
        outcome = ("passed" if not x["failed"] else
                   "kept (user chose Correct)" if x["kept"] else "original restored" if x["kept"] is False else "not finished (remaining)")
        rows.append(f"<tr><td><img src='data:image/png;base64,{x['png']}' width=280 height=140></td>"
                    f"<td>{e(x['photo'])}<br>#{x['id']} {e(x['kind'])} r={x['r']}<br><b>{e(outcome)}</b></td>{cells}"
                    f"<td><label><input type=checkbox> false alarm</label><br><label><input type=checkbox> looks wrong</label></td></tr>")
    head = "".join(f"<th>{e(k)}<br><small>{e(_limit(k))}</small></th>" for k in Q.CHECKS)
    pct = "".join(
        f"<tr><td>{e(k)}</td><td>{stats['failed'][k]}</td>"
        + ("".join(f"<td>{v:.2f}</td>" for v in stats["percentiles"][k].values()) if stats["percentiles"][k] else "<td colspan=4>not measurable</td>")
        + "</tr>" for k in Q.CHECKS)
    Path(path).write_text(f"""<!doctype html><meta charset="utf-8"><title>Repair quality review</title>
<style>body{{font:13px system-ui,sans-serif;margin:20px;color:#1b1b1f;background:#fff}}table{{border-collapse:collapse;margin:8px 0 20px}}
td,th{{border:1px solid #d4d4d8;padding:3px 6px;text-align:right;vertical-align:middle}}td:first-child,td:nth-child(2){{text-align:left}}
.bad{{background:#fde2e1;font-weight:600}}.note{{background:#fff7e6;border:1px solid #f5c26b;padding:8px 12px;border-radius:6px;max-width:860px}}</style>
<h1>Repair quality review</h1>
<p class="note">Each row is one repair: before | after. Red cells are checks that failed. Tick <b>false alarm</b> where a failed repair
looks natural, and <b>looks wrong</b> where a passed repair does not. Thresholds are set on the validation split only.</p>
<h2>Summary ({stats['repairs']} repairs)</h2>
<table><tr><th>Check</th><th>Failed</th><th>p5</th><th>p50</th><th>p95</th><th>p99</th></tr>{pct}</table>
<h2>Repairs</h2><table><tr><th>Before | after</th><th>Mark</th>{head}<th>Your call</th></tr>{''.join(rows)}</table>
""")
