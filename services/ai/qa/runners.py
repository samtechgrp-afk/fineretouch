"""Run the engine over a suite and score every photo.

Synthetic: generated skin (tests/perfection_fixtures.py) with exact truths.
Real: a folder with manifest.json, the photos and their .labels.json files (qa/dataset.py).
Both use the same engine path settings (defaults unless given) and the same scoring.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.engines.perfection import perfect_skin
from app.schemas import SkinPerfectionSettings
from tests.perfection_fixtures import make_scene, oval_face, oval_mask

from .dataset import load_labels, load_manifest, tone_group
from .memory_probe import peak_memory
from .scoring import Label, score_photo

FITZ = {"I": "fitz-I", "II": "fitz-II", "III": "fitz-III", "IV": "fitz-IV", "V": "fitz-V", "VI": "fitz-VI"}

# Synthetic suite: every tone at normal density, dense acne on three tones, hard light on a light
# and a deep tone, and one large scene. Seeds are the row index, so the suite is reproducible.
SYNTHETIC_SCENES = [
    dict(size=1200, tone="I", density="normal", lighting=0.15),
    dict(size=1200, tone="II", density="normal", lighting=0.15),
    dict(size=1200, tone="III", density="normal", lighting=0.15),
    dict(size=1200, tone="IV", density="normal", lighting=0.15),
    dict(size=1200, tone="V", density="normal", lighting=0.15),
    dict(size=1200, tone="VI", density="normal", lighting=0.15),
    dict(size=1600, tone="II", density="dense", lighting=0.15),
    dict(size=1600, tone="IV", density="dense", lighting=0.15),
    dict(size=1600, tone="VI", density="dense", lighting=0.15),
    dict(size=1200, tone="III", density="normal", lighting=0.45),
    dict(size=1200, tone="VI", density="normal", lighting=0.45),
    dict(size=2400, tone="V", density="normal", lighting=0.15),
]
QUICK = (0, 5, 8, 10)  # light, deep, deep dense, deep hard light


def _scene_id(i: int, sc: dict) -> str:
    light = "hard" if sc["lighting"] > 0.3 else "soft"
    return f"syn{i:02d}-{sc['tone']}-{sc['density']}-{light}-{sc['size']}"


def run_synthetic(quick: bool = False, settings: SkinPerfectionSettings | None = None, old: bool = False, log=print) -> list[dict]:
    s = settings or SkinPerfectionSettings(enabled=True)
    rows = []
    for i, sc in enumerate(SYNTHETIC_SCENES):
        if quick and i not in QUICK:
            continue
        img, truths, F = make_scene(seed=i, **sc)
        h, w = img.shape[:2]
        mask, faces = oval_mask(w, h), [oval_face(w, h)]
        with peak_memory() as mem:
            t0 = time.perf_counter()
            out, rep = perfect_skin(img, mask, faces, s)
            ms = (time.perf_counter() - t0) * 1000
        marks = [{"x": m.x, "y": m.y, "r": m.r, "kind": m.kind, "status": m.status, "confidence": m.confidence} for m in rep.marks]
        labels = [Label(t.x, t.y, t.r, t.kind) for t in truths]
        score = score_photo(img, out, labels, marks, mask > 0.9, F, ms=ms, peak_mb=mem.peak_mb)
        score["addedMB"] = mem.added_mb
        row = {"id": _scene_id(i, sc), "tone": FITZ[sc["tone"]], "density": sc["density"],
               "lighting": "hard" if sc["lighting"] > 0.3 else "soft", "size": sc["size"], "score": score}
        if old:
            from .legacy import run_old

            row["old"] = score_photo(img, run_old(img, F), labels, [], mask > 0.9, F)
        rows.append(row)
        log(f"{row['id']:<28} P {score['precision']:.2f} R {score['recall']:.2f} F1 {score['f1']:.2f} "
            f"removal {score['removalRate']:.2f} FP {score['falsePositives']} nat-dmg {score['naturalDamaged']} "
            f"tex {score['textureRetention']:.2f} {ms / 1000:.1f}s peak {mem.peak_mb} MB")
    return rows


def run_real(folder: str | Path, split: str | None = None, settings: SkinPerfectionSettings | None = None, log=print) -> list[dict]:
    """Real photos through the same path as the app's export (decode, faces, parsing, render)."""
    from app import imageio, raw_engine
    from app.depth import DepthEstimator
    from app.landmarks import FaceLandmarkService
    from app.pipeline import render
    from app.schemas import Adjustments

    folder = Path(folder)
    s = (settings or SkinPerfectionSettings()).model_copy(update={"enabled": True})
    faces_service, depth_service = FaceLandmarkService(), DepthEstimator()
    rows = []
    for p in load_manifest(folder):
        if split and p["split"] != split:
            continue
        photo = folder / p["file"]
        lf = load_labels(photo.with_name(photo.name + ".labels.json"))
        data = photo.read_bytes()
        hq = imageio.decode_hq(data, None)
        h, w = hq.shape[:2]
        # Labels are normalised; a camera RAW can decode a few px larger than its preview, so only
        # the shape must match (a different photo, or a rotated one, is caught here).
        if abs(lf.width / lf.height - w / h) > 0.01 * (w / h):
            raise ValueError(f"{photo}: labels are for a {lf.width}x{lf.height} photo, it decodes to {w}x{h}")
        lf.width, lf.height = w, h
        rgb = raw_engine.to_uint8(hq)
        adj = Adjustments(skinPerfection=s)
        with peak_memory() as mem:
            t0 = time.perf_counter()
            faces = faces_service.detect(rgb)
            parse = depth_service.parse(rgb) if adj.skinMask.useSegmentation else None
            reports: list = []
            out8 = render(hq, adj, faces, parse=parse, report=reports)
            ms = (time.perf_counter() - t0) * 1000
        before = hq.astype(np.float32) / (65535.0 if hq.dtype == np.uint16 else 255.0)
        after = out8.astype(np.float32) / 255.0
        L = float(max(w, h))
        rep = reports[0] if reports else None
        marks = [{"x": m.x, "y": m.y, "r": m.r, "kind": m.kind, "status": m.status, "confidence": m.confidence} for m in (rep.marks if rep else [])]
        labels = lf.to_px()
        # Scored skin: discs around the labels (the annotated skin), so background and hair do
        # not dilute texture and false-positive rates.
        region = np.zeros((h, w), np.uint8)
        for t in labels:
            cv2.circle(region, (int(t.x), int(t.y)), int(max(12 * t.r, 0.02 * L)), 1, -1)
        F = rep.face_width if rep and getattr(rep, "face_width", 0) else 0.15 * L
        score = score_photo(before, after, labels, marks, region > 0, F, ms=ms, peak_mb=mem.peak_mb)
        score["addedMB"] = mem.added_mb
        score["unconfirmedLabelsSkipped"] = lf.unconfirmed
        rows.append({"id": p["file"], "subject": p["subject"], "split": p["split"], "tone": p.get("tone"),
                     "toneGroup": tone_group(p.get("tone")), "lighting": p.get("lighting"),
                     "conditions": p.get("conditions", []), "size": max(w, h), "score": score})
        log(f"{p['file']:<28} {p['split']:<10} P {score['precision']:.2f} R {score['recall']:.2f} removal {score['removalRate']:.2f} "
            f"FP {score['falsePositives']} nat-dmg {score['naturalDamaged']} {ms / 1000:.1f}s")
        del hq, rgb, out8, before, after
    return rows


def run_perf(photo: str | Path, repeats: int = 1, log=print) -> dict:
    """Time and peak memory of one full-resolution Skin Perfection render (the export path) on
    this computer. No labels needed; for before / after performance comparisons."""
    from app import imageio, raw_engine
    from app.depth import DepthEstimator
    from app.landmarks import FaceLandmarkService
    from app.pipeline import render
    from app.schemas import Adjustments

    faces_service, depth_service = FaceLandmarkService(), DepthEstimator()
    data = Path(photo).read_bytes()
    runs = []
    for i in range(repeats + 1):  # the first run loads the models and is reported separately
        with peak_memory() as mem:
            t0 = time.perf_counter()
            hq = imageio.decode_hq(data, None)
            rgb = raw_engine.to_uint8(hq)
            faces = faces_service.detect(rgb)
            parse = depth_service.parse(rgb)
            reports: list = []
            render(hq, Adjustments(skinPerfection=SkinPerfectionSettings(enabled=True)), faces, parse=parse, report=reports)
            ms = (time.perf_counter() - t0) * 1000
        rep = reports[0]
        runs.append({"ms": ms, "engineMs": rep.ms, "peakMB": mem.peak_mb, "counts": rep.counts(), "tiles": rep.tiles})
        log(f"{'warm-up' if i == 0 else f'run {i}'}: {ms / 1000:.1f} s total, engine {rep.ms / 1000:.1f} s, peak {mem.peak_mb} MB, {rep.counts()}")
        del hq, rgb
    h, w = imageio.decode_hq(data, None).shape[:2]
    timed = runs[1:]
    return {"photo": Path(photo).name, "megapixels": round(w * h / 1e6, 1), "warmup": runs[0], "runs": timed,
            "msMedian": float(np.median([r["ms"] for r in timed])), "peakMB": max(r["peakMB"] or 0 for r in timed)}
