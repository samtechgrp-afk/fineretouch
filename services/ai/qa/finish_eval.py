"""Professional Skin Finish: method comparison and measurements (phase 1).

For each photo and each smoothing method (engines/finish.METHODS) at a given setting, it measures
on the finish mask:

  bands      RMS of the fine / pore / bump / blotch bands of L* before and after
  edgeKept   contrast of real contours (nostrils, eyelids, lips, jaw, shadow edges) after / before
  edgeHalo   mean |change of L*| in a thin band around those contours (halo and softening)
  texture    fine + pore energy after / before
  shift      mean change of L*, a*, b* on skin (complexion)
  ms         time of the finish stage

and, when a professional retouch of the same photo is given (a reference), the same bands of the
reference on the same mask and the distance in band space (mean |log2(after / reference)| over
pore, bump and blotch).

Synthetic scenes (tests/perfection_fixtures.make_scene, six tones, marks off since Skin
Perfection removes them first): texture bands are measured on plain skin relief, and edge damage
on the same skin with a soft shadow edge and a dark nostril-like ellipse added (known geometry).

Usage (from services/ai):
    python -m qa.finish_eval out.json [--real manifest.json] [--smoothing 80 --texture 60 --pores 45]
A real manifest is a JSON list of {"name", "path", "reference"?}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from app.engines import finish as FIN
from app.pipeline import (
    S0,
    blur,
    build_masks,
    face_skin_likelihood,
    finish_photo,
    skin_probability,
)
from app.schemas import SkinFinishSettings
from tests.perfection_fixtures import make_scene, oval_mask

TONES = ("I", "II", "III", "IV", "V", "VI")


def settings(smoothing: float, texture: float, pores: float, strength: float = 100) -> SkinFinishSettings:
    return SkinFinishSettings(
        enabled=True, strength=strength,
        smoothing={"amount": smoothing}, texturePreservation={"amount": texture}, porePreservation={"amount": pores},
    )


def contour_scene(tone: str, seed: int, size: int = 1200) -> tuple[np.ndarray, np.ndarray, float]:
    """Synthetic skin plus a shadow edge and a nostril-like dark ellipse (known contours)."""
    # Marks off (Skin Perfection removes them before the finish runs); faint dark spots stay.
    rgb, _, F = make_scene(size=size, tone=tone, seed=seed, with_natural=False, marks=False)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # A curved shadow edge (like the side of the nose against the cheek): 30% darker beyond it.
    d = (xx - (0.62 * w + 0.08 * w * np.sin(yy / h * 3.0))) / max(1.0, 0.004 * F)
    shade = 1.0 / (1.0 + np.exp(-np.clip(d, -50, 50)))
    lab[..., 0] *= 1.0 - 0.3 * shade
    # A nostril-like ellipse, much darker than the skin around it.
    nost = np.zeros((h, w), np.float32)
    cv2.ellipse(nost, (int(0.38 * w), int(0.6 * h)), (int(0.03 * F), int(0.018 * F)), 15, 0, 360, 1.0, -1)
    nost = cv2.GaussianBlur(nost, (0, 0), max(1.0, 0.003 * F))
    lab[..., 0] -= nost * 0.55 * lab[..., 0]
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1).astype(np.float32)
    return out, oval_mask(w, h), F


def real_inputs(path: str):
    """(rgb float, face skin weight, masks, parse) as the renderer builds them."""
    from app import main

    rgb8 = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    faces = main.faces_service.detect(rgb8)
    parse = main.depth_service.parse(rgb8)
    src = rgb8.astype(np.float32) / 255.0
    h, w = src.shape[:2]
    masks = build_masks(faces, h, w, parse, False)
    low0 = blur(src, S0 * max(h, w))
    p = skin_probability(low0)
    if masks.has_face:
        region = np.maximum(masks.skin, masks.forehead)
        own = face_skin_likelihood(low0, masks.skin, region)
        skin = np.maximum(masks.skin * (0.35 + 0.65 * p) * (0.15 + 0.85 * own), masks.forehead * own)
    else:
        skin = np.zeros((h, w), np.float32)
    return src, skin.astype(np.float32), masks, parse


def measure(L0: np.ndarray, L1: np.ndarray, lab0: np.ndarray, lab1: np.ndarray, mask: np.ndarray, F: float,
            ref_L: np.ndarray | None) -> dict:
    sel = mask > 0.5
    area = FIN.relief_area(L0, mask, F)
    b0, b1 = FIN.band_energy(L0, mask, F, area), FIN.band_energy(L1, mask, F, area)
    edges = FIN.feature_edges(L0, mask, F)
    e0 = FIN.edge_contrast(L0, edges, F)
    ring = cv2.dilate(edges.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    out = {
        "before": {k: round(v, 3) for k, v in b0.items()},
        "after": {k: round(v, 3) for k, v in b1.items()},
        "edgeKept": round(FIN.edge_contrast(L1, edges, F) / e0, 3) if e0 > 0 else None,
        "edgeHalo": round(float(np.abs(L1 - L0)[ring].mean()), 2) if ring.any() else None,
        "texture": round(math.hypot(b1["fine"], b1["pore"]) / max(math.hypot(b0["fine"], b0["pore"]), 1e-6), 3),
        "shift": [round(float(v), 2) for v in (lab1 - lab0)[sel].mean(axis=0)],
        "edgePixels": int(edges.sum()),
    }
    if ref_L is not None:
        br = FIN.band_energy(ref_L, mask, F, area)
        out["reference"] = {k: round(v, 3) for k, v in br.items()}
        out["refDistance"] = round(float(np.mean([abs(math.log2(max(b1[k], 1e-6) / max(br[k], 1e-6))) for k in ("pore", "bump", "blotch")])), 3)
        out["origDistance"] = round(float(np.mean([abs(math.log2(max(b0[k], 1e-6) / max(br[k], 1e-6))) for k in ("pore", "bump", "blotch")])), 3)
    return out


def run_one(src: np.ndarray, skin: np.ndarray, masks, parse, st: SkinFinishSettings, method: str, ref: np.ndarray | None,
            save: str | None = None) -> dict:
    t = time.perf_counter()
    if masks is None:  # synthetic: face weight given, no parse
        F = float(skin.shape[1]) * 0.94
        out, rep = FIN.finish_skin(src, skin, st, F, method=method, measure=False)
    else:
        out, rep = finish_photo(src, skin, masks, parse, st, method=method)
        F = rep.face_width
    ms = (time.perf_counter() - t) * 1000
    lab0, lab1 = cv2.cvtColor(src, cv2.COLOR_RGB2Lab), cv2.cvtColor(out, cv2.COLOR_RGB2Lab)
    mask = skin if masks is None else np.maximum(skin, 0)  # measured on face skin (the reference's retouch area)
    ref_L = None if ref is None else cv2.cvtColor(ref, cv2.COLOR_RGB2Lab)[..., 0]
    m = measure(lab0[..., 0], lab1[..., 0], lab0, lab1, mask, F, ref_L)
    m["ms"] = round(ms)
    m["faceWidth"] = round(F, 1)
    if save:
        cv2.imwrite(save, cv2.cvtColor((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return m


def main_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real")
    ap.add_argument("--smoothing", type=float, default=80)
    ap.add_argument("--texture", type=float, default=60)
    ap.add_argument("--pores", type=float, default=45)
    ap.add_argument("--methods", default=",".join(FIN.METHODS))
    ap.add_argument("--save-dir")
    ap.add_argument("--no-synthetic", action="store_true")
    a = ap.parse_args(argv)
    st = settings(a.smoothing, a.texture, a.pores)
    methods = a.methods.split(",")
    results: dict = {"settings": {"smoothing": a.smoothing, "texture": a.texture, "pores": a.pores}, "photos": {}}
    if not a.no_synthetic:
        for i, tone in enumerate(TONES):
            # Texture bands on plain relief (no contour to confound them), edge damage on the
            # same skin with a shadow edge and a nostril added.
            src, mask, _ = contour_scene(tone, 900 + i)
            plain, _, _ = make_scene(size=1200, tone=tone, seed=900 + i, with_natural=False, marks=False)
            res = {}
            for m in methods:
                r = run_one(plain, mask, None, None, st, m, None)
                c = run_one(src, mask, None, None, st, m, None)
                r.update(edgeKept=c["edgeKept"], edgeHalo=c["edgeHalo"], edgePixels=c["edgePixels"])
                res[m] = r
            results["photos"][f"synthetic-{tone}"] = res
            print(f"synthetic-{tone}", {m: (r["edgeKept"], r["after"]["bump"]) for m, r in results["photos"][f"synthetic-{tone}"].items()}, flush=True)
    if a.real:
        for row in json.loads(Path(a.real).read_text()):
            src, skin, masks, parse = real_inputs(row["path"])
            ref = None
            if row.get("reference"):
                r8 = cv2.cvtColor(cv2.imread(row["reference"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                ref = r8.astype(np.float32) / 255.0 if r8.shape[:2] == src.shape[:2] else None
            res = {}
            for m in methods:
                save = f"{a.save_dir}/{row['name']}-{m}.jpg" if a.save_dir else None
                res[m] = run_one(src, skin, masks, parse, st, m, ref, save)
            results["photos"][row["name"]] = res
            print(row["name"], {m: (r["edgeKept"], r.get("refDistance"), r["ms"]) for m, r in res.items()}, flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))  # a partial run still leaves its results
    Path(a.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
