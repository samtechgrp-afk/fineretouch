"""Golden renders: prove that a change leaves existing output untouched.

`record` renders a fixed set (synthetic skin on six tones, with several edit settings, plus any
real photos given in a manifest) through the full renderer and stores a SHA-256 of every output's
pixels. `check` renders the same set again and lists every case whose pixels differ.

Hashes depend on the OpenCV / NumPy build, so a baseline is only comparable on the machine (or
container image) that recorded it: record on the old commit, check on the new one, same machine.

Usage (from services/ai):
    python -m qa.golden record out.json [--manifest real.json]
    python -m qa.golden check out.json [--manifest real.json]

A real-photo manifest is a JSON list of {"name", "path", "adjustments"} (adjustments as the app
saves them). Real photos run with face landmarks and face parsing, as in the app's export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from app.pipeline import render
from app.schemas import Adjustments
from tests.perfection_fixtures import make_scene, oval_face

TONES = ("I", "II", "III", "IV", "V", "VI")

# Edit settings for the synthetic scenes: defaults, the classic skin sliders, Skin Perfection, and a
# heavy mix close to a real export (Auto Retouch plus strong skin and under-eye work).
SYNTH_EDITS: dict[str, dict] = {
    "defaults": {},
    "classic-skin": {"skinSmoothing": 60, "skinTexture": 65, "blemishRemoval": 80, "skinToneUnify": 25, "skinMultiplier": 2},
    "perfection": {"skinPerfection": {"enabled": True}},
    "heavy": {
        "exposure": 47, "contrast": -3, "temperature": -30, "tint": -18, "vibrance": 15, "skinSmoothing": 60,
        "skinTexture": 65, "blemishRemoval": 97, "skinToneUnify": 75, "acneRemoval": 100, "darkSpotRemoval": 85,
        "rednessReduction": 100, "skinNoise": 49, "underEyeTexture": 70, "eyeBrighten": 88, "darkCircles": 88,
        "sharpen": 10, "skinMultiplier": 2, "skinPerfection": {"enabled": True, "multiplier": 2},
    },
}


def digest(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def synthetic_cases():
    for i, tone in enumerate(TONES):
        rgb, _, _ = make_scene(size=900, tone=tone, seed=500 + i)
        rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        face = oval_face(900, 900)
        for name, edit in SYNTH_EDITS.items():
            yield f"synthetic-{tone}-{name}", rgb8, Adjustments(**edit), [face], None


def real_cases(manifest: str | None):
    if not manifest:
        return
    import cv2

    from app import main  # loads the face and parsing models

    for row in json.loads(Path(manifest).read_text()):
        bgr = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        adj = Adjustments(**row.get("adjustments", {}))
        faces = main.faces_service.detect(rgb8)
        parse = main.depth_service.parse(rgb8) if main._needs_skin_parse(adj) else None
        yield f"real-{row['name']}", rgb8, adj, faces, parse


def run(manifest: str | None) -> dict[str, str]:
    out = {}
    for name, rgb8, adj, faces, parse in [*synthetic_cases(), *real_cases(manifest)]:
        out[name] = digest(render(rgb8, adj, faces, parse=parse))
        print(name, out[name][:16], flush=True)
    return out


def main_cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("record", "check"))
    ap.add_argument("baseline")
    ap.add_argument("--manifest")
    a = ap.parse_args(argv)
    hashes = run(a.manifest)
    if a.mode == "record":
        Path(a.baseline).write_text(json.dumps(hashes, indent=1, sort_keys=True))
        return 0
    base = json.loads(Path(a.baseline).read_text())
    changed = sorted(k for k in base if hashes.get(k) != base[k])
    missing = sorted(k for k in base if k not in hashes)
    print(f"{len(base) - len(changed)}/{len(base)} identical" + (f"; changed: {changed}" if changed else "") + (f"; missing: {missing}" if missing else ""))
    return 1 if changed or missing else 0


if __name__ == "__main__":
    sys.exit(main_cli())
