"""Real-photo validation set: label files, the manifest, subject-based splits and coverage.

Label file (written by the app's label mode, next to the photo as `<photo>.labels.json`; the
photo itself is never written to):

    {
      "format": "fineretouch-labels", "version": 1,
      "image": {"name": "IMG_0012.CR3", "width": 6000, "height": 4000},
      "annotator": "...", "reviewer": "..." | null,
      "labels": [
        {"id": "l1", "x": 0.512, "y": 0.334, "r": 0.0021, "kind": "whitehead",
         "source": "manual" | "engine", "confirmed": true, "note": ""}
      ]
    }

x / y are shares of width / height; r is a share of the long edge (as in the engine report).
Labels pre-filled from the engine count only once a person confirmed them (confirmed: true), so
the benchmark never grades the engine against its own unreviewed output.

Manifest (`manifest.json` in the dataset folder; the folder stays on the user's machine):

    {
      "photos": [
        {"file": "S01/IMG_0012.jpg", "subject": "S01", "tone": "monk-8", "lighting": "hard",
         "region": "face" | "body" | "face+body", "conditions": ["whitehead", "stubble"],
         "consent": "internal-qa", "split": "validation" | "test" | null}
      ]
    }

Splits are by subject, never by photo: one person's photos all land in one split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .scoring import KEEP_KINDS, REMOVE_KINDS, Label

LABEL_FORMAT = "fineretouch-labels"
ALL_KINDS = REMOVE_KINDS + KEEP_KINDS
SPLITS = ("validation", "test")
VALIDATION_SHARE = 0.4
CONSENTS = ("internal-qa",)  # the only use photos in this set are cleared for

# Coverage matrix (docs/SKIN_PERFECTION_QA.md): tone groups x conditions, and x lighting.
TONE_GROUPS = {"monk-1-3": (1, 3), "monk-4-6": (4, 6), "monk-7-8": (7, 8), "monk-9-10": (9, 10)}
CONDITIONS = ("dense-acne", "whitehead", "blackhead", "hyperpigmentation", "post-acne", "freckles-moles", "stubble", "body-blemish")
LIGHTING = ("soft", "hard", "mixed")
MIN_PER_CELL = {"test": 5, "validation": 3}


@dataclass
class LabelFile:
    name: str
    width: int
    height: int
    labels: list[dict] = field(default_factory=list)
    annotator: str | None = None
    reviewer: str | None = None

    def to_px(self, include_unconfirmed: bool = False) -> list[Label]:
        L = float(max(self.width, self.height))
        out = []
        for d in self.labels:
            if d.get("kind") not in ALL_KINDS:
                continue
            if not d.get("confirmed", True) and not include_unconfirmed:
                continue
            out.append(Label(float(d["x"]) * self.width, float(d["y"]) * self.height, max(1.0, float(d["r"]) * L), d["kind"]))
        return out

    @property
    def unconfirmed(self) -> int:
        return sum(not d.get("confirmed", True) for d in self.labels)


def load_labels(path: str | Path) -> LabelFile:
    d = json.loads(Path(path).read_text())
    if d.get("format") != LABEL_FORMAT or int(d.get("version", 0)) != 1:
        raise ValueError(f"{path}: not a FineRetouch label file (format {d.get('format')!r}, version {d.get('version')!r})")
    im = d["image"]
    labels = []
    for i, x in enumerate(d.get("labels", [])):
        if not all(k in x for k in ("x", "y", "r", "kind")):
            raise ValueError(f"{path}: label {i} needs x, y, r and kind")
        if x["kind"] not in ALL_KINDS:
            raise ValueError(f"{path}: label {i} has unknown kind {x['kind']!r} (allowed: {', '.join(ALL_KINDS)})")
        if not (0 <= x["x"] <= 1 and 0 <= x["y"] <= 1 and 0 < x["r"] <= 0.1):
            raise ValueError(f"{path}: label {i} is out of range (x, y in 0..1, r in 0..0.1 of the long edge)")
        labels.append(x)
    return LabelFile(im["name"], int(im["width"]), int(im["height"]), labels, d.get("annotator"), d.get("reviewer"))


def tone_group(tone: str | None) -> str | None:
    """'monk-8' -> 'monk-7-8'."""
    if not tone or not tone.startswith("monk-"):
        return None
    try:
        v = int(tone.split("-")[1])
    except ValueError:
        return None
    return next((g for g, (lo, hi) in TONE_GROUPS.items() if lo <= v <= hi), None)


def assign_split(subject: str, validation_share: float = VALIDATION_SHARE) -> str:
    """Deterministic subject -> split (same subject, same split, on every machine)."""
    h = int(hashlib.sha256(subject.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if h < validation_share else "test"


def load_manifest(folder: str | Path) -> list[dict]:
    folder = Path(folder)
    photos = json.loads((folder / "manifest.json").read_text())["photos"]
    for p in photos:
        if p.get("consent") not in CONSENTS:
            raise ValueError(f"{p.get('file')}: consent must be one of {CONSENTS}; leave a photo out until it has one")
        p.setdefault("split", None)
        if p["split"] is None:
            p["split"] = assign_split(p["subject"])
        if p["split"] not in SPLITS:
            raise ValueError(f"{p['file']}: split must be validation or test")
    check_subject_separation(photos)
    return photos


def check_subject_separation(photos: list[dict]) -> None:
    seen: dict[str, str] = {}
    for p in photos:
        s = seen.setdefault(p["subject"], p["split"])
        if s != p["split"]:
            raise ValueError(f"subject {p['subject']} appears in both {s} and {p['split']}: splits must be by subject")


def coverage(photos: list[dict]) -> dict:
    """Photos per (split, tone group, condition) and (split, tone group, lighting), with the gaps
    against MIN_PER_CELL."""
    cells: dict[str, dict] = {}
    for split in SPLITS:
        cond = {g: {c: 0 for c in CONDITIONS} for g in TONE_GROUPS}
        light = {g: {lt: 0 for lt in LIGHTING} for g in TONE_GROUPS}
        for p in photos:
            g = tone_group(p.get("tone"))
            if p["split"] != split or g is None:
                continue
            for c in p.get("conditions", []):
                if c in cond[g]:
                    cond[g][c] += 1
            if p.get("lighting") in light[g]:
                light[g][p["lighting"]] += 1
        need = MIN_PER_CELL[split]
        gaps = [f"{g} / {c}: {n} of {need}" for g, row in cond.items() for c, n in row.items() if n < need]
        gaps += [f"{g} / {lt} light: {n} of {need}" for g, row in light.items() for lt, n in row.items() if n < need]
        cells[split] = {"conditions": cond, "lighting": light, "gaps": gaps,
                        "subjects": len({p["subject"] for p in photos if p["split"] == split}),
                        "photos": sum(p["split"] == split for p in photos)}
    return cells


def required_photos(conditions_per_photo: float = 2.5) -> dict:
    """Smallest set that can fill the matrix: each photo has one tone group and one lighting and
    shows about `conditions_per_photo` conditions (dense acne usually brings post-acne marks, etc.)."""
    out = {}
    for split, need in MIN_PER_CELL.items():
        by_light = len(TONE_GROUPS) * len(LIGHTING) * need
        by_cond = int(-(-len(TONE_GROUPS) * len(CONDITIONS) * need // conditions_per_photo))
        out[split] = max(by_light, by_cond)
    out["total"] = out["test"] + out["validation"]
    return out
