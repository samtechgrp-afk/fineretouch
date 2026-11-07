"""Batch style sync and the local batch queue (FineRetouch v0.6).

"Grade once, apply everywhere": the photographer finishes one reference photo and the
engine carries that look across hundreds of RAW or JPEG files on the same machine, with
no network and no cloud queue.

Reference style transfer
  smart_sync_adjustments   copy the reference's edits, compensating exposure and white
                           balance for each target's own original (a Python port of
                           smartSyncAdjustments in packages/shared/src/style.ts).
  palette_adjustments      the reference's retouch settings plus an exact L*a*b* colour and
                           tone-curve transfer to the graded reference (colorMatch).
  fit_lut                  a 3D LUT fitted from a before/after pair: the reference's grade as
                           a colour transform, whatever software made it. Saved and loaded
                           as Adobe/Resolve .cube files (lut_to_cube / parse_cube).
  plan_adjustments         the per-photo adjustments for a batch item in any sync mode.

Local queue
  BatchQueue    SQLite (WAL) job and item tables. Survives restarts: items that were
                running when the app quit go back to pending (recover()). Items are
                claimed atomically, so several workers never render the same file.
  BatchRunner   worker threads with a thermal profile (quiet / balanced / performance):
                how many photos render at once, how many OpenCV threads each may use, and a
                cool-down between photos, so a 500-photo job does not pin every core and
                spin the fans up for an hour.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .color import color_stats
from .color_engine import encode_lut, identity_lut
from .raw_engine import RAW_EXTENSIONS
from .schemas import (
    LUT_MAX_SIZE,
    Adjustments,
    BatchJobSettings,
    ColorGrading,
    ColorMatch,
    ColorStats,
    HslMixer,
    LookLut,
)

log = logging.getLogger("fineretouch.batch")

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"} | set(RAW_EXTENSIONS))
MAX_DISCOVERED = 50_000


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_images(sources: Iterable[str | Path], recursive: bool = True, limit: int = MAX_DISCOVERED) -> list[tuple[Path, Path]]:
    """Photos under each source (a folder or a single file), sorted, as (file, root) pairs.

    root is the folder the file was found under, so outputs can mirror sub-folders. Hidden
    files and folders (".", e.g. macOS "._" resource forks) are skipped; duplicates are
    listed once.
    """
    seen: set[Path] = set()
    out: list[tuple[Path, Path]] = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file():
            files, root = [p], p.parent
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            files, root = sorted(f for f in it if f.is_file()), p
        else:
            raise FileNotFoundError(f"No such file or folder: {p}")
        for f in files:
            rel = f.relative_to(root).parts
            if any(part.startswith(".") for part in rel) or f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = f.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append((f, root))
            if len(out) >= limit:
                return out
    return out


def output_path(source: Path, root: Path, output_dir: Path, fmt: str, prefix: str = "") -> Path:
    """Mirror source's place under root inside output_dir, with the export extension."""
    ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}[fmt]
    rel = source.relative_to(root)
    return output_dir / rel.parent / f"{prefix}{rel.stem}{ext}"


# ---------------------------------------------------------------------------
# Reference style transfer
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def _exposure_for_lightness(from_l: float, to_l: float) -> float:
    k = ((to_l + 16) / max(from_l + 16, 1)) ** 3
    return _clamp(50 * math.log2(max(k, 1e-3)), -100, 100)


def _round_half_up(v: float) -> int:
    """Math.round semantics (halves toward +infinity), so results match the TypeScript."""
    return math.floor(v + 0.5)


def smart_sync_adjustments(hero: Adjustments, hero_stats: ColorStats, target_stats: ColorStats) -> Adjustments:
    """Copy the hero's edits, plus the exposure and white-balance difference between the two
    originals, so photos shot in drifting light land on one look. Mirrors
    smartSyncAdjustments (packages/shared/src/style.ts); a test pins the two together."""
    hL, ha, hb = hero_stats.mean
    tL, ta, tb = target_stats.mean
    L = (hL + tL) / 2
    d_exp = _clamp(_exposure_for_lightness(tL, hL), -60, 60)
    d_temp = _clamp(_clamp(100 * (hb - tb) / (0.15 * max(L, 10)), -100, 100), -50, 50)
    d_tint = _clamp(_clamp(100 * (ha - ta) / (0.078 * max(L, 10)), -100, 100), -50, 50)
    return hero.model_copy(update={
        "exposure": _round_half_up(_clamp(hero.exposure + d_exp, -100, 100)),
        "temperature": _round_half_up(_clamp(hero.temperature + d_temp, -100, 100)),
        "tint": _round_half_up(_clamp(hero.tint + d_tint, -100, 100)),
    })


TONE_AND_COLOUR = ("exposure", "contrast", "highlights", "shadows", "temperature", "tint", "saturation", "vibrance")


def palette_adjustments(reference: Adjustments, graded_reference_stats: ColorStats, amount: float = 100) -> Adjustments:
    """The reference's retouch (skin, face, clothing, detail) with its colour and tone replaced
    by a statistical transfer to the *graded* reference: each target is matched to how the
    finished reference looks, from wherever its own original starts."""
    update: dict = {k: 0 for k in TONE_AND_COLOUR}
    update.update(hsl=HslMixer(), colorGrading=ColorGrading(), lut=None,
                  colorMatch=ColorMatch(reference=graded_reference_stats, amount=_clamp(amount, 0, 100)))
    return reference.model_copy(update=update)


def plan_adjustments(settings: BatchJobSettings, target_rgb8: np.ndarray | None) -> Adjustments:
    """Adjustments for one batch photo. target_rgb8 (a small proxy of the original) is only
    needed by the smart mode; the others are the same for every photo."""
    sync = settings.sync
    adj = settings.adjustments
    if sync.mode == "smart":
        if sync.referenceStats is None or target_rgb8 is None:
            return adj
        return smart_sync_adjustments(adj, sync.referenceStats, color_stats(target_rgb8.astype(np.float32) / 255.0))
    if sync.mode == "palette" and sync.referenceLookStats is not None:
        return palette_adjustments(adj, sync.referenceLookStats, sync.amount)
    return adj


# --- 3D LUTs -----------------------------------------------------------------------

def _smooth3(grid: np.ndarray, passes: int) -> np.ndarray:
    """Separable [1 2 1] smoothing along the three LUT axes, edge-clamped."""
    g = grid
    for _ in range(passes):
        for ax in range(3):
            pad = [(0, 0)] * g.ndim
            pad[ax] = (1, 1)
            p = np.pad(g, pad, mode="edge")
            sl = [slice(None)] * g.ndim
            lo, mid, hi = list(sl), list(sl), list(sl)
            lo[ax], mid[ax], hi[ax] = slice(0, -2), slice(1, -1), slice(2, None)
            g = 0.25 * p[tuple(lo)] + 0.5 * p[tuple(mid)] + 0.25 * p[tuple(hi)]
    return g


def fit_lut(before: np.ndarray, after: np.ndarray, size: int = 17, max_pixels: int = 600_000) -> np.ndarray:
    """Fit a size^3 RGB LUT mapping `before` colours to `after` colours.

    before / after: the same photo before and after grading (HxWx3, uint8 or float in 0..1,
    any size; `after` is resized to `before`). Each pixel votes for the trilinear LUT nodes
    around its original colour (splatting), the graded colour is averaged per node, and
    nodes no pixel reached take the mean *offset* of their neighbourhood, diffused outward
    from the colours that were seen. Far from the data the LUT falls back to identity
    plus that smooth offset, so colours the reference never contained still move with the
    grade instead of jumping. Returns float32 [b, g, r, 3].
    """
    if not 2 <= size <= LUT_MAX_SIZE:
        raise ValueError(f"LUT size must be 2..{LUT_MAX_SIZE}")
    b = before.astype(np.float32) / (255.0 if before.dtype == np.uint8 else 1.0)
    a = after.astype(np.float32) / (255.0 if after.dtype == np.uint8 else 1.0)
    if a.shape[:2] != b.shape[:2]:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    src = b.reshape(-1, 3)
    dst = a.reshape(-1, 3)
    if src.shape[0] > max_pixels:
        idx = np.random.default_rng(0).choice(src.shape[0], max_pixels, replace=False)
        src, dst = src[idx], dst[idx]
    src = np.clip(src, 0, 1)
    off = np.clip(dst, 0, 1) - src
    n = size
    c = src * (n - 1)
    i0 = np.minimum(np.floor(c).astype(np.int64), n - 2)
    f = c - i0
    acc = np.zeros((n * n * n, 3), np.float64)
    wsum = np.zeros(n * n * n, np.float64)
    for db in (0, 1):
        wb = f[:, 2] if db else 1 - f[:, 2]
        for dg in (0, 1):
            wg = f[:, 1] if dg else 1 - f[:, 1]
            for dr in (0, 1):
                wr = f[:, 0] if dr else 1 - f[:, 0]
                w = wb * wg * wr
                idx = ((i0[:, 2] + db) * n + (i0[:, 1] + dg)) * n + (i0[:, 0] + dr)
                np.add.at(wsum, idx, w)
                for ch in range(3):
                    np.add.at(acc[:, ch], idx, w * off[:, ch])
    acc = acc.reshape(n, n, n, 3)
    wsum = wsum.reshape(n, n, n)
    # Diffuse offsets into unseen colours: normalised convolution at growing radii.
    est = np.zeros_like(acc)
    have = np.zeros_like(wsum)
    for passes in (0, 1, 2, 4, 8, 16, 32):
        num = _smooth3(acc, passes) if passes else acc
        den = _smooth3(wsum, passes) if passes else wsum
        fill = (have < 0.5) & (den > 1e-6)
        est[fill] = num[fill] / den[fill][..., None]
        have[fill] = 1.0
    if (have < 0.5).any():  # no data at all
        est[have < 0.5] = 0.0
    confidence = np.clip(wsum / 4.0, 0.0, 1.0)[..., None]
    smooth_est = _smooth3(est, 1)
    offsets = est * confidence + smooth_est * (1 - confidence)
    return np.clip(identity_lut(n) + offsets.astype(np.float32), 0.0, 1.0).astype(np.float32)


def look_from_lut(lut: np.ndarray, name: str = "Look", amount: float = 100) -> LookLut:
    return LookLut(name=name[:120], size=int(lut.shape[0]), data=encode_lut(lut), amount=amount)


def lut_to_cube(lut: np.ndarray, title: str = "FineRetouch look") -> str:
    """Adobe / DaVinci .cube text (red index fastest)."""
    n = lut.shape[0]
    safe = title.replace('"', "'").replace("\n", " ")[:120]
    lines = [f'TITLE "{safe}"', f"LUT_3D_SIZE {n}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in lut.reshape(-1, 3)]
    return "\n".join(lines) + "\n"


def parse_cube(text: str) -> tuple[np.ndarray, str]:
    """Parse a .cube 3D LUT. Returns (lut [b, g, r, 3] in 0..1, title).

    Supports TITLE, LUT_3D_SIZE (2..LUT_MAX_SIZE), DOMAIN_MIN / DOMAIN_MAX (values are
    rescaled to 0..1); 1D LUT files are rejected.
    """
    size = None
    title = "Imported LUT"
    lo = np.zeros(3, np.float32)
    hi = np.ones(3, np.float32)
    values: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key = line.split()[0].upper()
        if key == "TITLE":
            title = line[5:].strip().strip('"') or title
        elif key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "LUT_1D_SIZE":
            raise ValueError("1D LUTs are not supported; export a 3D .cube")
        elif key == "DOMAIN_MIN":
            lo = np.array([float(v) for v in line.split()[1:4]], np.float32)
        elif key == "DOMAIN_MAX":
            hi = np.array([float(v) for v in line.split()[1:4]], np.float32)
        elif key[0].isdigit() or key[0] in "-.":
            values.append([float(v) for v in line.split()[:3]])
    if size is None or not 2 <= size <= LUT_MAX_SIZE:
        raise ValueError(f"LUT_3D_SIZE missing or outside 2..{LUT_MAX_SIZE}")
    if len(values) != size**3:
        raise ValueError(f"Expected {size ** 3} LUT entries, found {len(values)}")
    lut = (np.array(values, np.float32) - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(lut, 0.0, 1.0).reshape(size, size, size, 3), title


# ---------------------------------------------------------------------------
# SQLite queue
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,          -- queued | running | paused | done | cancelled
    output_dir TEXT NOT NULL,
    settings TEXT NOT NULL,        -- BatchJobSettings JSON
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    source TEXT NOT NULL,
    output TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending | running | done | failed | cancelled
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started REAL,
    finished REAL,
    ms REAL,
    UNIQUE (job_id, source)
);
CREATE INDEX IF NOT EXISTS items_job_status ON items (job_id, status, seq);
"""

JOB_ACTIVE = ("queued", "running")


@dataclass(frozen=True)
class BatchItem:
    id: int
    job_id: str
    source: str
    output: str
    attempts: int


class BatchQueue:
    """Thread-safe SQLite job queue. One connection, serialised by a lock (SQLite's own
    writer lock would serialise the writes anyway, and items take seconds to render)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- jobs ---------------------------------------------------------------------

    def create_job(self, name: str, files: Sequence[tuple[Path, Path]], output_dir: str | Path, settings: BatchJobSettings) -> str:
        """Queue every (file, root) pair. Outputs mirror sub-folders inside output_dir; two
        sources that would land on the same output name get a -2, -3, ... suffix."""
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        out_dir = Path(output_dir)
        used: set[str] = set()
        rows = []
        for seq, (src, root) in enumerate(files):
            out = output_path(src, root, out_dir, settings.output.format, settings.output.prefix)
            cand, k = out, 2
            while str(cand).lower() in used:
                cand = out.with_name(f"{out.stem}-{k}{out.suffix}")
                k += 1
            used.add(str(cand).lower())
            rows.append((job_id, seq, str(src), str(cand)))
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT INTO jobs (id, name, status, output_dir, settings, created, updated) VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                    (job_id, name, str(out_dir), settings.model_dump_json(), now, now),
                )
                self._db.executemany(
                    "INSERT INTO items (job_id, seq, source, output, status) VALUES (?, ?, ?, ?, 'pending')", rows
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return job_id

    def settings(self, job_id: str) -> BatchJobSettings:
        with self._lock:
            row = self._db.execute("SELECT settings FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return BatchJobSettings.model_validate_json(row["settings"])

    def set_status(self, job_id: str, action: str) -> None:
        """pause | resume | cancel | retry (failed items back to pending)."""
        now = time.time()
        with self._lock:
            row = self._db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            status = row["status"]
            self._db.execute("BEGIN IMMEDIATE")
            if action == "pause" and status in JOB_ACTIVE:
                self._db.execute("UPDATE jobs SET status = 'paused', updated = ? WHERE id = ?", (now, job_id))
            elif action == "resume" and status == "paused":
                self._db.execute("UPDATE jobs SET status = 'queued', updated = ? WHERE id = ?", (now, job_id))
            elif action == "cancel" and status not in ("done", "cancelled"):
                self._db.execute("UPDATE items SET status = 'cancelled' WHERE job_id = ? AND status = 'pending'", (job_id,))
                self._db.execute("UPDATE jobs SET status = 'cancelled', updated = ? WHERE id = ?", (now, job_id))
            elif action == "retry":
                n = self._db.execute(
                    "UPDATE items SET status = 'pending', attempts = 0, error = NULL WHERE job_id = ? AND status IN ('failed', 'cancelled')",
                    (job_id,),
                ).rowcount
                if n:
                    self._db.execute("UPDATE jobs SET status = 'queued', updated = ? WHERE id = ?", (now, job_id))
            elif action not in ("pause", "resume", "cancel", "retry"):
                self._db.execute("ROLLBACK")
                raise ValueError(f"Unknown action {action!r}")
            self._db.execute("COMMIT")

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def recover(self) -> int:
        """After a crash or quit: items left 'running' go back to pending, jobs to queued."""
        with self._lock:
            n = self._db.execute("UPDATE items SET status = 'pending' WHERE status = 'running'").rowcount
            self._db.execute("UPDATE jobs SET status = 'queued' WHERE status = 'running'")
        return n

    # --- items --------------------------------------------------------------------

    def claim(self) -> BatchItem | None:
        """Atomically take the next pending item of the oldest active job."""
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    """SELECT i.id, i.job_id, i.source, i.output, i.attempts FROM items i
                       JOIN jobs j ON j.id = i.job_id
                       WHERE i.status = 'pending' AND j.status IN ('queued', 'running')
                       ORDER BY j.created, i.seq LIMIT 1"""
                ).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return None
                self._db.execute(
                    "UPDATE items SET status = 'running', attempts = attempts + 1, started = ? WHERE id = ?", (now, row["id"])
                )
                self._db.execute("UPDATE jobs SET status = 'running', updated = ? WHERE id = ? AND status = 'queued'", (now, row["job_id"]))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return BatchItem(row["id"], row["job_id"], row["source"], row["output"], row["attempts"] + 1)

    def complete(self, item: BatchItem, ms: float) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE items SET status = 'done', finished = ?, ms = ?, error = NULL WHERE id = ? AND status = 'running'",
                (time.time(), ms, item.id),
            )
            self._finish_if_drained(item.job_id)

    def fail(self, item: BatchItem, error: str, max_attempts: int = 2) -> None:
        """Retry once (a file still being copied from a card, a transient lock), then fail."""
        with self._lock:
            status = "pending" if item.attempts < max_attempts else "failed"
            self._db.execute(
                "UPDATE items SET status = ?, error = ?, finished = ? WHERE id = ? AND status = 'running'",
                (status, error[:500], time.time(), item.id),
            )
            self._finish_if_drained(item.job_id)

    def release(self, item: BatchItem) -> None:
        """Hand an item back unrendered (the job was paused or the runner stopped)."""
        with self._lock:
            self._db.execute(
                "UPDATE items SET status = 'pending', attempts = MAX(0, attempts - 1) WHERE id = ? AND status = 'running'",
                (item.id,),
            )

    def job_status(self, job_id: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["status"] if row else None

    def _finish_if_drained(self, job_id: str) -> None:
        left = self._db.execute(
            "SELECT COUNT(*) FROM items WHERE job_id = ? AND status IN ('pending', 'running')", (job_id,)
        ).fetchone()[0]
        if left == 0:
            self._db.execute(
                "UPDATE jobs SET status = 'done', updated = ? WHERE id = ? AND status IN ('queued', 'running')",
                (time.time(), job_id),
            )

    # --- progress -----------------------------------------------------------------

    def progress(self, job_id: str, workers: int = 1, recent_errors: int = 5) -> dict:
        with self._lock:
            job = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            counts = {r["status"]: r["n"] for r in self._db.execute(
                "SELECT status, COUNT(*) AS n FROM items WHERE job_id = ? GROUP BY status", (job_id,))}
            avg = self._db.execute(
                "SELECT AVG(ms) FROM (SELECT ms FROM items WHERE job_id = ? AND status = 'done' ORDER BY finished DESC LIMIT 20)",
                (job_id,),
            ).fetchone()[0]
            errors = [dict(r) for r in self._db.execute(
                "SELECT source, error FROM items WHERE job_id = ? AND status = 'failed' ORDER BY finished DESC LIMIT ?",
                (job_id, recent_errors))]
            current = [r["source"] for r in self._db.execute(
                "SELECT source FROM items WHERE job_id = ? AND status = 'running' ORDER BY started", (job_id,))]
        total = sum(counts.values())
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        remaining = counts.get("pending", 0) + counts.get("running", 0)
        eta = (avg or 0.0) * remaining / max(1, workers) / 1000.0 if avg else None
        return {
            "id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "outputDir": job["output_dir"],
            "created": job["created"],
            "total": total,
            "done": done,
            "failed": failed,
            "cancelled": counts.get("cancelled", 0),
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "percent": round(100.0 * (done + failed) / total, 1) if total else 100.0,
            "avgMs": round(avg, 1) if avg else None,
            "etaSeconds": round(eta, 1) if eta is not None else None,
            "current": current,
            "errors": errors,
        }

    def list_jobs(self, workers: int = 1, limit: int = 50) -> list[dict]:
        with self._lock:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]
        return [self.progress(i, workers, recent_errors=2) for i in ids]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalProfile:
    workers: int
    cv_threads: int
    cooldown_s: float


def thermal_profile(name: str, cpus: int | None = None) -> ThermalProfile:
    """quiet: one photo at a time on half the cores, with a pause between photos (laptops,
    tethered shoots running next to the camera). balanced (default): leaves a core free for
    the editor. performance: every core, no pause (desktop with good cooling, overnight runs)."""
    n = max(1, cpus or os.cpu_count() or 2)
    if name == "quiet":
        return ThermalProfile(1, max(1, n // 2), 0.75)
    if name == "performance":
        workers = max(1, min(4, n // 4))
        return ThermalProfile(workers, max(1, n // workers), 0.0)
    if name == "balanced":
        workers = 1 if n < 8 else 2
        return ThermalProfile(workers, max(1, (n - 1) // workers), 0.1)
    raise ValueError(f"Unknown thermal profile {name!r}")


ProcessFn = Callable[[BatchItem, BatchJobSettings], None]


class BatchRunner:
    """Background workers that drain a BatchQueue. `process` renders one item (and raises on
    failure); it runs on a worker thread and must be thread-safe."""

    def __init__(self, queue: BatchQueue, process: ProcessFn, profile: str = "balanced"):
        self.queue = queue
        self.process = process
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._settings_cache: dict[str, BatchJobSettings] = {}
        self.set_profile(profile)

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def set_profile(self, name: str) -> None:
        prof = thermal_profile(name)
        with self._lock:
            self.profile_name, self.profile = name, prof
        cv2.setNumThreads(prof.cv_threads)
        if self.running:
            self._resize_pool()

    def start(self) -> None:
        with self._lock:
            self._stop.clear()
            self._resize_pool()
        self._wake.set()

    def notify(self) -> None:
        """New work queued: wake idle workers now instead of at their next poll."""
        if not self.running:
            self.start()
        self._wake.set()

    def stop(self, timeout: float | None = 30.0) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []

    def _resize_pool(self) -> None:
        self._threads = [t for t in self._threads if t.is_alive()]
        while len(self._threads) < self.profile.workers:
            t = threading.Thread(target=self._loop, args=(len(self._threads),), name=f"batch-{len(self._threads)}", daemon=True)
            self._threads.append(t)
            t.start()

    def _settings(self, job_id: str) -> BatchJobSettings:
        s = self._settings_cache.get(job_id)
        if s is None:
            s = self._settings_cache[job_id] = self.queue.settings(job_id)
        return s

    def _loop(self, index: int) -> None:
        while not self._stop.is_set():
            if index >= self.profile.workers:  # profile shrank: retire this worker
                return
            item = self.queue.claim()
            if item is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            if self.queue.job_status(item.job_id) not in JOB_ACTIVE:
                self.queue.release(item)
                continue
            t0 = time.perf_counter()
            try:
                self.process(item, self._settings(item.job_id))
            except Exception as exc:  # one bad file must not stop the batch
                log.warning("batch item %s failed: %s", item.source, exc)
                self.queue.fail(item, f"{type(exc).__name__}: {exc}")
            else:
                self.queue.complete(item, (time.perf_counter() - t0) * 1000.0)
            if self.profile.cooldown_s > 0:
                self._stop.wait(self.profile.cooldown_s)


def write_atomic(path: str | Path, data: bytes) -> None:
    """Write to a temporary file next to `path`, then rename: a crash never leaves a
    half-written JPEG where a finished one is expected."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, p)

