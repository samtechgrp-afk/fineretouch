#!/usr/bin/env python3
"""Build the FineRetouch desktop engine and stage everything Tauri bundles.

    python scripts/package_desktop.py              # models + engine for this machine
    python scripts/package_desktop.py --bundle     # ...then build the installer (npm run desktop:build)
    python scripts/package_desktop.py --models-only
    python scripts/package_desktop.py --target aarch64-apple-darwin   # name for a cross toolchain

What it does

1. Models. Makes sure services/ai/models holds the three model files (downloading the ones
   that are missing, the only step that needs the network) and copies them into
   src-tauri/resources/models, which tauri.conf.json bundles as the app resource "models/".
   At runtime the shell passes that folder to the engine, so the installed app is offline.

       face_landmarker.task      MediaPipe Face Landmarker (478-point mesh)    ~3.6 MB  required
       selfie_multiclass.tflite  MediaPipe selfie multiclass segmenter         ~16 MB   bokeh, hair
       midas_small.onnx          MiDaS v2.1 small monocular depth              ~67 MB   bokeh

2. Engine. Runs PyInstaller (one-file) over services/ai/desktop_entry.py, which pulls in
   app/main.py, retouch_engine.py, raw_engine.py, pipeline.py and FastAPI/uvicorn, plus the
   native parts of MediaPipe, OpenCV, rawpy (LibRaw) and exifread. The result is written to
   src-tauri/binaries/fineretouch-engine-<target-triple>[.exe], the name Tauri expects for
   the "binaries/fineretouch-engine" entry of bundle.externalBin.

3. Self-test. Runs the built binary with --self-test (loads the models, renders one frame)
   so a broken freeze fails the build here and not on a customer's machine.

PyInstaller cannot cross-compile: build each platform's engine on that platform (see
.github/workflows/desktop.yml). --target only changes the file name, for toolchains that
report a different triple from the one Tauri builds for.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AI_DIR = ROOT / "services" / "ai"
MODELS_SRC = AI_DIR / "models"
TAURI_DIR = ROOT / "src-tauri"
MODELS_DST = TAURI_DIR / "resources" / "models"
BIN_DIR = TAURI_DIR / "binaries"
ENGINE_NAME = "fineretouch-engine"
ENTRY = AI_DIR / "desktop_entry.py"


@dataclass(frozen=True)
class Model:
    name: str
    url: str
    min_bytes: int  # sanity check against truncated downloads and HTML error pages
    required: bool


MODELS = [
    Model(
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        3_000_000,
        True,
    ),
    Model(
        "selfie_multiclass.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        10_000_000,
        False,
    ),
    Model(
        "midas_small.onnx",
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        50_000_000,
        False,
    ),
]

# Imported by name at runtime (uvicorn picks its loop/protocol implementations by string),
# so PyInstaller's import graph cannot see them.
HIDDEN_IMPORTS = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "multipart",
    "python_multipart",
    "exifread",
    "rawpy",
]

# Packages whose data files and shared libraries must travel with them.
COLLECT_ALL = ["mediapipe"]
COLLECT_BINARIES = ["rawpy", "cv2"]
COLLECT_DATA = ["exifread"]

# Not used by the desktop engine; excluding them keeps the binary (and its start-up
# extraction) smaller. boto3 is only imported lazily by /v1/render-object, which the
# desktop app never calls (it renders from uploaded bytes, not object storage).
# MediaPipe's audio tasks import sounddevice optionally, and matplotlib (needed by its drawing
# utils) would otherwise drag in every GUI toolkit it has a backend for.
EXCLUDES = [
    "boto3",
    "botocore",
    "s3transfer",
    "sounddevice",
    "tkinter",
    "gi",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "pytest",
    "IPython",
    "notebook",
    "jupyter",
]


def log(msg: str) -> None:
    print(f"[package_desktop] {msg}", flush=True)


def host_target_triple() -> str:
    """The Rust host triple (what `tauri build` targets by default)."""
    try:
        out = subprocess.run(["rustc", "-vV"], check=True, capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    if sys.platform == "win32":
        return f"{arch}-pc-windows-msvc"
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin"
    return f"{arch}-unknown-linux-gnu"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(model: Model, dest: Path) -> None:
    log(f"downloading {model.name} from {model.url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            req = urllib.request.Request(model.url, headers={"User-Agent": "fineretouch-packager"})
            with urllib.request.urlopen(req, timeout=120) as res:
                shutil.copyfileobj(res, tmp, length=1 << 20)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    if tmp_path.stat().st_size < model.min_bytes:
        size = tmp_path.stat().st_size
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{model.name}: downloaded only {size} bytes, expected at least {model.min_bytes}")
    tmp_path.replace(dest)


def stage_models(skip_optional: bool, offline: bool) -> list[Path]:
    """Ensure models exist in services/ai/models and mirror them into the Tauri resources."""
    MODELS_DST.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for model in MODELS:
        if skip_optional and not model.required:
            (MODELS_DST / model.name).unlink(missing_ok=True)
            continue
        src = MODELS_SRC / model.name
        if not src.is_file() or src.stat().st_size < model.min_bytes:
            if offline:
                if model.required:
                    raise SystemExit(f"{src} is missing and --offline was given")
                log(f"skipping optional {model.name} (missing, --offline)")
                continue
            download(model, src)
        dst = MODELS_DST / model.name
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size or sha256(dst) != sha256(src):
            shutil.copy2(src, dst)
        staged.append(dst)
        log(f"model {model.name}: {dst.stat().st_size / 1e6:.1f} MB")
    total = sum(p.stat().st_size for p in staged) / 1e6
    log(f"{len(staged)} model(s), {total:.1f} MB staged in {MODELS_DST.relative_to(ROOT)}")
    return staged


def check_python_deps() -> None:
    missing = []
    for mod in ("PyInstaller", "fastapi", "uvicorn", "multipart", "numpy", "cv2", "mediapipe", "PIL", "rawpy", "exifread"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            "Missing Python packages: "
            + ", ".join(missing)
            + "\nInstall with: pip install -r services/ai/requirements.txt pyinstaller"
        )


def engine_filename(triple: str) -> str:
    return f"{ENGINE_NAME}-{triple}{'.exe' if 'windows' in triple else ''}"


def build_engine(triple: str, debug: bool) -> Path:
    check_python_deps()
    work = ROOT / "build" / "pyinstaller"
    dist = work / "dist"
    name = f"{ENGINE_NAME}-{triple}"
    sep = os.pathsep
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(ENTRY),
        "--name",
        name,
        "--onefile",
        "--noconfirm",
        "--clean",
        "--log-level",
        "WARN",
        "--distpath",
        str(dist),
        "--workpath",
        str(work / "build"),
        "--specpath",
        str(work),
        "--paths",
        str(AI_DIR),
        # Models live next to the app as a Tauri resource, not inside the binary: they stay
        # replaceable and the engine does not unpack 85 MB to a temp dir on every launch.
    ]
    # Console subsystem on purpose: Tauri spawns sidecars with CREATE_NO_WINDOW, so no window
    # appears, and a windowed (--noconsole) build would leave sys.stderr = None for logging.
    if not debug and sys.platform.startswith("linux"):
        # Stripping macOS libraries invalidates their code signatures, so Linux only.
        args.append("--strip")
    for mod in HIDDEN_IMPORTS:
        args += ["--hidden-import", mod]
    for pkg in COLLECT_ALL:
        args += ["--collect-all", pkg]
    for pkg in COLLECT_BINARIES:
        args += ["--collect-binaries", pkg]
    for pkg in COLLECT_DATA:
        args += ["--collect-data", pkg]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]
    # The engine's own package, so app/* is found even when imported indirectly.
    args += ["--add-data", f"{AI_DIR / 'app'}{sep}app"]
    log(f"running PyInstaller for {triple} (this takes a few minutes)")
    subprocess.run(args, check=True, cwd=AI_DIR)

    built = dist / (name + (".exe" if sys.platform == "win32" else ""))
    if not built.is_file():
        raise SystemExit(f"PyInstaller did not produce {built}")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    out = BIN_DIR / engine_filename(triple)
    shutil.copy2(built, out)
    if sys.platform != "win32":
        out.chmod(0o755)
    log(f"engine: {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


def self_test(binary: Path) -> None:
    log("self-test: loading models and rendering one frame with the frozen engine")
    env = {**os.environ, "FINERETOUCH_MODELS_DIR": str(MODELS_DST)}
    for var in ("FACE_LANDMARKER_MODEL", "SEGMENTER_MODEL", "MIDAS_MODEL"):
        env.pop(var, None)  # prove the bundled copies are what gets loaded
    res = subprocess.run([str(binary), "--self-test"], env=env, capture_output=True, text=True, timeout=600)
    report = next((line for line in res.stdout.splitlines() if line.startswith("version=")), "")
    if res.returncode != 0 or not report:
        sys.stderr.write(res.stdout[-4000:] + res.stderr[-4000:])
        raise SystemExit(f"self-test failed (exit {res.returncode})")
    log(f"self-test ok: {report}")
    if "faceBackend=mediapipe" not in report:
        log("warning: MediaPipe face model did not load; face tools will be disabled in this build")


def bundle() -> None:
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    log("building the installer: npm run desktop:build")
    subprocess.run([npm, "run", "desktop:build"], check=True, cwd=ROOT)
    log(f"installers are in {(TAURI_DIR / 'target' / 'release' / 'bundle').relative_to(ROOT)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", help="Rust target triple for the sidecar file name (default: rustc host)")
    p.add_argument("--models-only", action="store_true", help="only stage the models")
    p.add_argument("--skip-models", action="store_true", help="do not touch the models")
    p.add_argument("--skip-optional-models", action="store_true", help="bundle only the face model (no bokeh depth)")
    p.add_argument("--offline", action="store_true", help="never download; fail if a required model is missing")
    p.add_argument("--no-self-test", action="store_true")
    p.add_argument("--debug", action="store_true", help="keep symbols in the engine binary")
    p.add_argument("--bundle", action="store_true", help="run `npm run desktop:build` afterwards")
    args = p.parse_args()

    triple = args.target or host_target_triple()
    log(f"target {triple}")
    if not args.skip_models:
        stage_models(args.skip_optional_models, args.offline)
    if args.models_only:
        return 0
    binary = build_engine(triple, args.debug)
    if not args.no_self_test:
        self_test(binary)
    if args.bundle:
        bundle()
    return 0


if __name__ == "__main__":
    sys.exit(main())
