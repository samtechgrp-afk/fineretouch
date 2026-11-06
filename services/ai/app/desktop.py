"""Entry point of the FineRetouch desktop engine (the Tauri sidecar).

The desktop shell (src-tauri/src/engine.rs) starts this as ``fineretouch-engine`` with

    fineretouch-engine --port 51234 --parent-pid 4242

and these environment variables:

    FINERETOUCH_ENGINE_TOKEN   per-launch secret every request must carry (see main.py)
    FINERETOUCH_MODELS_DIR     bundled models (face_landmarker.task, selfie_multiclass.tflite,
                               midas_small.onnx); the per-model variables below win if set
    FINERETOUCH_DATA_DIR       %APPDATA%/FineRetouch (or the platform equivalent) for logs/cache
    CORS_ORIGINS               the webview origins (tauri://localhost, http://tauri.localhost)

Everything runs locally: the service binds to 127.0.0.1 only and never needs the network.
The process exits on its own when the desktop shell that started it goes away, so a crash
of the shell cannot leave an orphaned engine holding a port.

Extra flags for the packaging script and support:

    --version     print the engine version and exit
    --self-test   import every native dependency, load the models, run one render, exit 0/1
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

ENGINE_VERSION = "0.4.0"

_MODEL_FILES = {
    "FACE_LANDMARKER_MODEL": "face_landmarker.task",
    "SEGMENTER_MODEL": "selfie_multiclass.tflite",
    "MIDAS_MODEL": "midas_small.onnx",
}


def _bundle_dir() -> Path:
    """Directory holding bundled data: PyInstaller's extraction dir when frozen."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def configure_environment(models_dir: str | None, data_dir: str | None) -> None:
    """Point the model loaders at the bundled weights before ``app.main`` is imported.

    landmarks.py and depth.py read their paths at import time, so this must run first.
    """
    candidates = [models_dir, os.environ.get("FINERETOUCH_MODELS_DIR"), str(_bundle_dir() / "models")]
    for d in candidates:
        if d and Path(d).is_dir():
            for var, name in _MODEL_FILES.items():
                if not os.environ.get(var) and (Path(d) / name).is_file():
                    os.environ[var] = str(Path(d) / name)
            break
    if data_dir:
        os.environ["FINERETOUCH_DATA_DIR"] = data_dir
    # Desktop machines run one engine process; keep thread pools from oversubscribing the CPU
    # the webview's WebGL preview also needs.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1)))


def _log_file(data_dir: str | None) -> Path | None:
    base = data_dir or os.environ.get("FINERETOUCH_DATA_DIR")
    if not base:
        return None
    logs = Path(base) / "logs"
    try:
        logs.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return logs / "engine.log"


def _parent_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            WAIT_TIMEOUT = 0x102
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Not os.getppid(): in a one-file build the direct parent is PyInstaller's bootloader,
    # not the desktop shell, so the shell's own pid is what has to be watched.
    return True


def watch_parent(pid: int, interval: float = 2.0) -> None:
    def loop() -> None:
        while True:
            time.sleep(interval)
            if not _parent_alive(pid):
                logging.getLogger("fineretouch.desktop").warning("desktop shell %d exited; stopping engine", pid)
                os._exit(0)

    threading.Thread(target=loop, name="parent-watchdog", daemon=True).start()


def self_test() -> int:
    """Smoke test of the frozen binary: native libraries, models and one full render."""
    import numpy as np

    from . import main, raw_engine
    from .pipeline import render
    from .schemas import Adjustments

    report = {
        "version": ENGINE_VERSION,
        "faceBackend": main.faces_service.backend,
        "depthBackend": main.depth_service.backend,
        "rawpy": raw_engine.rawpy is not None,
        "exifread": raw_engine.exifread is not None,
    }
    rgb = (np.random.default_rng(0).random((256, 384, 3)) * 255).astype(np.uint8)
    faces = main.faces_service.detect(rgb)
    out = render(rgb.astype(np.uint16) * 257, Adjustments(), faces)
    report["render"] = list(out.shape)
    print(" ".join(f"{k}={v}" for k, v in report.items()), flush=True)
    return 0 if report["rawpy"] and out.shape[:2] == rgb.shape[:2] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fineretouch-engine", description="FineRetouch local AI engine")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (loopback only by default)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("FINERETOUCH_ENGINE_PORT", "8765")))
    parser.add_argument("--parent-pid", type=int, default=None, help="exit when this process exits")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        print(f"fineretouch-engine {ENGINE_VERSION}")
        return 0

    configure_environment(args.models_dir, args.data_dir)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if (log_path := _log_file(args.data_dir)) is not None:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=args.log_level.upper(), handlers=handlers, force=True)

    if args.self_test:
        return self_test()

    if args.parent_pid:
        watch_parent(args.parent_pid)

    import uvicorn

    from .main import app

    logging.getLogger("fineretouch.desktop").info(
        "engine %s listening on http://%s:%d (models: %s)",
        ENGINE_VERSION,
        args.host,
        args.port,
        os.environ.get("FACE_LANDMARKER_MODEL", "not bundled"),
    )
    # One process: the app object is passed directly (no import string), which is what a
    # frozen executable needs, and heavy endpoints already fan out over the thread pool.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(), access_log=False, workers=1)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
