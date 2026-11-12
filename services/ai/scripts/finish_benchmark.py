"""Professional Skin Finish performance on this computer: preview, export, stages, peak memory.

    cd services/ai
    python -m scripts.finish_benchmark D:/photos/IMG_0012.jpg D:/photos/IMG_0040.jpg --json finish-perf.json
    python -m scripts.finish_benchmark PHOTO --styles professional --repeats 1 --profile prof.txt
    python -m scripts.finish_benchmark PHOTO --settings edit.json      # your own Adjustments JSON

It calls the same functions as the running app: the preview is the /v1/perfection scan (full
resolution, returned at 2048 px), the export is a full-resolution JPEG render. Each photo is
measured with the finish off (the baseline, as before v0.14) and with each style on. Every
measurement runs in a fresh child process, so peak memory is that render's own peak and one
render's leftovers never count against the next. Nothing is downscaled or skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = [  # (label, module, attribute): wrapped with a timer in the child process
    ("decode", "app.imageio", "decode_hq"),
    ("faces", "app.landmarks", "FaceLandmarkService.detect"),
    ("parse", "app.depth", "DepthEstimator.parse"),
    ("skinPerfection", "app.pipeline", "perfect_skin"),
    ("finish (all)", "app.pipeline", "finish_skin"),
    ("finish: local corrections", "app.engines.finish", "apply_local"),
    ("  dark spots", "app.engines.finish_local", "correct_spots"),
    ("  tone evenness", "app.engines.finish_local", "even_tone"),
    ("  shine", "app.engines.finish_local", "reduce_shine"),
    ("  under-eye", "app.engines.finish_local", "refine_under_eye"),
    ("  dodge and burn", "app.engines.finish_local", "dodge_burn"),
    ("  measures", "app.engines.finish_local", "tone_rms"),
    ("dark circles", "app.pipeline", "correct_dark_circles"),
    ("encode", "app.imageio", "encode"),
]


def _wrap_stages(times: dict) -> None:
    import importlib

    for label, mod, attr in STAGES:
        m = importlib.import_module(mod)
        owner, name = m, attr
        if "." in attr:
            cls, name = attr.split(".")
            owner = getattr(m, cls)
        fn = getattr(owner, name, None)
        if fn is None:
            continue

        def timed(*a, __fn=fn, __label=label, **k):
            t0 = time.perf_counter()
            try:
                return __fn(*a, **k)
            finally:
                times[__label] = times.get(__label, 0.0) + (time.perf_counter() - t0) * 1000

        setattr(owner, name, timed)


def child(spec: dict) -> dict:
    """One measurement (runs in its own process)."""
    times: dict[str, float] = {}
    _wrap_stages(times)
    from app import main
    from app.schemas import Adjustments, OutputSpec, SkinFinishSettings
    from qa.memory_probe import peak_memory

    data = Path(spec["photo"]).read_bytes()
    base = json.loads(Path(spec["settings"]).read_text()) if spec.get("settings") else {}
    adj = Adjustments.model_validate(base)
    if spec["style"] == "off":
        adj = adj.model_copy(update={"skinFinish": SkinFinishSettings()})
    else:
        adj = adj.model_copy(update={"skinFinish": SkinFinishSettings.styled(spec["style"])})
    # Warm-up: load the models once (as the running app has them loaded) on a small image.
    import numpy as np
    from app import imageio
    small = np.full((256, 256, 3), 128, np.uint8)
    main.faces_service.detect(small)
    if adj.skinMask.useSegmentation:
        main.depth_service.parse(small)
    times.clear()
    prof = None
    if spec.get("profile"):
        import cProfile
        prof = cProfile.Profile()
    with peak_memory() as mem:
        t0 = time.perf_counter()
        if prof:
            prof.enable()
        if spec["kind"] == "preview":
            res = main._perfection_admitted(data, adj, 2048)
            extra = {"finish": res.get("finish"), "faces": res.get("faces")}
        else:
            enc, _, _ = main._render_admitted(data, adj, OutputSpec(format="jpeg", quality=95))
            extra = {"bytes": len(enc)}
        if prof:
            prof.disable()
        ms = (time.perf_counter() - t0) * 1000
    if prof:
        import io
        import pstats
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(40)
        Path(spec["profile"]).write_text(s.getvalue())
    w, h = imageio.oriented_size(data)
    return {"ms": round(ms), "peakMB": mem.peak_mb, "addedMB": mem.added_mb, "memMethod": mem.method,
            "megapixels": round(w * h / 1e6, 1), "stagesMs": {k: round(v) for k, v in times.items()}, **extra}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="*")
    ap.add_argument("--styles", default="off,natural,professional,advanced")
    ap.add_argument("--kinds", default="preview,export")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--settings", help="Adjustments JSON used as the edit (the finish is set per style)")
    ap.add_argument("--profile", help="write a cProfile of the first finished export to this file")
    ap.add_argument("--json")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        spec = json.loads(a.child)
        print("RESULT " + json.dumps(child(spec)), flush=True)
        return
    if not a.photos:
        ap.error("give at least one photo")
    info = {"machine": platform.processor() or platform.machine(), "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "cpus": os.cpu_count()}
    try:
        import psutil
        info["ramGB"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    print(json.dumps(info))
    rows, profiled = [], False
    for photo in a.photos:
        for style in a.styles.split(","):
            for kind in a.kinds.split(","):
                for r in range(a.repeats):
                    spec = {"photo": str(Path(photo).resolve()), "style": style, "kind": kind, "settings": a.settings}
                    if a.profile and not profiled and style != "off" and kind == "export":
                        spec["profile"], profiled = a.profile, True
                    p = subprocess.run([sys.executable, "-m", "scripts.finish_benchmark", "--child", json.dumps(spec)],
                                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
                    if not line:
                        print(f"FAILED {Path(photo).name} {style} {kind}:\n{p.stderr[-3000:]}")
                        continue
                    res = {"photo": Path(photo).name, "style": style, "kind": kind, "run": r + 1, **json.loads(line[7:])}
                    rows.append(res)
                    st = ", ".join(f"{k} {v / 1000:.1f}s" for k, v in res["stagesMs"].items() if v >= 200)
                    print(f"{res['photo']} {res['megapixels']} MP  {style:12s} {kind:7s} {res['ms'] / 1000:6.1f} s  "
                          f"peak {res['peakMB']} MB (+{res['addedMB']})  [{st}]", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"machine": info, "rows": rows}, indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
