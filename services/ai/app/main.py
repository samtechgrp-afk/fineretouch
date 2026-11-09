"""FineRetouch AI inference service.

Endpoints
  GET  /health                 liveness + which face backend is active
  POST /v1/analyze             landmarks, face regions, culling flags, colour stats
  POST /v1/color-stats         L*a*b* statistics of a reference image (for colour match)
  POST /v1/render              full-resolution render of an uploaded image (single export)
  POST /v1/render-object       render from object storage to object storage (batch worker)
  POST /v1/depth               relative depth map (8-bit PNG, 0 = subject, 255 = far) for the bokeh preview
  POST /v1/style/profile       style metrics of a reference photo
  POST /v1/style/sync          reference + N targets -> per-target adjustment patches ("Sync to All")
  POST /v1/raw/metadata        camera RAW format + EXIF (camera, lens, ISO, shutter, aperture, orientation)
  POST /v1/raw/preview         camera RAW -> JPEG (embedded preview, or a fast half-size demosaic)
  POST /v1/parse               face skin / body skin / hair masks (PNG, R face skin, G body skin, B hair)
  POST /v1/segment             person / hair / clothes masks (PNG, R clothes, G hair, B person) for the preview
  POST /v1/lut/fit             before + after of a graded reference -> 3D look LUT (+ .cube text)
  POST /v1/lut/parse           .cube file -> look LUT
  /v1/batch/...                local SQLite batch queue (desktop engine only, see LOCAL_BATCH)

Every image endpoint accepts camera RAW (CR2, CR3, NEF, ARW, RAF, DNG, ORF, RW2, PEF);
renders demosaic them at 16 bits per channel through raw_engine.

Heavy endpoints are plain `def` so FastAPI runs them in its thread pool; NumPy and
OpenCV release the GIL, so one process uses several cores. Scale out with more
replicas behind the gateway (see docker-compose.yml).
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import batch_engine, imageio, raw_engine, storage
from .color import color_stats
from .depth import DepthEstimator
from .landmarks import FaceLandmarkService
from .pipeline import render, sharpness
from .retouch_engine import style_profile, sync_style_arrays
from .schemas import (
    Adjustments,
    BatchJobRequest,
    BatchJobSettings,
    ColorStats,
    CullFlags,
    FaceAnalysis,
    ImageAnalysis,
    OutputSpec,
    RenderObjectRequest,
    RenderObjectResponse,
    StyleProfile,
    StyleSyncResponse,
)

log = logging.getLogger("fineretouch.ai")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "400")) * 1024 * 1024
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "40"))
EYES_CLOSED_THRESHOLD = float(os.environ.get("EYES_CLOSED_THRESHOLD", "0.2"))
ANALYZE_LONG_EDGE = 2048

app = FastAPI(title="FineRetouch AI", version="0.1.0")
# Desktop builds (services/ai/app/desktop.py) bind to loopback and set a per-launch token so
# other local programs, or web pages open in a browser, cannot drive the engine. Server
# deployments leave it unset and rely on the gateway instead.
ENGINE_TOKEN = os.environ.get("FINERETOUCH_ENGINE_TOKEN") or None


@app.middleware("http")
async def require_engine_token(request: Request, call_next):
    if ENGINE_TOKEN and request.method != "OPTIONS":
        supplied = request.headers.get("x-fineretouch-token", "")
        if not hmac.compare_digest(supplied, ENGINE_TOKEN):
            return JSONResponse({"detail": "Missing or invalid engine token"}, status_code=401)
    return await call_next(request)


# Added last so it is the outermost layer: even a 401 carries CORS headers the webview can read.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

faces_service = FaceLandmarkService()
depth_service = DepthEstimator()
MAX_SYNC_TARGETS = int(os.environ.get("MAX_SYNC_TARGETS", "200"))


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    return data


def _decode(data: bytes, max_long_edge: int | None = None) -> np.ndarray:
    try:
        return imageio.decode(data, max_long_edge)
    except Exception as exc:  # Pillow raises many types for bad input
        raise HTTPException(415, f"Unsupported or corrupt image: {exc}") from exc


def cull_flags(faces: list[FaceAnalysis], sharp: float) -> CullFlags:
    return CullFlags(
        blurry=sharp < BLUR_THRESHOLD,
        eyesClosed=any(f.landmarkCount > 0 and f.eyeOpenness < EYES_CLOSED_THRESHOLD for f in faces),
        noFace=len(faces) == 0,
    )


def analyze_array(rgb: np.ndarray, width: int, height: int) -> ImageAnalysis:
    faces = faces_service.detect(rgb)
    sharp = sharpness(rgb)
    return ImageAnalysis(
        width=width,
        height=height,
        faces=faces,
        sharpness=sharp,
        colorStats=color_stats(rgb.astype(np.float32) / 255.0),
        cull=cull_flags(faces, sharp),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "faceBackend": faces_service.backend, "depthBackend": depth_service.backend}


@app.post("/v1/analyze", response_model=ImageAnalysis)
async def analyze(file: UploadFile = File(...)) -> ImageAnalysis:
    data = await _read_upload(file)
    return _analyze_bytes(data)


def _analyze_bytes(data: bytes) -> ImageAnalysis:
    try:
        w, h = imageio.oriented_size(data)
    except Exception as exc:
        raise HTTPException(415, f"Unsupported or corrupt image: {exc}") from exc
    return analyze_array(_decode(data, ANALYZE_LONG_EDGE), w, h)


@app.post("/v1/color-stats", response_model=ColorStats)
async def reference_stats(file: UploadFile = File(...)) -> ColorStats:
    data = await _read_upload(file)
    rgb = _decode(data, 1024)
    return color_stats(rgb.astype(np.float32) / 255.0)


def _parse_adjustments(raw: str) -> Adjustments:
    try:
        return Adjustments.model_validate(json.loads(raw or "{}"))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(422, f"Invalid adjustments: {exc}") from exc


def _needs_skin_parse(adj: Adjustments) -> bool:
    """Face parsing refines the skin mask; skip it when no skin tool is on or the user turned it off."""
    a = adj
    skin_tools = (
        a.skinSmoothing > 0 or a.blemishRemoval > 0 or a.acneRemoval > 0 or a.darkSpotRemoval > 0
        or a.skinToneUnify > 0 or a.skinMultiplier > 1 or a.sharpen > 0
    )
    return skin_tools and a.skinMask.useSegmentation


def _render_bytes(
    data: bytes, adj: Adjustments | Callable[[np.ndarray], Adjustments], out: OutputSpec
) -> tuple[bytes, str, CullFlags]:
    """adj may be a function of a small proxy of the decoded photo (batch smart sync)."""
    try:
        hq = imageio.decode_hq(data, out.maxLongEdge)  # uint16 for camera RAW
    except Exception as exc:
        raise HTTPException(415, f"Unsupported or corrupt image: {exc}") from exc
    rgb = raw_engine.to_uint8(hq)
    if callable(adj):
        adj = adj(raw_engine.resize_long_edge(rgb, 1024))
    # Landmarks are resolution independent; detecting on the full image would only cost time.
    faces = faces_service.detect(rgb)
    depth = depth_service.estimate(rgb, faces) if adj.depthBokeh.enabled else None
    hair = clothes = None
    want_clothes = adj.clothing.enabled
    if (adj.strayHairRemoval.enabled and faces) or want_clothes:
        seg = depth_service.segment(rgb)
        if seg is not None:
            hair, clothes = seg[1], seg[2]
    parse = depth_service.parse(rgb) if faces and _needs_skin_parse(adj) else None
    result = render(hq, adj, faces, depth=depth, hair_prob=hair, clothes_prob=clothes if want_clothes else None, parse=parse)
    encoded, mime = imageio.encode(result, out.format, out.quality)
    return encoded, mime, cull_flags(faces, sharpness(rgb))


@app.post("/v1/render")
async def render_upload(
    file: UploadFile = File(...),
    adjustments: str = Form("{}"),
    format: str = Form("jpeg"),
    quality: int = Form(92),
    maxLongEdge: int | None = Form(None),
) -> Response:
    data = await _read_upload(file)
    adj = _parse_adjustments(adjustments)
    try:
        out = OutputSpec(format=format, quality=quality, maxLongEdge=maxLongEdge)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    t0 = time.perf_counter()
    encoded, mime, _ = await run_in_threadpool(_render_bytes, data, adj, out)
    return Response(encoded, media_type=mime, headers={"X-Render-Ms": f"{(time.perf_counter() - t0) * 1000:.0f}"})


@app.post("/v1/render-object", response_model=RenderObjectResponse)
def render_object(req: RenderObjectRequest) -> RenderObjectResponse:
    t0 = time.perf_counter()
    try:
        data = storage.get_bytes(req.sourceKey)
    except Exception as exc:
        raise HTTPException(404, f"Could not read {req.sourceKey}: {exc}") from exc
    encoded, mime, cull = _render_bytes(data, req.adjustments, req.output)
    storage.put_bytes(req.outputKey, encoded, mime)
    ms = (time.perf_counter() - t0) * 1000
    log.info("rendered %s -> %s in %.0f ms", req.sourceKey, req.outputKey, ms)
    return RenderObjectResponse(outputKey=req.outputKey, bytes=len(encoded), ms=ms, cull=cull)


@app.post("/v1/depth")
async def depth_map(file: UploadFile = File(...)) -> Response:
    """Depth for the live preview, computed on the same pixels the editor shows."""
    data = await _read_upload(file)

    def work() -> bytes:
        rgb = _decode(data, ANALYZE_LONG_EDGE)
        depth = depth_service.estimate(rgb, faces_service.detect(rgb))
        ok, png = cv2.imencode(".png", (np.clip(depth, 0, 1) * 255 + 0.5).astype(np.uint8))
        if not ok:
            raise HTTPException(500, "Could not encode depth map")
        return png.tobytes()

    png = await run_in_threadpool(work)
    return Response(png, media_type="image/png", headers={"X-Depth-Backend": depth_service.backend})


@app.post("/v1/style/profile", response_model=StyleProfile)
async def style_profile_endpoint(file: UploadFile = File(...)) -> StyleProfile:
    data = await _read_upload(file)
    return await run_in_threadpool(lambda: style_profile(_decode(data, 1024)))


@app.post("/v1/style/sync", response_model=StyleSyncResponse)
async def style_sync(
    reference: UploadFile = File(...),
    targets: list[UploadFile] = File(...),
    mode: str = Form("palette"),
    amount: float = Form(100),
) -> StyleSyncResponse:
    if mode not in ("palette", "sliders"):
        raise HTTPException(422, "mode must be 'palette' or 'sliders'")
    if len(targets) > MAX_SYNC_TARGETS:
        raise HTTPException(413, f"At most {MAX_SYNC_TARGETS} targets per request")
    ref_bytes = await _read_upload(reference)
    target_bytes = [(t.filename or f"target-{i}", await _read_upload(t)) for i, t in enumerate(targets)]

    def work() -> StyleSyncResponse:
        ref = _decode(ref_bytes, 1024)
        return sync_style_arrays(ref, [(n, _decode(b, 1024)) for n, b in target_bytes], mode, amount)

    return await run_in_threadpool(work)


def _require_raw(data: bytes, filename: str | None) -> raw_engine.RawFormat:
    fmt = raw_engine.detect_raw_format(data, filename)
    if fmt is None:
        raise HTTPException(415, "Not a supported camera RAW file (CR2, CR3, NEF, ARW, RAF, DNG, ORF, RW2, PEF)")
    return fmt


@app.post("/v1/raw/metadata")
async def raw_metadata(file: UploadFile = File(...)) -> dict:
    data = await _read_upload(file)
    _require_raw(data, file.filename)

    def work() -> dict:
        meta = raw_engine.read_raw_metadata(data, file.filename)
        try:
            meta.width, meta.height = raw_engine.oriented_size(data)
        except raw_engine.RawDecodeError:
            pass
        out = {k: v for k, v in vars(meta).items() if v is not None}
        out["shutter"] = meta.shutter
        return out

    return await run_in_threadpool(work)


@app.post("/v1/raw/preview")
async def raw_preview(
    file: UploadFile = File(...),
    maxLongEdge: int = Form(2560),
    mode: str = Form("embedded"),
    quality: int = Form(88),
) -> Response:
    """mode=embedded: the camera's own JPEG (instant). mode=demosaic: LibRaw at half size,
    camera white balance (what the export will look like before any edit)."""
    data = await _read_upload(file)
    _require_raw(data, file.filename)
    if mode not in ("embedded", "demosaic"):
        raise HTTPException(422, "mode must be 'embedded' or 'demosaic'")

    def work() -> tuple[bytes, str]:
        source = "embedded_preview"
        rgb = None
        if mode == "embedded" and (prev := raw_engine.extract_embedded_preview(data, file.filename)) is not None:
            rgb = raw_engine.resize_long_edge(prev[0], maxLongEdge)
        if rgb is None:
            res = raw_engine.decode_camera_raw(data, output_bps=8, max_long_edge=maxLongEdge, filename=file.filename)
            if not res.ok:
                raise HTTPException(415, f"Unreadable RAW file: {res.error}")
            rgb, source = res.image, res.source
        encoded, _ = imageio.encode(np.ascontiguousarray(rgb), "jpeg", max(1, min(100, quality)))
        return encoded, source

    encoded, source = await run_in_threadpool(work)
    return Response(encoded, media_type="image/jpeg", headers={"X-Raw-Source": source})


@app.post("/v1/parse")
async def parse_map(file: UploadFile = File(...)) -> Response:
    """Face parsing for the live preview's skin mask: PNG with R = face skin, G = body skin,
    B = hair (0..255) on the same pixels the editor shows. 404 when the segmenter model is not
    installed (the editor then keeps the landmark-only skin mask)."""
    data = await _read_upload(file)

    def work() -> bytes:
        rgb = _decode(data, ANALYZE_LONG_EDGE)
        p = depth_service.parse(rgb)
        if p is None:
            raise HTTPException(404, "Segmentation model not available")
        planes = np.stack([p.hair, p.body_skin, p.face_skin], axis=-1)  # BGR order for OpenCV
        ok, png = cv2.imencode(".png", (np.clip(planes, 0, 1) * 255 + 0.5).astype(np.uint8))
        if not ok:
            raise HTTPException(500, "Could not encode face parsing")
        return png.tobytes()

    return Response(await run_in_threadpool(work), media_type="image/png")


@app.post("/v1/segment")
async def segment_map(file: UploadFile = File(...)) -> Response:
    """Segmentation for the live preview: PNG with R = clothes, G = hair, B = person (0..255),
    on the same pixels the editor shows. 404 when the segmenter model is not installed."""
    data = await _read_upload(file)

    def work() -> bytes:
        rgb = _decode(data, ANALYZE_LONG_EDGE)
        seg = depth_service.segment(rgb)
        if seg is None:
            raise HTTPException(404, "Segmentation model not available")
        person, hair, clothes = seg
        planes = np.stack([person, hair, clothes], axis=-1)  # BGR order for OpenCV
        ok, png = cv2.imencode(".png", (np.clip(planes, 0, 1) * 255 + 0.5).astype(np.uint8))
        if not ok:
            raise HTTPException(500, "Could not encode segmentation")
        return png.tobytes()

    return Response(await run_in_threadpool(work), media_type="image/png")


def _lut_response(lut: np.ndarray, name: str) -> dict:
    look = batch_engine.look_from_lut(lut, name)
    return {**look.model_dump(), "cube": batch_engine.lut_to_cube(lut, name)}


@app.post("/v1/lut/fit")
async def lut_fit(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    size: int = Form(17),
    name: str = Form("Reference look"),
) -> dict:
    """Fit the reference's grade as a 3D LUT from its original and its graded version."""
    if not 2 <= size <= 33:
        raise HTTPException(422, "size must be 2..33")
    b_bytes, a_bytes = await _read_upload(before), await _read_upload(after)

    def work() -> dict:
        return _lut_response(batch_engine.fit_lut(_decode(b_bytes, 1024), _decode(a_bytes, 1024), size), name)

    return await run_in_threadpool(work)


@app.post("/v1/lut/parse")
async def lut_parse(file: UploadFile = File(...)) -> dict:
    data = await _read_upload(file)
    try:
        lut, title = batch_engine.parse_cube(data.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _lut_response(lut, title)


# --- local batch queue ------------------------------------------------------------------
#
# The queue reads and writes folders on this machine, so it only runs where the engine is
# the user's own local process: the desktop sidecar (FINERETOUCH_DATA_DIR set, token
# required) or a developer who opts in with FINERETOUCH_LOCAL_BATCH=1. Server deployments
# use services/worker (BullMQ + object storage) instead.

LOCAL_BATCH = os.environ.get("FINERETOUCH_LOCAL_BATCH", "1" if os.environ.get("FINERETOUCH_DATA_DIR") else "0") == "1"
_batch_lock = threading.Lock()
_batch: tuple[batch_engine.BatchQueue, batch_engine.BatchRunner] | None = None


def _batch_db_path() -> str:
    explicit = os.environ.get("FINERETOUCH_BATCH_DB")
    if explicit:
        return explicit
    base = os.environ.get("FINERETOUCH_DATA_DIR") or str(Path.home() / ".fineretouch")
    return str(Path(base) / "batch.sqlite3")


def _process_batch_item(item: batch_engine.BatchItem, settings: BatchJobSettings) -> None:
    data = Path(item.source).read_bytes()
    encoded, _, _ = _render_bytes(data, lambda proxy: batch_engine.plan_adjustments(settings, proxy), settings.output)
    batch_engine.write_atomic(item.output, encoded)


def batch_service() -> tuple[batch_engine.BatchQueue, batch_engine.BatchRunner]:
    global _batch
    if not LOCAL_BATCH:
        raise HTTPException(404, "The local batch queue is only available in the desktop engine")
    with _batch_lock:
        if _batch is None:
            queue = batch_engine.BatchQueue(_batch_db_path())
            recovered = queue.recover()
            if recovered:
                log.info("batch queue: %d interrupted item(s) back to pending", recovered)
            runner = batch_engine.BatchRunner(queue, _process_batch_item, os.environ.get("FINERETOUCH_BATCH_PROFILE", "balanced"))
            runner.start()
            _batch = (queue, runner)
        return _batch


@app.post("/v1/batch/jobs")
def batch_create(req: BatchJobRequest) -> dict:
    queue, runner = batch_service()
    try:
        files = batch_engine.discover_images(req.sources, req.recursive)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not files:
        raise HTTPException(422, "No photos (JPEG, PNG, TIFF, WebP or camera RAW) found in the selected folders")
    out_dir = Path(req.outputDir).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(422, f"Cannot create output folder {out_dir}: {exc}") from exc
    if req.profile:
        runner.set_profile(req.profile)
    job_id = queue.create_job(req.name, files, out_dir, req.settings)
    runner.notify()
    return queue.progress(job_id, runner.profile.workers)


@app.get("/v1/batch/jobs")
def batch_list() -> dict:
    queue, runner = batch_service()
    return {"profile": runner.profile_name, "workers": runner.profile.workers, "jobs": queue.list_jobs(runner.profile.workers)}


@app.get("/v1/batch/jobs/{job_id}")
def batch_get(job_id: str) -> dict:
    queue, runner = batch_service()
    try:
        return queue.progress(job_id, runner.profile.workers)
    except KeyError as exc:
        raise HTTPException(404, "No such batch") from exc


@app.post("/v1/batch/jobs/{job_id}/{action}")
def batch_action(job_id: str, action: str) -> dict:
    queue, runner = batch_service()
    if action not in ("pause", "resume", "cancel", "retry"):
        raise HTTPException(422, "action must be pause, resume, cancel or retry")
    try:
        queue.set_status(job_id, action)
    except KeyError as exc:
        raise HTTPException(404, "No such batch") from exc
    runner.notify()
    return queue.progress(job_id, runner.profile.workers)


@app.delete("/v1/batch/jobs/{job_id}")
def batch_delete(job_id: str) -> dict:
    queue, _ = batch_service()
    if queue.job_status(job_id) is None:
        raise HTTPException(404, "No such batch")
    queue.set_status(job_id, "cancel")
    queue.delete_job(job_id)
    return {"deleted": job_id}


@app.post("/v1/batch/profile")
def batch_profile(profile: str = Form(...)) -> dict:
    _, runner = batch_service()
    try:
        runner.set_profile(profile)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"profile": runner.profile_name, "workers": runner.profile.workers, "cvThreads": runner.profile.cv_threads}
