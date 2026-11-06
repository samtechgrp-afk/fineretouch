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

Every image endpoint accepts camera RAW (CR2, CR3, NEF, ARW, RAF, DNG, ORF, RW2, PEF);
renders demosaic them at 16 bits per channel through raw_engine.

Heavy endpoints are plain `def` so FastAPI runs them in its thread pool; NumPy and
OpenCV release the GIL, so one process uses several cores. Scale out with more
replicas behind the gateway (see docker-compose.yml).
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import imageio, raw_engine, storage
from .color import color_stats
from .depth import DepthEstimator
from .landmarks import FaceLandmarkService
from .pipeline import render, sharpness
from .retouch_engine import style_profile, sync_style_arrays
from .schemas import (
    Adjustments,
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


def _render_bytes(data: bytes, adj: Adjustments, out: OutputSpec) -> tuple[bytes, str, CullFlags]:
    try:
        hq = imageio.decode_hq(data, out.maxLongEdge)  # uint16 for camera RAW
    except Exception as exc:
        raise HTTPException(415, f"Unsupported or corrupt image: {exc}") from exc
    rgb = raw_engine.to_uint8(hq)
    # Landmarks are resolution independent; detecting on the full image would only cost time.
    faces = faces_service.detect(rgb)
    depth = depth_service.estimate(rgb, faces) if adj.depthBokeh.enabled else None
    hair = None
    if adj.strayHairRemoval.enabled and faces:
        seg = depth_service.segment(rgb)
        hair = seg[1] if seg is not None else None
    result = render(hq, adj, faces, depth=depth, hair_prob=hair)
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
