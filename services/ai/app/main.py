"""FineRetouch AI inference service.

Endpoints
  GET  /health                 liveness + which face backend is active
  POST /v1/analyze             landmarks, face regions, culling flags, colour stats
  POST /v1/color-stats         L*a*b* statistics of a reference image (for colour match)
  POST /v1/render              full-resolution render of an uploaded image (single export)
  POST /v1/render-object       render from object storage to object storage (batch worker)

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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import imageio, storage
from .color import color_stats
from .landmarks import FaceLandmarkService
from .pipeline import render, sharpness
from .schemas import (
    Adjustments,
    ColorStats,
    CullFlags,
    FaceAnalysis,
    ImageAnalysis,
    OutputSpec,
    RenderObjectRequest,
    RenderObjectResponse,
)

log = logging.getLogger("fineretouch.ai")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "150")) * 1024 * 1024
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
    return {"status": "ok", "faceBackend": faces_service.backend}


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
    rgb = _decode(data, out.maxLongEdge)
    # Landmarks are resolution independent; detecting on the full image would only cost time.
    faces = faces_service.detect(rgb)
    result = render(rgb, adj, faces)
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
