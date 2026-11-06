"""Decoding/encoding helpers. EXIF orientation is applied on decode."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000  # allow very large studio files, still bounded

_FORMATS = {"jpeg": ("JPEG", "image/jpeg"), "png": ("PNG", "image/png"), "webp": ("WEBP", "image/webp")}


def decode(data: bytes, max_long_edge: int | None = None) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_long_edge and max(im.size) > max_long_edge:
            im.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8).copy()


def oriented_size(data: bytes) -> tuple[int, int]:
    """(width, height) after EXIF rotation, without decoding pixels."""
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        orientation = im.getexif().get(0x0112, 1)
        return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def encode(rgb: np.ndarray, fmt: str = "jpeg", quality: int = 92) -> tuple[bytes, str]:
    pil_fmt, mime = _FORMATS[fmt]
    buf = io.BytesIO()
    kwargs: dict = {}
    if pil_fmt in ("JPEG", "WEBP"):
        kwargs["quality"] = quality
    if pil_fmt == "JPEG":
        kwargs.update(subsampling=0 if quality >= 90 else 2, optimize=True)
    Image.fromarray(rgb).save(buf, pil_fmt, **kwargs)
    return buf.getvalue(), mime
