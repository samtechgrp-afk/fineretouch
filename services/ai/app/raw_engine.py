"""Camera RAW decoding for the AI service.

Two jobs:

* ``decode_camera_raw``: full-precision decode through LibRaw (``rawpy``): white balance
  on the linear sensor data (camera, daylight or a Kelvin/tint target computed from the
  camera's XYZ matrix), highlight recovery, a chosen demosaic algorithm and an output
  colour space, returning an HxWx3 ``uint16`` (or ``uint8``) array that ``pipeline.render``
  consumes directly.
* Cheap metadata and preview access without LibRaw: magic-byte format detection, a small
  bounds-checked TIFF/EXIF reader (CR2, NEF, ARW, DNG, ORF, RW2, PEF), Canon CR3's
  ISOBMFF boxes and Fujifilm RAF's header. These also drive the fallback path: a damaged
  or unsupported file still yields its embedded camera JPEG, and if even that is missing
  the caller gets a structured placeholder result instead of an exception.

Supported containers: Canon CR2/CR3, Nikon NEF/NRW, Sony ARW/SR2/SRF, Fujifilm RAF, Adobe
DNG, Olympus ORF, Panasonic RW2, Pentax PEF. Canon has not published a ".cr4" format; the
extension is accepted so cards with unexpected names do not get rejected, and such a file
is decoded only if its bytes are one of the formats above.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

try:  # rawpy ships manylinux wheels with LibRaw bundled; keep the service importable without it.
    import rawpy
except ImportError:  # pragma: no cover - exercised only where rawpy is absent
    rawpy = None  # type: ignore[assignment]

log = logging.getLogger("fineretouch.raw")

RawFormat = Literal["cr2", "cr3", "nef", "arw", "raf", "dng", "orf", "rw2", "pef"]
DecodeSource = Literal["demosaic", "embedded_preview", "placeholder"]

#: File extensions the service accepts, mapped to the format we expect inside.
RAW_EXTENSIONS: dict[str, RawFormat | None] = {
    ".cr2": "cr2",
    ".cr3": "cr3",
    ".cr4": None,  # no such published format; detected by content if it ever appears
    ".nef": "nef",
    ".nrw": "nef",
    ".arw": "arw",
    ".srf": "arw",
    ".sr2": "arw",
    ".raf": "raf",
    ".dng": "dng",
    ".orf": "orf",
    ".rw2": "rw2",
    ".pef": "pef",
}

DEMOSAIC_ALIASES = {
    "linear": "LINEAR",
    "bilinear": "LINEAR",
    "vng": "VNG",
    "ppg": "PPG",
    "ahd": "AHD",
    "dcb": "DCB",
    "dht": "DHT",
    "dhd": "DHT",  # common misspelling of LibRaw's DHT
    "aahd": "AAHD",
}
HIGHLIGHT_MODES = ("clip", "ignore", "blend", "reconstruct")
COLOR_SPACES = {"srgb": ("sRGB", (2.4, 12.92)), "adobe": ("Adobe", (2.2, 0.0)), "prophoto": ("ProPhoto", (1.8, 0.0))}

_CANON_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
_MAX_IFDS = 64
_MAX_TAGS = 1024


class RawDecodeError(ValueError):
    """The file could not be decoded and no fallback was allowed or possible."""


@dataclass
class RawMetadata:
    format: RawFormat | None = None
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    iso: int | None = None
    exposure_time: float | None = None  # seconds
    f_number: float | None = None
    focal_length: float | None = None  # mm
    datetime_original: str | None = None
    orientation: int = 1  # EXIF 1..8
    color_space: str | None = None  # "sRGB" | "Adobe RGB" | "Uncalibrated"
    width: int | None = None  # sensor output size before orientation, when known
    height: int | None = None
    camera_white_balance: list[float] | None = None
    color_matrix: list[list[float]] | None = None  # camera RGB from XYZ (3x3)

    @property
    def shutter(self) -> str | None:
        t = self.exposure_time
        if not t:
            return None
        return f"1/{round(1 / t)}" if t < 0.5 else f"{t:g}s"


@dataclass
class RawDecodeResult:
    image: np.ndarray  # HxWx3, uint16 or uint8, oriented
    bits: int
    source: DecodeSource
    format: RawFormat | None
    metadata: RawMetadata
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.source != "placeholder"


# --------------------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------------------


def _is_tiff_like(head: bytes) -> bool:
    return head[:4] in (b"II*\x00", b"MM\x00*", b"IIRO", b"IIRS", b"MMOR", b"IIU\x00")


def detect_raw_format(data: bytes, filename: str | None = None) -> RawFormat | None:
    """Identify a camera RAW container from its first bytes; ``None`` for anything else
    (including ordinary TIFFs, which share the RAW container but carry no sensor data)."""
    head = bytes(data[:16])
    if head.startswith(b"FUJIFILMCCD-RAW"):
        return "raf"
    if head[4:8] == b"ftyp" and head[8:12] == b"crx ":
        return "cr3"
    if head[:4] in (b"IIRO", b"IIRS", b"MMOR"):
        return "orf"
    if head[:4] == b"IIU\x00":
        return "rw2"
    if head[:4] not in (b"II*\x00", b"MM\x00*"):
        return None
    if head[8:10] == b"CR":
        return "cr2"
    try:
        tiff = _Tiff(data)
        ifd0 = tiff.ifd(tiff.first_ifd)
    except ValueError:
        return None
    if 0xC612 in ifd0:  # DNGVersion
        return "dng"
    make = (tiff.text(ifd0, 0x010F) or "").strip().upper()
    for prefix, fmt in (("NIKON", "nef"), ("SONY", "arw"), ("PENTAX", "pef"), ("RICOH", "pef"), ("CANON", "cr2"), ("OLYMPUS", "orf"), ("OM DIGITAL", "orf"), ("PANASONIC", "rw2")):
        if make.startswith(prefix):
            return fmt  # type: ignore[return-value]
    if filename:
        return RAW_EXTENSIONS.get(Path(filename).suffix.lower())
    return None


def is_camera_raw(data: bytes, filename: str | None = None) -> bool:
    return detect_raw_format(data, filename) is not None


# --------------------------------------------------------------------------------------
# Minimal TIFF / EXIF reader (bounds checked; never trusts offsets)
# --------------------------------------------------------------------------------------

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


class _Tiff:
    def __init__(self, data: bytes, base: int = 0):
        self.data = memoryview(data)
        self.base = base
        order = bytes(self.data[base : base + 2])
        if order == b"II":
            self.e = "<"
        elif order == b"MM":
            self.e = ">"
        else:
            raise ValueError("not a TIFF structure")
        self.first_ifd = self.u32(base + 4)

    def _need(self, off: int, n: int) -> None:
        if off < 0 or off + n > len(self.data):
            raise ValueError("offset outside file")

    def u16(self, off: int) -> int:
        self._need(off, 2)
        return struct.unpack_from(self.e + "H", self.data, off)[0]

    def u32(self, off: int) -> int:
        self._need(off, 4)
        return struct.unpack_from(self.e + "I", self.data, off)[0]

    def ifd(self, rel: int) -> dict[int, tuple[int, int, int]]:
        """tag -> (type, count, absolute offset of the value bytes)."""
        off = self.base + rel
        n = self.u16(off)
        if n > _MAX_TAGS:
            raise ValueError("implausible IFD")
        out: dict[int, tuple[int, int, int]] = {}
        for i in range(n):
            e = off + 2 + i * 12
            tag, typ, count = self.u16(e), self.u16(e + 2), self.u32(e + 4)
            size = _TYPE_SIZES.get(typ, 1) * count
            value_at = e + 8 if size <= 4 else self.base + self.u32(e + 8)
            out[tag] = (typ, count, value_at)
        return out

    def next_ifd(self, rel: int) -> int:
        off = self.base + rel
        return self.u32(off + 2 + self.u16(off) * 12)

    def values(self, ifd: dict, tag: int) -> list[float]:
        if tag not in ifd:
            return []
        typ, count, at = ifd[tag]
        count = min(count, 4096)
        fmt = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d", 13: "I"}.get(typ)
        if fmt:
            n = struct.calcsize(fmt)
            self._need(at, n * count)
            return list(struct.unpack_from(self.e + fmt * count, self.data, at))
        if typ in (5, 10):
            self._need(at, 8 * count)
            raw = struct.unpack_from(self.e + ("I" if typ == 5 else "i") * (2 * count), self.data, at)
            return [raw[i] / raw[i + 1] if raw[i + 1] else 0.0 for i in range(0, len(raw), 2)]
        return []

    def first(self, ifd: dict, tag: int) -> float | None:
        v = self.values(ifd, tag)
        return v[0] if v else None

    def text(self, ifd: dict, tag: int) -> str | None:
        if tag not in ifd:
            return None
        typ, count, at = ifd[tag]
        if typ not in (1, 2, 7):
            return None
        count = min(count, 512)
        self._need(at, count)
        s = bytes(self.data[at : at + count]).split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
        return s or None


def _fill_from_tiff(meta: RawMetadata, tiff: _Tiff, ifd0_rel: int | None, exif_rel: int | None = None) -> None:
    """Copy IFD0 + Exif IFD fields into ``meta`` (only fields still unset)."""
    if ifd0_rel is not None:
        ifd0 = tiff.ifd(ifd0_rel)
        meta.make = meta.make or tiff.text(ifd0, 0x010F)
        meta.model = meta.model or tiff.text(ifd0, 0x0110)
        o = tiff.first(ifd0, 0x0112)
        if o and 1 <= int(o) <= 8 and meta.orientation == 1:
            meta.orientation = int(o)
        if exif_rel is None and 0x8769 in ifd0:
            exif_rel = int(tiff.first(ifd0, 0x8769) or 0) or None
    if exif_rel is None:
        return
    ex = tiff.ifd(exif_rel)
    if meta.exposure_time is None:
        meta.exposure_time = tiff.first(ex, 0x829A)
    if meta.f_number is None:
        meta.f_number = tiff.first(ex, 0x829D)
    if meta.iso is None:
        iso = tiff.first(ex, 0x8827) or tiff.first(ex, 0x8833)
        meta.iso = int(iso) if iso else None
    meta.datetime_original = meta.datetime_original or tiff.text(ex, 0x9003)
    if meta.focal_length is None:
        meta.focal_length = tiff.first(ex, 0x920A)
    meta.lens = meta.lens or tiff.text(ex, 0xA434)
    if meta.color_space is None:
        cs = tiff.first(ex, 0xA001)
        meta.color_space = {1: "sRGB", 2: "Adobe RGB", 0xFFFF: "Uncalibrated"}.get(int(cs)) if cs else None


def _iter_boxes(data: memoryview, start: int, end: int):
    """ISOBMFF boxes in [start, end): (type, payload_start, box_end)."""
    off = start
    while off + 8 <= end:
        size = struct.unpack_from(">I", data, off)[0]
        typ = bytes(data[off + 4 : off + 8])
        hdr = 8
        if size == 1:
            if off + 16 > end:
                return
            size = struct.unpack_from(">Q", data, off + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr or off + size > end:
            return
        yield typ, off + hdr, off + size
        off += size


def _cr3_metadata(data: bytes, meta: RawMetadata) -> None:
    mv = memoryview(data)
    for typ, start, end in _iter_boxes(mv, 0, len(mv)):
        if typ != b"moov":
            continue
        for t2, s2, e2 in _iter_boxes(mv, start, end):
            if t2 == b"uuid" and bytes(mv[s2 : s2 + 16]) == _CANON_UUID:
                for t3, s3, _e3 in _iter_boxes(mv, s2 + 16, e2):
                    if t3 in (b"CMT1", b"CMT2"):
                        tiff = _Tiff(data, s3)
                        if t3 == b"CMT1":
                            _fill_from_tiff(meta, tiff, tiff.first_ifd)
                        else:
                            _fill_from_tiff(meta, tiff, None, tiff.first_ifd)
        return


def _jpeg_exif(jpeg: bytes, meta: RawMetadata) -> None:
    """Fill ``meta`` from the APP1 Exif segment of a JPEG (used for RAF)."""
    off = 2
    while off + 4 <= len(jpeg) and jpeg[off] == 0xFF:
        marker = jpeg[off + 1]
        seg_len = struct.unpack_from(">H", jpeg, off + 2)[0]
        if marker == 0xE1 and jpeg[off + 4 : off + 10] == b"Exif\x00\x00":
            tiff = _Tiff(jpeg, off + 10)
            _fill_from_tiff(meta, tiff, tiff.first_ifd)
            return
        if marker == 0xDA:
            return
        off += 2 + seg_len


def _raf_jpeg(data: bytes) -> bytes | None:
    if len(data) < 92:
        return None
    off, length = struct.unpack_from(">II", data, 84)
    if off <= 0 or length <= 0 or off + length > len(data):
        return None
    return bytes(data[off : off + length])


def read_raw_metadata(data: bytes, filename: str | None = None) -> RawMetadata:
    """EXIF-level metadata without demosaicing. Never raises on damaged input."""
    fmt = detect_raw_format(data, filename)
    meta = RawMetadata(format=fmt)
    try:
        if fmt == "cr3":
            _cr3_metadata(data, meta)
        elif fmt == "raf":
            jpeg = _raf_jpeg(data)
            if jpeg:
                _jpeg_exif(jpeg, meta)
        elif _is_tiff_like(bytes(data[:4])):
            tiff = _Tiff(data)
            _fill_from_tiff(meta, tiff, tiff.first_ifd)
    except (ValueError, struct.error) as exc:
        log.debug("metadata parse stopped early: %s", exc)
    return meta


# --------------------------------------------------------------------------------------
# Embedded preview (fallback path and cheap thumbnails)
# --------------------------------------------------------------------------------------


def _jpeg_frame_size(data: memoryview, soi: int) -> tuple[int, int] | None:
    """(width, height) from the SOF of a baseline/progressive JPEG at ``soi``; ``None`` for
    lossless JPEG (the compressed sensor data in CR2/DNG) or anything malformed."""
    off = soi + 2
    end = len(data)
    for _ in range(64):
        if off + 4 > end or data[off] != 0xFF:
            return None
        marker = data[off + 1]
        if marker == 0xFF:
            off += 1
            continue
        seg_len = struct.unpack_from(">H", data, off + 2)[0]
        if marker in (0xC0, 0xC1, 0xC2):
            if off + 9 > end:
                return None
            h, w = struct.unpack_from(">HH", data, off + 5)
            return (w, h) if w and h else None
        if marker in (0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF, 0xDA, 0xD9):
            return None
        off += 2 + seg_len
    return None


def find_embedded_jpeg(data: bytes, max_candidates: int = 256) -> bytes | None:
    """The largest decodable JPEG embedded anywhere in the file (format independent)."""
    mv = memoryview(data)
    best: tuple[int, int] | None = None  # (pixels, offset)
    pos = 0
    for _ in range(max_candidates):
        pos = data.find(b"\xff\xd8\xff", pos)
        if pos < 0:
            break
        size = _jpeg_frame_size(mv, pos)
        if size and (best is None or size[0] * size[1] > best[0]):
            best = (size[0] * size[1], pos)
        pos += 3
    if best is None:
        return None
    start = best[1]
    end = data.find(b"\xff\xd9", start)
    # Pillow ignores trailing bytes, so an unknown end is fine; a found EOI keeps it small.
    return bytes(data[start : (end + 2 if end > 0 else len(data))])


def _orient(img: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        return img[:, ::-1]
    if orientation == 3:
        return img[::-1, ::-1]
    if orientation == 4:
        return img[::-1]
    if orientation == 5:
        return img.swapaxes(0, 1)
    if orientation == 6:
        return np.rot90(img, -1)
    if orientation == 7:
        return img.swapaxes(0, 1)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(img, 1)
    return img


def _decode_preview_jpeg(jpeg: bytes, raw_orientation: int) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(jpeg)) as im:
        own = im.getexif().get(0x0112, 1)
        if own and own != 1:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        else:  # camera previews usually carry no orientation of their own
            rgb = _orient(np.asarray(im.convert("RGB")), raw_orientation)
    return np.ascontiguousarray(rgb)


def extract_embedded_preview(data: bytes, filename: str | None = None) -> tuple[np.ndarray, RawMetadata] | None:
    """Oriented uint8 RGB of the camera's embedded preview, or ``None``."""
    meta = read_raw_metadata(data, filename)
    candidates: list[bytes] = []
    if meta.format == "raf" and (j := _raf_jpeg(data)):
        candidates.append(j)
    if rawpy is not None:
        try:
            with rawpy.imread(io.BytesIO(data)) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                candidates.append(bytes(thumb.data))
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                return np.ascontiguousarray(_orient(np.asarray(thumb.data, np.uint8), meta.orientation)), meta
        except Exception as exc:  # LibRawError subclasses, IO errors
            log.debug("rawpy could not extract a thumbnail: %s", exc)
    if j := find_embedded_jpeg(data):
        candidates.append(j)
    for jpeg in candidates:
        try:
            return _decode_preview_jpeg(jpeg, meta.orientation), meta
        except Exception as exc:
            log.debug("embedded JPEG did not decode: %s", exc)
    return None


# --------------------------------------------------------------------------------------
# White balance helpers
# --------------------------------------------------------------------------------------


def kelvin_to_xy(kelvin: float) -> tuple[float, float]:
    """CIE 1931 xy of a Planckian radiator (Kim et al. 2002 cubic fit, 1667-25000 K)."""
    t = float(np.clip(kelvin, 1667.0, 25000.0))
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def _apply_tint(mul: list[float], tint: float) -> list[float]:
    """Positive tint = magenta (red and blue up against green), like Lightroom's slider."""
    if not tint:
        return mul
    f = 2.0 ** (float(np.clip(tint, -150, 150)) / 150.0)
    return [mul[0] * f, mul[1], mul[2] * f, mul[3]]


def wb_multipliers_for_temperature(
    cam_xyz: np.ndarray, kelvin: float, tint: float = 0.0, base: list[float] | None = None
) -> list[float] | None:
    """LibRaw ``user_wb`` (R, G, B, G2) that renders a white under ``kelvin`` as neutral.

    The illuminant's XYZ is projected into camera space with the camera's XYZ->camera
    matrix; channel gains are its reciprocal, normalised so green is 1. These gains act on
    linear sensor values before demosaicing, i.e. in the camera's linear RGB space. ``base``
    multiplies the gains when ``cam_xyz`` maps into an already balanced space (see
    ``_camera_from_xyz``).
    """
    m = np.asarray(cam_xyz, np.float64)[:3, :3]
    if not np.isfinite(m).all() or np.abs(m).sum() < 1e-6:
        return None
    x, y = kelvin_to_xy(kelvin)
    white = np.array([x / y, 1.0, (1.0 - x - y) / y])
    cam = m @ white
    if (cam <= 1e-9).any():
        return None
    mul = 1.0 / cam
    if base is not None:
        mul = mul * np.asarray(base[:3], np.float64)
    mul = mul / mul[1]
    return _apply_tint([float(mul[0]), 1.0, float(mul[2]), 1.0], tint)


SRGB_FROM_XYZ = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def _camera_from_xyz(raw) -> tuple[np.ndarray, list[float] | None] | None:
    """XYZ -> camera matrix for Kelvin white balance, plus the base multipliers it assumes.

    LibRaw fills ``rgb_xyz_matrix`` from its per-model tables (CR2, NEF, ARW, ...). For
    DNG it stays empty and only ``color_matrix`` (camera -> sRGB, defined on daylight-balanced
    data) is set, so we invert that and stack it on the daylight multipliers.
    """
    cam_xyz = np.asarray(raw.rgb_xyz_matrix, np.float64)[:3, :3]
    if np.abs(cam_xyz).sum() > 1e-6:
        return cam_xyz, None
    rgb_cam = np.asarray(raw.color_matrix, np.float64)[:3, :3]
    day = [float(v) for v in raw.daylight_whitebalance]
    if np.abs(rgb_cam).sum() < 1e-6 or sum(day[:3]) <= 0 or abs(np.linalg.det(rgb_cam)) < 1e-9:
        return None
    return np.linalg.inv(rgb_cam) @ SRGB_FROM_XYZ, day


# --------------------------------------------------------------------------------------
# Full decode
# --------------------------------------------------------------------------------------


def _read_source(source: bytes | bytearray | memoryview | str | os.PathLike) -> tuple[bytes, str | None]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None
    p = Path(source)
    return p.read_bytes(), p.name


def _to_bits(img: np.ndarray, bits: int) -> np.ndarray:
    if bits == 16:
        return img.astype(np.uint16) * 257 if img.dtype == np.uint8 else img.astype(np.uint16)
    return img if img.dtype == np.uint8 else (img.astype(np.uint32) * 255 // 65535).astype(np.uint8)


def resize_long_edge(img: np.ndarray, max_long_edge: int | None) -> np.ndarray:
    h, w = img.shape[:2]
    if not max_long_edge or max(h, w) <= max_long_edge:
        return img
    import cv2

    s = max_long_edge / max(h, w)
    return cv2.resize(np.ascontiguousarray(img), (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _placeholder(meta: RawMetadata, bits: int, longest: int = 256) -> np.ndarray:
    w, h = meta.width or 3, meta.height or 2
    if meta.orientation in (5, 6, 7, 8):
        w, h = h, w
    s = longest / max(w, h)
    img = np.full((max(1, round(h * s)), max(1, round(w * s)), 3), 128, np.uint8)
    return _to_bits(img, bits)


def _postprocess(
    raw,
    *,
    use_camera_wb: bool,
    temperature: float | None,
    tint: float,
    demosaic: str,
    highlight_mode: str,
    color_space: str,
    bits: int,
    half_size: bool,
    auto_bright: bool,
    meta: RawMetadata,
    warnings: list[str],
) -> np.ndarray:
    algo_name = DEMOSAIC_ALIASES.get(demosaic.lower())
    if algo_name is None:
        raise ValueError(f"Unknown demosaic algorithm {demosaic!r}")
    algo = getattr(rawpy.DemosaicAlgorithm, algo_name)
    if not algo.isSupported:
        warnings.append(f"{algo_name} demosaic is not available in this LibRaw build; used AHD")
        algo = rawpy.DemosaicAlgorithm.AHD

    hl = highlight_mode.lower()
    if hl not in HIGHLIGHT_MODES:
        raise ValueError(f"Unknown highlight mode {highlight_mode!r}")
    hl_mode = {
        "clip": rawpy.HighlightMode.Clip,
        "ignore": rawpy.HighlightMode.Ignore,
        "blend": rawpy.HighlightMode.Blend,
        "reconstruct": rawpy.HighlightMode.ReconstructDefault,
    }[hl]

    cs = COLOR_SPACES.get(color_space.lower())
    if cs is None:
        raise ValueError(f"Unknown colour space {color_space!r}")
    cs_enum, gamma = getattr(rawpy.ColorSpace, cs[0]), cs[1]

    cam_wb = [float(v) for v in raw.camera_whitebalance]
    meta.camera_white_balance = cam_wb
    cam = _camera_from_xyz(raw)
    meta.color_matrix = cam[0].round(6).tolist() if cam else None
    has_cam_wb = sum(cam_wb[:3]) > 0

    wb: dict = {}
    if temperature is not None:
        mul = wb_multipliers_for_temperature(cam[0], temperature, tint, cam[1]) if cam else None
        if mul is not None:
            wb["user_wb"] = mul
        else:
            warnings.append("camera colour matrix unknown; Kelvin white balance ignored")
    if not wb:
        if use_camera_wb and has_cam_wb:
            base = cam_wb if cam_wb[3] else [cam_wb[0], cam_wb[1], cam_wb[2], cam_wb[1]]
            wb = {"user_wb": _apply_tint(base, tint)} if tint else {"use_camera_wb": True}
        else:
            if use_camera_wb:
                warnings.append("no camera white balance in file; used daylight")
            day = [float(v) for v in raw.daylight_whitebalance]
            wb = {"user_wb": _apply_tint(day, tint)} if tint and sum(day[:3]) > 0 else {"use_camera_wb": False}

    return raw.postprocess(
        demosaic_algorithm=algo,
        highlight_mode=hl_mode,
        output_color=cs_enum,
        gamma=gamma,
        output_bps=bits,
        half_size=half_size,
        no_auto_bright=not auto_bright,
        **wb,
    )


def decode_camera_raw(
    raw_bytes_or_path: bytes | bytearray | memoryview | str | os.PathLike,
    use_camera_wb: bool = True,
    output_bps: int = 16,
    *,
    temperature: float | None = None,
    tint: float = 0.0,
    demosaic: str = "AHD",
    highlight_mode: str = "blend",
    color_space: str = "srgb",
    max_long_edge: int | None = None,
    half_size: bool | None = None,
    auto_bright: bool = True,
    fallback: bool = True,
    filename: str | None = None,
) -> RawDecodeResult:
    """Demosaic a camera RAW into an oriented HxWx3 array (``uint16`` when ``output_bps=16``).

    temperature / tint: optional Kelvin target and magenta(+)/green(-) shift; overrides the
        camera white balance. With ``use_camera_wb=False`` and no temperature, daylight is used.
    demosaic: AHD (default), DHT ("DHD" accepted), PPG, VNG, DCB, AAHD or LINEAR.
    highlight_mode: clip, ignore, blend (default; keeps colour in clipped skies) or reconstruct.
    color_space: srgb, adobe or prophoto, each with its standard transfer curve.
    max_long_edge: downscale the result; when it is at most half the sensor size LibRaw's
        half-size mode skips demosaicing entirely (about 4x faster, ideal for analysis).
    fallback: on a damaged/unsupported file return the embedded preview, or a grey
        placeholder (``result.ok`` is False) instead of raising ``RawDecodeError``.
    """
    if output_bps not in (8, 16):
        raise ValueError("output_bps must be 8 or 16")
    data, path_name = _read_source(raw_bytes_or_path)
    filename = filename or path_name
    meta = read_raw_metadata(data, filename)
    warnings: list[str] = []
    error: str | None = None

    if rawpy is None:
        error = "rawpy (LibRaw) is not installed"
    else:
        try:
            with rawpy.imread(io.BytesIO(data)) as raw:
                sizes = raw.sizes
                meta.width, meta.height = int(sizes.width), int(sizes.height)
                flip = {0: 1, 3: 3, 5: 8, 6: 6}.get(int(sizes.flip))
                if flip:
                    meta.orientation = flip
                if half_size is None:
                    half_size = bool(max_long_edge) and max(meta.width, meta.height) // 2 >= max_long_edge  # type: ignore[operator]
                img = _postprocess(
                    raw,
                    use_camera_wb=use_camera_wb,
                    temperature=temperature,
                    tint=tint,
                    demosaic=demosaic,
                    highlight_mode=highlight_mode,
                    color_space=color_space,
                    bits=output_bps,
                    half_size=bool(half_size),
                    auto_bright=auto_bright,
                    meta=meta,
                    warnings=warnings,
                )
            img = resize_long_edge(img, max_long_edge)
            return RawDecodeResult(np.ascontiguousarray(img), output_bps, "demosaic", meta.format, meta, warnings)
        except ValueError as exc:
            if "Unknown" in str(exc):
                raise  # a caller error (bad option), not a damaged file
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # rawpy.LibRawError and friends
            error = f"{type(exc).__name__}: {exc}"

    if not fallback:
        raise RawDecodeError(error or "decode failed")
    log.warning("RAW demosaic failed (%s); falling back to embedded preview", error)
    preview = extract_embedded_preview(data, filename)
    if preview is not None:
        img, pmeta = preview
        meta.orientation = pmeta.orientation
        warnings.append("used the camera's embedded preview (8-bit, camera rendering)")
        img = resize_long_edge(img, max_long_edge)
        return RawDecodeResult(_to_bits(img, output_bps), output_bps, "embedded_preview", meta.format, meta, warnings, error)
    warnings.append("no decodable image data; returned a grey placeholder")
    return RawDecodeResult(_placeholder(meta, output_bps), output_bps, "placeholder", meta.format, meta, warnings, error)


def oriented_size(data: bytes) -> tuple[int, int]:
    """(width, height) of the decoded, oriented image without demosaicing."""
    if rawpy is not None:
        try:
            with rawpy.imread(io.BytesIO(data)) as raw:
                w, h, flip = int(raw.sizes.width), int(raw.sizes.height), int(raw.sizes.flip)
            return (h, w) if flip in (5, 6) else (w, h)
        except Exception:
            pass
    preview = extract_embedded_preview(data)
    if preview is None:
        raise RawDecodeError("unreadable RAW file")
    h, w = preview[0].shape[:2]
    return w, h


def to_uint8(img: np.ndarray) -> np.ndarray:
    """uint8 view of a decoded image (detectors and statistics work on 8-bit)."""
    return _to_bits(img, 8)
