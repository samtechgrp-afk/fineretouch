"""Synthetic camera RAW files for tests.

``make_dng`` writes a real, LibRaw-decodable DNG: an RGGB Bayer mosaic (12-bit data in
16-bit samples) in a SubIFD, a JPEG preview in IFD0 like cameras and converters write
it, an Exif IFD, and an sRGB-primaries colour matrix so expected colours are easy to
reason about. ``make_tiff_raw`` builds the same TIFF layout with another maker's tags, for
format detection; ``make_cr3`` and ``make_raf`` build the container headers of those formats.
"""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image

# TIFF types
BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED, SRATIONAL = 1, 2, 3, 4, 5, 7, 10
_SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, UNDEFINED: 1, SRATIONAL: 8}

# XYZ(D65) -> linear sRGB: a "camera" whose native primaries are sRGB.
SRGB_FROM_XYZ = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]


def _encode_values(typ: int, values) -> bytes:
    if typ == ASCII:
        return values.encode() + b"\x00"
    if typ in (BYTE, UNDEFINED):
        return bytes(values)
    if typ == SHORT:
        return struct.pack(f"<{len(values)}H", *values)
    if typ == LONG:
        return struct.pack(f"<{len(values)}I", *values)
    if typ in (RATIONAL, SRATIONAL):
        out = b""
        for v in values:
            num, den = v if isinstance(v, tuple) else (round(v * 10000), 10000)
            out += struct.pack("<ii" if typ == SRATIONAL else "<II", num, den)
        return out
    raise ValueError(typ)


def _count(typ: int, values) -> int:
    if typ == ASCII:
        return len(values) + 1
    return len(values)


def build_tiff(ifds: dict[str, list[tuple[int, int, object]]], blobs: dict[str, bytes], magic: bytes = b"II*\x00") -> bytes:
    """Little-endian TIFF from named IFDs. The first IFD is IFD0. A value given as the
    string "@name" is replaced with the file offset of IFD or blob ``name``."""
    names = list(ifds)
    sizes = {}
    for name in names:
        extra = 0
        for tag, typ, vals in ifds[name]:
            n = _count(typ, vals if not isinstance(vals, str) or typ == ASCII else [0]) * _SIZES[typ]
            extra += 0 if n <= 4 else n + (n & 1)
        sizes[name] = 2 + 12 * len(ifds[name]) + 4 + extra
    offsets: dict[str, int] = {}
    pos = 8
    for name in names:
        offsets[name] = pos
        pos += sizes[name]
    for name, blob in blobs.items():
        offsets[name] = pos
        pos += len(blob) + (len(blob) & 1)

    def resolve(v):
        return offsets[v[1:]] if isinstance(v, str) and v.startswith("@") else v

    out = bytearray(magic + struct.pack("<I", offsets[names[0]]))
    for name in names:
        entries = sorted(ifds[name], key=lambda e: e[0])
        start = offsets[name]
        data_pos = start + 2 + 12 * len(entries) + 4
        head = struct.pack("<H", len(entries))
        tail = b""
        for tag, typ, vals in entries:
            if typ != ASCII:
                vals = [resolve(v) for v in (vals if isinstance(vals, list) else [vals])]
            payload = _encode_values(typ, vals)
            count = _count(typ, vals)
            if len(payload) <= 4:
                head += struct.pack("<HHI", tag, typ, count) + payload.ljust(4, b"\x00")
            else:
                head += struct.pack("<HHII", tag, typ, count, data_pos)
                payload += b"\x00" * (len(payload) & 1)
                tail += payload
                data_pos += len(payload)
        assert len(out) == start
        out += head + struct.pack("<I", 0) + tail
    for name, blob in blobs.items():
        assert len(out) == offsets[name]
        out += blob + b"\x00" * (len(blob) & 1)
    return bytes(out)


def scene(h: int, w: int) -> np.ndarray:
    """Linear RGB in [0, 1]: a horizontal grey ramp on top, R/G/B patches below."""
    img = np.zeros((h, w, 3), np.float32)
    img[: h // 2] = np.linspace(0.05, 0.9, w, dtype=np.float32)[None, :, None]
    third = w // 3
    img[h // 2 :, :third] = (0.8, 0.1, 0.1)
    img[h // 2 :, third : 2 * third] = (0.1, 0.8, 0.1)
    img[h // 2 :, 2 * third :] = (0.1, 0.1, 0.8)
    return img


def mosaic_rggb(rgb: np.ndarray, white: int = 4095) -> np.ndarray:
    h, w = rgb.shape[:2]
    cfa = np.empty((h, w), np.uint16)
    v = np.clip(rgb * white, 0, white).astype(np.uint16)
    cfa[0::2, 0::2] = v[0::2, 0::2, 0]
    cfa[0::2, 1::2] = v[0::2, 1::2, 1]
    cfa[1::2, 0::2] = v[1::2, 0::2, 1]
    cfa[1::2, 1::2] = v[1::2, 1::2, 2]
    return cfa


def jpeg(rgb8: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb8).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def make_dng(
    width: int = 96,
    height: int = 64,
    *,
    orientation: int = 1,
    as_shot_neutral: tuple[float, float, float] = (1.0, 1.0, 1.0),
    preview: np.ndarray | None = None,
    make: str = "FineRetouch",
    model: str = "Synthetic DNG",
    iso: int = 400,
    exposure: tuple[int, int] = (1, 250),
    f_number: tuple[int, int] = (28, 10),
    focal_length: tuple[int, int] = (85, 1),
    lens: str = "FR 85mm F1.4",
    raw_rgb: np.ndarray | None = None,
) -> bytes:
    raw_rgb = scene(height, width) if raw_rgb is None else raw_rgb
    # Camera records a white surface as as_shot_neutral; bake that cast into the sensor data.
    cfa = mosaic_rggb(raw_rgb * np.array(as_shot_neutral, np.float32))
    if preview is None:
        preview = (np.clip(scene(height // 2, width // 2), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    pjpeg = jpeg(preview)
    ph, pw = preview.shape[:2]
    matrix = [v for row in SRGB_FROM_XYZ for v in row]
    ifds = {
        "ifd0": [
            (0x00FE, LONG, [1]),  # NewSubFileType: reduced-resolution preview
            (0x0100, LONG, [pw]),
            (0x0101, LONG, [ph]),
            (0x0102, SHORT, [8, 8, 8]),
            (0x0103, SHORT, [7]),  # JPEG
            (0x0106, SHORT, [6]),  # YCbCr
            (0x010F, ASCII, make),
            (0x0110, ASCII, model),
            (0x0111, LONG, ["@preview"]),
            (0x0112, SHORT, [orientation]),
            (0x0115, SHORT, [3]),
            (0x0116, LONG, [ph]),
            (0x0117, LONG, [len(pjpeg)]),
            (0x014A, LONG, ["@raw"]),  # SubIFDs
            (0x8769, LONG, ["@exif"]),
            (0xC612, BYTE, [1, 4, 0, 0]),  # DNGVersion
            (0xC613, BYTE, [1, 1, 0, 0]),
            (0xC614, ASCII, f"{make} {model}"),
            (0xC621, SRATIONAL, matrix),  # ColorMatrix1 (XYZ -> camera)
            (0xC628, RATIONAL, list(as_shot_neutral)),  # AsShotNeutral
            (0xC65A, SHORT, [21]),  # CalibrationIlluminant1 = D65
        ],
        "raw": [
            (0x00FE, LONG, [0]),
            (0x0100, LONG, [width]),
            (0x0101, LONG, [height]),
            (0x0102, SHORT, [16]),
            (0x0103, SHORT, [1]),
            (0x0106, SHORT, [32803]),  # CFA
            (0x0111, LONG, ["@cfa"]),
            (0x0115, SHORT, [1]),
            (0x0116, LONG, [height]),
            (0x0117, LONG, [cfa.nbytes]),
            (0x011C, SHORT, [1]),
            (0x828D, SHORT, [2, 2]),  # CFARepeatPatternDim
            (0x828E, BYTE, [0, 1, 1, 2]),  # CFAPattern RGGB
            (0xC61A, LONG, [0]),  # BlackLevel
            (0xC61D, LONG, [4095]),  # WhiteLevel
        ],
        "exif": [
            (0x829A, RATIONAL, [exposure]),
            (0x829D, RATIONAL, [f_number]),
            (0x8827, SHORT, [iso]),
            (0x9003, ASCII, "2026:11:06 10:30:00"),
            (0x920A, RATIONAL, [focal_length]),
            (0xA001, SHORT, [1]),
            (0xA434, ASCII, lens),
        ],
    }
    return build_tiff(ifds, {"preview": pjpeg, "cfa": cfa.astype("<u2").tobytes()})


def make_tiff_raw(make: str, *, cr2_marker: bool = False, magic: bytes = b"II*\x00", orientation: int = 1, preview: np.ndarray | None = None) -> bytes:
    """A TIFF-structured RAW header (no decodable sensor data) with an embedded JPEG."""
    preview = np.full((24, 36, 3), (200, 120, 90), np.uint8) if preview is None else preview
    pjpeg = jpeg(preview)
    ifds = {
        "ifd0": [
            (0x010F, ASCII, make),
            (0x0110, ASCII, f"{make} Test Body"),
            (0x0112, SHORT, [orientation]),
            (0x0201, LONG, ["@preview"]),  # JPEGInterchangeFormat
            (0x0202, LONG, [len(pjpeg)]),
            (0x8769, LONG, ["@exif"]),
        ],
        "exif": [(0x8827, SHORT, [3200]), (0x829A, RATIONAL, [(1, 60)]), (0x829D, RATIONAL, [(4, 1)])],
    }
    data = bytearray(build_tiff(ifds, {"preview": pjpeg}, magic=magic))
    if cr2_marker:
        # Real CR2s carry "CR\x02\x00" at byte 8; detection only needs the header, so the
        # IFD0 bytes it overwrites do not matter for callers that set this.
        data[8:12] = b"CR\x02\x00"
    return bytes(data)


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def make_cr3(model: str = "Canon EOS R5", orientation: int = 6, iso: int = 800, preview: np.ndarray | None = None) -> bytes:
    """ISOBMFF skeleton of a CR3: ftyp + moov/uuid(Canon)/CMT1+CMT2 + a JPEG in mdat."""
    cmt1 = build_tiff({"ifd0": [(0x010F, ASCII, "Canon"), (0x0110, ASCII, model), (0x0112, SHORT, [orientation])]}, {})
    cmt2 = build_tiff({"exif": [(0x8827, SHORT, [iso]), (0x829A, RATIONAL, [(1, 125)]), (0x829D, RATIONAL, [(56, 10)]), (0xA434, ASCII, "RF85mm F1.2 L USM")]}, {})
    canon = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48") + _box(b"CMT1", cmt1) + _box(b"CMT2", cmt2)
    preview = np.full((40, 60, 3), (90, 140, 200), np.uint8) if preview is None else preview
    return _box(b"ftyp", b"crx " + b"\x00\x00\x00\x01" + b"crx isom") + _box(b"moov", _box(b"uuid", canon)) + _box(b"mdat", jpeg(preview))


def make_raf(preview: np.ndarray | None = None, orientation: int = 1) -> bytes:
    """RAF header pointing at an embedded JPEG that carries its own Exif."""
    preview = np.full((30, 45, 3), (60, 160, 90), np.uint8) if preview is None else preview
    exif = Image.Exif()
    exif[0x010F] = "FUJIFILM"
    exif[0x0110] = "X-T5"
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.fromarray(preview).save(buf, "JPEG", quality=90, exif=exif.tobytes())
    pj = buf.getvalue()
    header = b"FUJIFILMCCD-RAW 0201FF383501" + b"\x00" * (84 - 28)
    header += struct.pack(">II", 100, len(pj)) + b"\x00" * (100 - 92)
    return header + pj
