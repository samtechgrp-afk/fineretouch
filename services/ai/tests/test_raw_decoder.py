"""Camera RAW decoding: detection, metadata, LibRaw demosaic, fallbacks and API wiring.

All inputs are synthetic (see raw_fixtures.py): a real Bayer DNG that LibRaw decodes, plus
container skeletons of CR2/NEF/ARW/PEF/ORF/RW2/CR3/RAF for detection and fallback paths.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import imageio, raw_engine
from app.main import app
from app.pipeline import render
from app.schemas import Adjustments

from .raw_fixtures import jpeg, make_cr3, make_dng, make_raf, make_tiff_raw

needs_rawpy = pytest.mark.skipif(raw_engine.rawpy is None, reason="rawpy not installed")
client = TestClient(app)


def as_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / (65535.0 if img.dtype == np.uint16 else 255.0)


def ramp(img: np.ndarray) -> np.ndarray:
    """Mean colour of the neutral ramp in the top half of the synthetic scene."""
    h = img.shape[0]
    return as_float(img)[2 : h // 2 - 2].reshape(-1, 3).mean(0)


# ---------------------------------------------------------------- detection


@pytest.mark.parametrize(
    ("data", "name", "expected"),
    [
        (make_dng(), None, "dng"),
        (make_tiff_raw("NIKON CORPORATION"), None, "nef"),
        (make_tiff_raw("SONY"), None, "arw"),
        (make_tiff_raw("PENTAX"), None, "pef"),
        (make_tiff_raw("Canon"), None, "cr2"),
        (make_tiff_raw("Canon", cr2_marker=True), None, "cr2"),
        (make_tiff_raw("OLYMPUS", magic=b"IIRO"), None, "orf"),
        (make_tiff_raw("Panasonic", magic=b"IIU\x00"), None, "rw2"),
        (make_cr3(), None, "cr3"),
        (make_cr3(), "IMG_0001.CR4", "cr3"),  # a .cr4 name is accepted; content decides
        (make_raf(), None, "raf"),
        (make_tiff_raw("Unknown Maker"), "DSC_1.NEF", "nef"),
    ],
)
def test_detects_raw_formats(data: bytes, name: str | None, expected: str) -> None:
    assert raw_engine.detect_raw_format(data, name) == expected


def test_ordinary_images_are_not_raw() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "TIFF")
    assert raw_engine.detect_raw_format(buf.getvalue(), "scan.tif") is None
    assert raw_engine.detect_raw_format(jpeg(np.zeros((8, 8, 3), np.uint8))) is None
    assert raw_engine.detect_raw_format(b"\x00" * 64, "IMG_0001.CR4") is None  # no such format


# ---------------------------------------------------------------- metadata


def test_reads_dng_exif() -> None:
    meta = raw_engine.read_raw_metadata(make_dng(orientation=8, iso=1600))
    assert meta.format == "dng"
    assert (meta.make, meta.model, meta.lens) == ("FineRetouch", "Synthetic DNG", "FR 85mm F1.4")
    assert meta.iso == 1600
    assert meta.exposure_time == pytest.approx(1 / 250) and meta.shutter == "1/250"
    assert meta.f_number == pytest.approx(2.8) and meta.focal_length == pytest.approx(85)
    assert meta.orientation == 8
    assert meta.color_space == "sRGB"
    assert meta.datetime_original == "2026:11:06 10:30:00"


def test_reads_cr3_canon_boxes() -> None:
    meta = raw_engine.read_raw_metadata(make_cr3(model="Canon EOS R6", orientation=6, iso=800))
    assert (meta.make, meta.model, meta.iso, meta.orientation) == ("Canon", "Canon EOS R6", 800, 6)
    assert meta.lens == "RF85mm F1.2 L USM" and meta.f_number == pytest.approx(5.6)


def test_reads_raf_exif_from_embedded_jpeg() -> None:
    meta = raw_engine.read_raw_metadata(make_raf(orientation=3))
    assert (meta.format, meta.make, meta.model, meta.orientation) == ("raf", "FUJIFILM", "X-T5", 3)


@pytest.mark.parametrize("junk", [b"", b"II*\x00\xff\xff\xff\xff", b"MM\x00*\x00\x00\x00\x08\xff\xff", b"FUJIFILMCCD-RAW" + b"\xff" * 80])
def test_metadata_never_raises_on_damaged_headers(junk: bytes) -> None:
    raw_engine.read_raw_metadata(junk, "broken.nef")


# ---------------------------------------------------------------- white balance maths


def test_kelvin_to_xy_follows_planckian_locus() -> None:
    x, y = raw_engine.kelvin_to_xy(6500)  # Planckian, i.e. slightly below D65 (0.3127, 0.3290)
    assert x == pytest.approx(0.3135, abs=0.001) and y == pytest.approx(0.3237, abs=0.001)
    x, y = raw_engine.kelvin_to_xy(2856)  # illuminant A lies on the locus
    assert x == pytest.approx(0.4476, abs=0.002) and y == pytest.approx(0.4074, abs=0.002)


def test_temperature_multipliers() -> None:
    from .raw_fixtures import SRGB_FROM_XYZ

    m = np.array(SRGB_FROM_XYZ)
    assert raw_engine.wb_multipliers_for_temperature(m, 6504) == pytest.approx([1, 1, 1, 1], abs=0.08)
    warm_light = raw_engine.wb_multipliers_for_temperature(m, 3200)
    assert warm_light[2] > 1.5 * warm_light[0]  # tungsten light needs a lot more blue gain
    tinted = raw_engine.wb_multipliers_for_temperature(m, 6504, tint=50)
    assert tinted[0] > 1.1 and tinted[2] > 1.1 and tinted[1] == 1
    assert raw_engine.wb_multipliers_for_temperature(np.zeros((4, 3)), 5000) is None


# ---------------------------------------------------------------- full decode


@needs_rawpy
def test_decodes_dng_to_16bit_rgb() -> None:
    res = raw_engine.decode_camera_raw(make_dng())
    assert res.ok and res.source == "demosaic" and res.format == "dng" and res.bits == 16
    assert res.image.dtype == np.uint16 and res.image.shape == (64, 96, 3)
    assert len(np.unique(res.image)) > 256  # genuinely more than 8 bits of tonal data
    r, g, b = ramp(res.image)
    assert abs(r - g) < 0.03 and abs(b - g) < 0.03  # neutral ramp stays neutral
    h = res.image.shape[0]
    patches = as_float(res.image)[h * 3 // 4, [16, 48, 80]]
    assert np.argmax(patches[0]) == 0 and np.argmax(patches[1]) == 1 and np.argmax(patches[2]) == 2
    assert res.metadata.width == 96 and res.metadata.height == 64
    assert res.metadata.camera_white_balance is not None


@needs_rawpy
def test_camera_white_balance_removes_cast() -> None:
    data = make_dng(as_shot_neutral=(0.55, 1.0, 0.8))
    as_shot = ramp(raw_engine.decode_camera_raw(data, use_camera_wb=True).image)
    daylight = ramp(raw_engine.decode_camera_raw(data, use_camera_wb=False).image)
    assert abs(as_shot[0] - as_shot[1]) < 0.03 and abs(as_shot[2] - as_shot[1]) < 0.03
    assert daylight[1] - daylight[0] > 0.08  # without the camera WB the green cast stays


@needs_rawpy
def test_temperature_and_tint_shift_colour() -> None:
    data = make_dng()
    warm = ramp(raw_engine.decode_camera_raw(data, temperature=8000).image)
    cool = ramp(raw_engine.decode_camera_raw(data, temperature=3500).image)
    assert warm[0] / warm[2] > cool[0] / cool[2]  # a lower Kelvin target renders bluer
    magenta = ramp(raw_engine.decode_camera_raw(data, tint=60).image)
    green = ramp(raw_engine.decode_camera_raw(data, tint=-60).image)
    assert magenta[1] / magenta[0] < green[1] / green[0]


@needs_rawpy
@pytest.mark.parametrize("algo", ["AHD", "DHD", "PPG", "VNG", "LINEAR"])
def test_demosaic_algorithms(algo: str) -> None:
    res = raw_engine.decode_camera_raw(make_dng(), demosaic=algo)
    assert res.source == "demosaic" and res.image.shape == (64, 96, 3)
    assert abs(ramp(res.image)[0] - ramp(res.image)[1]) < 0.03


@needs_rawpy
@pytest.mark.parametrize("space", ["srgb", "adobe", "prophoto"])
@pytest.mark.parametrize("highlights", ["clip", "blend", "reconstruct"])
def test_color_spaces_and_highlight_modes(space: str, highlights: str) -> None:
    res = raw_engine.decode_camera_raw(make_dng(), color_space=space, highlight_mode=highlights)
    assert res.ok and res.image.dtype == np.uint16


@needs_rawpy
def test_wider_gamuts_encode_red_less_saturated() -> None:
    data = make_dng()

    def red_saturation(space: str) -> float:
        px = as_float(raw_engine.decode_camera_raw(data, color_space=space).image)[56, 16]
        return float(px[0] - px[1:].max())

    assert red_saturation("srgb") > red_saturation("adobe") > red_saturation("prophoto")


def test_rejects_unknown_options() -> None:
    with pytest.raises(ValueError):
        raw_engine.decode_camera_raw(make_dng(), output_bps=12)
    if raw_engine.rawpy is not None:
        for kwargs in ({"demosaic": "magic"}, {"color_space": "cmyk"}, {"highlight_mode": "glow"}):
            with pytest.raises(ValueError):
                raw_engine.decode_camera_raw(make_dng(), **kwargs)


@needs_rawpy
@pytest.mark.parametrize(("orientation", "shape"), [(1, (64, 96, 3)), (3, (64, 96, 3)), (6, (96, 64, 3)), (8, (96, 64, 3))])
def test_applies_exif_orientation(orientation: int, shape: tuple[int, int, int]) -> None:
    res = raw_engine.decode_camera_raw(make_dng(orientation=orientation))
    assert res.image.shape == shape and res.metadata.orientation == orientation
    upright = raw_engine.decode_camera_raw(make_dng()).image
    expected = {1: upright, 3: upright[::-1, ::-1], 6: np.rot90(upright, -1), 8: np.rot90(upright, 1)}[orientation]
    assert np.abs(as_float(res.image) - as_float(expected)).mean() < 0.01


@needs_rawpy
def test_8bit_output_and_downscale() -> None:
    res = raw_engine.decode_camera_raw(make_dng(width=192, height=128), output_bps=8, max_long_edge=48)
    assert res.image.dtype == np.uint8 and max(res.image.shape[:2]) == 48
    assert res.image.shape[:2] == (32, 48)


@needs_rawpy
def test_accepts_a_path(tmp_path) -> None:
    p = tmp_path / "IMG_0042.DNG"
    p.write_bytes(make_dng())
    assert raw_engine.decode_camera_raw(p).source == "demosaic"


# ---------------------------------------------------------------- damaged files and fallbacks


def test_damaged_sensor_data_falls_back_to_embedded_preview() -> None:
    preview = np.zeros((24, 36, 3), np.uint8)
    preview[:, :18] = (220, 40, 40)
    data = make_tiff_raw("NIKON CORPORATION", orientation=6, preview=preview)
    res = raw_engine.decode_camera_raw(data, filename="DSC_0001.NEF")
    assert res.ok and res.source == "embedded_preview" and res.error
    assert res.image.dtype == np.uint16 and res.image.shape == (36, 24, 3)  # orientation 6 applied
    assert res.metadata.orientation == 6 and res.metadata.iso == 3200
    top = as_float(res.image)[2, 12]  # left half of the preview is now the top
    assert top[0] > 0.7 and top[1] < 0.35


def test_truncated_dng_still_yields_an_image() -> None:
    data = make_dng()
    res = raw_engine.decode_camera_raw(data[: len(data) - 4000])
    assert res.ok and res.image.ndim == 3


def test_preview_with_its_own_orientation_is_not_rotated_twice() -> None:
    res = raw_engine.decode_camera_raw(make_raf(orientation=6))
    assert res.source in ("embedded_preview", "demosaic")
    assert res.image.shape[:2] == (45, 30)


def test_garbage_returns_structured_placeholder() -> None:
    res = raw_engine.decode_camera_raw(b"\x13\x37" * 4096, filename="IMG_9999.CR3")
    assert not res.ok and res.source == "placeholder" and res.error
    assert res.image.dtype == np.uint16 and res.image.ndim == 3
    assert (res.image == 128 * 257).all()
    assert any("placeholder" in w for w in res.warnings)


def test_fallback_can_be_disabled() -> None:
    with pytest.raises(raw_engine.RawDecodeError):
        raw_engine.decode_camera_raw(make_tiff_raw("SONY"), fallback=False)


def test_finds_largest_baseline_jpeg() -> None:
    small, big = jpeg(np.zeros((8, 8, 3), np.uint8)), jpeg(np.full((40, 30, 3), 200, np.uint8))
    found = raw_engine.find_embedded_jpeg(b"junk" + small + b"\x00" * 10 + big + b"tail")
    assert found is not None and Image.open(io.BytesIO(found)).size == (30, 40)
    assert raw_engine.find_embedded_jpeg(b"\xff\xd8\xff\xc3\x00\x0b" + b"\x00" * 64) is None  # lossless JPEG


# ---------------------------------------------------------------- service integration


@needs_rawpy
def test_imageio_routes_raw_through_libraw() -> None:
    data = make_dng(orientation=6)
    assert imageio.decode(data).dtype == np.uint8
    hq = imageio.decode_hq(data)
    assert hq.dtype == np.uint16 and hq.shape == (96, 64, 3)
    assert imageio.oriented_size(data) == (64, 96)


def test_pipeline_accepts_16bit_input() -> None:
    rgb8 = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    adj = Adjustments(exposure=20, contrast=10)
    out8 = render(rgb8, adj, [])
    out16 = render(rgb8.astype(np.uint16) * 257, adj, [])
    assert np.abs(out8.astype(int) - out16.astype(int)).max() <= 1


@needs_rawpy
def test_render_endpoint_accepts_dng() -> None:
    res = client.post(
        "/v1/render",
        files={"file": ("IMG_0001.DNG", make_dng(), "image/x-adobe-dng")},
        data={"adjustments": json.dumps({"exposure": 10}), "format": "png"},
    )
    assert res.status_code == 200, res.text
    assert Image.open(io.BytesIO(res.content)).size == (96, 64)


@needs_rawpy
def test_raw_metadata_and_preview_endpoints() -> None:
    res = client.post("/v1/raw/metadata", files={"file": ("a.dng", make_dng(orientation=6), "application/octet-stream")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["format"] == "dng" and body["iso"] == 400 and body["shutter"] == "1/250"
    assert (body["width"], body["height"]) == (64, 96)

    res = client.post("/v1/raw/preview", files={"file": ("a.dng", make_dng(), "application/octet-stream")})
    assert res.status_code == 200 and res.headers["x-raw-source"] == "embedded_preview"
    assert Image.open(io.BytesIO(res.content)).format == "JPEG"
    res = client.post("/v1/raw/preview", files={"file": ("a.dng", make_dng(), "application/octet-stream")}, data={"mode": "demosaic"})
    assert res.status_code == 200 and res.headers["x-raw-source"] == "demosaic"

    res = client.post("/v1/raw/metadata", files={"file": ("a.jpg", jpeg(np.zeros((8, 8, 3), np.uint8)), "image/jpeg")})
    assert res.status_code == 415


def test_unreadable_raw_is_a_415_not_a_grey_render() -> None:
    res = client.post("/v1/render", files={"file": ("x.nef", make_tiff_raw("NIKON")[:40], "application/octet-stream")})
    assert res.status_code == 415
