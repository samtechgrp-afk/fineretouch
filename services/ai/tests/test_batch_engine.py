"""v0.6 batch style sync: LUT fitting and .cube files, sync planning, SQLite queue, runner, API."""

import io
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import batch_engine as be
from app import main
from app.color import color_stats
from app.color_engine import apply_lut, decode_lut, encode_lut, identity_lut
from app.pipeline import render
from app.schemas import Adjustments, BatchJobSettings, BatchSyncSpec, ColorStats, LookLut, OutputSpec

rng = np.random.default_rng(11)


def grade(rgb: np.ndarray) -> np.ndarray:
    """A typical 'warm, punchy' grade: S-curve, warm shadows, cooler highlights, a little desat."""
    c = rgb.astype(np.float32)
    c = c + 0.35 * (c - 0.5) * (1 - np.abs(2 * c - 1))  # S-curve
    y = c @ np.array([0.299, 0.587, 0.114], np.float32)
    c = y[..., None] + (c - y[..., None]) * 0.85
    c[..., 0] += 0.04 * (1 - y)
    c[..., 2] -= 0.03 * (1 - y) - 0.02 * y
    return np.clip(c, 0, 1)


def photo(h=96, w=128, seed=0) -> np.ndarray:
    r = np.random.default_rng(seed)
    base = r.random((h // 8, w // 8, 3)).astype(np.float32)
    img = np.kron(base, np.ones((8, 8, 1), np.float32))
    return np.clip(img + r.normal(0, 0.03, img.shape).astype(np.float32), 0, 1)


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


# --- LUTs --------------------------------------------------------------------------


def test_identity_pair_fits_identity_lut():
    img = photo()
    lut = be.fit_lut(img, img, 9)
    assert np.abs(lut - identity_lut(9)).max() < 1e-4


def test_fitted_lut_reproduces_the_grade_on_other_photos():
    ref = photo(seed=1)
    lut = be.fit_lut(ref, grade(ref), 17)
    other = photo(seed=2)
    err = np.abs(apply_lut(other, lut) - grade(other))
    assert err.mean() < 0.012, err.mean()
    assert np.percentile(err, 99) < 0.06


def test_lut_extrapolates_smoothly_outside_the_reference_colours():
    # Reference holds only warm, mid-bright colours; a saturated blue must not jump.
    ref = np.clip(np.array([0.7, 0.5, 0.4], np.float32) + rng.normal(0, 0.05, (64, 64, 3)).astype(np.float32), 0, 1)
    lut = be.fit_lut(ref, grade(ref), 17)
    blue = np.full((4, 4, 3), (0.1, 0.2, 0.8), np.float32)
    out = apply_lut(blue, lut)
    assert np.abs(out - blue).max() < 0.2
    assert np.isfinite(lut).all() and lut.min() >= 0 and lut.max() <= 1


def test_lut_encode_decode_and_cube_round_trip():
    lut = be.fit_lut(photo(seed=3), grade(photo(seed=3)), 5)
    data = encode_lut(lut)
    back = decode_lut(data, 5)
    assert np.abs(back - lut).max() < 1 / 65535 + 1e-6
    look = LookLut(name="Warm", size=5, data=data)
    assert look.size == 5
    parsed, title = be.parse_cube(be.lut_to_cube(lut, "Warm"))
    assert title == "Warm" and np.abs(parsed - lut).max() < 2e-6
    with pytest.raises(ValueError):
        decode_lut(data, 6)


def test_parse_cube_domain_and_errors():
    n = 2
    lines = ["# comment", "LUT_3D_SIZE 2", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 2 2 2"]
    for b in (0, 1):
        for g in (0, 1):
            for r in (0, 1):
                lines.append(f"{2 * r} {2 * g} {2 * b}")
    lut, title = be.parse_cube("\n".join(lines))
    assert title == "Imported LUT" and np.allclose(lut, identity_lut(n))
    with pytest.raises(ValueError):
        be.parse_cube("LUT_1D_SIZE 4\n0 0 0\n")
    with pytest.raises(ValueError):
        be.parse_cube("LUT_3D_SIZE 2\n0 0 0\n")


def test_identity_lut_in_render_is_a_no_op():
    rgb8 = (photo() * 255).astype(np.uint8)
    look = LookLut(name="id", size=9, data=encode_lut(identity_lut(9)))
    a = render(rgb8, Adjustments(), [])
    b = render(rgb8, Adjustments(lut=look), [])
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


# --- sync planning -----------------------------------------------------------------


def stats(L, a, b) -> ColorStats:
    return ColorStats(mean=(L, a, b), std=(20, 5, 5))


def test_smart_sync_matches_the_typescript_reference_values():
    # Same inputs and expected numbers as packages/shared/src/lut.test.ts.
    hero = Adjustments(exposure=10, temperature=5, tint=-3, skinSmoothing=40)
    out = be.smart_sync_adjustments(hero, stats(55, 4, 12), stats(48, 6, 6))
    assert (out.exposure, out.temperature, out.tint) == (32, 55, -53)
    assert out.skinSmoothing == 40
    out2 = be.smart_sync_adjustments(hero, stats(55, 4, 12), stats(54, 4.5, 11))
    assert (out2.exposure, out2.temperature, out2.tint) == (13, 17, -15)


def test_plan_adjustments_modes():
    adj = Adjustments(exposure=12, saturation=20, skinSmoothing=50, teethWhiten=30)
    look = color_stats(grade(photo()))
    ref = color_stats(photo())
    target = (photo(seed=5) * 255).astype(np.uint8)
    copy = be.plan_adjustments(BatchJobSettings(adjustments=adj, sync=BatchSyncSpec(mode="copy")), target)
    assert copy == adj
    smart = be.plan_adjustments(BatchJobSettings(adjustments=adj, sync=BatchSyncSpec(mode="smart", referenceStats=ref)), target)
    assert smart.skinSmoothing == 50 and smart.saturation == 20
    pal = be.plan_adjustments(BatchJobSettings(adjustments=adj, sync=BatchSyncSpec(mode="palette", referenceLookStats=look, amount=80)), target)
    assert pal.exposure == 0 and pal.saturation == 0 and pal.skinSmoothing == 50 and pal.teethWhiten == 30
    assert pal.colorMatch is not None and pal.colorMatch.amount == 80


# --- discovery ---------------------------------------------------------------------


def test_discover_images_and_output_paths(tmp_path: Path):
    (tmp_path / "day1" / "ceremony").mkdir(parents=True)
    (tmp_path / ".cache").mkdir()
    for p in ["a.jpg", "day1/b.JPG", "day1/ceremony/c.cr3", "day1/ceremony/d.nef", "notes.txt", ".cache/x.jpg", "._a.jpg"]:
        (tmp_path / p).write_bytes(b"x")
    files = be.discover_images([tmp_path])
    names = sorted(f.relative_to(tmp_path).as_posix() for f, _ in files)
    assert names == ["a.jpg", "day1/b.JPG", "day1/ceremony/c.cr3", "day1/ceremony/d.nef"]
    flat = be.discover_images([tmp_path], recursive=False)
    assert [f.name for f, _ in flat] == ["a.jpg"]
    f, root = next(x for x in files if x[0].name == "c.cr3")
    assert be.output_path(f, root, tmp_path / "out", "jpeg", "FR_") == tmp_path / "out" / "day1" / "ceremony" / "FR_c.jpg"
    with pytest.raises(FileNotFoundError):
        be.discover_images([tmp_path / "missing"])


# --- queue -------------------------------------------------------------------------


def make_queue(tmp_path: Path, n=3) -> tuple[be.BatchQueue, str, list[Path]]:
    q = be.BatchQueue(tmp_path / "q.sqlite3")
    files = []
    for i in range(n):
        p = tmp_path / "in" / f"img{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
        files.append(p)
    job = q.create_job("t", [(p, p.parent) for p in files], tmp_path / "out", BatchJobSettings())
    return q, job, files


def test_queue_claim_complete_fail_and_progress(tmp_path: Path):
    q, job, files = make_queue(tmp_path)
    a = q.claim()
    assert a.source == str(files[0]) and a.attempts == 1
    assert q.progress(job)["status"] == "running"
    b = q.claim()
    assert b.source == str(files[1])
    q.complete(a, 120.0)
    q.fail(b, "boom")  # first failure: back to pending for one retry
    c = q.claim()
    assert c.source == str(files[1]) and c.attempts == 2
    q.fail(c, "boom again")
    d = q.claim()
    assert d.source == str(files[2])
    q.complete(d, 80.0)
    assert q.claim() is None
    p = q.progress(job)
    assert p["status"] == "done" and p["done"] == 2 and p["failed"] == 1 and p["percent"] == 100.0
    assert p["errors"][0]["error"] == "boom again" and p["avgMs"] == 100.0
    q.set_status(job, "retry")
    assert q.progress(job)["status"] == "queued" and q.claim().source == str(files[1])


def test_queue_pause_resume_cancel_and_recover(tmp_path: Path):
    q, job, _ = make_queue(tmp_path)
    q.set_status(job, "pause")
    assert q.claim() is None
    q.set_status(job, "resume")
    item = q.claim()
    assert item is not None
    # The app quits mid-render; on the next launch the item is pending again.
    q2 = be.BatchQueue(tmp_path / "q.sqlite3")
    assert q2.recover() == 1
    assert q2.progress(job)["pending"] == 3
    q2.set_status(job, "cancel")
    p = q2.progress(job)
    assert p["status"] == "cancelled" and p["cancelled"] == 3 and q2.claim() is None
    with pytest.raises(KeyError):
        q2.progress("nope")


def test_output_name_collisions_get_suffixes(tmp_path: Path):
    q = be.BatchQueue(":memory:")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.jpg").write_bytes(b"x")
    (tmp_path / "a" / "x.png").write_bytes(b"x")
    job = q.create_job("t", be.discover_images([tmp_path / "a"]), tmp_path / "out", BatchJobSettings())
    outs = sorted(Path(q.claim().output).name for _ in range(2))
    assert outs == ["x-2.jpg", "x.jpg"]
    assert q.progress(job)["total"] == 2


def test_thermal_profiles():
    assert be.thermal_profile("quiet", 8) == be.ThermalProfile(1, 4, 0.75)
    assert be.thermal_profile("balanced", 8).workers == 2
    assert be.thermal_profile("balanced", 4).workers == 1
    assert be.thermal_profile("performance", 16).workers == 4
    with pytest.raises(ValueError):
        be.thermal_profile("turbo")


def wait_for(fn, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.05)
    return False


def test_runner_drains_the_queue_and_survives_bad_files(tmp_path: Path):
    q, job, files = make_queue(tmp_path, 5)
    files[2].write_bytes(b"bad")

    def process(item, settings):
        if Path(item.source).read_bytes() == b"bad":
            raise ValueError("corrupt")
        be.write_atomic(item.output, b"ok")

    runner = be.BatchRunner(q, process, "performance")
    runner.start()
    try:
        assert wait_for(lambda: q.progress(job)["status"] == "done")
    finally:
        runner.stop()
    p = q.progress(job)
    assert p["done"] == 4 and p["failed"] == 1
    assert sorted(x.name for x in (tmp_path / "out").iterdir()) == ["img0.jpg", "img1.jpg", "img3.jpg", "img4.jpg"]


# --- HTTP API --------------------------------------------------------------------------


@pytest.fixture()
def local_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LOCAL_BATCH", True)
    monkeypatch.setenv("FINERETOUCH_BATCH_DB", str(tmp_path / "engine.sqlite3"))
    monkeypatch.setenv("FINERETOUCH_BATCH_PROFILE", "performance")
    monkeypatch.setattr(main, "_batch", None)
    yield TestClient(main.app)
    if main._batch is not None:
        main._batch[1].stop()
        main._batch[0].close()
    monkeypatch.setattr(main, "_batch", None)


def test_batch_api_end_to_end(local_batch: TestClient, tmp_path: Path):
    src = tmp_path / "shoot"
    src.mkdir()
    for i in range(3):
        (src / f"p{i}.jpg").write_bytes(jpeg_bytes(photo(seed=10 + i) * (0.7 + 0.1 * i)))
    ref = color_stats(photo(seed=10) * 0.8)
    body = {
        "name": "Wedding",
        "sources": [str(src)],
        "outputDir": str(tmp_path / "out"),
        "settings": {
            "adjustments": {"exposure": 5, "skinSmoothing": 30},
            "sync": {"mode": "smart", "referenceStats": ref.model_dump()},
            "output": {"format": "jpeg", "quality": 85, "prefix": "FR_"},
        },
    }
    r = local_batch.post("/v1/batch/jobs", json=body)
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["total"] == 3
    assert wait_for(lambda: local_batch.get(f"/v1/batch/jobs/{job['id']}").json()["status"] == "done")
    p = local_batch.get(f"/v1/batch/jobs/{job['id']}").json()
    assert p["done"] == 3, p
    outs = sorted((tmp_path / "out").iterdir())
    assert [o.name for o in outs] == ["FR_p0.jpg", "FR_p1.jpg", "FR_p2.jpg"]
    assert Image.open(outs[0]).format == "JPEG"
    listing = local_batch.get("/v1/batch/jobs").json()
    assert listing["jobs"][0]["id"] == job["id"]
    assert local_batch.post(f"/v1/batch/jobs/{job['id']}/explode").status_code == 422
    assert local_batch.post("/v1/batch/profile", data={"profile": "quiet"}).json()["workers"] == 1
    assert local_batch.delete(f"/v1/batch/jobs/{job['id']}").status_code == 200
    assert local_batch.get(f"/v1/batch/jobs/{job['id']}").status_code == 404


def test_batch_api_rejects_empty_and_missing_folders(local_batch: TestClient, tmp_path: Path):
    (tmp_path / "empty").mkdir()
    base = {"outputDir": str(tmp_path / "out")}
    assert local_batch.post("/v1/batch/jobs", json={**base, "sources": [str(tmp_path / "empty")]}).status_code == 422
    assert local_batch.post("/v1/batch/jobs", json={**base, "sources": [str(tmp_path / "nope")]}).status_code == 404


def test_batch_api_is_off_on_servers(monkeypatch):
    monkeypatch.setattr(main, "LOCAL_BATCH", False)
    r = TestClient(main.app).get("/v1/batch/jobs")
    assert r.status_code == 404


def test_lut_fit_and_parse_endpoints():
    client = TestClient(main.app)
    ref = photo(seed=4)
    r = client.post("/v1/lut/fit", files={"before": ("b.jpg", jpeg_bytes(ref)), "after": ("a.jpg", jpeg_bytes(grade(ref)))},
                    data={"size": "9", "name": "Warm look"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["size"] == 9 and body["name"] == "Warm look" and body["cube"].startswith('TITLE "Warm look"')
    lut = decode_lut(body["data"], 9)
    other = photo(seed=8)
    assert np.abs(apply_lut(other, lut) - grade(other)).mean() < 0.03
    parsed = client.post("/v1/lut/parse", files={"file": ("x.cube", body["cube"].encode())}).json()
    assert parsed["size"] == 9 and parsed["name"] == "Warm look"
    assert client.post("/v1/lut/parse", files={"file": ("x.cube", b"nonsense")}).status_code == 422
    # Render with the fitted look applies it.
    rgb = (other * 255).astype(np.uint8)
    out = render(rgb, Adjustments(lut=LookLut(**{k: body[k] for k in ("name", "size", "data", "amount")})), [])
    assert np.abs(out.astype(float) / 255 - grade(other)).mean() < 0.035


def test_output_spec_defaults_used_by_batch():
    assert BatchJobSettings().output == OutputSpec()


def test_lut_limits_match_the_shared_package():
    import re

    from app.schemas import LUT_MAX_SIZE

    shared = Path(__file__).resolve().parents[3] / "packages" / "shared" / "src" / "index.ts"
    m = re.search(r"export const LUT_MAX_SIZE = (\d+);", shared.read_text())
    assert m and int(m.group(1)) == LUT_MAX_SIZE
