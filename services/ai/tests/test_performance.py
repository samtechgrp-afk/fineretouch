"""Phase 7 (v0.12): strip processing gives the full-frame result, lazy bands, memory gate."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from app import memory, pipeline
from app.schemas import Adjustments


def _photo(h: int = 300, w: int = 220) -> np.ndarray:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([150 + 60 * np.sin(xx / 23.0), 110 + 40 * np.cos(yy / 17.0), 90 + 30 * np.sin((xx + yy) / 31.0)], -1)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


EDITS = [
    {},
    {"exposure": 30, "contrast": 25, "vignette": -45, "sharpen": 35, "saturation": 15, "vibrance": 20, "highlights": -25, "shadows": 30, "temperature": 12, "tint": -6},
    {"skinSmoothing": 50, "blemishRemoval": 60, "vignette": 30, "sharpen": 20},
    {"colorMatch": {"amount": 70, "reference": {"mean": [60, 10, 15], "std": [18, 7, 9]}}, "hsl": {"orange": {"hue": 10, "saturation": -20, "luminance": 10}}},
]


@pytest.mark.parametrize("edit", EDITS)
def test_strips_match_one_strip(edit, monkeypatch):
    """The per-pixel stages give the same image in 7-row strips as in one full-frame pass."""
    rgb = _photo()
    adj = Adjustments(**edit)
    monkeypatch.setattr(pipeline, "STRIP_ROWS", 10_000)
    whole = pipeline.render(rgb, adj, [])
    monkeypatch.setattr(pipeline, "STRIP_ROWS", 7)
    strips = pipeline.render(rgb, adj, [])
    np.testing.assert_array_equal(whole, strips)


def test_coarse_bands_built_only_when_needed(monkeypatch):
    calls: list[float] = []
    real = pipeline.blur

    def counting(img, sigma):
        calls.append(sigma)
        return real(img, sigma)

    monkeypatch.setattr(pipeline, "blur", counting)
    pipeline.render(_photo(), Adjustments(exposure=20), [])
    assert len(calls) == 1  # only the fine band (sharpening detail / skin probability)


def test_estimate_scales_with_pixels():
    small, big = memory.render_estimate(1000, 1000), memory.render_estimate(6000, 4000)
    assert big > small
    assert 3.0e9 < big < 4.5e9  # ~24 MP camera file


def test_gate_limits_concurrent_renders():
    gate = memory.MemoryGate(budget=250)
    active, peak = [0], [0]
    lock = threading.Lock()

    def job():
        with gate.admit(100):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=job) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2
    assert gate.stats().in_flight == 0
    assert gate.stats().waited > 0


def test_gate_always_admits_one_oversized_render():
    gate = memory.MemoryGate(budget=10)
    with gate.admit(10_000):
        assert gate.stats().in_flight == 1


def test_budget_from_env(monkeypatch):
    monkeypatch.setenv("FINERETOUCH_RENDER_MEMORY_MB", "2048")
    assert memory.default_budget() == 2048 * 2**20
    monkeypatch.delenv("FINERETOUCH_RENDER_MEMORY_MB")
    total = memory.total_memory()
    assert total is None or memory.default_budget() == int(total * 0.6)
