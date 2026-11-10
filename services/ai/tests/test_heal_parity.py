"""Browser / Python healing parity (the Python side).

apps/web/src/engine/__fixtures__/heal-parity.json holds a crop of a Skin Perfection result with
repairs in it, four heal and clone strokes across those repairs and the Python result. The browser
test (apps/web/src/engine/heal.parity.test.ts) runs heal.ts on the same input and must match it
within one 8-bit level. This test makes sure the expected pixels are still what the Python engine
produces, so a change on either side fails one of the two tests.

Regenerate after an intended change to either implementation (then check both tests):
    python -m scripts.heal_parity_fixture
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.engines.local import apply_heal_strokes
from app.schemas import HealStroke
from scripts.heal_parity_fixture import FIXTURE, python_heal, unpack


@pytest.fixture(scope="module")
def fx():
    d = json.loads(FIXTURE.read_text())
    w, h = d["width"], d["height"]
    return d, unpack(d["input"], h, w), unpack(d["expected"], h, w)


def test_the_fixture_still_matches_the_python_engine(fx):
    d, inp, expected = fx
    assert np.array_equal(python_heal(inp, d["strokes"]), expected)


def test_the_strokes_cross_the_repairs(fx):
    d, inp, expected = fx
    changed = np.any(inp != expected, axis=2)
    yy, xx = np.mgrid[: inp.shape[0], : inp.shape[1]]
    for m in d["repairs"]:
        inside = (xx + 0.5 - m["x"]) ** 2 + (yy + 0.5 - m["y"]) ** 2 <= m["r"] ** 2
        assert changed[inside].mean() > 0.5


def test_rounding_between_strokes_is_the_only_difference_from_the_browser(fx):
    # The browser keeps 8-bit pixels and rounds after every stroke; the engine keeps floats and
    # rounds once. Doing the same in Python stays within one level of the engine's result.
    d, inp, expected = fx
    img = inp.astype(np.float32) / 255.0
    for s in d["strokes"]:
        img = apply_heal_strokes(img, [HealStroke(**s)])
        img = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0
    per_stroke = (img * 255.0 + 0.5).astype(np.uint8)
    diff = np.abs(per_stroke.astype(int) - expected.astype(int))
    assert diff.max() <= 1
    assert (diff > 0).mean() < 0.01
