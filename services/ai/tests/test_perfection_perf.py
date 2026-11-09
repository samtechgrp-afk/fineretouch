"""Phase 2E: the speed and memory changes must not change the result."""

import numpy as np

import app.engines.perfection as P
import app.pipeline as PL
from app.schemas import SkinPerfectionSettings
from tests.perfection_fixtures import make_scene, oval_face, oval_mask


def _run(img):
    h, w = img.shape[:2]
    return P.perfect_skin(img, oval_mask(w, h), [oval_face(w, h)], SkinPerfectionSettings(enabled=True))


def test_parallel_tiles_give_the_same_result_as_one_after_the_other(monkeypatch):
    img, _, _ = make_scene(size=1600, tone="VI", seed=3, density="dense")
    par, rep_par = _run(img)
    monkeypatch.setattr(P, "TILE_WORKERS", 1)
    seq, rep_seq = _run(img)
    assert rep_par.tiles > 1
    assert np.array_equal(par, seq)
    assert [(m.x, m.y, m.kind, m.status) for m in rep_par.marks] == [(m.x, m.y, m.kind, m.status) for m in rep_seq.marks]


def test_the_input_photo_is_never_modified():
    img, _, _ = make_scene(size=800, tone="III", seed=4)
    before = img.copy()
    _run(img)
    assert np.array_equal(img, before)


def test_fast_median_matches_numpy():
    rng = np.random.default_rng(0)
    for n in range(1, 40):
        v = (rng.random(n) * rng.random()).astype(np.float32)
        assert P._median(v) == float(np.median(v))


def test_parallel_strips_match_one_pass():
    c = np.random.default_rng(1).random((2200, 1000, 3)).astype(np.float32)
    assert np.array_equal(PL.skin_probability(c), PL._skin_probability(c))
