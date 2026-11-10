"""Professional Skin Finish engine (stage 0d, engines/finish.py): phase 1 tests.

Off means untouched; every control changes the rendered pixels in the band it claims; contours,
complexion and everything outside the skin are kept; texture never drops below its floor.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app import pipeline
from app.engines import finish as FIN
from app.pipeline import render
from app.schemas import Adjustments, SkinFinishSettings
from qa.finish_eval import contour_scene
from tests.perfection_fixtures import make_scene, oval_face, oval_mask

TONES = ("I", "II", "III", "IV", "V", "VI")


def st(smoothing=80, texture=60, pores=45, strength=100, **kw) -> SkinFinishSettings:
    return SkinFinishSettings(enabled=True, strength=strength, smoothing={"amount": smoothing},
                              texturePreservation={"amount": texture}, porePreservation={"amount": pores}, **kw)


@pytest.fixture(scope="module")
def plain():
    rgb, _, F = make_scene(size=700, tone="IV", seed=11, with_natural=False, marks=False)
    return rgb, oval_mask(700, 700), F


def lum(img):
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0])


def bands(img, mask, F):
    L = lum(img)
    return FIN.band_energy(L, mask, F, FIN.relief_area(L, mask, F))


# --- off is off --------------------------------------------------------------------------------

def test_defaults_are_off():
    s = Adjustments().skinFinish
    assert s.enabled is False
    assert (s.smoothing.amount, s.texturePreservation.amount, s.porePreservation.amount, s.strength) == (55, 75, 65, 100)


def test_render_with_finish_off_never_calls_the_engine(monkeypatch):
    rgb, _, _ = make_scene(size=500, tone="III", seed=3)
    rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
    face = [oval_face(500, 500)]
    adj = Adjustments(skinSmoothing=40, skinPerfection={"enabled": True})
    before = render(rgb8, adj, face)

    def boom(*a, **k):
        raise AssertionError("finish ran while off")

    monkeypatch.setattr(pipeline, "finish_photo", boom)
    # Off with non-default sliders, and on at strength 0: neither may touch the photo.
    off = adj.model_copy(update={"skinFinish": SkinFinishSettings(enabled=False, smoothing={"amount": 100})})
    zero = adj.model_copy(update={"skinFinish": SkinFinishSettings(enabled=True, strength=0)})
    assert np.array_equal(render(rgb8, off, face), before)
    assert np.array_equal(render(rgb8, zero, face), before)


def test_engine_disabled_returns_input(plain):
    rgb, mask, F = plain
    out, _ = FIN.finish_skin(rgb, mask, SkinFinishSettings(enabled=False), F)
    assert out is rgb


# --- every control does what it says ----------------------------------------------------------

def test_smoothing_reduces_bumps_and_blotches_monotonically(plain):
    rgb, mask, F = plain
    b0 = bands(rgb, mask, F)
    prev = None
    for amount in (0, 30, 60, 100):
        out, _ = FIN.finish_skin(rgb, mask, st(smoothing=amount), F)
        b = bands(out, mask, F)
        if amount == 0:
            assert b["bump"] == pytest.approx(b0["bump"], rel=0.02)
        else:
            assert b["bump"] < prev["bump"] and b["blotch"] < prev["blotch"]
        prev = b
    assert prev["bump"] < 0.6 * b0["bump"]


def test_pore_preservation_keeps_pores(plain):
    rgb, mask, F = plain
    low, _ = FIN.finish_skin(rgb, mask, st(pores=0), F)
    high, _ = FIN.finish_skin(rgb, mask, st(pores=100), F)
    b0, bl, bh = bands(rgb, mask, F), bands(low, mask, F), bands(high, mask, F)
    assert bl["pore"] < 0.7 * b0["pore"]
    assert bh["pore"] > 0.9 * b0["pore"]  # pores kept even at strong smoothing
    assert bh["bump"] < 0.8 * b0["bump"]  # while the bumps are still smoothed


def test_texture_preservation_keeps_grain(plain):
    rgb, mask, F = plain
    low, _ = FIN.finish_skin(rgb, mask, st(texture=0), F)
    high, _ = FIN.finish_skin(rgb, mask, st(texture=100), F)
    assert bands(high, mask, F)["fine"] > bands(low, mask, F)["fine"] * 1.1


def test_switching_a_control_off_changes_its_band(plain):
    rgb, mask, F = plain
    on, _ = FIN.finish_skin(rgb, mask, st(), F)
    b_on = bands(on, mask, F)
    for name, band in (("smoothing", "bump"), ("texturePreservation", "fine"), ("porePreservation", "pore")):
        s = st()
        getattr(s, name).enabled = False
        off, _ = FIN.finish_skin(rgb, mask, s, F)
        b_off = bands(off, mask, F)
        assert abs(b_off[band] / b_on[band] - 1.0) > 0.05, (name, b_on[band], b_off[band])


def test_overall_strength_blends_the_finish(plain):
    rgb, mask, F = plain
    full, _ = FIN.finish_skin(rgb, mask, st(texture=0), F)
    half, _ = FIN.finish_skin(rgb, mask, st(texture=0, strength=50), F)
    none, _ = FIN.finish_skin(rgb, mask, st(texture=0, strength=0), F)
    assert none is rgb
    d_full = lum(full) - lum(rgb)
    d_half = lum(half) - lum(rgb)
    sel = (mask > 0.5) & (np.abs(d_full) > 0.5)
    assert float(np.median(d_half[sel] / d_full[sel])) == pytest.approx(0.5, abs=0.08)


# --- what must not change ----------------------------------------------------------------------

@pytest.mark.parametrize("tone", TONES)
def test_contours_and_complexion_kept_on_every_tone(tone):
    src, mask, F = contour_scene(tone, 40 + TONES.index(tone), size=800)
    _, rep = FIN.finish_skin(src, mask, st(smoothing=100, pores=0, texture=0), F)
    assert rep.edge_kept is not None and rep.edge_kept >= 0.95, rep.edge_kept
    dL, da, db = rep.complexion_shift
    assert abs(dL) < 0.5 and abs(da) < 0.3 and abs(db) < 0.3, rep.complexion_shift
    # The plain Gaussian split (no edge weight) softens the same contours: the test can tell.
    _, g = FIN.finish_skin(src, mask, st(smoothing=100, pores=0, texture=0), F, method="gauss")
    assert g.edge_kept < rep.edge_kept - 0.02


def test_darker_skin_is_not_lightened():
    for tone in ("V", "VI"):
        rgb, _, F = make_scene(size=600, tone=tone, seed=5, with_natural=False, marks=False)
        mask = oval_mask(600, 600)
        out, _ = FIN.finish_skin(rgb, mask, st(smoothing=100, pores=0), F)
        sel = mask > 0.5
        assert lum(out)[sel].mean() - lum(rgb)[sel].mean() < 0.3


def test_pixels_outside_the_mask_are_bit_identical(plain):
    rgb, _, F = plain
    mask = np.zeros(rgb.shape[:2], np.float32)
    mask[200:500, 200:500] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    out, _ = FIN.finish_skin(rgb, mask, st(), F)
    outside = mask <= 1e-3
    assert np.array_equal(out[outside], rgb[outside])
    assert np.abs(out[mask > 0.5] - rgb[mask > 0.5]).max() > 0


def test_dark_cores_are_protected():
    src, mask, F = contour_scene("III", 7, size=800)
    out, _ = FIN.finish_skin(src, mask, st(smoothing=100, pores=0, texture=0), F)
    L0, L1 = lum(src), lum(out)
    core = L0 < np.percentile(L0[mask > 0.5], 1)  # the nostril-like ellipse
    assert np.abs(L1 - L0)[core].mean() < 0.3


# --- texture restoration ---------------------------------------------------------------------

def test_texture_restoration_refills_a_flattened_patch():
    rgb, _, F = make_scene(size=700, tone="III", seed=9, with_natural=False, marks=False)
    mask = oval_mask(700, 700)
    flat = rgb.copy()
    flat[250:420, 250:420] = cv2.GaussianBlur(rgb, (0, 0), 6)[250:420, 250:420]  # a waxy repair
    patch = np.zeros(mask.shape, bool)
    patch[280:390, 280:390] = True

    def fine(img):
        L = lum(img)
        hi = L - cv2.GaussianBlur(L, (0, 0), FIN.S_PORE * F)
        return float(np.sqrt(np.mean(hi[patch] ** 2)))

    kept, rep = FIN.finish_skin(flat, mask, st(smoothing=0, texture=100), F)
    _, rep0 = FIN.finish_skin(flat, mask, st(smoothing=0, texture=0), F)
    assert rep.restored_share > 0 and rep0.restored_share == 0
    assert fine(kept) > 2.0 * fine(flat)
    assert fine(kept) < 1.3 * fine(rgb)  # back to the person's own level, not beyond it
    again, _ = FIN.finish_skin(flat, mask, st(smoothing=0, texture=100), F)
    assert np.array_equal(kept, again)  # seeded: the same photo always gives the same grain


# --- in the renderer -------------------------------------------------------------------------

def test_render_finish_changes_skin_not_background():
    rgb, _, _ = make_scene(size=600, tone="V", seed=21, with_natural=False, marks=False, face_share=0.4)
    rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
    rgb8[:, :60] = (30, 140, 40)  # a green background strip outside the face
    face = oval_face(600, 600)
    base = render(rgb8, Adjustments(), [face])
    fin = render(rgb8, Adjustments(skinFinish=st()), [face])
    assert np.array_equal(fin[:, :60], base[:, :60])
    assert np.abs(fin.astype(int) - base.astype(int))[200:400, 200:400].max() > 0


def test_report_lists_the_measures(plain):
    rgb, mask, F = plain
    _, rep = FIN.finish_skin(rgb, mask, st(), F)
    d = rep.as_dict()
    for k in ("edgeKept", "textureKept", "complexionShift", "bandsBefore", "bandsAfter", "ms", "maskShare"):
        assert k in d
    assert 0.5 < d["textureKept"] <= 1.05
