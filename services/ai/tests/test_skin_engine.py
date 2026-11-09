"""Phase 4 skin engine tools (v0.9): redness reduction, skin noise reduction, under-eye texture.

Each test checks the tool does its job AND leaves what it must not touch: brightness and
pores for redness, pores and clean files for noise, pores and everything outside the
under-eye region for the line tool.
"""

import cv2
import numpy as np
from app.engines.common import _cbcr, _luma
from app.engines.noise import estimate_noise, reduce_skin_noise
from app.engines.redness import reduce_redness
from app.engines.under_eye import soften_under_eye_texture
from app.pipeline import S1, render
from app.schemas import Adjustments, FaceAnalysis, FaceRegions

SIZE = 320
L = float(SIZE)
SKIN = np.array([0.80, 0.60, 0.50], np.float32)
yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def blur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def pores(seed=1, sigma=1.6, amp=0.03):
    p = blur(np.random.default_rng(seed).normal(0, 1, (SIZE, SIZE)).astype(np.float32), sigma)
    return p * (amp / p.std())


def base_skin(seed=1):
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[:] = SKIN
    return np.clip(img + pores(seed)[..., None], 0, 1).astype(np.float32)


def disc(cy, cx, r):
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))[..., None]


def fine(img, s=1.2):
    y = _luma(img)
    return y - blur(y, s)


ONES = np.ones((SIZE, SIZE), np.float32)

# --- redness ---------------------------------------------------------------------------------


def red_image():
    img = base_skin(2)
    img = img + disc(110, 100, 26) * np.array([0.05, -0.035, -0.03], np.float32)  # flushed cheek
    img = img + disc(220, 220, 26) * np.array([-0.05, -0.05, -0.05], np.float32)  # neutral shadow
    return np.clip(img, 0, 1).astype(np.float32)


def test_redness_reduces_red_area_only_and_keeps_brightness_and_pores():
    img = red_image()
    out = reduce_redness(img, ONES, 100, sigma_tone=S1 * L, face_width=0.8 * L)
    _, cr0 = _cbcr(img)
    _, cr1 = _cbcr(out)
    ref = cr0[10:40, 10:40].mean()  # even skin far from both patches
    red_before = cr0[100:120, 90:110].mean() - ref
    red_after = cr1[100:120, 90:110].mean() - ref
    assert red_after < 0.4 * red_before
    # The neutral shadow is not red: barely changed.
    assert np.abs(out[210:230, 210:230] - img[210:230, 210:230]).mean() < 0.15 * np.abs(out[100:120, 90:110] - img[100:120, 90:110]).mean()
    # Luma never changes; the pore layer is intact.
    assert np.abs(_luma(out) - _luma(img)).max() < 1e-3
    assert np.corrcoef(fine(out).ravel(), fine(img).ravel())[0, 1] > 0.99


def test_redness_off_and_outside_skin():
    img = red_image()
    assert reduce_redness(img, ONES, 0, sigma_tone=S1 * L, face_width=0.8 * L) is img
    out = reduce_redness(img, np.zeros_like(ONES), 100, sigma_tone=S1 * L, face_width=0.8 * L)
    assert np.array_equal(out, img)


# --- noise -----------------------------------------------------------------------------------


def noisy(clean, sigma=0.02, seed=5):
    n = np.random.default_rng(seed).normal(0, sigma, clean.shape).astype(np.float32)
    return np.clip(clean + n, 0, 1).astype(np.float32)


def test_noise_reduction_removes_grain_and_colour_noise():
    clean = base_skin(3)
    img = noisy(clean)
    out = reduce_skin_noise(img, ONES, 100)
    err_in = np.abs(img - clean).mean()
    err_out = np.abs(out - clean).mean()
    assert err_out < 0.55 * err_in
    cb_n, cr_n = _cbcr(img - clean + 0.5)
    cb_o, cr_o = _cbcr(out - clean + 0.5)
    assert cr_o.std() < 0.5 * cr_n.std() and cb_o.std() < 0.5 * cb_n.std()


def test_noise_reduction_keeps_pores():
    clean = base_skin(3)
    out = reduce_skin_noise(noisy(clean), ONES, 100)
    # The pore pattern (1.6 px texture) survives: strong correlation with the clean pores.
    band = lambda x: blur(_luma(x), 1.0) - blur(_luma(x), 4.0)  # noqa: E731
    assert np.corrcoef(band(out).ravel(), band(clean).ravel())[0, 1] > 0.9
    assert band(out).std() > 0.7 * band(clean).std()


def test_noise_reduction_barely_touches_a_clean_photo():
    clean = base_skin(4)
    out = reduce_skin_noise(clean, ONES, 100)
    img = noisy(clean)
    assert np.abs(out - clean).mean() < 0.35 * np.abs(reduce_skin_noise(img, ONES, 100) - img).mean()
    level = lambda x: estimate_noise(fine(x, 0.9), *(c - blur(c, 0.9) for c in _cbcr(x)), ONES)  # noqa: E731
    assert level(noisy(clean)) > 4 * level(clean)
    # Strong pores alone do not read as noise (they carry no chroma).
    rough = np.clip(SKIN + pores(9, sigma=0.8, amp=0.05)[..., None], 0, 1).astype(np.float32)
    assert level(rough) < 0.25 * level(noisy(clean))


# --- under-eye texture ----------------------------------------------------------------------

REGION = np.zeros((SIZE, SIZE), np.float32)
REGION[100:200, 60:260] = 1.0
REGION = blur(REGION, 3)


def eye_skin(seed=6):
    """Under-eye skin: fine pores (about 1 px) under creases that are several pixels wide."""
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[:] = SKIN
    return img + pores(seed, sigma=0.8, amp=0.02)[..., None]


def lined():
    img = eye_skin().copy()
    for y in (130, 150, 170):  # creases, 4 px wide, softened edges
        img[y:y + 4, 70:250] -= 0.06
    return np.clip(img, 0, 1).astype(np.float32)


def under_eye(img, amount=100):
    return soften_under_eye_texture(img, REGION, amount, sigma_fine=1.5, sigma_line=6.0)


def line_depth(img):
    y = blur(_luma(img), 1.0)
    return float(y[[126, 137, 146, 157, 166, 177], 80:240].mean() - y[[131, 151, 171], 80:240].mean())


def test_under_eye_fades_lines_inside_region_only():
    img = lined()
    out = under_eye(img)
    assert line_depth(out) < 0.5 * line_depth(img)
    outside = REGION < 1e-3
    assert np.abs(out[outside] - img[outside]).max() < 1e-4


def test_under_eye_keeps_pores_and_scales_with_amount():
    img = eye_skin(7)
    out = under_eye(img)
    # Pores (finest band) between the lines are largely kept.
    f0 = fine(img, 0.8)[110:190, 80:240]
    f1 = fine(out, 0.8)[110:190, 80:240]
    assert f1.std() > 0.75 * f0.std()
    lines = lined()
    d40 = line_depth(under_eye(lines, 40))
    d100 = line_depth(under_eye(lines, 100))
    assert line_depth(lines) > d40 > d100


# --- through the renderer: 1x/2x/3x, switch and Protect --------------------------------------


def face_with_under_eyes() -> FaceAnalysis:
    t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    oval = [(0.5 + 0.4 * np.cos(a), 0.5 + 0.45 * np.sin(a)) for a in t]
    sq = lambda x0, y0, x1, y1: [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # noqa: E731
    return FaceAnalysis(
        box=(0.1, 0.05, 0.8, 0.9), landmarkCount=478, eyeOpenness=0.8,
        regions=FaceRegions(
            faceOval=oval, leftEye=sq(0.55, 0.32, 0.75, 0.38), rightEye=sq(0.25, 0.32, 0.45, 0.38),
            leftUnderEye=sq(0.22, 0.4, 0.78, 0.56), rightUnderEye=[], innerMouth=[], lips=[], leftBrow=[], rightBrow=[],
        ),
    )


def u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def test_new_tools_follow_multiplier_switch_and_protect():
    src = u8(lined())
    f = [face_with_under_eyes()]
    diff = lambda adj: np.abs(render(src, adj, f).astype(int) - src).sum()  # noqa: E731
    base = diff(Adjustments())
    one = diff(Adjustments(underEyeTexture=40))
    three = diff(Adjustments(underEyeTexture=40, effects={"underEyeTexture": {"multiplier": 3}}))
    off = diff(Adjustments(underEyeTexture=40, effects={"underEyeTexture": {"enabled": False}}))
    assert three > one > base and off == base
    # A Protect stroke over the whole under-eye region leaves it untouched.
    protect = {"feather": 0, "strokes": [{"mode": "erase", "size": 0.2, "points": [0.3, 0.48, 0.7, 0.48]}]}
    out = render(src, Adjustments(underEyeTexture=100, skinMask=protect), f)
    assert np.abs(out[140:170, 110:210].astype(int) - render(src, Adjustments(), f)[140:170, 110:210]).max() <= 1
    src2 = u8(noisy(red_image()))
    diff2 = lambda adj: np.abs(render(src2, adj, f).astype(int) - src2).sum()  # noqa: E731
    base2 = diff2(Adjustments())
    for key in ("rednessReduction", "skinNoise"):
        r1 = diff2(Adjustments(**{key: 30}))
        r3 = diff2(Adjustments(**{key: 30}, effects={key: {"multiplier": 3}}))
        off = diff2(Adjustments(**{key: 30}, effects={key: {"enabled": False}}))
        assert r3 > r1 > base2 and off == base2, key
