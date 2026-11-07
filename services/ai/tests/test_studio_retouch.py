"""v0.6 pro studio retouch: glasses glare, teeth, clothing wrinkles and colour."""

import cv2
import numpy as np

from app.color import rgb_to_lab
from app.pipeline import render
from app.retouch_engine import (
    clothing_mask,
    correct_teeth,
    detect_glare,
    glasses_lens_roi,
    reduce_glasses_glare,
    remove_clothing_wrinkles,
    shift_clothing_color,
    teeth_mask,
)
from app.schemas import Adjustments, ClothingSettings, FaceAnalysis, FaceRegions

rng = np.random.default_rng(7)
H = W = 400


def ring(cx: float, cy: float, rx: float, ry: float, n: int = 24) -> list[tuple[float, float]]:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(cx + rx * np.cos(a)), float(cy + ry * np.sin(a))) for a in t]


def face(**regions) -> FaceAnalysis:
    base = dict(faceOval=ring(0.5, 0.5, 0.4, 0.45), leftEye=[], rightEye=[], leftUnderEye=[], rightUnderEye=[],
                innerMouth=[], lips=[], leftBrow=[], rightBrow=[], nose=[])
    base.update(regions)
    return FaceAnalysis(box=(0.1, 0.05, 0.8, 0.9), landmarkCount=478, eyeOpenness=0.8, regions=FaceRegions(**base))


def skin(h: int = H, w: int = W) -> np.ndarray:
    img = np.ones((h, w, 3), np.float32) * np.array([0.78, 0.6, 0.5], np.float32)
    img += rng.normal(0, 0.01, (h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 1)


def blob(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))


# --- glasses glare -----------------------------------------------------------------

EYE_L = ring(0.35, 0.4, 0.06, 0.025)
EYE_R = ring(0.65, 0.4, 0.06, 0.025)


def glare_scene():
    clean = skin()
    for cx in (0.35, 0.65):
        # Dark iris inside a pale eye opening, as seen through the lens.
        cv2.ellipse(clean, (int(cx * W), int(0.4 * H)), (24, 10), 0, 0, 360, (0.85, 0.82, 0.8), -1)
        cv2.circle(clean, (int(cx * W), int(0.4 * H)), 9, (0.25, 0.17, 0.12), -1)
    clean[int(0.4 * H) - 3: int(0.4 * H) - 1, int(0.65 * W) + 2: int(0.65 * W) + 4] = 1.0  # catchlight
    veil = 0.65 * blob(H, W, 0.42 * H, 0.35 * W, 28)  # hazy flash reflection over the left lens
    glare_rgb = np.array([0.97, 0.98, 1.0], np.float32)
    observed = clean * (1 - veil[..., None]) + glare_rgb * veil[..., None]
    return clean, observed.astype(np.float32), veil


def test_lens_roi_covers_both_eyes_and_more():
    roi = glasses_lens_roi([face(leftEye=EYE_L, rightEye=EYE_R)], H, W)
    assert roi[int(0.4 * H), int(0.35 * W)] == 1 and roi[int(0.4 * H), int(0.65 * W)] == 1
    assert roi[int(0.4 * H), int(0.35 * W) + 50] == 1  # wider than the eye opening
    assert roi[int(0.9 * H), int(0.5 * W)] == 0
    assert glasses_lens_roi([face()], H, W).max() == 0  # no eye landmarks, no lenses


def test_glare_is_removed_and_eyes_stay_visible():
    clean, observed, veil = glare_scene()
    f = [face(leftEye=EYE_L, rightEye=EYE_R)]
    out = reduce_glasses_glare(observed, f, strength=80)
    region = veil > 0.15
    before = np.abs(observed - clean)[region].mean()
    after = np.abs(out - clean)[region].mean()
    assert after < 0.35 * before, (before, after)
    # The iris behind the reflection is visible again: dark, not washed out.
    iy, ix = int(0.4 * H), int(0.35 * W)
    assert out[iy, ix].mean() < observed[iy, ix].mean() - 0.2
    # The catchlight in the other (glare-free) eye and everything outside the lenses is untouched.
    assert np.allclose(out[int(0.4 * H) - 2, int(0.65 * W) + 3], 1.0, atol=0.02)
    roi = glasses_lens_roi(f, H, W)
    outside = cv2.dilate(roi, np.ones((25, 25), np.uint8)) == 0
    assert np.abs(out - observed)[outside].max() < 1e-6


def test_no_glare_means_no_change():
    clean, _, _ = glare_scene()
    out = reduce_glasses_glare(clean, [face(leftEye=EYE_L, rightEye=EYE_R)], strength=100)
    assert np.abs(out - clean).max() < 1e-6


def test_detect_glare_skips_catchlights():
    crop = skin(120, 200)
    eyes = np.zeros((120, 200), np.uint8)
    cv2.ellipse(eyes, (100, 60), (30, 12), 0, 0, 360, 1, -1)
    crop[57:62, 98:103] = 1.0  # catchlight
    veil, _, _ = detect_glare(crop, np.ones((120, 200), np.float32), eyes, 320.0, 1.0)
    assert veil.max() < 1e-3


# --- teeth -------------------------------------------------------------------------

MOUTH = ring(0.5, 0.72, 0.12, 0.04, 32)


def teeth_scene():
    img = skin()
    cx, cy = int(0.5 * W), int(0.72 * H)
    cv2.ellipse(img, (cx, cy), (48, 16), 0, 0, 360, (0.72, 0.3, 0.32), -1)  # gums / lips: red
    cv2.rectangle(img, (cx - 40, cy - 9), (cx + 40, cy + 6), (0.86, 0.78, 0.55), -1)  # yellowish teeth
    for x in range(cx - 30, cx + 40, 12):
        cv2.line(img, (x, cy - 9), (x, cy + 6), (0.18, 0.12, 0.1), 1)  # gaps between teeth
    cv2.circle(img, (cx + 15, cy - 2), 3, (0.7, 0.58, 0.3), -1)  # stain
    enamel = rng.normal(0, 0.012, img.shape[:2]).astype(np.float32)[..., None]
    img[cy - 9: cy + 7, cx - 40: cx + 41] += enamel[cy - 9: cy + 7, cx - 40: cx + 41]
    return np.clip(img, 0, 1).astype(np.float32), (cx, cy)


def lab_at(img, ys, xs):
    return rgb_to_lab(img[ys, xs].reshape(-1, 3))


def test_teeth_mask_excludes_gums_and_gaps():
    img, (cx, cy) = teeth_scene()
    m = teeth_mask(img, [face(innerMouth=MOUTH)])
    assert m[cy, cx - 24] > 0.6  # enamel
    assert m[cy + 13, cx] < 0.2  # gum below the teeth
    assert m[cy, cx - 30] < 0.5  # gap line
    assert m[int(0.3 * H), cx] == 0


def test_teeth_whitening_lowers_yellow_and_keeps_texture():
    img, (cx, cy) = teeth_scene()
    f = [face(innerMouth=MOUTH)]
    out = correct_teeth(img, f, whiten=80)
    ys, xs = slice(cy - 6, cy + 4), slice(cx - 28, cx - 20)
    b0, b1 = lab_at(img, ys, xs), lab_at(out, ys, xs)
    assert b1[:, 2].mean() < b0[:, 2].mean() - 8  # less yellow
    assert b1[:, 2].mean() > 0  # but not blue
    assert b1[:, 0].mean() > b0[:, 0].mean()  # brighter
    # Enamel texture: the pixel-to-pixel detail survives.
    d0 = np.std(img[ys, xs, 1] - cv2.blur(img, (5, 5))[ys, xs, 1])
    d1 = np.std(out[ys, xs, 1] - cv2.blur(out, (5, 5))[ys, xs, 1])
    assert d1 > 0.7 * d0
    # Gums keep their colour; outside the mouth nothing changes.
    g0, g1 = lab_at(img, slice(cy + 12, cy + 14), slice(cx - 5, cx + 5)), lab_at(out, slice(cy + 12, cy + 14), slice(cx - 5, cx + 5))
    assert abs(g1[:, 1].mean() - g0[:, 1].mean()) < 2
    assert np.abs(out[: int(0.6 * H)] - img[: int(0.6 * H)]).max() < 1e-6


def test_stain_removal_evens_a_stain():
    img, (cx, cy) = teeth_scene()
    out = correct_teeth(img, [face(innerMouth=MOUTH)], stain_removal=100)
    stain = lab_at(img, slice(cy - 3, cy), slice(cx + 14, cx + 17))[:, 2].mean()
    fixed = lab_at(out, slice(cy - 3, cy), slice(cx + 14, cx + 17))[:, 2].mean()
    ref = lab_at(img, slice(cy - 6, cy + 4), slice(cx - 28, cx - 20))[:, 2].mean()
    assert abs(fixed - ref) < 0.6 * abs(stain - ref)


# --- clothing ----------------------------------------------------------------------


def fabric(h=360, w=360):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = 0.06 * np.sin(xx * 2 * np.pi / 3.0) * np.sin(yy * 2 * np.pi / 3.0)  # threads, 3 px
    wrinkles = 0.22 * np.sin((xx * 0.8 + yy * 0.6) * 2 * np.pi / 36.0)  # creases, 36 px period
    drape = 0.35 * (yy / h - 0.5)  # broad shadow from top to bottom
    log_y = np.log(0.45) + weave + wrinkles + drape
    shade = np.exp(log_y - np.log(0.45))
    base = np.array([0.2, 0.32, 0.6], np.float32)  # blue shirt
    return np.clip(base * shade[..., None], 0, 1).astype(np.float32), weave, wrinkles, drape


def band_energy(log_y, period, h=360, w=360):
    """Amplitude of the pattern at the given period (via projection on its sinusoid)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if period == "wrinkle":
        basis = np.sin((xx * 0.8 + yy * 0.6) * 2 * np.pi / 36.0)
    else:
        basis = np.sin(xx * 2 * np.pi / 3.0) * np.sin(yy * 2 * np.pi / 3.0)
    sl = (slice(40, -40), slice(40, -40))
    return float((log_y[sl] * basis[sl]).sum() / (basis[sl] ** 2).sum())


def log_luma(img):
    return np.log(np.maximum(img @ np.array([0.299, 0.587, 0.114], np.float32), 1e-3))


def test_virtual_iron_removes_creases_keeps_weave_drape_and_colour():
    img, *_ = fabric()
    mask = np.ones(img.shape[:2], np.float32)
    out = remove_clothing_wrinkles(img, mask, strength=100, texture_preservation=100, face_width=200)
    l0, l1 = log_luma(img), log_luma(out)
    assert band_energy(l1, "wrinkle") < 0.35 * band_energy(l0, "wrinkle")
    assert band_energy(l1, "weave") > 0.85 * band_energy(l0, "weave")
    # Drape: the broad top-to-bottom shading survives.
    top, bottom = l1[60:100, 60:300].mean(), l1[260:300, 60:300].mean()
    top0, bottom0 = l0[60:100, 60:300].mean(), l0[260:300, 60:300].mean()
    assert (bottom - top) > 0.7 * (bottom0 - top0)
    # Chromaticity is untouched: wrinkles are shading.
    c0 = img / np.maximum(img.sum(-1, keepdims=True), 1e-4)
    c1 = out / np.maximum(out.sum(-1, keepdims=True), 1e-4)
    assert np.abs(c1 - c0)[40:-40, 40:-40].max() < 0.02


def test_virtual_iron_stays_inside_the_mask_and_keeps_edges():
    img, *_ = fabric()
    mask = np.zeros(img.shape[:2], np.float32)
    mask[:, 180:] = 1.0
    out = remove_clothing_wrinkles(img, mask, strength=100, face_width=200)
    assert np.abs(out[:, :170] - img[:, :170]).max() < 1e-6
    # A seam (hard dark line) inside the garment survives.
    seam = img.copy()
    seam[:, 270:273] *= 0.25
    out2 = remove_clothing_wrinkles(seam, mask, strength=100, face_width=200)
    assert out2[100:260, 271].mean() < 0.6 * out2[100:260, 262].mean()


def test_clothing_mask_from_segmenter_and_fallback():
    img = skin()
    seg = np.zeros((H, W), np.float32)
    seg[300:, :] = 0.9
    m = clothing_mask(img, [], clothes_prob=seg)
    assert m is not None and m[350, 200] > 0.8 and m[100, 200] == 0
    assert clothing_mask(img, []) is None
    # Fallback: below the chin, not skin.
    img[340:, :] = (0.15, 0.2, 0.45)
    small_face = FaceAnalysis(box=(0.4, 0.2, 0.2, 0.3), landmarkCount=478, eyeOpenness=0.8,
                              regions=face().regions)
    m2 = clothing_mask(img, [small_face])
    assert m2 is not None and m2[380, 200] > 0.5 and m2[100, 200] < 0.05


def test_clothing_colour_shift_rotates_hue_and_spares_skin():
    img = skin()
    img[200:, :] = (0.2, 0.32, 0.6)
    mask = np.zeros((H, W), np.float32)
    mask[150:, :] = 1.0  # sloppy mask that also covers skin rows 150..200
    out = shift_clothing_color(img, mask, hue_shift=120)
    lab0, lab1 = rgb_to_lab(img[300:310, 100:110].reshape(-1, 3)), rgb_to_lab(out[300:310, 100:110].reshape(-1, 3))
    h0 = np.degrees(np.arctan2(lab0[:, 2], lab0[:, 1])).mean()
    h1 = np.degrees(np.arctan2(lab1[:, 2], lab1[:, 1])).mean()
    assert abs(((h1 - h0 + 180) % 360) - 180 - 120) < 3
    assert np.abs(lab1[:, 0] - lab0[:, 0]).max() < 0.5  # lightness kept
    assert np.abs(out[160:190] - img[160:190]).max() < 0.03  # skin rows barely move


# --- pipeline integration -------------------------------------------------------------


def test_render_runs_studio_stages():
    clean, observed, _ = glare_scene()
    img, _ = teeth_scene()
    img[: int(0.6 * H)] = observed[: int(0.6 * H)]
    rgb8 = (img * 255 + 0.5).astype(np.uint8)
    f = [face(leftEye=EYE_L, rightEye=EYE_R, innerMouth=MOUTH)]
    base = render(rgb8, Adjustments(), f)
    adj = Adjustments(glareReduction=80, teethWhiten=60, teethStainRemoval=50,
                      clothing=ClothingSettings(enabled=True, wrinkleStrength=80, hueShift=30))
    seg = np.zeros((H, W), np.float32)
    seg[360:, :] = 1.0
    out = render(rgb8, adj, f, clothes_prob=seg)
    assert np.abs(out.astype(int) - base.astype(int))[150:190, 110:170].mean() > 5  # glare
    assert np.abs(out.astype(int) - base.astype(int))[283:292, 170:190].mean() > 2  # teeth
    # Defaults change nothing compared to v0.5 behaviour.
    assert np.array_equal(render(rgb8, Adjustments.model_validate({}), f), base)
