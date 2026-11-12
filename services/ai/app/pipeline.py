"""Full-resolution renderer. A NumPy/OpenCV port of the WebGL2 shader chain in
apps/web/src/engine/shaders.ts; any change to one must be mirrored in the other.

Stages (all on float32 sRGB in [0, 1]):
  0. Per-effect controls (app/effects.py): each effect's slider, 1x/2x/3x multiplier, opacity,
     enable switch and the global retouch strength resolve to one effective value per effect.
  1. Three Gaussian bands at radii proportional to the long edge (s0 < s1 < s2). The skin
     stages cap s1/s2 at a share of the face width (TONE_SIGMA_FACE / BASE_SIGMA_FACE), which
     only binds when the face is small in the frame (full-length and group shots), and run on
     the skin's bounding box only. The preview keeps long-edge radii.
  2. Skin probability (YCbCr box) gated by the face-oval mask from landmarks, extended up to
     the hairline and gated by the person's own skin colour (face_skin_likelihood), which also
     keeps braids and hair lying across the face out.
  3-4. Natural retouch (retouch_engine.natural_skin_retouch):
     spot healing first (acne, pimples and dark spots found by multi-scale LoG are
     inpainted in the tone layer and re-textured from healthy skin, one at a time; the
     1x/2x/3x multiplier widens detection), then frequency separation into a tone layer
     (s1) and a detail layer: the blotch band (s1..s2) is suppressed and an adaptive
     median evens the tone layer, and the detail layer is re-injected by the Texture
     retention factor. The preview approximates spot healing and the median.
  5. Skin tone unification: chroma pulled toward the local average.
  5b. Boosted skin (2x/3x) and stray-hair cleanup (retouch_engine; the preview
      approximates these, so they are excluded from the parity guarantee).
  5c. Skin-tone guard (retouch_engine.skin_tone_guard): CIELAB chroma floor and hue
      lock so retouched skin never turns ashen or shifts away from its undertone.
  5d. Glasses glare (retouch_engine.reduce_glasses_glare) and 5e. clothing virtual iron and
      recolour (remove_clothing_wrinkles / shift_clothing_color, v0.6).
  6. Landmark-driven features: eyes, dark circles, teeth (correct_teeth: CIELAB whitening and
     stain removal on an enamel mask).
  7. L*a*b* colour match (covariance transform + tone curve when available).
  8. Global tone: exposure / white balance (linear light), highlights, shadows,
     contrast, saturation, vibrance; 8b. HSL mixer and colour grading (color_engine);
     8c. look LUT (color_engine.apply_lut); then sharpening and vignette.
  9. Depth-aware bokeh (retouch_engine.apply_depth_bokeh), when enabled and a depth map is given.
  0d. Professional Skin Finish (v0.14, engines/finish.py), after 0c and the heal strokes: edge-aware
     smoothing of the bump, blotch, pore and grain bands of face and body skin, texture
     restoration and verification. Off by default (then nothing in it runs).
  Pro tools (v0.10, engines/local.py): 0b. healing and clone strokes on the original pixels
  before stage 1 (right after the Skin Perfection engine, 0c, when that is on: the preview's order); 10. local retouch layers (frequency separation, dodge & burn, correction
  brushes with their own masks) after stage 8 and before the bokeh of stage 9.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .color import color_stats, linear_to_srgb, srgb_to_linear, transfer_lab
from .color_engine import apply_color_engine, apply_lut, decode_lut
from .effects import detector_multiplier, resolve_adjustments
from .engines.finish import FinishReport, finish_skin
from .engines.glasses import wears_glasses
from .engines.local import apply_heal_strokes, apply_local_layers
from .engines.noise import reduce_skin_noise
from .engines.perfection import PerfectionReport, _trim_heap, perfect_skin
from .engines.redness import reduce_redness
from .engines.under_eye import UNDER_REF_SIGMA, correct_dark_circles, soften_under_eye_texture
from .masking import FaceParse, apply_mask_edits, face_zone, hair_keep, protect_weight
from .retouch_engine import (
    apply_boosted_skin_retouch,
    apply_depth_bokeh,
    clothing_mask,
    correct_teeth,
    faces_in_crop,
    natural_skin_retouch,
    reduce_glasses_glare,
    remove_acne,
    remove_clothing_wrinkles,
    remove_dark_spots,
    remove_stray_hairs,
    shift_clothing_color,
    skin_tone_guard,
)
from .schemas import (
    Adjustments,
    ColorStats,
    FaceAnalysis,
    SkinFinishSettings,
    SkinPerfectionSettings,
)

S0, S1, S2 = 0.0012, 0.004, 0.012
SKIN_BRIGHT_GAIN = 0.12  # skin brightening at 100: x1.12 (about +5..+7 L* on mid skin)
SKIN_BRIGHT_KEEP = (0.75, 0.95)  # local tone luma above which skin highlights are left as shot
# Skin-stage blur radii never exceed these shares of the face width. In a headshot the face is
# 30-45% of the long edge and S1/S2 already sit below the caps; in a full-length or group shot
# the long-edge radii would be several times the size of a cheek and flatten facial shading.
TONE_SIGMA_FACE, BASE_SIGMA_FACE = 0.012, 0.036
# The landmark oval stops at mid-forehead: raise its top by this share of its half-height to
# reach the hairline, keeping only pixels that match the face's own skin colour.
FOREHEAD_LIFT = 0.4
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luma(c: np.ndarray) -> np.ndarray:
    return c @ LUMA


def cbcr(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return cb, cr


def from_ycbcr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb, cr = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


STRIP_WORKERS = 4  # at most; never more than OpenCV's thread count (the batch thermal profile sets it)


def _rows(fn, h: int, rows: int = 256) -> None:
    """fn(y0, y1) over horizontal strips, side by side on the CPU's cores. Only for per-pixel
    work that writes its own rows, so the result is the same as one strip after the other."""
    spans = [(y0, min(h, y0 + rows)) for y0 in range(0, h, rows)]
    n = min(STRIP_WORKERS, max(1, cv2.getNumThreads()), len(spans))
    if n <= 1:
        for a, b in spans:
            fn(a, b)
        return
    with ThreadPoolExecutor(max_workers=n) as ex:
        for f in [ex.submit(fn, a, b) for a, b in spans]:
            f.result()
    _trim_heap()


def skin_probability(c: np.ndarray) -> np.ndarray:
    """Per-pixel YCbCr skin box (0..1), computed in parallel strips on a large image."""
    if c.shape[0] * c.shape[1] < 2_000_000:
        return _skin_probability(c)
    out = np.empty(c.shape[:2], np.float32)

    def run(y0: int, y1: int) -> None:
        out[y0:y1] = _skin_probability(c[y0:y1])

    _rows(run, c.shape[0])
    return out


def _skin_probability(c: np.ndarray) -> np.ndarray:
    cb, cr = cbcr(c)
    return (
        smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.66, 0.71, cr))
        * smoothstep(0.27, 0.32, cb) * (1.0 - smoothstep(0.48, 0.53, cb))
    )


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sigma, 0.6), borderType=cv2.BORDER_REFLECT)


@dataclass
class FaceMasks:
    skin: np.ndarray
    eyes: np.ndarray
    under_eye: np.ndarray
    mouth: np.ndarray
    has_face: bool
    forehead: np.ndarray | None = None  # upper forehead above the landmark oval (gate by colour)
    face_width: float = 0.0  # widest face, px
    folds: np.ndarray | None = None  # nose and the nose-to-mouth folds (landmarks): the face's shape
    face_core: np.ndarray | None = None  # inside the landmark face oval, off its edge (no jaw, neck, body)


def build_masks(
    faces: list[FaceAnalysis], h: int, w: int, parse: FaceParse | None = None, include_body: bool = False,
) -> FaceMasks:
    """Rasterise normalised landmark polygons into feathered float masks.

    parse: optional face parsing (app/masking.py). Semantic face skin near a detected face
    extends the skin mask to the hairline, ears and jaw; hair removes skin where it lies across
    the face; include_body adds neck / chest / arm skin. Eyes, brows and lips stay cut out.
    """
    skin = np.zeros((h, w), np.float32)
    eyes = np.zeros((h, w), np.float32)
    under = np.zeros((h, w), np.float32)
    mouth = np.zeros((h, w), np.float32)
    forehead = np.zeros((h, w), np.float32)
    if not faces:
        return FaceMasks(skin, eyes, under, mouth, False)

    def poly(points) -> np.ndarray:
        return np.round(np.array(points, np.float32) * np.array([w, h], np.float32)).astype(np.int32)

    face_w = 1.0
    for f in faces:
        face_w = max(face_w, f.box[2] * w)
        r = f.regions
        oval = poly(r.faceOval)
        cv2.fillPoly(skin, [oval], 1.0)
        if len(oval) >= 3:
            cy = float(oval[:, 1].mean())
            lifted = oval.astype(np.float32).copy()
            up = lifted[:, 1] < cy
            lifted[up, 1] = cy - (cy - lifted[up, 1]) * (1.0 + FOREHEAD_LIFT)
            cv2.fillPoly(forehead, [np.round(lifted).astype(np.int32)], 1.0)
        cutouts = [p for p in (r.leftEye, r.rightEye, r.lips, r.leftBrow, r.rightBrow) if len(p) >= 3]
        if cutouts:
            pts = [poly(p) for p in cutouts]
            cv2.fillPoly(skin, pts, 0.0)
            cv2.polylines(skin, pts, True, 0.0, thickness=max(1, int(face_w * 0.012)))
        for region, canvas in ((r.leftEye, eyes), (r.rightEye, eyes), (r.leftUnderEye, under),
                               (r.rightUnderEye, under), (r.innerMouth, mouth)):
            if len(region) >= 3:
                cv2.fillPoly(canvas, [poly(region)], 1.0)

    folds = fold_mask(faces, h, w, face_w)

    # Overlap the oval by a few blur radii so the two feathered masks meet without a seam.
    inside = np.zeros((h, w), np.uint8)
    features = np.zeros((h, w), np.uint8)
    for f in faces:
        r = f.regions
        cv2.fillPoly(inside, [poly(r.faceOval)], 1)
        for region in (r.leftEye, r.rightEye, r.leftBrow, r.rightBrow, r.lips):
            if len(region) >= 3:
                cv2.fillPoly(features, [poly(region)], 1)
    inside = cv2.erode(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(face_w * 0.1) | 1,) * 2))
    # Facial skin for the finish's any-shape dark patches: the oval kept off its edge, so the
    # jaw line, the neck and the body (whose shading is shape, not pigment) never count.
    face_core = blur(inside.astype(np.float32), face_w * 0.02)
    features = cv2.dilate(features, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(face_w * 0.03) | 1,) * 2))
    forehead[(inside > 0) | (features > 0)] = 0.0
    skin = blur(skin, face_w * 0.025)
    forehead = blur(forehead, face_w * 0.025)
    if parse is not None:
        p = parse.resized(h, w)
        cut = 1.0 - blur(features.astype(np.float32), face_w * 0.006)
        zone = face_zone([(f.box[0] * w, f.box[1] * h, f.box[2] * w, f.box[3] * h) for f in faces], h, w)
        skin = np.maximum(skin, blur(p.face_skin * zone, face_w * 0.004) * cut)
        if include_body:
            skin = np.maximum(skin, blur(p.body_skin, face_w * 0.004) * cut)
        keep = hair_keep(blur(p.hair, face_w * 0.004))
        skin = skin * keep
        forehead = forehead * keep
    return FaceMasks(
        forehead=forehead,
        face_width=face_w,
        skin=skin,
        eyes=blur(eyes, face_w * 0.006),
        under_eye=blur(under, face_w * 0.02),
        mouth=blur(mouth, face_w * 0.004),
        has_face=True,
        folds=folds,
        face_core=face_core,
    )


FOLD_WIDTH, FOLD_NOSE = 0.06, 0.04  # x face width: the band along each fold; around the nose


def fold_mask(faces: list[FaceAnalysis], h: int, w: int, face_w: float) -> np.ndarray | None:
    """Where the face's own shape makes deep, elongated shadows that are not marks: the lower
    nose (alae, nostrils) and the nasolabial folds, drawn from each side of the nose to the
    mouth corner on that side and a little past it. None without nose and lip landmarks."""
    out = None
    for f in faces:
        r = f.regions
        if len(r.nose) < 3 or len(r.lips) < 3:
            continue
        if out is None:
            out = np.zeros((h, w), np.float32)
        nose = np.array(r.nose, np.float32) * [w, h]
        lips = np.array(r.lips, np.float32) * [w, h]
        fw = f.box[2] * w
        cv2.fillPoly(out, [np.round(nose).astype(np.int32)], 1.0)
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(FOLD_NOSE * fw) | 1,) * 2))
        width = max(1, int(FOLD_WIDTH * fw))
        cx = float(nose[:, 0].mean())

        def seg(p, q):
            cv2.line(out, tuple(int(v) for v in np.round(p)), tuple(int(v) for v in np.round(q)), 1.0, width)

        for side, eye in ((np.argmin, r.leftEye), (np.argmax, r.rightEye)):
            a, b = nose[side(nose[:, 0])], lips[side(lips[:, 0])]
            seg(a, b + 0.35 * (b - a))  # the fold runs on past the mouth corner
            if len(eye) >= 3:  # the side wall of the nose, up to the eye's inner corner
                e = np.array(eye, np.float32) * [w, h]
                inner = e[np.argmin(np.abs(e[:, 0] - cx))]
                if (inner[0] - cx) * (a[0] - cx) > 0:  # the eye on the same side as this ala
                    seg(a, inner)
    return None if out is None else blur(out, face_w * 0.015)


def face_skin_likelihood(low: np.ndarray, core: np.ndarray, region: np.ndarray) -> np.ndarray:
    """How well each pixel of `region` matches the face's own skin colour (CIELAB, 0..1).

    The fixed YCbCr skin box accepts braids, warm backgrounds and most hair on darker
    complexions, so the forehead extension is gated by the person's own skin instead:
    robust median and spread of L*a*b* inside the landmark oval, then a soft distance.
    """
    out = np.zeros(region.shape, np.float32)
    sel = core > 0.8
    ys, xs = np.nonzero(region > 1e-3)
    if sel.sum() < 64 or ys.size == 0:
        return out
    lab_core = cv2.cvtColor(np.ascontiguousarray(low[sel][None], np.float32), cv2.COLOR_RGB2Lab)[0]
    med = np.median(lab_core, axis=0)
    spread = 1.4826 * np.median(np.abs(lab_core - med), axis=0)
    spread = np.maximum(spread, np.array([4.0, 2.0, 2.0], np.float32))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = cv2.cvtColor(np.ascontiguousarray(low[y0:y1, x0:x1], np.float32), cv2.COLOR_RGB2Lab)
    d = (lab - med) / (2.5 * spread)
    # Darker than the face is the main hair cue; lighter (shine on the forehead) is allowed more.
    d[..., 0] = np.where(d[..., 0] > 0, d[..., 0] * 0.5, d[..., 0] * 1.6)
    dist = np.sqrt((d * d).sum(axis=-1))
    out[y0:y1, x0:x1] = 1.0 - smoothstep(1.0, 1.8, dist)
    return out


def _work_box(weight: np.ndarray, pad: float) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(weight > 1e-3)
    if ys.size == 0:
        return None
    h, w = weight.shape
    p = int(np.ceil(pad))
    return max(0, ys.min() - p), min(h, ys.max() + p + 1), max(0, xs.min() - p), min(w, xs.max() + p + 1)


def _skin_stages(
    c: np.ndarray, src: np.ndarray, skin_w: np.ndarray, faces: list[FaceAnalysis], adj: Adjustments,
    sigma_tone: float, sigma_base: float, bands: tuple[np.ndarray, np.ndarray] | None,
    spot_multiplier: int = 1,
    tool_multipliers: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Stages 3-5c on one crop (all arrays share its shape; faces normalised to it)."""
    c = c.copy()
    # 2a. Skin noise reduction (v0.9) first: grain and colour noise would otherwise read as
    #     texture to the spot detectors and the smoothing.
    if adj.skinNoise > 0:
        c = reduce_skin_noise(np.clip(c, 0.0, 1.0).astype(np.float32), skin_w, adj.skinNoise)
        bands = None
    # 2b. Separate skin tools (v0.7): acne first (red spots, bumps, then post-acne redness), then
    #     dark spots (isolated marks healed, uneven pigmentation evened on the tone layer).
    if adj.acneRemoval > 0:
        c = remove_acne(c, skin_w, faces, adj.acneRemoval, tool_multipliers[0], adj.protectBeautyMarks)
    if adj.darkSpotRemoval > 0:
        c = remove_dark_spots(
            c, skin_w, faces, adj.darkSpotRemoval, tool_multipliers[1], adj.protectBeautyMarks, adj.keepFreckles,
            sigma_tone=sigma_tone, sigma_base=sigma_base,
        )
    if adj.acneRemoval > 0 or adj.darkSpotRemoval > 0:
        bands = None  # the precomputed blurs no longer describe c
    # 3 + 4. Natural frequency-separation retouch: spots healed one by one (tone inpainted,
    #    healthy pores cloned in), then blotches smoothed and median-evened on the tone layer
    #    only, and the detail layer re-injected by the Texture retention factor.
    if adj.skinSmoothing > 0 or adj.blemishRemoval > 0:
        c = natural_skin_retouch(
            c, skin_w, faces,
            smoothing=adj.skinSmoothing, texture_retention=adj.skinTexture,
            spot_amount=adj.blemishRemoval, multiplier=int(spot_multiplier),
            protect_beauty_marks=adj.protectBeautyMarks, keep_freckles=adj.keepFreckles,
            sigma_tone=sigma_tone, sigma_base=sigma_base, bands=bands,
        )
    base = bands[1] if bands is not None else blur(src, sigma_base)

    # 4b. Redness reduction (v0.9): diffuse redness pulled to the person's surrounding skin tone.
    if adj.rednessReduction > 0:
        h_, w_ = c.shape[:2]
        face_w = max((f.box[2] * w_ for f in faces), default=0.0) or 0.4 * max(h_, w_)
        c = reduce_redness(np.clip(c, 0.0, 1.0).astype(np.float32), skin_w, adj.rednessReduction, sigma_tone=sigma_tone, face_width=face_w)

    # 5. Skin tone unification.
    if adj.skinToneUnify > 0:
        y = luma(c)
        cb, cr = cbcr(c)
        cb2, cr2 = cbcr(base)
        t = (adj.skinToneUnify / 100.0) * 0.8 * skin_w
        c = from_ycbcr(y, cb + (cb2 - cb) * t, cr + (cr2 - cr) * t)

    # 5b. Boosted retouch.
    if adj.skinMultiplier > 1:
        c = apply_boosted_skin_retouch(np.clip(c, 0.0, 1.0).astype(np.float32), skin_w, float(adj.skinMultiplier))
    # 5c. Skin-tone guard: CIELAB a*/b* lock against ashen or off-hue skin.
    if adj.skinToneGuard:
        c = skin_tone_guard(np.clip(c, 0.0, 1.0).astype(np.float32), src, base, skin_w)
    return c


def render(
    rgb8: np.ndarray,
    adj: Adjustments,
    faces: list[FaceAnalysis],
    source_stats: ColorStats | None = None,
    depth: np.ndarray | None = None,
    hair_prob: np.ndarray | None = None,
    clothes_prob: np.ndarray | None = None,
    parse: FaceParse | None = None,
    report: list | None = None,
) -> np.ndarray:
    """rgb8: HxWx3 uint8 (or uint16, e.g. a demosaiced camera RAW) sRGB -> HxWx3 uint8 sRGB.

    depth: optional relative depth (0 subject .. 1 far, any size) for the bokeh stage.
    hair_prob: optional hair probability (HxW) that keeps stray-hair cleanup out of the hairstyle.
    parse: optional face parsing (face skin, body skin, hair; app/masking.py) for the skin mask.
    report: optional list; the Skin Perfection report (engines/perfection.PerfectionReport) is
    appended to it when that engine runs.
    clothes_prob: optional garment probability (HxW, the segmenter's clothes class) for the
    clothing iron and recolour; without it a portrait prior below the chin is used.
    """
    # Per-effect enable / multiplier / opacity and global strength -> effective engine values.
    tool_multipliers = (detector_multiplier(adj, "acneRemoval"), detector_multiplier(adj, "darkSpotRemoval"))
    adj, spot_multiplier = resolve_adjustments(adj)
    src = rgb8.astype(np.float32) / (65535.0 if rgb8.dtype == np.uint16 else 255.0)
    # 0b. Healing and clone strokes (v0.10) on the original pixels: every later stage, and the
    #     sharpening detail, work from the cleaned photo. With Skin Perfection on they run right
    #     after it instead (stage 0c): the strokes were painted on the perfected photo the preview
    #     shows (POST /v1/perfection, then the browser's heal), so the export uses that order too.
    #     With the Skin Finish on (stage 0d, also engine-only) they run after the finish, for the
    #     same reason: the preview's source is then the engine's perfected and finished photo.
    perfection_on = adj.skinPerfection.enabled and adj.skinPerfection.strength > 0
    finish_on = adj.skinFinish.enabled and adj.skinFinish.strength > 0
    if adj.localRetouch.heal and not perfection_on and not finish_on:
        src = apply_heal_strokes(src, adj.localRetouch.heal)
    h, w = src.shape[:2]
    L = float(max(h, w))
    low0 = blur(src, S0 * L)
    # The coarser bands are only built when a stage reads them (each is a full float copy).
    bands: dict[float, np.ndarray] = {}

    def band(sigma_share: float) -> np.ndarray:
        if sigma_share not in bands:
            bands[sigma_share] = blur(src, sigma_share * L)
        return bands[sigma_share]
    sm = adj.skinMask
    masks = build_masks(faces, h, w, parse if sm.useSegmentation else None, sm.includeBodySkin)

    p = skin_probability(low0)
    if masks.has_face:
        # The person's own skin colour also keeps hair and braids that cross the face out.
        region = masks.skin if masks.forehead is None else np.maximum(masks.skin, masks.forehead)
        own = face_skin_likelihood(low0, masks.skin, region)
        skin = masks.skin * (0.35 + 0.65 * p) * (0.15 + 0.85 * own)
        if masks.forehead is not None:
            skin = np.maximum(skin, masks.forehead * own)
    else:
        skin = p * 0.8
    # User mask strokes: "add" paints skin in, "erase" protects an area from every skin tool.
    skin = apply_mask_edits(skin.astype(np.float32), sm)
    if masks.has_face and (adj.darkCircles > 0 or adj.underEyeTexture > 0):
        keep = protect_weight(sm, h, w)
        if keep is not None:
            masks.under_eye = masks.under_eye * keep
    # 0c. Skin Perfection engine (v0.13): detect, classify, correct and verify imperfections at
    #     full resolution on face and body skin. It edits the photo the rest of the chain starts
    #     from (like the healing brush), so every later stage sees the cleaned skin.
    sp = adj.skinPerfection
    if perfection_on:
        # Without a face the colour-only skin box would take in wood, sand and warm walls, so only
        # semantic body skin and the user's own Add-skin strokes count then.
        face_skin = skin if masks.has_face else apply_mask_edits(np.zeros_like(skin), sm)
        # The blurred copies of the photo are rebuilt from the cleaned skin below; freeing them
        # first keeps a 30 MP render inside its memory budget (each is a full float copy).
        low0 = p = own = region = None
        bands.clear()
        src, rep = perfect_photo(src, face_skin, masks, faces, parse, sp, protect_weight(sm, h, w))
        if report is not None:
            report.append(rep)
        if adj.localRetouch.heal and not finish_on:  # 0b, after the engine: the order the preview uses
            src = apply_heal_strokes(src, adj.localRetouch.heal)
        low0 = blur(src, S0 * L)
        bands.clear()
    # 0d. Professional Skin Finish (v0.14): texture finish and local corrections (shine, dark
    #     spots, tone, under-eye, dodge and burn) of face and body skin on the photo Skin
    #     Perfection left, before every older skin stage; the heal strokes follow it (see 0b).
    #     Off by default; when off nothing here runs, so the output is unchanged bit for bit.
    fs = adj.skinFinish
    if fs.enabled and fs.strength > 0:
        face_skin = skin if masks.has_face else apply_mask_edits(np.zeros_like(skin), sm)
        src, frep = finish_photo(src, face_skin, masks, parse, fs, protect_weight(sm, h, w))
        if report is not None:
            report.append(frep)
        if adj.localRetouch.heal:  # 0b, after the finish (see above): the order the preview uses
            src = apply_heal_strokes(src, adj.localRetouch.heal)
        low0 = blur(src, S0 * L)
        bands.clear()
    skin = skin[..., None]

    c = src.copy()
    skin_w = skin[..., 0]
    smooth = adj.skinSmoothing / 100.0

    # 3-5c run on the skin's bounding box only: in a full-length shot that is a few percent of
    # the frame, and every pixel outside it would come back unchanged anyway.
    skin_work = (
        adj.skinSmoothing > 0 or adj.blemishRemoval > 0 or adj.skinToneUnify > 0 or adj.skinMultiplier > 1
        or adj.acneRemoval > 0 or adj.darkSpotRemoval > 0 or adj.rednessReduction > 0 or adj.skinNoise > 0
    )
    if skin_work:
        sigma_tone, sigma_base = S1 * L, S2 * L
        if masks.has_face:
            sigma_tone = min(sigma_tone, TONE_SIGMA_FACE * masks.face_width)
            sigma_base = min(sigma_base, BASE_SIGMA_FACE * masks.face_width)
        box = _work_box(skin_w, 3.0 * sigma_base + 0.15 * max(masks.face_width, 32.0))
        if box is not None:
            y0, y1, x0, x1 = box
            local = faces_in_crop(faces, x0, y0, x1 - x0, y1 - y0, w, h)
            same_bands = sigma_tone == S1 * L and sigma_base == S2 * L
            c[y0:y1, x0:x1] = _skin_stages(
                c[y0:y1, x0:x1], src[y0:y1, x0:x1], skin_w[y0:y1, x0:x1], local, adj,
                sigma_tone, sigma_base,
                (band(S1)[y0:y1, x0:x1], band(S2)[y0:y1, x0:x1]) if same_bands else None,
                spot_multiplier, tool_multipliers,
            )
    # 5c2. Skin brightening (phase 3b): skin only, by the skin weight, so the background, hair
    #      and clothing are untouched; a gain (shading ratios kept) that fades out on the skin's
    #      own highlights, measured on the local tone so texture is not modulated.
    if adj.skinBrighten > 0 and masks.has_face:
        keep = 1.0 - smoothstep(*SKIN_BRIGHT_KEEP, luma(band(S1)))
        c = c * (1.0 + SKIN_BRIGHT_GAIN * (adj.skinBrighten / 100.0) * skin_w * keep)[..., None]
        del keep
    if adj.strayHairRemoval.enabled and adj.strayHairRemoval.strength > 0 and faces:
        c = remove_stray_hairs(np.clip(c, 0.0, 1.0).astype(np.float32), faces, adj.strayHairRemoval.strength, hair_prob)
    # 5d. Glasses glare (v0.6): reflection model inverted on the lenses, cores inpainted. Phase 3b:
    #     only on faces whose glasses were confidently detected (or every face with the override).
    glare_faces = [f for f in faces if wears_glasses(f, adj.glareForce)] if adj.glareReduction > 0 else []
    if glare_faces:
        c = reduce_glasses_glare(np.clip(c, 0.0, 1.0).astype(np.float32), glare_faces, adj.glareReduction)
    # 5e. Clothing: virtual iron, then recolour.
    cl = adj.clothing
    if cl.enabled and (cl.wrinkleStrength > 0 or cl.hueShift != 0 or cl.saturation != 0):
        garment = clothing_mask(src, faces, clothes_prob)
        if garment is not None:
            face_w = max((f.box[2] * w for f in faces), default=0.0) or None
            c = np.clip(c, 0.0, 1.0).astype(np.float32)
            if cl.wrinkleStrength > 0:
                c = remove_clothing_wrinkles(c, garment, cl.wrinkleStrength, cl.texturePreservation, face_w)
            c = shift_clothing_color(c, garment, cl.hueShift, cl.saturation)

    # 6. Face features.
    if masks.has_face:
        if adj.eyeBrighten > 0:
            e = (adj.eyeBrighten / 100.0) * masks.eyes[..., None]
            c = c * (1.0 + 0.35 * e) + 0.6 * e * (src - low0)
        if adj.darkCircles > 0:
            # v0.14: the shadow (band S1) is lifted as a whole toward the surrounding face skin,
            # so texture is kept and nothing grey comes in (engines/under_eye.correct_dark_circles).
            c = correct_dark_circles(
                np.clip(c, 0.0, 1.0).astype(np.float32), src, band(S1), masks.skin, masks.under_eye,
                adj.darkCircles, sigma_ref=UNDER_REF_SIGMA * masks.face_width,
            )
        # 6b. Under-eye texture (v0.9): the crease band (S0..S1, capped by face width) faded,
        #     pores and the tone layer kept.
        if adj.underEyeTexture > 0:
            sig_line = min(S1 * L, TONE_SIGMA_FACE * masks.face_width)
            c = soften_under_eye_texture(
                np.clip(c, 0.0, 1.0).astype(np.float32), masks.under_eye, adj.underEyeTexture,
                sigma_fine=min(S0 * L, 0.3 * sig_line), sigma_line=sig_line,
            )
        if adj.teethWhiten > 0 or adj.teethStainRemoval > 0:
            c = correct_teeth(np.clip(c, 0.0, 1.0).astype(np.float32), faces, adj.teethWhiten, adj.teethStainRemoval)

    c = np.clip(c, 0.0, 1.0)

    bands.clear()
    # 7-9 are per-pixel, so they run in horizontal strips (v0.12): the temporaries of the tone,
    # colour and LUT maths then cost a strip's worth of memory instead of several full-frame
    # float copies, with the same result.
    stats = None
    if adj.colorMatch is not None and adj.colorMatch.amount > 0:
        stats = source_stats or color_stats(src)
    lut = decode_lut(adj.lut.data, adj.lut.size) if adj.lut is not None and adj.lut.amount > 0 else None
    def finish(y0: int, y1: int) -> None:
        c[y0:y1] = _finish_strip(c[y0:y1], src[y0:y1], low0[y0:y1], skin[y0:y1], y0, h, w, adj, smooth, stats, lut)

    _rows(finish, h)

    # 10. Local retouch layers (v0.10): frequency separation, dodge & burn and correction
    #     brushes, each through its own mask (applied before the lens blur, as in the preview).
    if adj.localRetouch.layers:
        c = apply_local_layers(np.clip(c, 0.0, 1.0).astype(np.float32), adj.localRetouch, skin_w)

    # 9. Depth-of-field.
    b = adj.depthBokeh
    if b.enabled and depth is not None and b.blurRadius > 0:
        c = apply_depth_bokeh(
            np.clip(c, 0.0, 1.0).astype(np.float32), depth, b.bokehMultiplier / 100.0, b.blurRadius,
            b.apertureShape, b.highlightBoost,
        )

    return (np.clip(c, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


STRIP_ROWS = 256


def perfect_photo(
    src: np.ndarray, skin: np.ndarray, masks: FaceMasks, faces: list[FaceAnalysis], parse: FaceParse | None,
    settings: SkinPerfectionSettings, protect: np.ndarray | None = None,
) -> tuple[np.ndarray, PerfectionReport]:
    """Stage 0c: the Skin Perfection engine on the photo (float32 sRGB), with face skin from the
    renderer's skin weight and body skin from face parsing. Clothing, hair and background are
    outside both; the user's Protect strokes are already in `skin` and are re-applied to body
    skin here."""
    h, w = src.shape[:2]
    body = None
    if settings.bodyCoverage and parse is not None:
        p = parse.resized(h, w)
        fw = masks.face_width or 0.25 * max(h, w)
        body = blur(p.body_skin, fw * 0.003) * hair_keep(blur(p.hair, fw * 0.004))
        body = body * (0.4 + 0.6 * skin_probability(blur(src, S0 * max(h, w))))
        if protect is not None:
            body = body * protect
    return perfect_skin(
        src.astype(np.float32, copy=False), skin, faces, settings, body_skin=body,
        under_eye=masks.under_eye if masks.has_face else None, face_skin=masks.skin if masks.has_face else skin,
    )


def finish_photo(
    src: np.ndarray, skin: np.ndarray, masks: FaceMasks, parse: FaceParse | None,
    settings: SkinFinishSettings, protect: np.ndarray | None = None, method: str = "edge",
) -> tuple[np.ndarray, FinishReport]:
    """Stage 0d: the Professional Skin Finish engine on the photo (float32 sRGB), with face skin
    from the renderer's skin weight and body skin from face parsing (gated by the skin colour box,
    hair kept out, the user's Protect strokes applied). Clothing, hair and background are outside
    both, and their pixels come back bit for bit."""
    h, w = src.shape[:2]
    fw = masks.face_width or 0.25 * max(h, w)
    body = None
    if settings.bodyCoverage and parse is not None:
        p = parse.resized(h, w)
        body = blur(p.body_skin, fw * 0.003) * hair_keep(blur(p.hair, fw * 0.004))
        body = body * (0.4 + 0.6 * skin_probability(blur(src, S0 * max(h, w))))
        if protect is not None:
            body = body * protect
    return finish_skin(src.astype(np.float32, copy=False), skin.astype(np.float32, copy=False), settings, fw,
                       body=body, method=method, under_eye=None if not masks.has_face else masks.under_eye if protect is None else masks.under_eye * protect,
                       folds=masks.folds if masks.has_face else None,
                       face_core=masks.face_core if masks.has_face and masks.face_core is not None else 0.0)


def _finish_strip(
    c: np.ndarray, src: np.ndarray, low0: np.ndarray, skin: np.ndarray, y0: int, h: int, w: int,
    adj: Adjustments, smooth: float, stats: ColorStats | None, lut: np.ndarray | None,
) -> np.ndarray:
    """Stages 7-9 (colour match, global tone, colour engine, LUT, sharpening, vignette) on rows
    y0.. of the frame. Every operation is per pixel, so strips give the full-frame result."""
    c = np.clip(c, 0.0, 1.0)

    # 7. Colour match.
    if stats is not None:
        c = transfer_lab(c, stats, adj.colorMatch.reference, adj.colorMatch.amount / 100.0)

    # 8. Global tone. Exposure and white balance in linear light.
    lin = srgb_to_linear(c)
    lin = lin * (2.0 ** (adj.exposure / 50.0))
    T, Ti = adj.temperature / 100.0, adj.tint / 100.0
    lin = lin * np.array([1.0 + 0.2 * T, 1.0 - 0.12 * Ti, 1.0 - 0.2 * T], np.float32)
    c = linear_to_srgb(lin)

    y = luma(c)[..., None]
    H, S = adj.highlights / 100.0, adj.shadows / 100.0
    c = c + 0.25 * H * smoothstep(0.5, 1.0, y) + 0.25 * S * (1.0 - smoothstep(0.0, 0.5, y))
    c = (c - 0.5) * (1.0 + 0.5 * adj.contrast / 100.0) + 0.5

    y = luma(c)[..., None]
    sat = (c.max(axis=-1) - c.min(axis=-1))[..., None]
    c = y + (c - y) * (1.0 + adj.saturation / 100.0)
    c = y + (c - y) * (1.0 + (adj.vibrance / 100.0) * (1.0 - np.clip(sat, 0.0, 1.0)))

    # 8b. HSL colour mixer and 3-way colour grading (app/color_engine.py).
    c = apply_color_engine(np.clip(c, 0.0, 1.0).astype(np.float32), adj.hsl, adj.colorGrading)
    # 8c. Look LUT (a reference grade fitted by batch_engine.fit_lut, or an imported .cube).
    if lut is not None:
        c = apply_lut(c, lut, adj.lut.amount / 100.0)

    if adj.sharpen > 0:
        c = c + (adj.sharpen / 100.0) * 1.2 * (src - low0) * (1.0 - 0.8 * skin * smooth)

    if adj.vignette != 0:
        yy, xx = np.mgrid[y0:y0 + c.shape[0], 0:w].astype(np.float32)
        dx = (xx + 0.5) / w - 0.5
        dy = (yy + 0.5) / h - 0.5
        d = np.sqrt(dx * dx + dy * dy) / np.sqrt(0.5)
        c = c * (1.0 + (adj.vignette / 100.0) * 0.6 * smoothstep(0.35, 1.0, d))[..., None]
    return c


def sharpness(rgb8: np.ndarray) -> float:
    """Variance of the Laplacian on a 1024px grey proxy (scale-normalised blur metric)."""
    h, w = rgb8.shape[:2]
    s = min(1.0, 1024.0 / max(h, w))
    grey = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    if s < 1:
        grey = cv2.resize(grey, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())
