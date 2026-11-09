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
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import color_stats, linear_to_srgb, srgb_to_linear, transfer_lab
from .color_engine import apply_color_engine, apply_lut, decode_lut
from .effects import detector_multiplier, resolve_adjustments
from .masking import FaceParse, apply_mask_edits, face_zone, hair_keep
from .retouch_engine import (
    apply_boosted_skin_retouch,
    apply_depth_bokeh,
    clothing_mask,
    correct_teeth,
    faces_in_crop,
    natural_skin_retouch,
    remove_acne,
    remove_dark_spots,
    reduce_glasses_glare,
    remove_clothing_wrinkles,
    remove_stray_hairs,
    shift_clothing_color,
    skin_tone_guard,
)
from .schemas import Adjustments, ColorStats, FaceAnalysis

S0, S1, S2 = 0.0012, 0.004, 0.012
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


def skin_probability(c: np.ndarray) -> np.ndarray:
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
    )


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
) -> np.ndarray:
    """rgb8: HxWx3 uint8 (or uint16, e.g. a demosaiced camera RAW) sRGB -> HxWx3 uint8 sRGB.

    depth: optional relative depth (0 subject .. 1 far, any size) for the bokeh stage.
    hair_prob: optional hair probability (HxW) that keeps stray-hair cleanup out of the hairstyle.
    parse: optional face parsing (face skin, body skin, hair; app/masking.py) for the skin mask.
    clothes_prob: optional garment probability (HxW, the segmenter's clothes class) for the
    clothing iron and recolour; without it a portrait prior below the chin is used.
    """
    # Per-effect enable / multiplier / opacity and global strength -> effective engine values.
    tool_multipliers = (detector_multiplier(adj, "acneRemoval"), detector_multiplier(adj, "darkSpotRemoval"))
    adj, spot_multiplier = resolve_adjustments(adj)
    src = rgb8.astype(np.float32) / (65535.0 if rgb8.dtype == np.uint16 else 255.0)
    h, w = src.shape[:2]
    L = float(max(h, w))
    low0 = blur(src, S0 * L)
    low1 = blur(src, S1 * L)
    low2 = blur(src, S2 * L)
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
    skin = skin[..., None]

    c = src.copy()
    skin_w = skin[..., 0]
    smooth = adj.skinSmoothing / 100.0

    # 3-5c run on the skin's bounding box only: in a full-length shot that is a few percent of
    # the frame, and every pixel outside it would come back unchanged anyway.
    skin_work = (
        adj.skinSmoothing > 0 or adj.blemishRemoval > 0 or adj.skinToneUnify > 0 or adj.skinMultiplier > 1
        or adj.acneRemoval > 0 or adj.darkSpotRemoval > 0
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
                (low1[y0:y1, x0:x1], low2[y0:y1, x0:x1]) if same_bands else None,
                spot_multiplier, tool_multipliers,
            )
    if adj.strayHairRemoval.enabled and adj.strayHairRemoval.strength > 0 and faces:
        c = remove_stray_hairs(np.clip(c, 0.0, 1.0).astype(np.float32), faces, adj.strayHairRemoval.strength, hair_prob)
    # 5d. Glasses glare (v0.6): reflection model inverted on the lenses, cores inpainted.
    if adj.glareReduction > 0 and faces:
        c = reduce_glasses_glare(np.clip(c, 0.0, 1.0).astype(np.float32), faces, adj.glareReduction)
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
            d = (adj.darkCircles / 100.0) * masks.under_eye
            y = luma(c)
            cb, cr = cbcr(c)
            cb2, cr2 = cbcr(low2)
            y = y + np.maximum(0.0, luma(low2) * 1.06 - y) * d
            c = from_ycbcr(y, cb + (cb2 - cb) * 0.5 * d, cr + (cr2 - cr) * 0.5 * d)
        if adj.teethWhiten > 0 or adj.teethStainRemoval > 0:
            c = correct_teeth(np.clip(c, 0.0, 1.0).astype(np.float32), faces, adj.teethWhiten, adj.teethStainRemoval)

    c = np.clip(c, 0.0, 1.0)

    # 7. Colour match.
    if adj.colorMatch is not None and adj.colorMatch.amount > 0:
        stats = source_stats or color_stats(src)
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
    if adj.lut is not None and adj.lut.amount > 0:
        c = apply_lut(c, decode_lut(adj.lut.data, adj.lut.size), adj.lut.amount / 100.0)

    if adj.sharpen > 0:
        c = c + (adj.sharpen / 100.0) * 1.2 * (src - low0) * (1.0 - 0.8 * skin * smooth)

    if adj.vignette != 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = (xx + 0.5) / w - 0.5
        dy = (yy + 0.5) / h - 0.5
        d = np.sqrt(dx * dx + dy * dy) / np.sqrt(0.5)
        c = c * (1.0 + (adj.vignette / 100.0) * 0.6 * smoothstep(0.35, 1.0, d))[..., None]

    # 9. Depth-of-field.
    b = adj.depthBokeh
    if b.enabled and depth is not None and b.blurRadius > 0:
        c = apply_depth_bokeh(
            np.clip(c, 0.0, 1.0).astype(np.float32), depth, b.bokehMultiplier / 100.0, b.blurRadius,
            b.apertureShape, b.highlightBoost,
        )

    return (np.clip(c, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def sharpness(rgb8: np.ndarray) -> float:
    """Variance of the Laplacian on a 1024px grey proxy (scale-normalised blur metric)."""
    h, w = rgb8.shape[:2]
    s = min(1.0, 1024.0 / max(h, w))
    grey = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    if s < 1:
        grey = cv2.resize(grey, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())
