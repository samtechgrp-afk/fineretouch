"""Skin Perfection engine (v0.13): detect, classify, correct and verify skin imperfections.

Why a new engine (see docs/SKIN_PERFECTION.md for the full audit): the v0.5-v0.12 spot healer
looked for marks on a copy shrunk until the face was 420 px wide, median-filtered the signal
(which deletes pore-sized blackheads), merged touching acne into one component that then failed
the size / roundness tests, blended every repair at 77% at 1x, never looked at its own result,
and covered the face only. On a GPU the export used a per-pixel shader approximation instead.

Pipeline (all stages real image processing; no stage is a placeholder):

  A. High-resolution analysis: the skin's bounding box is analysed with the face about
     WORK_FACE px wide (never upscaled), in overlapping tiles. Each tile owns the marks whose
     centre lies in its core, and its apron is wider than the largest mark plus its context
     ring, so marks on tile seams are neither missed nor counted twice.
  B/C. Skin coverage: the caller's face skin weight plus semantic body skin (neck, chest,
     shoulders, arms, hands, back, legs) from face parsing; eyes, brows, lips, nostrils, hair
     and anything erased with the Protect brush are excluded.
  D. Multi-scale detection: a scale-space of normalised LoG responses on two signals,
     darkness+redness relative to a robust local skin baseline (marks are left out of the
     baseline, so dense acne does not hide itself) and brightness (whiteheads). Each scale is
     thresholded in units of its own robust spread, so pores set the noise floor at the fine
     scales and blackheads must beat the pore texture around them. Candidates are scale-space
     maxima, so neighbouring pimples stay separate marks.
  E. Classification: size relative to the face, darkness, redness (a*), brownness (b*),
     edge sharpness, ring isotropy, Hessian anisotropy (lines and hair), pore-relative
     contrast and neighbour density classify each mark as acne, pimple, blackhead, whitehead,
     dark spot, post-acne mark, small dot or blemish, or as a protected natural feature:
     mole, freckle, birthmark or hair follicle. Each mark gets a confidence.
  F. Targeted correction at full resolution: heal marks (acne, pimples, blackheads,
     whiteheads, blemishes) get their tone layer filled from the skin around them (push-pull
     membrane) and texture cloned from the best-matching healthy patch nearby; pigment marks
     (dark spots, post-acne marks, small dots) only have their tone layer rebuilt, so the
     pores inside stay. Area corrections: post-acne redness halo, diffuse redness,
     discoloration, under-eye darkness, uneven texture and scars.
  G. Texture reconstruction: donor texture is mixed with the original pore band by Pore
     preservation, and a smooth fill is never left where Texture preservation asks for grain.
  H. Residual scan: the corrected image is scanned again (more sensitively, by Residual
     scanning intensity). Each mark is verified by measuring its contrast after correction;
     only marks that are actually gone count as corrected. Remaining marks are retried in the
     next pass, at most MAX_ATTEMPTS times each, and passes stop as soon as nothing is left to
     try, so the engine cannot loop or keep working the same area.
  I. Quality verification: counts (detected / corrected / remaining / protected / ignored),
     mean confidence and fine-texture retention around corrected marks.

Everything here is classical computer vision on top of the MediaPipe models FineRetouch
already ships (face landmarks, selfie multiclass segmentation); no new model or dependency.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import cv2
import numpy as np

from ..schemas import FaceAnalysis, PerfectionOverride, SkinPerfectionSettings
from .common import _cbcr, _ellipse, _from_ycbcr, _gauss, _luma, _masked_blur, _smoothstep
from .local import push_pull
from .redness import reduce_redness
from .skin import correct_pigmentation
from . import perfection_quality as Q
from .spots import face_width_px, faces_in_crop, feature_protection_mask

# --- A. analysis scale and tiling ----------------------------------------------------------
WORK_FACE = 1100.0  # analysis resolution: face width in px (old detector: 420)
TILE = 1024  # tile core, work px
NATIVE_SCALES = 3  # the finest scales run on full-resolution pixels when the face is shrunk for analysis ...
NATIVE_BELOW = 0.9  # ... by more than 10%
NATIVE_KOFS = 200  # kidx offset of native-pass candidates (their response statistics are their own)
# Marks are looked for at these LoG sigmas (share of face width F). Blob radius ~ sigma * sqrt 2,
# so the finest scale finds marks ~0.35% of the face wide (a blackhead), the coarsest ~6%.
SIGMAS = (0.0012, 0.002, 0.0032, 0.005, 0.008, 0.012, 0.018, 0.026)
BRIGHT_SIGMAS = SIGMAS[:5]  # whiteheads are small
RING = 2.2  # context ring radius in mark radii
BASE_CONTEXT = 0.03  # local skin baseline: masked blur at this share of F

# --- E. classification (sizes are diameters as a share of F; contrast in luma 0..1) ----------
TINY = 0.007  # blackhead / small dot size
MOLE_MIN_D, MOLE_MIN_REL = 0.013, 0.22  # beauty marks: big enough and much darker than the skin
FRECKLE_MAX_D, FRECKLE_MAX_REL = 0.022, 0.14
FRECKLE_NEIGHBOURS, FRECKLE_REACH = 4, 0.12  # freckles come in groups of similar marks
FOLLICLE_NEIGHBOURS, FOLLICLE_REACH = 5, 0.035  # stubble: a dense field of tiny dark dots
BIRTHMARK_D = 0.06  # larger, low-contrast patches are birthmarks (or shading), never "spots"
RED_A = 2.2  # a* above the skin that makes a mark "red" (acne family)
BROWN_B = 1.0  # b* above the skin that makes a dark mark "brown" (pigment)
MIN_REL_CONTRAST = 0.035  # below this relative contrast a mark is invisible at 100%
# Colourless marks need more contrast than red or brown ones (skin relief and shine are colourless).
BLEMISH_MIN_REL, BLEMISH_MAX_D = 0.07, 0.025
WHITEHEAD_MIN_REL, WHITEHEAD_MAX_D = 0.045, 0.011
BODY_MAX_D = 0.03  # largest body mark (share of F)
TINY_MIN_REL = 0.06  # a pore-sized dark dot must be at least this dark to be a blackhead / dot
MAX_CANDIDATES_PER_TILE = 6000
PORE_RATIO = 2.0  # a pore-sized mark must be this many times the typical pore's contrast (at 100% sensitivity)
# Tone fairness (v0.13 phase 2B). Marks are compared with this person's own skin texture, measured
# separately for dark texture (pores), bright texture (sheen between pores) and colour texture (a*),
# each with the measured sensor noise taken out. Deep skin has the same marks at the same relative
# contrast, but more relative noise and texture; one shared dark-pore reference held whiteheads and
# red acne there to a standard set by the pores.
# Tone-fair texture tests (bright texture, colour texture) are an alternative to the strict pore
# test, and only for a clearly significant mark outside lit texture (FAIR_MARGIN, FIELD_*):
BRIGHT_PORE_RELIEF = 0.5  # a bright dot beats bright texture with this much less margin than a dark dot beats pores
RED_REF_MIN = 1.5  # a* texture reference floor (a* units), so smooth even-toned skin does not make every red dot "strong"
CLEAR_RED_A = 2.0 * RED_A  # colour contrast only counts for the texture test when the mark is clearly red
LUMA_REF_MIN = 0.01  # luma texture reference floor (relative)
RIM_MIN_RATIO = 1.5  # a bright dot this much fainter than a dark mark right beside it is that mark's rim ...
FAIR_MARGIN = 2.0  # ... and only for a mark at least this many times over its detection threshold
FIELD_REACH, FIELD_DOTS = 0.1, 20  # this many light dots within this reach (share of F) = lit texture or shimmer
RIM_MAX_PORE = 3.0  # ... unless it clearly beats the bright texture (a rim is faint; a whitehead beside a spot is not)

# Ambiguity (v0.13 phase 2C). A mark that could be either a natural feature or an imperfection is
# never corrected automatically: it is reported for review with the reason.
FIELD_OUTLIER_SIZE, FIELD_OUTLIER_DARK = 1.5, 1.3  # a dot this much bigger AND darker than its stubble field
FRECKLE_OUTLIER_SIZE, FRECKLE_OUTLIER_DARK = 1.4, 1.8  # a spot this much bigger AND darker than its freckle group
HAIR_COHERENCE = 0.5  # orientation coherence around a beard-area dot that says "hair runs here"
FIELD_LIKE = 1.3  # a lone dot within this factor of a nearby stubble field's dots looks like one of them
CLUSTER_PEAK_RATIO = 1.5  # a large flat mark with 2+ marks inside this much stronger is a cluster, not one patch
REVIEW_REASONS = ("lowConfidence", "mayBeFreckle", "mayBeFollicle", "repairFailed")
FINE_RESIDUAL = 0.6  # the residual check also rescans the mark at this share of its radius

# --- F/G. correction ------------------------------------------------------------------------
SHARP_KINDS = ("blackhead", "whitehead", "smallDot", "darkSpot", "postAcne")  # edges that leave a ghost ring
SHARP_SMALL = 6.0  # a sharp mark up to this many pore scales across is all edge
HEAL_KINDS = ("acne", "pimple", "blackhead", "whitehead", "blemish", "manual")
PIGMENT_KINDS = ("darkSpot", "postAcne", "smallDot")
PROTECTED_KINDS = ("mole", "freckle", "birthmark", "follicle")
MULTIPLIER_SENS = {1: 0.0, 2: 10.0, 3: 18.0}  # sensitivity points added by 2x / 3x
MULTIPLIER_RADIUS = {1: 1.0, 2: 1.12, 3: 1.25}
MULTIPLIER_AREA = {1: 1.0, 2: 1.35, 3: 1.6}  # area tools (redness, tone, texture) scale up gently

# --- H. verification ------------------------------------------------------------------------
VERIFY_RATIO = 0.4  # a mark is gone when its contrast fell below this share of the original ...
MAX_ATTEMPTS = 2  # ... and no mark is corrected more than this many times
REVIEW_BAND = 0.15  # confidence this far below the threshold -> "needs review", not corrected
RESIDUE_REACH = 1.8  # a residual hit within this many radii of a corrected mark is that mark's residue ...
RESIDUE_GROW = 1.5  # ... and widens it by at most this factor
REPAIR_MARGIN = 1.0  # px added to every repair radius
RETRY_GROW = 1.35  # a retry after a failed quality check repairs this much wider (within RESIDUE_GROW)


def _fast_gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide blurs run on a shrunk copy (a 200 px kernel would cost seconds)."""
    if sigma <= 6.0:
        return _gauss(img, sigma)
    h, w = img.shape[:2]
    f = sigma / 3.0
    size = (max(1, round(w / f)), max(1, round(h / f)))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small = _gauss(small, 3.0 * min(size[0] / w, size[1] / h) * f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _fast_masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask[..., None] if img.ndim == 3 else mask
    num = _fast_gauss(np.ascontiguousarray(img * m, np.float32), sigma)
    den = _fast_gauss(np.ascontiguousarray(mask, np.float32), sigma)
    den = den[..., None] if img.ndim == 3 else den
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), img)


def _lowres(fn, image: np.ndarray, maps: list[np.ndarray], F: float, target_face: float = 320.0) -> np.ndarray:
    """Run a tone-scale correction fn(image, *maps, F) on a copy shrunk to a face ~target_face px
    wide and add its (smooth) change back at full size, so wide blurs stay cheap."""
    s = min(1.0, target_face / max(F, 1.0))
    if s >= 0.999:
        return fn(image, *maps, F)
    h, w = image.shape[:2]
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    sm = [cv2.resize(np.ascontiguousarray(m, np.float32), size, interpolation=cv2.INTER_AREA) for m in maps]
    delta = fn(small, *sm, F * s) - small
    out = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
    out += image  # one full-size array, not three (a 30 MP float copy is 345 MB)
    return np.clip(out, 0.0, 1.0, out=out)


@dataclass
class Mark:
    """One imperfection or protected feature, in full-resolution pixel coordinates."""

    id: int
    x: float
    y: float
    r: float  # radius, px
    kind: str
    confidence: float  # 0..1
    region: str  # "face" | "body"
    contrast: float  # relative contrast when first found
    status: str = "detected"  # corrected | remaining | review | protected | ignored
    attempts: int = 0
    residual: bool = False  # first found by a residual scan
    after: float | None = None  # relative contrast after the last correction
    r0: float | None = None  # radius when first found (residue may widen r up to RESIDUE_GROW x this)
    reason: str | None = None  # why a "review" mark needs a decision (REVIEW_REASONS)
    quality: dict | None = None  # the correction quality checks of its last repair (perfection_quality)
    fine0: float | None = None  # contrast at the finer residual scale, before any repair
    forced: bool = False  # the user chose Correct for it (its repair is kept even if it fails a check)
    bulk: bool = False  # "Correct all remaining" chose it: its repair must pass every check, or it is undone


@dataclass
class PerfectionReport:
    width: int
    height: int
    marks: list[Mark] = field(default_factory=list)
    passes: int = 0
    tiles: int = 0
    work_scale: float = 1.0
    face_width: float = 0.0
    texture_retention: float | None = None  # fine-band energy around corrected marks, after / before
    ms: float = 0.0
    areas: dict[str, float] = field(default_factory=dict)  # area corrections applied (strength 0..1)
    quality: dict[str, int] = field(default_factory=dict)  # final repairs failing each check (perfection_quality)

    def counts(self) -> dict[str, int]:
        st = [m.status for m in self.marks]
        detected = sum(s in ("corrected", "remaining", "review", "ignored") for s in st)
        return {
            "detected": detected,
            "corrected": st.count("corrected"),
            "remaining": st.count("remaining") + st.count("review"),
            "review": st.count("review"),
            "protected": st.count("protected"),
            "ignored": st.count("ignored"),
        }

    def to_json(self, max_marks: int = 3000) -> dict:
        marks = sorted(self.marks, key=lambda m: (m.status == "protected", -m.confidence))[:max_marks]
        L = float(max(self.width, self.height))
        conf = [m.confidence for m in self.marks if m.status != "protected"]
        by_kind: dict[str, int] = {}
        for m in self.marks:
            by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
        return {
            "width": self.width, "height": self.height, "passes": self.passes, "tiles": self.tiles,
            "workScale": round(self.work_scale, 4), "faceWidth": round(self.face_width, 1), "ms": round(self.ms),
            "counts": self.counts(), "byKind": by_kind,
            "meanConfidence": round(float(np.mean(conf)), 3) if conf else None,
            "textureRetention": None if self.texture_retention is None else round(self.texture_retention, 3),
            "areas": {k: round(v, 3) for k, v in self.areas.items()},
            "quality": dict(self.quality),
            "marks": [
                {**{k: v for k, v in asdict(m).items() if k not in ("x", "y", "r", "after", "contrast", "r0", "fine0", "forced", "bulk")},
                 "x": round(m.x / self.width, 5), "y": round(m.y / self.height, 5), "r": round(m.r / L, 5),
                 "contrast": round(m.contrast, 3), "after": None if m.after is None else round(m.after, 3),
                 "confidence": round(m.confidence, 3)}
                for m in marks
            ],
        }


# =============================================================================================
# A-E. Detection
# =============================================================================================


@dataclass
class _Cand:
    x: np.ndarray  # work px (within the analysis crop)
    y: np.ndarray
    sigma: np.ndarray  # work px
    score: np.ndarray  # response / threshold-spread (dimensionless)
    bright: np.ndarray  # bool: from the brightness signal (whitehead family)
    dark: np.ndarray  # relative darkness at the centre (luma / skin luma)
    red: np.ndarray  # a* above the skin
    brown: np.ndarray  # b* above the skin
    iso: np.ndarray  # ring isotropy 0..1
    aniso: np.ndarray  # Hessian eigenvalue ratio 0..1 (1 = round)
    pore: np.ndarray  # contrast / typical pore contrast on the tile (dark pore reference, as in PR #13)
    edge: np.ndarray  # structure gradient around the mark (relative)
    body: np.ndarray  # bool: outside the face skin
    resp: np.ndarray  # raw LoG response
    med: np.ndarray  # tile median of that scale's response on skin
    spread: np.ndarray  # tile robust spread of that scale's response on skin
    kidx: np.ndarray  # scale index (bright scales offset by 100)
    shine: np.ndarray  # 0..1 specular / oily highlight zone
    inskin: np.ndarray  # share of the context ring that lies on skin
    cpore: np.ndarray  # redness / typical a* texture peak on the tile (colour contrast against this person's skin)
    tpore: np.ndarray  # contrast / typical texture peak of the same polarity (bright marks: bright texture)
    coh: np.ndarray  # 0..1 orientation coherence of the skin around the mark (hair shafts run one way)


_BOOL_FIELDS, _INT_FIELDS = ("bright", "body"), ("kidx",)


def _cat(parts: list[_Cand]) -> _Cand:
    if not parts:
        return _Cand(**{f: np.zeros(0, bool if f in _BOOL_FIELDS else np.int32 if f in _INT_FIELDS else np.float32)
                        for f in _Cand.__dataclass_fields__})
    return _Cand(*[np.concatenate([getattr(p, f) for p in parts]) for f in _Cand.__dataclass_fields__])


def _take(c: _Cand, sel: np.ndarray) -> _Cand:
    return _Cand(*[getattr(c, f)[sel] for f in _Cand.__dataclass_fields__])


def _robust(v: np.ndarray) -> tuple[float, float]:
    if v.size == 0:
        return 0.0, 1e-5
    med = float(np.median(v))
    return med, 1.4826 * float(np.median(np.abs(v - med))) + 1e-6


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[np.clip(np.round(y).astype(np.int64), 0, h - 1), np.clip(np.round(x).astype(np.int64), 0, w - 1)]


def _robust_baseline(v: np.ndarray, core: np.ndarray, sigma: float, outlier: np.ndarray | None = None) -> np.ndarray:
    """Masked blur of v over skin, recomputed with the marks left out (darker / redder outliers)."""
    w = core.astype(np.float32)
    base = _fast_masked_blur(v, w, sigma)
    if outlier is not None:
        w2 = w * (1.0 - _smoothstep(0.5, 1.0, outlier(base)))
        base = _fast_masked_blur(v, w2, sigma)
    return base


def _noise_sigma(v: np.ndarray, core: np.ndarray) -> float:
    """Per-pixel sensor noise of v on core (Immerkaer's Laplacian-difference estimator, with the
    median instead of the mean so pores and marks do not count as noise)."""
    r = np.abs(cv2.filter2D(np.ascontiguousarray(v, np.float32), -1, _NOISE_K, borderType=cv2.BORDER_REFLECT))[core]
    return float(np.median(r)) / (0.6745 * 6.0) if r.size else 0.0


_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
_NOISE_FIELD: list[np.ndarray] = []
_NOISE_PEAK: dict[float, float] = {}
_NOISE_LOCK = threading.Lock()


def _noise_peak(sigma: float) -> float:
    """Typical (75th percentile) positive peak of unit white noise blurred at sigma."""
    key = round(sigma, 2)
    with _NOISE_LOCK:  # tiles run in threads
        return _noise_peak_locked(key)


def _noise_peak_locked(key: float) -> float:
    if key not in _NOISE_PEAK:
        if not _NOISE_FIELD:
            _NOISE_FIELD.append(np.random.default_rng(7).normal(0.0, 1.0, (256, 256)).astype(np.float32))
        g = cv2.GaussianBlur(_NOISE_FIELD[0], (0, 0), sigmaX=key, borderType=cv2.BORDER_REFLECT)
        pk = (g >= cv2.dilate(g, np.ones((3, 3), np.uint8))) & (g > 0)
        _NOISE_PEAK[key] = float(np.percentile(g[pk], 75))
    return _NOISE_PEAK[key]


def _texture_ref(f: np.ndarray, core: np.ndarray, noise_peak: float, floor: float, default: float) -> float:
    """The typical texture peak of f on core (upper quartile of its positive local maxima), with
    the part sensor noise explains taken out in quadrature (never below half of the raw value)."""
    peaks = (f >= cv2.dilate(f, np.ones((3, 3), np.uint8))) & core & (f > 0)
    if peaks.sum() <= 50:
        return default
    raw = float(np.percentile(f[peaks], 75))
    return max(floor, math.sqrt(max(raw * raw - noise_peak * noise_peak, 0.25 * raw * raw)))


TILE_WORKERS = 4  # at most; never more than OpenCV's thread count (the batch thermal profile sets it)
TILE_BUDGET_MB = 900.0  # working memory for tiles in flight (measured: about 200 bytes per tile pixel)
TILE_BYTES_PER_PX = 200.0


def _trim_heap() -> None:
    """Give memory freed by worker threads back to the system. glibc keeps each thread's freed
    tile buffers in that thread's own arena, which would add about 1 GB to a 30 MP render's
    footprint; other platforms return it themselves."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _load_libc():
    if not sys.platform.startswith("linux"):
        return None
    try:
        lib = ctypes.CDLL("libc.so.6")
        lib.malloc_trim.argtypes = [ctypes.c_size_t]
        return lib
    except (OSError, AttributeError):
        return None


_LIBC = _load_libc()


def _map_tiles(fn, jobs: list, px: int = 0) -> list:
    """fn over jobs, in order; px: pixels of the largest job, to keep tiles in flight in budget."""
    n = min(TILE_WORKERS, max(1, cv2.getNumThreads()), len(jobs))
    if px:
        n = min(n, max(1, int(TILE_BUDGET_MB * 2**20 // (px * TILE_BYTES_PER_PX))))
    if n <= 1:
        return [fn(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(fn, jobs))
    _trim_heap()
    return res


def _detect_tile(
    rgb: np.ndarray, skin: np.ndarray, body: np.ndarray, protect: np.ndarray, Fw: float,
    core_box: tuple[int, int, int, int], dark_scales: tuple[int, ...] = tuple(range(len(SIGMAS))),
    bright_scales: tuple[int, ...] = tuple(range(len(BRIGHT_SIGMAS))), kofs: int = 0,
) -> tuple[_Cand, dict[int, tuple[float, float, float]]]:
    """Candidates whose centre is in core_box (tile-local x0, y0, x1, y1) at the given scale
    indices, plus per-scale (median, spread, skin pixels) of the responses on this tile's skin,
    keyed by the candidates' kidx (scale index, +100 for bright, + kofs for the native pass)."""
    cx0, cy0, cx1, cy1 = core_box
    core = (skin > 0.5) & (protect == 0)
    if core.sum() < 64:
        return _cat([]), {}
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2Lab)
    y = _luma(rgb).astype(np.float32)
    ctx = max(4.0, BASE_CONTEXT * Fw)
    # Robust skin baseline: a first estimate, then again without the pixels that are clearly
    # darker (spots) so a dense acne field does not drag its own reference down.
    noise = max(0.004, float(np.std(y[core] - _gauss(y, 1.5)[core])))
    base_y = _robust_baseline(y, core, ctx, lambda b: (b - y) / (2.5 * noise + 0.01))
    w2 = core.astype(np.float32) * (1.0 - _smoothstep(0.5, 1.0, (base_y - y) / (2.5 * noise + 0.01)))
    base_a = _fast_masked_blur(lab[..., 1], w2, ctx)
    base_b = _fast_masked_blur(lab[..., 2], w2, ctx)
    skin_y = float(np.median(base_y[core]))
    rel = np.float32(1.0 / max(skin_y, 0.05))
    dark = (base_y - y) * rel  # relative darkness
    red = lab[..., 1] - base_a
    brown = lab[..., 2] - base_b
    sig_dark = (dark + 0.012 * np.maximum(red, 0.0)).astype(np.float32)
    sig_bright = (-dark).astype(np.float32)
    # Texture references. Pores are by far the most numerous pore-sized dark peaks, so the upper
    # quartile of all such peaks on this tile's skin is the "typical pore" a blackhead must beat.
    # A light dot is held to the bright texture (sheen between pores) and a red dot also to the
    # a* texture: three independent references, each with the sensor noise taken out, so a deep,
    # noisier skin tone is not held to a stricter standard than a light one.
    gs = max(0.6, 0.5 * SIGMAS[0] * math.sqrt(2.0) * Fw)
    gp = _gauss(dark, gs)
    n_luma = _noise_peak(gs) * _noise_sigma(y, core) * float(rel)
    pore_ref = _texture_ref(gp, core, n_luma, LUMA_REF_MIN, 0.03)
    bright_ref = _texture_ref(-gp, core, n_luma, LUMA_REF_MIN, 0.03)
    # Colour noise is not taken out: a red dot has to beat whatever a* speckle this skin shows.
    red_ref = _texture_ref(_gauss(red, gs), core, 0.0, RED_REF_MIN, 3.0)
    # Specular / oily highlight zones (forehead, nose, chin shine): the tone there is well above
    # the person's skin, and the bright dots inside are skin texture lit by the shine.
    # Local, not global: a key light makes one side of the face brighter, a highlight is a
    # bright patch against the skin right around it.
    shine = _smoothstep(1.06, 1.2, _fast_gauss(y, max(2.0, 0.008 * Fw)) / np.maximum(_fast_gauss(base_y, max(4.0, 0.06 * Fw)), 1e-3)).astype(np.float32)
    # Hair direction: structure-tensor coherence at hair-shaft scale. Stubble and body hair make
    # the skin around a dot run one way; acne fields and pores do not.
    gy_ = _gauss(y, max(0.7, 0.0015 * Fw))
    ix, iy = cv2.Sobel(gy_, cv2.CV_32F, 1, 0), cv2.Sobel(gy_, cv2.CV_32F, 0, 1)
    si = max(1.5, 0.012 * Fw)
    jxx, jyy, jxy = _fast_gauss(ix * ix, si), _fast_gauss(iy * iy, si), _fast_gauss(ix * iy, si)
    coh = (np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1e-9)).astype(np.float32)
    # Structure edges (nose side, jaw, hairline): gradient of the baseline-scale luma.
    ys = _gauss(y, max(1.0, 0.006 * Fw))
    edge = np.hypot(cv2.Sobel(ys, cv2.CV_32F, 1, 0), cv2.Sobel(ys, cv2.CV_32F, 0, 1)) / 8.0 * rel * max(1.0, 0.006 * Fw)

    parts: list[_Cand] = []
    stats: dict[int, tuple[float, float, float]] = {}
    sub = core[::3, ::3]
    for signal, emit, is_bright in ((sig_dark, dark_scales, False), (sig_bright, bright_scales, True)):
        if not emit:
            continue
        # One guard scale on each side of the emitted ones (not emitted), so a pass that covers
        # only part of the scale range finds the same scale-space maxima as the whole range.
        n_sc = len(BRIGHT_SIGMAS) if is_bright else len(SIGMAS)
        scales = tuple(range(max(0, min(emit) - 1), min(n_sc, max(emit) + 2)))
        sp = [max(0.7, SIGMAS[k] * Fw) for k in scales]
        resp = []
        blurred = []
        for sg in sp:
            g = cv2.GaussianBlur(signal, (0, 0), sigmaX=sg, borderType=cv2.BORDER_REFLECT)
            blurred.append(g)
            resp.append(-(sg * sg) * cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        spatial = [cv2.dilate(r, np.ones((3, 3), np.uint8)) for r in resp]
        for k, sg in enumerate(sp):
            if scales[k] not in emit:
                continue
            r = resp[k]
            med, spread = _robust(r[::3, ::3][sub])
            kidx = kofs + scales[k] + (100 if is_bright else 0)
            stats[kidx] = (med, spread, float(sub.sum()))
            is_max = r >= spatial[k]
            if k > 0:
                is_max &= r >= spatial[k - 1]
            if k + 1 < len(sp):
                is_max &= r >= spatial[k + 1]
            cand = is_max & core & (r > med + 1.6 * spread) & (r > 0.25 * MIN_REL_CONTRAST)
            cand[:cy0] = False
            cand[cy1:] = False
            cand[:, :cx0] = False
            cand[:, cx1:] = False
            yy, xx = np.nonzero(cand)
            if yy.size == 0:
                continue
            score = (r[yy, xx] - med) / spread
            if yy.size > MAX_CANDIDATES_PER_TILE:
                top = np.argsort(-score)[:MAX_CANDIDATES_PER_TILE]
                yy, xx, score = yy[top], xx[top], score[top]
            fx, fy = xx.astype(np.float32), yy.astype(np.float32)
            rad = sg * math.sqrt(2.0)
            sm = max(0.6, 0.35 * rad)
            d_k = _gauss(dark, sm)
            c_dark = d_k[yy, xx]
            c_red = _gauss(red, sm)[yy, xx]
            c_brown = _gauss(brown, sm)[yy, xx]
            # Ring isotropy: share of 12 ring samples the mark stands out from.
            sign = -1.0 if is_bright else 1.0
            center = sign * c_dark
            hits = np.zeros(yy.size, np.float32)
            inside = np.zeros(yy.size, np.float32)
            for t in np.linspace(0.0, 2 * math.pi, 12, endpoint=False):
                px, py = fx + RING * rad * math.cos(t), fy + RING * rad * math.sin(t)
                ring = sign * _sample(d_k, px, py)
                hits += (ring < 0.45 * center).astype(np.float32)
                inside += (_sample(skin, px, py) > 0.5).astype(np.float32)
            iso = hits / 12.0
            # Hessian anisotropy at the mark's scale (lines, hairs and creases are elongated).
            g = blurred[k]
            dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)[yy, xx]
            dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)[yy, xx]
            dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)[yy, xx]
            tr, det_root = dxx + dyy, np.sqrt(np.maximum((dxx - dyy) ** 2 + 4 * dxy * dxy, 0.0))
            l1, l2 = np.abs(0.5 * (tr + det_root)), np.abs(0.5 * (tr - det_root))
            aniso = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1e-9)
            parts.append(_Cand(
                fx, fy, np.full(yy.size, sg, np.float32), score.astype(np.float32),
                np.full(yy.size, is_bright), c_dark, c_red, c_brown, iso, aniso.astype(np.float32),
                (np.abs(c_dark) / pore_ref).astype(np.float32),
                _sample(edge, fx, fy).astype(np.float32), body[yy, xx] > 0.5,
                r[yy, xx].astype(np.float32), np.full(yy.size, med, np.float32), np.full(yy.size, spread, np.float32),
                np.full(yy.size, kidx, np.int32), shine[yy, xx], inside / 12.0,
                (np.maximum(c_red, 0.0) / red_ref).astype(np.float32),
                (np.abs(c_dark) / (bright_ref if is_bright else pore_ref)).astype(np.float32),
                coh[yy, xx],
            ))
    return _cat(parts), stats


def _neighbour_counts(x: np.ndarray, y: np.ndarray, sel: np.ndarray, reach: float) -> np.ndarray:
    """For every point, about how many points of `sel` lie within `reach` (excluding itself).

    Points are binned on a grid a quarter of `reach` wide and the bins summed over a disc, so
    the cost is one small convolution however many candidates a full-length photo has.
    """
    out = np.zeros(x.size, np.int32)
    if x.size == 0 or not sel.any():
        return out
    cell = max(reach / 4.0, 0.5)
    gx = np.floor(x / cell).astype(np.int64)
    gy = np.floor(y / cell).astype(np.int64)
    ox, oy = gx.min() - 5, gy.min() - 5
    gx, gy = gx - ox, gy - oy
    grid = np.zeros((int(gy.max()) + 6, int(gx.max()) + 6), np.float32)
    np.add.at(grid, (gy[sel], gx[sel]), 1.0)
    k = _ellipse(9).astype(np.float32)  # a disc of radius ~reach in grid cells
    dens = cv2.filter2D(grid, -1, k, borderType=cv2.BORDER_CONSTANT)
    out = np.round(dens[gy, gx]).astype(np.int32) - sel.astype(np.int32)
    return np.maximum(out, 0)


def _neighbour_medians(
    x: np.ndarray, y: np.ndarray, query: np.ndarray, pool: np.ndarray, reach: float, *vals: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """For each index in query: how many pool points lie within reach (excluding itself), and
    the median of each of vals over them (NaN when none)."""
    cnt = np.zeros(query.size, np.int32)
    meds = [np.full(query.size, np.nan, np.float32) for _ in vals]
    if query.size == 0 or pool.size == 0:
        return cnt, meds
    order = pool[np.argsort(x[pool])]
    px = x[order]
    for q, i in enumerate(query):
        a, b = np.searchsorted(px, x[i] - reach), np.searchsorted(px, x[i] + reach)
        w = order[a:b]
        w = w[(w != i) & ((x[w] - x[i]) ** 2 + (y[w] - y[i]) ** 2 <= reach * reach)]
        cnt[q] = w.size
        if w.size:
            for v, m in zip(vals, meds):
                m[q] = _median(v[w])
    return cnt, meds


def _median(v: np.ndarray) -> float:
    """np.median of a small float32 array, without its per-call overhead (same result: the
    middle value, or the float32 mean of the middle two)."""
    s = np.sort(v)
    h = s.size // 2
    return float(s[h]) if s.size % 2 else float((s[h - 1] + s[h]) / s.dtype.type(2))


def _nms(c: _Cand, order: np.ndarray, share: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Non-maximum suppression in scale space.

    Candidates of similar scale (radius ratio <= 2) that overlap keep only the strongest. Then a
    small mark whose centre lies inside a much larger kept mark is absorbed by it: a pimple's
    bright head and its red base, or a stubble dot inside a mole, are one mark, corrected (or
    kept) as a whole. Returns (keep, extent radius in work px).
    """
    keep = np.zeros(c.x.size, bool)
    rad = c.sigma * math.sqrt(2.0)
    extent = rad.astype(np.float32).copy()
    if c.x.size == 0:
        return keep, extent
    cell = max(4.0, float(np.max(rad)) * 2.0)
    grid: dict[tuple[int, int], list[int]] = {}
    for i in order:
        gx, gy = int(c.x[i] // cell), int(c.y[i] // cell)
        clash = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    big, small = max(rad[i], rad[j]), min(rad[i], rad[j])
                    if big > 2.0 * small:
                        continue
                    if (c.x[j] - c.x[i]) ** 2 + (c.y[j] - c.y[i]) ** 2 < (share * big) ** 2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            keep[i] = True
            grid.setdefault((gx, gy), []).append(i)
    # Absorb small marks into much larger ones that contain them.
    kept = np.nonzero(keep)[0]
    for i in kept[np.argsort(-rad[kept])]:
        if not keep[i]:
            continue
        inside = keep & (rad < 0.5 * rad[i]) & ((c.x - c.x[i]) ** 2 + (c.y - c.y[i]) ** 2 < (0.9 * rad[i]) ** 2)
        if inside.any():
            keep[inside] = False
            extent[i] = max(extent[i], float(np.max(np.hypot(c.x[inside] - c.x[i], c.y[inside] - c.y[i]) + rad[inside])))
    return keep, extent


def _k_for(sens: float, multiplier: int) -> float:
    """Detection threshold in robust spreads for a 0..100 sensitivity (higher = more marks)."""
    s = float(np.clip(sens + MULTIPLIER_SENS.get(multiplier, 0.0), 0.0, 120.0)) / 100.0
    # 0 -> ~11.5 spreads (only blatant marks), 70 -> ~3.7, 100 -> 2.5; 2x / 3x reach below that.
    return 2.5 + 9.0 * max(0.0, 1.0 - s) ** 1.7 - 0.6 * max(0.0, s - 1.0) / 0.2


def _classify(
    c: _Cand, Fw: float, settings: SkinPerfectionSettings, sens_boost: float = 0.0, zones: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(kind, confidence, accepted, review reason) per candidate. accepted = passes its category's
    threshold; a reason (not None) means the mark is ambiguous and must not be corrected unasked."""
    n = c.x.size
    diam = 2.0 * c.sigma * math.sqrt(2.0) / max(Fw, 1.0)
    rel = np.abs(c.dark)
    # Visible contrast: darkness (or lightness) plus redness, in relative luma units.
    contrast = rel + 0.012 * np.maximum(c.red, 0.0)
    kinds = np.full(n, "blemish", dtype=object)
    is_red = (c.red > RED_A) & (c.red * 0.012 > 0.35 * np.maximum(c.dark, 0.0))
    tiny = diam < TINY
    darkish = ~c.bright
    # Bright family: whiteheads (small light domes); a light dome in a red ring is a pimple.
    kinds[c.bright] = "whitehead"
    kinds[c.bright & (c.red > RED_A)] = "pimple"
    # Shine: light dots inside an oily highlight, or a dense field of them, are lit skin texture.
    bright_dots = c.bright & (diam < WHITEHEAD_MAX_D) & (rel >= 0.8 * WHITEHEAD_MIN_REL) & (c.score >= _k_for(70, 1))
    # Lit texture: on real skin, sheen, oil and body shimmer make wide fields of light dots. The
    # tone-fair tests below (bright texture, colour texture, pimple fusion) never apply inside such
    # a field; there a mark has to pass the strict pore test alone.
    in_field = _neighbour_counts(c.x, c.y, bright_dots, FIELD_REACH * Fw) >= FIELD_DOTS
    glossy = bright_dots & ((c.shine > 0.4) | (_neighbour_counts(c.x, c.y, bright_dots, FOLLICLE_REACH * Fw) >= 4))
    kinds[glossy & (c.red <= RED_A)] = "shine"
    # Dark / red family.
    kinds[darkish & is_red] = "acne"
    kinds[darkish & is_red & (diam >= 0.014) & (c.iso >= 0.8)] = "pimple"
    flat_red = darkish & is_red & (c.brown > BROWN_B) & (c.pore < 4.0) & (diam >= 0.01)
    kinds[flat_red] = "postAcne"
    nonred = darkish & ~is_red
    kinds[nonred & (c.brown > BROWN_B)] = "darkSpot"
    kinds[nonred & tiny] = "smallDot"
    kinds[nonred & tiny & (np.abs(c.brown) <= 2.5) & (rel >= 0.1)] = "blackhead"
    # Protected natural features.
    mole = nonred & (diam >= MOLE_MIN_D) & (rel >= MOLE_MIN_REL) & (diam < BIRTHMARK_D)
    kinds[mole] = "mole"
    kinds[darkish & (diam >= BIRTHMARK_D) & (rel < MOLE_MIN_REL)] = "birthmark"
    small_brown = nonred & (diam < FRECKLE_MAX_D) & (rel < FRECKLE_MAX_REL) & (c.brown > 0.3) & ~mole
    near = _neighbour_counts(c.x, c.y, small_brown, FRECKLE_REACH * Fw)
    kinds[small_brown & (near >= FRECKLE_NEIGHBOURS)] = "freckle"
    # A pore-sized dot only counts at all when it clearly beats the pore texture around it.
    pore_ok = (c.pore >= PORE_RATIO + 0.5) & (rel >= 0.08)
    tiny_dark = nonred & tiny & ~c.bright & pore_ok
    dense = _neighbour_counts(c.x, c.y, tiny_dark, FOLLICLE_REACH * Fw)
    # Blackheads crowd on the nose too, so a dense field there stays blackheads.
    zone = zones[np.clip(c.y.astype(int), 0, zones.shape[0] - 1), np.clip(c.x.astype(int), 0, zones.shape[1] - 1)] \
        if zones is not None else np.full(n, 2, np.uint8)
    # Hair follicles only where hair grows: the beard area (below the nose) and body skin. With
    # landmarks, a cluster of dark dots on the forehead, nose or upper cheeks is blackheads / dots.
    can_hair = (zone == 2) | c.body
    on_nose = zone == 1
    fol = tiny_dark & (dense >= FOLLICLE_NEIGHBOURS) & can_hair
    # Grow the stubble field once: dots at its edge have fewer neighbours but sit next to it.
    edge_dots = tiny_dark & ~fol & can_hair & ~on_nose & (_neighbour_counts(c.x, c.y, fol, FOLLICLE_REACH * Fw) >= 1)
    kinds[fol | edge_dots] = "follicle"
    kinds[darkish & tiny & (c.aniso < 0.25)] = "follicle"  # a short hair stub

    # Ambiguity (2C): natural feature or imperfection? Judged against the mark's own group, and
    # only for a clearly significant mark (faint relief on body skin is left as it was).
    strong = c.score >= FAIR_MARGIN * _k_for(70, 1)
    reason = np.full(n, None, dtype=object)
    was_feature = np.zeros(n, bool)  # classified as a natural feature, now in doubt
    diam_v, rel_v = diam.astype(np.float32), rel.astype(np.float32)
    # Stubble field: follicles are alike. A dot much bigger and darker than the field's dots is
    # likely a blackhead in the beard, but could be a thicker hair: review, never automatic.
    field = np.nonzero((kinds == "follicle") & (c.aniso >= 0.25))[0]
    if field.size >= FOLLICLE_NEIGHBOURS:
        cnt, (md, mr) = _neighbour_medians(c.x, c.y, field, field, 2.0 * FOLLICLE_REACH * Fw, diam_v, rel_v)
        out = (cnt >= 3) & strong[field] & (diam_v[field] >= FIELD_OUTLIER_SIZE * md) & (rel_v[field] >= FIELD_OUTLIER_DARK * mr)
        kinds[field[out]] = "blackhead"
        reason[field[out]] = "mayBeFollicle"
        was_feature[field[out]] = True
        # A lone pore-sized dark dot in the hair zone next to a stubble field, and like its
        # dots, could be a stray follicle: review instead of correcting it as a blackhead.
        lone = np.nonzero(np.isin(kinds, ("blackhead", "smallDot")) & tiny & can_hair & (reason == None))[0]  # noqa: E711
        cnt2, (md2, mr2) = _neighbour_medians(c.x, c.y, lone, field, 3.0 * FOLLICLE_REACH * Fw, diam_v, rel_v)
        alike = (cnt2 >= 1) & (diam_v[lone] <= FIELD_LIKE * md2) & (rel_v[lone] <= FIELD_LIKE * mr2)
        reason[lone[alike]] = "mayBeFollicle"
    # A pore-sized dark dot where hair grows and the skin around it runs one way (hair shafts)
    # may be a follicle, however few dots are near it: review, not a blackhead correction.
    haired = np.isin(kinds, ("blackhead", "smallDot")) & tiny & can_hair & (c.coh >= HAIR_COHERENCE) & (reason == None)  # noqa: E711
    reason[haired] = "mayBeFollicle"
    # Freckle group: freckles are alike. One clearly bigger or darker than its group is likely a
    # dark spot among freckles, but could be a larger freckle: review, never automatic.
    frk = np.nonzero(kinds == "freckle")[0]
    if frk.size >= FRECKLE_NEIGHBOURS:
        cnt, (md, mr) = _neighbour_medians(c.x, c.y, frk, frk, FRECKLE_REACH * Fw, diam_v, rel_v)
        out = (cnt >= 3) & strong[frk] & (diam_v[frk] >= FRECKLE_OUTLIER_SIZE * md) & (rel_v[frk] >= FRECKLE_OUTLIER_DARK * mr)
        kinds[frk[out]] = "darkSpot"
        reason[frk[out]] = "mayBeFreckle"
        was_feature[frk[out]] = True

    rad = c.sigma * math.sqrt(2.0)
    # Rims: a mark darker than the skin baseline leaves a faint light ring where the baseline meets
    # it (most visibly beside moles on deep skin). A light dot right at a much darker mark is
    # that rim, not a whitehead.
    strong_dark = np.nonzero(darkish & (rel >= 0.1))[0]
    light = c.bright & (rel >= 0.5 * WHITEHEAD_MIN_REL) & (c.tpore < RIM_MAX_PORE)
    if strong_dark.size and light.any():
        li = np.nonzero(light & (c.red <= RED_A))[0]
        li = li[np.argsort(c.x[li])]
        lx, rmax = c.x[li], float(rad[li].max(initial=0.0))
        for j in strong_dark:
            reach = 2.0 * rad[j] + rmax
            a, b = np.searchsorted(lx, c.x[j] - reach), np.searchsorted(lx, c.x[j] + reach)
            if a == b:
                continue
            w = li[a:b]
            hit = ((c.x[w] - c.x[j]) ** 2 + (c.y[w] - c.y[j]) ** 2 < (2.0 * rad[j] + rad[w]) ** 2) & (rel[j] >= RIM_MIN_RATIO * rel[w])
            kinds[w[hit]] = "rim"
    # Pimples: a light head on a red dome. The head breaks the dome's ring test (the centre is not
    # darker than the ring), so a red dome with a light, red head at its centre is one pimple and
    # its roundness is judged with the head counted in.
    fused = np.zeros(n, bool)
    heads = np.nonzero(c.bright & (c.red > RED_A) & (c.iso >= 0.5) & ~in_field)[0]
    domes = np.nonzero(darkish & is_red)[0]
    domes = domes[np.argsort(c.x[domes])]
    dx, dmax = c.x[domes], float(rad[domes].max(initial=0.0))
    for i in heads if domes.size else ():
        a, b = np.searchsorted(dx, c.x[i] - 0.5 * dmax), np.searchsorted(dx, c.x[i] + 0.5 * dmax)
        w = domes[a:b]
        inside = ((c.x[w] - c.x[i]) ** 2 + (c.y[w] - c.y[i]) ** 2 < (0.5 * rad[w]) ** 2) & (rad[w] > 1.5 * rad[i])
        if inside.any():
            fused[w[inside][np.argmax(c.score[w[inside]])]] = True
    kinds[fused] = "pimple"

    # Threshold per category (and body sensitivity on body skin).
    accepted = np.zeros(n, bool)
    conf = np.zeros(n, np.float32)
    ep = settings.edgeProtection / 100.0
    min_aniso = 0.18 + 0.3 * ep
    cats: dict[str, object] = {}
    ks: dict[float, float] = {}
    edge_sm = _smoothstep(0.15, 0.45, c.edge)  # elementwise, as it was per candidate
    for i in range(n):
        kind = kinds[i]
        if kind in ("shine", "rim"):
            continue
        cat = cats.get(kind)
        if cat is None:
            cat = settings.category("acne" if kind in ("mole", "freckle", "birthmark", "follicle") else kind)
            if kind in PROTECTED_KINDS:
                # Protected marks are judged with the general dark-mark sensitivity of their family.
                cat = settings.category("darkSpot" if kind != "follicle" else "blackhead")
            cats[kind] = cat
        sens = cat.sensitivity
        if c.body[i]:
            sens = min(100.0, sens * settings.bodySensitivity / 70.0)
        # Colourless marks carry no colour evidence: the more sensitive residual scan does not
        # lower their bar (on its own it finds skin relief, most of all on deep skin).
        boost = 0.0 if kind == "blemish" else sens_boost
        k = ks.get(sens + boost)
        if k is None:
            k = ks[sens + boost] = _k_for(sens + boost, settings.multiplier)
        floor = MIN_REL_CONTRAST * (1.6 - 0.9 * min(sens + boost, 100.0) / 100.0)
        margin = c.score[i] / k
        ok = margin >= 1.0 and contrast[i] >= floor and cat.enabled
        if diam[i] < 1.6 * TINY:
            # Texture test against this skin's own texture of the same kind (see _detect_tile):
            # light dots against bright texture, red dots against luma or colour texture.
            need = PORE_RATIO + 1.5 - 1.5 * min(sens + boost, 100.0) / 100.0
            texture_ok = c.pore[i] >= need
            if not in_field[i] and margin >= FAIR_MARGIN:
                if c.bright[i]:
                    texture_ok |= c.tpore[i] >= need - BRIGHT_PORE_RELIEF
                if is_red[i] and c.red[i] >= CLEAR_RED_A:
                    texture_ok |= c.cpore[i] >= need
            ok &= texture_ok
        iso = max(c.iso[i], 0.75) if fused[i] else c.iso[i]
        # Shape: blobs only. Lines, hairs and shading edges fail isotropy or anisotropy.
        shape = min(1.0, iso / 0.75) * min(1.0, c.aniso[i] / max(min_aniso, 1e-3))
        ok &= c.iso[i] >= 0.5 and c.aniso[i] >= min_aniso * 0.8
        # The whole context ring must be skin: a dark blob where a finger meets the next one, a
        # nail or the edge of a sleeve is a boundary, not a mark.
        ok &= c.inskin[i] >= 0.85
        if diam[i] > 0.03:
            # Large marks are where shading lives (between the brows, beside the nose): only
            # clearly round, isolated ones away from structure edges count.
            ok &= (c.iso[i] >= 0.6 if fused[i] else c.iso[i] >= 0.9) and c.edge[i] < 0.25 and c.shine[i] < 0.3
        if c.body[i]:
            # Body skin folds (knuckles, elbows, creases) and shading are long or large; body
            # marks must be compact, round and of moderate size.
            ok &= diam[i] <= BODY_MAX_D and c.iso[i] >= 0.65 and c.aniso[i] >= min_aniso
        sfac = 1.25 - 0.5 * min(sens + boost, 100.0) / 100.0
        if kind == "blemish":
            # Colourless marks: compact, clearly darker, and never inside an oily highlight
            # (there a "dark blob" is the gap between two shine patches).
            ok &= rel[i] >= BLEMISH_MIN_REL * sfac and diam[i] <= BLEMISH_MAX_D and c.shine[i] < 0.4
        if kind in ("blackhead", "smallDot"):
            ok &= rel[i] >= TINY_MIN_REL * sfac
        if kind == "whitehead":
            ok &= rel[i] >= WHITEHEAD_MIN_REL * sfac and diam[i] <= WHITEHEAD_MAX_D
        # Next to a strong structure edge only clear marks count (edge protection).
        edge_pen = 1.0 - ep * float(edge_sm[i])
        conf[i] = float(np.clip((1.0 - math.exp(-1.2 * max(margin, 0.0))) * shape * (0.4 + 0.6 * edge_pen), 0.0, 1.0))
        accepted[i] = bool(ok)
    # A feature in doubt is reported (for review) whenever it would have been kept as a feature;
    # any other ambiguous mark only when it passed its own category's threshold.
    accepted |= was_feature & (c.score >= _k_for(70, 1)) & ((diam >= 1.6 * TINY) | (c.pore >= PORE_RATIO))
    return list(kinds), conf, accepted, reason


@dataclass
class _Analysis:
    x0: int
    y0: int
    scale: float
    Fw: float
    tiles: int


def _zones(local: list[FaceAnalysis], H: int, W: int, Fw: float) -> np.ndarray | None:
    """Zones from landmarks: 1 = nose (blackheads crowd there), 2 = beard area (below the nose)."""
    zones = None
    if any(len(f.regions.nose) >= 3 for f in local):
        zones = np.zeros((H, W), np.uint8)
        for f in local:
            if len(f.regions.nose) < 3 or len(f.regions.faceOval) < 3:
                continue
            nose_pts = np.round(np.array(f.regions.nose, np.float32) * [W, H]).astype(np.int32)
            oval = np.round(np.array(f.regions.faceOval, np.float32) * [W, H]).astype(np.int32)
            lower = np.zeros((H, W), np.uint8)
            cv2.fillPoly(lower, [oval], 2)
            lower[: max(0, int(nose_pts[:, 1].max())), :] = 0
            zones = np.maximum(zones, cv2.dilate(lower, _ellipse(max(3, int(0.15 * Fw) | 1))))
            nz = np.zeros((H, W), np.uint8)
            cv2.fillPoly(nz, [nose_pts], 1)
            nz = cv2.dilate(nz, _ellipse(max(3, int(0.08 * Fw) | 1)))
            zones[nz > 0] = 1
    return zones


def _analyse(
    image: np.ndarray, skin: np.ndarray, body: np.ndarray, faces: list[FaceAnalysis], F: float,
    settings: SkinPerfectionSettings, sens_boost: float = 0.0, cache: dict | None = None,
) -> tuple[list[tuple], _Analysis | None]:
    """Run A-E on the skin's bounding box. Returns raw marks (x, y, r, kind, conf, body, contrast,
    review reason or None) in full-resolution px, accepted or protected only.

    cache: one dict per render. The residual scans see the same skin, body and faces as the
    first detection (only the pixels change), so the masks built from them are made once."""
    h, w = image.shape[:2]
    geo = cache.get("geometry") if cache is not None else None
    if geo is None:
        ys, xs = np.nonzero(skin > 0.3)
        if ys.size < 64:
            return [], None
        pad = int(0.08 * F) + 8
        geo = (max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1))
    y0, y1, x0, x1 = geo
    s = min(1.0, WORK_FACE / max(F, 1.0))
    crop_full = image[y0:y1, x0:x1]
    ch, cw = crop_full.shape[:2]
    size = (max(1, round(cw * s)), max(1, round(ch * s)))
    crop = crop_full if s >= 1 else cv2.resize(crop_full, size, interpolation=cv2.INTER_AREA)
    Fw = F * s
    local = faces_in_crop(faces, x0, y0, x1 - x0, y1 - y0, w, h)
    masks = cache.get("masks") if cache is not None else None
    if masks is None:
        sk_full = skin[y0:y1, x0:x1]
        bd_full = body[y0:y1, x0:x1]
        sk, bd = sk_full, bd_full
        if s < 1:
            sk = cv2.resize(sk, size, interpolation=cv2.INTER_AREA)
            bd = cv2.resize(bd, size, interpolation=cv2.INTER_AREA)
        protect = feature_protection_mask(local, size[1], size[0])
        # Close the skin mask over spot-sized holes (a red pimple fails a colour skin test).
        # Face skin is colour-gated, so close it over pimple-sized holes; semantic body skin has no
        # such holes and is only closed a little, so the gap between two fingers stays open.
        face_part = np.clip(sk - bd, 0.0, 1.0).astype(np.float32)
        sk = np.maximum(cv2.morphologyEx(face_part, cv2.MORPH_CLOSE, _ellipse(max(3, int(0.05 * Fw) | 1))),
                        cv2.morphologyEx(bd.astype(np.float32), cv2.MORPH_CLOSE, _ellipse(max(3, int(0.01 * Fw) | 1))))
        sk_n = protect_n = None
        if s < NATIVE_BELOW:
            sk_n = cv2.resize(sk, (cw, ch), interpolation=cv2.INTER_LINEAR)  # the closed mask, not a second closing
            protect_n = feature_protection_mask(local, ch, cw)
        masks = (sk, bd, bd_full, protect, sk_n, protect_n, _zones(local, size[1], size[0], Fw))
        if cache is not None:
            cache["geometry"], cache["masks"] = geo, masks
    sk, bd, bd_full, protect, sk_n, protect_n, zones = masks

    # Native resolution for the finest scales: when the face is wider than the analysis scale,
    # the smallest marks (blackheads, whiteheads, pore-sized dots) are looked for on the
    # full-resolution pixels, not on the shrunk copy, so a large photo loses no detail there.
    native = s < NATIVE_BELOW
    fine = tuple(range(NATIVE_SCALES)) if native else ()
    passes = [(crop, sk, bd, protect, Fw, 1.0, tuple(k for k in range(len(SIGMAS)) if k not in fine),
               tuple(k for k in range(len(BRIGHT_SIGMAS)) if k not in fine), 0)]
    if native:
        passes.append((crop_full, sk_n, bd_full, protect_n, F, s, fine, fine, NATIVE_KOFS))
    parts: list[_Cand] = []
    stats: list[dict[int, tuple[float, float, float]]] = []
    jobs = []
    for p_crop, p_sk, p_bd, p_prot, p_F, p_s, ks, kb, kofs in passes:
        smax = SIGMAS[min(len(SIGMAS) - 1, max(ks + kb) + 1)]  # + the guard scale
        apron = int(math.ceil(3.0 * smax * p_F + RING * smax * math.sqrt(2) * p_F + BASE_CONTEXT * p_F)) + 4
        PH, PW = p_crop.shape[:2]
        for ty in range(0, PH, TILE):
            for tx in range(0, PW, TILE):
                cy0, cy1, cx0, cx1 = ty, min(PH, ty + TILE), tx, min(PW, tx + TILE)
                if not (p_sk[cy0:cy1, cx0:cx1] > 0.5).any():
                    continue
                ay0, ay1, ax0, ax1 = max(0, cy0 - apron), min(PH, cy1 + apron), max(0, cx0 - apron), min(PW, cx1 + apron)
                jobs.append(((p_crop[ay0:ay1, ax0:ax1], p_sk[ay0:ay1, ax0:ax1], p_bd[ay0:ay1, ax0:ax1], p_prot[ay0:ay1, ax0:ax1], p_F,
                              (cx0 - ax0, cy0 - ay0, cx1 - ax0, cy1 - ay0), ks, kb, kofs), ax0, ay0, p_s))
    # Tiles are independent, so they run side by side (OpenCV and NumPy release the GIL); the
    # results are gathered in tile order, so the output is the same as one after the other.
    px = max((j[0][0].shape[0] * j[0][0].shape[1] for j in jobs), default=0)
    for (cand, st), (_, ax0, ay0, p_s) in zip(_map_tiles(lambda j: _detect_tile(*j[0]), jobs, px), jobs):
        if st:
            stats.append(st)
        if cand.x.size:
            # Into analysis (work) coordinates, so all candidates share one frame.
            cand.x = (cand.x + ax0) * p_s
            cand.y = (cand.y + ay0) * p_s
            cand.sigma = cand.sigma * p_s
            parts.append(cand)
    tiles = len(jobs)
    H, W = size[1], size[0]
    ana = _Analysis(x0, y0, s, Fw, tiles)
    c = _cat(parts)
    if c.x.size == 0:
        return [], ana
    # Global spread consistency: a tile's own response spread (inflated by a dense acne field,
    # deflated on very clean skin) may differ from the photo's skin-weighted median for that
    # scale by at most -20% / +25%, so every tile is held to about the same standard.
    for kidx in np.unique(c.kidx):
        st = np.array([t[int(kidx)] for t in stats if int(kidx) in t])
        if st.size == 0:
            continue
        order_s = np.argsort(st[:, 1])
        cum = np.cumsum(st[order_s, 2])
        g = float(st[order_s, 1][np.searchsorted(cum, 0.5 * cum[-1])])
        sel = c.kidx == kidx
        spread = np.clip(c.spread[sel], 0.8 * g, 1.25 * g)
        c.score[sel] = (c.resp[sel] - c.med[sel]) / spread
    kinds, conf, ok, reason = _classify(c, Fw, settings, sens_boost, zones)
    kinds_a = np.array(kinds, dtype=object)
    diam = 2.0 * c.sigma * math.sqrt(2.0) / max(Fw, 1.0)
    prot = np.isin(kinds_a, PROTECTED_KINDS) & (c.score >= _k_for(70, 1)) & ((diam >= 1.6 * TINY) | (c.pore >= PORE_RATIO))
    keep_any = ok | prot
    # Clusters (2C): a large flat mark (a "birthmark" or broad blemish) holding two or more
    # clearly stronger marks is those marks side by side, merged at a coarse scale. The large
    # one is dropped so the marks inside are judged (and corrected or kept) one by one.
    rad_all = c.sigma * math.sqrt(2.0)
    small = np.nonzero(ok & ~np.isin(kinds_a, PROTECTED_KINDS))[0]
    for j in np.nonzero(keep_any & np.isin(kinds_a, ("birthmark", "blemish")) & (diam >= 0.02))[0]:
        if not small.size:
            break
        inside = small[((c.x[small] - c.x[j]) ** 2 + (c.y[small] - c.y[j]) ** 2 < (0.9 * rad_all[j]) ** 2)
                       & (rad_all[small] < 0.5 * rad_all[j])
                       & (np.abs(c.dark[small]) + 0.012 * np.maximum(c.red[small], 0.0)
                          >= CLUSTER_PEAK_RATIO * (abs(c.dark[j]) + 0.012 * max(c.red[j], 0.0)))]
        if inside.size >= 2:
            keep_any[j] = False
    # Non-maximum suppression across scales and signals, strongest first.
    order = np.argsort(-(c.score / np.maximum(np.array([_k_for(70, 1)] * c.x.size), 1e-6)))
    order = order[keep_any[order]]
    kept, extent = _nms(c, order, share=1.0)
    out = []
    for i in np.nonzero(kept)[0]:
        out.append((
            float(c.x[i] / s + x0), float(c.y[i] / s + y0), float(extent[i] / s),
            str(kinds_a[i]), float(conf[i]), bool(c.body[i]), float(abs(c.dark[i]) + 0.012 * max(c.red[i], 0.0)),
            reason[i],
        ))
    return out, ana


# =============================================================================================
# F/G. Correction
# =============================================================================================


def _mark_contrast(image: np.ndarray, skin: np.ndarray, m: Mark) -> float:
    """Relative contrast of the mark against its ring on `image` (same measure as detection)."""
    h, w = image.shape[:2]
    R = max(1.5, m.r)
    reach = int(math.ceil(RING * R * 1.6 + 3))
    x0, x1 = max(0, int(m.x) - reach), min(w, int(m.x) + reach + 1)
    y0, y1 = max(0, int(m.y) - reach), min(h, int(m.y) + reach + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    crop = image[y0:y1, x0:x1]
    lab = cv2.cvtColor(np.ascontiguousarray(_gauss(crop, max(0.6, 0.35 * R)), np.float32), cv2.COLOR_RGB2Lab)
    yl = _luma(_gauss(crop, max(0.6, 0.35 * R)))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(xx - m.x, yy - m.y)
    inner = d <= max(1.0, 0.6 * R)
    ring = (d >= 1.8 * R) & (d <= 2.6 * R) & (skin[y0:y1, x0:x1] > 0.3)
    if not inner.any() or ring.sum() < 4:
        return 0.0
    base = float(np.median(yl[ring]))
    dark = (base - float(np.mean(yl[inner]))) / max(base, 0.05)
    red = float(np.mean(lab[..., 1][inner])) - float(np.median(lab[..., 1][ring]))
    if m.kind == "whitehead":
        return max(0.0, -dark)
    return max(0.0, dark) + 0.012 * max(red, 0.0)


def _cap_fill(low: np.ndarray, fill: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """A repair may not come out lighter than the median of the skin right around it: next to
    a highlight the membrane fill would otherwise carry the shine into the mark as a pale blob."""
    ring = cv2.dilate(sel.astype(np.uint8), _ellipse(5)).astype(bool) & ~sel
    if ring.sum() < 4:
        return fill
    yl, yf = _luma(low), _luma(fill)
    cap = float(np.median(yl[ring]))
    over = (yf > cap) & (yf > yl + 1e-6)
    if not over.any():
        return fill
    t = np.ones(yl.shape, np.float32)
    t[over] = np.clip((np.maximum(cap, yl[over]) - yl[over]) / (yf[over] - yl[over]), 0.0, 1.0)
    return low + (fill - low) * t[..., None]


def _grow(settings: SkinPerfectionSettings) -> float:
    """Repair radius / mark radius."""
    return (1.25 + 1.0 * settings.correctionRadius / 100.0) * MULTIPLIER_RADIUS.get(int(settings.multiplier), 1.0)


def _repair_radius(m: Mark, settings: SkinPerfectionSettings) -> float:
    # The margin covers a pixel of centring error, which on a pore-sized mark leaves a fragment.
    return max(1.5, m.r * _grow(settings) + REPAIR_MARGIN)


def _natural_failures(m: Mark) -> list[str]:
    """Failed checks of the mark's last repair that say it changed the skin wrongly (not only that
    some of the mark is left: still visible, or its colour partly left)."""
    q = m.quality or {}
    return [k for k in q.get("failed", []) if k in Q.NATURALNESS and not (k == "colour" and q.get("incomplete"))]


DONOR_MAX_TAKEN = 0.35  # a donor patch may overlap other marks / features by at most this share ...
DONOR_TAKEN_COST = 0.05  # ... at this cost per unit of overlap (tone mismatch is in relative luma)


def _ranked_donor(
    box: tuple[int, int, int, int], low_y: np.ndarray, ok: np.ndarray, taken: np.ndarray, distance: int, rank: int = 0
) -> tuple[int, int] | None:
    """Offset (dy, dx) of a nearby skin patch whose tone matches the mark's rim: the best one, or
    for a retry after a failed quality check the next best (rank 1, 2, ...). A patch must be skin
    (`ok`) throughout; overlap with other marks (`taken`) is allowed up to DONOR_MAX_TAKEN and
    costs, so in a dense acne field a mostly clear patch still gives texture."""
    x, y, bw, bh = box
    h, w = ok.shape
    pad = max(2, max(bw, bh) // 3)
    y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
    target = float(np.median(low_y[y0:y1, x0:x1]))
    found: list[tuple[float, tuple[int, int]]] = []
    for i in range(12):
        t = 2 * math.pi * i / 12
        for dist in (distance, int(distance * 1.6), int(distance * 2.4), int(distance * 3.6)):
            dy, dx = round(math.sin(t) * dist), round(math.cos(t) * dist)
            yy0, yy1, xx0, xx1 = y0 + dy, y1 + dy, x0 + dx, x1 + dx
            if yy0 < 0 or xx0 < 0 or yy1 > h or xx1 > w or not ok[yy0:yy1, xx0:xx1].all():
                continue
            cover = float(taken[yy0:yy1, xx0:xx1].mean())
            if cover > DONOR_MAX_TAKEN:
                continue
            cost = abs(float(np.median(low_y[yy0:yy1, xx0:xx1])) - target) + 0.002 * dist / max(1, distance) + DONOR_TAKEN_COST * cover
            found.append((cost, (dy, dx)))
        if rank == 0 and found and min(found)[0] < 0.01:
            break
    if not found:
        return None
    found.sort(key=lambda c: c[0])
    return found[min(rank, len(found) - 1)][1]


def _known_blur(img: np.ndarray, known: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of the known pixels only (normalised convolution); unknown pixels far from
    any known one keep a plain blur, and are replaced by the fill anyway."""
    den = _gauss(known, sigma)[..., None]
    num = _gauss(img * known[..., None], sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), _gauss(img, sigma)).astype(np.float32)


def _donor_patch(
    img: np.ndarray, skin: np.ndarray, hole: np.ndarray, box: tuple[int, int, int, int], size: int, sigma: float, rank: int,
) -> np.ndarray | None:
    """Fine texture (above `sigma`) of the healthy patch that best matches the mark's rim, from a
    window wide enough for every offset the search tries (box in image pixels). `hole` marks
    every repair and every detected feature or mark: a donor's pixels on one are filled from the
    donor's clean texture around them, so no stubble dot, freckle or blackhead is stamped into
    the repair."""
    x, y, bw, bh = box
    h, w = img.shape[:2]
    dist = max(4, int(size * 1.3))
    pad = int(math.ceil(3.6 * dist + size / 3 + 2 * sigma)) + 2
    y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
    crop = img[y0:y1, x0:x1].astype(np.float32)
    low = _gauss(crop, sigma)
    ly, sk = _luma(low), skin[y0:y1, x0:x1]
    taken = cv2.dilate(hole[y0:y1, x0:x1], _ellipse(3)).astype(np.float32)
    b = (x - x0, y - y0, bw, bh)
    off = _ranked_donor(b, ly, sk > 0.6, taken, dist, rank)
    if off is None:
        off = _ranked_donor(b, ly, sk > 0.3, taken, dist, rank)
    if off is None:
        return None
    dy, dx = off
    yy, xx = y - y0 + dy, x - x0 + dx
    patch = (crop - low)[yy:yy + bh, xx:xx + bw]
    bad = taken[yy:yy + bh, xx:xx + bw]
    if bad.any():
        # Fill the overlap from the patch's clean texture (pores mirrored in from around it).
        clean = 1.0 - bad
        fill = push_pull(patch, clean)
        turn = (lambda a: a[::-1, ::-1]) if (x + y) % 2 else (lambda a: a[::-1, :])  # noqa: E731
        tex = np.where(turn(clean)[..., None] > 0, turn(patch), fill)
        patch = np.where(bad[..., None] > 0, tex, patch)
    return patch


def _correct_marks(
    image: np.ndarray, skin: np.ndarray, marks: list[Mark], settings: SkinPerfectionSettings, F: float,
    edge_guard: np.ndarray | None, avoid: list[Mark] = (),
) -> np.ndarray:
    """Heal / pigment-correct the given marks at full resolution, cluster by cluster. `avoid`:
    every other detected mark and feature, never used as donor skin."""
    if not marks:
        return image
    h, w = image.shape[:2]
    overall = settings.strength / 100.0
    tp = settings.texturePreservation / 100.0
    pp = settings.porePreservation / 100.0
    heal = np.zeros((h, w), np.float32)  # strength where the texture is rebuilt
    pig = np.zeros((h, w), np.float32)  # strength where only the tone is rebuilt
    hole = np.zeros((h, w), np.uint8)
    core = np.zeros((h, w), np.float32)  # the marks themselves: their fine band is the mark, not skin
    sf_ = max(0.7, 0.0012 * F)
    for m in marks:
        R = _repair_radius(m, settings)
        cat = settings.category("blemish" if m.kind == "manual" else m.kind)
        st = float(np.clip(cat.strength / 100.0 * overall, 0.0, 1.0))
        if st <= 0:
            continue
        canvas = heal if m.kind in HEAL_KINDS else pig
        cv2.circle(canvas, (int(round(m.x)), int(round(m.y))), int(math.ceil(R)), st, -1, lineType=cv2.LINE_AA)
        cv2.circle(hole, (int(round(m.x)), int(round(m.y))), int(math.ceil(R)), 1, -1)
        if m.kind in SHARP_KINDS:
            # A sharp mark's fine band is its own edge. A small one is all edge; on a large one only
            # the rim is replaced, so the pores and follicles inside it are kept.
            c = (int(round(m.x)), int(round(m.y)))
            cv2.circle(core, c, int(math.ceil(min(1.6 * m.r, R))), 1.0, -1, lineType=cv2.LINE_AA)
            if m.r > SHARP_SMALL * sf_:
                cv2.circle(core, c, int(0.6 * m.r), 0.0, -1, lineType=cv2.LINE_AA)
    if not hole.any():
        return image
    taken = hole.copy()
    for m in avoid:
        cv2.circle(taken, (int(round(m.x)), int(round(m.y))), int(math.ceil(1.6 * m.r)) + 1, 1, -1)
    out = image.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    # A cluster holding a mark whose last repair failed a naturalness check is retried with the
    # next-best donor patch.
    rank = np.zeros(n, np.int32)
    for m in marks:
        if _natural_failures(m):
            lab = labels[min(h - 1, max(0, int(round(m.y)))), min(w - 1, max(0, int(round(m.x))))]
            rank[lab] = max(rank[lab], m.attempts)
    sf = max(0.7, 0.0012 * F)  # pore band
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        size = max(bw, bh)
        pad = max(6, int(1.6 * size))
        cy0, cy1, cx0, cx1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
        # Read from `out`: a neighbouring cluster's window may already hold a repair.
        crop = out[cy0:cy1, cx0:cx1].astype(np.float32)
        sel = labels[cy0:cy1, cx0:cx1] == i
        known = (~sel).astype(np.float32)
        sigma = max(0.8, 0.3 * size)
        # Tone from the skin around the mark only: a plain blur spreads the mark into the ring the
        # fill is built from, and the repair comes out tinted by the mark it replaced.
        low = _known_blur(crop, known, sigma)
        high = crop - low
        low_fill = _cap_fill(low, push_pull(low, known), sel)
        # The fine band (pores) inside the repair is kept, except over the mark itself: there it
        # is the mark's own sharp edge, and keeping it leaves a ghost ring.
        mcore = _gauss(core[cy0:cy1, cx0:cx1], 0.6)[..., None] * sel[..., None]
        low_p = _gauss(crop, sf)
        fine_o = crop - low_p
        # Heal marks: rebuilt tone + texture from the best healthy patch nearby.
        sk = skin[cy0:cy1, cx0:cx1]
        texture = push_pull(high, known)  # smooth fallback (no grain)
        donor_high = _donor_patch(out, skin, taken, (x, y, bw, bh), size, sigma, int(rank[i]))
        fine_fill = push_pull(fine_o, known * (1.0 - mcore[..., 0]))  # mark core: no texture at worst
        if donor_high is not None:
            donor = np.zeros_like(high)
            donor[y - cy0:y - cy0 + bh, x - cx0:x - cx0 + bw] = donor_high
            fine_d = donor - _gauss(donor, sf)
            fine_fill = np.where(sel[..., None], fine_d, fine_o)
            # Pore preservation: on marks bigger than a pore, keep the original pore band around
            # the mark (not over it).
            big = float(_smoothstep(3.0 * sf, 6.0 * sf, np.array(0.5 * size)))
            donor = donor + (fine_o - fine_d) * (pp * big) * (1.0 - mcore)
            texture = texture * (1.0 - tp) + donor * tp
            texture = np.where(sel[..., None], texture, high)
        # Pigment marks: rebuild everything above the pore scale; the pores around the mark stay,
        # and over the mark the donor's pores stand in for the mark's edge.
        fine_keep = fine_o * (1.0 - mcore) + fine_fill * mcore
        pig_out = _cap_fill(low_p, push_pull(low_p, known), sel) + fine_keep
        heal_out = low_fill + texture
        own = sel.astype(np.float32)  # this cluster only; neighbours get their own pass
        wh = _gauss(heal[cy0:cy1, cx0:cx1] * own, max(0.6, 0.12 * size))
        wp = _gauss(pig[cy0:cy1, cx0:cx1] * own, max(0.6, 0.12 * size))
        if edge_guard is not None:
            g = edge_guard[cy0:cy1, cx0:cx1]
            wh, wp = wh * g, wp * g
        wh, wp = wh[..., None], wp[..., None]
        res = crop + (heal_out - crop) * wh
        res = res + (pig_out - res) * wp * (1.0 - wh)
        out[cy0:cy1, cx0:cx1] = res
    return np.clip(out, 0.0, 1.0, out=out)


def _acne_halo(image: np.ndarray, skin: np.ndarray, marks: list[Mark], F: float, amount: float) -> np.ndarray:
    """Post-acne redness around corrected acne: pull excess Cr toward the skin around it."""
    red_marks = [m for m in marks if m.kind in ("acne", "pimple", "postAcne")]
    if amount <= 0 or not red_marks:
        return image
    h, w = image.shape[:2]
    halo = np.zeros((h, w), np.float32)
    for m in red_marks:
        cv2.circle(halo, (int(round(m.x)), int(round(m.y))), int(math.ceil(m.r * 3.0 + 2)), 1.0, -1)
    ys, xs = np.nonzero(halo)
    pad = int(0.06 * F) + 4
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = image[y0:y1, x0:x1]
    sk = skin[y0:y1, x0:x1].astype(np.float32)
    hl = _gauss(halo[y0:y1, x0:x1], max(1.0, 0.004 * F)) * sk
    y = _luma(crop)
    cb, cr = _cbcr(crop)
    base_cr = _masked_blur(cr, sk * (1.0 - halo[y0:y1, x0:x1]), 0.04 * F)
    excess = np.maximum(_gauss(cr, max(0.8, 0.002 * F)) - base_cr, 0.0)
    out = image.copy()
    out[y0:y1, x0:x1] = np.clip(_from_ycbcr(y, cb, cr - excess * 0.85 * (amount / 100.0) * hl), 0.0, 1.0)
    return out


def smooth_relief(image: np.ndarray, skin: np.ndarray, F: float, amount: float, sensitivity: float, pore_keep: float) -> np.ndarray:
    """Uneven texture and shallow acne scars: the relief band (bumps and pits a few pores wide)
    is reduced where its local energy is well above the person's typical skin, so rough patches
    calm down while even skin and the pore band (finer than the relief band) are untouched."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not (skin > 0.3).any():
        return image
    ys, xs = np.nonzero(skin > 0.05)
    h, w = skin.shape
    s_hi, s_lo = max(0.8, 0.0025 * F), max(2.0, 0.012 * F)
    pad = int(3 * s_lo) + 2
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = image[y0:y1, x0:x1]
    sk = skin[y0:y1, x0:x1].astype(np.float32)
    band = _gauss(crop, s_hi)
    band -= _gauss(crop, s_lo)
    e = np.sqrt(np.maximum(_fast_masked_blur(_luma(band) ** 2, sk, 3 * s_lo), 1e-10))
    typical = float(np.median(e[sk > 0.5])) if (sk > 0.5).any() else float(np.median(e))
    ratio = e / max(typical, 1e-6)
    lo = 2.2 - 0.9 * sensitivity / 100.0  # where reduction starts (x typical energy)
    rough = _smoothstep(lo, lo + 0.8, ratio)
    del e, ratio
    # Oily highlights look "rough" only because the shine lights up every pore: leave them.
    yl = _luma(crop)
    gloss = _fast_gauss(yl, max(2.0, 0.008 * F)) / np.maximum(_fast_masked_blur(yl, sk, max(4.0, 0.06 * F)), 1e-3)
    rough = rough * (1.0 - _smoothstep(1.06, 1.2, gloss))
    del yl, gloss
    k = (t * 0.8 * rough * sk)[..., None]
    del rough
    out = image.copy()
    # Pores live above s_hi and are untouched; pore_keep only matters for very coarse pores.
    band *= k
    band *= 1.0 - 0.3 * pore_keep
    np.subtract(crop, band, out=band)  # in place: on a full-length shot the crop is most of the photo
    out[y0:y1, x0:x1] = np.clip(band, 0.0, 1.0, out=band)
    return out


def lift_under_eye(image: np.ndarray, under_eye: np.ndarray, F: float, amount: float) -> np.ndarray:
    """Under-eye darkness: luma lifted toward the cheek below it and the colour cast pulled to
    it, on the tone layer only (lash line, creases and pores stay)."""
    t = float(np.clip(amount / 100.0, 0.0, 1.0))
    if t <= 0 or not (under_eye > 1e-3).any():
        return image
    ys, xs = np.nonzero(under_eye > 1e-3)
    h, w = under_eye.shape
    pad = int(0.08 * F) + 4
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + 2 * pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = image[y0:y1, x0:x1]
    region = under_eye[y0:y1, x0:x1]
    sigma = max(1.0, 0.012 * F)
    tone = _gauss(crop, sigma)
    # Reference: the skin just below the region (cheek), shifted up.
    shift = int(0.06 * F) + 1
    ref = np.roll(_gauss(crop, 2 * sigma), -shift, axis=0)
    dy = np.maximum(_luma(ref) - _luma(tone), 0.0)
    k = (t * 0.8 * region)
    y = _luma(crop) + dy * k
    cb, cr = _cbcr(crop)
    cb_r, cr_r = _cbcr(ref)
    cb_t, cr_t = _cbcr(tone)
    out = image.copy()
    out[y0:y1, x0:x1] = np.clip(_from_ycbcr(y, cb + (cb_r - cb_t) * 0.6 * k, cr + (cr_r - cr_t) * 0.6 * k), 0.0, 1.0)
    return out


def _fine_energy(image: np.ndarray, marks: list[Mark], F: float) -> float:
    """Pore-band energy in rings around the given marks (texture-retention check)."""
    if not marks:
        return 0.0
    y = _luma(image)
    fine = y - _gauss(y, max(0.8, 0.0025 * F))
    tot, cnt = 0.0, 0
    h, w = y.shape
    for m in marks[:400]:
        R = max(2.0, m.r)
        r0 = int(math.ceil(3.0 * R))
        x0, x1, y0, y1 = max(0, int(m.x) - r0), min(w, int(m.x) + r0 + 1), max(0, int(m.y) - r0), min(h, int(m.y) + r0 + 1)
        f = fine[y0:y1, x0:x1]
        tot += float((f * f).sum())
        cnt += f.size
    return tot / max(cnt, 1)


# =============================================================================================
# Driver: passes, residual scans, verification and user overrides
# =============================================================================================


def _needs_own_decision(m: Mark) -> bool:
    """A protected natural feature, or a mark under review because it may be one."""
    return m.status == "protected" or (m.status == "review" and m.reason not in (None, "lowConfidence"))


def _verify(
    image: np.ndarray, out: np.ndarray, skin: np.ndarray, todo: list[Mark], earlier: list[Mark],
    settings: SkinPerfectionSettings, F: float, features: list[Mark] = (), count: bool = True,
) -> None:
    """The quality gate (perfection_quality): a mark is corrected only if its repair passes every
    check. A failure leaves it "remaining", so the next pass retries it (with another donor
    when a naturalness check failed)."""
    h, w = out.shape[:2]
    holes = np.zeros((h, w), np.uint8)
    for m in todo + earlier:
        cv2.circle(holes, (int(round(m.x)), int(round(m.y))), int(math.ceil(_repair_radius(m, settings))) + 1, 1, -1)
    for m in features:  # moles, freckles, marks left alone: not "the skin around" a repair
        cv2.circle(holes, (int(round(m.x)), int(round(m.y))), int(math.ceil(1.6 * m.r)) + 1, 1, -1)
    tk, pk = settings.texturePreservation / 100.0, settings.porePreservation / 100.0
    sigs, xs, ys, rs, quals = [], [], [], [], []
    for m in todo:
        m.attempts += int(count)
        m.after = _mark_contrast(out, skin, m)
        R = _repair_radius(m, settings)
        residual = None
        if m.kind != "manual" and m.contrast > 0:
            fine = replace(m, r=FINE_RESIDUAL * m.r)
            if m.fine0 is None:
                m.fine0 = _mark_contrast(image, skin, fine)
            floor = 0.5 * MIN_REL_CONTRAST
            residual = (m.after / max(VERIFY_RATIO * m.contrast, floor),
                        _mark_contrast(out, skin, fine) / max(VERIFY_RATIO * m.fine0, floor))
        q = Q.assess(Q.window(image, out, skin, holes, m.x, m.y, R), R, F, residual, tk, pk)
        quals.append(q)
        sigs.append(Q.texture_signature(out, m.x, m.y, R, F))
        xs.append(m.x), ys.append(m.y), rs.append(R)
    # Repetition, against every repair so far (this pass and earlier ones).
    for m in earlier:
        R = _repair_radius(m, settings)
        sigs.append(Q.texture_signature(out, m.x, m.y, R, F))
        xs.append(m.x), ys.append(m.y), rs.append(R)
    rep_ = Q.repetition(sigs, xs, ys, rs)
    for m, q, r in zip(todo, quals, rep_):
        q.values["repetition"] = r
        q.failed = Q.failures(q.values, tk, pk)
        m.quality = q.to_json()
        m.status = "corrected" if not q.failed else "remaining"


def _restore(image: np.ndarray, out: np.ndarray, failed: list[Mark], kept: list[Mark], settings: SkinPerfectionSettings) -> np.ndarray:
    """Put the original pixels back over failed repairs, but not over repairs that passed."""
    h, w = out.shape[:2]
    if not failed:
        return out
    sig = max(0.6, 0.15 * float(np.median([_repair_radius(m, settings) for m in failed])))
    # Only the box around the failed marks changes (plus the blur's reach), so the masks and
    # the blend are built there, not on the whole photo.
    reach = int(math.ceil(4.0 * sig)) + 2
    rad = [int(math.ceil(_repair_radius(m, settings) * 1.25)) + 1 for m in failed]
    y0 = max(0, min(int(round(m.y)) - r for m, r in zip(failed, rad)) - reach)
    y1 = min(h, max(int(round(m.y)) + r for m, r in zip(failed, rad)) + reach + 1)
    x0 = max(0, min(int(round(m.x)) - r for m, r in zip(failed, rad)) - reach)
    x1 = min(w, max(int(round(m.x)) + r for m, r in zip(failed, rad)) + reach + 1)
    if y1 <= y0 or x1 <= x0:
        return out
    back = np.zeros((y1 - y0, x1 - x0), np.float32)
    keep = np.zeros_like(back)
    for m, r in zip(failed, rad):
        cv2.circle(back, (int(round(m.x)) - x0, int(round(m.y)) - y0), r, 1.0, -1)
    for m in kept:
        cv2.circle(keep, (int(round(m.x)) - x0, int(round(m.y)) - y0), int(math.ceil(_repair_radius(m, settings))) + 1, 1.0, -1)
    if not back.any():
        return out
    back = np.clip(_gauss(back, sig) * (1.0 - _gauss(keep, sig)), 0.0, 1.0)[..., None]
    res = out.copy()
    o = out[y0:y1, x0:x1]
    res[y0:y1, x0:x1] = o + (image[y0:y1, x0:x1] - o) * back
    return res


def _restore_core(image: np.ndarray, out: np.ndarray, marks: list[Mark], feather: float = 1.5) -> np.ndarray:
    """Original pixels inside each mark's radius (exactly), blended back over `feather` px outside."""
    h, w = out.shape[:2]
    res = out if out is not image else out.copy()
    for m in marks:
        R = m.r + feather
        x0, x1 = max(0, int(math.floor(m.x - R))), min(w, int(math.ceil(m.x + R)) + 1)
        y0, y1 = max(0, int(math.floor(m.y - R))), min(h, int(math.ceil(m.y + R)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - m.x) ** 2 + (yy - m.y) ** 2)
        wgt = np.clip((R - d) / feather, 0.0, 1.0)[..., None]
        o = res[y0:y1, x0:x1]
        res[y0:y1, x0:x1] = np.where(wgt >= 1.0, image[y0:y1, x0:x1], o + (image[y0:y1, x0:x1] - o) * wgt)
    return res


def _apply_overrides(marks: list[Mark], overrides: list[PerfectionOverride], w: int, h: int, next_id: int) -> tuple[list[Mark], int]:
    L = float(max(w, h))
    for o in overrides:
        ox, oy, orad = o.x * w, o.y * h, max(1.5, o.radius * L)
        hit = [m for m in marks if (m.x - ox) ** 2 + (m.y - oy) ** 2 <= (orad + m.r) ** 2]
        if o.action == "ignore":
            for m in hit:
                m.status = "ignored"
        elif o.bulk:
            # "Correct all remaining" (B1): it retries ordinary marks only, and its repairs go
            # through the quality gate like any automatic repair (undone if one check fails).
            # Never a protected feature or a mark that needs its own decision, even the nearest
            # one; never a mark whose centre lies outside the circle; and no "manual" spot where
            # nothing is detected any more (a bulk choice is not a hand-placed correction).
            for m in hit:
                if _needs_own_decision(m) or m.forced or (m.x - ox) ** 2 + (m.y - oy) ** 2 > orad * orad:
                    continue
                if m.status != "corrected":
                    m.status, m.reason, m.bulk = "forced", None, True
        else:
            if hit:
                for m in hit:
                    m.bulk = False  # the user's own choice wins over an earlier bulk one
                # The mark nearest the override's centre is the one the user chose, so it is forced
                # whatever it is. A neighbour inside the circle is forced only if it is an ordinary
                # mark: a protected feature, or one waiting for review because it may be one, needs
                # its own decision (one "Correct all remaining" override per blemish must not
                # correct the follicle next to it). Nor is a neighbour that only touches the circle:
                # its centre must be inside it (a failed repair next to a remaining blemish is
                # measured afresh, and must not be corrected because its neighbour was).
                target = min(hit, key=lambda m: (m.x - ox) ** 2 + (m.y - oy) ** 2)
                for m in hit:
                    if m is not target and (_needs_own_decision(m) or (m.x - ox) ** 2 + (m.y - oy) ** 2 > orad * orad):
                        continue
                    m.status, m.reason, m.forced = "forced", None, True
            else:
                marks.append(Mark(next_id, ox, oy, orad, "manual", 1.0, "face", 0.0, status="forced", forced=True))
                next_id += 1
    return marks, next_id


def perfect_skin(
    image: np.ndarray,
    skin: np.ndarray,
    faces: list[FaceAnalysis],
    settings: SkinPerfectionSettings,
    *,
    body_skin: np.ndarray | None = None,
    under_eye: np.ndarray | None = None,
    face_skin: np.ndarray | None = None,
) -> tuple[np.ndarray, PerfectionReport]:
    """Run the Skin Perfection engine. image: HxWx3 float32 sRGB 0..1 (not modified).

    skin: HxW face (and any user-added) skin weight; body_skin: optional HxW body-skin weight
    (used when settings.bodyCoverage); under_eye: optional HxW under-eye region; face_skin:
    HxW weight that marks "face" for the body / face split (defaults to skin).
    """
    t0 = time.perf_counter()
    h, w = image.shape[:2]
    rep = PerfectionReport(width=w, height=h)
    if not settings.enabled or settings.strength <= 0:
        return image, rep
    face_w = face_skin if face_skin is not None else skin
    region = skin.astype(np.float32)
    body = np.zeros((h, w), np.float32)
    if settings.bodyCoverage and body_skin is not None:
        body = np.clip(body_skin.astype(np.float32) - face_w, 0.0, 1.0)
        region = np.maximum(region, body)
    F = face_width_px(faces, region, w)
    rep.face_width = F
    if not (region > 0.3).any():
        rep.ms = (time.perf_counter() - t0) * 1000
        return image, rep
    mult = int(settings.multiplier)
    # Edge guard: strong structure (nose side, jaw line, hairline, clothing edges) damps repairs.
    ep = settings.edgeProtection / 100.0
    edge_guard = guard = None
    if ep > 0:
        ys_, xs_ = np.nonzero(region > 0.05)
        guard = np.ones((h, w), np.float32)
        pad = int(0.05 * F) + 4
        y0, y1, x0, x1 = max(0, ys_.min() - pad), min(h, ys_.max() + pad + 1), max(0, xs_.min() - pad), min(w, xs_.max() + pad + 1)
        yl = _luma(_gauss(image[y0:y1, x0:x1], max(1.0, 0.01 * F)))
        g = np.hypot(cv2.Sobel(yl, cv2.CV_32F, 1, 0), cv2.Sobel(yl, cv2.CV_32F, 0, 1)) / 8.0 * max(1.0, 0.01 * F)
        g = g / max(float(np.median(yl)), 0.05)
        guard[y0:y1, x0:x1] = 1.0 - ep * _smoothstep(0.2, 0.6, g)
        # The mask edge itself (skin meets hair, cloth, background) is never repaired over.
        # (Closed only over spot-sized holes, so the gap between two fingers is never bridged.)
        guard *= np.clip(cv2.morphologyEx(region, cv2.MORPH_CLOSE, _ellipse(max(3, int(0.015 * F) | 1))) * 1.6, 0.0, 1.0)
        edge_guard = guard

    scan_cache: dict = {}
    raw, ana = _analyse(image, region, body, faces, F, settings, cache=scan_cache)
    if ana is not None:
        rep.tiles, rep.work_scale = ana.tiles, ana.scale
    marks: list[Mark] = []
    thr = settings.confidenceThreshold / 100.0
    for i, (x, y, r, kind, conf, is_body, contrast, why) in enumerate(raw):
        status = "detected"
        if kind in PROTECTED_KINDS:
            keep = {"mole": settings.preserveMoles, "freckle": settings.preserveFreckles,
                    "birthmark": settings.preserveBirthmarks, "follicle": True}[kind]
            if keep:
                status = "protected"
            else:
                kind = {"mole": "darkSpot", "freckle": "darkSpot", "birthmark": "darkSpot"}[kind]
        if status == "detected" and why:
            # Ambiguous (could be a natural feature): kept, shown for review with the reason.
            status = "review"
        elif status == "detected" and conf < thr:
            # Low confidence: preserved by default, shown for review.
            status, why = "review", "lowConfidence"
        marks.append(Mark(i, x, y, r, kind, conf, "body" if is_body else "face", contrast, status=status,
                          reason=why if status == "review" else None))
    next_id = len(marks)
    marks, next_id = _apply_overrides(marks, settings.overrides, w, h, next_id)

    # Protected natural features (moles, freckles, birthmarks, follicles), and marks waiting for a
    # decision because they may be one, are kept out of every repair window and every area
    # correction, so nothing nearby can wash them out.
    keep = np.zeros((h, w), np.float32)
    for m in marks:
        if _needs_own_decision(m):
            cv2.circle(keep, (int(round(m.x)), int(round(m.y))), int(math.ceil(1.6 * m.r + 1)), 1.0, -1, lineType=cv2.LINE_AA)
    if keep.any():
        keep = _gauss(keep, max(0.8, 0.002 * F))
        edge_guard = (edge_guard if edge_guard is not None else np.ones((h, w), np.float32)) * (1.0 - keep)
    area_region = region * (1.0 - keep)
    del keep, guard  # folded into edge_guard and area_region (each a full-size float mask)

    out = image
    before_energy = None
    for p in range(settings.passes):
        todo = [m for m in marks if m.status in ("detected", "forced") or (m.status == "remaining" and m.attempts < MAX_ATTEMPTS)]
        if not todo:
            break
        if before_energy is None:
            before_energy = _fine_energy(image, todo, F)
        # A retry after a failed quality check is a fresh repair from the original skin (with
        # another donor), not a repair of the failed repair: stacking would wear away the pores
        # and follicles inside the mark each time.
        # It is also wider: a ghost ring or seam usually means the mark reaches past its measured
        # radius, so half its edge was left in the blend.
        retry = [m for m in todo if _natural_failures(m)]
        if retry:
            out = _restore(image, out, retry, [m for m in marks if m.status == "corrected"], settings)
            for m in retry:
                if m.r0 is None:
                    m.r0 = m.r
                m.r = min(RESIDUE_GROW * m.r0, RETRY_GROW * m.r)
        prev = out if p > 0 else None
        out = _correct_marks(out, region, todo, settings, F, edge_guard, [m for m in marks if m not in todo])
        rep.passes = p + 1
        # The first residual scan looks at the whole face again, more sensitively (it finds marks
        # the first detection missed, most often on deep skin). Later scans look for what the
        # repairs left: where a pass changed nothing, the last scan already saw these pixels, and a
        # new hit there is its tile statistics shifting around a repair, not a new mark.
        changed = None
        if p > 0:
            diff = np.zeros((h, w), np.uint8)
            for r0 in range(0, h, 256):  # in strips: a full-size difference would be another 345 MB at 30 MP
                diff[r0:r0 + 256] = np.abs(out[r0:r0 + 256] - prev[r0:r0 + 256]).max(axis=2) > 1e-4
            changed = cv2.dilate(diff, np.ones((3, 3), np.uint8), iterations=max(1, int(round(0.01 * F)))) > 0
        prev = None
        _verify(image, out, region, todo, [m for m in marks if m.attempts > 0 and m not in todo], settings, F,
                [m for m in marks if m.attempts == 0 and m not in todo])
        # H. Residual scan of the corrected image (more sensitive), new marks join the next pass.
        boost = 20.0 * settings.residualIntensity / 100.0
        res, _ = _analyse(out, region, body, faces, F, settings, sens_boost=boost, cache=scan_cache)
        for (x, y, r, kind, conf, is_body, contrast, why) in res:
            if kind in PROTECTED_KINDS:
                continue
            near = [m for m in marks if (m.x - x) ** 2 + (m.y - y) ** 2 <= (max(m.r, r) * 1.2) ** 2]
            if near:
                # A known mark: its status comes from measuring it (m.after), not from this hit.
                continue
            # Residue: a hit at the edge of a mark that was already corrected (the rest of a
            # pimple's halo, a dark rim the repair left) is that mark, not a new one. The mark
            # is widened to cover it and goes back for another pass, or is reported remaining.
            owners = [m for m in marks if m.attempts > 0 and m.kind not in PROTECTED_KINDS
                      and (m.x - x) ** 2 + (m.y - y) ** 2 <= (RESIDUE_REACH * m.r + r) ** 2]
            if owners:
                m = min(owners, key=lambda o: (o.x - x) ** 2 + (o.y - y) ** 2)
                if m.r0 is None:
                    m.r0 = m.r
                m.r = min(RESIDUE_GROW * m.r0, max(m.r, math.hypot(m.x - x, m.y - y) + r))
                if m.status == "corrected":
                    m.status = "remaining"
                continue
            if changed is not None and not changed[min(h - 1, max(0, int(round(y)))), min(w - 1, max(0, int(round(x))))]:
                continue
            st = "review" if why or conf < thr else "detected"
            if p + 1 >= settings.passes and st == "detected":
                st = "remaining"  # no pass left to correct it
            marks.append(Mark(next_id, x, y, r, kind, conf, "body" if is_body else "face", contrast, status=st, residual=True,
                              reason=(why or "lowConfidence") if st == "review" else None))
            next_id += 1
        marks, _ = _apply_overrides(marks, [o for o in settings.overrides if o.action == "ignore"], w, h, next_id)
    for m in marks:
        if m.status in ("detected", "forced"):
            m.status = "remaining"
    # A repair that still fails a naturalness check after its last attempt is not "corrected": it
    # is undone (the original skin comes back) and the mark waits for review with the failed
    # checks. A mark the user chose to correct keeps its repair and shows the checks it failed.
    # A mark "Correct all remaining" chose must pass every check, "still visible" included: a
    # bulk repair that is only partly right is undone too (B1).
    failed = [m for m in marks if m.status == "remaining" and (_natural_failures(m) or (m.bulk and m.attempts > 0))]
    for m in failed:
        if m.bulk and m.quality is not None and not m.quality["failed"]:
            m.quality["failed"] = ["residual"]  # it passed the checks, then the rescan still found some of it
    undo = [m for m in failed if not m.forced]
    if undo:
        out = _restore(image, out, undo, [m for m in marks if m.status == "corrected" or (m.forced and m in failed)], settings)
    for m in failed:
        m.quality["kept"] = m.forced
        if m.forced:
            m.status = "corrected"
        else:
            m.status, m.reason = "review", "repairFailed"
        m.after = _mark_contrast(out, region, m)

    corrected = [m for m in marks if m.status in ("corrected", "remaining") and m.attempts > 0]
    if before_energy:
        rep.texture_retention = _fine_energy(out, corrected, F) / max(before_energy, 1e-12)

    # Area corrections (after the marks, so they see clean skin).
    def amt(name: str) -> float:
        c = settings.category(name)
        return 0.0 if not c.enabled else float(np.clip(c.strength * settings.strength / 100.0 * MULTIPLIER_AREA[mult], 0.0, 100.0))

    if amt("redness") > 0:
        out = _acne_halo(out, area_region, [m for m in marks if m.attempts > 0], F, amt("redness"))
        a_red = amt("redness") * settings.category("redness").sensitivity / 100.0
        out = _lowres(lambda im, sk, f: reduce_redness(im, sk, a_red, sigma_tone=max(1.0, 0.012 * f), face_width=f), out, [area_region], F)
        rep.areas["redness"] = amt("redness") / 100.0
    if amt("discoloration") > 0:
        disc = amt("discoloration") * (0.6 if settings.preserveFreckles else 1.0)
        a_disc = disc * settings.category("discoloration").sensitivity / 70.0
        out = _lowres(lambda im, sk, f: correct_pigmentation(im, sk, max(1.0, 0.036 * f), a_disc), out, [area_region], F)
        rep.areas["discoloration"] = disc / 100.0
    tex = max(amt("texture"), amt("scar"))
    if tex > 0:
        sens = max(settings.category("texture").sensitivity, settings.category("scar").sensitivity)
        out = smooth_relief(out, area_region, F, tex, sens, settings.porePreservation / 100.0)
        rep.areas["texture"] = tex / 100.0
    if amt("underEye") > 0 and under_eye is not None:
        out = lift_under_eye(out, under_eye, F, amt("underEye"))
    # A mark waiting for review is untouched by every tool, the area tools included: the tone
    # tool would otherwise lighten a restored dark spot (or a freckle) the engine was unsure of.
    waiting = [m for m in marks if m.status == "review"]
    if waiting:
        out = _restore(image, out, waiting, [m for m in marks if m.status == "corrected"], settings)
    # A failed bulk repair is undone completely (B1): the restore above gives way to a corrected
    # neighbour's feathered edge, which can leave part of that neighbour's repair over this mark.
    # Its own disc gets the original pixels back exactly, with a 1.5 px feather outside it.
    bulk_failed = [m for m in marks if m.bulk and m.reason == "repairFailed"]
    if bulk_failed:
        out = _restore_core(image, out, bulk_failed)
    # A repair left "remaining" only because some of the mark's colour was left is measured again
    # once the redness and tone tools have run: they often finish it.
    unfinished = [m for m in marks if m.status == "remaining" and m.attempts > 0 and m.quality
                  and set(m.quality["failed"]) <= {"colour", "residual"} and m.quality.get("incomplete", True)]
    if unfinished:
        _verify(image, out, region, unfinished, [m for m in marks if m.status == "corrected"], settings, F,
                [m for m in marks if m.attempts == 0], count=False)
    for m in marks:
        if m.attempts > 0 and m.quality:
            for k in m.quality["failed"]:
                rep.quality[k] = rep.quality.get(k, 0) + 1
        rep.areas["underEye"] = amt("underEye") / 100.0
    rep.marks = marks
    # Nothing outside skin (clothing, hair, jewellery, background) ever changes, not even by
    # rounding: pixels there are copied back bit for bit. User-forced corrections are kept.
    allowed = region > 1e-3
    for m in marks:
        if m.kind == "manual" and m.attempts > 0:
            cv2.circle(allowed.view(np.uint8), (int(round(m.x)), int(round(m.y))), int(math.ceil(m.r * 2.5 + 2)), 1, -1)
    out = out.copy() if out is image else out  # the input is never modified
    np.clip(out, 0.0, 1.0, out=out)
    np.copyto(out, image, where=~allowed[..., None])
    rep.ms = (time.perf_counter() - t0) * 1000
    return out.astype(np.float32, copy=False), rep
