"""Pydantic mirror of packages/shared/src/index.ts. Keep the two in sync."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]


TONE_QUANTILE_COUNT = 17


class ColorStats(BaseModel):
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    # Optional row-major 3x3 L*a*b* covariance and L* quantiles (see packages/shared).
    cov: list[float] | None = Field(None, min_length=9, max_length=9)
    lQuantiles: list[float] | None = Field(None, min_length=TONE_QUANTILE_COUNT, max_length=TONE_QUANTILE_COUNT)


class ColorMatch(BaseModel):
    reference: ColorStats
    amount: float = Field(100, ge=0, le=100)


class StrayHairSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    strength: float = Field(60, ge=0, le=100)


class DepthBokehSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    blurRadius: float = Field(50, ge=0, le=100)
    bokehMultiplier: float = Field(100, ge=100, le=300)
    apertureShape: Literal["circle", "hex"] = "circle"
    highlightBoost: float = Field(40, ge=0, le=100)


class HslValue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hue: float = Field(0, ge=-100, le=100)
    saturation: float = Field(0, ge=-100, le=100)
    luminance: float = Field(0, ge=-100, le=100)


class HslMixer(BaseModel):
    """8-channel colour mixer; see app/color_engine.py."""

    model_config = ConfigDict(extra="ignore")

    red: HslValue = HslValue()
    orange: HslValue = HslValue()
    yellow: HslValue = HslValue()
    green: HslValue = HslValue()
    aqua: HslValue = HslValue()
    blue: HslValue = HslValue()
    purple: HslValue = HslValue()
    magenta: HslValue = HslValue()


class GradeWheel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hue: float = Field(0, ge=0, le=360)
    saturation: float = Field(0, ge=0, le=100)
    luminance: float = Field(0, ge=-100, le=100)


class ColorGrading(BaseModel):
    """3-way shadows / midtones / highlights wheels; see app/color_engine.py."""

    model_config = ConfigDict(extra="ignore")

    shadows: GradeWheel = GradeWheel()
    midtones: GradeWheel = GradeWheel()
    highlights: GradeWheel = GradeWheel()
    blending: float = Field(50, ge=0, le=100)
    balance: float = Field(0, ge=-100, le=100)


class ClothingSettings(BaseModel):
    """Clothing "virtual iron" and recolour (retouch_engine.remove_clothing_wrinkles / shift_clothing_color)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    wrinkleStrength: float = Field(60, ge=0, le=100)
    texturePreservation: float = Field(80, ge=0, le=100)
    hueShift: float = Field(0, ge=-180, le=180)
    saturation: float = Field(0, ge=-100, le=100)


LUT_MAX_SIZE = 33


class LookLut(BaseModel):
    """A 3D colour lookup table ("look"), e.g. fitted from a graded reference or loaded from a .cube.

    data: base64 of size^3 RGB triples as little-endian uint16 (0..65535), red index fastest
    (the .cube order). See app/color_engine.py decode_lut / apply_lut.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field("Look", max_length=120)
    size: int = Field(17, ge=2, le=LUT_MAX_SIZE)
    data: str = Field(..., max_length=(LUT_MAX_SIZE**3 * 6 * 4) // 3 + 8)
    amount: float = Field(100, ge=0, le=100)


class EffectControl(BaseModel):
    """Per-effect enable / multiplier / opacity (mirror of packages/shared/src/effects.ts)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    multiplier: Literal[1, 2, 3] = 1
    opacity: float = Field(100, ge=0, le=100)


class EffectControls(BaseModel):
    model_config = ConfigDict(extra="ignore")

    skinSmoothing: EffectControl = EffectControl()
    blemishRemoval: EffectControl = EffectControl()
    acneRemoval: EffectControl = EffectControl()
    darkSpotRemoval: EffectControl = EffectControl()
    rednessReduction: EffectControl = EffectControl()
    skinNoise: EffectControl = EffectControl()
    skinToneUnify: EffectControl = EffectControl()
    darkCircles: EffectControl = EffectControl()
    underEyeTexture: EffectControl = EffectControl()
    eyeBrighten: EffectControl = EffectControl()
    teethWhiten: EffectControl = EffectControl()
    teethStainRemoval: EffectControl = EffectControl()
    glareReduction: EffectControl = EffectControl()
    sharpen: EffectControl = EffectControl()
    strayHair: EffectControl = EffectControl()
    clothingIron: EffectControl = EffectControl()


MAX_MASK_STROKES = 200
MAX_STROKE_POINTS = 4000


class MaskStroke(BaseModel):
    """One brush stroke on the skin mask (mirror of packages/shared MaskStroke)."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["add", "erase"] = "add"
    size: float = Field(0.02, ge=0.001, le=0.2)  # brush radius, share of the long edge
    points: list[float] = Field(..., min_length=2, max_length=2 * MAX_STROKE_POINTS)  # normalised x, y pairs


class SkinMaskSettings(BaseModel):
    """Skin mask refinement (app/masking.py): AI face parsing and the user's brush strokes."""

    model_config = ConfigDict(extra="ignore")

    useSegmentation: bool = True
    includeBodySkin: bool = False
    feather: float = Field(50, ge=0, le=100)
    strokes: list[MaskStroke] = Field(default_factory=list, max_length=MAX_MASK_STROKES)


MAX_HEAL_STROKES = 300
MAX_LAYERS = 8
MAX_LAYER_STROKES = 200
MAX_TOOL_POINTS = 4000


class HealStroke(BaseModel):
    """Healing / clone brush stroke (mirror of packages/shared localRetouch.ts HealStroke)."""

    model_config = ConfigDict(extra="ignore")

    mode: Literal["heal", "clone"] = "heal"
    size: float = Field(0.02, ge=0.001, le=0.2)
    hardness: float = Field(50, ge=0, le=100)
    opacity: float = Field(100, ge=0, le=100)
    offset: tuple[float, float]  # source - destination, normalised to width / height
    points: list[float] = Field(..., min_length=2, max_length=2 * MAX_TOOL_POINTS)


class LayerStroke(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["add", "erase", "dodge", "burn"] = "add"
    size: float = Field(0.02, ge=0.001, le=0.2)
    flow: float = Field(100, ge=1, le=100)
    points: list[float] = Field(..., min_length=2, max_length=2 * MAX_TOOL_POINTS)


class LayerMask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base: Literal["none", "skin", "all"] = "none"
    feather: float = Field(50, ge=0, le=100)
    invert: bool = False
    lumaMin: float = Field(0, ge=0, le=100)
    lumaMax: float = Field(100, ge=0, le=100)
    lumaSoftness: float = Field(20, ge=0, le=100)
    density: float = Field(100, ge=0, le=100)
    strokes: list[LayerStroke] = Field(default_factory=list, max_length=MAX_LAYER_STROKES)


class RetouchLayer(BaseModel):
    """Local retouch layer (frequency separation, dodge & burn or correction brush)."""

    model_config = ConfigDict(extra="ignore")

    id: str = "layer"
    name: str = ""
    kind: Literal["frequency", "dodgeBurn", "correction"] = "correction"
    enabled: bool = True
    opacity: float = Field(100, ge=0, le=100)
    mask: LayerMask = LayerMask()
    splitRadius: float = Field(40, ge=0, le=100)
    toneSmoothing: float = Field(0, ge=0, le=100)
    texture: float = Field(100, ge=0, le=200)
    dodgeBurnStrength: float = Field(50, ge=0, le=100)
    dodgeBurnRange: Literal["all", "shadows", "midtones", "highlights"] = "all"
    exposure: float = Field(0, ge=-100, le=100)
    contrast: float = Field(0, ge=-100, le=100)
    saturation: float = Field(0, ge=-100, le=100)
    temperature: float = Field(0, ge=-100, le=100)
    tint: float = Field(0, ge=-100, le=100)


class LocalRetouch(BaseModel):
    """Pro tools (app/engines/local.py): healing / clone strokes and local retouch layers."""

    model_config = ConfigDict(extra="ignore")

    heal: list[HealStroke] = Field(default_factory=list, max_length=MAX_HEAL_STROKES)
    layers: list[RetouchLayer] = Field(default_factory=list, max_length=MAX_LAYERS)


# --- Skin Perfection engine (v0.13, app/engines/perfection.py) -------------------------------

PERFECTION_CATEGORIES = (
    "acne", "pimple", "blackhead", "whitehead", "darkSpot", "postAcne", "smallDot", "blemish",
    "discoloration", "scar", "texture", "redness", "underEye",
)
# Default (sensitivity, strength) per category. Discrete marks are removed at full strength;
# area corrections (tone, redness, texture, under-eye) default lower so skin keeps its life.
PERFECTION_DEFAULTS: dict[str, tuple[float, float]] = {
    "acne": (75, 100), "pimple": (75, 100), "blackhead": (70, 100), "whitehead": (70, 100),
    "darkSpot": (70, 100), "postAcne": (70, 90), "smallDot": (70, 100), "blemish": (75, 100),
    "discoloration": (60, 50), "scar": (50, 25), "texture": (50, 20), "redness": (60, 50),
    "underEye": (50, 35),
}
MAX_PERFECTION_OVERRIDES = 2000


class PerfectionCategory(BaseModel):
    """Detection sensitivity and correction strength of one imperfection category (0..100)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    sensitivity: float = Field(70, ge=0, le=100)
    strength: float = Field(100, ge=0, le=100)


def _default_categories() -> dict[str, PerfectionCategory]:
    return {k: PerfectionCategory(sensitivity=s, strength=t) for k, (s, t) in PERFECTION_DEFAULTS.items()}


class PerfectionOverride(BaseModel):
    """A user decision on one spot: ignore a detected mark, or correct this spot regardless."""

    model_config = ConfigDict(extra="ignore")

    x: float = Field(..., ge=0, le=1)  # normalised centre
    y: float = Field(..., ge=0, le=1)
    radius: float = Field(0.004, ge=0.0003, le=0.05)  # share of the long edge
    action: Literal["ignore", "correct"] = "ignore"
    # Set by "Correct all remaining": the correction must pass every quality check, or it is undone
    # (original pixels) and the mark becomes a failed repair. A hand-picked correction is kept.
    bulk: bool = False


class SkinPerfectionSettings(BaseModel):
    """Skin Perfection Studio (mirror of packages/shared/src/perfection.ts)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    strength: float = Field(100, ge=0, le=100)  # overall Skin Perfection strength
    multiplier: int = Field(1, ge=1, le=3)  # 1x natural, 2x strong, 3x maximum
    categories: dict[str, PerfectionCategory] = Field(default_factory=_default_categories)
    bodyCoverage: bool = True  # neck, chest, shoulders, arms, hands, back, legs
    bodySensitivity: float = Field(70, ge=0, le=100)
    texturePreservation: float = Field(90, ge=0, le=100)
    porePreservation: float = Field(90, ge=0, le=100)
    correctionRadius: float = Field(50, ge=0, le=100)
    edgeProtection: float = Field(60, ge=0, le=100)
    confidenceThreshold: float = Field(50, ge=0, le=100)
    residualIntensity: float = Field(70, ge=0, le=100)
    passes: int = Field(2, ge=1, le=4)
    preserveFreckles: bool = True
    preserveMoles: bool = True
    preserveBirthmarks: bool = True
    overrides: list[PerfectionOverride] = Field(default_factory=list, max_length=MAX_PERFECTION_OVERRIDES)

    def category(self, name: str) -> PerfectionCategory:
        if name in self.categories:
            return self.categories[name]
        s, t = PERFECTION_DEFAULTS.get(name, (70, 100))
        return PerfectionCategory(sensitivity=s, strength=t)


class FinishControl(BaseModel):
    """One Professional Skin Finish slider: its own on/off switch and amount (0..100)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    amount: float = Field(50, ge=0, le=100)


# The panel's styles (packages/shared/src/finish.ts FINISH_STYLES mirrors this): every amount.
FINISH_STYLES: dict[str, dict[str, float]] = {
    "natural": dict(smoothing=35, texturePreservation=85, porePreservation=80, shine=25, darkSpots=45, toneEvenness=25, underEye=25, dodgeBurn=15),
    "professional": dict(smoothing=55, texturePreservation=75, porePreservation=65, shine=45, darkSpots=70, toneEvenness=45, underEye=45, dodgeBurn=30),
    "advanced": dict(smoothing=80, texturePreservation=60, porePreservation=45, shine=60, darkSpots=95, toneEvenness=65, underEye=60, dodgeBurn=45),
}


class SkinFinishSettings(BaseModel):
    """Professional Skin Finish (v0.14, stage 0d; engines/finish.py). Off by default: with it off
    the renderer's output is bit-identical to a build without it.

    Texture (phase 1): smoothing (bumps and blotches), texturePreservation (grain and the texture
    restoration floor), porePreservation (pores) and the overall strength.
    Local corrections (phase 2, engines/finish_local.py): shine, darkSpots (post-acne marks, dark
    spots, small hyperpigmentation), toneEvenness, underEye and dodgeBurn. They are 0 unless set,
    so a finish saved with only the phase 1 controls renders as it did; the panel's styles
    (Natural, Professional, Advanced) set all of them. `style` records which one was picked
    ("custom" after any slider change)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    strength: float = Field(100, ge=0, le=100)  # Overall Skin Finish Strength: blends the whole finish
    smoothing: FinishControl = FinishControl(amount=55)
    texturePreservation: FinishControl = FinishControl(amount=75)
    porePreservation: FinishControl = FinishControl(amount=65)
    shine: FinishControl = FinishControl(amount=0)
    darkSpots: FinishControl = FinishControl(amount=0)
    toneEvenness: FinishControl = FinishControl(amount=0)
    underEye: FinishControl = FinishControl(amount=0)
    dodgeBurn: FinishControl = FinishControl(amount=0)
    style: str = "custom"
    bodyCoverage: bool = True  # neck, shoulders, arms, hands, legs (face parsing)

    @classmethod
    def styled(cls, style: str, **kw) -> "SkinFinishSettings":
        """On, with every amount from one of FINISH_STYLES."""
        return cls(enabled=True, style=style, **{k: FinishControl(amount=v) for k, v in FINISH_STYLES[style].items()}, **kw)


class Adjustments(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exposure: float = Field(0, ge=-100, le=100)
    contrast: float = Field(0, ge=-100, le=100)
    highlights: float = Field(0, ge=-100, le=100)
    shadows: float = Field(0, ge=-100, le=100)
    temperature: float = Field(0, ge=-100, le=100)
    tint: float = Field(0, ge=-100, le=100)
    saturation: float = Field(0, ge=-100, le=100)
    vibrance: float = Field(0, ge=-100, le=100)
    skinSmoothing: float = Field(0, ge=0, le=100)
    skinTexture: float = Field(50, ge=0, le=100)
    blemishRemoval: float = Field(0, ge=0, le=100)
    skinToneUnify: float = Field(0, ge=0, le=100)
    # v0.7 separate skin tools: acne (red spots, bumps, post-acne redness) and dark spots
    # (isolated dark marks + uneven pigmentation). blemishRemoval stays as general blemish removal.
    acneRemoval: float = Field(0, ge=0, le=100)
    darkSpotRemoval: float = Field(0, ge=0, le=100)
    keepFreckles: bool = False
    # v0.9 skin engine (app/engines/redness.py, noise.py, under_eye.py).
    rednessReduction: float = Field(0, ge=0, le=100)
    skinNoise: float = Field(0, ge=0, le=100)
    underEyeTexture: float = Field(0, ge=0, le=100)
    eyeBrighten: float = Field(0, ge=0, le=100)
    darkCircles: float = Field(0, ge=0, le=100)
    teethWhiten: float = Field(0, ge=0, le=100)
    sharpen: float = Field(0, ge=0, le=100)
    vignette: float = Field(0, ge=-100, le=100)
    colorMatch: ColorMatch | None = None
    skinMultiplier: int = Field(1, ge=1, le=3)
    strayHairRemoval: StrayHairSettings = StrayHairSettings()
    depthBokeh: DepthBokehSettings = DepthBokehSettings()
    # v0.5: spot healing keeps moles / beauty marks; colour mixer and grading wheels.
    protectBeautyMarks: bool = True
    # CIELAB a*/b* lock: retouched skin may not turn ashen or drift off its undertone.
    skinToneGuard: bool = True
    hsl: HslMixer = HslMixer()
    colorGrading: ColorGrading = ColorGrading()
    # v0.6 pro studio: glasses glare, teeth stains, clothing iron / recolour, look LUT.
    glareReduction: float = Field(0, ge=0, le=100)
    teethStainRemoval: float = Field(0, ge=0, le=100)
    clothing: ClothingSettings = ClothingSettings()
    lut: LookLut | None = None
    # v0.7 per-effect controls and master retouch strength (app/effects.py).
    effects: EffectControls = EffectControls()
    globalStrength: float = Field(100, ge=0, le=100)
    # v0.8 skin mask: face parsing + user brush strokes (app/masking.py).
    skinMask: SkinMaskSettings = SkinMaskSettings()
    localRetouch: LocalRetouch = LocalRetouch()
    # v0.13 Skin Perfection engine: tiled multi-scale detection, classification, targeted
    # correction and verified residual passes (app/engines/perfection.py).
    skinPerfection: SkinPerfectionSettings = SkinPerfectionSettings()
    skinFinish: SkinFinishSettings = SkinFinishSettings()


class StyleProfile(BaseModel):
    """Mirror of StyleProfile in packages/shared/src/style.ts."""

    stats: ColorStats
    meanL: float
    contrast: float
    warmth: float
    tintAxis: float
    chroma: float
    sharpness: float
    skinLuma: float | None
    skinTexture: float | None


class StyleSyncTarget(BaseModel):
    name: str
    profile: StyleProfile
    adjustments: dict


class StyleSyncResponse(BaseModel):
    reference: StyleProfile
    targets: list[StyleSyncTarget]


class FaceRegions(BaseModel):
    faceOval: list[Point]
    leftEye: list[Point]
    rightEye: list[Point]
    leftUnderEye: list[Point]
    rightUnderEye: list[Point]
    innerMouth: list[Point]
    lips: list[Point]
    leftBrow: list[Point]
    rightBrow: list[Point]
    # Lower nose (nostrils and alae), protected from spot healing. Optional: older clients omit it.
    nose: list[Point] = []


class FaceAnalysis(BaseModel):
    box: tuple[float, float, float, float]
    landmarkCount: int
    regions: FaceRegions
    eyeOpenness: float


class CullFlags(BaseModel):
    blurry: bool
    eyesClosed: bool
    noFace: bool


class ImageAnalysis(BaseModel):
    width: int
    height: int
    faces: list[FaceAnalysis]
    sharpness: float
    colorStats: ColorStats
    cull: CullFlags


class OutputSpec(BaseModel):
    format: Literal["jpeg", "png", "webp"] = "jpeg"
    quality: int = Field(92, ge=1, le=100)
    maxLongEdge: int | None = Field(None, ge=64, le=20000)
    prefix: str = ""


class RenderObjectRequest(BaseModel):
    """Used by services/worker: read the original from object storage, write the export back."""

    sourceKey: str
    outputKey: str
    adjustments: Adjustments
    output: OutputSpec = OutputSpec()


class RenderObjectResponse(BaseModel):
    outputKey: str
    bytes: int
    ms: float
    cull: CullFlags


class BatchSyncSpec(BaseModel):
    """How a batch carries the reference look to each photo (app/batch_engine.py).

    copy     the same adjustments on every photo.
    smart    the reference's edits plus per-photo exposure / white-balance compensation
             against the reference's original (referenceStats).
    palette  the reference's retouch plus a colour and tone transfer to the graded reference
             (referenceLookStats).
    A look LUT, when set in the adjustments, applies in every mode.
    """

    mode: Literal["copy", "smart", "palette"] = "smart"
    referenceStats: ColorStats | None = None
    referenceLookStats: ColorStats | None = None
    amount: float = Field(100, ge=0, le=100)


class BatchJobSettings(BaseModel):
    adjustments: Adjustments = Adjustments()
    sync: BatchSyncSpec = BatchSyncSpec()
    output: OutputSpec = OutputSpec()


class BatchJobRequest(BaseModel):
    """POST /v1/batch/jobs (desktop / local engine only)."""

    name: str = Field("Batch", max_length=120)
    sources: list[str] = Field(..., min_length=1, max_length=64)
    recursive: bool = True
    outputDir: str = Field(..., min_length=1)
    settings: BatchJobSettings = BatchJobSettings()
    profile: Literal["quiet", "balanced", "performance"] | None = None
