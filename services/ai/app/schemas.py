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
