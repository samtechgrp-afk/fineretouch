"""Pydantic mirror of packages/shared/src/index.ts. Keep the two in sync."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]


class ColorStats(BaseModel):
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class ColorMatch(BaseModel):
    reference: ColorStats
    amount: float = Field(100, ge=0, le=100)


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
