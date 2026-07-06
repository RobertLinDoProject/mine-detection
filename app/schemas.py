from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    device: str
    task: Literal["object_detection"]
    weights: str
    class_count: int
    class_map: dict[int, str]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    detection_id: int = Field(ge=1)
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: BoundingBox
    bbox_normalized: BoundingBox
    area_px: float = Field(ge=0.0)


class DetectResponse(BaseModel):
    model_name: str
    model_version: str
    image_filename: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    detections: list[Detection]
