from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str
    confidence: float


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(PredictionItem):
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    task: str
    model_source: str
    top_prediction: Optional[PredictionItem] = None
    predictions: List[PredictionItem] = Field(default_factory=list)
    detections: List[DetectionItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    task: Optional[str] = None
    model_source: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_source: str
    resolved_model_file: str
    task: str
    top_k: int
