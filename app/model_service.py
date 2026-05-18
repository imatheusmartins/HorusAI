from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from app.schemas import BoundingBox, DetectionItem, PredictionItem, PredictionResponse


class ModelService:
    def __init__(self) -> None:
        self.model_source = os.getenv("MODEL_PATH", "best.pt")
        self.top_k = int(os.getenv("TOP_K", "3"))
        self.resolved_model_file = self._resolve_model_file(self.model_source)
        self.model = YOLO(self.resolved_model_file)
        self.task = getattr(self.model, "task", "unknown")

    def _resolve_model_file(self, model_source: str) -> str:
        path = Path(model_source)

        if path.is_file() and path.suffix == ".pt":
            return str(path)

        raise FileNotFoundError(f"Modelo .pt nao encontrado: {model_source}")

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        results = self.model.predict(image, verbose=False)
        if not results:
            return PredictionResponse(task=self.task, model_source=self.model_source)

        result = results[0]
        task = getattr(result, "task", self.task)

        if task == "classify":
            return self._serialize_classification(result)

        if task == "detect":
            return self._serialize_detection(result)

        return PredictionResponse(task=task or "unknown", model_source=self.model_source)

    def _serialize_classification(self, result: Any) -> PredictionResponse:
        names = getattr(result, "names", {}) or {}
        probs = getattr(result, "probs", None)

        if probs is None:
            return PredictionResponse(task="classify", model_source=self.model_source)

        top_indices = probs.top5[: self.top_k]
        predictions = [
            PredictionItem(
                label=str(names.get(index, index)),
                confidence=round(float(probs.data[index]), 6),
            )
            for index in top_indices
        ]

        top_prediction = predictions[0] if predictions else None

        return PredictionResponse(
            task="classify",
            model_source=self.model_source,
            top_prediction=top_prediction,
            predictions=predictions,
        )

    def _serialize_detection(self, result: Any) -> PredictionResponse:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)

        detections: list[DetectionItem] = []
        if boxes is not None:
            xyxy_list = boxes.xyxy.tolist()
            conf_list = boxes.conf.tolist()
            cls_list = boxes.cls.tolist()

            for xyxy, confidence, class_id in zip(xyxy_list, conf_list, cls_list):
                detections.append(
                    DetectionItem(
                        label=str(names.get(int(class_id), int(class_id))),
                        confidence=round(float(confidence), 6),
                        bbox=BoundingBox(
                            x1=round(float(xyxy[0]), 2),
                            y1=round(float(xyxy[1]), 2),
                            x2=round(float(xyxy[2]), 2),
                            y2=round(float(xyxy[3]), 2),
                        ),
                    )
                )

        top_prediction = None
        predictions = []
        if detections:
            top = max(detections, key=lambda item: item.confidence)
            top_prediction = PredictionItem(label=top.label, confidence=top.confidence)
            predictions = [PredictionItem(label=item.label, confidence=item.confidence) for item in detections]

        return PredictionResponse(
            task="detect",
            model_source=self.model_source,
            top_prediction=top_prediction,
            predictions=predictions,
            detections=detections,
        )
