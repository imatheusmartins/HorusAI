from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model_service import ModelService
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse


app = FastAPI(title=os.getenv("API_TITLE", "Retinopathy Inference API"))
model_service = ModelService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.model is not None,
        task=model_service.task,
        model_source=model_service.model_source,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_source=model_service.model_source,
        resolved_model_file=model_service.resolved_model_file,
        task=model_service.task,
        top_k=model_service.top_k,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem valido.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        return model_service.predict(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na inferencia: {exc}") from exc
