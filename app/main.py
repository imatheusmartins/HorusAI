from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.model_service import ModelService
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse


app = FastAPI(title=os.getenv("API_TITLE", "Retinopathy Inference API"))
model_service = ModelService()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html lang="pt-BR">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>HorusAI Model API</title>
        <style>
          :root {
            color-scheme: light dark;
            --bg: #0f172a;
            --card: #111827;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #38bdf8;
            --border: #263244;
          }

          body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, Helvetica, sans-serif;
            background: radial-gradient(circle at top, #1e3a8a 0, var(--bg) 42%);
            color: var(--text);
            display: grid;
            place-items: center;
            padding: 32px 16px;
          }

          main {
            width: min(920px, 100%);
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 28px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
          }

          h1 {
            margin: 0 0 8px;
            font-size: clamp(28px, 4vw, 42px);
          }

          p {
            color: var(--muted);
            line-height: 1.6;
          }

          form {
            display: grid;
            gap: 16px;
            margin: 28px 0;
          }

          input, button {
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 14px;
            font-size: 16px;
          }

          input {
            background: #0b1220;
            color: var(--text);
          }

          button {
            cursor: pointer;
            border: 0;
            background: var(--accent);
            color: #082f49;
            font-weight: 700;
          }

          button:disabled {
            cursor: wait;
            opacity: 0.75;
          }

          .links {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 18px;
          }

          a {
            color: var(--accent);
            text-decoration: none;
          }

          img {
            max-width: 100%;
            border-radius: 16px;
            border: 1px solid var(--border);
            margin-top: 12px;
          }

          pre {
            overflow: auto;
            background: #020617;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
            min-height: 96px;
          }

          .result-title {
            margin-top: 24px;
            color: var(--text);
            font-weight: 700;
          }
        </style>
      </head>
      <body>
        <main>
          <h1>HorusAI Model API</h1>
          <p>
            Envie uma imagem de fundo de olho para executar a inferencia no modelo YOLO.
            A API tambem continua disponivel via Swagger e endpoint HTTP.
          </p>

          <form id="predict-form">
            <input id="file" name="file" type="file" accept="image/*" required />
            <button id="submit" type="submit">Executar inferencia</button>
          </form>

          <div id="preview"></div>

          <div class="result-title">Resultado</div>
          <pre id="result">Aguardando envio de imagem...</pre>

          <div class="links">
            <a href="/docs">Swagger UI</a>
            <a href="/health">Health check</a>
            <a href="/model/info">Informacoes do modelo</a>
          </div>
        </main>

        <script>
          const form = document.getElementById("predict-form");
          const fileInput = document.getElementById("file");
          const button = document.getElementById("submit");
          const result = document.getElementById("result");
          const preview = document.getElementById("preview");

          fileInput.addEventListener("change", () => {
            const file = fileInput.files[0];
            preview.innerHTML = "";

            if (file) {
              const img = document.createElement("img");
              img.src = URL.createObjectURL(file);
              img.alt = "Imagem selecionada";
              preview.appendChild(img);
            }
          });

          form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const file = fileInput.files[0];
            if (!file) return;

            const data = new FormData();
            data.append("file", file);

            button.disabled = true;
            result.textContent = "Processando...";

            try {
              const response = await fetch("/predict", {
                method: "POST",
                body: data
              });

              const payload = await response.json();
              result.textContent = JSON.stringify(payload, null, 2);
            } catch (error) {
              result.textContent = "Falha ao executar a inferencia: " + error;
            } finally {
              button.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """


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
