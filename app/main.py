from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.model_service import ModelService
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse


SEVERITY_LABELS = {
    "0": "Grau 0 — sem retinopatia diabética",
    "1": "Grau 1 — retinopatia leve",
    "2": "Grau 2 — retinopatia moderada",
    "3": "Grau 3 — retinopatia grave",
    "4": "Grau 4 — retinopatia diabética proliferativa",
}

BASE_CSS = """
:root {
  color-scheme: light dark;
  --bg: #08111f;
  --panel: rgba(15, 23, 42, 0.94);
  --panel-soft: rgba(30, 41, 59, 0.72);
  --text: #e5e7eb;
  --muted: #a8b3c7;
  --accent: #38bdf8;
  --accent-strong: #0ea5e9;
  --border: rgba(148, 163, 184, 0.22);
  --ok: #22c55e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 18% 0%, rgba(14, 165, 233, 0.28), transparent 32%),
    radial-gradient(circle at 88% 8%, rgba(99, 102, 241, 0.22), transparent 30%),
    var(--bg);
  color: var(--text);
}
main { width: min(1120px, 100%); margin: 0 auto; padding: 40px 18px; }
.hero {
  display: grid;
  gap: 22px;
  grid-template-columns: 1.15fr 0.85fr;
  align-items: stretch;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 28px;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
}
.eyebrow { color: var(--accent); font-size: 13px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
h1 { margin: 10px 0 12px; font-size: clamp(34px, 6vw, 58px); line-height: 1.02; letter-spacing: -0.04em; }
h2 { margin: 0 0 14px; font-size: 24px; }
p { color: var(--muted); line-height: 1.65; }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 20px; }
.info { background: var(--panel-soft); border: 1px solid var(--border); border-radius: 18px; padding: 16px; }
.info strong { display: block; margin-bottom: 6px; }
form { display: grid; gap: 14px; margin-top: 18px; }
input, button, .button {
  border-radius: 14px;
  border: 1px solid var(--border);
  padding: 14px 16px;
  font-size: 16px;
}
input { background: #020617; color: var(--text); width: 100%; }
button, .button {
  cursor: pointer;
  border: 0;
  background: linear-gradient(135deg, var(--accent), var(--accent-strong));
  color: #02111f;
  font-weight: 800;
  text-align: center;
  text-decoration: none;
}
.button.secondary { background: transparent; color: var(--accent); border: 1px solid var(--border); }
button:disabled { cursor: wait; opacity: .72; }
.links { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }
a { color: var(--accent); }
.preview img, .sample img { width: 100%; border-radius: 18px; border: 1px solid var(--border); margin-top: 14px; background: #020617; }
.result-card { margin-top: 22px; }
.result-summary { display: none; gap: 12px; align-items: center; background: rgba(56, 189, 248, 0.09); border: 1px solid rgba(56, 189, 248, 0.28); border-radius: 18px; padding: 16px; margin-bottom: 14px; }
.badge { display: inline-flex; align-items: center; justify-content: center; min-width: 44px; height: 44px; border-radius: 999px; background: var(--accent); color: #082f49; font-weight: 900; }
pre { overflow: auto; background: #020617; border: 1px solid var(--border); border-radius: 18px; padding: 16px; min-height: 120px; }
.scale { display: grid; gap: 10px; margin-top: 12px; }
.scale div { display: flex; gap: 10px; align-items: flex-start; color: var(--muted); }
.scale b { color: var(--text); min-width: 72px; }
.samples-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; margin-top: 22px; }
.sample { background: var(--panel); border: 1px solid var(--border); border-radius: 22px; padding: 16px; }
.sample h2 { font-size: 19px; margin-top: 14px; overflow-wrap: anywhere; }
.sample-meta { color: var(--muted); margin: 8px 0 16px; }
.sample-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.empty { background: var(--panel-soft); border: 1px dashed var(--border); border-radius: 20px; padding: 24px; margin-top: 22px; }
.notice { border-left: 4px solid var(--accent); padding-left: 14px; }
@media (max-width: 860px) { .hero, .grid { grid-template-columns: 1fr; } main { padding-top: 20px; } }
"""

app = FastAPI(title=os.getenv("API_TITLE", "Retinopathy Inference API"))
model_service = ModelService()

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


def discover_sample_images() -> list[dict[str, str]]:
    samples_dir = static_dir / "samples"
    if not samples_dir.exists():
        return []

    samples: list[dict[str, str]] = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for file_path in sorted(samples_dir.iterdir(), key=lambda path: path.name.lower()):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_extensions:
            continue

        match = re.search(r"_(?P<grade>[0-4])\.[^.]+$", file_path.name)
        if not match:
            continue

        grade = match.group("grade")
        severity = SEVERITY_LABELS[grade].split(" — ", 1)[1]

        samples.append(
            {
                "filename": file_path.name,
                "grade": grade,
                "severity": severity,
                "src": f"/static/samples/{file_path.name}",
                "title": file_path.stem.replace("_", " "),
            }
        )

    return sorted(samples, key=lambda item: (item["grade"], item["filename"].lower()))


def page_shell(title: str, body: str) -> str:
    return f"""
    <!doctype html>
    <html lang="pt-BR">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title}</title>
        <style>{BASE_CSS}</style>
      </head>
      <body>{body}</body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    severity_json = SEVERITY_LABELS
    return page_shell(
        "HorusAI — Classificação de Retinopatia Diabética",
        f"""
        <main>
          <section class="hero">
            <div class="card">
              <div class="eyebrow">Protótipo de inferência</div>
              <h1>Classificação de retinopatia diabética por imagem de fundo de olho</h1>
              <p>
                Envie uma retinografia colorida para estimar o grau de retinopatia diabética em uma escala de 0 a 4.
                O resultado é uma predição computacional para fins acadêmicos e não substitui avaliação médica.
              </p>
              <div class="links">
                <a class="button" href="#upload">Testar imagem</a>
                <a class="button secondary" href="/samples">Ver imagens de exemplo</a>
                <a class="button secondary" href="/docs">Swagger da API</a>
              </div>
            </div>

            <aside class="card">
              <h2>Imagem esperada</h2>
              <p class="notice">
                Use imagens de fundo de olho, preferencialmente retinografias coloridas com retina visível,
                boa iluminação, foco adequado e campo circular preservado.
              </p>
              <div class="grid">
                <div class="info"><strong>Indicado</strong>Retinografia colorida em JPG, JPEG ou PNG.</div>
                <div class="info"><strong>Evitar</strong>OCT, laudos em PDF, prints de tela, selfies ou imagens externas do olho.</div>
                <div class="info"><strong>Qualidade</strong>Evite imagens muito desfocadas, cortadas ou com reflexos fortes.</div>
                <div class="info"><strong>Escala</strong>O modelo retorna graus numéricos de 0 a 4.</div>
              </div>
            </aside>
          </section>

          <section id="upload" class="card" style="margin-top: 22px;">
            <h2>Executar inferência</h2>
            <form id="predict-form">
              <input id="file" name="file" type="file" accept="image/*" required />
              <button id="submit" type="submit">Analisar imagem</button>
            </form>
            <div id="preview" class="preview"></div>

            <div class="result-card">
              <div id="summary" class="result-summary">
                <span id="grade" class="badge"></span>
                <div>
                  <strong id="severity-title"></strong>
                  <p id="confidence" style="margin: 4px 0 0;"></p>
                </div>
              </div>
              <pre id="result">Aguardando envio de imagem...</pre>
            </div>
          </section>

          <section class="card" style="margin-top: 22px;">
            <h2>Escala de saída do modelo</h2>
            <div class="scale">
              <div><b>Grau 0</b><span>Sem retinopatia diabética.</span></div>
              <div><b>Grau 1</b><span>Retinopatia leve.</span></div>
              <div><b>Grau 2</b><span>Retinopatia moderada.</span></div>
              <div><b>Grau 3</b><span>Retinopatia grave.</span></div>
              <div><b>Grau 4</b><span>Retinopatia diabética proliferativa.</span></div>
            </div>
          </section>
        </main>

        <script>
          const severityLabels = {severity_json};
          const form = document.getElementById("predict-form");
          const fileInput = document.getElementById("file");
          const button = document.getElementById("submit");
          const result = document.getElementById("result");
          const preview = document.getElementById("preview");
          const summary = document.getElementById("summary");
          const grade = document.getElementById("grade");
          const severityTitle = document.getElementById("severity-title");
          const confidence = document.getElementById("confidence");

          fileInput.addEventListener("change", () => {{
            const file = fileInput.files[0];
            preview.innerHTML = "";
            summary.style.display = "none";

            if (file) {{
              const img = document.createElement("img");
              img.src = URL.createObjectURL(file);
              img.alt = "Imagem selecionada para inferência";
              preview.appendChild(img);
            }}
          }});

          form.addEventListener("submit", async (event) => {{
            event.preventDefault();

            const file = fileInput.files[0];
            if (!file) return;

            const data = new FormData();
            data.append("file", file);

            button.disabled = true;
            summary.style.display = "none";
            result.textContent = "Processando a imagem. A primeira inferência pode levar alguns segundos...";

            try {{
              const response = await fetch("/predict", {{ method: "POST", body: data }});
              const payload = await response.json();
              result.textContent = JSON.stringify(payload, null, 2);

              if (payload.top_prediction) {{
                const label = String(payload.top_prediction.label);
                const conf = Number(payload.top_prediction.confidence || 0);
                grade.textContent = label;
                severityTitle.textContent = severityLabels[label] || `Grau ${{label}}`;
                confidence.textContent = `Confiança: ${{(conf * 100).toFixed(2)}}%`;
                summary.style.display = "flex";
              }}
            }} catch (error) {{
              result.textContent = "Falha ao executar a inferência: " + error;
            }} finally {{
              button.disabled = false;
            }}
          }});
        </script>
        """,
    )


@app.get("/samples", response_class=HTMLResponse)
def samples() -> str:
    sample_images = discover_sample_images()

    if sample_images:
        cards = "".join(
            f"""
            <article class="sample">
              <img src="{escape(item['src'])}" alt="Imagem de fundo de olho {escape(item['filename'])}" />
              <h2>{escape(item['filename'])}</h2>
              <p class="sample-meta"><strong>Classificação esperada:</strong> Grau {item['grade']} — {escape(item['severity'])}</p>
              <div class="sample-actions">
                <a class="button" href="{escape(item['src'])}" download>Baixar imagem</a>
                <a class="button secondary" href="{escape(item['src'])}" target="_blank" rel="noreferrer">Abrir imagem</a>
              </div>
            </article>
            """
            for item in sample_images
        )
    else:
        cards = """
        <div class="empty">
          <h2>Imagens de exemplo ainda não cadastradas</h2>
          <p>
            Esta página está pronta para receber imagens de teste. Cada imagem pode ser apresentada com grau esperado,
            descrição e botão de download para facilitar a demonstração do protótipo.
          </p>
        </div>
        """

    return page_shell(
        "HorusAI — Imagens de exemplo",
        f"""
        <main>
          <section class="card">
            <div class="eyebrow">Base de demonstração</div>
            <h1>Imagens para teste do modelo</h1>
            <p>
              Use estas imagens para demonstrar a classificação do protótipo em diferentes graus de retinopatia diabética.
              O grau esperado é identificado no final do nome do arquivo, antes da extensão.
            </p>
            <div class="links">
              <a class="button" href="/">Voltar para inferência</a>
              <a class="button secondary" href="/docs">Swagger da API</a>
            </div>
          </section>
          <section class="samples-grid">{cards}</section>
        </main>
        """,
    )


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
