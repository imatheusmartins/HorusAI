---
title: HorusAI Model API
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# API de Inferencia para Retinopatia Diabetica

API HTTP desenvolvida com FastAPI para executar inferencias com um modelo YOLO treinado para analise de imagens de fundo de olho.

## Tecnologias

- Python
- FastAPI
- Ultralytics YOLO
- Pillow
- Uvicorn

## Estrutura do projeto

- `app/main.py`: definicao dos endpoints HTTP
- `app/model_service.py`: carregamento do modelo e processamento das inferencias
- `app/schemas.py`: modelos de resposta da API
- `best.pt`: modelo utilizado pela aplicacao
- `requirements.txt`: dependencias do projeto
- `Dockerfile`: imagem para execucao em container

## Endpoints

| Metodo | Rota | Descricao |
| --- | --- | --- |
| `GET` | `/health` | Verifica o status da API e do modelo carregado |
| `GET` | `/model/info` | Retorna informacoes do modelo em uso |
| `POST` | `/predict` | Recebe uma imagem e retorna a inferencia |

## Execucao local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Com a aplicacao em execucao:

- Interface web: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Variaveis de ambiente

| Variavel | Valor padrao | Descricao |
| --- | --- | --- |
| `MODEL_PATH` | `best.pt` | Caminho do arquivo do modelo |
| `TOP_K` | `3` | Quantidade maxima de predicoes retornadas |
| `API_TITLE` | `Retinopathy Inference API` | Titulo exibido na documentacao da API |

## Exemplo de requisicao

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\caminho\imagem_fundo_olho.jpg"
```

## Exemplo de resposta

```json
{
  "task": "detect",
  "model_source": "best.pt",
  "top_prediction": {
    "label": "microaneurysm",
    "confidence": 0.91
  },
  "predictions": [
    {
      "label": "microaneurysm",
      "confidence": 0.91
    }
  ],
  "detections": [
    {
      "label": "microaneurysm",
      "confidence": 0.91,
      "bbox": {
        "x1": 125.4,
        "y1": 88.2,
        "x2": 164.7,
        "y2": 120.9
      }
    }
  ]
}
```

## Integracao

A API pode ser consumida por outros servicos via HTTP usando `multipart/form-data` no endpoint `POST /predict`. No projeto principal, ela pode ser chamada pelo backend Spring Boot, mantendo a inferencia isolada da aplicacao web.

## Hugging Face Spaces

O projeto tambem pode ser executado como Docker Space no Hugging Face. A rota raiz `/` disponibiliza uma interface simples para upload de imagem e teste do modelo. A documentacao interativa da API permanece disponivel em `/docs`.
