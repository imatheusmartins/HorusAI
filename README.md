---
title: HorusAI Model API
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# API de Inferência para Retinopatia Diabética

API HTTP desenvolvida com FastAPI para classificar imagens de fundo de olho usando um modelo YOLO treinado para retinopatia diabética.

O serviço recebe uma imagem por `multipart/form-data` e retorna o grau estimado da doença em uma escala de 0 a 4.

## Tecnologias

- Python
- FastAPI
- Ultralytics YOLO
- Pillow
- Uvicorn
- Docker

## Estrutura do projeto

- `app/main.py`: endpoints HTTP e interface web do protótipo
- `app/model_service.py`: carregamento do modelo e execução da inferência
- `app/schemas.py`: modelos de resposta da API
- `app/static/samples`: imagens de exemplo para demonstração
- `best.pt`: modelo utilizado pela aplicação
- `requirements.txt`: dependências do projeto
- `Dockerfile`: imagem para execução em container

## Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `GET` | `/` | Interface web para upload e teste do modelo |
| `GET` | `/samples` | Página com imagens de exemplo para teste |
| `GET` | `/health` | Verifica o status da API e do modelo carregado |
| `GET` | `/model/info` | Retorna informações do modelo em uso |
| `POST` | `/predict` | Recebe uma imagem e retorna a classificação |
| `GET` | `/docs` | Documentação interativa da API |

## Tipo de imagem esperado

A entrada esperada é uma imagem de fundo de olho, preferencialmente uma retinografia colorida em `JPG`, `JPEG` ou `PNG`.

Recomendações:

- retina visível e centralizada;
- boa iluminação e foco adequado;
- campo circular preservado;
- evitar imagens cortadas, muito escuras, desfocadas ou com reflexos fortes.

O modelo não foi projetado para receber OCT, laudos em PDF, prints de tela, selfies ou imagens externas do olho.

## Escala de classificação

O modelo retorna a classe como um grau numérico:

| Grau | Interpretação |
| --- | --- |
| `0` | Sem retinopatia diabética aparente |
| `1` | Retinopatia diabética leve |
| `2` | Retinopatia diabética moderada |
| `3` | Retinopatia diabética severa |
| `4` | Retinopatia diabética proliferativa |

O grau `4` representa o estágio mais grave na escala utilizada pelo protótipo.

## Execução local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Com a aplicação em execução:

- Interface web: `http://localhost:8000/`
- Imagens de exemplo: `http://localhost:8000/samples`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Variáveis de ambiente

| Variável | Valor padrão | Descrição |
| --- | --- | --- |
| `MODEL_PATH` | `best.pt` | Caminho do arquivo do modelo |
| `TOP_K` | `3` | Quantidade máxima de predições retornadas |
| `API_TITLE` | `Retinopathy Inference API` | Título exibido na documentação da API |

## Exemplo de requisição

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\caminho\imagem_fundo_olho.jpg"
```

## Exemplo de resposta

```json
{
  "task": "classify",
  "model_source": "best.pt",
  "top_prediction": {
    "label": "1",
    "confidence": 0.444444
  },
  "predictions": [
    {
      "label": "1",
      "confidence": 0.444444
    },
    {
      "label": "2",
      "confidence": 0.406893
    },
    {
      "label": "0",
      "confidence": 0.141952
    }
  ],
  "detections": []
}
```

Nesse exemplo, a principal predição do modelo é o grau `1`, correspondente a retinopatia diabética leve.

## Integração

A API pode ser consumida por outros serviços via HTTP usando `multipart/form-data` no endpoint `POST /predict`. No projeto principal, ela pode ser chamada pelo backend Spring Boot, mantendo a inferência isolada da aplicação web.

## Hugging Face Spaces

O projeto pode ser executado como Docker Space no Hugging Face. A rota raiz `/` disponibiliza uma interface web para demonstração do protótipo, enquanto `/docs` mantém a documentação interativa da API.
