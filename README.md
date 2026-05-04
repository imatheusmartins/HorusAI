# API de Inferencia para Retinopatia Diabetica

Este projeto expoe uma API HTTP para servir um modelo YOLO a partir da sua aplicacao Java com Spring Boot.

## Qual modelo faz mais sentido?

Pelos artefatos encontrados nesta pasta:

- `yolo11m-cls.pt` tem nomenclatura de classificacao (`cls`), o que combina com o caso de uso de classificar um exame de fundo de olho em categorias diagnosticas.
- `yolo26n.pt` parece um checkpoint mais completo de treino/inferencia para deteccao, com estrutura bem maior e tipica de modelos que retornam caixas/objetos.

Na configuracao atual deste projeto, o modelo principal carregado pela API e `yolo26n.pt`.

Observacao:

- Agora o projeto esta configurado para carregar diretamente um arquivo `.pt` real.
- O modelo principal esperado e `yolo26n.pt`.
- Como esse checkpoint aparenta ser de deteccao, a resposta principal da API passa a ser orientada a objetos detectados.

## Arquitetura sugerida

- Spring Boot: interface web, autenticacao, fluxo SaaS, persistencia e orquestracao.
- FastAPI + Ultralytics: servico isolado de inferencia.
- Comunicacao: `multipart/form-data` enviando a imagem para `POST /predict`.

## Estrutura

- `app/main.py`: endpoints HTTP
- `app/model_service.py`: carga do modelo e inferencia
- `app/schemas.py`: contratos de resposta
- `requirements.txt`: dependencias Python
- `Dockerfile`: containerizacao

## Endpoints

- `GET /health`
- `GET /model/info`
- `POST /predict`

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Variaveis de ambiente

```bash
MODEL_PATH=yolo26n.pt
TOP_K=3
API_TITLE=Retinopathy Inference API
```

## Exemplo com curl

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\caminho\imagem_fundo_olho.jpg"
```

## Exemplo de consumo no Spring Boot

Com `WebClient`, o fluxo fica assim:

```java
MultipartBodyBuilder builder = new MultipartBodyBuilder();
builder.part("file", imageResource);

PredictionResponse response = webClient.post()
    .uri("http://localhost:8000/predict")
    .contentType(MediaType.MULTIPART_FORM_DATA)
    .body(BodyInserters.fromMultipartData(builder.build()))
    .retrieve()
    .bodyToMono(PredictionResponse.class)
    .block();
```

## Resposta esperada

Para deteccao, a API retorna algo neste formato:

```json
{
  "task": "detect",
  "model_source": "yolo26n.pt",
  "top_prediction": {
    "label": "microaneurysm",
    "confidence": 0.91
  },
  "predictions": [
    {
      "label": "microaneurysm",
      "confidence": 0.91
    },
    {
      "label": "hemorrhage",
      "confidence": 0.06
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

Se depois voce voltar para um modelo de classificacao, a API ja tem suporte para retornar `top_prediction` e `predictions` sem a lista de `detections`.

## Observacao academica

Para o TCC, esta separacao entre aplicacao web e microservico de inferencia e uma boa decisao arquitetural porque:

- desacopla o ciclo de vida do modelo da aplicacao Java;
- facilita troca de modelo sem reescrever a camada web;
- permite escalar inferencia separadamente;
- deixa claro o papel do servico de IA dentro da arquitetura do sistema.
