# backend-food-ia

API FastAPI del sistema de recomendación semántica de platos.

## Ejecución local

```bash
pip install -r requirements.txt
MODEL_BACKEND=fastembed uvicorn app.main:app --reload --port 8000
```

La API queda disponible en `http://localhost:8000/docs`.

## Pruebas

```bash
MODEL_BACKEND=tfidf pytest -q
```

El backend aplica filtros deterministas de seguridad y utiliza embeddings multilingües mediante FastEmbed/ONNX Runtime.
