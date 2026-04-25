# FastAPI MVP

## Endpoints

```
GET  /        - Status du service
POST /predict - Analyser une image

```

## Lancer

```bash
cd fastapi-mvp
pip install fastapi uvicorn
uvicorn src.main:app --reload
```

## Swagger

Accessible sur `http://localhost:8000/docs`
