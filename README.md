# Project: Sentiment Analysis API (Industry-Ready)

## Overview
A FastAPI-based sentiment analysis API using a fine-tuned DistilBERT model. Includes model versioning, batch prediction, metrics visualization, Docker deployment, CI/CD, and tests.

---

## Features
- **Model Training & Saving:** Fine-tune and save models/tokenizers to `models/v1`.
- **Prediction API:** Single/batch prediction, probability output, error handling, logging.
- **Metrics Visualization:** Training loss/accuracy plots at `/metrics` endpoint.
- **Dockerized:** Production-ready Dockerfile.
- **CI/CD:** Automated testing, linting, and Docker build with GitHub Actions.
- **Unit Tests:** Located in `tests/`.
- **Model Versioning:** Store models in `models/v1`, `models/v2`, etc.

---

## File Structure
- `app.py` - FastAPI app, endpoints for prediction and metrics.
- `model_utils.py` - Model/tokenizer loading, prediction logic, logging.
- `train_model.py` - Training script, saves model/tokenizer, plots metrics.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - For containerization.
- `.dockerignore` - Ignore files for Docker build context.
- `.github/workflows/ci.yml` - CI/CD pipeline.
- `tests/` - Unit tests.
- `models/` - Saved models/tokenizers (versioned).
- `results/` - Training metrics plots.

---

## Usage

### 1. Train Model
```bash
python train_model.py
```

### 2. Run API (Locally)
```bash
uvicorn app:app --reload
```

### 3. Run with Docker
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### 4. API Endpoints
- `POST /predict` - Single text prediction
- `POST /predict_batch` - Batch prediction
- `GET /metrics` - Training metrics plot

### 5. Run Tests
```bash
pip install pytest
pytest
```

---

## CI/CD
- On every push/PR: lint, test, and build Docker image via GitHub Actions.

---

## Model Versioning
- Save new models to `models/v2`, update `MODEL_DIR` env var if needed.

---

## Notes
- For production, consider using a process manager (e.g., Gunicorn) and HTTPS.
- Extend with authentication, monitoring, and more advanced logging as needed.
