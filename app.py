from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_utils import predict_sentiment
from fastapi.responses import FileResponse, JSONResponse
import logging
import json

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict(data: TextInput):
    try:
        result = predict_sentiment(data.text)
        return {"sentiment": result}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

@app.post("/predict_batch")
def predict_batch(data: BatchInput):
    try:
        results = predict_sentiment(data.texts)
        return {"sentiments": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")

@app.get("/metrics")
def get_metrics():
    try:
        return FileResponse("results/training_metrics.png")
    except Exception as e:
        logger.error(f"Metrics file error: {e}")
        raise HTTPException(status_code=404, detail="Metrics file not found.")

@app.get("/metrics/summary")
def get_metrics_summary():
    """
    Returns a simple JSON summary of the latest training metrics (accuracy, loss, etc).
    Expects a metrics file at results/metrics_summary.json (to be generated after training).
    """
    try:
        with open("results/metrics_summary.json", "r") as f:
            summary = json.load(f)
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Metrics summary error: {e}")
        raise HTTPException(status_code=404, detail="Metrics summary not found.")
