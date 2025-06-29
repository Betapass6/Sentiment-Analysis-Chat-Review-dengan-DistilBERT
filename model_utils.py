import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models/v1")

def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        logger.info(f"Loaded model and tokenizer from {MODEL_DIR}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

# Load once
try:
    tokenizer, model = load_model_and_tokenizer()
except Exception as e:
    tokenizer, model = None, None

def predict_sentiment(texts, return_proba=False):
    if tokenizer is None or model is None:
        raise RuntimeError("Model or tokenizer not loaded.")
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).tolist()
        sentiments = ["Negative" if p == 0 else "Positive" for p in preds]
        if return_proba:
            return list(zip(sentiments, probs.tolist()))
        return sentiments if len(sentiments) > 1 else sentiments[0]
