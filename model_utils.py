from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer dan model sekali saja
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "Positive" if prediction == 1 else "Negative"
