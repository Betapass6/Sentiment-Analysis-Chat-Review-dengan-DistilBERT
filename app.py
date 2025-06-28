from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_sentiment

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    result = predict_sentiment(data.text)
    return {"sentiment": result}
