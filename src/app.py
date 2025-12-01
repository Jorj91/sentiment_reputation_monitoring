from fastapi import FastAPI
from pydantic import BaseModel
from .sentiment import predict_sentiment

app = FastAPI(title="Sentiment Monitoring API")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to MachineInnovators Sentiment API"}

@app.post("/predict")
def predict(input_data: TextInput):
    return predict_sentiment(input_data.text)