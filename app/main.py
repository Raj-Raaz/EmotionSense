from fastapi import FastAPI
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch

from app.schemas import EmotionRequest, EmotionResponse

app = FastAPI(title="EmotionSense")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("emotion_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("emotion_model")

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

@app.get("/")
def home():
    return {"message": "EmotionSense API is running"}

@app.post("/predict", response_model=EmotionResponse)
def predict(request: EmotionRequest):
    inputs = tokenizer(request.text, padding="max_length", truncation=True, max_length=50, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

    return EmotionResponse(
        emotion=label_names[prediction],
        confidence=float(probabilities[0][prediction])
    )