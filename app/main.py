from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from flask import Flask, render_template, request
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from app.schemas import EmotionRequest, EmotionResponse


MODEL_PATH = Path(__file__).resolve().parent.parent / "emotion_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

app = FastAPI(
    title="EmotionSense",
    description="Emotion Classification using DistilBERT",
    version="1.0.0"
)


@app.get("/api")
def api_home():
    return {"message": "EmotionSense API is running"}


@app.post("/predict", response_model=EmotionResponse)
def predict(data: EmotionRequest):
    inputs = tokenizer(
        data.text,
        padding="max_length",
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

    return EmotionResponse(
        emotion=label_names[prediction],
        confidence=float(probabilities[0][prediction])
    )


flask_app = Flask(__name__, template_folder="templates")


@flask_app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    emotion = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]

        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

        emotion = label_names[prediction]
        confidence = float(probabilities[0][prediction])

    return render_template(
        "index.html",
        text=text,
        emotion=emotion,
        confidence=confidence
    )


app.mount("/", WSGIMiddleware(flask_app))