# EmotionSense — Text Emotion Classification API

A machine learning web application that predicts the emotion expressed in a piece of text (sadness, joy, love, anger, fear, or surprise). The project compares three architectures — LSTM, GRU, and a fine-tuned DistilBERT — and serves the best-performing model through a **FastAPI** backend with a simple HTML form for interactive predictions, containerized with **Docker**.

## Features

- REST API endpoint (`/predict`) that returns the predicted emotion and per-class probabilities for a piece of text
- Simple web UI (`/`) for submitting text and viewing the prediction
- Input validation via Pydantic
- Fine-tuned `distilbert-base-uncased` model, selected as the best of three architectures compared (LSTM, GRU, DistilBERT) during training
- Dockerized for easy deployment

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Frontend | Jinja2 templates (HTML form) |
| ML Model | PyTorch, Hugging Face Transformers (DistilBERT) |
| Data | pandas, NumPy, scikit-learn |
| Server | Uvicorn |
| Containerization | Docker |

## Dataset

Model trained on [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) from Hugging Face — 16,000 training / 2,000 test English sentences labeled with one of 6 emotions.

## Project Structure

```
EmotionSense-Space/
├── app/
│   ├── main.py                          # FastAPI app, /predict endpoint
│   ├── schemas.py                       # Pydantic input schema & validation
│   └── templates/
│       └── index.html                   # Frontend prediction form
├── Emotions.ipynb                       # Model training / comparison notebook
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model Comparison

Three architectures were trained and evaluated on the same split, with class-weighted loss to handle the dataset's class imbalance:

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| LSTM | 87.75% | 0.839 | 0.880 |
| GRU | 84.65% | 0.818 | 0.850 |
| **DistilBERT** | **92.20%** | **0.887** | **0.924** |

DistilBERT was selected as the deployed model. See `Emotions.ipynb` for the full training, comparison, and error analysis.

## API Reference

> Confirm these against your actual `main.py`/`schemas.py` — filled in here with the standard shape for this kind of endpoint.

### `GET /api`
Health check.

**Response**
```json
{ "message": "EmotionSense API is running" }
```

### `POST /predict`
Returns the predicted emotion for a piece of text.

**Request body**

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | Yes | The sentence to classify |

**Example request**
```json
{
   "text": "I am very happy today"
}
```

**Example response**
```json
{
  "emotion": "joy",
  "confidence": 0.95
}
```

## Getting Started

### Prerequisites
- Python 3.11+
- pip
- (Optional) Docker

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/EmotionSense.git
cd EmotionSense

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The app will be available at:
- Web UI: `http://localhost:8000/`
- API docs (Swagger): `http://localhost:8000/docs`
- API health check: `http://localhost:8000/api`

### Run with Docker

```bash
# Build the image
docker build -t raj2903/emotionsense:latest .

# Run the container
docker run -d -p 8000:8000 --name emotionsense-app raj2903/emotionsense:latest
```

Then visit `http://localhost:8000/`.

### Pull from Docker Hub

```bash
docker pull raj2903/emotionsense:latest
docker run -d -p 8000:8000 raj2903/emotionsense:latest
```

## Model

The deployed model is a fine-tuned `distilbert-base-uncased` (Hugging Face Transformers), selected after comparing it against an LSTM and a GRU trained from scratch on the same data. Training used class-weighted loss to handle the dataset's imbalance, early stopping with best-checkpoint restoration, and packed sequences for the recurrent models. See `Emotions.ipynb` for the full training and evaluation workflow, including error analysis.

## License

This project is available for educational and portfolio purposes. Add a license of your choice (e.g., MIT) if you intend to distribute it.

## Author

**Raj** — [GitHub](https://github.com/Raj-Raaz) · [LinkedIn](https://www.linkedin.com/in/iitmraj)
