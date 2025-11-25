from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
vectorizer = bundle["vectorizer"]

NEGATIVE_KEYWORDS = [
    "battery", "drain", "drains", "heating", "overheating",
    "slow", "lag", "hang", "stuck",
    "refund", "fake", "scam", "cheat",
    "late delivery", "wrong item", "not working",
    "worst", "useless", "poor"
]

app = FastAPI(
    title="Sentiment Prediction API",
    description="Hybrid ML + Rule-based sentiment classifier for e-commerce and device reviews",
    version="1.0.0"
)

class ReviewInput(BaseModel):
    review_text: str

@app.get("/")
def root():
    return {"message": "API Running", "model_loaded_from": str(MODEL_PATH)}

@app.post("/predict")
def predict(payload: ReviewInput):
    text = payload.review_text.lower()

    if any(k in text for k in NEGATIVE_KEYWORDS):
        return {
            "input_text": payload.review_text,
            "predicted_label": "Negative",
            "confidence": 0.99,
            "note": "Rule-boosted (keyword hit)"
        }

    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    proba = model.predict_proba(X_vec)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    return {
        "input_text": payload.review_text,
        "predicted_label": sentiment,
        "probabilities": {
            "negative": float(proba[0]),
            "positive": float(proba[1])
        }
    }
