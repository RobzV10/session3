from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

LABELS = {0: "negative", 1: "positive"}


class SentimentNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_artifacts():
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
    with open(MODEL_DIR / "config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    input_dim = len(vectorizer.get_feature_names_out())
    model = SentimentNet(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
    model.eval()
    return model, vectorizer


def predict_text(model: nn.Module, vectorizer, text: str) -> dict[str, object]:
    features = vectorizer.transform([text]).toarray()
    logits = model(torch.tensor(features, dtype=torch.float32))
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    label_id = int(torch.argmax(logits, dim=1).item())
    return {
        "text": text,
        "label": LABELS[label_id],
        "score": float(probabilities[label_id]),
        "probabilities": {LABELS[i]: float(probabilities[i]) for i in range(len(probabilities))},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment using saved model artifacts.")
    parser.add_argument("--text", type=str, default=None, help="Sentence to classify")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.text is None:
        args.text = input("Enter a movie review sentence: ").strip()
    if not args.text:
        raise SystemExit("No text provided for prediction.")

    model, vectorizer = load_artifacts()
    result = predict_text(model, vectorizer, args.text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
