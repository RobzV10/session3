from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
CSV_PATH = DATA_DIR / "imdb_balanced_10k.csv"

MAX_FEATURES = 3000
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

LABELS = {0: "negative", 1: "positive"}


class TextDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


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


def ensure_dataset_exists() -> None:
    if CSV_PATH.exists():
        print(f"Dataset file already exists: {CSV_PATH}")
        return

    print("Downloading IMDb dataset and generating balanced CSV file...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    combined = pd.concat([train_df, test_df], ignore_index=True)
    balanced = combined.groupby("label", group_keys=False).apply(
        lambda group: group.sample(n=5000, random_state=RANDOM_SEED)
    )
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    balanced.to_csv(CSV_PATH, index=False)
    print(f"Saved balanced dataset to {CSV_PATH}")


def load_data() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(CSV_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV dataset must contain 'text' and 'label' columns.")
    return df["text"], df["label"]


def build_vectorizer(texts: pd.Series) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    ).fit(texts)


def build_data_loaders(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series,
    X_val: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[DataLoader, DataLoader]:
    X_train_transformed = vectorizer.transform(X_train)
    X_val_transformed = vectorizer.transform(X_val)
    train_dataset = TextDataset(X_train_transformed, y_train.to_numpy())
    val_dataset = TextDataset(X_val_transformed, y_val.to_numpy())
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
    )


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> None:
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            targets.extend(y_batch.cpu().numpy())
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="weighted")
    return {"accuracy": float(accuracy), "f1_score": float(f1)}


def save_artifacts(
    model: nn.Module,
    vectorizer: TfidfVectorizer,
    config: dict,
    metrics: dict,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"Saved model artifacts in {MODEL_DIR}")


def run_training(args: argparse.Namespace) -> None:
    ensure_dataset_exists()
    texts, labels = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    vectorizer = build_vectorizer(X_train)
    train_loader, val_loader = build_data_loaders(vectorizer, X_train, X_val, y_train, y_val)
    input_dim = len(vectorizer.get_feature_names_out())
    model = SentimentNet(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, loss_fn)
        metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} - accuracy={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f}")

    final_metrics = evaluate(model, val_loader)
    config = {
        "dataset_csv": str(CSV_PATH),
        "model_type": "feedforward_tfidf",
        "vectorizer": {
            "max_features": MAX_FEATURES,
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
        },
        "labels": LABELS,
    }
    save_artifacts(model, vectorizer, config, final_metrics)
    print("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IMDb sentiment classifier.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_training(args)
