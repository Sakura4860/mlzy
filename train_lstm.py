"""
Train and evaluate LSTM model on the prepared dataset.
Uses the scaled train/test splits exported by main.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.config import METRICS_DIR, MODELS_DIR, MODEL_PARAMS, RESULTS_DIR, VALIDATION_SIZE
from src.models.lstm_model import LSTMModel


def load_feature_list() -> Optional[List[str]]:
    """Load lite feature list if it exists; otherwise return None."""
    feature_file = RESULTS_DIR / "models" / "lite_features.txt"
    if feature_file.exists():
        lines = feature_file.read_text().splitlines()
        features = [line.strip() for line in lines if line.strip()]
        return features if features else None
    return None


def load_data(features: Optional[List[str]] = None):
    """Load scaled train/test splits and optionally select subset features."""
    X_train = pd.read_csv(METRICS_DIR / "X_train_full.csv")
    X_test = pd.read_csv(METRICS_DIR / "X_test_full.csv")
    y_train = pd.read_csv(METRICS_DIR / "y_train.csv").squeeze().values
    y_test = pd.read_csv(METRICS_DIR / "y_test.csv").squeeze().values

    if features is not None:
        missing = [f for f in features if f not in X_train.columns]
        if missing:
            raise ValueError(f"Missing features in data: {missing}")
        X_train = X_train[features]
        X_test = X_test[features]

    return (
        X_train.values.astype(np.float32),
        X_test.values.astype(np.float32),
        y_train.astype(np.float32),
        y_test.astype(np.float32),
        list(X_train.columns),
    )


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float) -> tuple:
    """Split chronologically into train/val using the tail as validation."""
    val_size = max(1, int(len(X) * val_ratio))
    if val_size >= len(X):
        raise ValueError("Validation size too large for available data")
    return (
        X[:-val_size],
        X[-val_size:],
        y[:-val_size],
        y[-val_size:],
    )


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    features = load_feature_list()
    X_train, X_test, y_train, y_test, columns = load_data(features)

    lstm_cfg = MODEL_PARAMS["lstm"]
    seq_len = lstm_cfg["sequence_length"]

    if len(X_train) <= seq_len:
        raise ValueError(f"Not enough samples for sequence_length={seq_len}")

    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train, VALIDATION_SIZE)

    model = LSTMModel(
        input_size=X_train.shape[1],
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        batch_size=lstm_cfg["batch_size"],
        epochs=lstm_cfg["epochs"],
        learning_rate=lstm_cfg["learning_rate"],
    )

    model.train(X_tr, y_tr, X_val, y_val, sequence_length=seq_len)
    metrics = model.evaluate(X_test, y_test, sequence_length=seq_len)

    results = {
        "model": "LSTM",
        "feature_set": "lite" if features else "full",
        "num_features": len(columns),
        "sequence_length": seq_len,
        **metrics,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([results]).to_csv(METRICS_DIR / "lstm_metrics.csv", index=False)
    (METRICS_DIR / "lstm_metrics.json").write_text(json.dumps(results, indent=2))

    model_path = MODELS_DIR / "lstm.pth"
    model.save_model(model_path)

    print("LSTM evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Feature set: {'lite' if features else 'full'} ({len(columns)} features)")


if __name__ == "__main__":
    main()
