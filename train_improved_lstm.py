"""
Train and evaluate Improved LSTM model with better architecture and training strategies.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.config import METRICS_DIR, MODELS_DIR, RESULTS_DIR, VALIDATION_SIZE
from src.models.improved_lstm_model import ImprovedLSTMModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    # 改进的LSTM配置
    improved_lstm_cfg = {
        'hidden_size': 128,        # 保持较大的隐藏层
        'num_layers': 3,           # 增加层数（2→3）
        'dropout': 0.3,            # 增加dropout（0.2→0.3）
        'batch_size': 64,          # 较小batch以更好学习（128→64）
        'epochs': 100,             # 增加最大轮数（20→100），但有早停
        'learning_rate': 0.001,    # 初始学习率
        'weight_decay': 1e-5,      # L2正则化
        'patience': 15,            # 早停耐心值
        'sequence_length': 48      # 更长序列（24→48）
    }

    logger.info("=" * 60)
    logger.info("Training Improved LSTM Model")
    logger.info("=" * 60)
    logger.info(f"Configuration: {json.dumps(improved_lstm_cfg, indent=2)}")

    features = load_feature_list()
    X_train, X_test, y_train, y_test, columns = load_data(features)

    seq_len = improved_lstm_cfg["sequence_length"]
    logger.info(f"Feature set: {'lite' if features else 'full'} ({len(columns)} features)")
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    if len(X_train) <= seq_len:
        raise ValueError(f"Not enough samples for sequence_length={seq_len}")

    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train, VALIDATION_SIZE)
    logger.info(f"After split: Train={len(X_tr)}, Val={len(X_val)}")

    model = ImprovedLSTMModel(
        input_size=X_train.shape[1],
        hidden_size=improved_lstm_cfg["hidden_size"],
        num_layers=improved_lstm_cfg["num_layers"],
        dropout=improved_lstm_cfg["dropout"],
        batch_size=improved_lstm_cfg["batch_size"],
        epochs=improved_lstm_cfg["epochs"],
        learning_rate=improved_lstm_cfg["learning_rate"],
        weight_decay=improved_lstm_cfg["weight_decay"],
        patience=improved_lstm_cfg["patience"],
    )

    logger.info("\n" + "=" * 60)
    logger.info("Starting Training...")
    logger.info("=" * 60)
    
    model.train(X_tr, y_tr, X_val, y_val, sequence_length=seq_len)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on Test Set...")
    logger.info("=" * 60)
    
    metrics = model.evaluate(X_test, y_test, sequence_length=seq_len)

    results = {
        "model": "ImprovedLSTM",
        "feature_set": "lite" if features else "full",
        "num_features": len(columns),
        "sequence_length": seq_len,
        "hidden_size": improved_lstm_cfg["hidden_size"],
        "num_layers": improved_lstm_cfg["num_layers"],
        "final_train_loss": model.train_losses[-1] if model.train_losses else None,
        "best_val_loss": model.best_val_loss if hasattr(model, 'best_val_loss') else None,
        "epochs_trained": len(model.train_losses),
        **metrics,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([results]).to_csv(METRICS_DIR / "improved_lstm_metrics.csv", index=False)
    (METRICS_DIR / "improved_lstm_metrics.json").write_text(json.dumps(results, indent=2))

    model_path = MODELS_DIR / "improved_lstm.pth"
    model.save_model(model_path)

    logger.info("\n" + "=" * 60)
    logger.info("Improved LSTM Evaluation Results")
    logger.info("=" * 60)
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:<20} {v:>15.4f}")
    print("-" * 40)
    print(f"\nModel saved to: {model_path}")
    print(f"Feature set: {'lite' if features else 'full'} ({len(columns)} features)")
    print(f"Sequence length: {seq_len}")
    print(f"Epochs trained: {len(model.train_losses)}")
    if hasattr(model, 'best_val_loss'):
        print(f"Best validation loss: {model.best_val_loss:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
