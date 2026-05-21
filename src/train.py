"""Train the fraud classifier on joined ``raw_transactions`` + feature tables.

Usage:
    python -m src.train                # default 15 epochs
    python -m src.train --epochs 30    # longer run
    python -m src.train --device cuda  # GPU (if available)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src import db
from src.config import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from src.model import FraudDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


JOIN_QUERY = """
SELECT
    t.amount,
    t.is_fraud,
    t.merchant_id,
    COALESCE(u.avg_transaction_amount, 0)   AS user_avg_amount,
    COALESCE(u.total_transactions, 0)       AS user_total_txns,
    COALESCE(u.fraud_rate, 0)               AS user_fraud_rate,
    COALESCE(u.transaction_velocity_24h, 0) AS velocity_24h,
    COALESCE(u.unique_merchants, 0)         AS unique_merchants,
    COALESCE(u.unique_countries, 0)         AS unique_countries,
    COALESCE(m.merchant_fraud_rate, 0)      AS merchant_fraud_rate,
    COALESCE(m.merchant_avg_amount, 0)      AS merchant_avg_amount,
    COALESCE(m.merchant_total_transactions, 0) AS merchant_total_txns
FROM raw_transactions t
LEFT JOIN user_features u     ON t.user_id = u.user_id
LEFT JOIN merchant_features m ON t.merchant_id = m.merchant_id
"""


def load_dataset() -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    # SQLAlchemy engine handles both SQLite and Postgres transparently.
    engine = db.get_engine()
    log.info("Loading joined dataset from %s...", db.describe()["backend"])
    df = pd.read_sql(JOIN_QUERY, engine)

    if df.empty:
        raise RuntimeError(
            "No training data. Run `python -m src.ingest` and "
            "`python -m src.features` first."
        )

    # Order columns the same way the API will assemble them at inference.
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.float32)
    log.info("Dataset: %d rows, %d features, fraud rate %.3f%%",
             len(df), X.shape[1], 100 * y.mean())

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y, scaler


def train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[FraudDetector, dict]:
    if y.sum() == 0 or y.sum() == len(y):
        raise RuntimeError("Need both fraud and non-fraud examples to train.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FraudDetector(input_dim=X.shape[1]).to(device)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max(1, (y_train == 1).sum())], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = float("inf")
    best_state = None
    metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        probs = 1 / (1 + np.exp(-np.concatenate(all_logits)))
        y_true = np.concatenate(all_y)
        try:
            auc = roc_auc_score(y_true, probs)
            ap = average_precision_score(y_true, probs)
        except ValueError:
            auc, ap = float("nan"), float("nan")

        log.info(
            "epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  AUC=%.4f  AP=%.4f",
            epoch, epochs, train_loss, val_loss, auc, ap,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            metrics = {"val_loss": val_loss, "auc": auc, "ap": ap, "epoch": epoch}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    started = time.time()
    X, y, scaler = load_dataset()
    model, metrics = train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1]}, MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    log.info("Saved model to %s", MODEL_PATH)
    log.info("Saved scaler to %s", SCALER_PATH)
    log.info("Best metrics: %s", metrics)
    log.info("Total time: %.1fs", time.time() - started)


if __name__ == "__main__":
    main()
