"""Compute user + merchant features from ``raw_transactions``.

Usage:
    python -m src.features
"""
from __future__ import annotations

import logging
import time

import pandas as pd
from sqlalchemy import text

from src import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("features")


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    with db.get_connection() as conn:
        db.init_schema(conn)

    engine = db.get_engine()
    log.info("Loading raw_transactions from %s...", db.describe()["backend"])
    df = pd.read_sql("SELECT * FROM raw_transactions", engine)

    if df.empty:
        raise RuntimeError(
            "raw_transactions is empty. Run `python -m src.ingest` first."
        )

    log.info("Computing user features over %d rows...", len(df))
    now = time.time()
    last_24h = now - 86_400
    updated_at = now

    user_features = df.groupby("user_id").agg(
        total_transactions=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        avg_transaction_amount=("amount", "mean"),
        max_transaction_amount=("amount", "max"),
        unique_merchants=("merchant_id", "nunique"),
        unique_countries=("country", "nunique"),
        fraud_rate=("is_fraud", "mean"),
        last_transaction_ts=("timestamp", "max"),
    ).reset_index()

    velocity = (
        df[df["timestamp"] > last_24h]
        .groupby("user_id")
        .size()
        .rename("transaction_velocity_24h")
        .reset_index()
    )
    user_features = user_features.merge(velocity, on="user_id", how="left").fillna(
        {"transaction_velocity_24h": 0}
    )
    user_features["transaction_velocity_24h"] = user_features[
        "transaction_velocity_24h"
    ].astype(int)
    user_features["updated_at"] = updated_at

    log.info("Computing merchant features...")
    merchant_features = df.groupby("merchant_id").agg(
        merchant_total_transactions=("transaction_id", "count"),
        merchant_avg_amount=("amount", "mean"),
        merchant_fraud_rate=("is_fraud", "mean"),
        merchant_unique_users=("user_id", "nunique"),
    ).reset_index()
    merchant_features["updated_at"] = updated_at

    return user_features, merchant_features


def write(user_features: pd.DataFrame, merchant_features: pd.DataFrame) -> None:
    """Replace feature tables wholesale inside a single transaction.
    """
    engine = db.get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM user_features"))
        conn.execute(text("DELETE FROM merchant_features"))
        user_features.to_sql(
            "user_features", conn, if_exists="append",
            index=False, method="multi", chunksize=1000,
        )
        merchant_features.to_sql(
            "merchant_features", conn, if_exists="append",
            index=False, method="multi", chunksize=1000,
        )
    log.info(
        "Wrote %d user_features and %d merchant_features.",
        len(user_features), len(merchant_features),
    )


def main() -> None:
    user_features, merchant_features = compute()
    write(user_features, merchant_features)


if __name__ == "__main__":
    main()
