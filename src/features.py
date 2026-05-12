"""Stage 2: compute user + merchant features from raw_transactions.

The MVP uses pandas. The Spark equivalent in PROJECT.md does the same
aggregations, just distributed. Swap implementations later without changing
downstream consumers.

Usage:
    python -m src.features
"""
from __future__ import annotations

import logging
import sqlite3
import time

import pandas as pd

from src.config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("features")


SCHEMA = """
CREATE TABLE IF NOT EXISTS user_features (
    user_id                  INTEGER PRIMARY KEY,
    total_transactions       INTEGER,
    total_spend              REAL,
    avg_transaction_amount   REAL,
    max_transaction_amount   REAL,
    transaction_velocity_24h INTEGER,
    unique_merchants         INTEGER,
    unique_countries         INTEGER,
    fraud_rate               REAL,
    last_transaction_ts      REAL,
    updated_at               REAL
);

CREATE TABLE IF NOT EXISTS merchant_features (
    merchant_id                 INTEGER PRIMARY KEY,
    merchant_total_transactions INTEGER,
    merchant_avg_amount         REAL,
    merchant_fraud_rate         REAL,
    merchant_unique_users       INTEGER,
    updated_at                  REAL
);
"""


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        log.info("Loading raw_transactions from SQLite...")
        df = pd.read_sql("SELECT * FROM raw_transactions", conn)
    finally:
        conn.close()

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
    conn = sqlite3.connect(DB_PATH)
    try:
        # Replace the table contents wholesale; this matches the Spark
        # `mode="overwrite"` behavior in PROJECT.md and keeps things idempotent.
        conn.execute("DELETE FROM user_features")
        conn.execute("DELETE FROM merchant_features")
        user_features.to_sql("user_features", conn, if_exists="append", index=False)
        merchant_features.to_sql("merchant_features", conn, if_exists="append", index=False)
        conn.commit()
        log.info(
            "Wrote %d user_features and %d merchant_features.",
            len(user_features), len(merchant_features),
        )
    finally:
        conn.close()


def main() -> None:
    user_features, merchant_features = compute()
    write(user_features, merchant_features)


if __name__ == "__main__":
    main()
