"""
If the IEEE-CIS CSV is present in data/, use it. Otherwise generate a
synthetic dataset so the rest of the pipeline can run end-to-end without
external downloads. Real IEEE-CIS data is preferred for meaningful AUC.

Usage:
    python -m src.ingest                  # auto: real if present, else synthetic
    python -m src.ingest --synthetic      # force synthetic
    python -m src.ingest --rows 50000     # limit rows (real or synthetic)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH, IEEE_CIS_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT UNIQUE,
    user_id         INTEGER,
    amount          REAL,
    merchant_id     INTEGER,
    merchant_category TEXT,
    device          TEXT,
    country         TEXT,
    timestamp       REAL,
    is_fraud        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_raw_user ON raw_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_raw_merchant ON raw_transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_transactions(timestamp);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def _load_real(rows: Optional[int]) -> pd.DataFrame:
    log.info("Reading IEEE-CIS CSV from %s", IEEE_CIS_CSV)
    usecols = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
               "card1", "DeviceType", "P_emaildomain", "isFraud"]
    df = pd.read_csv(IEEE_CIS_CSV, usecols=lambda c: c in usecols, nrows=rows)

    now = time.time()
    out = pd.DataFrame({
        "transaction_id": "txn_" + df["TransactionID"].astype(str),
        # card1 is a card identifier; treat it as user_id proxy
        "user_id": df["card1"].fillna(0).astype(int),
        "amount": df["TransactionAmt"].astype(float),
        # Stable hash of ProductCD to integer merchant id (kept small for SQLite)
        "merchant_id": df["ProductCD"].fillna("unk").astype(str).map(
            lambda s: abs(hash(s)) % 5000
        ),
        "merchant_category": df["ProductCD"].fillna("unk").astype(str),
        "device": df["DeviceType"].fillna("unknown").astype(str),
        "country": df["P_emaildomain"].fillna("unknown").astype(str),
        # IEEE-CIS TransactionDT is seconds-since-reference; offset to "recent"
        # so 24h velocity windowing has data on both sides of `now`.
        "timestamp": now - (df["TransactionDT"].max() - df["TransactionDT"]).astype(float),
        "is_fraud": df["isFraud"].fillna(0).astype(int),
    })
    return out


def _load_synthetic(rows: int) -> pd.DataFrame:
    """Generate a fake dataset with realistic-ish structure and ~3.5% fraud rate."""
    log.info("Generating %d synthetic transactions", rows)
    rng = np.random.default_rng(42)

    n_users = max(1000, rows // 50)
    n_merchants = max(200, rows // 200)

    user_id = rng.integers(1, n_users + 1, size=rows)
    merchant_id = rng.integers(1, n_merchants + 1, size=rows)

    # Most amounts are small; a long right tail of larger ones.
    amount = np.round(rng.lognormal(mean=3.5, sigma=1.1, size=rows), 2)

    devices = rng.choice(["mobile", "desktop", "tablet", "unknown"],
                         size=rows, p=[0.55, 0.30, 0.10, 0.05])
    countries = rng.choice(["US", "GB", "CA", "DE", "FR", "BR", "IN", "unknown"],
                           size=rows, p=[0.45, 0.10, 0.08, 0.07, 0.06, 0.08, 0.10, 0.06])
    categories = rng.choice(["W", "C", "R", "H", "S"], size=rows)

    now = time.time()
    # Spread across the last 7 days so 24h velocity has signal
    timestamp = now - rng.uniform(0, 7 * 86400, size=rows)

    # Fraud signal: high amount + non-US + new device raises probability
    base = 0.015
    risk = (
        base
        + 0.06 * (amount > np.quantile(amount, 0.95))
        + 0.04 * (countries != "US")
        + 0.03 * (devices == "unknown")
    )
    is_fraud = (rng.random(rows) < risk).astype(int)

    df = pd.DataFrame({
        "transaction_id": [f"syn_{i}" for i in range(rows)],
        "user_id": user_id.astype(int),
        "amount": amount,
        "merchant_id": merchant_id.astype(int),
        "merchant_category": categories,
        "device": devices,
        "country": countries,
        "timestamp": timestamp,
        "is_fraud": is_fraud,
    })
    log.info("Synthetic fraud rate: %.2f%%", 100 * df["is_fraud"].mean())
    return df


def write_to_sqlite(df: pd.DataFrame) -> int:
    conn = _connect()
    try:
        # Use INSERT OR IGNORE for idempotency on transaction_id.
        cur = conn.cursor()
        rows = df[[
            "transaction_id", "user_id", "amount", "merchant_id",
            "merchant_category", "device", "country", "timestamp", "is_fraud",
        ]].itertuples(index=False, name=None)
        cur.executemany(
            """
            INSERT OR IGNORE INTO raw_transactions
              (transaction_id, user_id, amount, merchant_id,
               merchant_category, device, country, timestamp, is_fraud)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic dataset even if IEEE-CIS CSV exists.")
    parser.add_argument("--rows", type=int, default=None,
                        help="Cap on rows ingested (real or synthetic).")
    args = parser.parse_args()

    use_real = IEEE_CIS_CSV.exists() and not args.synthetic
    if use_real:
        df = _load_real(args.rows)
    else:
        if not args.synthetic and not IEEE_CIS_CSV.exists():
            log.warning("IEEE-CIS CSV not found at %s — using synthetic data.",
                        IEEE_CIS_CSV)
        df = _load_synthetic(args.rows or 100_000)

    log.info("Writing %d rows to %s", len(df), DB_PATH)
    inserted = write_to_sqlite(df)
    log.info("Inserted %d new rows (duplicates skipped).", inserted)


if __name__ == "__main__":
    main()
