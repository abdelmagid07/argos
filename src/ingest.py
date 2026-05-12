"""Stage 1: ingest raw transactions into SQLite.

If the IEEE-CIS CSV is present in data/, use it. Otherwise generate a
synthetic dataset so the rest of the pipeline can run end-to-end without
external downloads. Real IEEE-CIS data is preferred for meaningful AUC.

Usage:
    python -m src.ingest                  # auto: real if present, else synthetic
    python -m src.ingest --synthetic      # force synthetic
    python -m src.ingest --rows 50000     # limit rows (real or synthetic)
    python -m src.ingest --reset          # truncate tables first (use after
                                          # changing merchant_id mapping)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DB_PATH, IEEE_CIS_CSV


def _stable_hash(s: str, mod: int) -> int:
    """Deterministic hash that survives process restarts.

    Python's built-in hash() is salted per interpreter, so the same
    (ProductCD, addr1) pair would map to different merchant_ids across runs.
    md5 is fine here — we just need stability, not cryptographic strength.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % mod

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
    """Map IEEE-CIS columns onto our generic transaction schema.

    IEEE-CIS does not expose an explicit merchant identifier. We synthesize
    one from (ProductCD, addr1) — i.e. "(product category, billing region)" —
    which yields ~1000+ distinct buckets instead of the 5 values of ProductCD
    alone. addr1 has nulls in the dataset, which we bucket as 0.
    """
    log.info("Reading IEEE-CIS CSV from %s", IEEE_CIS_CSV)
    usecols = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
               "card1", "card4", "addr1", "DeviceType", "P_emaildomain", "isFraud"]
    df = pd.read_csv(IEEE_CIS_CSV, usecols=lambda c: c in usecols, nrows=rows)

    def col(name: str, default) -> pd.Series:
        """Tolerant column getter: some IEEE-CIS exports drop optional fields."""
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    product_cd = col("ProductCD", "unk").fillna("unk").astype(str)
    addr1 = col("addr1", 0).fillna(0).astype(int).astype(str)
    merchant_key = product_cd.str.cat(addr1, sep="|")

    now = time.time()
    transaction_dt = col("TransactionDT", 0).astype(float)
    out = pd.DataFrame({
        "transaction_id": "txn_" + df["TransactionID"].astype(str),
        # card1 is a card identifier; treat it as user_id proxy.
        "user_id": col("card1", 0).fillna(0).astype(int),
        "amount": df["TransactionAmt"].astype(float),
        # Hash (ProductCD, addr1) so the same pair always maps to the same
        # merchant_id across runs (Python's hash() is randomized).
        "merchant_id": merchant_key.map(lambda s: _stable_hash(s, 10_000)),
        # Keep the category readable for debugging; e.g. "W|264.0".
        "merchant_category": merchant_key,
        "device": col("DeviceType", "unknown").fillna("unknown").astype(str),
        "country": col("P_emaildomain", "unknown").fillna("unknown").astype(str),
        # IEEE-CIS TransactionDT is seconds-since-reference; offset to "recent"
        # so 24h velocity windowing has data on both sides of `now`.
        "timestamp": now - (transaction_dt.max() - transaction_dt),
        "is_fraud": col("isFraud", 0).fillna(0).astype(int),
    })

    log.info(
        "Loaded %d real transactions | %d unique users | %d unique merchants",
        len(out), out["user_id"].nunique(), out["merchant_id"].nunique(),
    )
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


def write_to_sqlite(df: pd.DataFrame, reset: bool = False) -> int:
    conn = _connect()
    try:
        cur = conn.cursor()
        if reset:
            # Clear raw_transactions plus any downstream features so a re-ingest
            # under a new merchant_id scheme doesn't leave stale rows behind.
            log.info("--reset: truncating raw_transactions + feature tables")
            cur.execute("DELETE FROM raw_transactions")
            for tbl in ("user_features", "merchant_features"):
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,),
                )
                if cur.fetchone():
                    cur.execute(f"DELETE FROM {tbl}")

        rows = df[[
            "transaction_id", "user_id", "amount", "merchant_id",
            "merchant_category", "device", "country", "timestamp", "is_fraud",
        ]].itertuples(index=False, name=None)
        # INSERT OR IGNORE keeps the script idempotent on transaction_id.
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
    parser.add_argument("--reset", action="store_true",
                        help="Truncate existing tables before insert. Use when "
                             "the merchant_id scheme or other schema-affecting "
                             "logic has changed.")
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
    inserted = write_to_sqlite(df, reset=args.reset)
    log.info("Inserted %d new rows (duplicates skipped).", inserted)


if __name__ == "__main__":
    main()
