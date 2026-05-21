"""Ingest raw transactions into the active DB backend.

Backend is chosen by :mod:`src.db` (Postgres if ``DATABASE_URL`` is set, else
SQLite). If ``data/train_transaction.csv`` (Kaggle IEEE-CIS) is present we use
it; otherwise we synthesize a comparable dataset so the rest of the pipeline
can run end-to-end without external downloads.

Usage:
    python -m src.ingest                  # auto: real if present, else synthetic
    python -m src.ingest --synthetic      # force synthetic
    python -m src.ingest --rows 50000     # cap rows (real or synthetic)
    python -m src.ingest --reset          # truncate tables first
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from src import db
from src.config import IEEE_CIS_CSV


def _stable_hash(s: str, mod: int) -> int:
    """Deterministic hash that survives process restarts.

    Python's built-in hash() is salted per interpreter, so the same
    (ProductCD, addr1) pair would map to different merchant_ids across runs.
    md5 is fine here — we just need stability, not cryptographic strength.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


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


COLUMNS = [
    "transaction_id", "user_id", "amount", "merchant_id",
    "merchant_category", "device", "country", "timestamp", "is_fraud",
]


def write_to_db(df: pd.DataFrame, reset: bool = False) -> int:
    """Idempotent bulk insert into raw_transactions on the active backend.

    Dispatches to db.bulk_insert_ignore_conflicts which uses
    `INSERT OR IGNORE` (SQLite) or `INSERT ... ON CONFLICT DO NOTHING`
    (Postgres) so re-running ingest never produces duplicates.
    """
    with db.get_connection() as conn:
        db.init_schema(conn)
        if reset:
            # Clear raw_transactions plus any downstream features so a re-ingest
            # under a new merchant_id scheme doesn't leave stale rows behind.
            log.info("--reset: truncating raw_transactions + feature tables")
            cur = conn.cursor()
            for tbl in ("user_features", "merchant_features", "raw_transactions"):
                cur.execute(f"DELETE FROM {tbl}")
            conn.commit()

        # Convert numpy/pandas scalar types to native Python so psycopg2 and
        # sqlite3 don't choke on numpy.int64 etc.
        rows = [tuple(map(to_python_scalar, r)) for r in df[COLUMNS].itertuples(index=False, name=None)]
        return db.bulk_insert_ignore_conflicts(
            conn,
            table="raw_transactions",
            columns=COLUMNS,
            rows=rows,
            conflict_col="transaction_id",
        )


def to_python_scalar(v):
    """Convert numpy/pandas scalars to native Python for DB drivers."""
    if hasattr(v, "item"):
        return v.item()
    return v


def load_transactions_dataframe(
    *,
    synthetic: bool = False,
    rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the same transaction rows `main()` would ingest.

    Used by the Kafka producer so CSV → topic → consumer reproduces the same
    schema as `python -m src.ingest` without duplicating mapping logic.
    """
    use_real = IEEE_CIS_CSV.exists() and not synthetic
    if use_real:
        return _load_real(rows)
    if not synthetic and not IEEE_CIS_CSV.exists():
        log.warning(
            "IEEE-CIS CSV not found at %s — using synthetic data.", IEEE_CIS_CSV
        )
    return _load_synthetic(rows or 100_000)


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

    df = load_transactions_dataframe(synthetic=args.synthetic, rows=args.rows)

    backend = db.describe()
    log.info("Writing %d rows to %s backend", len(df), backend["backend"])
    inserted = write_to_db(df, reset=args.reset)
    log.info("Wrote %d rows (duplicates ignored).", inserted)


if __name__ == "__main__":
    main()
