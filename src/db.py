"""Stage 2: dual SQLite / Postgres backend.

Selection rule
--------------
If env var DATABASE_URL is set (and non-empty), use Postgres.
Otherwise fall back to local SQLite at config.DB_PATH.

Public API
----------
    using_postgres() -> bool
    get_engine()                  - SQLAlchemy engine (use with pandas)
    get_connection()              - context manager, raw DBAPI connection
    placeholder()                 - "?" for SQLite, "%s" for Postgres
    init_schema(conn)             - create tables idempotently
    bulk_insert_ignore_conflicts(conn, table, columns, rows, conflict_col)
                                    backend-aware bulk insert
"""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

from dotenv import load_dotenv

from src.config import DB_PATH

# Load .env from the project root once at import time. python-dotenv is a
# no-op if .env doesn't exist, which is what we want for production / CI.
load_dotenv()

log = logging.getLogger("db")


# Backend selection
def _raw_database_url() -> str | None:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if url and url.strip():
        return url.strip()
    return None


def using_postgres() -> bool:
    return _raw_database_url() is not None


def get_db_url() -> str:
    """Return a SQLAlchemy-style URL for the active backend."""
    raw = _raw_database_url()
    if raw:
        # SQLAlchemy 2.x rejects the legacy `postgres://` scheme.
        if raw.startswith("postgres://"):
            raw = "postgresql://" + raw[len("postgres://"):]
        return raw
    return f"sqlite:///{DB_PATH}"


# Connections / engines
_engine = None  # cached SQLAlchemy engine


def get_engine():
    """Return a process-wide SQLAlchemy engine.

    Used by pandas (read_sql, to_sql) because pandas 2.x prefers a SQLAlchemy
    connectable over a raw DBAPI connection. The engine pools connections, so
    we only pay the handshake cost once.
    """
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine

        url = get_db_url()
        if url.startswith("sqlite"):
            # check_same_thread=False lets FastAPI's threadpool share the file.
            _engine = create_engine(
                url, connect_args={"check_same_thread": False}, future=True
            )
        else:
            # pool_pre_ping survives idle connection drops on PgBouncer.
            # Small pool because Supabase free-tier caps total client conns.
            _engine = create_engine(
                url, pool_pre_ping=True, pool_size=5, max_overflow=5, future=True
            )
        log.info("Initialized %s engine", "postgres" if using_postgres() else "sqlite")
    return _engine


@contextmanager
def get_connection() -> Iterator:
    """Yield a raw DBAPI connection. Caller is responsible for commit.

    Use this when you need driver-specific features (executemany, COPY,
    psycopg2.extras.execute_values) that SQLAlchemy abstracts away.
    """
    if using_postgres():
        import psycopg2

        # SQLAlchemy URLs and psycopg2 URLs are compatible at the libpq level.
        # We just strip any +driver suffix SQLAlchemy might have added.
        url = get_db_url()
        if "+" in url.split("://", 1)[0]:
            scheme, rest = url.split("://", 1)
            url = scheme.split("+")[0] + "://" + rest
        conn = psycopg2.connect(url)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        try:
            yield conn
        finally:
            conn.close()


def placeholder() -> str:
    """Parameter placeholder for the active backend."""
    return "%s" if using_postgres() else "?"


# Schema
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_transactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id    TEXT UNIQUE,
    user_id           INTEGER,
    amount            REAL,
    merchant_id       INTEGER,
    merchant_category TEXT,
    device            TEXT,
    country           TEXT,
    timestamp         REAL,
    is_fraud          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_raw_user      ON raw_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_raw_merchant  ON raw_transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_transactions(timestamp);

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


def init_schema(conn) -> None:
    """Create the three Argos tables if they don't already exist.

    On Postgres we assume the user ran `schema.sql` via the Supabase SQL
    editor. We still issue idempotent CREATE TABLE IF NOT EXISTS here so
    fresh dev environments don't have to remember the manual step.
    """
    if using_postgres():
        with conn.cursor() as cur:
            cur.execute(_POSTGRES_SCHEMA)
        conn.commit()
    else:
        conn.executescript(_SQLITE_SCHEMA)


_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_transactions (
    id                SERIAL PRIMARY KEY,
    transaction_id    TEXT UNIQUE,
    user_id           BIGINT,
    amount            DOUBLE PRECISION,
    merchant_id       BIGINT,
    merchant_category TEXT,
    device            TEXT,
    country           TEXT,
    timestamp         DOUBLE PRECISION,
    is_fraud          INTEGER,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_raw_user      ON raw_transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_raw_merchant  ON raw_transactions (merchant_id);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_transactions (timestamp);

CREATE TABLE IF NOT EXISTS user_features (
    user_id                  BIGINT PRIMARY KEY,
    total_transactions       INTEGER,
    total_spend              DOUBLE PRECISION,
    avg_transaction_amount   DOUBLE PRECISION,
    max_transaction_amount   DOUBLE PRECISION,
    transaction_velocity_24h INTEGER,
    unique_merchants         INTEGER,
    unique_countries         INTEGER,
    fraud_rate               DOUBLE PRECISION,
    last_transaction_ts      DOUBLE PRECISION,
    updated_at               DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS merchant_features (
    merchant_id                 BIGINT PRIMARY KEY,
    merchant_total_transactions INTEGER,
    merchant_avg_amount         DOUBLE PRECISION,
    merchant_fraud_rate         DOUBLE PRECISION,
    merchant_unique_users       INTEGER,
    updated_at                  DOUBLE PRECISION
);
"""


# Bulk insert helper
def bulk_insert_ignore_conflicts(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[tuple],
    conflict_col: str,
    page_size: int = 1000,
) -> int:
    """Idempotent bulk insert. Returns number of attempted rows.

    The "rowcount inserted" semantics differ between SQLite (returns the
    number of rows the executemany touched) and Postgres ON CONFLICT
    (returns 0 for ignored rows). To keep callers happy we just return the
    total number of input rows; "exactly how many were new" is rarely the
    metric you want.
    """
    cols_sql = ", ".join(columns)
    rows_list = list(rows)  # need len() and possibly multiple passes

    if using_postgres():
        # execute_values batches into a single INSERT VALUES (...), (...), ...
        # statement. Roughly 10-100x faster than executemany on Supabase's
        # pooled connection, where every round-trip costs network latency.
        from psycopg2.extras import execute_values

        sql = (
            f"INSERT INTO {table} ({cols_sql}) VALUES %s "
            f"ON CONFLICT ({conflict_col}) DO NOTHING"
        )
        with conn.cursor() as cur:
            execute_values(cur, sql, rows_list, page_size=page_size)
    else:
        placeholders = ", ".join(["?"] * len(columns))
        sql = (
            f"INSERT OR IGNORE INTO {table} ({cols_sql}) "
            f"VALUES ({placeholders})"
        )
        conn.executemany(sql, rows_list)

    conn.commit()
    return len(rows_list)


# Convenience
def describe() -> dict:
    """Useful for debugging which backend is active."""
    return {
        "backend": "postgres" if using_postgres() else "sqlite",
        "url": (
            "***REDACTED***"
            if using_postgres()
            else f"sqlite:///{DB_PATH}"
        ),
    }
