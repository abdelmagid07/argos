"""Push the offline DB's feature tables into Redis.

Usage:
    python -m src.sync_to_redis
    python -m src.sync_to_redis --ttl 7200      # custom TTL (default 1h)
    python -m src.sync_to_redis --no-meta       # skip writing argos:meta
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import redis
from sqlalchemy import text

from src import db
from src.redis_store import META_KEY, MERCHANT_KEY_FMT, USER_KEY_FMT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sync_to_redis")


DEFAULT_TTL = 3600
PIPELINE_BATCH = 5000


def _flush(pipe: redis.client.Pipeline) -> None:
    pipe.execute()


def _sync_table(
    conn,
    pipe: redis.client.Pipeline,
    r: redis.Redis,
    *,
    select_sql: str,
    id_col: str,
    key_fmt: str,
    ttl: int,
) -> int:
    """Pipeline every row in a table into Redis hashes with TTL.

    Pipelining batches network round-trips — without it, syncing 13k users
    would take 13k * RTT_ms. With batch=5000, it's roughly 3 round trips.
    """
    count = 0
    for row in conn.execute(text(select_sql)).mappings():
        d = dict(row)
        entity_id = int(d.pop(id_col))
        # Redis HSET refuses None values; stringify everything so we don't
        # have to special-case nulls coming out of LEFT JOINs.
        mapping = {k: ("" if v is None else str(v)) for k, v in d.items()}
        if not mapping:
            continue
        key = key_fmt.format(entity_id)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, ttl)
        count += 1
        if count % PIPELINE_BATCH == 0:
            _flush(pipe)
            pipe = r.pipeline()
            log.info("  ... %d rows pushed", count)
    _flush(pipe)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL,
                        help="Seconds before keys expire (default: 3600).")
    parser.add_argument("--no-meta", action="store_true",
                        help="Skip writing argos:meta hash.")
    args = parser.parse_args()

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise SystemExit(
            "REDIS_URL is not set in .env. Start Redis "
            "(`docker compose up -d redis`) and set REDIS_URL first."
        )

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    log.info("Connected to Redis at %s", redis_url)

    engine = db.get_engine()
    log.info("Reading features from %s backend", db.describe()["backend"])

    started = time.perf_counter()
    with engine.connect() as conn:
        log.info("Syncing user_features -> Redis...")
        user_count = _sync_table(
            conn, r.pipeline(), r,
            select_sql="SELECT * FROM user_features",
            id_col="user_id",
            key_fmt=USER_KEY_FMT,
            ttl=args.ttl,
        )
        log.info("Synced %d user feature rows", user_count)

        log.info("Syncing merchant_features -> Redis...")
        merchant_count = _sync_table(
            conn, r.pipeline(), r,
            select_sql="SELECT * FROM merchant_features",
            id_col="merchant_id",
            key_fmt=MERCHANT_KEY_FMT,
            ttl=args.ttl,
        )
        log.info("Synced %d merchant feature rows", merchant_count)

    elapsed = time.perf_counter() - started

    if not args.no_meta:
        # /health reads this; cheaper than SCAN-counting keys on every call.
        r.hset(META_KEY, mapping={
            "user_count": str(user_count),
            "merchant_count": str(merchant_count),
            "synced_at": f"{time.time():.3f}",
            "ttl_seconds": str(args.ttl),
            "source_backend": db.describe()["backend"],
        })

    log.info(
        "Sync complete: %d users, %d merchants in %.2fs (TTL=%ds)",
        user_count, merchant_count, elapsed, args.ttl,
    )


if __name__ == "__main__":
    main()
