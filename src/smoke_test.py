"""Quick end-to-end check against a running serve.py.

Usage (in another terminal, with the API running):
    python -m src.smoke_test
    python -m src.smoke_test --requests 500 --host http://localhost:8000
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import sys
import time

import httpx

from src.config import DB_PATH


def sample_real_keys(n: int) -> list[tuple[int, int]]:
    """Pull (user_id, merchant_id) pairs that actually exist in the feature
    store so we exercise the warm path, not just cold-start."""
    conn = sqlite3.connect(DB_PATH)
    try:
        users = [r[0] for r in conn.execute(
            "SELECT user_id FROM user_features ORDER BY RANDOM() LIMIT ?", (n,)
        )]
        merchants = [r[0] for r in conn.execute(
            "SELECT merchant_id FROM merchant_features ORDER BY RANDOM() LIMIT ?", (n,)
        )]
    finally:
        conn.close()
    if not users or not merchants:
        return []
    return [(random.choice(users), random.choice(merchants)) for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.host, timeout=args.timeout) as client:
        try:
            r = client.get("/health")
            r.raise_for_status()
            print("health:", r.json())
        except Exception as e:
            print(f"health check failed: {e}", file=sys.stderr)
            return 1

        keys = sample_real_keys(args.requests) or [
            (random.randint(1, 10_000), random.randint(1, 5_000))
            for _ in range(args.requests)
        ]

        latencies = []
        scores = []
        labels = {"low": 0, "medium": 0, "high": 0}
        errors = 0

        t0 = time.perf_counter()
        for user_id, merchant_id in keys:
            payload = {
                "user_id": int(user_id),
                "merchant_id": int(merchant_id),
                "amount": round(random.uniform(5, 5000), 2),
            }
            t = time.perf_counter()
            try:
                r = client.post("/predict", json=payload)
                latencies.append((time.perf_counter() - t) * 1000)
                if r.status_code == 200:
                    body = r.json()
                    scores.append(body["fraud_score"])
                    labels[body["risk_label"]] += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
        elapsed = time.perf_counter() - t0

    if not latencies:
        print("no successful requests", file=sys.stderr)
        return 1

    latencies.sort()
    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(len(latencies) * p))]

    print()
    print(f"requests:    {len(keys)}  errors: {errors}  in {elapsed:.2f}s")
    print(f"throughput:  {len(keys) / elapsed:,.1f} req/s (single client)")
    print("latency_ms:")
    print(f"  mean       {statistics.mean(latencies):.2f}")
    print(f"  p50        {pct(0.50):.2f}")
    print(f"  p95        {pct(0.95):.2f}")
    print(f"  p99        {pct(0.99):.2f}")
    print("score distribution:")
    print(f"  mean       {statistics.mean(scores):.3f}")
    print(f"  min/max    {min(scores):.3f} / {max(scores):.3f}")
    print(f"  labels     {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
