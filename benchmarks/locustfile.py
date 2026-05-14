"""Locust load test for POST /predict.

Run from the **repository root** so `src` imports resolve.

Prerequisites: trained model, feature store populated (Redis recommended), API
running (e.g. `uvicorn src.serve:app --port 8000`).

Easiest stats: add **`--html benchmarks/results/locust_report.html`** and open
that file in a browser (response-time **percentiles**, charts, RPS). Use
**`--only-summary`** for a short terminal; the HTML still writes at exit.

Example (~200 RPS: ``constant_throughput(1)`` × 200 users):

    pip install -r benchmarks/requirements.txt
    python -m locust -f benchmarks/locustfile.py --headless \\
      --host http://127.0.0.1:8000 -u 200 -r 50 --run-time 3m \\
      --only-summary --html benchmarks/results/locust_report.html

Tune `-u` if RPS in the HTML summary drifts from the target.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from locust import HttpUser, constant_throughput, events, task
from locust.runners import WorkerRunner

load_dotenv(ROOT / ".env")

from src.smoke_test import sample_real_keys  # noqa: E402


def _build_key_pool(n: int) -> list[tuple[int, int]]:
    keys = sample_real_keys(min(n, 5000))
    if not keys:
        return [
            (random.randint(1, 10_000), random.randint(1, 5_000)) for _ in range(n)
        ]
    return keys


_POOL_SIZE = int(os.getenv("LOCUST_KEY_POOL", "2000"))
_KEY_POOL = _build_key_pool(_POOL_SIZE)


@events.init.add_listener
def _on_locust_init(environment, **_kwargs) -> None:
    if isinstance(environment.runner, WorkerRunner):
        return
    print(
        f"[locust] key_pool={len(_KEY_POOL)} "
        f"(DATABASE_URL + ingest/features for real IDs; else random fallback)",
        flush=True,
    )


class PredictUser(HttpUser):
    """~1 task/sec per user → `-u N` ≈ N RPS when the server keeps up."""

    host = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8000")
    wait_time = constant_throughput(1)

    @task
    def predict_(self) -> None:
        user_id, merchant_id = random.choice(_KEY_POOL)
        post_kw: dict = {
            "json": {
                "user_id": int(user_id),
                "merchant_id": int(merchant_id),
                "amount": round(random.uniform(5.0, 5000.0), 2),
            },
            "name": "POST /predict",
            "timeout": 30.0,
        }
        if os.getenv("LOCUST_PREDICT_TIMINGS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            post_kw["params"] = {"timings": "true"}
        self.client.post("/predict", **post_kw)
