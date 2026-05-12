# Argos — MVP

Real-time-ish ML fraud detection. This branch is the **minimum viable
pipeline**: ingest → features → train → serve, all running on your laptop
with no external services. Production-grade infra (Kafka, Spark, Redis,
Supabase, Kubernetes, Prometheus) is layered in over later stages — see
[`PROJECT.md`](PROJECT.md) for the full target architecture.

## Architecture

```
[CSV or synthetic]  →  src/ingest.py        →  raw_transactions
                                                 ↓
                       src/features.py      →  user_features, merchant_features  (offline)
                                                 ↓
                       src/sync_to_redis.py →  Redis hashes                       (online cache)
                                                 ↓
                       src/train.py         →  models/fraud_detector_v1.pt + scaler.pkl
                                                 ↓
                       src/serve.py         →  FastAPI /predict on :8000
```

Backends are pluggable:

| Layer | Default | Override |
|---|---|---|
| Offline tables | SQLite (`argos.db`) | `DATABASE_URL=postgresql://...` → Supabase |
| Online cache | In-memory dict | `REDIS_URL=redis://localhost:6379` → Redis |

All backend selection happens in `src/db.py` and `src/feature_store.py`.
Every other script is backend-agnostic, so the API code is the same whether
you're running on a laptop with SQLite + in-memory or on Supabase + Redis.

## Quickstart — SQLite (no setup)

```bash
# 1. Set up Python env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 2-5. Pipeline (SQLite is automatic if DATABASE_URL is not set)
python -m src.ingest
python -m src.features
python -m src.train
uvicorn src.serve:app --port 8000

# 6. (in another terminal) smoke test
python -m src.smoke_test --requests 200
```

To train on real data, drop `train_transaction.csv` from the
[IEEE-CIS Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection)
into `data/` and re-run from `ingest`.

## Quickstart — Supabase Postgres

1. Create a Supabase project (free tier is fine).
2. In **Settings → Database → Connection string**, copy the
   **Transaction pooler** URL (port 6543). This URL goes through PgBouncer,
   which is required because Supabase free tier caps total connections.
3. Copy `.env.example` to `.env` and paste the URL after `DATABASE_URL=`.
4. Open the **SQL Editor** in Supabase, paste the contents of
   [`schema.sql`](schema.sql), and run it.
5. Re-run the same pipeline as above:

```bash
python -m src.ingest      # writes to Postgres now
python -m src.features    # writes user/merchant features
python -m src.train       # reads joined data from Postgres
uvicorn src.serve:app --port 8000
```

Switch back to SQLite anytime by emptying the `DATABASE_URL` line in `.env`.

## Quickstart — Redis online feature store

This adds an online cache between the offline DB and the API. Requires
Docker Desktop running.

1. Bring up Redis locally:
   ```bash
   docker compose up -d redis
   docker compose ps    # should show argos-redis as healthy
   ```
2. Add `REDIS_URL=redis://localhost:6379` to your `.env`.
3. Sync the latest feature tables into Redis (run after `features.py`):
   ```bash
   python -m src.sync_to_redis
   ```
4. Restart the API. It will detect `REDIS_URL` and use the Redis backend
   automatically:
   ```bash
   uvicorn src.serve:app --port 8000
   ```
5. Confirm via `/health` — should show `"feature_store": "redis"`.

The cache uses a 1-hour TTL (override with `--ttl`). Rerun `sync_to_redis`
whenever you want fresh aggregates; the API picks them up immediately.

To go back to the in-memory store, comment out `REDIS_URL` and restart.

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1234, "merchant_id": 42, "amount": 250.0}'
```

```json
{
  "fraud_score": 0.0123,
  "risk_label": "low",
  "model_version": "v1",
  "latency_ms": 1.85,
  "used_user_features": true,
  "used_merchant_features": true
}
```

Other endpoints:

| Endpoint | What |
|---|---|
| `GET /health` | liveness + how many users/merchants are loaded |
| `GET /stats`  | request counters and rolling p50/p95/p99 latency |

## Layout

```
argos/
├── PROJECT.md             # full production target
├── README.md              # this file
├── requirements.txt
├── docker-compose.yml     # local infra services (Redis today, Kafka soon)
├── schema.sql             # Postgres DDL for Supabase SQL editor
├── .env.example           # template; copy to .env and fill in URLs
├── data/                  # raw inputs (CSV); gitignored
├── models/                # trained artifacts; gitignored
├── argos.db               # SQLite store; gitignored
└── src/
    ├── config.py          # paths, feature order
    ├── db.py              # offline backend selector (SQLite ↔ Postgres)
    ├── ingest.py          # CSV → DB (synthetic fallback)
    ├── features.py        # pandas aggregates → user/merchant feature tables
    ├── feature_store.py   # Protocol + InMemoryFeatureStore + factory
    ├── redis_store.py     # RedisFeatureStore (online cache implementation)
    ├── sync_to_redis.py   # push feature tables from DB into Redis
    ├── model.py           # PyTorch FraudDetector
    ├── train.py           # training loop + metrics + artifact save
    ├── serve.py           # FastAPI app
    └── smoke_test.py      # quick end-to-end latency check
```

## Roadmap to the full PROJECT.md architecture

Each stage swaps **one** component without touching the others.

1. **MVP** — SQLite, in-memory store, single uvicorn process. ✅
2. **Postgres / Supabase** — `src/db.py` selects backend via `DATABASE_URL`; SQLite still works offline. ✅
3. **Redis online store** — `src/feature_store.py` is a factory; `REDIS_URL` switches to `RedisFeatureStore`. ✅
4. **Kafka** — replace direct CSV ingest with a Kafka producer/consumer pair writing to Postgres.
5. **Spark** — replace pandas in `features.py` with a Spark job that reads/writes Postgres.
6. **Docker** — wrap `serve.py` in a Dockerfile.
7. **Kubernetes** — deployment + HPA + service.
8. **Prometheus / Grafana** — replace `/stats` with `prometheus_client` instrumentation.


