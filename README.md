# Argos — MVP

Real-time-ish ML fraud detection. This branch is the **minimum viable
pipeline**: ingest → features → train → serve, all running on your laptop
with no external services. Production-grade infra (Kafka, Spark, Redis,
Supabase, Kubernetes, Prometheus) is layered in over later stages — see
[`PROJECT.md`](PROJECT.md) for the full target architecture.

## Architecture

```
[CSV or synthetic]  →  src/ingest.py  ────────────────────────────┐
      │                                                             │
      └──── optional Kafka path ─→  Kafka topic `transactions`      │
                      │                                             │
                      └── src.kafka_ingest.consumer → raw_transactions
                                                                  │
                                                                  ▼
                       src/features.py      →  user_features, merchant_features  (offline)
                                                 ↓
                       src/sync_to_redis.py →  Redis hashes                       (online cache)
                                                 ↓
                       src/train.py         →  models/fraud_detector_v1.pt + scaler.pkl
                                                 ↓
                       src/serve.py         →  FastAPI /predict on :8000
```

Either **batch ingest** (`python -m src.ingest`) or **streaming ingest**
(`producer` → Kafka → `consumer`) lands in `raw_transactions`; downstream
is identical.

Backends are pluggable:

| Layer | Default | Override |
|---|---|---|
| Offline tables | SQLite (`argos.db`) | `DATABASE_URL=postgresql://...` → Supabase |
| Online cache | In-memory dict | `REDIS_URL=redis://localhost:6380` → Redis (host port from [`docker-compose.yml`](docker-compose.yml)) |

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

# 2. Run the whole pipeline end-to-end (ingest -> features -> [sync_to_redis] -> train -> serve -> smoke_test).
python run_all.py
# Useful variants:
#   python run_all.py --synthetic --reset    # rebuild from synthetic data
#   python run_all.py --skip train           # reuse last model
#   python run_all.py --no-server            # data pipeline only
#   python run_all.py --keep-server          # leave the API running at the end
```

Manual equivalent (handy if you want to run a single stage):

```bash
python -m src.ingest
python -m src.features
python -m src.train
uvicorn src.serve:app --port 8000
python -m src.smoke_test --requests 200       # in another terminal
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
2. Add `REDIS_URL=redis://localhost:6380` to your `.env` (or whatever host port
   [`docker-compose.yml`](docker-compose.yml) maps to Redis).
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

## Quickstart — Kafka streaming ingest

Requires Docker (Zookeeper + Kafka). Uses **port 9092** on the host — stop any
other broker bound there first.

1. Start infra:
   ```bash
   docker compose up -d zookeeper kafka
   docker compose ps    # wait until argos-kafka is healthy (~40s first boot)
   ```
2. Install deps: `pip install -r requirements.txt` (pulls `kafka-python`).
3. Optional `.env` entries — defaults match [`docker-compose.yml`](docker-compose.yml):
   `KAFKA_BOOTSTRAP_SERVERS=localhost:9092`, `KAFKA_TOPIC=transactions`,
   `KAFKA_DLQ_TOPIC=transactions-dlq`.

4. **Terminal A — consumer** (writes to the same DB as `src.ingest`):
   ```bash
   python -m src.kafka_ingest.consumer
   ```

5. **Terminal B — producer** (reads IEEE-CIS CSV or `--synthetic`):
   ```bash
   python -m src.kafka_ingest.producer
   # or a smaller demo:
   python -m src.kafka_ingest.producer --synthetic --rows 5000
   ```

6. Continue the pipeline as usual: `features` → `train` → `sync_to_redis` → `serve`.

Direct CSV ingest (`python -m src.ingest`) still works and does **not** require
Kafka — pick one path per environment.

## Quickstart — Docker (containerized API)

Bundles the FastAPI service into an image so it runs identically anywhere — and
gives us the building block for the Kubernetes step.

Prerequisite: you've already trained at least once, so `models/` contains
`fraud_detector_v1.pt`, `scaler.pkl`, and `feature_columns.json`. The build
will fail loudly if any of those are missing.

1. Build + run via compose. This also pulls Redis up if it isn't already:
   ```bash
   docker compose up -d --build api
   docker compose ps    # argos-api should report (healthy) after ~20-30s
   ```
2. Hit it on the host — same port (`8000`) as the local uvicorn path:
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"user_id": 1234, "merchant_id": 42, "amount": 250.0}'
   ```
3. Or run the whole pipeline end-to-end with the container as the server:
   ```bash
   python run_all.py --via-docker
   ```
4. Tear down: `docker compose stop api` (or `docker compose down` for everything).

Container details worth knowing:

- **Image tag:** `argos-api:dev` (also reused by future k8s manifests).
- **CPU-only torch wheel** is installed during the build to keep the image at
  ~600MB instead of ~2GB.
- **Non-root** user (`argos`) at runtime.
- **`REDIS_URL` is overridden** inside the container to
  `redis://redis:6379` so it talks to the Redis service over the compose
  network — `localhost:6380` only works from your host shell.
- **`DATABASE_URL`** is read from `.env` (or compose's env_file) unchanged;
  Supabase URLs work from inside the container because they're just external
  DNS names.

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
├── Dockerfile             # API image (CPU torch, non-root, healthcheck)
├── .dockerignore
├── docker-compose.yml     # local infra services + the API container
├── run_all.py             # one-shot end-to-end test runner
├── schema.sql             # Postgres DDL for Supabase SQL editor
├── .env.example           # template; copy to .env and fill in URLs
├── data/                  # raw inputs (CSV); gitignored
├── models/                # trained artifacts; gitignored
├── argos.db               # SQLite store; gitignored
└── src/
    ├── kafka_ingest/      # Kafka producer / consumer / topic bootstrap
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
4. **Kafka** — `src/kafka_ingest/` producer + consumer → same `raw_transactions` schema; optional vs `src.ingest`. ✅
5. **Spark** — *(deferred)* swap pandas in `features.py` for a Spark job. Parked because on this data size Spark would be slower than pandas, and Spark-on-Windows setup is painful. Will revisit if/when scaling demands it.
6. **Docker** — `Dockerfile` + `argos-api:dev` image + compose `api` service; `python run_all.py --via-docker` runs the whole pipeline with the containerized API. ✅
7. **Kubernetes** — deployment + HPA + service for the `argos-api:dev` image.



