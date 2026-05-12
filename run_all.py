"""End-to-end test runner for Argos.

One script that does the whole pipeline.
 Future roadmap stages (Spark, Docker, k8s, Prometheus,
and an alternate Kafka ingest path) are commented out at the bottom.

Pipeline (default):
    ingest -> features -> sync_to_redis -> train -> serve -> smoke_test

Examples:
    python run_all.py                          # full pipeline
    python run_all.py --synthetic --reset      # rebuild from synthetic data
    python run_all.py --skip train             # reuse last model
    python run_all.py --only smoke_test        # assumes a server is up
    python run_all.py --no-server              # data pipeline only
    python run_all.py --keep-server            # leave API running at the end
    python run_all.py --rows 50000 --epochs 5  # quick smoke run end-to-end

Stages that don't apply silently no-op (e.g. sync_to_redis when REDIS_URL is unset).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # uses your .venv automatically when activated

# Load .env once so REDIS_URL / DATABASE_URL gates work without manual export.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv missing -> rely on real env vars only
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_for_health(url: str, timeout: float = 60.0) -> bool:
    """Poll url every 500ms until it 200s or we hit timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_ingest(args) -> None:
    section("Stage 1: ingest")
    cmd = [PY, "-m", "src.ingest"]
    if args.synthetic:
        cmd.append("--synthetic")
    if args.reset:
        cmd.append("--reset")
    if args.rows:
        cmd += ["--rows", str(args.rows)]
    run(cmd)


def stage_features(_args) -> None:
    section("Stage 2: features")
    run([PY, "-m", "src.features"])


def stage_sync_to_redis(_args) -> None:
    section("Stage 3: sync_to_redis (skipped if REDIS_URL unset)")
    if not (os.getenv("REDIS_URL") or "").strip():
        print("REDIS_URL not set in env/.env — skipping.")
        return
    run([PY, "-m", "src.sync_to_redis"])


def stage_train(args) -> None:
    section("Stage 4: train")
    cmd = [PY, "-m", "src.train"]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    run(cmd)


def stage_serve_and_smoke(args) -> None:
    section("Stage 5: serve + smoke_test")
    serve_cmd = [
        PY, "-m", "uvicorn", "src.serve:app",
        "--port", str(args.port),
        "--log-level", "warning",
    ]
    print(f"$ {' '.join(serve_cmd)}  (background)", flush=True)
    server = subprocess.Popen(serve_cmd, cwd=ROOT)

    try:
        health_url = f"http://localhost:{args.port}/health"
        print(f"waiting for {health_url} ...", flush=True)
        if not wait_for_health(health_url, timeout=60):
            raise SystemExit("server failed to become healthy within 60s")

        run([
            PY, "-m", "src.smoke_test",
            "--host", f"http://localhost:{args.port}",
            "--requests", str(args.requests),
        ])

        if args.keep_server:
            print(f"\nserver left running (pid={server.pid}). Ctrl+C to stop.")
            server.wait()
    finally:
        if not args.keep_server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()


# ---------------------------------------------------------------------------
# Future stages — uncomment as the roadmap lands
# ---------------------------------------------------------------------------

# def stage_kafka_streaming(args) -> None:
#     """Stream-through-Kafka alternative to direct ingest.
#
#     Brings up Zookeeper + Kafka via docker compose, starts the consumer in
#     the background, then runs the producer in the foreground. Use INSTEAD
#     of stage_ingest.
#     """
#     section("Optional: Kafka streaming ingest")
#     run(["docker", "compose", "up", "-d", "zookeeper", "kafka"])
#     consumer = subprocess.Popen([PY, "-m", "src.kafka_ingest.consumer"], cwd=ROOT)
#     try:
#         time.sleep(5)  # give consumer time to join the group
#         run([
#             PY, "-m", "src.kafka_ingest.producer",
#             *(["--synthetic"] if args.synthetic else []),
#             *(["--rows", str(args.rows)] if args.rows else []),
#         ])
#     finally:
#         consumer.terminate()
#         consumer.wait(timeout=10)
#
#
# def stage_spark(args) -> None:
#     """Replace pandas feature compute with Spark batch job."""
#     section("Stage 5: Spark batch feature computation")
#     run([
#         "spark-submit",
#         "--jars", "spark/jars/postgresql-42.6.0.jar",
#         "spark/compute_features.py",
#     ])
#
#
# def stage_docker_build(args) -> None:
#     section("Stage 6: build API container image")
#     run(["docker", "build", "-t", "argos-api:v1", "-f", "api/Dockerfile", "."])
#
#
# def stage_k8s_deploy(args) -> None:
#     section("Stage 7: kubernetes apply")
#     run(["kind", "load", "docker-image", "argos-api:v1", "--name", "argos"])
#     run(["kubectl", "apply", "-f", "k8s/api-deployment.yaml"])
#     run(["kubectl", "apply", "-f", "k8s/api-hpa.yaml"])
#
#
# def stage_prometheus(args) -> None:
#     section("Stage 8: Prometheus + Grafana")
#     run([
#         "helm", "install", "prometheus",
#         "prometheus-community/kube-prometheus-stack",
#     ])
#
#
# def stage_locust(args) -> None:
#     section("Stage 8b: Locust load test")
#     run(["locust", "-f", "load_test/locustfile.py",
#          "--host", f"http://localhost:{args.port}"])


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

STAGE_HANDLERS = {
    "ingest": stage_ingest,
    "features": stage_features,
    "sync_to_redis": stage_sync_to_redis,
    "train": stage_train,
    "serve": stage_serve_and_smoke,
    # As you uncomment future stages above, also register them here, e.g.:
    # "kafka": stage_kafka_streaming,
    # "spark": stage_spark,
    # "docker": stage_docker_build,
    # "k8s": stage_k8s_deploy,
    # "prometheus": stage_prometheus,
    # "locust": stage_locust,
}

DEFAULT_ORDER = ["ingest", "features", "sync_to_redis", "train", "serve"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--skip", default="",
                   help="Comma-separated stage names to skip.")
    p.add_argument("--only", default="",
                   help="Comma-separated stages to run (overrides --skip).")

    # ingest passthroughs
    p.add_argument("--synthetic", action="store_true",
                   help="Force synthetic data instead of IEEE-CIS CSV.")
    p.add_argument("--reset", action="store_true",
                   help="Truncate tables before ingest.")
    p.add_argument("--rows", type=int, default=None,
                   help="Row cap (real or synthetic).")

    # train passthrough
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training epoch count.")

    # serve / smoke passthroughs
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--requests", type=int, default=200,
                   help="Number of requests for smoke_test.")
    p.add_argument("--no-server", action="store_true",
                   help="Skip the serve+smoke stage (data pipeline only).")
    p.add_argument("--keep-server", action="store_true",
                   help="Leave the API running after smoke_test.")

    return p.parse_args()


def select_stages(args) -> list[str]:
    stages = list(DEFAULT_ORDER)
    if args.no_server:
        stages = [s for s in stages if s != "serve"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        stages = [s for s in stages if s in wanted]
    else:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        stages = [s for s in stages if s not in skip]
    return stages


def main() -> int:
    args = parse_args()
    stages = select_stages(args)
    if not stages:
        print("nothing to do", file=sys.stderr)
        return 1
    print(f"running stages: {stages}", flush=True)

    started = time.time()
    try:
        for name in stages:
            STAGE_HANDLERS[name](args)
    except subprocess.CalledProcessError as e:
        print(f"\nstage failed: {' '.join(e.cmd)} (exit {e.returncode})",
              file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    print(f"\ntotal: {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
