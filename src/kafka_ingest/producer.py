"""Stream transaction rows from CSV/synthetic data into Kafka.

Usage:
    docker compose up -d zookeeper kafka
    python -m src.kafka_ingest.producer
    python -m src.kafka_ingest.producer --synthetic --rows 10000
    python -m src.kafka_ingest.producer --stats-interval 0.5

Pairs with `python -m src.kafka_ingest.consumer` in another terminal.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

from src.ingest import COLUMNS, load_transactions_dataframe, to_python_scalar
from src.kafka_ingest.topics import ensure_topics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kafka_ingest.producer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers (comma-separated). Env: KAFKA_BOOTSTRAP_SERVERS.",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", "transactions"),
        help="Topic name. Env: KAFKA_TOPIC.",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="If >0, sleep this many seconds after every --batch-size sends (throttle).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="When --sleep is set, sleep after every N sends. Producer batching "
        "uses linger_ms and batch_size (not this flag).",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=1.0,
        help="Seconds between log lines with recent publish throughput.",
    )
    args = parser.parse_args()

    ensure_topics(
        args.bootstrap,
        [
            (args.topic, 3, 1),
            (
                os.getenv("KAFKA_DLQ_TOPIC", "transactions-dlq"),
                1,
                1,
            ),
        ],
    )

    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap.split(","),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        compression_type="lz4",
        acks="all",
        enable_idempotence=True,
        # kafka-python requires 1 when idempotent; >1 raises KafkaConfigurationError
        max_in_flight_requests_per_connection=1,
        retries=5,
        linger_ms=5,
        batch_size=32768,
    )

    df = load_transactions_dataframe(synthetic=args.synthetic, rows=args.rows)
    log.info("Streaming %d rows to topic %s", len(df), args.topic)

    sent = 0
    t0 = time.perf_counter()
    last_tick = t0
    last_sent = 0

    def _maybe_log_stats() -> None:
        nonlocal last_tick, last_sent
        now = time.perf_counter()
        if args.stats_interval <= 0 or now - last_tick < args.stats_interval:
            return
        window = now - last_tick
        n = sent - last_sent
        log.info(
            "Throughput: %.0f events/s (avg over %.2fs) | total_sent=%d",
            n / window if window > 0 else 0.0,
            window,
            sent,
        )
        last_tick = now
        last_sent = sent

    for tup in df[COLUMNS].itertuples(index=False, name=None):
        row = {col: to_python_scalar(val) for col, val in zip(COLUMNS, tup)}
        producer.send(args.topic, row)
        sent += 1
        if sent % 5000 == 0:
            log.info("Sent %d events", sent)
        _maybe_log_stats()
        if args.sleep and sent % args.batch_size == 0:
            time.sleep(args.sleep)


    now = time.perf_counter()
    if sent > last_sent:
        window = now - last_tick
        if window > 0:
            n = sent - last_sent
            log.info(
                "Throughput: %.0f events/s (avg over %.2fs) | total_sent=%d",
                n / window,
                window,
                sent,
            )

    t_flush = time.perf_counter()
    producer.flush()
    log.info(
        "Flush complete: %.2fs waiting for broker acks on pending batches",
        time.perf_counter() - t_flush,
    )
    producer.close()
    elapsed = time.perf_counter() - t0
    log.info(
        "Done. total_sent=%d  wall_s=%.2f  avg=%.0f events/s",
        sent,
        elapsed,
        sent / elapsed if elapsed > 0 else 0.0,
    )


if __name__ == "__main__":
    main()
