"""Stream transaction rows from CSV/synthetic data into Kafka.

Usage:
    docker compose up -d zookeeper kafka
    python -m src.kafka_ingest.producer
    python -m src.kafka_ingest.producer --synthetic --rows 10000
    python -m src.kafka_ingest.producer --bootstrap localhost:9092

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
        help="Seconds between batches (throttle demo traffic).",
    )
    parser.add_argument("--batch-size", type=int, default=500)
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
        acks="all",
        enable_idempotence=True,
        max_in_flight_requests_per_connection=5,
        retries=5,
        linger_ms=5,
        batch_size=32768,
    )

    df = load_transactions_dataframe(synthetic=args.synthetic, rows=args.rows)
    log.info("Streaming %d rows to topic %s", len(df), args.topic)

    sent = 0
    batch: list[dict] = []
    for tup in df[COLUMNS].itertuples(index=False, name=None):
        row = {col: to_python_scalar(val) for col, val in zip(COLUMNS, tup)}
        batch.append(row)
        if len(batch) >= args.batch_size:
            for ev in batch:
                producer.send(args.topic, ev)
            producer.flush()
            sent += len(batch)
            if sent % 5000 == 0:
                log.info("Sent %d events", sent)
            batch.clear()
            if args.sleep:
                time.sleep(args.sleep)
    for ev in batch:
        producer.send(args.topic, ev)
    producer.flush()
    sent += len(batch)
    producer.close()
    log.info("Done. Total sent: %d", sent)


if __name__ == "__main__":
    main()
