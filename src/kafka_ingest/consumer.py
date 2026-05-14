"""Read transaction events from Kafka and INSERT into raw_transactions.

Semantics mirror PROJECT.md: manual offset commits after successful DB insert;
failed parses / inserts go to the DLQ topic and the offset is still committed
so a poison message cannot wedge the consumer forever.

Usage:
    python -m src.kafka_ingest.consumer
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

from src import db
from src.ingest import COLUMNS, to_python_scalar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kafka_ingest.consumer")


def _event_to_row(event: dict) -> tuple:
    missing = [c for c in COLUMNS if c not in event]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    return tuple(to_python_scalar(event[c]) for c in COLUMNS)


def _flush_batch(rows: list[tuple]) -> None:
    if not rows:
        return
    with db.get_connection() as conn:
        db.init_schema(conn)
        db.bulk_insert_ignore_conflicts(
            conn,
            table="raw_transactions",
            columns=COLUMNS,
            rows=rows,
            conflict_col="transaction_id",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", "transactions"),
    )
    parser.add_argument(
        "--dlq-topic",
        default=os.getenv("KAFKA_DLQ_TOPIC", "transactions-dlq"),
    )
    parser.add_argument(
        "--group-id",
        default=os.getenv("KAFKA_GROUP_ID", "postgres-sink"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("KAFKA_CONSUMER_BATCH", "500")),
    )
    args = parser.parse_args()

    from kafka import KafkaConsumer, KafkaProducer

    dlq = KafkaProducer(
        bootstrap_servers=args.bootstrap.split(","),
        value_serializer=lambda v: v,
    )

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap.split(","),
        group_id=args.group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        # Raw bytes — we json.loads in the loop so bad payloads can go to DLQ.
        value_deserializer=lambda v: v,
    )

    log.info(
        "Consumer started topic=%s group=%s batch=%d — waiting...",
        args.topic,
        args.group_id,
        args.batch_size,
    )

    processed = 0
    batch: list[tuple] = []
    t_wall = time.perf_counter()
    last_tick = t_wall
    processed_at_tick = 0

    def _maybe_log_throughput() -> None:
        nonlocal last_tick, processed_at_tick
        now = time.perf_counter()
        window = now - last_tick
        if window < 1.0:
            return
        n = processed - processed_at_tick
        log.info(
            "Sink throughput: %.0f committed rows/s (1s window) | total=%d",
            n / window,
            processed,
        )
        last_tick = now
        processed_at_tick = processed

    try:
        for message in consumer:
            try:
                payload = json.loads(message.value.decode("utf-8"))
                row = _event_to_row(payload)
                batch.append(row)
                if len(batch) >= args.batch_size:
                    _flush_batch(batch)
                    processed += len(batch)
                    batch.clear()
                    consumer.commit()
                    if processed % 5000 == 0:
                        log.info("Committed %d rows", processed)
                    _maybe_log_throughput()
            except Exception as e:
                log.exception("Bad message offset=%s: %s", message.offset, e)
                dlq.send(args.dlq_topic, message.value)
                dlq.flush()
                consumer.commit()
    except KeyboardInterrupt:
        log.info("Shutdown — flushing %d buffered rows", len(batch))
        if batch:
            _flush_batch(batch)
            processed += len(batch)
            consumer.commit()
        elapsed = time.perf_counter() - t_wall
        if processed and elapsed > 0:
            log.info(
                "Stopped. total_committed=%d wall_s=%.1f avg=%.0f rows/s",
                processed,
                elapsed,
                processed / elapsed,
            )
        raise


if __name__ == "__main__":
    main()
