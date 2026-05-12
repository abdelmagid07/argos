"""Create Kafka topics idempotently (local docker-compose broker)."""
from __future__ import annotations

import logging

log = logging.getLogger("kafka_ingest.topics")


def ensure_topics(
    bootstrap_servers: str,
    topics: list[tuple[str, int, int]],
) -> None:
    """Create topics if missing.

    Each tuple is (name, num_partitions, replication_factor).
    replication_factor must be **1** for single-broker local Kafka.
    """
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
    except ImportError as e:
        raise ImportError(
            "kafka-python is required. pip install kafka-python"
        ) from e

    admin = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers.split(","),
        client_id="argos-topic-bootstrap",
    )
    try:
        for name, partitions, replication in topics:
            try:
                fs = admin.create_topics(
                    [
                        NewTopic(
                            name=name,
                            num_partitions=partitions,
                            replication_factor=replication,
                        )
                    ],
                    validate_only=False,
                )
                # kafka-python returns Dict[str, Future]; errors surface on .result()
                if fs:
                    for fut in fs.values():
                        fut.result(timeout=60)
                log.info("Created topic %s (%d partitions)", name, partitions)
            except TopicAlreadyExistsError:
                log.info("Topic %s already exists — skipping", name)
    finally:
        admin.close()
