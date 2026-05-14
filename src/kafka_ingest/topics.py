"""Create Kafka topics idempotently (local docker-compose broker)."""
from __future__ import annotations

import logging

log = logging.getLogger("kafka_ingest.topics")


def _raise_for_create_topics_result(result, *, timeout_sec: float = 60.0) -> None:
    """Drain `KafkaAdminClient.create_topics` return value across kafka-python versions.

    Older versions returned ``{topic: Future}``; newer ones return a
    ``CreateTopicsResponse``-like object or a dict from ``to_dict()``.
    """
    import kafka.errors as Errors

    if result is None:
        return
    if isinstance(result, dict):
        if not result:
            return
        first = next(iter(result.values()))
        if callable(getattr(first, "result", None)):
            for fut in result.values():
                fut.result(timeout=timeout_sec)
            return
        for t in result.get("topics", ()):
            code = t.get("error_code", 0) if isinstance(t, dict) else getattr(t, "error_code", 0)
            err_typ = Errors.for_code(code)
            if err_typ not in (Errors.NoError, Errors.TopicAlreadyExistsError):
                raise err_typ()
        return
    topics = getattr(result, "topics", None)
    if topics is not None:
        for t in topics:
            code = getattr(t, "error_code", 0)
            err_typ = Errors.for_code(code)
            if err_typ not in (Errors.NoError, Errors.TopicAlreadyExistsError):
                raise err_typ()
        return
    if callable(getattr(result, "result", None)):
        result.result(timeout=timeout_sec)


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
        new_topics = [
            NewTopic(name=name, num_partitions=parts, replication_factor=repl)
            for name, parts, repl in topics
        ]
        try:
            out = admin.create_topics(new_topics, validate_only=False)
            _raise_for_create_topics_result(out)
            log.info("Ensured topics exist: %s", [t[0] for t in topics])
        except TopicAlreadyExistsError:
            log.info(
                "Kafka topic(s) already exist, continuing: %s",
                [t[0] for t in topics],
            )
    finally:
        admin.close()
