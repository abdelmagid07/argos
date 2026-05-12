"""Redis-backed online feature store.

Keys (set by `src.sync_to_redis`):
    user:{user_id}:features         HASH of feature_name -> stringified value
    merchant:{merchant_id}:features HASH of feature_name -> stringified value
    argos:meta                      HASH with sync metadata

Miss policy: returns {} on miss (same contract as InMemoryFeatureStore).
"""
from __future__ import annotations

import logging
from typing import Optional

import redis

log = logging.getLogger("redis_store")

USER_KEY_FMT = "user:{}:features"
MERCHANT_KEY_FMT = "merchant:{}:features"
META_KEY = "argos:meta"


class RedisFeatureStore:
    def __init__(self, url: str, max_connections: int = 20) -> None:
        self.client = redis.Redis.from_url(
            url,
            decode_responses=True,
            max_connections=max_connections,
            socket_keepalive=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        self.client.ping()
        self._url = url
        log.info("RedisFeatureStore connected to %s", url)

    # lookups 

    def get_user_features(self, user_id: int) -> dict:
        data = self.client.hgetall(USER_KEY_FMT.format(int(user_id)))
        return self._coerce(data)

    def get_merchant_features(self, merchant_id: int) -> dict:
        data = self.client.hgetall(MERCHANT_KEY_FMT.format(int(merchant_id)))
        return self._coerce(data)

    @staticmethod
    def _coerce(data: dict[str, str]) -> dict:
        """Redis HGETALL returns all-strings; serve.py expects numerics."""
        out: dict = {}
        for k, v in data.items():
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                out[k] = v
        return out

    # introspection (used by /health) 

    def _meta(self) -> dict[str, str]:
        try:
            return self.client.hgetall(META_KEY) or {}
        except redis.RedisError as e:
            log.warning("Could not read %s: %s", META_KEY, e)
            return {}

    @property
    def num_users(self) -> int:
        meta = self._meta()
        return int(float(meta.get("user_count", 0)))

    @property
    def num_merchants(self) -> int:
        meta = self._meta()
        return int(float(meta.get("merchant_count", 0)))

    @property
    def last_synced_at(self) -> Optional[float]:
        meta = self._meta()
        v = meta.get("synced_at")
        return float(v) if v else None

    @property
    def backend_name(self) -> str:
        return "redis"
