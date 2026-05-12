"""Feature store.

Backends
--------
- `InMemoryFeatureStore`: loads `user_features` and
  `merchant_features` rows from the offline DB into Python dicts at API
  startup. Zero external dependencies, but features are stale until the
  API is restarted, and every replica holds its own copy in RAM.

- `RedisFeatureStore` (enabled when REDIS_URL is set in .env): looks up
  features in Redis hashes on each request. Features can be refreshed
  without restarting the API, and
  many API replicas share the same cache.

Both implement the same `FeatureStore` Protocol, so `src/serve.py` doesn't
care which one is active.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sqlalchemy import text

from src import db

log = logging.getLogger("feature_store")


@runtime_checkable
class FeatureStore(Protocol):
    """Interface every feature-store backend implements.

    Both `get_user_features` and `get_merchant_features` MUST return an empty
    dict on miss — callers (serve.py) use the emptiness to set the
    `used_user_features` / `used_merchant_features` flags in the response.
    """

    def get_user_features(self, user_id: int) -> dict: ...
    def get_merchant_features(self, merchant_id: int) -> dict: ...

    @property
    def num_users(self) -> int: ...
    @property
    def num_merchants(self) -> int: ...
    @property
    def backend_name(self) -> str: ...


@dataclass
class InMemoryFeatureStore:
    users: dict[int, dict] = field(default_factory=dict)
    merchants: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "InMemoryFeatureStore":
        """Pull both feature tables into RAM via the SQLAlchemy engine.

        One-shot, blocking. Called from FastAPI's lifespan handler on boot.
        """
        store = cls()
        engine = db.get_engine()
        with engine.connect() as conn:
            for row in conn.execute(text("SELECT * FROM user_features")).mappings():
                d = dict(row)
                store.users[int(d.pop("user_id"))] = d
            for row in conn.execute(text("SELECT * FROM merchant_features")).mappings():
                d = dict(row)
                store.merchants[int(d.pop("merchant_id"))] = d
        log.info(
            "InMemoryFeatureStore loaded from %s: %d users, %d merchants",
            db.describe()["backend"], len(store.users), len(store.merchants),
        )
        return store

    def get_user_features(self, user_id: int) -> dict:
        return self.users.get(int(user_id), {})

    def get_merchant_features(self, merchant_id: int) -> dict:
        return self.merchants.get(int(merchant_id), {})

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_merchants(self) -> int:
        return len(self.merchants)

    @property
    def backend_name(self) -> str:
        return "in_memory"


def load_feature_store() -> FeatureStore:
    """Pick the right backend based on env vars.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url and redis_url.strip():
        # Import here so InMemory users don't need redis installed.
        from src.redis_store import RedisFeatureStore
        return RedisFeatureStore(redis_url.strip())
    return InMemoryFeatureStore.load()
