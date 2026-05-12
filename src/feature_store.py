"""Stage 4 (MVP): in-memory feature store.

Loads user_features and merchant_features from the active backend (SQLite or
Postgres) into Python dicts at startup. Lookups are O(1). When we graduate
to Redis, only this file changes; the API contract (`get_user_features`,
`get_merchant_features`) stays the same.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import text

from src import db

log = logging.getLogger("feature_store")


@dataclass
class FeatureStore:
    users: dict[int, dict] = field(default_factory=dict)
    merchants: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "FeatureStore":
        """Pull both feature tables into RAM via the SQLAlchemy engine.

        Using the engine keeps this backend-agnostic. Column names come back
        as dict keys regardless of whether we hit SQLite or Postgres.
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
            "FeatureStore loaded from %s: %d users, %d merchants",
            db.describe()["backend"], len(store.users), len(store.merchants),
        )
        return store

    def get_user_features(self, user_id: int) -> dict:
        return self.users.get(int(user_id), {})

    def get_merchant_features(self, merchant_id: int) -> dict:
        return self.merchants.get(int(merchant_id), {})
