"""Stage 4 (MVP): in-memory feature store.

Loads user_features and merchant_features from SQLite into Python dicts at
startup. Lookups are O(1). When we graduate to Redis, only this file changes;
the API contract (`get_user_features`, `get_merchant_features`) stays the
same.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

from src.config import DB_PATH

log = logging.getLogger("feature_store")


@dataclass
class FeatureStore:
    users: dict[int, dict] = field(default_factory=dict)
    merchants: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "FeatureStore":
        store = cls()
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT * FROM user_features"):
                d = dict(row)
                store.users[int(d.pop("user_id"))] = d
            for row in conn.execute("SELECT * FROM merchant_features"):
                d = dict(row)
                store.merchants[int(d.pop("merchant_id"))] = d
        finally:
            conn.close()
        log.info(
            "FeatureStore loaded: %d users, %d merchants",
            len(store.users), len(store.merchants),
        )
        return store

    def get_user_features(self, user_id: int) -> dict:
        return self.users.get(int(user_id), {})

    def get_merchant_features(self, merchant_id: int) -> dict:
        return self.merchants.get(int(merchant_id), {})
