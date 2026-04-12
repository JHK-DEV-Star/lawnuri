"""
SQLite implementations of the cache and anonymization repositories.

- **SQLiteCacheRepo**: General-purpose key-value cache scoped per debate.
- **SQLiteAnonymizationRepo**: PII anonymization token mapping storage.

Both store their payloads as JSON text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.db.base_repository import BaseAnonymizationRepo, BaseCacheRepo
from app.utils.logger import logger


# =========================================================================
# Cache
# =========================================================================

class SQLiteCacheRepo(BaseCacheRepo):
    """SQLite-backed key-value cache, scoped to individual debates."""

    async def get(
        self, db: aiosqlite.Connection, debate_id: str, cache_key: str
    ) -> Any | None:
        """
        Retrieve a cached value by debate ID and key.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            cache_key: The cache entry key.

        Returns:
            The deserialized cached value, or *None* if the key does
            not exist.
        """
        try:
            async with db.execute(
                """
                SELECT data FROM legal_api_cache
                WHERE debate_id = ? AND cache_key = ?
                """,
                (debate_id, cache_key),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return None

            return json.loads(row[0])

        except Exception:
            logger.exception(
                "Cache get failed (debate=%s, key=%s).", debate_id, cache_key,
            )
            return None

    async def set(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        cache_key: str,
        data: Any,
    ) -> None:
        """
        Store a value in the cache (upsert).

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            cache_key: The cache entry key.
            data: Any JSON-serializable value.
        """
        now = datetime.now(timezone.utc).isoformat()
        data_json = json.dumps(data, ensure_ascii=False, default=str)

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO legal_api_cache
                    (cache_key, debate_id, data, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, debate_id, data_json, now),
            )
            await db.commit()
            logger.debug(
                "Cache set (debate=%s, key=%s, %d bytes).",
                debate_id, cache_key, len(data_json),
            )

        except Exception:
            logger.exception(
                "Cache set failed (debate=%s, key=%s).", debate_id, cache_key,
            )
            raise

    async def delete_all(
        self, db: aiosqlite.Connection, debate_id: str
    ) -> int:
        """
        Remove all cache entries for a debate.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.

        Returns:
            The number of rows deleted.
        """
        try:
            cursor = await db.execute(
                "DELETE FROM legal_api_cache WHERE debate_id = ?",
                (debate_id,),
            )
            await db.commit()
            removed = cursor.rowcount
            logger.info(
                "Cache cleared: %d entries removed (debate=%s).",
                removed, debate_id,
            )
            return removed

        except Exception:
            logger.exception(
                "Cache delete_all failed (debate=%s).", debate_id,
            )
            raise


# =========================================================================
# Anonymization mapping
# =========================================================================

class SQLiteAnonymizationRepo(BaseAnonymizationRepo):
    """SQLite-backed PII anonymization mapping storage."""

    async def load(
        self, db: aiosqlite.Connection, debate_id: str
    ) -> dict:
        """
        Load the anonymization mapping for a debate.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.

        Returns:
            A ``{token: original_value}`` dict, or an empty dict if no
            mapping has been saved yet.
        """
        try:
            async with db.execute(
                """
                SELECT mapping FROM anonymization_maps
                WHERE debate_id = ?
                """,
                (debate_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return {}

            return json.loads(row[0])

        except Exception:
            logger.exception(
                "Failed to load anonymization mapping (debate=%s).", debate_id,
            )
            return {}

    async def save(
        self, db: aiosqlite.Connection, debate_id: str, mapping: dict
    ) -> None:
        """
        Replace the anonymization mapping for a debate.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            mapping: The complete ``{token: original_value}`` dict.
        """
        now = datetime.now(timezone.utc).isoformat()
        mapping_json = json.dumps(mapping, ensure_ascii=False)

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO anonymization_maps
                    (debate_id, mapping, updated_at)
                VALUES (?, ?, ?)
                """,
                (debate_id, mapping_json, now),
            )
            await db.commit()
            logger.debug(
                "Anonymization mapping saved (debate=%s, %d entries).",
                debate_id, len(mapping),
            )

        except Exception:
            logger.exception(
                "Failed to save anonymization mapping (debate=%s).", debate_id,
            )
            raise

    async def merge(
        self, db: aiosqlite.Connection, debate_id: str, new_mapping: dict
    ) -> dict:
        """
        Merge new tokens into the existing anonymization mapping.

        Existing tokens are preserved; new tokens are added.  The
        merged result is persisted and returned.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            new_mapping: Additional ``{token: original_value}`` entries.

        Returns:
            The merged mapping dict.
        """
        current = await self.load(db, debate_id)
        current.update(new_mapping)
        await self.save(db, debate_id, current)
        logger.debug(
            "Anonymization mapping merged (debate=%s, %d total entries).",
            debate_id, len(current),
        )
        return current
