"""
SQLite implementation of the settings repository.

Settings are stored as a single JSON blob in a singleton row (id=1).
Encryption of sensitive values is handled by the caller (SettingsManager);
this layer only persists and retrieves the raw JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from app.db.base_repository import BaseSettingsRepo
from app.utils.logger import logger


class SQLiteSettingsRepo(BaseSettingsRepo):
    """SQLite-backed settings storage using a singleton row pattern."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self, db: aiosqlite.Connection) -> dict:
        """
        Load the full settings dictionary from the database.

        Returns:
            The parsed settings dict, or an empty dict if no settings
            have been saved yet.
        """
        try:
            async with db.execute(
                "SELECT data FROM settings WHERE id = 1"
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                logger.debug("No settings row found; returning empty dict.")
                return {}

            data: dict = json.loads(row[0])
            logger.debug("Settings loaded (%d top-level keys).", len(data))
            return data

        except Exception:
            logger.exception("Failed to load settings from database.")
            return {}

    async def save(self, db: aiosqlite.Connection, data: dict) -> None:
        """
        Persist the full settings dictionary.

        Uses INSERT OR REPLACE to create or update the singleton row.

        Args:
            db: Active aiosqlite connection.
            data: The complete settings dictionary to store.
        """
        now = datetime.now(timezone.utc).isoformat()
        json_data = json.dumps(data, ensure_ascii=False)

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO settings (id, data, updated_at)
                VALUES (1, ?, ?)
                """,
                (json_data, now),
            )
            await db.commit()
            logger.info("Settings saved (%d bytes).", len(json_data))

        except Exception:
            logger.exception("Failed to save settings to database.")
            raise

    async def load_masked(self, db: aiosqlite.Connection) -> dict:
        """
        Load settings from the database.

        This method returns the raw stored data; actual masking of
        sensitive fields (API keys, passwords, etc.) is the
        responsibility of the caller (typically SettingsManager).

        Returns:
            The parsed settings dict, or an empty dict if none exist.
        """
        return await self.load(db)
