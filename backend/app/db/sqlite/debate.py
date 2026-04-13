"""
SQLite implementation of the debate repository.

Each debate's full LangGraph state is stored as a JSON blob in the
``state`` column.  Frequently queried fields (status, current_round,
situation_brief, etc.) are also stored in dedicated columns for
efficient listing and filtering without deserializing the full state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.db.base_repository import BaseDebateRepo
from app.utils.logger import logger


class SQLiteDebateRepo(BaseDebateRepo):
    """SQLite-backed debate state persistence."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC time as an ISO 8601 string."""
        return datetime.now().isoformat()

    @staticmethod
    def _row_to_state(row: aiosqlite.Row) -> dict:
        """
        Reconstruct a full state dict from a database row.

        The row is expected to come from a ``SELECT *`` against the
        ``debates`` table with ``row_factory = aiosqlite.Row``.
        """
        state: dict = json.loads(row["state"]) if row["state"] else {}

        # Ensure top-level fields from dedicated columns are present in
        # the returned state dict (they are the source of truth for
        # listing queries, but the JSON blob is authoritative for the
        # full state).
        state.setdefault("debate_id", row["debate_id"])
        state.setdefault("situation_brief", row["situation_brief"])
        state.setdefault("status", row["status"])
        state.setdefault("current_round", row["current_round"])
        state.setdefault("created_at", row["created_at"])
        state.setdefault("updated_at", row["updated_at"])

        if row["analysis"]:
            state.setdefault("analysis", json.loads(row["analysis"]))

        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self, db: aiosqlite.Connection, debate_id: str) -> dict | None:
        """
        Load a single debate's full state by its ID.

        Args:
            db: Active aiosqlite connection.
            debate_id: The unique debate identifier.

        Returns:
            The full state dict, or *None* if no debate with this ID exists.
        """
        try:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM debates WHERE debate_id = ?",
                (debate_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                logger.debug("Debate '%s' not found.", debate_id)
                return None

            state = self._row_to_state(row)
            logger.debug("Debate '%s' loaded (status=%s).", debate_id, state.get("status"))
            return state

        except Exception:
            logger.exception("Failed to load debate '%s'.", debate_id)
            return None

    async def load_status(self, db: aiosqlite.Connection, debate_id: str) -> str | None:
        """Load only the status column (no JSON deserialization)."""
        try:
            async with db.execute(
                "SELECT status FROM debates WHERE debate_id = ?",
                (debate_id,),
            ) as cursor:
                row = await cursor.fetchone()
            return row[0] if row else None
        except Exception:
            logger.exception("Failed to load status for debate '%s'.", debate_id)
            return None

    async def save(self, db: aiosqlite.Connection, debate_id: str, state: dict) -> None:
        """
        Create or replace a debate's full state.

        Extracts frequently queried fields into dedicated columns and
        stores the complete state as a JSON blob.

        Args:
            db: Active aiosqlite connection.
            debate_id: The unique debate identifier.
            state: The complete LangGraph state dictionary to persist.
        """
        now = self._now_iso()

        # Extract top-level summary fields for indexed columns.
        situation_brief = state.get("situation_brief", "")
        status = state.get("status", "created")
        current_round = state.get("current_round", state.get("round", 0))
        created_at = state.get("created_at", now)
        updated_at = now

        # Analysis may be a dict or already serialized.
        analysis_raw = state.get("analysis")
        analysis_json: str | None = None
        if analysis_raw is not None:
            if isinstance(analysis_raw, dict):
                analysis_json = json.dumps(analysis_raw, ensure_ascii=False)
            else:
                analysis_json = str(analysis_raw)

        # Ensure timestamps are present in the state blob as well.
        state["created_at"] = created_at
        state["updated_at"] = updated_at

        state_json = json.dumps(state, ensure_ascii=False, default=str)

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO debates
                    (debate_id, situation_brief, status, analysis,
                     current_round, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    debate_id,
                    situation_brief,
                    status,
                    analysis_json,
                    current_round,
                    state_json,
                    created_at,
                    updated_at,
                ),
            )
            await db.commit()
            logger.info(
                "Debate '%s' saved (status=%s, round=%s).",
                debate_id, status, current_round,
            )

        except Exception:
            logger.exception("Failed to save debate '%s'.", debate_id)
            raise

    async def list_all(self, db: aiosqlite.Connection) -> list[dict]:
        """
        Return summary dicts for every debate, ordered newest first.

        Only the indexed columns are returned (no full state blob),
        making this efficient for dashboard listing.

        Returns:
            A list of summary dicts with keys: debate_id, situation_brief,
            status, analysis, current_round, created_at, updated_at.
        """
        try:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT debate_id, situation_brief, status, analysis,
                       current_round, created_at, updated_at
                FROM debates
                ORDER BY updated_at DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()

            results: list[dict] = []
            for row in rows:
                entry: dict[str, Any] = {
                    "debate_id": row["debate_id"],
                    "situation_brief": row["situation_brief"],
                    "status": row["status"],
                    "current_round": row["current_round"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                if row["analysis"]:
                    try:
                        entry["analysis"] = json.loads(row["analysis"])
                    except (json.JSONDecodeError, TypeError):
                        entry["analysis"] = None
                else:
                    entry["analysis"] = None
                results.append(entry)

            logger.debug("list_all returned %d debates.", len(results))
            return results

        except Exception:
            logger.exception("Failed to list debates.")
            return []

    async def update(
        self, db: aiosqlite.Connection, debate_id: str, **kwargs: Any
    ) -> dict | None:
        """
        Merge keyword arguments into an existing debate's state and persist.

        This is a convenience method that loads the current state, applies
        the updates, and saves the result.

        Args:
            db: Active aiosqlite connection.
            debate_id: The debate to update.
            **kwargs: Fields to merge into the state dict.

        Returns:
            The updated state dict, or *None* if the debate was not found.
        """
        state = await self.load(db, debate_id)
        if state is None:
            logger.warning("update: debate '%s' not found.", debate_id)
            return None

        state.update(kwargs)
        await self.save(db, debate_id, state)
        return state

    async def delete(self, db: aiosqlite.Connection, debate_id: str) -> bool:
        """
        Delete a debate by its ID.

        Args:
            db: Active aiosqlite connection.
            debate_id: The debate to delete.

        Returns:
            *True* if a row was removed, *False* if the debate did not exist.
        """
        try:
            cursor = await db.execute(
                "DELETE FROM debates WHERE debate_id = ?",
                (debate_id,),
            )
            await db.commit()
            removed = cursor.rowcount > 0
            if removed:
                logger.info("Debate '%s' deleted.", debate_id)
            else:
                logger.debug("delete: debate '%s' not found.", debate_id)
            return removed

        except Exception:
            logger.exception("Failed to delete debate '%s'.", debate_id)
            raise
