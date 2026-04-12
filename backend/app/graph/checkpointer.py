"""
LangGraph checkpointer factory for persistent state across graph invocations.

Uses AsyncSqliteSaver to store node-boundary snapshots so that:
  - Pause/resume restores from the last completed node
  - Server restarts can pick up where the graph left off

Singleton pattern: one aiosqlite connection is reused across all debates
to avoid "database is locked" errors from concurrent connections.
"""

from __future__ import annotations

import asyncio
import os

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import config

_DB_PATH = os.path.join(config.DATA_DIR, "checkpoints.db")

# Module-level singleton
_saver: AsyncSqliteSaver | None = None
_conn: aiosqlite.Connection | None = None
_init_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Lazily create the lock (must be created inside an event loop)."""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


async def get_checkpointer() -> AsyncSqliteSaver:
    """Return a singleton AsyncSqliteSaver connected to the local checkpoint DB."""
    global _saver, _conn

    if _saver is not None:
        return _saver

    async with _get_lock():
        # Double-check after acquiring lock
        if _saver is not None:
            return _saver

        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        _conn = await aiosqlite.connect(_DB_PATH)
        _saver = AsyncSqliteSaver(_conn)
        await _saver.setup()
        return _saver


async def close_checkpointer() -> None:
    """Close the singleton connection (call on shutdown)."""
    global _saver, _conn, _init_lock
    _saver = None
    _init_lock = None
    if _conn is not None:
        await _conn.close()
        _conn = None
