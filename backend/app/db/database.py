"""Async database connection management.

Provides connection lifecycle functions (init, close, get) for use with
FastAPI's lifespan and dependency injection. Uses aiosqlite with WAL mode
for concurrent read access. The DB_BACKEND environment variable is reserved
for a future PostgreSQL switch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

from app.config import config
from app.utils.logger import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_BACKEND: str = os.getenv("DB_BACKEND", "sqlite").lower()
"""Database backend selector.  Only ``'sqlite'`` is implemented today."""

DB_PATH: str = os.getenv(
    "DB_PATH",
    os.path.join(config.DATA_DIR, "lawnuri.db"),
)
"""Filesystem path for the SQLite database file."""

_SCHEMA_PATH: str = os.path.join(os.path.dirname(__file__), "schema.sql")
"""Path to the SQL schema file applied on first startup."""

# ---------------------------------------------------------------------------
# Module-level connection reference
# ---------------------------------------------------------------------------

_connection: aiosqlite.Connection | None = None


# ---------------------------------------------------------------------------
# Public lifecycle functions
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Initialise the database connection and apply the schema.

    This function is idempotent -- it may be called multiple times safely.
    It should be invoked once during application startup (e.g. in the
    FastAPI lifespan handler).

    Behaviour:
        1. Ensure the parent directory for the database file exists.
        2. Open a persistent ``aiosqlite`` connection.
        3. Enable WAL journal mode for better concurrency.
        4. Enable foreign key enforcement.
        5. Execute ``schema.sql`` to create tables (IF NOT EXISTS).
    """
    global _connection

    if _connection is not None:
        logger.debug("init_db called but connection already exists; skipping.")
        return

    if DB_BACKEND != "sqlite":
        raise NotImplementedError(
            f"Database backend '{DB_BACKEND}' is not yet supported. "
            "Set DB_BACKEND=sqlite or leave it unset."
        )

    # Ensure the data directory exists
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Opening SQLite database at %s", DB_PATH)

    _connection = await aiosqlite.connect(DB_PATH)

    # Row factory so we can access columns by name
    _connection.row_factory = aiosqlite.Row

    # Enable WAL mode for concurrent readers
    await _connection.execute("PRAGMA journal_mode=WAL")
    # Enforce foreign key constraints
    await _connection.execute("PRAGMA foreign_keys=ON")
    # Reasonable busy timeout (5 seconds) for write contention
    await _connection.execute("PRAGMA busy_timeout=5000")

    # Apply schema
    try:
        schema_sql = Path(_SCHEMA_PATH).read_text(encoding="utf-8")
        await _connection.executescript(schema_sql)
        await _connection.commit()
        logger.info("Database schema applied successfully.")
    except FileNotFoundError:
        logger.error("Schema file not found at %s", _SCHEMA_PATH)
        raise
    except Exception as exc:
        logger.error("Failed to apply database schema: %s", exc)
        raise

    logger.info("Database initialised (backend=%s, WAL=on, FK=on).", DB_BACKEND)


async def close_db() -> None:
    """
    Close the database connection gracefully.

    Should be called during application shutdown.  Safe to call even if the
    connection was never opened or has already been closed.
    """
    global _connection

    if _connection is None:
        logger.debug("close_db called but no connection exists; nothing to do.")
        return

    try:
        await _connection.close()
        logger.info("Database connection closed.")
    except Exception as exc:
        logger.warning("Error closing database connection: %s", exc)
    finally:
        _connection = None


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    FastAPI dependency that yields the shared database connection.

    Usage::

        @router.get("/example")
        async def example(db=Depends(get_db)):
            cursor = await db.execute("SELECT 1")
            ...

    Raises:
        RuntimeError: If ``init_db()`` has not been called yet.

    Yields:
        The active ``aiosqlite.Connection`` instance.
    """
    if _connection is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() during application startup."
        )
    yield _connection


async def get_db_connection() -> aiosqlite.Connection | None:
    """
    Get the raw database connection (non-generator).

    For use in non-FastAPI contexts (LangGraph nodes, migration scripts).
    Returns None if the database has not been initialised.
    """
    return _connection
