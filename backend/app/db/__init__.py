"""
Database abstraction layer for LawNuri.

Provides async database access, connection lifecycle management,
and abstract repository interfaces for all persistence operations.
"""

from app.db.database import init_db, close_db, get_db, get_db_connection

__all__ = ["init_db", "close_db", "get_db", "get_db_connection"]
