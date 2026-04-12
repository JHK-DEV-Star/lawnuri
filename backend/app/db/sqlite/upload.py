"""
SQLite implementation of the file-upload metadata repository.

Stores metadata about uploaded files (filename, path, size, MIME type)
scoped to individual debates.  The ``content`` column stores the
server-side file path; the actual file content lives on disk.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.db.base_repository import BaseUploadRepo
from app.utils.logger import logger


class SQLiteUploadRepo(BaseUploadRepo):
    """SQLite-backed upload metadata storage."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mime_from_filename(filename: str) -> str:
        """Infer MIME type from a filename extension."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mime_map = {
            "pdf": "application/pdf",
            "txt": "text/plain",
            "md": "text/markdown",
            "markdown": "text/markdown",
        }
        return mime_map.get(ext, "application/octet-stream")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        filename: str,
        file_path: str,
        file_size: int,
    ) -> dict:
        """
        Record a new file upload.

        The ``content`` column of the ``uploaded_documents`` table is
        used to store the server-side file path.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical grouping (e.g. ``"team_a"``, ``"shared"``).
                Currently stored only in the returned dict; the schema
                does not have a dedicated pool column for uploads.
            filename: Original filename as uploaded by the user.
            file_path: Server-side path where the file is stored.
            file_size: File size in bytes.

        Returns:
            A metadata dict for the newly created record, including the
            generated ``doc_id`` and ``uploaded_at`` timestamp.
        """
        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        mime_type = self._mime_from_filename(filename)

        try:
            await db.execute(
                """
                INSERT INTO uploaded_documents
                    (doc_id, debate_id, filename, content, mime_type,
                     size_bytes, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, debate_id, filename, file_path, mime_type,
                 file_size, now),
            )
            await db.commit()

            result: dict[str, Any] = {
                "doc_id": doc_id,
                "debate_id": debate_id,
                "pool": pool,
                "filename": filename,
                "file_path": file_path,
                "mime_type": mime_type,
                "size_bytes": file_size,
                "uploaded_at": now,
            }
            logger.info(
                "Upload recorded: %s (%s, %d bytes, debate=%s).",
                filename, mime_type, file_size, debate_id,
            )
            return result

        except Exception:
            logger.exception(
                "Failed to record upload '%s' (debate=%s, pool=%s).",
                filename, debate_id, pool,
            )
            raise

    async def list_by_debate(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str | None = None,
    ) -> list[dict]:
        """
        List uploaded documents for a debate.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Optional filter (currently unused at the SQL level).

        Returns:
            A list of document metadata dicts, newest first.
        """
        try:
            async with db.execute(
                """
                SELECT doc_id, debate_id, filename, content,
                       mime_type, size_bytes, uploaded_at
                FROM uploaded_documents
                WHERE debate_id = ?
                ORDER BY uploaded_at DESC
                """,
                (debate_id,),
            ) as cursor:
                rows = await cursor.fetchall()

            results: list[dict] = []
            for row in rows:
                results.append({
                    "doc_id": row[0],
                    "debate_id": row[1],
                    "filename": row[2],
                    "file_path": row[3],  # stored in 'content' column
                    "mime_type": row[4],
                    "size_bytes": row[5],
                    "uploaded_at": row[6],
                })

            logger.debug(
                "list_by_debate returned %d documents (debate=%s).",
                len(results), debate_id,
            )
            return results

        except Exception:
            logger.exception(
                "Failed to list uploads (debate=%s).", debate_id,
            )
            return []

    async def delete_by_debate(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
    ) -> int:
        """
        Delete all upload records for a debate.

        Note: This only removes database metadata.  The caller is
        responsible for deleting the actual files from disk.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.

        Returns:
            The number of rows deleted.
        """
        try:
            cursor = await db.execute(
                "DELETE FROM uploaded_documents WHERE debate_id = ?",
                (debate_id,),
            )
            await db.commit()
            removed = cursor.rowcount
            logger.info(
                "Deleted %d upload records (debate=%s).",
                removed, debate_id,
            )
            return removed

        except Exception:
            logger.exception(
                "Failed to delete upload records (debate=%s).", debate_id,
            )
            raise
