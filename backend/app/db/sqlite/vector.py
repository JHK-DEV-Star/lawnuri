"""
SQLite implementation of the vector (embedding) repository.

Embeddings are stored as BLOBs using ``struct.pack`` with float32 arrays.
Cosine similarity search is performed in Python since the sqlite-vec
extension may not be available in all environments.
"""

from __future__ import annotations

import json
import math
import struct
import uuid

import aiosqlite

from app.db.base_repository import BaseVectorRepo
from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Embedding serialization helpers
# ---------------------------------------------------------------------------

def _floats_to_blob(floats: list[float]) -> bytes:
    """Serialize a float list to a binary blob (float32 little-endian)."""
    return struct.pack(f"{len(floats)}f", *floats)


def _blob_to_floats(blob: bytes) -> list[float]:
    """Deserialize a binary blob back to a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SQLiteVectorRepo(BaseVectorRepo):
    """SQLite-backed vector chunk storage with Python cosine similarity."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_chunks(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Batch-insert text chunks with their embedding vectors.

        Each entry in *chunks* should be a dict with at least a
        ``"content"`` key.  Additional metadata keys (``chunk_id``,
        ``doc_id``, ``doc_name``, ``chunk_index``, etc.) are preserved
        as JSON in the ``metadata`` column.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical collection name (e.g. ``"team_a"``, ``"shared"``).
            chunks: List of chunk dicts, each containing at least ``"content"``.
            embeddings: Parallel list of embedding vectors (one per chunk).

        Raises:
            ValueError: If *chunks* and *embeddings* have different lengths.
        """
        if not chunks:
            logger.debug("add_chunks called with empty list; skipping.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length. "
                f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        rows: list[tuple[str, str, str, str, str, bytes]] = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
            content = chunk.get("content", "")
            metadata = {k: v for k, v in chunk.items() if k != "content"}
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            emb_blob = _floats_to_blob(embedding)
            rows.append((chunk_id, debate_id, pool, content, metadata_json, emb_blob))

        try:
            await db.executemany(
                """
                INSERT INTO vector_chunks
                    (chunk_id, debate_id, pool, content, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            await db.commit()
            logger.info(
                "Added %d vector chunks (debate=%s, pool=%s).",
                len(rows), debate_id, pool,
            )

        except Exception:
            logger.exception(
                "Failed to add vector chunks (debate=%s, pool=%s).",
                debate_id, pool,
            )
            raise

    async def search(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the *top_k* most similar chunks by cosine similarity.

        All embeddings for the given debate + pool are loaded and
        compared in Python.  This is acceptable for moderate collection
        sizes typical in a single-debate context.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical collection name.
            query_embedding: The query vector to compare against.
            top_k: Maximum number of results to return.

        Returns:
            A list of result dicts ordered by descending similarity, each
            containing: ``chunk_id``, ``content``, ``metadata``,
            ``similarity``.
        """
        try:
            async with db.execute(
                """
                SELECT chunk_id, content, metadata, embedding
                FROM vector_chunks
                WHERE debate_id = ? AND pool = ?
                """,
                (debate_id, pool),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                logger.debug(
                    "Vector search found 0 candidates (debate=%s, pool=%s).",
                    debate_id, pool,
                )
                return []

            scored: list[tuple[float, dict]] = []
            for chunk_id, content, metadata_json, emb_blob in rows:
                stored_emb = _blob_to_floats(emb_blob)
                sim = _cosine_similarity(query_embedding, stored_emb)

                metadata: dict = {}
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except (json.JSONDecodeError, TypeError):
                        pass

                scored.append((
                    sim,
                    {
                        "chunk_id": chunk_id,
                        "content": content,
                        "metadata": metadata,
                        "similarity": sim,
                    },
                ))

            # Sort by similarity descending and take top_k.
            scored.sort(key=lambda t: t[0], reverse=True)
            results = [item for _, item in scored[:top_k]]

            logger.debug(
                "Vector search returned %d/%d results (debate=%s, pool=%s).",
                len(results), len(rows), debate_id, pool,
            )
            return results

        except Exception:
            logger.exception(
                "Vector search failed (debate=%s, pool=%s).",
                debate_id, pool,
            )
            return []

    async def delete_collection(
        self, db: aiosqlite.Connection, debate_id: str, pool: str
    ) -> int:
        """
        Delete all chunks for a debate + pool combination.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical collection name.

        Returns:
            The number of rows deleted.
        """
        try:
            cursor = await db.execute(
                "DELETE FROM vector_chunks WHERE debate_id = ? AND pool = ?",
                (debate_id, pool),
            )
            await db.commit()
            removed = cursor.rowcount
            logger.info(
                "Deleted %d vector chunks (debate=%s, pool=%s).",
                removed, debate_id, pool,
            )
            return removed

        except Exception:
            logger.exception(
                "Failed to delete vector chunks (debate=%s, pool=%s).",
                debate_id, pool,
            )
            raise

    async def count(
        self, db: aiosqlite.Connection, debate_id: str, pool: str
    ) -> int:
        """
        Return the number of stored chunks for a debate + pool.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical collection name.

        Returns:
            The chunk count.
        """
        try:
            async with db.execute(
                "SELECT COUNT(*) FROM vector_chunks WHERE debate_id = ? AND pool = ?",
                (debate_id, pool),
            ) as cursor:
                row = await cursor.fetchone()

            return row[0] if row else 0

        except Exception:
            logger.exception(
                "Failed to count vector chunks (debate=%s, pool=%s).",
                debate_id, pool,
            )
            return 0
