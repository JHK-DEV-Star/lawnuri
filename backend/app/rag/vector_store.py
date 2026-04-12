"""
ChromaDB wrapper for storing and searching text chunk embeddings.

Provides persistent vector storage with metadata filtering,
used by the GraphRAG pipeline to retrieve relevant legal text chunks.

Supports dual-write: SQLite (primary) with ChromaDB fallback.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Helper: obtain the SQLite connection outside of FastAPI DI
# ---------------------------------------------------------------------------

async def _try_get_db():
    """Return the active SQLite connection, or None if unavailable."""
    try:
        from app.db.database import get_db_connection
        return await get_db_connection()
    except Exception:
        return None


class VectorStore:
    """Wrapper around ChromaDB for embedding-based vector search.

    When a SQLite connection is available, all writes go to SQLite first
    (primary) and then to ChromaDB (backward compatibility).  Reads prefer
    SQLite when available, falling back to ChromaDB.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "default",
        debate_id: str = "",
        pool: str = "",
    ) -> None:
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory path where ChromaDB stores its data on disk.
            collection_name: Name of the collection to use or create.
            debate_id: Debate identifier used for SQLite scoping.
            pool: Logical collection / pool name for SQLite scoping.
        """
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._debate_id = debate_id
        self._pool = pool or collection_name

        logger.info(
            "Initializing VectorStore (persist_dir=%s, collection=%s)",
            persist_dir,
            collection_name,
        )

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create_collection handles the case where the collection
        # does not exist yet -- it will be created automatically.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "VectorStore ready. Collection '%s' holds %d chunks.",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> None:
        """
        Add text chunks with pre-computed embeddings and metadata.

        Writes to SQLite first (if available), then ChromaDB for backward
        compatibility.

        Each metadata dict should ideally contain:
            - chunk_id:       Unique identifier for the chunk.
            - doc_id:         Identifier for the parent document.
            - doc_name:       Human-readable document name.
            - chunk_index:    Integer position of this chunk within the document.
            - original_text:  Truncated preview of the chunk text (for display).

        Args:
            chunks: Raw text strings for each chunk.
            metadatas: List of metadata dicts (one per chunk).
            embeddings: List of embedding vectors (one per chunk).
            ids: Optional explicit IDs. If *None*, UUIDs are generated.
        """
        if not chunks:
            logger.warning("add_chunks called with an empty chunk list; skipping.")
            return

        if len(chunks) != len(metadatas) or len(chunks) != len(embeddings):
            raise ValueError(
                "chunks, metadatas, and embeddings must all have the same length. "
                f"Got {len(chunks)}, {len(metadatas)}, {len(embeddings)}."
            )

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Ensure every metadata dict includes a truncated original_text preview
        prepared_metadatas: list[dict[str, Any]] = []
        for i, meta in enumerate(metadatas):
            entry = dict(meta)
            if "original_text" not in entry:
                entry["original_text"] = chunks[i][:200]
            prepared_metadatas.append(entry)

        logger.debug("Adding %d chunks to collection '%s'.", len(chunks), self._collection_name)

        # --- SQLite primary write ---
        if self._debate_id:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.vector import SQLiteVectorRepo
                    repo = SQLiteVectorRepo()
                    sqlite_chunks = [
                        {"chunk_id": ids[i], "content": chunks[i], **prepared_metadatas[i]}
                        for i in range(len(chunks))
                    ]
                    await repo.add_chunks(db, self._debate_id, self._pool, sqlite_chunks, embeddings)
                    logger.info("SQLite vector write succeeded (%d chunks).", len(chunks))
                except Exception:
                    logger.warning("SQLite vector write failed; ChromaDB will still be written.", exc_info=True)

        # --- ChromaDB fallback / backward-compat write ---
        # ChromaDB handles batching internally, but very large inserts
        # may need to be split. We use a batch size of 5000 to stay safe.
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                documents=chunks[start:end],
                metadatas=prepared_metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        logger.info(
            "Successfully added %d chunks. Collection now holds %d total.",
            len(chunks),
            self._collection.count(),
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Search by embedding vector, returning the closest chunks.

        Uses SQLite if available, otherwise falls back to ChromaDB.

        Args:
            query_embedding: The query vector to compare against stored embeddings.
            top_k: Maximum number of results to return.
            where: Optional ChromaDB *where* filter dict for metadata filtering
                   (e.g. ``{"doc_id": "some-doc-id"}``).

        Returns:
            A list of dicts, each containing:
                - id:        Chunk ID.
                - document:  Original chunk text.
                - metadata:  Full metadata dict.
                - distance:  Cosine distance to the query vector (lower = closer).
        """
        logger.debug(
            "Searching collection '%s' (top_k=%d, where=%s).",
            self._collection_name,
            top_k,
            where,
        )

        # --- Try SQLite first ---
        if self._debate_id:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.vector import SQLiteVectorRepo
                    repo = SQLiteVectorRepo()
                    sqlite_results = await repo.search(
                        db, self._debate_id, self._pool, query_embedding, top_k,
                    )
                    if sqlite_results:
                        # Convert SQLite result format to match ChromaDB format
                        output: list[dict] = []
                        for r in sqlite_results:
                            output.append({
                                "id": r["chunk_id"],
                                "document": r["content"],
                                "metadata": r.get("metadata", {}),
                                "distance": 1.0 - r.get("similarity", 0.0),
                            })
                        logger.debug("SQLite search returned %d results.", len(output))
                        return output
                except Exception:
                    logger.warning("SQLite vector search failed; falling back to ChromaDB.", exc_info=True)

        # --- ChromaDB fallback ---
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_params["where"] = where

        results = self._collection.query(**query_params)

        # ChromaDB returns lists-of-lists; unwrap the outer layer.
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: list[dict] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            output.append(
                {
                    "id": chunk_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )

        logger.debug("Search returned %d results.", len(output))
        return output

    async def delete_collection(self) -> None:
        """Delete the entire collection from both SQLite and ChromaDB."""
        logger.warning("Deleting collection '%s'.", self._collection_name)

        # --- SQLite ---
        if self._debate_id:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.vector import SQLiteVectorRepo
                    repo = SQLiteVectorRepo()
                    await repo.delete_collection(db, self._debate_id, self._pool)
                except Exception:
                    logger.warning("SQLite delete_collection failed.", exc_info=True)

        # --- ChromaDB ---
        self._client.delete_collection(name=self._collection_name)
        # Re-create so the instance remains usable after deletion.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' deleted and re-created (empty).", self._collection_name)

    async def count(self) -> int:
        """Return the number of stored chunks in the collection.

        Prefers SQLite count when available.
        """
        if self._debate_id:
            db = await _try_get_db()
            if db is not None:
                try:
                    from app.db.sqlite.vector import SQLiteVectorRepo
                    repo = SQLiteVectorRepo()
                    return await repo.count(db, self._debate_id, self._pool)
                except Exception:
                    logger.warning("SQLite count failed; falling back to ChromaDB.", exc_info=True)

        return self._collection.count()
