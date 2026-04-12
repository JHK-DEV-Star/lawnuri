"""
Abstract base classes for all LawNuri repository interfaces.

Each base class defines the contract that concrete implementations
(SQLite, PostgreSQL, etc.) must fulfil. Business logic depends only
on these abstractions, keeping storage details swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class BaseSettingsRepo(ABC):
    """Persistence interface for application settings."""

    @abstractmethod
    async def load(self, db: Any) -> dict:
        """Load the full settings dictionary."""

    @abstractmethod
    async def save(self, db: Any, data: dict) -> None:
        """Persist the full settings dictionary."""

    @abstractmethod
    async def load_masked(self, db: Any) -> dict:
        """Load settings with sensitive values masked."""


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

class BaseDebateRepo(ABC):
    """Persistence interface for debate sessions and their LangGraph state."""

    @abstractmethod
    async def load(self, db: Any, debate_id: str) -> dict | None:
        """Load a single debate's full state by ID."""

    @abstractmethod
    async def save(self, db: Any, debate_id: str, state: dict) -> None:
        """Create or replace a debate's full state."""

    @abstractmethod
    async def list_all(self, db: Any) -> list[dict]:
        """Return summary dicts for every debate, newest first."""

    @abstractmethod
    async def update(self, db: Any, debate_id: str, **kwargs: Any) -> dict | None:
        """Merge *kwargs* into an existing debate's state and persist."""

    @abstractmethod
    async def delete(self, db: Any, debate_id: str) -> bool:
        """Delete a debate. Returns True if a row was actually removed."""


# ---------------------------------------------------------------------------
# Vector (embeddings)
# ---------------------------------------------------------------------------

class BaseVectorRepo(ABC):
    """Persistence interface for text-chunk embeddings."""

    @abstractmethod
    async def add_chunks(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Batch-insert text chunks with their embedding vectors."""

    @abstractmethod
    async def search(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Return the *top_k* most similar chunks for a query embedding."""

    @abstractmethod
    async def delete_collection(
        self, db: Any, debate_id: str, pool: str
    ) -> int:
        """Delete all chunks for a debate + pool. Returns rows removed."""

    @abstractmethod
    async def count(self, db: Any, debate_id: str, pool: str) -> int:
        """Return the number of stored chunks."""


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class BaseGraphRepo(ABC):
    """Persistence interface for entity-relation knowledge graphs."""

    @abstractmethod
    async def add_entity(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        entity_id: str,
        label: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> None:
        """Insert or update a graph entity (node)."""

    @abstractmethod
    async def add_relation(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict | None = None,
    ) -> None:
        """Insert a directed relation (edge) between two entities."""

    @abstractmethod
    async def get_neighbors(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        entity_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """BFS traversal returning neighbouring entities up to *depth* hops."""

    @abstractmethod
    async def search_entities(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        query: str,
    ) -> list[dict]:
        """Search entities by label substring (case-insensitive)."""

    @abstractmethod
    async def to_dict(
        self, db: Any, debate_id: str, pool: str
    ) -> dict:
        """Return ``{"nodes": [...], "edges": [...]}`` for the full graph."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class BaseCacheRepo(ABC):
    """Key-value cache scoped to a debate."""

    @abstractmethod
    async def get(self, db: Any, debate_id: str, cache_key: str) -> Any | None:
        """Retrieve a cached value, or *None* if missing."""

    @abstractmethod
    async def set(self, db: Any, debate_id: str, cache_key: str, data: Any) -> None:
        """Store a value in the cache (upsert)."""

    @abstractmethod
    async def delete_all(self, db: Any, debate_id: str) -> int:
        """Remove all cache entries for a debate. Returns rows removed."""


# ---------------------------------------------------------------------------
# Anonymization mapping
# ---------------------------------------------------------------------------

class BaseAnonymizationRepo(ABC):
    """Persistence for PII anonymization token mappings."""

    @abstractmethod
    async def load(self, db: Any, debate_id: str) -> dict:
        """Load the anonymization mapping for a debate."""

    @abstractmethod
    async def save(self, db: Any, debate_id: str, mapping: dict) -> None:
        """Replace the anonymization mapping for a debate."""

    @abstractmethod
    async def merge(self, db: Any, debate_id: str, new_mapping: dict) -> dict:
        """Merge *new_mapping* into the existing mapping and return the result."""


# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------

class BaseUploadRepo(ABC):
    """Persistence for uploaded file metadata."""

    @abstractmethod
    async def add(
        self,
        db: Any,
        debate_id: str,
        pool: str,
        filename: str,
        file_path: str,
        file_size: int,
    ) -> dict:
        """Record a new upload and return its metadata dict."""

    @abstractmethod
    async def list_by_debate(
        self, db: Any, debate_id: str, pool: str | None = None
    ) -> list[dict]:
        """List uploads for a debate, optionally filtered by pool."""

    @abstractmethod
    async def delete_by_debate(self, db: Any, debate_id: str) -> int:
        """Delete all upload records for a debate. Returns rows removed."""
