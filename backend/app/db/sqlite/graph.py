"""
SQLite implementation of the knowledge-graph repository.

Entities (nodes) and relations (edges) are stored in dedicated tables
scoped by ``debate_id`` and ``pool``.  Neighbour traversal uses
iterative BFS with SQL queries at each depth level.
"""

from __future__ import annotations

import json
import uuid
from collections import deque

import aiosqlite

from app.db.base_repository import BaseGraphRepo
from app.utils.logger import logger


class SQLiteGraphRepo(BaseGraphRepo):
    """SQLite-backed entity-relation graph storage."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_properties(raw: str | None) -> dict:
        """Safely parse a JSON properties string, returning an empty dict on failure."""
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Public API -- mutations
    # ------------------------------------------------------------------

    async def add_entity(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        entity_id: str,
        label: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> None:
        """
        Insert or update a graph entity (node).

        If an entity with the same (debate_id, pool, entity_id) already
        exists, its label, type, and properties are replaced.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical graph partition (e.g. ``"shared"``).
            entity_id: Unique node identifier within the graph.
            label: Human-readable label for the entity.
            entity_type: Category (e.g. ``"person"``, ``"law"``, ``"case"``).
            properties: Optional extra attributes stored as JSON.
        """
        props_json = json.dumps(properties or {}, ensure_ascii=False)

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO graph_entities
                    (entity_id, debate_id, pool, label, entity_type, properties)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (entity_id, debate_id, pool, label, entity_type, props_json),
            )
            await db.commit()
            logger.debug(
                "Entity upserted: %s (type=%s, debate=%s, pool=%s).",
                entity_id, entity_type, debate_id, pool,
            )

        except Exception:
            logger.exception(
                "Failed to upsert entity '%s' (debate=%s, pool=%s).",
                entity_id, debate_id, pool,
            )
            raise

    async def add_relation(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict | None = None,
    ) -> None:
        """
        Insert a directed relation (edge) between two entities.

        Duplicate edges (same source, target, relation_type within the
        same debate + pool) are allowed; each insertion creates a new row.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical graph partition.
            source_id: Source entity ID.
            target_id: Target entity ID.
            relation_type: Label for the relationship (e.g. ``"cites"``).
            properties: Optional extra attributes stored as JSON.
        """
        props_json = json.dumps(properties or {}, ensure_ascii=False)

        try:
            relation_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO graph_relations
                    (relation_id, debate_id, pool, source_id, target_id,
                     relation_type, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (relation_id, debate_id, pool, source_id, target_id,
                 relation_type, props_json),
            )
            await db.commit()
            logger.debug(
                "Relation added: %s -[%s]-> %s (debate=%s, pool=%s).",
                source_id, relation_type, target_id, debate_id, pool,
            )

        except Exception:
            logger.exception(
                "Failed to add relation %s -> %s (debate=%s, pool=%s).",
                source_id, target_id, debate_id, pool,
            )
            raise

    # ------------------------------------------------------------------
    # Public API -- queries
    # ------------------------------------------------------------------

    async def get_neighbors(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        entity_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """
        BFS traversal returning neighbouring entities up to *depth* hops.

        Both outgoing and incoming edges are followed (bidirectional).
        The starting entity is **not** included in the results.

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical graph partition.
            entity_id: The starting node for BFS.
            depth: Maximum number of hops.

        Returns:
            A list of entity dicts (entity_id, label, entity_type,
            properties) reachable within *depth* hops.
        """
        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        results: list[dict] = []

        try:
            while queue:
                current, current_depth = queue.popleft()
                if current_depth >= depth:
                    continue

                # Find neighbours via outgoing edges.
                async with db.execute(
                    """
                    SELECT target_id FROM graph_relations
                    WHERE debate_id = ? AND pool = ? AND source_id = ?
                    """,
                    (debate_id, pool, current),
                ) as cursor:
                    outgoing = [row[0] for row in await cursor.fetchall()]

                # Find neighbours via incoming edges.
                async with db.execute(
                    """
                    SELECT source_id FROM graph_relations
                    WHERE debate_id = ? AND pool = ? AND target_id = ?
                    """,
                    (debate_id, pool, current),
                ) as cursor:
                    incoming = [row[0] for row in await cursor.fetchall()]

                neighbor_ids = set(outgoing + incoming) - visited

                for nid in neighbor_ids:
                    visited.add(nid)

                    # Fetch entity details.
                    async with db.execute(
                        """
                        SELECT entity_id, label, entity_type, properties
                        FROM graph_entities
                        WHERE debate_id = ? AND pool = ? AND entity_id = ?
                        """,
                        (debate_id, pool, nid),
                    ) as cursor:
                        erow = await cursor.fetchone()

                    if erow is not None:
                        results.append({
                            "entity_id": erow[0],
                            "label": erow[1],
                            "entity_type": erow[2],
                            "properties": self._parse_properties(erow[3]),
                        })

                    queue.append((nid, current_depth + 1))

            logger.debug(
                "get_neighbors('%s', depth=%d) -> %d results (debate=%s, pool=%s).",
                entity_id, depth, len(results), debate_id, pool,
            )
            return results

        except Exception:
            logger.exception(
                "get_neighbors failed for entity '%s' (debate=%s, pool=%s).",
                entity_id, debate_id, pool,
            )
            return []

    async def search_entities(
        self,
        db: aiosqlite.Connection,
        debate_id: str,
        pool: str,
        query: str,
    ) -> list[dict]:
        """
        Search entities by label substring (case-insensitive LIKE).

        Args:
            db: Active aiosqlite connection.
            debate_id: Owning debate identifier.
            pool: Logical graph partition.
            query: Substring to search for in entity labels.

        Returns:
            A list of matching entity dicts.
        """
        pattern = f"%{query}%"

        try:
            async with db.execute(
                """
                SELECT entity_id, label, entity_type, properties
                FROM graph_entities
                WHERE debate_id = ? AND pool = ? AND label LIKE ?
                """,
                (debate_id, pool, pattern),
            ) as cursor:
                rows = await cursor.fetchall()

            results: list[dict] = []
            for eid, label, etype, props_raw in rows:
                results.append({
                    "entity_id": eid,
                    "label": label,
                    "entity_type": etype,
                    "properties": self._parse_properties(props_raw),
                })

            logger.debug(
                "search_entities(query='%s') -> %d results (debate=%s, pool=%s).",
                query, len(results), debate_id, pool,
            )
            return results

        except Exception:
            logger.exception(
                "search_entities failed (query='%s', debate=%s, pool=%s).",
                query, debate_id, pool,
            )
            return []

    async def to_dict(
        self, db: aiosqlite.Connection, debate_id: str, pool: str
    ) -> dict:
        """
        Export the full graph as a dict for serialization or NetworkX import.

        Returns:
            ``{"nodes": [...], "edges": [...]}`` where each node has
            entity_id, label, entity_type, properties and each edge has
            source, target, relation_type, properties.
        """
        try:
            # Fetch all entities.
            async with db.execute(
                """
                SELECT entity_id, label, entity_type, properties
                FROM graph_entities
                WHERE debate_id = ? AND pool = ?
                """,
                (debate_id, pool),
            ) as cursor:
                entity_rows = await cursor.fetchall()

            nodes: list[dict] = []
            for eid, label, etype, props_raw in entity_rows:
                nodes.append({
                    "entity_id": eid,
                    "label": label,
                    "entity_type": etype,
                    "properties": self._parse_properties(props_raw),
                })

            # Fetch all relations.
            async with db.execute(
                """
                SELECT source_id, target_id, relation_type, properties
                FROM graph_relations
                WHERE debate_id = ? AND pool = ?
                """,
                (debate_id, pool),
            ) as cursor:
                relation_rows = await cursor.fetchall()

            edges: list[dict] = []
            for src, tgt, rtype, props_raw in relation_rows:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relation_type": rtype,
                    "properties": self._parse_properties(props_raw),
                })

            logger.debug(
                "to_dict: %d nodes, %d edges (debate=%s, pool=%s).",
                len(nodes), len(edges), debate_id, pool,
            )
            return {"nodes": nodes, "edges": edges}

        except Exception:
            logger.exception(
                "to_dict failed (debate=%s, pool=%s).", debate_id, pool,
            )
            return {"nodes": [], "edges": []}
