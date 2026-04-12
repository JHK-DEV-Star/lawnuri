"""
NetworkX wrapper for storing entity-relation graphs.

Provides a directed graph with SQLite persistence, used by the
GraphRAG pipeline to represent legal entities and their relationships.

The graph is kept in memory (NetworkX DiGraph) for O(1) lookups and
fast BFS traversal.  Mutations are tracked via a ``_dirty`` flag so
that ``save()`` only writes to SQLite when something actually changed.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Source → entity_type mapping for legal search results
# ---------------------------------------------------------------------------

_SOURCE_TYPE_MAP: dict[str, str] = {
    "court_precedent": "case",
    "constitutional_decision": "case",
    "admin_tribunal_decision": "case",
    "legal_statute": "law",
    "local_ordinance": "law",
    "administrative_rule": "law",
    "treaty": "law",
    "legal_interpretation": "interpretation",
}


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


class GraphStore:
    """Wrapper around ``networkx.DiGraph`` with SQLite persistence.

    The graph lives in memory for fast access.  Call :meth:`save` to
    flush accumulated changes to SQLite, and :meth:`load` to hydrate
    the in-memory graph from SQLite.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        debate_id: str = "",
        pool: str = "",
    ) -> None:
        self._graph = nx.DiGraph()
        self._persist_path = persist_path   # kept for indexer compat (no-op)
        self._debate_id = debate_id
        self._pool = pool
        self._dirty = False

        logger.info(
            "GraphStore initialized (debate=%s, pool=%s).", debate_id, pool,
        )

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        label: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> None:
        """Add or update a node (entity) in the graph."""
        attrs: dict[str, Any] = {
            "label": label,
            "entity_type": entity_type,
        }
        if properties:
            attrs.update(properties)

        self._graph.add_node(entity_id, **attrs)
        self._dirty = True
        logger.debug("Added entity '%s' (type=%s, label=%s).", entity_id, entity_type, label)

    def get_entity(self, entity_id: str) -> dict | None:
        """Retrieve an entity with all its properties, or *None*."""
        if entity_id not in self._graph:
            return None
        data = dict(self._graph.nodes[entity_id])
        data["entity_id"] = entity_id
        return data

    def has_entity(self, entity_id: str) -> bool:
        """Check if an entity exists in the graph."""
        return entity_id in self._graph

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict | None = None,
    ) -> None:
        """Add a directed edge (relation) between two entities."""
        if source_id not in self._graph:
            logger.warning("Source entity '%s' does not exist; it will be auto-created.", source_id)
        if target_id not in self._graph:
            logger.warning("Target entity '%s' does not exist; it will be auto-created.", target_id)

        attrs: dict[str, Any] = {"relation_type": relation_type}
        if properties:
            attrs.update(properties)

        self._graph.add_edge(source_id, target_id, **attrs)
        self._dirty = True
        logger.debug("Added relation %s -[%s]-> %s.", source_id, relation_type, target_id)

    # ------------------------------------------------------------------
    # Traversal / search
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: str | None = None,
        depth: int = 1,
    ) -> list[dict]:
        """Return neighboring entities via bidirectional BFS traversal."""
        if entity_id not in self._graph:
            logger.warning("get_neighbors: entity '%s' not found.", entity_id)
            return []

        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        results: list[dict] = []

        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            neighbors: list[tuple[str, dict]] = []
            for _, target, edge_data in self._graph.out_edges(current, data=True):
                neighbors.append((target, edge_data))
            for source, _, edge_data in self._graph.in_edges(current, data=True):
                neighbors.append((source, edge_data))

            for neighbor_id, edge_data in neighbors:
                if neighbor_id in visited:
                    continue
                if relation_type is not None and edge_data.get("relation_type") != relation_type:
                    continue
                visited.add(neighbor_id)
                entity = self.get_entity(neighbor_id)
                if entity is not None:
                    results.append(entity)
                queue.append((neighbor_id, current_depth + 1))

        logger.debug(
            "get_neighbors('%s', relation_type=%s, depth=%d) -> %d results.",
            entity_id, relation_type, depth, len(results),
        )
        return results

    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Search entities by label substring match (case-insensitive)."""
        query_lower = query.lower()
        results: list[dict] = []

        for node_id, data in self._graph.nodes(data=True):
            label: str = data.get("label", "")
            if query_lower not in label.lower():
                continue
            if entity_type is not None and data.get("entity_type") != entity_type:
                continue
            entry = dict(data)
            entry["entity_id"] = node_id
            results.append(entry)

        logger.debug(
            "search_entities(query='%s', entity_type=%s) -> %d results.",
            query, entity_type, len(results),
        )
        return results

    def get_subgraph(self, entity_ids: list[str]) -> dict:
        """Extract a subgraph containing only the specified entities."""
        id_set = set(entity_ids)
        nodes: list[dict] = []
        edges: list[dict] = []

        for node_id in entity_ids:
            if node_id not in self._graph:
                continue
            entry = dict(self._graph.nodes[node_id])
            entry["entity_id"] = node_id
            nodes.append(entry)

        for source, target, data in self._graph.edges(data=True):
            if source in id_set and target in id_set:
                edge_entry = dict(data)
                edge_entry["source"] = source
                edge_entry["target"] = target
                edges.append(edge_entry)

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Search-result ingestion (lightweight, no LLM calls)
    # ------------------------------------------------------------------

    def ingest_search_results(self, results: list[dict]) -> int:
        """Convert structured search results into graph entities/relations.

        Uses deterministic field mapping — no LLM calls.  Each result
        dict is expected to have some combination of ``case_number``,
        ``law_name``, ``title``, ``source``, ``content``, and nested
        ``metadata``.

        Returns the number of *new* entities added (existing ones are
        skipped).
        """
        added = 0
        # Collect all entity_ids created in this batch for inter-entity relations
        batch_entities: list[tuple[str, str]] = []  # (entity_id, entity_type)

        for r in results:
            meta = r.get("metadata", {})

            # --- Determine entity_id ---
            case_number = (
                r.get("case_number")
                or meta.get("case_number")
                or r.get("case_id")
                or meta.get("case_id")
                or ""
            )
            law_name = (
                r.get("law_name")
                or meta.get("law_name")
                or ""
            )
            item_id = r.get("_item_id") or r.get("item_id") or r.get("id") or meta.get("item_id") or ""

            entity_id = case_number or law_name or item_id
            if not entity_id:
                continue  # can't identify this result

            # Skip if already in graph
            if self.has_entity(entity_id):
                batch_entities.append((entity_id, self._graph.nodes[entity_id].get("entity_type", "")))
                continue

            # --- Determine entity_type ---
            source = r.get("source") or meta.get("source") or ""
            entity_type = _SOURCE_TYPE_MAP.get(source, "")
            if not entity_type:
                # Heuristic: case_number present → case, law_name present → law
                if case_number:
                    entity_type = "case"
                elif law_name:
                    entity_type = "law"
                else:
                    entity_type = "unknown"

            # --- Determine label ---
            title = r.get("title") or meta.get("title") or ""
            if entity_type == "case":
                label = case_number or title
            else:
                label = law_name or title or entity_id

            # --- Description (first 500 chars of content) ---
            content = r.get("content") or ""
            description = content[:500] if content else ""

            props: dict[str, Any] = {}
            if description:
                props["description"] = description
            if title and title != label:
                props["title"] = title
            if case_number and case_number != entity_id:
                props["case_number"] = case_number
            if law_name and law_name != entity_id:
                props["law_name"] = law_name

            self.add_entity(entity_id, label=label, entity_type=entity_type, properties=props)
            batch_entities.append((entity_id, entity_type))
            added += 1

        # --- Create inter-entity relations ---
        # If a result references both a case AND a law, link them.
        cases = [(eid, et) for eid, et in batch_entities if et == "case"]
        laws = [(eid, et) for eid, et in batch_entities if et == "law"]

        for case_id, _ in cases:
            for law_id, _ in laws:
                # Avoid duplicate edges
                if not self._graph.has_edge(case_id, law_id):
                    self.add_relation(case_id, law_id, "references")

        if added:
            logger.info(
                "GraphStore ingested %d new entities from %d search results.",
                added, len(results),
            )
        return added

    # ------------------------------------------------------------------
    # Persistence (SQLite only)
    # ------------------------------------------------------------------

    async def save(self, path: str | None = None) -> None:
        """Persist the graph to SQLite.

        Skips the write if nothing changed since the last save
        (``_dirty`` flag is False).
        """
        if not self._dirty:
            logger.debug("GraphStore.save() skipped — no changes.")
            return

        if not self._debate_id:
            logger.warning("GraphStore.save() skipped — no debate_id set.")
            return

        db = await _try_get_db()
        if db is None:
            logger.warning("GraphStore.save() failed — no DB connection.")
            return

        try:
            from app.db.sqlite.graph import SQLiteGraphRepo
            repo = SQLiteGraphRepo()
            data = self.to_dict()

            for node in data.get("nodes", []):
                props = {k: v for k, v in node.items()
                         if k not in ("entity_id", "label", "entity_type")}
                await repo.add_entity(
                    db,
                    self._debate_id,
                    self._pool,
                    entity_id=node["entity_id"],
                    label=node.get("label", ""),
                    entity_type=node.get("entity_type", ""),
                    properties=props or None,
                )

            for edge in data.get("edges", []):
                props = {k: v for k, v in edge.items()
                         if k not in ("source", "target", "relation_type")}
                await repo.add_relation(
                    db,
                    self._debate_id,
                    self._pool,
                    source_id=edge["source"],
                    target_id=edge["target"],
                    relation_type=edge.get("relation_type", ""),
                    properties=props or None,
                )

            self._dirty = False
            logger.info(
                "Graph saved to SQLite (debate=%s, pool=%s, %d nodes, %d edges).",
                self._debate_id, self._pool,
                self.count_entities(), self.count_relations(),
            )
        except Exception:
            logger.warning("SQLite graph save failed.", exc_info=True)

    async def load(self, path: str | None = None) -> None:
        """Load graph data from SQLite, replacing the current in-memory graph."""
        if not self._debate_id:
            logger.debug("GraphStore.load() skipped — no debate_id.")
            return

        db = await _try_get_db()
        if db is None:
            logger.debug("GraphStore.load() skipped — no DB connection.")
            return

        try:
            from app.db.sqlite.graph import SQLiteGraphRepo
            repo = SQLiteGraphRepo()
            data = await repo.to_dict(db, self._debate_id, self._pool)
            if data.get("nodes"):
                self._load_from_dict(data)
                self._dirty = False
                logger.info(
                    "Graph loaded from SQLite (debate=%s, pool=%s, %d nodes, %d edges).",
                    self._debate_id, self._pool,
                    self.count_entities(), self.count_relations(),
                )
        except Exception:
            logger.warning("SQLite graph load failed.", exc_info=True)

    def _load_from_dict(self, data: dict) -> None:
        """Replace the current graph from a nodes/edges dict."""
        self._graph = nx.DiGraph()

        for node in data.get("nodes", []):
            node_copy = dict(node)
            entity_id = node_copy.pop("entity_id")
            self._graph.add_node(entity_id, **node_copy)

        for edge in data.get("edges", []):
            edge_copy = dict(edge)
            source = edge_copy.pop("source")
            target = edge_copy.pop("target")
            self._graph.add_edge(source, target, **edge_copy)

    def to_dict(self) -> dict:
        """Convert the graph to a serializable dict."""
        nodes: list[dict] = []
        for node_id, data in self._graph.nodes(data=True):
            entry = dict(data)
            entry["entity_id"] = node_id
            nodes.append(entry)

        edges: list[dict] = []
        for source, target, data in self._graph.edges(data=True):
            entry = dict(data)
            entry["source"] = source
            entry["target"] = target
            edges.append(entry)

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def count_entities(self) -> int:
        """Return the number of entities (nodes) in the graph."""
        return self._graph.number_of_nodes()

    def count_relations(self) -> int:
        """Return the number of relations (edges) in the graph."""
        return self._graph.number_of_edges()
