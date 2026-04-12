"""
Unified search interface combining vector search (ChromaDB),
graph traversal (NetworkX), and external legal API results.

Provides a single entry point for retrieving relevant legal
documents and entities across multiple storage backends.
"""

from __future__ import annotations

import os

from app.rag.graph_store import GraphStore
from app.rag.vector_store import VectorStore
from app.utils.embedding_client import EmbeddingClient
from app.utils.logger import logger

# Valid pool names
_VALID_POOLS = {"common", "team_a", "team_b"}

# Pools accessible by each team
_TEAM_POOLS: dict[str, list[str]] = {
    "team_a": ["common", "team_a"],
    "team_b": ["common", "team_b"],
}


class Searcher:
    """
    Unified search across vector store, knowledge graph, and external APIs.

    Combines results from ChromaDB (vector similarity), NetworkX (graph
    traversal), and optionally external legal APIs into a single ranked
    result list with deduplication.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        data_dir: str,
    ) -> None:
        """
        Initialize the searcher.

        Args:
            embedding_client: EmbeddingClient instance for generating
                query embeddings during vector search.
            data_dir: Root data directory (e.g. "backend/data").
        """
        self._embedding_client = embedding_client
        self._data_dir = data_dir
        self._graph_cache: dict[tuple[str, str], GraphStore] = {}

        logger.info("Searcher initialized (data_dir=%s).", data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        debate_id: str,
        pool: str,
        search_type: str = "both",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform a unified search across configured backends.

        Args:
            query: The search query string.
            debate_id: Identifier for the debate whose data to search.
            pool: Document pool to search ("common", "team_a", "team_b").
            search_type: One of "vector", "graph", or "both".
                - "vector": embedding-based similarity search only.
                - "graph": entity label matching + neighbor traversal only.
                - "both": combine and deduplicate results from both.
            top_k: Maximum number of results to return.

        Returns:
            A list of result dicts, each containing:
                - content (str): The text content or entity description.
                - source (str): Origin identifier (document name, entity ID, etc.).
                - score (float): Relevance score (higher is better).
                - type (str): Result type ("vector" or "graph").
                - metadata (dict): Additional metadata.
        """
        if search_type == "vector":
            return await self.search_vector(query, debate_id, pool, top_k=top_k)

        if search_type == "graph":
            return await self.search_graph(query, debate_id, pool)

        # "both" - combine vector and graph results
        vector_results = await self.search_vector(
            query, debate_id, pool, top_k=top_k
        )
        graph_results = await self.search_graph(query, debate_id, pool)

        combined = self._merge_and_deduplicate(vector_results, graph_results)

        # Sort by score descending and limit to top_k
        combined.sort(key=lambda r: r["score"], reverse=True)
        return combined[:top_k]

    async def search_vector(
        self,
        query: str,
        debate_id: str,
        pool: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform vector-only search using ChromaDB.

        Embeds the query, then finds the closest stored chunks by
        cosine similarity.

        Args:
            query: The search query string.
            debate_id: Identifier for the debate.
            pool: Document pool to search.
            top_k: Maximum number of results.

        Returns:
            List of result dicts with type="vector".
        """
        try:
            vector_store = self._get_vector_store(debate_id, pool)
        except Exception as exc:
            logger.warning(
                "Could not open vector store for debate=%s pool=%s: %s",
                debate_id, pool, exc,
            )
            return []

        # Embed the query
        try:
            query_embeddings = await self._embedding_client.aembed(query)
            query_vector = query_embeddings[0]
        except Exception as exc:
            logger.warning("Failed to embed query: %s", exc)
            return []

        # Search ChromaDB
        try:
            raw_results = await vector_store.search(
                query_embedding=query_vector,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

        results: list[dict] = []
        for item in raw_results:
            # Convert cosine distance to a similarity score (0..1)
            distance = item.get("distance", 1.0)
            score = max(0.0, 1.0 - distance)

            metadata = item.get("metadata", {})
            results.append({
                "content": item.get("document", ""),
                "source": metadata.get("doc_name", "unknown"),
                "score": score,
                "type": "vector",
                "metadata": metadata,
            })

        logger.info(
            "Vector search for '%s' (debate=%s, pool=%s) returned %d results.",
            query, debate_id, pool, len(results),
        )
        return results

    async def search_graph(
        self,
        query: str,
        debate_id: str,
        pool: str,
    ) -> list[dict]:
        """
        Perform graph-only search using NetworkX.

        Searches for entities whose labels match the query, then
        retrieves their neighbors to build a context of related
        entities and relations.

        Args:
            query: The search query string.
            debate_id: Identifier for the debate.
            pool: Document pool to search.

        Returns:
            List of result dicts with type="graph".
        """
        try:
            graph_store = await self._get_graph_store(debate_id, pool)
        except Exception as exc:
            logger.warning(
                "Could not open graph store for debate=%s pool=%s: %s",
                debate_id, pool, exc,
            )
            return []

        # Find entities matching the query by label
        matching_entities = graph_store.search_entities(query)

        results: list[dict] = []
        seen_ids: set[str] = set()

        for entity in matching_entities:
            entity_id = entity.get("entity_id", "")
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)

            # Build a content string from entity properties
            label = entity.get("label", "")
            entity_type = entity.get("entity_type", "")
            description = entity.get("description", "")
            content = f"[{entity_type}] {label}"
            if description:
                content += f": {description}"

            results.append({
                "content": content,
                "source": f"graph:{entity_id}",
                "score": 0.8,  # Direct match gets a high base score
                "type": "graph",
                "metadata": {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "label": label,
                    "match_type": "direct",
                },
            })

            # Get neighbors (1-hop) for additional context
            neighbors = graph_store.get_neighbors(entity_id, depth=1)
            for neighbor in neighbors:
                neighbor_id = neighbor.get("entity_id", "")
                if neighbor_id in seen_ids:
                    continue
                seen_ids.add(neighbor_id)

                n_label = neighbor.get("label", "")
                n_type = neighbor.get("entity_type", "")
                n_desc = neighbor.get("description", "")
                n_content = f"[{n_type}] {n_label}"
                if n_desc:
                    n_content += f": {n_desc}"

                results.append({
                    "content": n_content,
                    "source": f"graph:{neighbor_id}",
                    "score": 0.5,  # Neighbor match gets a lower score
                    "type": "graph",
                    "metadata": {
                        "entity_id": neighbor_id,
                        "entity_type": n_type,
                        "label": n_label,
                        "match_type": "neighbor",
                        "related_to": entity_id,
                    },
                })

        logger.info(
            "Graph search for '%s' (debate=%s, pool=%s) returned %d results.",
            query, debate_id, pool, len(results),
        )
        return results

    async def search_all_pools(
        self,
        query: str,
        debate_id: str,
        team: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search all pools accessible to a given team.

        Access rules:
            - team_a can access: common + team_a
            - team_b can access: common + team_b

        Results from all accessible pools are merged, deduplicated,
        and ranked by relevance score.

        Args:
            query: The search query string.
            debate_id: Identifier for the debate.
            team: Team identifier ("team_a" or "team_b").
            top_k: Maximum number of results.

        Returns:
            Merged and deduplicated list of result dicts.
        """
        pools = _TEAM_POOLS.get(team, ["common"])
        logger.info(
            "Searching all pools for team '%s': %s", team, pools
        )

        all_results: list[dict] = []

        for pool in pools:
            pool_results = await self.search(
                query=query,
                debate_id=debate_id,
                pool=pool,
                search_type="both",
                top_k=top_k,
            )
            all_results.extend(pool_results)

        # Deduplicate and rank
        deduplicated = self._deduplicate(all_results)
        deduplicated.sort(key=lambda r: r["score"], reverse=True)

        logger.info(
            "search_all_pools for '%s' (team=%s) returned %d results "
            "(from %d total before dedup).",
            query, team, min(len(deduplicated), top_k), len(all_results),
        )
        return deduplicated[:top_k]

    # ------------------------------------------------------------------
    # Store accessors
    # ------------------------------------------------------------------

    def _get_vector_store(self, debate_id: str, pool: str) -> VectorStore:
        """
        Open (or create) a VectorStore for the specified debate and pool.

        Args:
            debate_id: Debate identifier.
            pool: Pool name (common, team_a, team_b).

        Returns:
            A VectorStore instance backed by the on-disk ChromaDB data.
        """
        persist_dir = os.path.join(
            self._data_dir, "debates", debate_id, "rag", pool, "chroma"
        )
        return VectorStore(persist_dir=persist_dir, collection_name=pool, debate_id=debate_id, pool=pool)

    async def _get_graph_store(self, debate_id: str, pool: str) -> GraphStore:
        """
        Return a cached GraphStore for ``(debate_id, pool)``.

        On the first call for a given key the store is loaded from SQLite
        and then kept in memory for the rest of the debate.  Subsequent
        calls return the same instance — this allows the graph to
        accumulate new entities across rounds.
        """
        key = (debate_id, pool)
        if key in self._graph_cache:
            return self._graph_cache[key]

        store = GraphStore(debate_id=debate_id, pool=pool)
        await store.load()
        self._graph_cache[key] = store
        logger.info(
            "GraphStore cached for (%s, %s) — %d nodes, %d edges.",
            debate_id, pool, store.count_entities(), store.count_relations(),
        )
        return store

    # ------------------------------------------------------------------
    # Public graph access (for search_node ingestion)
    # ------------------------------------------------------------------

    async def get_graph_store(self, debate_id: str, pool: str) -> GraphStore:
        """Public accessor — delegates to the cached ``_get_graph_store``."""
        return await self._get_graph_store(debate_id, pool)

    async def save_graph(self, debate_id: str, pool: str) -> None:
        """Persist a specific pool's graph to SQLite (if dirty)."""
        key = (debate_id, pool)
        store = self._graph_cache.get(key)
        if store is not None:
            await store.save()

    async def save_all_graphs(self) -> None:
        """Persist all cached graphs to SQLite."""
        for store in self._graph_cache.values():
            await store.save()

    # ------------------------------------------------------------------
    # Deduplication and merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_deduplicate(
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> list[dict]:
        """
        Merge results from vector and graph search, removing duplicates.

        Deduplication is based on content similarity: if two results
        share the same content text (stripped and lowered), only the
        one with the higher score is kept.

        Args:
            vector_results: Results from vector search.
            graph_results: Results from graph search.

        Returns:
            Merged and deduplicated list of results.
        """
        seen: dict[str, dict] = {}

        for result in vector_results + graph_results:
            key = result["content"].strip().lower()[:200]
            existing = seen.get(key)
            if existing is None or result["score"] > existing["score"]:
                seen[key] = result

        return list(seen.values())

    @staticmethod
    def _deduplicate(results: list[dict]) -> list[dict]:
        """
        Remove duplicate results based on content text.

        Keeps the highest-scoring entry when duplicates are found.

        Args:
            results: List of result dicts.

        Returns:
            Deduplicated list.
        """
        seen: dict[str, dict] = {}

        for result in results:
            key = result["content"].strip().lower()[:200]
            existing = seen.get(key)
            if existing is None or result["score"] > existing["score"]:
                seen[key] = result

        return list(seen.values())
