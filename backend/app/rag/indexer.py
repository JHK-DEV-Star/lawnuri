"""
Main indexing pipeline for the GraphRAG system.

Orchestrates the full document processing flow:
    document -> anonymize -> chunk -> embed -> store vectors ->
    extract entities -> extract relations -> store graph.

Supports per-debate, per-pool (common / team_a / team_b) storage with
progress tracking via TaskManager.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from app.models.task import TaskManager, TaskStatus
from app.rag.anonymizer import Anonymizer
from app.rag.entity_extractor import EntityExtractor
from app.rag.graph_store import GraphStore
from app.rag.relation_extractor import RelationExtractor
from app.rag.vector_store import VectorStore
from app.utils.embedding_client import EmbeddingClient
from app.utils.file_parser import FileParser, split_text_into_chunks
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Valid pool names
_VALID_POOLS = {"common", "team_a", "team_b"}


class Indexer:
    """
    End-to-end indexing pipeline for legal documents.

    Each indexed document goes through parsing, anonymization, chunking,
    embedding, vector storage, entity extraction, relation extraction,
    and graph storage.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        data_dir: str,
    ) -> None:
        """
        Initialize the indexer with required clients and data directory.

        Args:
            llm_client: LLMClient instance for entity/relation extraction.
            embedding_client: EmbeddingClient instance for generating embeddings.
            data_dir: Root data directory (e.g. "backend/data").
        """
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self._data_dir = data_dir

        self._anonymizer = Anonymizer()
        self._entity_extractor = EntityExtractor(llm_client)
        self._relation_extractor = RelationExtractor(llm_client)
        self._task_manager = TaskManager()

        logger.info("Indexer initialized (data_dir=%s).", data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def index_document(
        self,
        file_path: str,
        pool: str,
        debate_id: str,
        task_id: str | None = None,
    ) -> dict:
        """
        Run the full indexing pipeline for a single document.

        Args:
            file_path: Path to the source document file.
            pool: Target pool — one of "common", "team_a", "team_b".
            debate_id: Unique identifier for the debate this document belongs to.
            task_id: Optional task ID for progress tracking via TaskManager.

        Returns:
            Summary dict with counts:
                - file: original file path
                - chunks: number of chunks created
                - entities: number of unique entities extracted
                - relations: number of relations extracted
        """
        if pool not in _VALID_POOLS:
            raise ValueError(
                f"Invalid pool '{pool}'. Must be one of: {_VALID_POOLS}"
            )

        file_name = Path(file_path).name
        doc_id = str(uuid.uuid4())
        logger.info(
            "[Indexer] Starting indexing: file=%s, pool=%s, debate=%s",
            file_name,
            pool,
            debate_id,
        )

        self._update_progress(task_id, 0.0, f"Parsing file: {file_name}")

        # Step 1: Parse the document
        raw_text = FileParser.extract_text(file_path)
        logger.info("[Indexer] Parsed %d characters from %s.", len(raw_text), file_name)

        self._update_progress(task_id, 0.1, f"Anonymizing text: {file_name}")

        # Step 2: Anonymize text
        anonymized_text, mapping = self._anonymizer.anonymize(raw_text)
        logger.info(
            "[Indexer] Anonymization complete: %d mappings created.", len(mapping)
        )

        # Step 3: Save anonymization mapping (merged with existing if any)
        mapping_path = self._get_mapping_path(debate_id)
        await self._save_mapping(mapping, mapping_path, debate_id)

        self._update_progress(task_id, 0.2, f"Splitting into chunks: {file_name}")

        # Step 4: Split into chunks
        chunks = split_text_into_chunks(anonymized_text)
        if not chunks:
            logger.warning("[Indexer] No chunks produced from %s.", file_name)
            self._update_progress(task_id, 1.0, "No content to index.")
            return {
                "file": file_path,
                "chunks": 0,
                "entities": 0,
                "relations": 0,
            }

        logger.info("[Indexer] Split text into %d chunks.", len(chunks))

        self._update_progress(task_id, 0.3, f"Generating embeddings: {file_name}")

        # Step 5: Generate embeddings for all chunks
        embeddings = await self._embedding_client.aembed(chunks)
        logger.info("[Indexer] Generated %d embedding vectors.", len(embeddings))

        self._update_progress(task_id, 0.5, f"Storing vectors: {file_name}")

        # Step 6: Store chunks + embeddings in ChromaDB
        vector_dir = self._get_vector_store_dir(debate_id, pool)
        vector_store = VectorStore(persist_dir=vector_dir, collection_name=pool, debate_id=debate_id, pool=pool)

        metadatas = [
            {
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_name": file_name,
                "chunk_index": i,
                "pool": pool,
                "debate_id": debate_id,
            }
            for i in range(len(chunks))
        ]
        await vector_store.add_chunks(
            chunks=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("[Indexer] Stored %d chunks in VectorStore.", len(chunks))

        self._update_progress(task_id, 0.6, f"Extracting entities: {file_name}")

        # Step 7: Extract entities from chunks
        all_entities = await self._entity_extractor.extract_batch(chunks)
        logger.info("[Indexer] Extracted %d unique entities.", len(all_entities))

        # Also extract per-chunk entities for relation extraction
        chunk_entities: list[list[dict]] = []
        for chunk in chunks:
            ents = await self._entity_extractor.extract(chunk)
            chunk_entities.append(ents)

        self._update_progress(task_id, 0.8, f"Extracting relations: {file_name}")

        # Step 8: Extract relations between entities
        all_relations = await self._relation_extractor.extract_batch(
            chunks, chunk_entities
        )
        logger.info("[Indexer] Extracted %d relations.", len(all_relations))

        self._update_progress(task_id, 0.9, f"Building knowledge graph: {file_name}")

        # Step 9: Store entities and relations in the graph
        graph_store = GraphStore(debate_id=debate_id, pool=pool)

        # Build a label -> entity_id lookup for linking relations
        label_to_id: dict[str, str] = {}
        for entity in all_entities:
            graph_store.add_entity(
                entity_id=entity["entity_id"],
                label=entity["label"],
                entity_type=entity["type"],
                properties={"description": entity.get("description", "")},
            )
            label_to_id[entity["label"].lower()] = entity["entity_id"]

        for relation in all_relations:
            source_id = label_to_id.get(relation["source"].lower())
            target_id = label_to_id.get(relation["target"].lower())
            if source_id and target_id:
                graph_store.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation["relation_type"],
                    properties={"description": relation.get("description", "")},
                )

        # Step 10: Persist graph to SQLite
        await graph_store.save()
        logger.info("[Indexer] Graph saved to SQLite (debate=%s, pool=%s).", debate_id, pool)

        self._update_progress(task_id, 1.0, f"Indexing complete: {file_name}")

        summary = {
            "file": file_path,
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relations": len(all_relations),
        }
        logger.info("[Indexer] Document indexed: %s", summary)
        return summary

    async def index_documents(
        self,
        file_paths: list[str],
        pool: str,
        debate_id: str,
        task_id: str | None = None,
    ) -> dict:
        """
        Index multiple documents sequentially.

        Args:
            file_paths: List of file paths to index.
            pool: Target pool — one of "common", "team_a", "team_b".
            debate_id: Unique identifier for the debate.
            task_id: Optional task ID for progress tracking.

        Returns:
            Aggregated summary dict:
                - files: number of files processed
                - total_chunks: total chunks across all files
                - total_entities: total unique entities
                - total_relations: total relations
                - details: list of per-file summary dicts
        """
        total_chunks = 0
        total_entities = 0
        total_relations = 0
        details: list[dict] = []

        for idx, fp in enumerate(file_paths):
            logger.info(
                "[Indexer] Indexing file %d/%d: %s", idx + 1, len(file_paths), fp
            )

            # Calculate sub-progress within the overall task
            if task_id:
                base_progress = idx / len(file_paths)
                self._update_progress(
                    task_id,
                    base_progress,
                    f"Indexing file {idx + 1}/{len(file_paths)}: {Path(fp).name}",
                )

            try:
                result = await self.index_document(
                    file_path=fp,
                    pool=pool,
                    debate_id=debate_id,
                    # Do not pass task_id to individual calls to avoid
                    # conflicting progress updates; we manage it here.
                    task_id=None,
                )
                total_chunks += result["chunks"]
                total_entities += result["entities"]
                total_relations += result["relations"]
                details.append(result)
            except Exception:
                logger.exception("[Indexer] Failed to index file: %s", fp)
                details.append({"file": fp, "error": "indexing failed"})

        if task_id:
            self._update_progress(task_id, 1.0, "All documents indexed.")

        summary = {
            "files": len(file_paths),
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "details": details,
        }
        logger.info("[Indexer] Batch indexing complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _get_vector_store_dir(self, debate_id: str, pool: str) -> str:
        """
        Return the ChromaDB persist directory for a given debate and pool.

        Args:
            debate_id: The debate identifier.
            pool: The pool name (common, team_a, team_b).

        Returns:
            Absolute path string to the chroma directory.
        """
        return os.path.join(
            self._data_dir, "debates", debate_id, "rag", pool, "chroma"
        )

    def _get_graph_store_path(self, debate_id: str, pool: str) -> str:
        """
        Return the graph JSON file path for a given debate and pool.

        Args:
            debate_id: The debate identifier.
            pool: The pool name (common, team_a, team_b).

        Returns:
            Absolute path string to the graph.json file.
        """
        return os.path.join(
            self._data_dir, "debates", debate_id, "rag", pool, "graph.json"
        )

    def _get_mapping_path(self, debate_id: str) -> str:
        """
        Return the anonymization mapping file path for a debate.

        Args:
            debate_id: The debate identifier.

        Returns:
            Absolute path string to anonymization_map.json.
        """
        return os.path.join(
            self._data_dir, "debates", debate_id, "anonymization_map.json"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_progress(
        self,
        task_id: str | None,
        progress: float,
        message: str,
    ) -> None:
        """
        Update task progress if a task_id is provided.

        Args:
            task_id: Optional task identifier. No-op if None.
            progress: Progress value between 0.0 and 1.0.
            message: Human-readable status message.
        """
        if task_id is None:
            return

        status = TaskStatus.COMPLETED if progress >= 1.0 else TaskStatus.PROCESSING
        self._task_manager.update_task(
            task_id,
            progress=progress,
            message=message,
            status=status,
        )
        logger.debug(
            "[Indexer] Task %s progress: %.0f%% — %s",
            task_id,
            progress * 100,
            message,
        )

    async def _save_mapping(self, mapping: dict[str, str], path: str, debate_id: str = "") -> None:
        """
        Save an anonymization mapping, merging with any existing data.

        If a mapping already exists (in SQLite or file), the new entries
        are merged into it (existing keys are preserved).

        Args:
            mapping: New token-to-original mapping entries.
            path: Destination file path.
            debate_id: Debate identifier for SQLite scoping.
        """
        existing: dict[str, str] = {}
        try:
            existing = await self._anonymizer.load_mapping(path, debate_id=debate_id)
        except Exception:
            logger.warning("[Indexer] Could not load existing mapping at %s.", path)

        merged = {**existing, **mapping}
        await self._anonymizer.save_mapping(merged, path, debate_id=debate_id)
        logger.info(
            "[Indexer] Anonymization mapping saved: %d entries (%d new).",
            len(merged),
            len(mapping),
        )
