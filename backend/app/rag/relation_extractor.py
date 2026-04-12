"""
LLM-based relation extraction for legal texts.

Uses an LLM to identify directed relationships between known entities
within a text chunk. Designed to work in tandem with EntityExtractor.
"""

from __future__ import annotations

from typing import Any

from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Valid relation types between entities
RELATION_TYPES = (
    "filed_against",
    "represented_by",
    "violates",
    "cites",
    "ruled_by",
    "related_to",
    "part_of",
    "occurred_at",
    "contracted_with",
)

# System prompt instructing the LLM how to extract relations
_SYSTEM_PROMPT = (
    "You are a legal relation extraction assistant. "
    "Given a text chunk and a list of known entities found in that chunk, "
    "identify all meaningful relationships between them.\n\n"
    "For each relationship, return:\n"
    '- "source": the label of the source entity (must match one of the provided entities)\n'
    '- "target": the label of the target entity (must match one of the provided entities)\n'
    '- "relation_type": one of: "filed_against", "represented_by", "violates", '
    '"cites", "ruled_by", "related_to", "part_of", "occurred_at", "contracted_with"\n'
    '- "description": a brief one-sentence description of the relationship\n\n'
    "Rules:\n"
    "- Only create relations between entities in the provided list.\n"
    "- Source and target must be different entities.\n"
    "- Use the most specific relation type that applies.\n"
    '- Use "related_to" only when no more specific type fits.\n'
    "- A single pair of entities may have multiple different relation types.\n\n"
    "Return a JSON array of objects. If no relationships are found, return an empty array [].\n"
    "Output ONLY valid JSON, no extra text."
)


def _build_user_message(text: str, entities: list[dict]) -> str:
    """
    Build the user message combining the text and entity list.

    Args:
        text: The source text chunk.
        entities: List of entity dicts (must contain at least "label" and "type").

    Returns:
        Formatted user prompt string.
    """
    entity_lines = "\n".join(
        f'- {e.get("label", "?")} (type: {e.get("type", "unknown")})'
        for e in entities
    )
    return (
        f"TEXT:\n{text}\n\n"
        f"KNOWN ENTITIES:\n{entity_lines}\n\n"
        "Extract all relationships between the entities listed above."
    )


class RelationExtractor:
    """Extract directed relations between entities from text using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Initialize the relation extractor.

        Args:
            llm_client: An LLMClient instance used for LLM inference.
        """
        self._llm = llm_client
        logger.info("RelationExtractor initialized.")

    async def extract(self, text: str, entities: list[dict]) -> list[dict]:
        """
        Extract relations from a text chunk given known entities.

        Args:
            text: The text chunk to analyze.
            entities: List of entity dicts present in this chunk. Each dict
                      should contain at least "label" and "type" keys.

        Returns:
            A list of relation dicts, each containing:
                - source: Label of the source entity.
                - target: Label of the target entity.
                - relation_type: Category of the relationship.
                - description: Brief contextual description.
        """
        if not text or not text.strip() or len(entities) < 2:
            logger.debug(
                "[RelationExtractor] Skipping extraction: "
                "text empty or fewer than 2 entities."
            )
            return []

        user_message = _build_user_message(text, entities)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            raw: Any = await self._llm.achat_json(messages, temperature=0.1)
        except Exception:
            logger.exception("[RelationExtractor] LLM call failed during relation extraction.")
            return []

        # Build a set of known entity labels for validation
        known_labels = {e.get("label", "").lower() for e in entities}

        relations = self._parse_response(raw, known_labels)
        logger.info(
            "[RelationExtractor] Extracted %d relations from chunk.", len(relations)
        )
        return relations

    async def extract_batch(
        self,
        chunks: list[str],
        chunk_entities: list[list[dict]],
    ) -> list[dict]:
        """
        Extract relations from multiple text chunks.

        Args:
            chunks: List of text chunks.
            chunk_entities: Parallel list of entity lists — one per chunk.
                            chunk_entities[i] contains the entities found in chunks[i].

        Returns:
            Aggregated list of all relation dicts across all chunks.
        """
        if not chunks or not chunk_entities:
            return []

        if len(chunks) != len(chunk_entities):
            logger.error(
                "[RelationExtractor] Mismatch: %d chunks vs %d entity lists.",
                len(chunks),
                len(chunk_entities),
            )
            return []

        all_relations: list[dict] = []
        for idx, (chunk, entities) in enumerate(zip(chunks, chunk_entities)):
            logger.debug(
                "[RelationExtractor] Processing chunk %d/%d (%d entities).",
                idx + 1,
                len(chunks),
                len(entities),
            )
            relations = await self.extract(chunk, entities)
            all_relations.extend(relations)

        logger.info(
            "[RelationExtractor] Batch extraction complete: %d total relations.",
            len(all_relations),
        )
        return all_relations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: Any, known_labels: set[str]) -> list[dict]:
        """
        Validate and normalize the raw LLM JSON response into relation dicts.

        Only relations whose source and target both appear in *known_labels*
        are retained.

        Args:
            raw: Parsed JSON from the LLM (expected to be a list of dicts).
            known_labels: Set of lowercase entity labels for validation.

        Returns:
            List of well-formed relation dicts.
        """
        if not isinstance(raw, list):
            logger.warning(
                "[RelationExtractor] Expected a JSON array but got %s; wrapping.",
                type(raw).__name__,
            )
            raw = [raw] if isinstance(raw, dict) else []

        relations: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation_type = str(item.get("relation_type", "")).strip().lower()
            description = str(item.get("description", "")).strip()

            if not source or not target:
                continue

            # Ensure source and target are different
            if source.lower() == target.lower():
                continue

            # Validate that both endpoints are known entities
            if source.lower() not in known_labels or target.lower() not in known_labels:
                logger.debug(
                    "[RelationExtractor] Dropping relation '%s' -> '%s': "
                    "entity not in known set.",
                    source,
                    target,
                )
                continue

            # Normalize unknown relation types to "related_to"
            if relation_type not in RELATION_TYPES:
                relation_type = "related_to"

            relations.append(
                {
                    "source": source,
                    "target": target,
                    "relation_type": relation_type,
                    "description": description,
                }
            )

        return relations
