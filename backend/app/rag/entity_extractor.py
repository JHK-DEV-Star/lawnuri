"""
LLM-based entity extraction for legal texts.

Uses an LLM to identify named entities (persons, organizations, laws,
cases, dates, locations, concepts, documents) from anonymized text chunks.
Supports single-chunk and batch extraction with deduplication.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Valid entity types recognized by the extractor
ENTITY_TYPES = (
    "person",
    "organization",
    "law",
    "case",
    "date",
    "location",
    "concept",
    "document",
)

# System prompt instructing the LLM how to extract legal entities
_SYSTEM_PROMPT = (
    "You are a legal entity extraction assistant. "
    "Given a text chunk from a legal document, identify and extract all named entities.\n\n"
    "For each entity, return:\n"
    '- "label": the entity name as it appears in the text\n'
    '- "type": one of: "person", "organization", "law", "case", '
    '"date", "location", "concept", "document"\n'
    '- "description": a brief one-sentence description of the entity '
    "based on the context\n\n"
    "Rules:\n"
    "- Extract every distinct entity mentioned in the text.\n"
    "- Use the most specific type that applies.\n"
    '- For anonymized placeholders like Person_A, Company_B, use type "person" '
    'or "organization" respectively.\n'
    '- For legal concepts (e.g. "duty of care"), use type "concept".\n'
    '- For specific statutes or regulations, use type "law".\n'
    '- For case references or case numbers, use type "case".\n\n'
    "Return a JSON array of objects. If no entities are found, return an empty array [].\n"
    "Output ONLY valid JSON, no extra text."
)


class EntityExtractor:
    """Extract named entities from text using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Initialize the entity extractor.

        Args:
            llm_client: An LLMClient instance used for LLM inference.
        """
        self._llm = llm_client
        logger.info("EntityExtractor initialized.")

    async def extract(self, text: str) -> list[dict]:
        """
        Extract entities from a single text chunk.

        Args:
            text: The text to extract entities from.

        Returns:
            A list of entity dicts, each containing:
                - entity_id: Unique identifier (UUID).
                - label: Entity name as found in text.
                - type: Entity category string.
                - description: Brief contextual description.
        """
        if not text or not text.strip():
            logger.debug("[EntityExtractor] Empty text provided; returning no entities.")
            return []

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            raw: Any = await self._llm.achat_json(messages, temperature=0.1)
        except Exception:
            logger.exception("[EntityExtractor] LLM call failed during entity extraction.")
            return []

        entities = self._parse_response(raw)
        logger.info("[EntityExtractor] Extracted %d entities from chunk.", len(entities))
        return entities

    async def extract_batch(self, chunks: list[str]) -> list[dict]:
        """
        Extract entities from multiple text chunks with deduplication.

        Entities are deduplicated by label (case-insensitive). When duplicates
        are found the first occurrence is kept.

        Args:
            chunks: List of text chunks to process.

        Returns:
            A deduplicated list of entity dicts.
        """
        if not chunks:
            return []

        all_entities: list[dict] = []
        for idx, chunk in enumerate(chunks):
            logger.debug(
                "[EntityExtractor] Processing chunk %d/%d (%d chars).",
                idx + 1,
                len(chunks),
                len(chunk),
            )
            entities = await self.extract(chunk)
            all_entities.extend(entities)

        deduplicated = self._deduplicate(all_entities)
        logger.info(
            "[EntityExtractor] Batch extraction complete: %d total -> %d unique entities.",
            len(all_entities),
            len(deduplicated),
        )
        return deduplicated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: Any) -> list[dict]:
        """
        Validate and normalize the raw LLM JSON response into entity dicts.

        Args:
            raw: Parsed JSON from the LLM (expected to be a list of dicts).

        Returns:
            List of well-formed entity dicts with generated entity_id values.
        """
        if not isinstance(raw, list):
            logger.warning(
                "[EntityExtractor] Expected a JSON array but got %s; wrapping.",
                type(raw).__name__,
            )
            raw = [raw] if isinstance(raw, dict) else []

        entities: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            label = str(item.get("label", "")).strip()
            entity_type = str(item.get("type", "")).strip().lower()
            description = str(item.get("description", "")).strip()

            if not label:
                continue

            # Normalize unknown types to "concept"
            if entity_type not in ENTITY_TYPES:
                entity_type = "concept"

            entities.append(
                {
                    "entity_id": str(uuid.uuid4()),
                    "label": label,
                    "type": entity_type,
                    "description": description,
                }
            )

        return entities

    @staticmethod
    def _deduplicate(entities: list[dict]) -> list[dict]:
        """
        Remove duplicate entities by label (case-insensitive).

        Keeps the first occurrence of each unique label.

        Args:
            entities: List of entity dicts potentially containing duplicates.

        Returns:
            Deduplicated list preserving original order.
        """
        seen: set[str] = set()
        unique: list[dict] = []

        for entity in entities:
            key = entity["label"].lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)

        return unique
