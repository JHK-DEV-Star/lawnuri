"""
Evidence model for LawNuri.

Represents pieces of evidence submitted during a debate,
including uploaded documents, legal statutes, court precedents,
and knowledge graph relations.
"""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A single piece of evidence submitted during a debate round."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source_type: Literal[
        "uploaded_document",
        "legal_statute",
        "court_precedent",
        "graph_relation",
    ]
    source_detail: str  # e.g., "Patent Act Article 126", "document.pdf p.15"
    submitted_by: str  # team_a or team_b
    round: int
    speaker: str  # agent_id
    relevance: str = ""
