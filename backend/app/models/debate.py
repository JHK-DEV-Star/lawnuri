"""
Debate state models for LawNuri.

Defines the data structures for creating, analyzing, and tracking
the full lifecycle of a debate session.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from .agent import AgentProfile


class DebateCreate(BaseModel):
    """Request payload to create a new debate."""

    situation_brief: str
    default_model: str  # e.g., "gpt-4o-mini"


class DebateAnalysis(BaseModel):
    """AI-generated analysis of the debate topic."""

    topic: str
    opinion_a: str
    opinion_b: str
    key_issues: list[str]
    team_a_cautions: list[str] = []
    team_b_cautions: list[str] = []
    # Deep analysis fields
    parties: list[dict] = []
    timeline: list[dict] = []
    causal_chain: list[str] = []
    key_facts: list[dict] = []
    focus_points: dict = {}
    missing_information: list[str] = []


class DebateConfig(BaseModel):
    """Full debate configuration and runtime state."""

    debate_id: str
    situation_brief: str
    analysis: DebateAnalysis
    default_model: str
    agents: list[AgentProfile] = []
    status: Literal[
        "created",
        "analyzing",
        "ready",
        "running",
        "paused",
        "stopped",
        "completed",
        "extended",
    ]
    min_rounds: int = 3
    max_rounds: int = 10
    current_round: int = 0
    created_at: datetime
    updated_at: datetime
