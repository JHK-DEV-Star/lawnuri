"""
Agent profile model for LawNuri.

Defines the structure for debate agents (debaters and judges),
including their personality, specialty, and LLM configuration.
"""

from typing import Literal, Optional
from pydantic import BaseModel


class AgentProfile(BaseModel):
    """Profile defining a debate participant or judge."""

    agent_id: str
    name: str
    role: Literal["debater", "judge"]
    team: Optional[Literal["team_a", "team_b"]] = None  # None for judges
    specialty: str = ""
    personality: str = ""
    debate_style: str = ""  # or judgment_style for judges
    background: str = ""
    llm_override: Optional[str] = None  # Override model for this agent
