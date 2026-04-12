"""
LangGraph edge (routing) definitions for the LawNuri debate engine.

Edges determine the conditional transitions between nodes based on
the current debate state.
"""

from app.graph.edges.decide_next import decide_next
from app.graph.edges.should_continue import should_continue
from app.graph.edges.team_consensus import team_consensus

__all__ = [
    "decide_next",
    "should_continue",
    "team_consensus",
]
