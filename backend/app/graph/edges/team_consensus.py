"""
Conditional edge function for team internal discussion subgraph.

Checks whether the team has reached consensus or exhausted
the maximum allowed discussion turns, routing accordingly.
"""

from __future__ import annotations

_MAX_INTERNAL_TURNS = 3


def team_consensus(state: dict) -> str:
    """
    Conditional edge function for team subgraph internal discussion.

    Checks if the team has reached consensus by examining whether an
    agreed_strategy has been produced or the maximum discussion turns
    have been exhausted.

    Args:
        state: Current TeamState dict with internal_discussion and
            agreed_strategy fields.

    Returns:
        "search_evidence" if consensus reached or max turns exhausted,
        "internal_discuss" to continue discussion.
    """
    strategy = state.get("agreed_strategy", "")
    discussion = state.get("internal_discussion", [])
    max_turns = _MAX_INTERNAL_TURNS

    if strategy or len(discussion) >= max_turns:
        return "search_evidence"

    return "internal_discuss"
