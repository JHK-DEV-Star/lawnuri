"""
Conditional edge function for judge Q&A routing.

Routes to ``agent_answer`` if there are pending judge questions,
otherwise continues to ``route_next``.
"""

from __future__ import annotations


def decide_judge_qa(state: dict) -> str:
    """
    Route based on whether any judge questions are pending.

    Args:
        state: Current DebateState dict.

    Returns:
        ``"agent_answer"`` if questions remain, ``"route_next"`` otherwise.
    """
    pending = state.get("pending_judge_questions", [])
    if pending:
        return "agent_answer"
    return "route_next"
