"""
Conditional edge function for round-end continuation decision.

After a round ends, determines whether the debate should continue
to the next round, proceed to final judgment, or pause.
"""

from __future__ import annotations


def should_continue(state: dict) -> str:
    """
    Conditional edge function after the round_end node.

    Routes based on state["next_action"]:
      - "continue" -> "user_interrupt" (start next round)
      - "stop"     -> "final_judgment"
      - "paused"   -> "__end__" (will resume later via checkpoint)

    Args:
        state: Current DebateState dict after round_end processing.

    Returns:
        Target node name string for the conditional edge.
        Returns "__end__" for the LangGraph END sentinel.
    """
    action = state.get("next_action", "continue")

    if action == "stop":
        return "final_judgment"
    elif action == "paused":
        return "__end__"

    # Default: continue to the next round
    return "user_interrupt"
