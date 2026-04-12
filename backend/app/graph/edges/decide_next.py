"""
Conditional edge function for the main debate graph routing.

Routes the debate flow after the route_next node based on the
determined next_action in the state.
"""

from __future__ import annotations


def decide_next(state: dict) -> str:
    """
    Conditional edge function for the main graph.

    Routes based on state["next_action"]:
      - "opponent_rebut" -> "user_interrupt" (back to loop)
      - "same_team_add" -> "user_interrupt" (back to loop)
      - "opponent_chain" -> "user_interrupt" (back to loop)
      - "round_end"     -> "round_end"

    Args:
        state: Current DebateState dict.

    Returns:
        Target node name string for the conditional edge.
    """
    action = state.get("next_action", "opponent_rebut")

    if action in ("opponent_rebut", "same_team_add", "opponent_chain"):
        return "user_interrupt"
    elif action == "round_end":
        return "round_end"

    # Default fallback: loop back for another turn
    return "user_interrupt"
