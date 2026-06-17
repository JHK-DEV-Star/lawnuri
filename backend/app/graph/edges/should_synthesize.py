"""
Conditional edge function after final_judgment.

In complaint mode, route to the synthesis node (which drafts the 소장);
in debate mode, end the graph exactly as before. Default keeps the
original debate behavior (final_judgment -> END).
"""

from __future__ import annotations


def should_synthesize(state: dict) -> str:
    """
    Route after final_judgment based on mode.

    - mode == "complaint" -> "synthesis" (draft the 소장)
    - otherwise           -> "__end__" (original debate path)
    """
    if state.get("mode", "debate") == "complaint":
        return "synthesis"
    return "__end__"
