"""
LangGraph node for handling user intervention (human-in-the-loop).

When a user injects evidence, hints, or documents mid-debate, this node
processes the pending intervention and routes it to the target team's
extra_evidence pool before the next team processing step.
"""

from __future__ import annotations

from app.graph.state import DebateState
from app.utils.logger import logger


async def user_interrupt_node(state: DebateState) -> dict:
    """
    Check for pending user intervention and inject it into the target team.

    If ``user_interrupt`` is set in the state, build an evidence entry from
    its contents and append it to the target team's ``extra_evidence`` list.
    The intervention data is then cleared so it is not re-processed.

    Expected ``user_interrupt`` dict shape::

        {
            "target_team": "team_a" | "team_b",
            "content": "...",
            "type": "hint" | "document"
        }

    Returns:
        Partial state update with the modified team state, cleared
        ``user_interrupt``, and the new evidence appended to
        ``all_evidences``.  Returns an empty dict when there is no
        pending intervention.
    """
    interrupt_data = state.get("user_interrupt")
    if not interrupt_data:
        logger.debug("[user_interrupt] No pending intervention, passing through.")
        return {}

    target_team: str = interrupt_data.get("target_team", "team_a")
    content: str = interrupt_data.get("content", "")
    content_type: str = interrupt_data.get("type", "hint")  # "hint" or "document"

    if not content:
        logger.warning("[user_interrupt] Intervention has empty content, skipping.")
        return {"user_interrupt": None}

    # Determine target teams
    if target_team == "both":
        teams = ["team_a", "team_b"]
    else:
        teams = [target_team]

    logger.info(
        "[user_interrupt] Processing intervention for %s (type=%s, round=%d).",
        target_team,
        content_type,
        state.get("round", 0),
    )

    # Build evidence entries compatible with the Evidence model schema
    _type_labels = {"hint": "힌트", "evidence": "증거"}
    _type_label = _type_labels.get(content_type, content_type)

    updates: dict = {"user_interrupt": None, "all_evidences": []}

    for t in teams:
        evidence = {
            "content": content,
            "source_type": "user_injected",
            "source_detail": f"사용자 개입: {_type_label} (round {state.get('round', 0)})",
            "submitted_by": t,
            "round": state.get("round", 0),
            "speaker": "user",
            "type": content_type,
        }

        # Update the appropriate team's extra evidence list
        team_state_key = f"{t}_state"
        team_state = dict(state.get(team_state_key, {}))
        extra = list(team_state.get("extra_evidence", []))
        extra.append(evidence)
        team_state["extra_evidence"] = extra
        updates[team_state_key] = team_state
        updates["all_evidences"].append(evidence)

        logger.info(
            "[user_interrupt] Injected evidence into %s (total extra_evidence: %d).",
            t,
            len(extra),
        )

    return updates
