"""
LangGraph node for dynamic routing after a team speaks.

Determines the next action in the debate flow based on the current state:
- opponent_rebut: the opposing team gets to respond
- same_team_add: the current team adds a supplementary statement
- round_end: move to round-end evaluation

Uses LLM-based analysis of the debate context to make routing decisions,
defaulting to balanced alternation (opponent_rebut) for fair debate.
"""

from __future__ import annotations


from app.graph.state import DebateState
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Maximum retries for JSON parsing
_MAX_JSON_RETRIES = 2

# System prompt for routing decision
_ROUTING_SYSTEM_PROMPT = """\
You are a debate moderator deciding what should happen next in a legal debate.

After reviewing the current state of the debate, choose the next action:

1. "opponent_rebut" - The opposing team responds to the statement just made. \
This is the DEFAULT and most common action for balanced debate.
2. "same_team_add" - The same team adds a supplementary statement. Use this \
ONLY if the team's previous statement was clearly incomplete or they raised \
a new point that needs immediate elaboration.
3. "round_end" - End the current round. Use this when BOTH teams have spoken \
at least once in this round.

Rules:
- Default to "opponent_rebut" for fair, balanced debate.
- "same_team_add" should be rare (less than 10% of decisions).
- "round_end" should only be chosen after both sides have spoken.
- Never let one team dominate with consecutive statements.

Output ONLY a valid JSON object:
{
    "next_action": "opponent_rebut" | "same_team_add" | "round_end",
    "reason": "Brief explanation for the routing decision"
}
"""


async def route_next_node(
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Determine what happens next in the debate flow.

    Analyses the debate state to decide whether the opponent should rebut,
    the same team should add to their statement, or the round should end.

    Args:
        state: Current DebateState with debate_log, current_team, and
            round information.
        llm_client: LLM client for the routing decision.

    Returns:
        Partial state update with ``next_action`` and ``current_team``.
    """
    current_team = state.get("current_team", "team_a")
    current_round = state.get("round", 0)
    debate_log = state.get("debate_log", [])

    logger.info(
        "[route_next] Routing after %s speaks (round %d, %d log entries).",
        current_team, current_round, len(debate_log),
    )

    # Quick heuristic: check if both teams have spoken this round
    teams_spoken_this_round = set()
    for entry in debate_log:
        if entry.get("round", -1) == current_round:
            teams_spoken_this_round.add(entry.get("team", ""))

    both_spoken = "team_a" in teams_spoken_this_round and "team_b" in teams_spoken_this_round

    # If only one team has spoken, route to opponent rebuttal
    if not both_spoken:
        other_team = "team_b" if current_team == "team_a" else "team_a"
        logger.info(
            "[route_next] Only %s has spoken this round. Routing to %s (opponent_rebut).",
            current_team, other_team,
        )
        return {
            "next_action": "opponent_rebut",
            "current_team": other_team,
        }

    # Both teams have spoken — end round immediately (no LLM decision needed)
    # The next round's first speaker will be the team that spoke second this round
    # (fairness: alternate who goes first)
    next_first_team = "team_b" if current_team == "team_a" else "team_a"
    logger.info(
        "[route_next] Both teams spoke in round %d. Ending round. Next first: %s.",
        current_round, next_first_team,
    )
    return {
        "next_action": "round_end",
        "current_team": next_first_team,
    }
