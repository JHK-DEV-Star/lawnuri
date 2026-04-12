"""
LangGraph node for round-end evaluation.

Increments the round counter and checks termination conditions to decide
whether the debate should continue, stop (final judgment), or pause
(awaiting user intervention).
"""

from __future__ import annotations

import asyncio

from app.graph.state import DebateState
from app.utils.logger import logger

# Default threshold: how many judges must vote "sufficient" to trigger early stop.
# With 3 judges, requiring 2 means a majority must agree.
_EARLY_STOP_MAJORITY_THRESHOLD = 2

_SPEAKING_ORDER_PROMPT = (
    "You are the presiding judge of a legal debate.\n"
    "Decide which team should speak FIRST in the next round.\n\n"
    "Consider:\n"
    "1. Defense necessity — which team needs to respond to a strong argument?\n"
    "2. Fairness — alternate when possible (review the history below).\n"
    "3. Debate flow — who has momentum or a pending rebuttal?\n"
    "4. Evaluation scores — give the weaker team a chance to recover.\n\n"
    "You MUST respond with JSON only:\n"
    '{"team": "team_a" or "team_b", "reasoning": "one sentence explanation"}\n'
)

_IMPROVEMENT_FEEDBACK_PROMPT = (
    "You are a panel of legal debate judges providing constructive feedback.\n"
    "Based on the evaluation notes from this round, generate concise, actionable\n"
    "improvement feedback for the team.\n\n"
    "Each bullet point MUST be a specific, executable instruction. NOT abstract advice.\n\n"
    "BAD (too vague):\n"
    "- 'Strengthen your evidence'\n"
    "- 'Improve your legal reasoning'\n"
    "- 'Address the opponent's arguments better'\n\n"
    "GOOD (specific and actionable):\n"
    "- 'Your argument about identity verification lacked a supporting precedent. Find a case where a court invalidated a contract due to failed identity checks.'\n"
    "- 'The opponent cited Article 750 to argue negligence. You did not respond to this. Counter it by arguing [specific approach].'\n"
    "- 'You claimed the contract was void but cited a precedent about wage disputes, which is unrelated. Replace it with a precedent about insurance contract fraud.'\n\n"
    "Focus on:\n"
    "- Which specific arguments lacked evidence or had weak/irrelevant precedents\n"
    "- Which opponent arguments were left unanswered and how to counter them\n"
    "- Which precedents should be replaced and what to search for instead\n\n"
    "Do NOT repeat praise. Do NOT give abstract advice.\n"
    "Output 3-5 bullet points of specific, actionable feedback.\n"
    "Write in the same language as the evaluation notes."
)


async def _decide_speaking_order(
    state: DebateState,
    current_round: int,
    llm_client,
) -> tuple[str, str]:
    """Decide which team speaks first in the next round.

    Returns:
        (team_id, reasoning) — e.g. ("team_b", "Team A needs to defend.")
    """
    # Round 1 always starts with team_a — no LLM call needed
    if current_round <= 1:
        return ("team_a", "First round default.")

    # --- Build compact context for the judge ---
    debate_log = state.get("debate_log", [])

    # 1) Speaking order history per round
    order_history_lines: list[str] = []
    seen_rounds: dict[int, str] = {}
    for entry in debate_log:
        r = entry.get("round", 0)
        if r and r not in seen_rounds:
            seen_rounds[r] = entry.get("team", entry.get("speaker", "?"))
    for r in sorted(seen_rounds):
        order_history_lines.append(f"  Round {r}: {seen_rounds[r]} spoke first")
    order_history = "\n".join(order_history_lines) or "  (no history)"

    # 2) Recent judge notes (current round, max 6, truncated)
    judge_notes = state.get("judge_notes", [])
    recent_notes = [
        n for n in judge_notes if n.get("round") == current_round
    ][:6]
    notes_text = "\n".join(
        f"  - {n.get('content', '')[:200]}" for n in recent_notes
    ) or "  (none)"

    # 3) Last statements from each team (truncated to 300 chars)
    last_a = ""
    last_b = ""
    for entry in reversed(debate_log):
        team = entry.get("team", "")
        stmt = entry.get("statement", "")
        if team == "team_a" and not last_a:
            last_a = stmt[:300]
        elif team == "team_b" and not last_b:
            last_b = stmt[:300]
        if last_a and last_b:
            break

    user_msg = (
        f"TOPIC: {state.get('topic', '')}\n\n"
        f"SPEAKING ORDER HISTORY:\n{order_history}\n\n"
        f"JUDGE NOTES (Round {current_round}):\n{notes_text}\n\n"
        f"LAST TEAM A STATEMENT:\n  {last_a or '(none)'}\n\n"
        f"LAST TEAM B STATEMENT:\n  {last_b or '(none)'}\n\n"
        f"Which team should speak FIRST in Round {current_round + 1}?"
    )

    try:
        result = await llm_client.achat_json(
            messages=[
                {"role": "system", "content": _SPEAKING_ORDER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        team = result.get("team", "team_a")
        reasoning = result.get("reasoning", "")
        if team not in ("team_a", "team_b"):
            logger.warning("[round_end] Invalid team '%s' from LLM, defaulting to team_a.", team)
            team = "team_a"
        logger.info("[round_end] Judge decided next round first speaker: %s (%s)", team, reasoning)
        return (team, reasoning)
    except Exception as exc:
        logger.warning("[round_end] Speaking order decision failed: %s. Defaulting to team_a.", exc)
        return ("team_a", "Fallback default.")


async def _generate_improvement_feedback(
    team_id: str,
    judge_notes_for_team: list[dict],
    topic: str,
    llm_client,
) -> str:
    """Generate improvement feedback for a team based on judge notes."""
    if not judge_notes_for_team:
        return ""

    notes_text = "\n\n".join(
        f"[Judge {n.get('judge_name', '?')}]: {n.get('content', '')}"
        for n in judge_notes_for_team
    )

    user_msg = (
        f"DEBATE TOPIC: {topic}\n"
        f"TEAM: {team_id}\n\n"
        f"JUDGE EVALUATION NOTES FROM THIS ROUND:\n{notes_text}\n\n"
        f"Generate improvement feedback for this team."
    )

    try:
        response = await llm_client.achat(
            messages=[
                {"role": "system", "content": _IMPROVEMENT_FEEDBACK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.strip()
    except Exception as exc:
        logger.warning("[round_end] Failed to generate improvement feedback for %s: %s", team_id, exc)
        return ""


async def round_end_node(state: DebateState, llm_client=None) -> dict:
    """
    Increment the round counter and check termination conditions.

    Termination conditions (checked in priority order):
    1. Status is "paused" or "stopped" by user -> pause/stop
    2. Round >= max_rounds -> stop (proceed to final judgment)
    3. Early-stop votes have sufficient majority -> stop
    4. Otherwise -> continue to next round

    Args:
        state: Current DebateState with round, max_rounds, min_rounds,
            early_stop_votes, and status information.

    Returns:
        Partial state update with incremented ``round`` and
        ``next_action`` set to one of: "continue", "stop", "paused".
    """
    current_round = state.get("round", 0)
    new_round = current_round + 1
    max_rounds = state.get("max_rounds", 10)
    min_rounds = state.get("min_rounds", 3)
    status = state.get("status", "running")
    early_stop_votes = state.get("early_stop_votes", [])

    logger.info(
        "[round_end] Round %d -> %d (max=%d, min=%d, status=%s).",
        current_round, new_round, max_rounds, min_rounds, status,
    )

    # Check 1: User-initiated pause or stop
    if status in ("paused", "stopped"):
        logger.info("[round_end] Debate %s by user.", status)
        return {
            "round": current_round,  # Don't increment — verdict/report are not rounds
            "next_action": "paused" if status == "paused" else "stop",
        }

    # Check 2: Maximum rounds reached
    if new_round > max_rounds:
        logger.info(
            "[round_end] Maximum rounds (%d) reached. Stopping debate.",
            max_rounds,
        )
        return {
            "round": current_round,  # Keep at last debate round (not max+1)
            "next_action": "stop",
        }

    # Check 3: Early stop by judge consensus (only after min_rounds)
    if new_round >= min_rounds and early_stop_votes:
        sufficient_count = sum(
            1 for v in early_stop_votes if v.get("vote") == "sufficient"
        )
        total_votes = len(early_stop_votes)

        logger.info(
            "[round_end] Early-stop votes: %d/%d 'sufficient' (threshold=%d).",
            sufficient_count, total_votes, _EARLY_STOP_MAJORITY_THRESHOLD,
        )

        if sufficient_count >= _EARLY_STOP_MAJORITY_THRESHOLD:
            # Collect reasons from sufficient-voting judges
            reasons = [
                v.get("reason", "")
                for v in early_stop_votes
                if v.get("vote") == "sufficient"
            ]
            logger.info(
                "[round_end] Early stop triggered by judge consensus. Reasons: %s",
                "; ".join(r[:80] for r in reasons if r),
            )
            return {
                "round": current_round,  # Stay at last debate round
                "next_action": "stop",
            }

    # Default: continue to next round
    logger.info("[round_end] Continuing to round %d.", new_round)

    # Generate improvement feedback for each team
    feedback = {}
    if llm_client:
        judge_notes = state.get("judge_notes", [])
        topic = state.get("topic", "")
        team_a_notes = [n for n in judge_notes if n.get("round") == current_round and n.get("team_evaluated") == "team_a"]
        team_b_notes = [n for n in judge_notes if n.get("round") == current_round and n.get("team_evaluated") == "team_b"]

        fb_a, fb_b, (next_team, order_reasoning) = await asyncio.gather(
            _generate_improvement_feedback("team_a", team_a_notes, topic, llm_client),
            _generate_improvement_feedback("team_b", team_b_notes, topic, llm_client),
            _decide_speaking_order(state, current_round, llm_client),
        )
        feedback = {"team_a": fb_a, "team_b": fb_b}
        logger.info("[round_end] Generated improvement feedback: team_a=%d chars, team_b=%d chars", len(fb_a), len(fb_b))
    else:
        next_team = "team_a"
        order_reasoning = "No LLM client — default."

    return {
        "round": new_round,
        "next_action": "continue",
        "current_team": next_team,
        "judge_improvement_feedback": feedback,
        "speaking_order_reasoning": order_reasoning,
        "pending_judge_questions": [],  # Clear leftover questions from previous round
    }
