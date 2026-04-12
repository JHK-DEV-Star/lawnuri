"""
LangGraph node for judge note accumulation.

After each team statement, every judge reviews the latest contribution
and accumulates evaluation notes. When the minimum round threshold has
been reached, judges also vote on whether early termination is warranted.
"""

from __future__ import annotations

import asyncio
import json

from app.agents.language import get_language_instruction
from app.graph.state import DebateState
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Maximum retries for JSON parsing in early-stop vote
_MAX_JSON_RETRIES = 2

# System prompt for judge note-taking
_NOTE_SYSTEM_PROMPT = """\
You are {judge_name}, a legal judge evaluating the latest debate statement.
Background: {background}. Judgment style: {judgment_style}.

Evaluate using these weighted criteria:

1. **Legal Accuracy (30%)**: Are cited statutes/precedents correct and applicable?
   Is the legal reasoning sound? Any misapplication of law?

2. **Evidence Quality (25%)**: Does the evidence support the claims?
   Are sources authoritative (Supreme Court > lower courts)?
   Are citations properly used in context?

3. **Argumentation Structure (20%)**: Is the argument logically organized (IRAC)?
   Does each claim flow from evidence to conclusion?

4. **Rebuttal Effectiveness (15%)**: Does it address opponent's strongest points?
   Are counterarguments substantive or merely dismissive?

5. **Persuasiveness (10%)**: Overall conviction and clarity of presentation.

Additionally, check for SUSPICIOUS precedent citations:
- Placeholder case numbers (e.g., 20XX다XXXXX, 2008다NNNNN) are FABRICATED.
- Case numbers not matching any known search result are suspect.
- If detected, you MUST flag this in your evaluation and reduce the Evidence Quality score.
- Note: "This team cited a fabricated/unverifiable precedent [case number], which undermines
  their evidence credibility."

Output a structured evaluation with one assessment per criterion (5 lines).
Format: "[Criterion] (score/10): brief assessment"
Example: "[Legal Accuracy] (8/10): Correctly applied Article 750 but overlooked..."
"""

# System prompt for early-stop voting
_EARLY_STOP_SYSTEM_PROMPT = """\
You are {judge_name}, an experienced legal judge.
You must decide whether this debate has reached a point where further rounds \
would not add meaningful value.

Vote "sufficient" ONLY if ALL of the following conditions are met:
1. Both sides have thoroughly presented their core arguments
2. Key evidence has been cited and addressed
3. Arguments are beginning to repeat without adding new substance
4. Additional rounds would not significantly change the outcome
5. All major legal perspectives and issues relevant to the case have been addressed \
(e.g., contractual validity, tortious liability, procedural defects, statutory violations). \
If there are legal angles that neither side has explored yet, vote "continue"

Be CONSERVATIVE: when in doubt, vote "continue". Premature termination \
deprives teams of their right to fully argue their case.

Output ONLY a valid JSON object:
{{
    "vote": "sufficient" | "continue",
    "reason": "Brief explanation for your vote"
}}
"""


async def _judge_evaluate_statement(
    judge: dict,
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Have a single judge evaluate the latest debate statement.

    Args:
        judge: Judge agent profile dict.
        state: Current DebateState.
        llm_client: LLM client for the evaluation.

    Returns:
        A note dict with judge_id, judge_name, round, team, and content.
    """
    judge_name = judge.get("name", "Judge")
    judge_id = judge.get("agent_id", "unknown")
    background = judge.get("background", "experienced legal professional")
    judgment_style = judge.get("debate_style", "balanced and thorough")

    # Get this round's statements from both teams
    debate_log = state.get("debate_log", [])
    current_round = state.get("round", 0)
    round_entries = [
        e for e in debate_log
        if isinstance(e, dict) and e.get("round") == current_round
    ]
    # Build both teams' statements for evaluation
    team_a_stmt = ""
    team_b_stmt = ""
    for entry in round_entries:
        team = entry.get("team", "")
        stmt = entry.get("statement", entry.get("content", ""))
        speaker = entry.get("speaker", "unknown")
        if team == "team_a":
            team_a_stmt = f"[{speaker}]: {stmt}"
        elif team == "team_b":
            team_b_stmt = f"[{speaker}]: {stmt}"

    # Build context from debate log (summarized)
    context_lines = []
    for entry in debate_log[-6:]:  # Last 6 entries for context
        team = entry.get("team", "?")
        speaker = entry.get("speaker", "?")
        content_text = str(entry.get("statement", entry.get("content", "")))
        context_lines.append(f"[{team}] {speaker}: {content_text}")
    context_text = "\n".join(context_lines) if context_lines else "(empty debate log)"

    # Load language from settings
    try:
        from app.api.settings import settings_mgr
        _settings = settings_mgr.load()
        _language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        _language = "ko"

    system_msg = _NOTE_SYSTEM_PROMPT.format(
        judge_name=judge_name,
        background=background,
        judgment_style=judgment_style,
    ) + get_language_instruction(_language)

    # Build evidence summary for this round
    current_round = state.get("round", 0)
    all_ev = state.get("all_evidences", [])
    round_ev = [e for e in all_ev if isinstance(e, dict) and e.get("round") == current_round]
    ev_lines_acc = []
    for ei, ev in enumerate(round_ev[:15]):
        team = ev.get("submitted_by", "?")
        src_type = ev.get("source_type", "?")
        detail = ev.get("source_detail", "")
        content = str(ev.get("content", ""))
        ev_lines_acc.append(f"[{ei+1}] ({team}) [{src_type}] {detail}: {content}")
    evidence_text = "\n".join(ev_lines_acc) if ev_lines_acc else "(no evidence cited this round)"

    _team_a_name = state.get("team_a_name", "Team A")
    _team_b_name = state.get("team_b_name", "Team B")
    user_msg = (
        f"Debate topic: {state.get('topic', '?')}\n"
        f"Opinion A ({_team_a_name}): {state.get('opinion_a', '?')}\n"
        f"Opinion B ({_team_b_name}): {state.get('opinion_b', '?')}\n"
        f"Current round: {current_round}\n\n"
        f"Recent debate context:\n{context_text}\n\n"
        f"=== {_team_a_name}'s Statement (Round {current_round}) ===\n"
        f"{team_a_stmt}\n\n"
        f"=== {_team_b_name}'s Statement (Round {current_round}) ===\n"
        f"{team_b_stmt}\n\n"
        f"Evidence cited this round:\n{evidence_text}\n\n"
        f"Evaluate BOTH teams' statements. Compare their arguments, evidence quality, "
        f"and persuasiveness. Identify which team made stronger points in this round."
    )

    try:
        notes_text = await llm_client.achat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1000,
        )
    except Exception as exc:
        logger.error(
            "[judge_accumulate] Judge %s evaluation error: %s", judge_name, exc
        )
        notes_text = f"(Evaluation error for judge {judge_name}: {exc})"

    return {
        "judge_id": judge_id,
        "judge_name": judge_name,
        "round": state.get("round", 0),
        "team_evaluated": "both",
        "content": notes_text,
    }


async def _judge_early_stop_vote(
    judge: dict,
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Have a single judge vote on whether the debate should stop early.

    Only called when the current round exceeds the minimum round threshold.

    Args:
        judge: Judge agent profile dict.
        state: Current DebateState.
        llm_client: LLM client for the vote.

    Returns:
        A vote dict with judge_id, vote ("sufficient" or "continue"),
        and reason.
    """
    judge_name = judge.get("name", "Judge")
    judge_id = judge.get("agent_id", "unknown")

    # Summarize the debate so far
    debate_log = state.get("debate_log", [])
    summary_lines = []
    for entry in debate_log:
        team = entry.get("team", "?")
        content_text = str(entry.get("statement", entry.get("content", "")))
        summary_lines.append(f"[{team}] {content_text}")
    debate_summary = "\n".join(summary_lines) if summary_lines else "(no debate entries)"

    # Load language from settings
    try:
        from app.api.settings import settings_mgr
        _settings = settings_mgr.load()
        _language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        _language = "ko"

    system_msg = _EARLY_STOP_SYSTEM_PROMPT.format(judge_name=judge_name) + get_language_instruction(_language)

    user_msg = (
        f"Debate topic: {state.get('topic', '?')}\n"
        f"Round: {state.get('round', 0)} / max {state.get('max_rounds', 10)}\n"
        f"Min rounds required: {state.get('min_rounds', 3)}\n\n"
        f"Full debate so far:\n{debate_summary}\n\n"
        f"Should this debate continue or is it sufficient to conclude?"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Try to get valid JSON vote
    for attempt in range(_MAX_JSON_RETRIES + 1):
        try:
            vote_data = await llm_client.achat_json(
                messages, temperature=0.2, max_tokens=300
            )
            vote = vote_data.get("vote", "continue")
            reason = vote_data.get("reason", "")
            return {
                "judge_id": judge_id,
                "judge_name": judge_name,
                "vote": vote if vote in ("sufficient", "continue") else "continue",
                "reason": reason,
                "round": state.get("round", 0),
            }
        except json.JSONDecodeError:
            if attempt < _MAX_JSON_RETRIES:
                messages.append({
                    "role": "user",
                    "content": "Invalid JSON. Output ONLY the JSON object.",
                })
            else:
                logger.warning(
                    "[judge_accumulate] Judge %s early-stop vote JSON failed.", judge_name
                )

    return {
        "judge_id": judge_id,
        "judge_name": judge_name,
        "vote": "continue",
        "reason": "Failed to parse vote response; defaulting to continue.",
        "round": state.get("round", 0),
    }


async def judge_accumulate_node(
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Each judge reviews the latest statement and accumulates evaluation notes.

    After the minimum rounds threshold is reached, judges also vote on
    whether early termination is warranted.

    Args:
        state: Current DebateState with debate_log, judge_agents, and
            round information.
        llm_client: LLM client for judge evaluations.

    Returns:
        Partial state update with ``judge_notes`` (appended) and
        ``early_stop_votes`` (replaced each evaluation cycle).
    """
    judges = state.get("judge_agents", [])
    current_round = state.get("round", 0)
    min_rounds = state.get("min_rounds", 3)

    if not judges:
        logger.warning("[judge_accumulate] No judge agents configured.")
        return {"judge_notes": []}

    # Check if both teams have spoken in this round before evaluating
    debate_log = state.get("debate_log", [])
    teams_spoken = set()
    for entry in debate_log:
        if isinstance(entry, dict) and entry.get("round") == current_round:
            teams_spoken.add(entry.get("team", ""))
    both_teams_spoke = "team_a" in teams_spoken and "team_b" in teams_spoken

    if not both_teams_spoke:
        logger.info(
            "[judge_accumulate] Only %s spoke so far in round %d — skipping evaluation until both teams speak.",
            teams_spoken, current_round,
        )
        return {"judge_notes": []}

    logger.info(
        "[judge_accumulate] Both teams spoke in round %d — %d judges evaluating.",
        current_round, len(judges),
    )

    # Phase 1: All judges evaluate the latest statement in parallel
    eval_tasks = [
        _judge_evaluate_statement(judge, state, llm_client)
        for judge in judges
    ]
    notes_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

    new_notes: list[dict] = []
    for result in notes_results:
        if isinstance(result, dict):
            new_notes.append(result)
        elif isinstance(result, Exception):
            logger.error("[judge_accumulate] Judge evaluation exception: %s", result)

    logger.info(
        "[judge_accumulate] Collected %d evaluation notes.", len(new_notes)
    )

    # Phase 2: Early stop voting (only after min_rounds)
    early_stop_votes: list[dict] = []
    if current_round >= min_rounds:
        logger.info(
            "[judge_accumulate] Round %d >= min_rounds %d, collecting early-stop votes.",
            current_round, min_rounds,
        )
        vote_tasks = [
            _judge_early_stop_vote(judge, state, llm_client)
            for judge in judges
        ]
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        for result in vote_results:
            if isinstance(result, dict):
                early_stop_votes.append(result)
            elif isinstance(result, Exception):
                logger.error("[judge_accumulate] Early-stop vote exception: %s", result)

        sufficient_count = sum(
            1 for v in early_stop_votes if v.get("vote") == "sufficient"
        )
        logger.info(
            "[judge_accumulate] Early-stop votes: %d/%d 'sufficient'.",
            sufficient_count, len(early_stop_votes),
        )

    update: dict = {
        "judge_notes": new_notes,  # Annotated with operator.add, will append
    }

    # Only update early_stop_votes if we actually collected them this round
    if early_stop_votes:
        update["early_stop_votes"] = early_stop_votes

    return update
