"""
LangGraph node for dynamically assigning tasks to team members.

Analyses the opponent's latest statement and the current debate context
to distribute specific research and argumentation tasks to each member
of the active team, leveraging their individual specialties.
"""

from __future__ import annotations

import json

from app.agents.language import get_language_instruction
from app.graph.state import TeamState
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Maximum number of retries for JSON parsing from LLM output
_MAX_JSON_RETRIES = 2

# System prompt for the role-assignment LLM call
_SYSTEM_PROMPT = """\
You are a debate team coordinator. Your job is to assign specific tasks AND \
unique argument angles to each team member for the current debate round.

Analyze the opponent's statement and distribute work based on each member's \
specialty and personality. Each member should receive a task that best fits \
their expertise, AND a unique argument angle they must focus on.

Output ONLY a valid JSON object with this exact structure:
{
    "strategy_note": "Brief overall strategy description for this round",
    "assignments": [
        {
            "agent_id": "agent's unique ID",
            "agent_name": "agent's name",
            "task": "Specific task description (e.g., Search for relevant statutes on...)",
            "search_type": "statute|precedent|document|graph",
            "priority": "primary|supporting",
            "argument_angle": ["primary_angle", "secondary_angle"]
        }
    ]
}

Rules:
- Assign exactly one task per team member.
- At least one member must have priority "primary" (lead argument builder).
- Search types: "statute" for legal code lookup, "precedent" for case law, \
"document" for uploaded documents, "graph" for knowledge graph relations.
- Tasks should be complementary, not overlapping.
- Consider previous round assignments to rotate responsibilities when possible.

Primary Speaker Selection Criteria:
- Assign "priority": "primary" to the agent whose specialty is MOST relevant \
to the key issues being debated THIS round.
- Selection criteria (in priority order):
  1. RELEVANCE: Match agent specialty to the round's main legal issues.
  2. EXPERTISE MATCH: Procedural issues → procedural specialist. \
Precedent analysis → precedent expert.
  3. DIVERSITY: Previous speakers: {previous_speakers}. Prefer agents who \
haven't spoken recently, unless another agent is clearly more qualified.
  4. FRESHNESS: An agent who hasn't spoken yet may bring a new perspective.
- Do NOT always assign the same agent as primary.

Argument Angle Rules:
- Each agent MUST have at least 2 argument_angles as a list: [primary, secondary].
- The primary angle is the agent's main focus; the secondary provides depth.
- No two agents should have the same PRIMARY angle (first element).
- Choose from (but not limited to) these angles:
  * "statutory" — specific law articles, regulations, and their interpretation
  * "precedent" — court precedent cases and their holdings
  * "procedural" — procedural defects, compliance failures, due process
  * "factual" — factual evidence, circumstantial analysis, timeline reconstruction
  * "policy" — policy implications, fairness, equity, public interest
  * "constitutional" — constitutional principles, fundamental rights
  * "comparative" — comparative law, similar cases in other jurisdictions
  * "contractual" — contract terms, obligations, breach analysis
  * "damages" — harm assessment, causation, remedy calculation
  * "credibility" — witness reliability, evidence authenticity, burden of proof
  * "temporal" — statute of limitations, timing of actions, prescription periods
  * "regulatory" — administrative regulations, regulatory compliance, agency guidance
  * "custom_practice" — industry customs, standard practices, professional norms
  * "intent" — party intentions, mens rea, good/bad faith analysis
  * "rights_balance" — competing rights, proportionality, interests balancing
- You may create a custom angle if none above fit the specific case.
- Choose angles that best match each agent's specialty AND the legal issues at hand.
"""


def _build_user_prompt(state: TeamState) -> str:
    """Build the user prompt with current debate context for role assignment."""
    members_desc = []
    for m in state.get("members", []):
        members_desc.append(
            f"- {m.get('name', 'Unknown')} (ID: {m.get('agent_id', '?')}): "
            f"Specialty: {m.get('specialty', 'general')}, "
            f"Style: {m.get('debate_style', 'balanced')}, "
            f"Personality: {m.get('personality', 'neutral')}"
        )
    members_text = "\n".join(members_desc) if members_desc else "(no members)"

    opponent_stmt = state.get("opponent_statement", "")
    if not opponent_stmt:
        opponent_stmt = "(Opening round - no opponent statement yet)"

    # Summarize previous assignments if available
    prev_assignments = state.get("role_assignments", [])
    prev_summary = ""
    if prev_assignments:
        prev_lines = []
        for a in prev_assignments:
            prev_lines.append(
                f"  - {a.get('agent_name', '?')}: {a.get('task', '?')} "
                f"[{a.get('search_type', '?')}]"
            )
        prev_summary = "Previous round assignments:\n" + "\n".join(prev_lines)
    else:
        prev_summary = "Previous round assignments: (none - first round)"

    # Debate context summary (last few entries)
    context_entries = state.get("debate_context", [])
    context_summary = ""
    if context_entries:
        recent = context_entries[-3:]  # Last 3 entries
        context_lines = []
        for entry in recent:
            speaker = entry.get("speaker", "?")
            team = entry.get("team", "?")
            snippet = str(entry.get("statement", entry.get("content", "")))[:200]
            context_lines.append(f"  [{team}] {speaker}: {snippet}...")
        context_summary = "Recent debate context:\n" + "\n".join(context_lines)
    else:
        context_summary = "Recent debate context: (none yet)"

    # Extra evidence available
    extra_evidence = state.get("extra_evidence", [])
    extra_note = ""
    if extra_evidence:
        extra_note = (
            f"\nNote: The team has {len(extra_evidence)} user-injected "
            f"evidence item(s) available. Consider incorporating them."
        )

    team_opinion = state.get("team_opinion", "")
    opponent_opinion = state.get("opponent_opinion", "")

    return f"""\
Round: {state.get('round', 1)}
Team opinion (we advocate): {team_opinion}
Opponent opinion: {opponent_opinion}

Team members:
{members_text}

Opponent's latest statement:
{opponent_stmt}

{prev_summary}

{context_summary}
{extra_note}

Assign tasks to each team member for this round."""


async def assign_roles_node(state: TeamState, llm_client: LLMClient) -> dict:
    """
    Assign specific tasks to each team member for the current round.

    Uses the LLM to analyze the opponent's statement, the debate history,
    and each member's specialty to produce complementary role assignments.

    Args:
        state: Current TeamState containing members, opponent statement,
            debate context, and previous assignments.
        llm_client: LLMClient instance for making LLM calls.

    Returns:
        Partial state update with ``role_assignments`` list and
        ``agreed_strategy`` string.
    """
    team_id = state.get("team_id", "unknown")
    members = state.get("members", [])
    if not members:
        logger.error("[assign_roles] No members for %s - returning empty assignments.", team_id)
        return {"role_assignments": [], "agreed_strategy": "No members available"}

    logger.info(
        "[assign_roles] Assigning roles for %s (round %d, %d members).",
        team_id,
        state.get("round", 0),
        len(members),
    )

    from app.api.settings import settings_mgr
    try:
        settings = settings_mgr.load()
        language = settings.get("debate", {}).get("language", "ko")
    except Exception:
        language = "ko"
    # Build previous speakers list for diversity
    # TeamState uses "debate_context" for the full debate log (not "debate_log")
    debate_log = state.get("debate_context", state.get("debate_log", []))
    prev_speakers = [e.get("speaker", "?") for e in debate_log if e.get("team") == team_id]
    prev_speakers_text = ", ".join(prev_speakers[-5:]) if prev_speakers else "(none yet)"
    system_prompt = _SYSTEM_PROMPT.replace("{previous_speakers}", prev_speakers_text) + get_language_instruction(language)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_prompt(state)},
    ]

    # Attempt to get valid JSON from the LLM with retries
    assignments_data = None
    for attempt in range(_MAX_JSON_RETRIES + 1):
        try:
            assignments_data = await llm_client.achat_json(
                messages, temperature=0.3, max_tokens=2000
            )
            break
        except json.JSONDecodeError:
            if attempt < _MAX_JSON_RETRIES:
                logger.warning(
                    "[assign_roles] JSON parse failed (attempt %d/%d), retrying.",
                    attempt + 1,
                    _MAX_JSON_RETRIES + 1,
                )
                # Add a hint to produce valid JSON
                messages.append({
                    "role": "user",
                    "content": "Your response was not valid JSON. Please output ONLY "
                               "the JSON object with no extra text.",
                })
            else:
                logger.error("[assign_roles] Failed to get valid JSON after retries.")

    # Fallback: create default assignments if LLM failed
    if assignments_data is None:
        logger.warning("[assign_roles] Using fallback default assignments.")
        members = state.get("members", [])
        assignments_data = {
            "strategy_note": "Default strategy: each member searches in their specialty area.",
            "assignments": [
                {
                    "agent_id": m.get("agent_id", f"agent_{i}"),
                    "agent_name": m.get("name", f"Agent {i}"),
                    "task": f"Search for evidence related to your specialty: {m.get('specialty', 'general')}",
                    "search_type": "document",
                    "priority": "primary" if i == (state.get("round", 1) - 1) % len(members) else "supporting",
                }
                for i, m in enumerate(members)
            ],
        }

    role_assignments = assignments_data.get("assignments", [])
    strategy_note = assignments_data.get("strategy_note", "")

    logger.info(
        "[assign_roles] Assigned %d roles for %s. Strategy: %s",
        len(role_assignments),
        team_id,
        strategy_note[:100],
    )

    return {
        "role_assignments": role_assignments,
        "agreed_strategy": strategy_note,
    }
