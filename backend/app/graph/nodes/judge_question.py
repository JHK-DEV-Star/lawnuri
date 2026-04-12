"""
LangGraph nodes for judge Q&A during debate rounds.

After both teams have spoken in a round, each judge may ask a targeted
question to a specific agent.  The agent then answers, and the exchange
is recorded in the debate log and a dedicated judge_qa_log.
"""

from __future__ import annotations

import asyncio
import json

from app.agents.language import get_language_instruction, SIMULATION_FRAME_ADVOCATE, SIMULATION_FRAME_JUDGE
from app.graph.state import DebateState
from app.utils.llm_client import LLMClient
from app.utils.logger import logger

# Maximum retries for JSON parsing in judge question decision
_MAX_JSON_RETRIES = 2

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_QUESTION_DECISION_PROMPT = """\
You are {judge_name}, a legal judge observing an ongoing debate.
Background: {background}. Judgment style: {judgment_style}.

After reviewing the statements from this round, decide whether you want \
to ask a clarifying question to one of the debating agents.

You SHOULD ask a question when you observe ANY of the following:

1. **Irrelevant Legal Sources**: A cited law or precedent appears unrelated to the \
debate topic. Ask the agent to explain the specific connection between the cited \
source and the dispute at hand. If the relevance is unclear, request that they \
withdraw the citation or provide a direct legal nexus.

2. **Unsupported Claims**: A legal assertion is made without citing specific \
statutes (조문), case numbers (사건번호), or court rulings. Ask the agent to \
provide the exact legal authority — statute article number, case number, or \
ruling date — that supports their claim.

3. **Contradictory Evidence**: Cited evidence appears to undermine rather than \
support the agent's own position. Ask the agent to explain how the cited source \
supports their argument, or whether they acknowledge the contradiction.

4. **Misinterpreted Precedents**: A cited case's holding, ratio decidendi, or \
factual context may not align with how the agent is applying it. Ask the agent \
to clarify the specific holding of the cited case and how it applies to the \
current facts.

5. **Missing Key Issues**: A critical legal issue directly relevant to the topic \
(e.g., statute of limitations, burden of proof, consent requirements) has not \
been addressed by either team. Ask the relevant team to present their position \
on the unaddressed issue.

6. **Outdated or Amended Law**: A cited law may have been amended since the \
cited precedent was decided, or a precedent may have been overruled by a later \
Supreme Court decision. Ask the agent to confirm whether the cited authority \
reflects current law and whether any subsequent amendments affect their argument.

7. **Repetitive Arguments**: A team is repeating the same argument across \
multiple rounds without introducing new evidence or analysis. Ask the agent \
to present a new perspective, additional evidence, or address a different aspect \
of the issue.

Your questions can take various forms depending on the situation:
- Critique: pointing out logical errors or misapplied citations
- Clarification: requesting additional explanation or detail
- Curiosity: asking for deeper analysis on an interesting point
- Supplementation: requesting the team address a gap in their argument

Do NOT ask about minor stylistic issues, presentation format, or points that \
are already well-established in the debate. It is perfectly acceptable to have \
no question — not every round requires one.

# Relevance Challenge (IMPORTANT)
- If an agent cites a law or precedent that appears UNRELATED to the case topic, \
you MUST challenge them to explain its specific relevance.
  Example: If the case is about guarantee insurance but the agent cites the \
Franchise Business Act, ask: "How is [cited law] relevant to this guarantee \
insurance dispute? Please explain the specific connection."
- If an agent makes abstract claims like "procedural defects existed" without \
specifying WHAT the defects were, ask them to be specific.
- If an agent's cited evidence cannot be verified from the search results, \
note this as a credibility concern.

Note: You evaluate independently of other judges. Even if another judge asks \
a question on a different issue, you should still ask YOUR question if you \
identify a separate problem or area needing clarification. Different judges \
noticing different issues is expected and valuable for thorough evaluation.

Round {round} statements:
{round_statements}

Available agents you may question:
{team_a_label}: {team_a_names}
{team_b_label}: {team_b_names}

If you have a question, output ONLY a valid JSON object:
{{
    "has_question": true,
    "target_agent_id": "<agent_id of the agent you want to question>",
    "question": "<your question>"
}}

If you have no question, output ONLY:
{{
    "has_question": false,
    "target_agent_id": "",
    "question": ""
}}
"""

_AGENT_ANSWER_PROMPT = """\
You are {agent_name}, a debater in a legal debate.
Your team: {team_id}.

## YOUR CORE POSITION (ANCHOR — NEVER ABANDON)
{team_opinion}

You are answering a judge's question. Your answer MUST support YOUR team's position above.
Do NOT argue for the opponent's position. Do NOT present a balanced view.
You are an ADVOCATE for your team, even when answering questions.
If the judge's question challenges your position, defend it with evidence.

## Self-Harmful Statement Prevention
- NEVER make statements that strengthen the opponent's position.
- NEVER voluntarily admit weaknesses without immediately providing a stronger counter-argument.
- If a judge asks about a weakness, acknowledge it minimally and immediately redirect
  to your team's strongest counter-point.
  BAD: "Yes, we admit the notification was indeed delayed by 2 years."
  GOOD: "While there was a delay in notification, the key issue is that no valid
  consent was obtained in the first place, making the notification timeline irrelevant."
- NEVER concede a factual point without reframing it in your team's favor.

## Original Situation
{situation_brief}

## Key Facts
{analysis_summary}

You MUST answer with SPECIFIC facts from the Original Situation and
VERIFIED evidence from YOUR team's pool below.

## Your Team's Evidence Pool (cite ONLY from this list)
{team_evidence_summary}

## Opponent's Cited Evidence (for reference/rebuttal ONLY — do NOT cite as your own)
{opponent_cited_summary}

## Citation Format (MANDATORY)
- When citing a precedent, you MUST include a VERBATIM QUOTE from the holding text
  in 「」 brackets, then state whether the court upheld or denied the claim.
  BAD: "This is supported by [판례: CASE_NUMBER]."
  BAD: "[판례: CASE_NUMBER] held that the court recognized X" (paraphrase, no verbatim quote)
  GOOD: "[판례: CASE_NUMBER] 「verbatim quote from holding」 → The court upheld/denied ... This applies because..."
- If the holding CONTRADICTS your position, you may distinguish it but NEVER misrepresent it.
- When citing a statute, include the specific article content.
  BAD: "This violates [법령: LAW_NAME]."
  GOOD: "[법령: LAW_NAME Article N] stipulates that '[quote the provision].' This means..."
- NEVER cite evidence not in your team's pool above
- NEVER invent or approximate case numbers

## Anti-Hallucination Rules
- ONLY cite precedents/laws that appear in your evidence pool above.
- NEVER invent, guess, approximate, or truncate case numbers.
- NEVER use hypothetical or fictional case numbers for any reason.
- If the opponent cited a case number not in your pool, do NOT re-cite it.
- If you cannot answer with verified evidence, state that additional research is needed.
- You MAY state general legal principles without a specific case number,
  but you must NOT invent one to support your statement.

## Judge's Question
{question}

Provide a focused, evidence-based answer.
"""


# ---------------------------------------------------------------------------
# Helper: build round statements summary
# ---------------------------------------------------------------------------

def _get_round_statements(debate_log: list, current_round: int) -> str:
    """Return a text summary of all statements in the current round."""
    lines: list[str] = []
    for entry in debate_log:
        if entry.get("round") == current_round:
            team = entry.get("team", "?")
            speaker = entry.get("speaker", "?")
            stmt = str(entry.get("statement", ""))[:400]
            lines.append(f"[{team}] {speaker}: {stmt}")
    return "\n".join(lines) if lines else "(no statements this round)"


def _agent_names_list(agents: list) -> str:
    """Format agent list as 'name (agent_id)' entries."""
    parts = []
    for a in agents:
        parts.append(f"{a.get('name', '?')} ({a.get('agent_id', '?')})")
    return ", ".join(parts) if parts else "(none)"


# ---------------------------------------------------------------------------
# Node: judge_question_node
# ---------------------------------------------------------------------------

async def _judge_decide_question(
    judge: dict,
    state: DebateState,
    llm_client: LLMClient,
    language: str,
) -> dict | None:
    """Have a single judge decide whether to ask a question."""
    judge_name = judge.get("name", "Judge")
    judge_id = judge.get("agent_id", "unknown")
    background = judge.get("background", "experienced legal professional")
    judgment_style = judge.get("debate_style", "balanced and thorough")
    current_round = state.get("round", 0)

    debate_log = state.get("debate_log", [])
    round_statements = _get_round_statements(debate_log, current_round)
    team_a_names = _agent_names_list(state.get("team_a_agents", []))
    team_b_names = _agent_names_list(state.get("team_b_agents", []))

    # Build previous Q&A history for dedup (include both questions AND answers)
    qa_log = state.get("judge_qa_log", [])
    prev_qa_lines = []
    for q in qa_log:
        if q.get("round") == current_round:
            line = (
                f"- [{q.get('judge_name', '?')}] Q: {q.get('question', '')[:200]}"
            )
            answer = q.get("answer", "")
            if answer:
                line += f"\n  A ({q.get('target_agent_id', '?')}): {answer[:200]}"
            prev_qa_lines.append(line)
    prev_qa_section = ""
    if prev_qa_lines:
        prev_qa_section = (
            "\n\n## Questions Already Asked and Answered This Round\n"
            + "\n".join(prev_qa_lines)
            + "\n\nCRITICAL RULES:\n"
            "- Do NOT generate a question that is the same as or substantially similar to any above.\n"
            "- If a question was already asked AND answered, do NOT ask the same thing again.\n"
            "- If the answer was unsatisfactory, you may ask a specific FOLLOW-UP,\n"
            "  but it MUST address a DIFFERENT aspect of the issue.\n"
            "- If you have no genuinely NEW question, respond with has_question: false.\n"
        )

    # Build evidence summary: ONLY evidence directly cited in statements
    import re as _re
    all_ev = state.get("all_evidences", [])
    cited_ids: set[str] = set()
    for entry in debate_log:
        if entry.get("round") == current_round:
            stmt = str(entry.get("statement", ""))
            for m in _re.finditer(r'\[(판례|법령|헌재|행심|문서|court_precedent|legal_statute):\s*([^\]]+)\]', stmt):
                cited_ids.add(m.group(2).strip())
    cited_ev = [
        ev for ev in all_ev
        if ev.get("evidence_id", "") in cited_ids
        or ev.get("source_detail", "") in cited_ids
    ]
    ev_lines: list[str] = []
    for ei, ev in enumerate(cited_ev[:30]):
        team = ev.get("submitted_by", "?")
        src_type = ev.get("source_type", "?")
        detail = ev.get("source_detail", "")[:100]
        content = str(ev.get("content", ""))[:200]
        ev_lines.append(f"[{ei+1}] ({team}) [{src_type}] {detail}: {content}")
    evidence_section = ""
    if ev_lines:
        evidence_section = (
            "\n\n## Evidence Directly Cited in This Round's Statements\n"
            + "\n".join(ev_lines)
            + "\n\nYou may ONLY ask questions about evidence that was DIRECTLY CITED in the\n"
            "representative statements above. Do NOT ask about evidence that was found\n"
            "during internal search but not mentioned in the statements.\n"
            "Do NOT score or rate evidence quality during questioning.\n"
            "Your role here is to ASK, not to EVALUATE.\n"
        )

    team_a_label = state.get("team_a_name", "Team A")
    team_b_label = state.get("team_b_name", "Team B")

    system_msg = SIMULATION_FRAME_JUDGE + _QUESTION_DECISION_PROMPT.format(
        judge_name=judge_name,
        background=background,
        judgment_style=judgment_style,
        round=current_round,
        round_statements=round_statements,
        team_a_names=team_a_names,
        team_b_names=team_b_names,
        team_a_label=team_a_label,
        team_b_label=team_b_label,
    ) + evidence_section + prev_qa_section + get_language_instruction(language)

    user_msg = (
        f"Debate topic: {state.get('topic', '?')}\n"
        f"Opinion A: {state.get('opinion_a', '?')}\n"
        f"Opinion B: {state.get('opinion_b', '?')}\n\n"
        f"Decide whether you have a clarifying question for any agent."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(_MAX_JSON_RETRIES + 1):
        try:
            data = await llm_client.achat_json(
                messages, temperature=0.3, max_tokens=1000,
            )
            if data.get("has_question"):
                return {
                    "judge_id": judge_id,
                    "judge_name": judge_name,
                    "target_agent_id": data.get("target_agent_id", ""),
                    "question": data.get("question", ""),
                    "round": current_round,
                }
            # No question
            return None
        except json.JSONDecodeError:
            if attempt < _MAX_JSON_RETRIES:
                messages.append({
                    "role": "user",
                    "content": "Invalid JSON. Output ONLY the JSON object.",
                })
            else:
                logger.warning(
                    "[judge_question] Judge %s question decision JSON failed.",
                    judge_name,
                )
    return None


async def judge_question_node(
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    After both teams have spoken in the current round, each judge decides
    whether to ask a clarifying question to a specific agent.

    Questions are collected into ``pending_judge_questions`` and also
    recorded in ``judge_qa_log``.

    Args:
        state: Current DebateState.
        llm_client: LLM client for judge decisions.

    Returns:
        Partial state update with ``pending_judge_questions`` and
        ``judge_qa_log``.
    """
    judges = state.get("judge_agents", [])
    if not judges:
        logger.warning("[judge_question] No judge agents configured.")
        return {"pending_judge_questions": [], "judge_qa_log": []}

    # Limit Q&A to max 3 per round
    MAX_QA_PER_ROUND = 3
    qa_log = state.get("judge_qa_log", [])
    current_round = state.get("round", 0)
    qa_this_round = sum(
        1 for q in qa_log
        if q.get("round") == current_round and q.get("answer") is not None
    )
    if qa_this_round >= MAX_QA_PER_ROUND:
        logger.info(
            "[judge_question] Max Q&A (%d) reached for round %d. Skipping.",
            MAX_QA_PER_ROUND, current_round,
        )
        return {"pending_judge_questions": [], "judge_qa_log": []}

    # Check that both teams have spoken this round
    debate_log = state.get("debate_log", [])
    current_round = state.get("round", 0)
    teams_spoken_this_round: set[str] = set()
    for entry in debate_log:
        if entry.get("round") == current_round:
            teams_spoken_this_round.add(entry.get("team", ""))

    if not ("team_a" in teams_spoken_this_round and "team_b" in teams_spoken_this_round):
        logger.info(
            "[judge_question] Not both teams have spoken yet (round %d). Skipping.",
            current_round,
        )
        return {"pending_judge_questions": [], "judge_qa_log": []}

    # Load language setting
    try:
        from app.api.settings import settings_mgr
        _settings = settings_mgr.load()
        language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        language = "ko"

    logger.info(
        "[judge_question] %d judges deciding on questions (round %d).",
        len(judges), current_round,
    )

    # Run all judges in parallel
    tasks = [
        _judge_decide_question(judge, state, llm_client, language)
        for judge in judges
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect all candidate questions
    candidates: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("[judge_question] Judge question exception: %s", result)
            continue
        if result is not None:
            candidates.append(result)

    logger.info(
        "[judge_question] %d candidate questions from judges.", len(candidates)
    )

    # Select the best question (1 per round)
    selected: dict | None = None
    if len(candidates) == 0:
        selected = None
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        # Multiple candidates — LLM selects the most impactful one
        try:
            selection_system = (
                "You are selecting the most impactful question for a legal debate.\n\n"
                "Selection criteria (in priority order):\n"
                "1. SPECIFICITY: Prefer questions that target specific claims, evidence, "
                "or legal reasoning over general/abstract questions.\n"
                "2. IMPACT: Prefer questions that could change the direction of the debate "
                "or expose a critical weakness in an argument.\n"
                "3. EVIDENCE-BASED: Prefer questions that reference specific cited evidence "
                "and ask about its application or relevance.\n"
                "4. DIVERSITY: If two questions are equally strong, prefer the one from "
                "a judge who hasn't asked recently.\n"
                "5. AVOID REPETITION: Do not select a question that has already been "
                "asked in a previous round.\n\n"
                "Reply with ONLY the number of your selection."
            )
            # Add this round's Q&A history to selection context
            round_qa = [q for q in qa_log if q.get("round") == current_round]
            qa_hist_for_selection = ""
            if round_qa:
                qa_hist_for_selection = "\nQuestions already asked this round (DO NOT select similar ones):\n"
                for q in round_qa:
                    qa_hist_for_selection += f"- Q: {q.get('question', '')[:150]}\n"
                    if q.get("answer"):
                        qa_hist_for_selection += f"  A: {q.get('answer', '')[:100]}\n"
                qa_hist_for_selection += "\nIf ALL candidates are too similar to already-asked questions, reply with 0.\n"

            selection_user = f"Select the best question:{qa_hist_for_selection}\n\nCandidates:\n"
            for i, c in enumerate(candidates):
                selection_user += f"{i+1}. [{c['judge_name']}] → {c.get('target_agent_id', '?')}: {c['question']}\n"

            chosen = await llm_client.achat(
                messages=[
                    {"role": "system", "content": selection_system},
                    {"role": "user", "content": selection_user},
                ],
                temperature=0.1,
                max_tokens=5,
            )
            idx = int(chosen.strip().replace(".", "").replace(",", "")) - 1
            if idx < 0:
                # LLM determined all candidates are too similar to existing questions
                logger.info("[judge_question] All candidates too similar to existing questions. Skipping.")
                return {"pending_judge_questions": [], "judge_qa_log": []}
            selected = candidates[max(0, min(idx, len(candidates) - 1))]
            logger.info(
                "[judge_question] Selected question %d/%d from %s.",
                idx + 1, len(candidates), selected.get("judge_name", "?"),
            )
        except Exception as exc:
            logger.warning("[judge_question] Selection failed (%s), using first candidate.", exc)
            selected = candidates[0]

    # Build return
    questions_list: list[dict] = []
    qa_entries: list[dict] = []
    if selected:
        questions_list.append(selected)
        qa_entries.append({
            "judge_id": selected["judge_id"],
            "judge_name": selected["judge_name"],
            "target_agent_id": selected["target_agent_id"],
            "question": selected["question"],
            "answer": None,
            "round": selected["round"],
        })

    logger.info(
        "[judge_question] Final: %d question(s) selected.", len(questions_list)
    )

    return {
        "pending_judge_questions": questions_list,
        "judge_qa_log": qa_entries,
    }


# ---------------------------------------------------------------------------
# Node: agent_answer_node
# ---------------------------------------------------------------------------

def _find_agent_profile(
    agent_id: str,
    team_a_agents: list,
    team_b_agents: list,
) -> tuple[dict | None, str]:
    """Find an agent profile by id and return (profile, team)."""
    for agent in team_a_agents:
        if agent.get("agent_id") == agent_id:
            return agent, "team_a"
    for agent in team_b_agents:
        if agent.get("agent_id") == agent_id:
            return agent, "team_b"
    return None, ""


async def agent_answer_node(
    state: DebateState,
    llm_client: LLMClient,
) -> dict:
    """
    Answer the first pending judge question.

    Finds the targeted agent, generates an answer using the agent's
    debater profile, and records both the question and answer in the
    ``debate_log`` and ``judge_qa_log``.

    Args:
        state: Current DebateState with pending_judge_questions.
        llm_client: LLM client for the agent answer.

    Returns:
        Partial state update with ``pending_judge_questions``,
        ``judge_qa_log``, and ``debate_log``.
    """
    pending = list(state.get("pending_judge_questions", []))
    if not pending:
        logger.info("[agent_answer] No pending questions.")
        return {
            "pending_judge_questions": [],
            "judge_qa_log": [],
            "debate_log": [],
        }

    # Take the first pending question
    question_entry = pending[0]
    remaining = pending[1:]

    judge_id = question_entry.get("judge_id", "unknown")
    judge_name = question_entry.get("judge_name", "Judge")
    target_agent_id = question_entry.get("target_agent_id", "")
    question_text = question_entry.get("question", "")
    current_round = question_entry.get("round", state.get("round", 0))

    team_a_agents = state.get("team_a_agents", [])
    team_b_agents = state.get("team_b_agents", [])

    agent_profile, agent_team = _find_agent_profile(
        target_agent_id, team_a_agents, team_b_agents
    )

    if agent_profile is None:
        logger.error(
            "[agent_answer] Target agent %s not found. Skipping.", target_agent_id
        )
        return {
            "pending_judge_questions": remaining,
            "judge_qa_log": [],
            "debate_log": [],
        }

    agent_name = agent_profile.get("name", "Agent")
    team_opinion = (
        state.get("opinion_a", "") if agent_team == "team_a"
        else state.get("opinion_b", "")
    )

    # Load language setting
    try:
        from app.api.settings import settings_mgr
        _settings = settings_mgr.load()
        language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        language = "ko"

    # Build team evidence summary for the answering agent
    _all_ev = state.get("all_evidences", [])
    _team_ev = [e for e in _all_ev if isinstance(e, dict) and e.get("submitted_by") == agent_team]
    _ev_lines: list[str] = []
    for _e in _team_ev[:20]:
        _cn = _e.get("case_number", _e.get("case_id", ""))
        _ln = _e.get("law_name", "")
        _det = _e.get("source_detail", "")[:100]
        if _cn:
            _ev_lines.append(f"- [판례] {_cn}: {_det}")
        elif _ln:
            _ev_lines.append(f"- [법령] {_ln}: {_det}")
    _team_ev_summary = "\n".join(_ev_lines) if _ev_lines else "- (no verified evidence available)"

    # Build analysis summary
    _situation = state.get("situation_brief", "")
    _topic = state.get("topic", "")
    _key_issues = state.get("key_issues", [])
    _analysis_parts = [f"Topic: {_topic}"]
    if _key_issues:
        _analysis_parts.append("Key issues: " + ", ".join(str(i) for i in _key_issues[:5]))
    _analysis_summary = "\n".join(_analysis_parts)

    # Build opponent's cited evidence summary
    _opp_ev = [e for e in _all_ev if isinstance(e, dict) and e.get("submitted_by") != agent_team]
    _opp_lines: list[str] = []
    _opp_seen: set[str] = set()
    for _e in _opp_ev[:20]:
        _cn = _e.get("case_number", _e.get("case_id", ""))
        _ln = _e.get("law_name", "")
        _det = _e.get("source_detail", "")[:80]
        _key = _cn or _ln or _det
        if _key and _key not in _opp_seen:
            _opp_seen.add(_key)
            if _cn:
                _opp_lines.append(f"- [판례] {_cn}: {_det}")
            elif _ln:
                _opp_lines.append(f"- [법령] {_ln}: {_det}")
    _opp_summary = "\n".join(_opp_lines) if _opp_lines else "- (none)"

    # Build blacklist section
    _blacklist = state.get("blacklisted_evidence", [])
    _blacklist_section = ""
    if _blacklist:
        _blacklist_section = (
            "\n## BLACKLISTED EVIDENCE (ABSOLUTELY DO NOT CITE)\n"
            "Your team identified these as IRRELEVANT during internal discussion. "
            "You MUST NOT cite any of these under any circumstances:\n"
            + "\n".join(f"- {item}" for item in _blacklist) + "\n\n"
        )

    system_msg = SIMULATION_FRAME_ADVOCATE + _AGENT_ANSWER_PROMPT.format(
        agent_name=agent_name,
        team_id=agent_team,
        team_opinion=team_opinion,
        situation_brief=_situation,
        analysis_summary=_analysis_summary,
        team_evidence_summary=_team_ev_summary,
        opponent_cited_summary=_opp_summary,
        question=question_text,
    ) + _blacklist_section + get_language_instruction(language)

    user_msg = (
        f"Judge {judge_name} asks:\n{question_text}\n\n"
        f"Provide your answer."
    )

    try:
        answer_text = await llm_client.achat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=1500,
        )
    except Exception as exc:
        logger.error(
            "[agent_answer] Agent %s answer error: %s", agent_name, exc
        )
        answer_text = f"(Answer error for agent {agent_name}: {exc})"

    logger.info(
        "[agent_answer] Agent %s answered judge %s's question.",
        agent_name, judge_name,
    )

    # Build log entries
    answer_entry = {
        "judge_id": judge_id,
        "judge_name": judge_name,
        "target_agent_id": target_agent_id,
        "target_agent_name": agent_name,
        "question": question_text,
        "answer": answer_text,
        "round": current_round,
    }

    q_log_entry = {
        "team": "judge",
        "speaker": judge_id,
        "statement": question_text,
        "round": current_round,
        "entry_type": "judge_question",
    }

    a_log_entry = {
        "team": agent_team,
        "speaker": target_agent_id,
        "statement": answer_text,
        "round": current_round,
        "entry_type": "qa_answer",
    }

    return {
        "pending_judge_questions": remaining,
        "judge_qa_log": [answer_entry],
        "debate_log": [q_log_entry, a_log_entry],
    }
