"""
Discussion subgraph node — conducts internal team deliberation.

Implements the true multi-agent discussion pipeline:
  Phase 2: Independent analysis (each agent sees only own search results + memory)
  Phase 3: Team discussion via _conduct_discussion() (shared board)
  Phase 4: Memory extraction (each agent summarizes key insights for next round)

Per-agent context isolation:
  - Own search results: visible only to that agent
  - Own memories: visible only to that agent
  - Own analysis: visible only to that agent
  - Shared board (discussion messages): visible to all
  - Shared evidence pool: visible to all
"""

from __future__ import annotations

import re as _re

from app.graph.state import TeamState
from app.graph.nodes.team_speak import _conduct_discussion
from app.agents.language import get_language_instruction
from app.rag.legal_api import LegalAPIClient
from app.utils.llm_client import LLMClient
from app.utils.logger import logger


# ------------------------------------------------------------------
# Phase 2: Independent analysis per agent
# ------------------------------------------------------------------

async def _run_independent_analyses(
    members: list[dict],
    assignments: list[dict],
    agent_search_results: dict[str, list],
    agent_memories: dict[str, list],
    situation_brief: str,
    team_opinion: str,
    opponent_stmt: str,
    topic: str,
    current_round: int,
    llm_client: LLMClient,
    language: str = "ko",
    team_id: str = "team_a",
    extra_evidence: list[dict] | None = None,
) -> dict[str, str]:
    """
    Each agent independently analyzes the situation using ONLY their own
    search results and memories. Other agents' data is NOT visible.

    Returns: {agent_id: analysis_text}
    """
    independent_analyses: dict[str, str] = {}

    for agent in members:
        agent_id = agent.get("agent_id", "")
        agent_name = agent.get("name", "Unknown")
        persona = agent.get("persona", agent.get("system_prompt", ""))

        # Find agent's assignment
        agent_angle = ""
        for a in assignments:
            if a.get("agent_id") == agent_id:
                angles = a.get("argument_angle", [])
                if isinstance(angles, list) and angles:
                    agent_angle = ", ".join(angles)
                elif isinstance(angles, str):
                    agent_angle = angles
                break

        # Build agent's PRIVATE context
        own_results = agent_search_results.get(agent_id, [])
        own_memories = agent_memories.get(agent_id, [])

        # Format own search results (truncated)
        results_text = ""
        if own_results:
            result_lines = []
            for r in own_results[:5]:
                cn = r.get("case_number", "")
                ln = r.get("law_name", "")
                title = r.get("title", "")
                content = str(r.get("content", ""))[:300]
                label = cn or ln or title or "unknown"
                result_lines.append(f"- [{label}]: {content}")
            results_text = "\n".join(result_lines)

        # Format own memories from previous rounds
        memory_text = ""
        if own_memories:
            mem_lines = []
            for m in own_memories[-5:]:  # Last 5 memories
                rnd = m.get("round", "?")
                content = m.get("content", "")
                mem_lines.append(f"Round {rnd}: {content}")
            memory_text = "\n".join(mem_lines)

        # Frame situation differently per team to prevent position drift
        if team_id != "team_a":
            _situation_header = "== SITUATION (OPPOSING PARTY'S ACCOUNT — counter this, do NOT sympathize) =="
        else:
            _situation_header = "== SITUATION (YOUR CLIENT'S ACCOUNT — argue FOR this) =="

        prompt = (
            f"You are {agent_name}, a legal professional.\n"
            f"Specialty: {persona}\n"
            f"Your assigned angle: {agent_angle}\n\n"
            f"{_situation_header}\n{situation_brief[:3000]}\n\n"
            f"== YOUR TEAM'S POSITION (you MUST argue FOR this) ==\n{team_opinion}\n\n"
            f"== OPPONENT'S LATEST ARGUMENT (you MUST argue AGAINST this) ==\n{opponent_stmt[:2000]}\n\n"
        )

        if extra_evidence:
            _type_labels = {"hint": "Hint", "evidence": "Evidence"}
            _extra_lines = [
                f"- [{_type_labels.get(e.get('type', 'hint'), e.get('type', 'hint'))}] {e.get('content', '')}"
                for e in extra_evidence
            ]
            prompt += (
                f"== ⚡ USER INTERVENTIONS (address these FIRST) ==\n"
                + "\n".join(_extra_lines) + "\n\n"
            )

        if results_text:
            prompt += f"== YOUR SEARCH RESULTS (private to you) ==\n{results_text}\n\n"

        if memory_text:
            prompt += f"== YOUR MEMORY FROM PREVIOUS ROUNDS ==\n{memory_text}\n\n"

        prompt += (
            f"Independently analyze this situation from YOUR unique perspective.\n"
            f"Focus on:\n"
            f"1. Key legal arguments from YOUR specialty angle\n"
            f"2. Strengths/weaknesses of evidence YOU found\n"
            f"3. Strategy recommendations based on YOUR analysis\n"
            f"4. What the opponent is likely to argue next\n\n"
            f"Be concise (2-3 paragraphs). This is YOUR independent analysis "
            f"before team discussion — maintain your unique perspective."
        )
        prompt += get_language_instruction(language)

        try:
            _agent_model = agent.get("llm_override") or None
            analysis = await llm_client.achat(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2000,
                model_override=_agent_model,
            )
            independent_analyses[agent_id] = analysis
            logger.info(
                "[discuss_node] Independent analysis from %s: %d chars",
                agent_name, len(analysis),
            )
        except Exception as exc:
            logger.warning("[discuss_node] Independent analysis failed for %s: %s", agent_name, exc)
            independent_analyses[agent_id] = ""

    return independent_analyses


# ------------------------------------------------------------------
# Phase 4: Memory extraction per agent
# ------------------------------------------------------------------

async def _extract_agent_memories(
    members: list[dict],
    discussion_log: list[dict],
    current_round: int,
    llm_client: LLMClient,
    language: str = "ko",
) -> dict[str, list]:
    """
    After discussion, each agent extracts key insights to remember
    for the next round.

    Returns: {agent_id: [new_memory_entry, ...]}
    """
    new_memories: dict[str, list] = {}

    for agent in members:
        agent_id = agent.get("agent_id", "")
        agent_name = agent.get("name", "Unknown")

        # Collect this agent's messages from discussion
        agent_msgs = [m for m in discussion_log if m.get("agent_id") == agent_id]
        if not agent_msgs:
            new_memories[agent_id] = []
            continue

        # Build context of what this agent said and what others said
        discussion_summary = ""
        for msg in discussion_log[-10:]:  # Last 10 messages
            speaker = msg.get("speaker", "?")
            content = msg.get("content", "")[:300]
            discussion_summary += f"{speaker}: {content}\n"

        prompt = (
            f"You are {agent_name}. You just participated in a team legal debate discussion.\n\n"
            f"Discussion summary:\n{discussion_summary}\n\n"
            f"Extract 2-3 KEY INSIGHTS you want to remember for the next round:\n"
            f"- Opponent weaknesses you discovered\n"
            f"- Evidence strategies that worked or failed\n"
            f"- Important facts or contradictions noted\n"
            f"- Strategic recommendations for next round\n\n"
            f"Format each insight as a single concise sentence."
        )
        prompt += get_language_instruction(language)

        try:
            _agent_model = agent.get("llm_override") or None
            memory_text = await llm_client.achat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                model_override=_agent_model,
            )
            new_memories[agent_id] = [{
                "round": current_round,
                "type": "insight",
                "content": memory_text,
            }]
            logger.info("[discuss_node] Memory extracted for %s: %d chars", agent_name, len(memory_text))
        except Exception as exc:
            logger.warning("[discuss_node] Memory extraction failed for %s: %s", agent_name, exc)
            new_memories[agent_id] = []

    return new_memories


# ------------------------------------------------------------------
# Main discuss_node
# ------------------------------------------------------------------

async def discuss_node(
    state: TeamState,
    llm_client: LLMClient,
    legal_api: LegalAPIClient | None = None,
) -> dict:
    """
    Subgraph node: conduct internal team discussion with multi-agent features.

    Pipeline:
      Phase 2 — Independent analysis (each agent sees only own data)
      Phase 3 — Team discussion (shared board, _conduct_discussion)
      Phase 4 — Memory extraction (each agent summarizes insights)

    Returns partial TeamState update.
    """
    # Pause check before discussion
    _debate_id = state.get("debate_id", "")
    if _debate_id:
        from app.graph.pause_check import check_pause
        await check_pause(_debate_id)

    team_id = state.get("team_id", "unknown")
    members = state.get("members", [])
    if not members:
        logger.warning("[discuss_node] No team members for %s — skipping discussion.", team_id)
        return {
            "internal_discussion": [],
            "discussion_log": [],
            "accepted_cases": [],
            "blacklisted_items": [],
            "blacklisted_evidence": [],
            "discussed_cases": [],
            "agreed_strategy": "",
            "analysis_summary": "",
            "agent_memories": state.get("agent_memories", {}),
            "_opponent_cited_summary": "",
        }
    assignments = state.get("role_assignments", [])
    all_search_results = state.get("search_results", [])
    case_id_map = state.get("case_id_map", {})
    current_round = state.get("round", 1)
    agent_search_results = state.get("agent_search_results", {})
    agent_memories = state.get("agent_memories", {})

    # Merge user interventions (extra_evidence) into search results
    extra_evidence = state.get("extra_evidence", [])
    if extra_evidence:
        for ev in extra_evidence:
            all_search_results.append({
                "content": ev.get("content", ""),
                "source_type": "user_injected",
                "source_detail": ev.get("source_detail", "User intervention"),
                "type": ev.get("type", "hint"),
                "round": ev.get("round", 0),
                "case_number": "",
                "law_name": "",
                "title": f"[User {ev.get('type', 'hint')}] {ev.get('content', '')[:50]}",
            })
        logger.info(
            "[discuss_node] Merged %d user interventions into search results",
            len(extra_evidence),
        )

    # Load settings
    from app.api.settings import settings_mgr
    try:
        _settings = settings_mgr.load()
        language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        language = "ko"

    # Load team display names
    team_a_name = state.get("team_a_name", "Team A")
    team_b_name = state.get("team_b_name", "Team B")
    team_display_name = team_a_name if team_id == "team_a" else team_b_name

    # Build analysis summary
    _analysis_parts = [f"Topic: {state.get('topic', '')}"]
    _ki = state.get("key_issues", [])
    if _ki:
        _analysis_parts.append("Key issues: " + ", ".join(str(i) for i in _ki[:5]))
    _analysis_summary = "\n".join(_analysis_parts)

    # Extract opponent's cited evidence
    _opp_cited = []
    for _entry in state.get("debate_context", []):
        if _entry.get("team") != team_id:
            for _ev in _entry.get("evidence", []):
                if isinstance(_ev, dict):
                    _opp_cited.append(_ev)
    _opp_lines: list[str] = []
    _opp_seen: set[str] = set()
    for _ev in _opp_cited:
        _cn = _ev.get("case_number", _ev.get("case_id", ""))
        _ln = _ev.get("law_name", "")
        _det = _ev.get("source_detail", "")[:80]
        _key = _cn or _ln or _det
        if _key and _key not in _opp_seen:
            _opp_seen.add(_key)
            if _cn:
                _opp_lines.append(f"- [판례] {_cn}: {_det}")
            elif _ln:
                _opp_lines.append(f"- [법령] {_ln}: {_det}")
    _opp_summary = "\n".join(_opp_lines) if _opp_lines else ""

    # Extract our previous statement
    _our_prev_stmt = ""
    for _entry in reversed(state.get("debate_context", [])):
        if (_entry.get("team") == team_id
                and _entry.get("entry_type") not in ("judge_question", "qa_answer")):
            _our_prev_stmt = _entry.get("statement", _entry.get("content", ""))[:1500]
            break

    # ================================================================
    # Phase 2: Independent analysis (each agent sees ONLY own data)
    # ================================================================
    independent_analyses = await _run_independent_analyses(
        members=members,
        assignments=assignments,
        agent_search_results=agent_search_results,
        agent_memories=agent_memories,
        situation_brief=state.get("situation_brief", ""),
        team_opinion=state.get("team_opinion", ""),
        opponent_stmt=state.get("opponent_statement", "(none)"),
        topic=state.get("topic", ""),
        current_round=current_round,
        llm_client=llm_client,
        language=language,
        team_id=team_id,
        extra_evidence=extra_evidence,
    )

    # ================================================================
    # Phase 3: Team discussion (delegates to existing _conduct_discussion)
    # Independent analyses are injected per-agent via the members list
    # ================================================================

    # Inject independent analyses into member dicts for _conduct_discussion
    # _conduct_discussion reads agent's system_prompt — we append analysis there
    enriched_members = []
    for m in members:
        m_copy = dict(m)
        aid = m_copy.get("agent_id", "")
        own_analysis = independent_analyses.get(aid, "")
        if own_analysis:
            # Store as a field that _conduct_discussion can read
            m_copy["_independent_analysis"] = own_analysis
        enriched_members.append(m_copy)

    # Cross-round REVIEW_MORE / ACCEPT state — loaded from TeamState, mutated in-place
    # by _conduct_discussion, written back after the call.
    _prior_rm_raw = state.get("review_more_persist", {}) or {}
    _prior_accept_raw = state.get("accept_votes_persist", {}) or {}
    prior_review_more_agents: dict[str, set] = {
        str(_cn): set(_aids) for _cn, _aids in _prior_rm_raw.items()
    }
    prior_accept_votes: dict[str, set] = {
        str(_cn): set(_aids) for _cn, _aids in _prior_accept_raw.items()
    }

    discussion_log, accepted_cases = await _conduct_discussion(
        members=enriched_members,
        search_results=all_search_results,
        team_opinion=state.get("team_opinion", ""),
        opponent_stmt=state.get("opponent_statement", "(none)"),
        llm_client=llm_client,
        language=language,
        debate_id=state.get("debate_id", ""),
        team_id=team_id,
        judge_qa_log=state.get("judge_qa_log", []),
        assignments=assignments,
        situation_brief=state.get("situation_brief", ""),
        analysis_summary=_analysis_summary,
        opponent_cited_summary=_opp_summary,
        team_display_name=team_display_name,
        case_id_map=case_id_map,
        legal_api=legal_api,
        topic=state.get("topic", ""),
        our_prev_stmt=_our_prev_stmt,
        judge_improvement_feedback=state.get("judge_improvement_feedback", ""),
        extra_evidence=extra_evidence if extra_evidence else None,
        prior_review_more_agents=prior_review_more_agents,
        prior_accept_votes=prior_accept_votes,
    )

    # Serialize cross-round state back (sets → sorted lists for JSON/checkpointer)
    updated_review_more_persist = {
        _cn: sorted(_aids) for _cn, _aids in prior_review_more_agents.items()
    }
    updated_accept_votes_persist = {
        _cn: sorted(_aids) for _cn, _aids in prior_accept_votes.items()
    }

    # ================================================================
    # Phase 4: Memory extraction (each agent summarizes insights)
    # ================================================================
    new_memories = await _extract_agent_memories(
        members=members,
        discussion_log=discussion_log,
        current_round=current_round,
        llm_client=llm_client,
        language=language,
    )

    # Merge new memories with existing
    updated_memories = dict(agent_memories)
    for aid, mems in new_memories.items():
        if aid not in updated_memories:
            updated_memories[aid] = []
        updated_memories[aid].extend(mems)
        # Keep at most 10 memories per agent (oldest trimmed)
        if len(updated_memories[aid]) > 10:
            updated_memories[aid] = updated_memories[aid][-10:]

    # --- Parse blacklist from discussion ---
    blacklisted_items: list[str] = []
    for entry in discussion_log:
        content = entry.get("content", "")
        if "FINAL_BLACKLIST:" in content:
            bl_line = content.split("FINAL_BLACKLIST:")[-1].strip()
            blacklisted_items.extend(
                item.strip().strip("[]") for item in bl_line.split(",") if item.strip()
            )
        elif "BLACKLIST:" in content and "FINAL_BLACKLIST" not in content:
            bl_line = content.split("BLACKLIST:")[-1].split("—")[0].strip()
            if bl_line:
                blacklisted_items.append(bl_line.strip().strip("[]"))
    blacklisted_items = list(set(blacklisted_items))
    if blacklisted_items:
        logger.info("[discuss_node] Blacklisted evidence: %s", blacklisted_items)

    # Parse rejected precedents
    rejected_cases: set[str] = set()
    for entry in discussion_log:
        content = entry.get("content", "")
        for _m in _re.findall(r'\[REJECT:\s*([^\]]+)\]', content):
            rejected_cases.add(_m.strip())
    if rejected_cases:
        logger.info("[discuss_node] Rejected precedents: %s", rejected_cases)
        blacklisted_items.extend(rejected_cases)
        blacklisted_items = list(set(blacklisted_items))

    # Parse discussed/reviewed cases
    discussed_cases: set[str] = set()
    for entry in discussion_log:
        _dc_content = entry.get("content", "")
        for _m in _re.findall(r'\[판례:\s*([^\]]+)\]', _dc_content):
            discussed_cases.add(_m.strip())
        for _m in _re.findall(r'\[법령:\s*([^\]]+)\]', _dc_content):
            discussed_cases.add(_m.strip())
        for _tag in ['ACCEPT', 'REJECT', 'BLACKLIST', 'REVIEW_MORE']:
            for _m in _re.findall(rf'\[{_tag}:\s*([^\]]+)\]', _dc_content):
                discussed_cases.add(_m.strip())
    if discussed_cases:
        logger.info("[discuss_node] Discussed cases: %d items", len(discussed_cases))

    # --- Generate consensus strategy ---
    _team_opinion = state.get("team_opinion", "")
    consensus_prompt = (
        "Based on the team discussion below, summarize the agreed strategy "
        "in 1-2 sentences. What is the team's main argument and which "
        "evidence will be cited?\n\n"
    )
    for entry in discussion_log:
        consensus_prompt += f"{entry['speaker']}: {entry['content']}\n"

    _consensus_system = (
        f"You are a debate team coordinator for {team_display_name}.\n"
        f"Your team's core position: {_team_opinion}\n\n"
        f"Summarize the agreed strategy that SUPPORTS this position.\n"
        f"The strategy must argue FOR your team's position, not against it."
        + get_language_instruction(language)
    )
    try:
        agreed_strategy = await llm_client.achat(
            messages=[
                {"role": "system", "content": _consensus_system},
                {"role": "user", "content": consensus_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
    except Exception as exc:
        logger.error("[discuss_node] Consensus generation error: %s", exc)
        agreed_strategy = state.get("agreed_strategy", "") or "Proceed with available evidence."

    logger.info(
        "[discuss_node] Discussion complete for %s: %d messages, strategy: %s",
        team_id, len(discussion_log), agreed_strategy[:80],
    )

    return {
        "internal_discussion": discussion_log,
        "discussion_log": discussion_log,
        "accepted_cases": list(accepted_cases),
        "blacklisted_items": blacklisted_items,
        "blacklisted_evidence": blacklisted_items,
        "discussed_cases": list(discussed_cases),
        "agreed_strategy": agreed_strategy,
        "analysis_summary": _analysis_summary,
        "agent_memories": updated_memories,
        # Cross-round REVIEW_MORE / ACCEPT persistence
        "review_more_persist": updated_review_more_persist,
        "accept_votes_persist": updated_accept_votes_persist,
        # Store opponent summary for statement_node
        "_opponent_cited_summary": _opp_summary,
    }
