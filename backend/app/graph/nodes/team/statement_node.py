"""
Statement subgraph node — produces the final representative statement.

Wraps _produce_statement() from team_speak.py. Selects the primary
speaker, generates the debate statement with citations, and builds
the evidence list.
"""

from __future__ import annotations

from app.graph.state import TeamState
from app.graph.nodes.team_speak import _produce_statement
from app.utils.llm_client import LLMClient
from app.utils.logger import logger


async def statement_node(
    state: TeamState,
    llm_client: LLMClient,
) -> dict:
    """
    Subgraph node: produce the final representative statement.

    Phase 3 of the team subgraph pipeline. Uses the discussion results,
    agreed strategy, and search results to generate the team's official
    debate statement.

    Returns partial TeamState update with output_statement, output_evidence,
    and selected_speaker.
    """
    team_id = state.get("team_id", "unknown")

    # Load settings
    from app.api.settings import settings_mgr
    try:
        _settings = settings_mgr.load()
        language = _settings.get("debate", {}).get("language", "ko")
    except Exception:
        language = "ko"

    # Load team display name
    team_a_name = state.get("team_a_name", "Team A")
    team_b_name = state.get("team_b_name", "Team B")
    team_display_name = team_a_name if team_id == "team_a" else team_b_name

    # Pause check before expensive statement generation
    _debate_id = state.get("debate_id", "")
    if _debate_id:
        from app.graph.pause_check import check_pause
        await check_pause(_debate_id)

    # Get discussion results
    discussion_log = state.get("discussion_log", state.get("internal_discussion", []))
    accepted_cases = set(state.get("accepted_cases", []))
    blacklisted_items = state.get("blacklisted_items", [])
    discussed_cases = set(state.get("discussed_cases", []))
    agreed_strategy = state.get("agreed_strategy", "")
    all_search_results = state.get("search_results", [])

    # Build updated state for _produce_statement
    updated_state = dict(state)
    updated_state["agreed_strategy"] = agreed_strategy
    # Pass through opponent cited summary if available
    _opp_summary = state.get("_opponent_cited_summary", "")
    updated_state["_opponent_cited_summary"] = _opp_summary

    selected_speaker, statement, evidence_list = await _produce_statement(
        state=updated_state,
        search_results=all_search_results,
        discussion_log=discussion_log,
        agreed_strategy=agreed_strategy,
        llm_client=llm_client,
        language=language,
        team_display_name=team_display_name,
        blacklisted_items=blacklisted_items,
        discussed_cases=discussed_cases,
        accepted_cases=accepted_cases,
    )

    logger.info(
        "[statement_node] Team %s complete. Speaker: %s, Statement: %d chars, Evidence: %d.",
        team_id, selected_speaker, len(statement), len(evidence_list),
    )

    return {
        "output_statement": statement,
        "output_evidence": evidence_list,
        "selected_speaker": selected_speaker,
    }
