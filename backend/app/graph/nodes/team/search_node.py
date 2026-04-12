"""
Search subgraph node — executes per-agent evidence searches.

Wraps the search orchestration logic from team_speak_node Phase 1.
Each agent searches independently; results are stored per-agent (private)
in agent_search_results, and a shared_evidence_pool carries over from
previous rounds.
"""

from __future__ import annotations

import asyncio
import re as _re

from app.graph.state import TeamState
from app.graph.nodes.team_speak import (
    _execute_agent_search,
    _llm_relevance_filter,
    _build_case_id_map,
)
from app.rag.legal_api import LegalAPIClient
from app.rag.searcher import Searcher
from app.utils.llm_client import LLMClient
from app.utils.logger import logger


async def search_node(
    state: TeamState,
    llm_client: LLMClient,
    searcher: Searcher | None = None,
    legal_api: LegalAPIClient | None = None,
) -> dict:
    """
    Subgraph node: execute per-agent evidence searches.

    Phase 1 of the team subgraph pipeline. Each agent searches based on
    their role assignment. Results are stored privately per agent in
    ``agent_search_results``, while shared carry-over evidence goes to
    ``shared_evidence_pool``.

    Returns partial TeamState update.
    """
    # Pause check before expensive search operations
    _debate_id = state.get("debate_id", "")
    if _debate_id:
        from app.graph.pause_check import check_pause
        await check_pause(_debate_id)

    team_id = state.get("team_id", "unknown")
    members = state.get("members", [])
    assignments = state.get("role_assignments", [])

    # Load settings
    from app.api.settings import settings_mgr
    try:
        _settings = settings_mgr.load()
        _debate_cfg = _settings.get("debate", {})
        max_tool_rounds = _debate_cfg.get("max_api_calls_per_round", 10)
        language = _debate_cfg.get("language", "ko")
    except Exception:
        max_tool_rounds = 10
        language = "ko"

    logger.info(
        "[search_node] Starting searches for %s (%d members, max_tool_rounds=%d).",
        team_id, len(members), max_tool_rounds,
    )

    # --- Re-verify previously unverified citations ---
    _unverified_pat = _re.compile(r'\[⚠ 미확인 인용:\s*([^\]]+)\]')
    _all_evidence = state.get("all_evidences", [])
    _verified_ids: set[str] = set()
    for _ev in _all_evidence:
        if isinstance(_ev, dict):
            for _key in ('case_number', 'case_id', 'law_name', 'title', 'evidence_id'):
                _val = _ev.get(_key, '')
                if _val and not (len(str(_val)) == 36 and '-' in str(_val)):
                    _verified_ids.add(str(_val).strip())

    _debate_log = state.get("debate_context", [])
    _log_updated = False
    for _entry in _debate_log:
        _stmt = _entry.get("statement", "")
        for _m in list(_unverified_pat.finditer(_stmt)):
            _cited = _m.group(1).strip()
            if _cited in _verified_ids:
                _ctype = "판례" if any(c in _cited for c in "다가나합") else "법령"
                _entry["statement"] = _stmt.replace(_m.group(0), f"[{_ctype}: {_cited}]")
                _log_updated = True
                logger.info("[search_node] Previously unverified '%s' now verified", _cited)

    _unverified_for_prompt: list[str] = []
    for _entry in _debate_log:
        _stmt = _entry.get("statement", "")
        for _m in _unverified_pat.finditer(_stmt):
            _cited = _m.group(1).strip()
            _spk = _entry.get("speaker", "?")
            _rnd = _entry.get("round", "?")
            _unverified_for_prompt.append(f"- {_cited} (Round {_rnd}, {_spk})")

    if _log_updated:
        _debate_id = state.get("debate_id", "")
        if _debate_id:
            try:
                from app.api.debate import DebateStore
                await DebateStore.asave(_debate_id, state)
            except Exception:
                pass

    # --- Build per-agent search tasks ---
    search_tasks = []
    agent_task_map: list[str] = []  # parallel list: agent_id per task
    for member in members:
        agent_id = member.get("agent_id", "")
        agent_assignment = None
        for a in assignments:
            if a.get("agent_id") == agent_id:
                agent_assignment = a
                break
        if agent_assignment is None:
            agent_assignment = {
                "agent_id": agent_id,
                "task": "Search for general supporting evidence",
                "search_type": "document",
                "priority": "supporting",
            }

        search_tasks.append(
            _execute_agent_search(
                agent=member,
                assignment=agent_assignment,
                state=state,
                llm_client=llm_client,
                searcher=searcher,
                legal_api=legal_api,
                max_tool_rounds=max_tool_rounds,
                language=language,
                unverified_citations=_unverified_for_prompt,
            )
        )
        agent_task_map.append(agent_id)

    # --- Carry over previous round results (shared pool) ---
    _prev_search = state.get("search_results", [])
    _prev_blacklist = set(state.get("blacklisted_evidence", []))
    carried_results: list[dict] = []
    existing_keys: set[str] = set()
    for sr in _prev_search:
        if not isinstance(sr, dict):
            continue
        cn = sr.get("case_number", sr.get("case_id", ""))
        ln = sr.get("law_name", "")
        key = cn or ln
        if key and key in _prev_blacklist:
            continue
        carried_results.append(sr)
        if key:
            existing_keys.add(key)

    # --- Run agent searches in batches (max 3 concurrent) ---
    _SEARCH_BATCH_SIZE = 3
    agent_search_results: dict[str, list[dict]] = {}
    all_new_results: list[dict] = []

    if search_tasks:
        search_results_nested: list = []
        for _batch_start in range(0, len(search_tasks), _SEARCH_BATCH_SIZE):
            _batch = search_tasks[_batch_start:_batch_start + _SEARCH_BATCH_SIZE]
            _batch_results = await asyncio.gather(*_batch, return_exceptions=True)
            search_results_nested.extend(_batch_results)
        for idx, result in enumerate(search_results_nested):
            agent_id = agent_task_map[idx]
            agent_results: list[dict] = []
            if isinstance(result, list):
                for sr in result:
                    if not isinstance(sr, dict):
                        continue
                    cn = sr.get("case_number", sr.get("case_id", ""))
                    ln = sr.get("law_name", "")
                    key = cn or ln
                    if key and (key in _prev_blacklist or key in existing_keys):
                        continue
                    if key:
                        existing_keys.add(key)
                    agent_results.append(sr)
                    all_new_results.append(sr)
            elif isinstance(result, Exception):
                logger.error("[search_node] Agent %s search exception: %s", agent_id, result)
            agent_search_results[agent_id] = agent_results

    # Combine for backward compatibility (flat list used by statement_node)
    all_search_results = carried_results + all_new_results

    # LLM relevance filter
    if all_search_results:
        all_search_results = await _llm_relevance_filter(
            items=all_search_results,
            topic=state.get("topic", ""),
            team_opinion=state.get("team_opinion", ""),
            llm_client=llm_client,
            language=language,
        )

    # Track used search queries
    _used_queries = set(state.get("used_search_queries", []))
    for sr in all_new_results:
        _q = sr.get("_search_query", "")
        if _q:
            _used_queries.add(_q)

    # Build case_id_map
    case_id_map = _build_case_id_map(all_search_results) if all_search_results else {}
    if case_id_map:
        logger.info("[search_node] Built case_id_map with %d entries.", len(case_id_map))

    # --- Ingest new results into the knowledge graph for cross-round accumulation ---
    _debate_id = state.get("debate_id", "")
    if searcher is not None and all_new_results and _debate_id and team_id:
        try:
            _gs = await searcher.get_graph_store(_debate_id, team_id)
            _n_added = _gs.ingest_search_results(all_new_results)
            if _n_added:
                logger.info(
                    "[search_node] Ingested %d new entities into graph (%s/%s).",
                    _n_added, _debate_id, team_id,
                )
        except Exception as _exc:
            logger.warning("[search_node] Graph ingestion failed: %s", _exc)

    logger.info(
        "[search_node] Search complete for %s: %d carried + %d new = %d total.",
        team_id, len(carried_results), len(all_new_results), len(all_search_results),
    )

    return {
        "search_results": all_search_results,
        "agent_search_results": agent_search_results,
        "shared_evidence_pool": carried_results,
        "used_search_queries": list(_used_queries),
        "case_id_map": case_id_map,
    }
