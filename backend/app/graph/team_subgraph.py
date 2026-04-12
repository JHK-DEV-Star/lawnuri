"""
Team internal processing subgraph for LawNuri.

A 4-node LangGraph StateGraph that orchestrates the full team pipeline:
  1. assign_roles: Analyse opponent statement and assign tasks to members
  2. search: Execute per-agent evidence searches (private results)
  3. discuss: Internal team discussion with voting and consensus
  4. statement: Produce the final representative statement with citations

Each node boundary is a checkpoint opportunity for LangGraph's
checkpointing system (Step 2).
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.nodes.assign_roles import assign_roles_node
from app.graph.nodes.team.search_node import search_node
from app.graph.nodes.team.discuss_node import discuss_node
from app.graph.nodes.team.statement_node import statement_node
from app.graph.state import TeamState
from app.rag.legal_api import LegalAPIClient
from app.rag.searcher import Searcher
from app.utils.llm_client import LLMClient
from app.utils.logger import logger


async def _assign_roles_wrapper(
    state: TeamState,
    *,
    llm_client: LLMClient,
) -> dict:
    """Bind the LLM client to assign_roles_node."""
    return await assign_roles_node(state, llm_client)


async def _search_wrapper(
    state: TeamState,
    *,
    llm_client: LLMClient,
    searcher: Searcher | None = None,
    legal_api: LegalAPIClient | None = None,
) -> dict:
    """Bind dependencies to search_node."""
    debate_id = state.get("debate_id", "")
    if debate_id:
        try:
            from app.api.debate import DebateStore
            current = await DebateStore.aload(debate_id)
            current["current_phase"] = "searching"
            await DebateStore.asave(debate_id, current)
        except Exception:
            pass
    return await search_node(state, llm_client, searcher, legal_api)


async def _discuss_wrapper(
    state: TeamState,
    *,
    llm_client: LLMClient,
    legal_api: LegalAPIClient | None = None,
) -> dict:
    """Bind dependencies to discuss_node."""
    debate_id = state.get("debate_id", "")
    if debate_id:
        try:
            from app.api.debate import DebateStore
            current = await DebateStore.aload(debate_id)
            current["current_phase"] = "discussing"
            await DebateStore.asave(debate_id, current)
        except Exception:
            pass
    return await discuss_node(state, llm_client, legal_api)


async def _statement_wrapper(
    state: TeamState,
    *,
    llm_client: LLMClient,
) -> dict:
    """Bind LLM client to statement_node."""
    debate_id = state.get("debate_id", "")
    if debate_id:
        try:
            from app.api.debate import DebateStore
            current = await DebateStore.aload(debate_id)
            current["current_phase"] = "statement"
            await DebateStore.asave(debate_id, current)
        except Exception:
            pass
    return await statement_node(state, llm_client)


def build_team_subgraph(
    llm_client: LLMClient,
    searcher: Searcher | None = None,
    legal_api: LegalAPIClient | None = None,
) -> Any:
    """
    Build and compile the 4-node team subgraph.

    Flow: START -> assign_roles -> search -> discuss -> statement -> END

    Args:
        llm_client: LLMClient instance for all LLM operations.
        searcher: Optional Searcher for document/graph search.
        legal_api: Optional LegalAPIClient for statute/precedent search.

    Returns:
        A compiled LangGraph StateGraph(TeamState).
    """
    logger.info("[team_subgraph] Building 4-node team subgraph.")

    bound_assign_roles = partial(_assign_roles_wrapper, llm_client=llm_client)
    bound_search = partial(_search_wrapper, llm_client=llm_client, searcher=searcher, legal_api=legal_api)
    bound_discuss = partial(_discuss_wrapper, llm_client=llm_client, legal_api=legal_api)
    bound_statement = partial(_statement_wrapper, llm_client=llm_client)

    graph = StateGraph(TeamState)

    graph.add_node("assign_roles", bound_assign_roles)
    graph.add_node("search", bound_search)
    graph.add_node("discuss", bound_discuss)
    graph.add_node("statement", bound_statement)

    graph.set_entry_point("assign_roles")
    graph.add_edge("assign_roles", "search")
    graph.add_edge("search", "discuss")
    graph.add_edge("discuss", "statement")
    graph.add_edge("statement", END)

    compiled = graph.compile()

    logger.info("[team_subgraph] Team subgraph compiled successfully.")
    return compiled
