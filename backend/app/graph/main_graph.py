"""
Main debate orchestration graph for LawNuri.

Builds and compiles the top-level LangGraph StateGraph that coordinates
the full debate lifecycle:

    START -> user_interrupt -> team_speak -> judge_accumulate -> route_next
                ^                                                    |
                |    (opponent_rebut / same_team_add) ---------------+
                |    (round_end) -> round_end --------+
                |                                     |
                +---- (continue) ---------------------+
                                                      |
                                        (stop) -> final_judgment -> END
                                        (paused) -> END

Each node receives an LLM client via dependency injection through
functools.partial, keeping the node functions themselves testable
with explicit parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.edges.decide_judge_qa import decide_judge_qa
from app.graph.edges.decide_next import decide_next
from app.graph.edges.should_continue import should_continue
from app.graph.nodes.final_judgment import final_judgment_node
from app.graph.nodes.judge_accumulate import judge_accumulate_node
from app.graph.nodes.judge_question import judge_question_node, agent_answer_node
from app.graph.nodes.round_end import round_end_node
from app.graph.nodes.route_next import route_next_node
from app.graph.nodes.user_interrupt import user_interrupt_node
from app.graph.state import DebateState
from app.graph.team_subgraph import build_team_subgraph
from app.rag.legal_api import LegalAPIClient
from app.rag.searcher import Searcher
from app.utils.llm_client import LLMClient
from app.utils.logger import logger



async def _user_interrupt_wrapper(state: DebateState) -> dict:
    """Thin pass-through; user_interrupt_node needs no LLM client."""
    return await user_interrupt_node(state)


async def _team_speak_wrapper(
    state: DebateState,
    *,
    llm_client: LLMClient,
    searcher: Searcher | None = None,
    legal_api: LegalAPIClient | None = None,
) -> dict:
    """
    Orchestrate the currently-speaking team's full processing pipeline.

    Extracts the relevant TeamState from the top-level DebateState,
    runs the 4-node team subgraph (assign_roles → search → discuss →
    statement), and writes the results back into the top-level state.
    """
    current_team = state.get("current_team", "team_a")
    team_agents_key = f"{current_team}_agents"
    team_state_key = f"{current_team}_state"
    other_team = "team_b" if current_team == "team_a" else "team_a"

    # Determine opinions and cautions per team
    analysis = state.get("analysis", {}) if isinstance(state.get("analysis"), dict) else {}
    if current_team == "team_a":
        team_opinion = state.get("opinion_a", "")
        opponent_opinion = state.get("opinion_b", "")
        team_cautions = analysis.get("team_a_cautions", [])
    else:
        team_opinion = state.get("opinion_b", "")
        opponent_opinion = state.get("opinion_a", "")
        team_cautions = analysis.get("team_b_cautions", [])

    # Get opponent's last statement from debate log
    opponent_statement = ""
    for entry in reversed(state.get("debate_log", [])):
        if entry.get("team") == other_team:
            opponent_statement = entry.get("statement", entry.get("content", ""))
            break

    # Load configurable team display names from state
    team_a_name = state.get("team_a_name", "Team A")
    team_b_name = state.get("team_b_name", "Team B")

    # Build TeamState from DebateState
    existing_team_state = state.get(team_state_key, {})
    team_state = {
        "team_id": current_team,
        "members": state.get(team_agents_key, []),
        "opponent_statement": opponent_statement,
        "debate_context": state.get("debate_log", []),
        "extra_evidence": existing_team_state.get("extra_evidence", []),
        "internal_discussion": existing_team_state.get("internal_discussion", []),
        "agreed_strategy": existing_team_state.get("agreed_strategy", ""),
        "search_results": existing_team_state.get("search_results", []),
        "used_search_queries": existing_team_state.get("used_search_queries", []),
        "role_assignments": existing_team_state.get("role_assignments", []),
        "selected_speaker": existing_team_state.get("selected_speaker", ""),
        "output_statement": "",
        "output_evidence": [],
        "round": state.get("round", 1),
        "debate_id": state.get("debate_id", ""),
        "team_opinion": team_opinion,
        "opponent_opinion": opponent_opinion,
        "team_cautions": team_cautions,
        "judge_qa_log": state.get("judge_qa_log", []),
        "team_a_name": team_a_name,
        "team_b_name": team_b_name,
        "judge_improvement_feedback": (state.get("judge_improvement_feedback") or {}).get(current_team, ""),
        # --- Fields read by subgraph nodes ---
        "all_evidences": state.get("all_evidences", []),
        "blacklisted_evidence": state.get("blacklisted_evidence", []),
        # --- New subgraph inter-node fields ---
        "situation_brief": state.get("situation_brief", ""),
        "topic": state.get("topic", ""),
        "key_issues": state.get("key_issues", []),
        "analysis_summary": "",
        "discussion_log": [],
        "accepted_cases": [],
        "blacklisted_items": [],
        "discussed_cases": [],
        "case_id_map": {},
        # --- Multi-agent fields ---
        "agent_memories": state.get("agent_memories", {}),
        "agent_search_results": {},
        "shared_evidence_pool": [],
        # --- Inter-node private fields ---
        "_opponent_cited_summary": "",
    }

    # Save phase for frontend polling
    debate_id = state.get("debate_id", "")
    if debate_id:
        state["current_phase"] = "preparing"
        state["current_team"] = current_team
        try:
            from app.api.debate import DebateStore
            await DebateStore.asave(debate_id, state)
        except Exception:
            pass

    # Snapshot timing and token usage before subgraph execution
    _round_start = datetime.now(timezone.utc)
    _tokens_before = llm_client.usage_summary

    # Run the 4-node team subgraph
    team_subgraph = build_team_subgraph(
        llm_client=llm_client,
        searcher=searcher,
        legal_api=legal_api,
    )

    from app.graph.pause_check import DebatePausedError
    try:
        team_state = await team_subgraph.ainvoke(team_state)
    except DebatePausedError:
        # Subgraph detected pause — fire interrupt so the main graph's
        # checkpointer saves state and astream yields "__interrupt__".
        logger.info("[main_graph] %s paused/stopped inside subgraph — firing interrupt.", current_team)
        from langgraph.types import interrupt
        interrupt({"reason": "paused", "debate_id": debate_id})
        # If interrupt() did not raise (version-dependent), return partial state
        # to prevent empty statement from being recorded.
        return {team_state_key: team_state}

    # Extract results from subgraph output
    statement = team_state.get("output_statement", "")
    speaker = team_state.get("selected_speaker", "unknown")
    evidence_list = team_state.get("output_evidence", [])

    # Build debate log entry (include internal_discussion for frontend display)
    _now = datetime.now(timezone.utc)
    _tokens_after = llm_client.usage_summary
    debate_log_entry = {
        "team": current_team,
        "speaker": speaker,
        "statement": statement,
        "round": state.get("round", 1),
        "evidence_count": len(evidence_list),
        "evidence": evidence_list,
        "internal_discussion": team_state.get("internal_discussion", []),
        "timestamp": _now.isoformat(),
        "elapsed_seconds": int((_now - _round_start).total_seconds()),
        "token_usage": {
            "input": _tokens_after["input_tokens"] - _tokens_before["input_tokens"],
            "output": _tokens_after["output_tokens"] - _tokens_before["output_tokens"],
            "total": _tokens_after["total_tokens"] - _tokens_before["total_tokens"],
            "calls": _tokens_after["call_count"] - _tokens_before["call_count"],
        },
    }

    # Store internal discussion separately
    internal_discussions = list(state.get("internal_discussions", []))
    internal_discussions.append({
        "team": current_team,
        "round": state.get("round", 1),
        "discussion": team_state.get("internal_discussion", []),
    })

    # Save evidence + log immediately so graph/timeline update in real-time
    if debate_id:
        state["current_phase"] = "discussing"
        try:
            from app.api.debate import DebateStore
            current_saved = await DebateStore.aload(debate_id)
            # Append evidence
            saved_evidence = current_saved.get("all_evidences", [])
            saved_evidence.extend(evidence_list)
            current_saved["all_evidences"] = saved_evidence
            # Append debate log entry
            saved_log = current_saved.get("debate_log", [])
            saved_log.append(debate_log_entry)
            current_saved["debate_log"] = saved_log
            current_saved["current_phase"] = "discussing"
            current_saved["current_round"] = state.get("round", 1)
            # Persist agent lists for graph (they may not exist in saved state yet)
            for key in ("team_a_agents", "team_b_agents", "judge_agents"):
                if key not in current_saved and key in state:
                    current_saved[key] = state.get(key, [])
            await DebateStore.asave(debate_id, current_saved)
        except Exception:
            pass

    logger.info(
        "[main_graph] Team %s produced statement (%d chars, %d evidence).",
        current_team,
        len(statement),
        len(evidence_list),
    )

    # Propagate agent memories back to DebateState for cross-round persistence
    # Merge (not replace) to preserve the other team's memories
    existing_memories = dict(state.get("agent_memories", {}))
    existing_memories.update(team_state.get("agent_memories", {}))

    # Propagate new blacklisted evidence (Annotated[list, operator.add] — append only)
    prev_blacklisted = set(str(b) for b in state.get("blacklisted_evidence", []))
    new_blacklisted = [
        b for b in team_state.get("blacklisted_items", [])
        if str(b) not in prev_blacklisted
    ]

    # Persist accumulated graph knowledge after team's turn
    if searcher is not None and debate_id:
        try:
            await searcher.save_graph(debate_id, current_team)
            await searcher.save_graph(debate_id, "common")
        except Exception:
            logger.warning("[main_graph] Graph save after %s turn failed.", current_team, exc_info=True)

    return {
        team_state_key: team_state,
        "debate_log": [debate_log_entry],       # Annotated list: appended
        "all_evidences": evidence_list,          # Annotated list: appended
        "blacklisted_evidence": new_blacklisted, # Annotated list: appended
        "internal_discussions": internal_discussions,
        "agent_memories": existing_memories,
    }


async def _judge_accumulate_wrapper(
    state: DebateState,
    *,
    llm_client: LLMClient,
) -> dict:
    """Bind LLM client to judge_accumulate_node."""
    # Save judging phase for frontend polling (lazy import)
    debate_id = state.get("debate_id", "")
    if debate_id:
        state["current_phase"] = "judging"
        state["current_team"] = ""
        try:
            from app.api.debate import DebateStore
            await DebateStore.asave(debate_id, state)
        except Exception:
            pass
    return await judge_accumulate_node(state, llm_client)


async def _route_next_wrapper(
    state: DebateState,
    *,
    llm_client: LLMClient,
) -> dict:
    """Bind LLM client to route_next_node."""
    return await route_next_node(state, llm_client)


async def _round_end_wrapper(state: DebateState, *, llm_client: LLMClient) -> dict:
    """Bind LLM client to round_end_node for improvement feedback generation."""
    return await round_end_node(state, llm_client=llm_client)


async def _final_judgment_wrapper(
    state: DebateState,
    *,
    llm_client: LLMClient,
    searcher: Searcher | None = None,
) -> dict:
    """Bind LLM client to final_judgment_node and save accumulated graphs."""
    result = await final_judgment_node(state, llm_client)

    # Persist all accumulated graph knowledge at debate end
    if searcher is not None:
        try:
            await searcher.save_all_graphs()
            logger.info("[main_graph] All graphs saved at debate end.")
        except Exception:
            logger.warning("[main_graph] Final graph save failed.", exc_info=True)

    return result


async def _judge_question_wrapper(state: DebateState, *, llm_client: LLMClient) -> dict:
    """Bind LLM client to judge_question_node."""
    return await judge_question_node(state, llm_client)

async def _agent_answer_wrapper(state: DebateState, *, llm_client: LLMClient) -> dict:
    """Bind LLM client to agent_answer_node."""
    return await agent_answer_node(state, llm_client)



def _create_llm_client(llm_config: dict, model_override: str | None = None) -> LLMClient:
    """
    Create an LLMClient from the provided configuration dict.

    The llm_config dict should contain:
      - api_key: API key for the provider
      - base_url: Base URL of the API endpoint
      - model: Default model identifier

    Args:
        llm_config: Provider configuration dict.
        model_override: Optional model to use instead of the config default.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(
        api_key=llm_config.get("api_key", ""),
        base_url=llm_config.get("base_url", ""),
        model=model_override or llm_config.get("model", "gpt-5.4-nano"),
        provider=llm_config.get("provider", ""),
        vertex_project_id=llm_config.get("vertex_project_id", ""),
        vertex_location=llm_config.get("vertex_location", "global"),
    )


def build_debate_graph(
    llm_config: dict,
    searcher: Searcher | None = None,
    legal_api: LegalAPIClient | None = None,
    checkpointer=None,
) -> Any:
    """
    Build and compile the main debate LangGraph.

    Creates LLMClient instances from the provider config, binds them to
    node wrapper functions via functools.partial, assembles the full
    StateGraph with conditional edges, and compiles it.

    Args:
        llm_config: Dict with provider configs. Expected keys:
            - api_key: str
            - base_url: str
            - model: str (default model identifier)
        searcher: Optional Searcher instance for document/graph search.
        legal_api: Optional LegalAPIClient for statute/precedent search.

    Returns:
        A compiled LangGraph StateGraph (CompiledGraph) ready for
        invocation with an initial DebateState.
    """
    logger.info("[main_graph] Building debate graph with model=%s", llm_config.get("model"))

    # Create the shared LLM client
    llm_client = _create_llm_client(llm_config)

    # Bind dependencies to node wrappers
    bound_team_speak = partial(
        _team_speak_wrapper,
        llm_client=llm_client,
        searcher=searcher,
        legal_api=legal_api,
    )
    bound_judge_accumulate = partial(
        _judge_accumulate_wrapper,
        llm_client=llm_client,
    )
    bound_route_next = partial(
        _route_next_wrapper,
        llm_client=llm_client,
    )
    bound_final_judgment = partial(
        _final_judgment_wrapper,
        llm_client=llm_client,
        searcher=searcher,
    )
    bound_judge_question = partial(_judge_question_wrapper, llm_client=llm_client)
    bound_agent_answer = partial(_agent_answer_wrapper, llm_client=llm_client)

    # Build the StateGraph
    graph = StateGraph(DebateState)

    # Add nodes
    graph.add_node("user_interrupt", _user_interrupt_wrapper)
    graph.add_node("team_speak", bound_team_speak)
    graph.add_node("judge_accumulate", bound_judge_accumulate)
    graph.add_node("route_next", bound_route_next)
    bound_round_end = partial(_round_end_wrapper, llm_client=llm_client)
    graph.add_node("round_end", bound_round_end)
    graph.add_node("final_judgment", bound_final_judgment)
    graph.add_node("judge_question", bound_judge_question)
    graph.add_node("agent_answer", bound_agent_answer)

    # Set entry point
    graph.set_entry_point("user_interrupt")

    # Linear edges: user_interrupt -> team_speak -> judge_accumulate -> route_next
    graph.add_edge("user_interrupt", "team_speak")
    graph.add_edge("team_speak", "judge_accumulate")
    graph.add_edge("judge_accumulate", "judge_question")

    # Conditional edge from judge_question: answer or proceed
    graph.add_conditional_edges(
        "judge_question",
        decide_judge_qa,
        {
            "agent_answer": "agent_answer",
            "route_next": "route_next",
        },
    )

    # Edge from agent_answer back to judge_question (for multi-question loop)
    graph.add_edge("agent_answer", "judge_question")

    # Conditional edge from route_next: decide whether to loop or end round
    graph.add_conditional_edges(
        "route_next",
        decide_next,
        {
            "user_interrupt": "user_interrupt",
            "round_end": "round_end",
        },
    )

    # Conditional edge from round_end: continue, stop, or pause
    graph.add_conditional_edges(
        "round_end",
        should_continue,
        {
            "user_interrupt": "user_interrupt",
            "final_judgment": "final_judgment",
            "__end__": END,
        },
    )

    # Final judgment leads to END
    graph.add_edge("final_judgment", END)

    # Compile the graph (with optional checkpointer for persistent state)
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("[main_graph] Debate graph compiled successfully.")
    return compiled
