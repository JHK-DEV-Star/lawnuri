"""
LangGraph state definitions for the LawNuri debate engine.

Defines the TypedDict states used by LangGraph to manage debate flow:
- TeamState: Internal state for a single debating team's subgraph.
- DebateState: Top-level state for the full debate orchestration graph.
"""

from typing import Annotated, TypedDict

import operator


class TeamState(TypedDict):
    """State for a single debate team's internal processing subgraph."""

    team_id: str                        # "team_a" | "team_b"
    members: list                       # List of AgentProfile dicts
    opponent_statement: str             # Opponent's last statement
    debate_context: list                # Full debate context so far
    extra_evidence: list                # User-injected evidence for this team
    internal_discussion: list           # Internal team discussion log
    agreed_strategy: str                # Consensus strategy
    search_results: list                # RAG search results from all agents
    role_assignments: list              # Current round role assignments
    selected_speaker: str               # Selected representative speaker agent_id
    output_statement: str               # Final representative statement
    output_evidence: list               # Evidence used (list of Evidence dicts)
    round: int                          # Current round
    debate_id: str                      # For RAG access
    team_opinion: str                   # The opinion this team advocates
    opponent_opinion: str               # The opposing opinion
    judge_qa_log: list                  # Previous judge Q&A history for context
    used_search_queries: list             # Search queries used in previous rounds
    judge_improvement_feedback: str       # Judge improvement feedback for this team

    team_a_name: str                        # Display name for Team A
    team_b_name: str                        # Display name for Team B
    team_cautions: list                     # Strategic warnings for this team

    # --- Fields from DebateState (read by subgraph nodes) ---
    all_evidences: list                   # All submitted evidence so far (for citation re-verification)
    blacklisted_evidence: list            # Evidence blacklisted in previous rounds

    # --- Subgraph inter-node fields (Step 1) ---
    discussion_log: list                  # discuss → statement: raw discussion messages
    accepted_cases: list                  # discuss → statement: ACCEPT-voted case numbers
    blacklisted_items: list               # discuss → statement: blacklisted evidence IDs
    discussed_cases: list                 # discuss → statement: all mentioned case numbers
    case_id_map: dict                     # search → discuss: case_number → item_id mapping
    situation_brief: str                  # from DebateState (for prompts)
    topic: str                            # from DebateState (for prompts)
    key_issues: list                      # from DebateState analysis
    analysis_summary: str                 # pre-built analysis summary string

    # --- Multi-agent fields (Step 5) ---
    agent_memories: dict                  # {agent_id: [memory_entry, ...]} — persistent across rounds
    agent_search_results: dict            # {agent_id: [result, ...]} — private per agent
    shared_evidence_pool: list            # evidence shared during discussion

    # --- Inter-node private fields ---
    _opponent_cited_summary: str          # discuss → statement: opponent's cited evidence summary

    # --- Cross-round review state (persisted across rounds) ---
    review_more_persist: dict             # {canonical_case_number: [agent_id, ...]} — REVIEW_MORE requesters accumulated across rounds
    accept_votes_persist: dict            # {canonical_case_number: [agent_id, ...]} — ACCEPT voters accumulated across rounds


class DebateState(TypedDict):
    """Full debate state. Fields annotated with Annotated[list, operator.add] are append-only."""

    debate_id: str                                          # Unique debate identifier
    situation_brief: str                                    # Raw user input text
    topic: str                                              # Extracted debate topic (by LLM)
    opinion_a: str                                          # Extracted opinion 1 (by LLM)
    opinion_b: str                                          # Extracted opinion 2 (by LLM)
    key_issues: list                                        # Extracted key issues (by LLM)
    round: int                                              # Current round number
    min_rounds: int                                         # Minimum rounds before early stop allowed
    max_rounds: int                                         # Maximum rounds
    debate_log: Annotated[list, operator.add]               # Full statement log (append-only)
    all_evidences: Annotated[list, operator.add]            # All submitted evidence (append-only)
    team_a_state: dict                                      # Current TeamState for team_a
    team_b_state: dict                                      # Current TeamState for team_b
    team_a_agents: list                                     # Team A agent profiles
    team_b_agents: list                                     # Team B agent profiles
    judge_agents: list                                      # Judge agent profiles
    judge_notes: Annotated[list, operator.add]              # Judge accumulated notes
    current_team: str                                       # Currently speaking team ("team_a" | "team_b")
    next_action: str                                        # Routing result
    early_stop_votes: list                                  # Judge early-stop votes
    verdicts: list                                          # Final verdicts
    status: str                                             # "running" | "paused" | "stopped" | "completed" | "extended"
    default_model: str                                      # Default LLM model
    llm_config: dict                                        # LLM provider config (api_key, base_url per provider)
    user_interrupt: dict | None                             # Pending user intervention data
    team_a_name: str                                         # Display name for Team A
    team_b_name: str                                         # Display name for Team B
    internal_discussions: list                               # Internal team discussions (separate from debate_log)
    judge_qa_log: Annotated[list, operator.add]              # Judge Q&A exchanges
    pending_judge_questions: list                             # Currently pending questions
    blacklisted_evidence: Annotated[list, operator.add]      # Evidence items blacklisted by teams
    # Analysis data (preserved from analyze phase, not modified by graph nodes)
    analysis: dict                                           # Full analysis dict
    parties: list                                            # Party information
    timeline: list                                           # Event timeline
    causal_chain: list                                       # Causal chain
    key_facts: list                                          # Key facts
    focus_points: dict                                       # Team focus points
    missing_information: list                                # Missing information
    team_a_cautions: list                                    # Team A strategic warnings
    team_b_cautions: list                                    # Team B strategic warnings
    judge_improvement_feedback: dict                         # {"team_a": str, "team_b": str} — replaced each round
    speaking_order_reasoning: str                             # Why the judge chose this round's first speaker
    # --- Multi-agent fields (Step 5) ---
    agent_memories: dict                                     # {agent_id: [memory_entry, ...]} — persistent across rounds
